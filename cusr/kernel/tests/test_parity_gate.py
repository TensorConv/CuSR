"""test_parity_gate.py — kernel 换代质量闸门 (W0).

规则: **任何新 kernel 版本必须过这道闸, 才允许替换 SR 实验在用的版本.**
性能优化 (fused / devjac / K-bucket / masking / ...) 随便做, 质量不许退.

流程: 跑 BATCH_LM 于 data/pop.bin (真实 EvoGP dump, 1000 棵) → verify.py
对拍 scipy-fp64 (相对步长 FD + MINPACK 内部缩放, 金标准) → 断言阈值.

阈值 (按 2026-06 修复后基线 94.0 / 92.4 / 99.8 留余量定; 量级扫描另见
test_fixture_scale.py, 两道都过才算过闸):
- loss-down rate (K>0)        >= 93%
- within 1.05x scipy loss     >= 90%
- within 10x scipy loss       >= 99%

用法:
    uv run python test_parity_gate.py                     # 闸 baseline ./batch_lm
    BATCH_LM=../batch_lm_fusedfd uv run python ...        # 闸某个变体
    GATE_NPROC=8 ...                                      # scipy worker 数
    GATE_REPORT=../data/verify_report_xxx.md ...          # 保留报告 artifact

注意: 阈值在 RTX 5070 Ti (laptop) 上校准; 换机器 (A100) 重测时若有 ±1pp
级别漂移属正常, 大于该量级要查.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = HERE.parents[2]  # cusr/kernel/tests/ -> cusr/kernel -> cusr -> repo root
BATCH_LM = Path(os.environ.get("BATCH_LM", PROJ / "batch_lm"))
POP = ROOT / "data" / "fixtures" / "pop.bin"

THRESHOLDS = {
    "loss_down": 93.0,
    "within_1p05": 90.0,
    "within_10": 99.0,
}


def main() -> int:
    if not BATCH_LM.exists():
        print(f"binary not found: {BATCH_LM}"); return 1
    if not POP.exists():
        print(f"{POP} 不存在, 先跑 dump_evogp.py"); return 1

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        print(f"[gate] running {BATCH_LM.name} on {POP.name} ...")
        r = subprocess.run([str(BATCH_LM), str(POP), str(out_dir)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(r.stdout); print(r.stderr); return 1
        wall_line = [l for l in r.stdout.splitlines() if "总耗时" in l]
        if wall_line:
            print(f"[gate] {wall_line[0].strip()}")

        # GATE_REPORT 相对路径按调用方 cwd 解析 (verify.py 以 cwd=PROJ 跑, 不解析会写歪)
        report = str(Path(os.environ.get("GATE_REPORT", out_dir / "verify_report.md")).resolve())
        nproc = os.environ.get("GATE_NPROC", "20")
        print(f"[gate] scipy parity check (nproc={nproc}) ...")
        v = subprocess.run(
            [sys.executable, str(PROJ / "verify.py"),
             "--pop", str(POP),
             "--c-final", str(out_dir / "c_final.bin"),
             "--status", str(out_dir / "status.bin"),
             "--report", report, "--nproc", nproc],
            capture_output=True, text=True, encoding="utf-8", cwd=PROJ)
        if v.returncode != 0:
            print(v.stdout); print(v.stderr); return 1

        tail = v.stdout.strip().splitlines()[-3:]
        print("\n".join(f"[verify] {l}" for l in tail))

        m_down = re.search(r"loss-down rate \(K>0\): ([\d.]+)%", v.stdout)
        m_env = re.search(r"1\.05×=([\d.]+)%\s+2×=([\d.]+)%\s+10×=([\d.]+)%", v.stdout)
        if not (m_down and m_env):
            print("[gate] 解析 verify.py 输出失败 — 格式变了?"); return 1
        got = {
            "loss_down": float(m_down.group(1)),
            "within_1p05": float(m_env.group(1)),
            "within_10": float(m_env.group(3)),
        }

    print(f"\n== parity gate ({BATCH_LM.name}) ==")
    n_pass = 0
    for key, thr in THRESHOLDS.items():
        ok = got[key] >= thr
        n_pass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {key:<12} {got[key]:.1f}% (>= {thr}%)")
    print(f"result: {n_pass}/{len(THRESHOLDS)} PASS")
    return 0 if n_pass == len(THRESHOLDS) else 1


if __name__ == "__main__":
    sys.exit(main())
