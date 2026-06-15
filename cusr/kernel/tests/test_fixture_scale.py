"""test_fixture_scale.py — 常数量级扫描 (W0: 相对 eps_fd 修复的验收测试).

跑每个 pop_fixture_scale_*.bin, 验:
- tier1 (单常数 / zero_init): status==0 且 per-constant 相对误差 < 5e-3 → 硬断言.
  注意这里是 **逐常数** 相对误差 max_k |c_k-c*_k|/|c*_k|, 不是混合范数 ——
  量级悬殊时混合范数会把小常数的错误淹没掉.
- tier2 (混合量级): 只打表诊断, 不计入 exit code (列缩放决策的输入).

修复前预期: tier1 在 s 远离 1 时 FAIL (绝对 eps_fd=1e-3 的 bug 暴露).
修复后预期: tier1 全 PASS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
# BATCH_LM=../batch_lm_fusedfd 可指定变体二进制 (默认 baseline)
BATCH_LM = Path(os.environ.get("BATCH_LM", PROJ / "batch_lm"))
TRUTH = HERE / "pop_fixture_scale_truth.json"

STATUS_NAMES = {0: "CONVERGED", 1: "MAXITER", 2: "FAIL_NAN", 3: "K0_SKIP", 4: "FAIL_CHOLESKY"}
C_REL_TOL = 5e-3


def run_batch_lm(fixture: Path, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(BATCH_LM), str(fixture), str(out_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise RuntimeError(f"batch_lm exit={r.returncode} on {fixture.name}")
    status = np.fromfile(out_dir / "status.bin", dtype=np.int32)
    c_final = np.fromfile(out_dir / "c_final.bin", dtype=np.float32)
    return status, c_final


def main():
    if not BATCH_LM.exists():
        print(f"batch_lm not built. Run `make batch_lm` in {PROJ}."); return 1
    if not TRUTH.exists():
        print("fixtures not generated. Run pop_fixture_scale_gen.py first."); return 1

    manifest = json.loads(TRUTH.read_text())["files"]
    t1_pass = t1_total = t2_pass = t2_total = 0

    for entry in manifest:
        fixture = HERE / entry["bin"]
        out_dir = HERE / "_out_scale" / fixture.stem
        status, c_final = run_batch_lm(fixture, out_dir)
        print(f"\n== scale={entry['scale']:.0e} ({entry['bin']}) ==")
        for i, fx in enumerate(entry["fixtures"]):
            K, off, c_true = fx["K"], fx["c_offset"], np.asarray(fx["c_true"])
            c_got = c_final[off:off + K].astype(np.float64)
            # 逐常数相对误差 (量级悬殊时混合范数会淹没小常数的错); 分母下限防未来 c_true=0 的 fixture
            c_rel = float(np.max(np.abs(c_got - c_true) / np.maximum(np.abs(c_true), 1e-30)))
            ok = (status[i] == 0) and (c_rel < C_REL_TOL)
            tag = "PASS" if ok else "FAIL"
            tier = fx["tier"]
            if tier == 1:
                t1_total += 1; t1_pass += ok
            else:
                t2_total += 1; t2_pass += ok
            print(f"  [{tag}] t{tier} {fx['name']:<16} status={STATUS_NAMES.get(int(status[i]), status[i]):<13}"
                  f" c_rel={c_rel:.3e}  c_got={np.array2string(c_got, precision=4)}")

    print(f"\nresult: tier1 {t1_pass}/{t1_total} PASS (hard assert), "
          f"tier2 {t2_pass}/{t2_total} pass (diagnostic only)")
    return 0 if t1_pass == t1_total else 1


if __name__ == "__main__":
    sys.exit(main())
