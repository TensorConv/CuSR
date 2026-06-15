"""probe_fp64solve.py — A/B 实测: 混合精度 K×K 解 (fp64) 救回多少 tier-B.

exp012 交叉表发现: inner-const-heavy 真实 tier-B 64.8%, 1396 个 miss 里
520 (37%) 是 fp32 Cholesky 崩, 757 (54%) 是"收敛却差 5%". 这脚本实测
B (-DSOLVE_FP64) 相对 A (fp32 基线) 把那 520 救回了多少, 以及 the cost.

A/B 必须同源同 flag (只差 -DSOLVE_FP64)、同机现编 (见 008/build cmd).
gate #1: A 必须复现冻结基线 (inner-const-heavy: chol≈652, tier-B≈2566).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import cusr.kernel
from cusr.benchmark import interp, popio, runner

_KERNEL = Path(cusr.kernel.__file__).parent
BIN_A = (_KERNEL / "batch_lm_fusedfd_A").resolve()
BIN_B = (_KERNEL / "batch_lm_fusedfd_B").resolve()
CACHE = Path("_cache")
TIME_RE = re.compile(r"总耗时 ([\d.]+) 秒")
RAWNAME = {0: "conv", 1: "maxit", 2: "nan", 3: "k0", 4: "chol"}


def run_binary(binp: Path, pop: dict, reps: int = 3):
    """跑 binary, 返回 (raw_status[M], c_final, core_wall_median)."""
    walls = []
    raw = cfin = None
    with tempfile.TemporaryDirectory() as td:
        pp = Path(td) / "pop.bin"
        popio.save_pop_bin(pop, pp)
        for _ in range(reps):
            r = subprocess.run([str(binp), str(pp), td, "--quiet"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{binp.name} exit={r.returncode}: {r.stderr[-300:]}")
            mt = TIME_RE.search(r.stdout or "")
            if mt:
                walls.append(float(mt.group(1)))
        raw = np.fromfile(Path(td) / "status.bin", dtype=np.int32)
        cfin = np.fromfile(Path(td) / "c_final.bin", dtype=np.float32).astype(np.float64)
    core = float(np.median(walls)) if walls else float("nan")
    return raw, cfin, core


def classify(pop, cfin, Lstar):
    K = pop["metas"][:, 3]
    Lb = interp.loss_pop(pop, cfin)
    tA, tB, elig = runner.classify_tiers(Lb, Lstar, K)
    return Lb, tA, tB, elig


def probe(preset: str, detailed: bool):
    path = runner.resolve_pop(f"preset:{preset}")
    pop = popio.load_pop_bin(path)
    Lstar = runner.get_oracle(path, pop, None, CACHE)["loss_star"]
    elig_n = int((pop["metas"][:, 3] >= 1).sum())

    rawA, cA, coreA = run_binary(BIN_A, pop)
    rawB, cB, coreB = run_binary(BIN_B, pop)
    _, tAa, tBa, elig = classify(pop, cA, Lstar)
    _, tAb, tBb, _ = classify(pop, cB, Lstar)

    cholA = int((rawA == 4).sum())
    cholB = int((rawB == 4).sum())
    print(f"\n{'='*72}\n{preset}  (M={pop['M']}, eligible={elig_n})")
    print(f"  {'':14s} {'A(fp32)':>10s} {'B(fp64解)':>10s} {'Δ':>8s}")
    print(f"  {'tier-B 数':14s} {int(tBa.sum()):>10d} {int(tBb.sum()):>10d} {int(tBb.sum()-tBa.sum()):>+8d}")
    print(f"  {'tier-B %':14s} {100*tBa.sum()/elig_n:>9.1f}% {100*tBb.sum()/elig_n:>9.1f}% {100*(tBb.sum()-tBa.sum())/elig_n:>+7.1f}%")
    print(f"  {'tier-A 数':14s} {int(tAa.sum()):>10d} {int(tAb.sum()):>10d} {int(tAb.sum()-tAa.sum()):>+8d}")
    print(f"  {'Cholesky崩':14s} {cholA:>10d} {cholB:>10d} {cholB-cholA:>+8d}")
    print(f"  {'核心耗时(s)':14s} {coreA:>10.3f} {coreB:>10.3f} {coreB/coreA if coreA else 0:>7.2f}x")

    # no-harm (集合级): 净 tier-B 不降, 回退要少
    gained = int((tBb & ~tBa).sum())
    lost = int((tBa & ~tBb).sum())
    print(f"  no-harm: 新进 tier-B {gained}, 回退 {lost}, 净 {gained-lost:+d}")

    if detailed:
        # "那 520 棵 chol-miss 去哪了": A 里 chol崩且没达B 的树, 在 B 里的归宿
        cholmiss_A = (rawA == 4) & elig & ~tBa
        n = int(cholmiss_A.sum())
        to_B = int((cholmiss_A & tBb).sum())
        still_chol = int((cholmiss_A & (rawB == 4)).sum())
        conv_short = int((cholmiss_A & (rawB == 0) & ~tBb).sum())
        other = n - to_B - still_chol - conv_short
        print(f"\n  >> A 的 chol-miss {n} 棵在 B 里的归宿:")
        print(f"       → 救回 tier-B          : {to_B:5d}  ({100*to_B/n:4.1f}%)  [fp64 修好了条件数]")
        print(f"       → 仍 Cholesky 崩 (s≤0) : {still_chol:5d}  ({100*still_chol/n:4.1f}%)  [真·秩亏, fp64 也修不了]")
        print(f"       → 收敛却仍差5% (短)    : {conv_short:5d}  ({100*conv_short/n:4.1f}%)  [fp32 Jacobian 天花板]")
        print(f"       → 其它(maxit/nan)      : {other:5d}  ({100*other/n:4.1f}%)")


def main():
    if not (BIN_A.exists() and BIN_B.exists()):
        print(f"binaries 缺失: A={BIN_A.exists()} B={BIN_B.exists()}")
        return 1
    # gate #1: A 复现冻结基线
    path = runner.resolve_pop("preset:inner-const-heavy")
    pop = popio.load_pop_bin(path)
    Lstar = runner.get_oracle(path, pop, None, CACHE)["loss_star"]
    rawA, cA, _ = run_binary(BIN_A, pop, reps=1)
    _, _, tBa, _ = classify(pop, cA, Lstar)
    cholA, tBn = int((rawA == 4).sum()), int(tBa.sum())
    ok = abs(cholA - 652) <= 5 and abs(tBn - 2566) <= 5
    print(f"[gate#1] 编辑后 fp32-A: Cholesky={cholA} (期望~652), tier-B={tBn} (期望~2566)"
          f"  → {'PASS 复现基线' if ok else 'FAIL 源/arch 漂移, 先查清再读 B!'}")
    if not ok:
        return 1

    for preset in ["inner-const-heavy", "early-gen", "late-gen-bloated",
                   "synth-inner-const-heavy"]:
        probe(preset, detailed=(preset == "inner-const-heavy"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
