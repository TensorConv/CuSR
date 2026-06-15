"""probe_pivotfloor.py — rung C: 给 Cholesky 主元加正地板 (不 abort 退化方向).

判别 advisor 的问题: B (fp64) 证明 520 chol-miss 是秩亏不是精度. 那"让 solve
放行退化方向"(s=max(s,ε)) 能不能救回 tier-B? 还是这些树放行后仍拟合不好,
只是从 chol-崩 迁移到"收敛却短" (→ tier-B 65% 部分不可约)?

floor 只碰退化主元 → 良态树与 A 逐位相同 → tier-B 变化纯来自放行退化方向.
ε 两点 (1e-9 激进 / 1e-3 适中) 防单点误导. 盯 NaN: 多 = 需真阻尼非只地板.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import cusr.kernel
from cusr.benchmark import popio, runner

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling run-scripts
from probe_fp64solve import CACHE, classify, run_binary  # noqa: E402

_KERNEL = Path(cusr.kernel.__file__).parent
BIN_A = (_KERNEL / "batch_lm_fusedfd_A").resolve()
FLOORS = {"F9 (ε=1e-9)": (_KERNEL / "batch_lm_fusedfd_F9").resolve(),
          "F3 (ε=1e-3)": (_KERNEL / "batch_lm_fusedfd_F3").resolve()}


def main():
    path = runner.resolve_pop("preset:inner-const-heavy")
    pop = popio.load_pop_bin(path)
    Lstar = runner.get_oracle(path, pop, None, CACHE)["loss_star"]
    elig_n = int((pop["metas"][:, 3] >= 1).sum())

    rawA, cA, _ = run_binary(BIN_A, pop, reps=1)
    LbA, tAa, tBa, elig = classify(pop, cA, Lstar)
    cholmiss_A = (rawA == 4) & elig & ~tBa
    n520 = int(cholmiss_A.sum())
    print(f"[gate#1] A: chol={int((rawA==4).sum())} tier-B={int(tBa.sum())} "
          f"(期望 652/2566); chol-miss={n520}")

    for label, binp in FLOORS.items():
        rawF, cF, _ = run_binary(binp, pop, reps=1)
        LbF, tAf, tBf, _ = classify(pop, cF, Lstar)
        nan_F = int((~np.isfinite(cF)).reshape(-1).sum())  # c_final 里 NaN/Inf 个数
        nan_tree = int((rawF == 2).sum())                  # status=FAIL_NAN 的树
        print(f"\n{'='*64}\n{label}  vs A")
        print(f"  tier-B : {int(tBa.sum())} → {int(tBf.sum())}  ({int(tBf.sum()-tBa.sum()):+d}, "
              f"{100*(tBf.sum()-tBa.sum())/elig_n:+.1f}pp)")
        print(f"  tier-A : {int(tAa.sum())} → {int(tAf.sum())}  ({int(tAf.sum()-tAa.sum()):+d})")
        print(f"  Cholesky崩: {int((rawA==4).sum())} → {int((rawF==4).sum())}  (放行后应≈0)")
        print(f"  NaN: c_final 非有限 {nan_F} 个, status=FAIL_NAN {nan_tree} 棵")
        # 520 chol-miss 去哪
        to_B = int((cholmiss_A & tBf).sum())
        to_nan = int((cholmiss_A & (rawF == 2)).sum())
        to_conv_short = int((cholmiss_A & (rawF == 0) & ~tBf).sum())
        to_maxit_short = int((cholmiss_A & (rawF == 1) & ~tBf).sum())
        still_chol = int((cholmiss_A & (rawF == 4)).sum())
        other = n520 - to_B - to_nan - to_conv_short - to_maxit_short - still_chol
        print(f"  >> A 的 chol-miss {n520} 棵在此档归宿:")
        print(f"       → 救回 tier-B        : {to_B:5d} ({100*to_B/n520:4.1f}%)")
        print(f"       → 收敛却仍短         : {to_conv_short:5d} ({100*to_conv_short/n520:4.1f}%)")
        print(f"       → maxiter 仍短       : {to_maxit_short:5d} ({100*to_maxit_short/n520:4.1f}%)")
        print(f"       → NaN (退化方向爆)   : {to_nan:5d} ({100*to_nan/n520:4.1f}%)")
        print(f"       → 仍 chol / 其它     : {still_chol+other:5d}")
        # no-harm: 良态树 (A 非 chol) loss 是否动
        wellcond = elig & (rawA != 4)
        churn = int((tBf[wellcond] != tBa[wellcond]).sum())
        print(f"  no-harm: 良态树 (A 非 chol) tier-B 归属变动 {churn} 棵 (应≈0, floor 不碰它们)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
