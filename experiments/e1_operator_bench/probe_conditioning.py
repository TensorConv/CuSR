"""probe_conditioning.py — 决定 rank-revealing-on-JtJ (便宜) vs QR-on-J (贵).

advisor: 决定权在"近奇异桶"有多大.
  精确秩亏 (σmin≈0, 死/冗余常数)     → 任何 rank-reveal 都治, 条件数平方无所谓 → 层次1 够.
  近奇异   (κ ~1e4-1e10, 有限但大)   → fp32 形成 JtJ 把 κ 平方 → 毁掉 → 需 QR-on-J 或 fp64-JtJ.
  良态但短 (κ 小)                    → fp32 FD-Jacobian 噪声, 两层都救不了 (deferred 桶).

在 fp64 里对 (A 的 520 chol-miss) + (F3 比 A 多救回但非 chol-miss 的近奇异候选) 算 κ.
J 在各树的 A-endpoint (cA) 处用中心差分构造 (失败树 ≈ c_init, 没动过).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import cusr.kernel
from cusr.benchmark import interp, popio, runner

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling run-scripts
from probe_fp64solve import CACHE, classify, run_binary  # noqa: E402

_KERNEL = Path(cusr.kernel.__file__).parent
BIN = {k: (_KERNEL / f"batch_lm_fusedfd_{k}").resolve()
       for k in ("A", "F9", "F3")}


def fp64_jacobian(pop, m, c):
    """中心差分 fp64 Jacobian J[N,K]. 非有限 → 返回 None."""
    K = len(c)
    N = pop["xs"].shape[0]
    J = np.empty((N, K), dtype=np.float64)
    for j in range(K):
        h = 1e-6 * max(abs(float(c[j])), 1.0)
        cp = c.astype(np.float64).copy(); cp[j] += h
        cm = c.astype(np.float64).copy(); cm[j] -= h
        yp = interp.eval_pop_tree(pop, m, cp)
        ym = interp.eval_pop_tree(pop, m, cm)
        J[:, j] = (yp - ym) / (2.0 * h)
    return J if np.isfinite(J).all() else None


def kappa_of(pop, metas, c_all, m):
    """返回 (κ, rank_deficit). rank_deficit = K - #(σ > 1e-6·σmax):
    >0 = 结构秩亏 (truncation territory, rank-reveal 即治, 与精度无关);
    =0 但 κ 大 = 满秩近奇异 (条件数有限, 形成 JtJ 才会被平方毁掉 → 才真需 QR/fp64)."""
    K = int(metas[m, 3]); coff = int(metas[m, 2])
    c = c_all[coff:coff + K]
    J = fp64_jacobian(pop, m, c)
    if J is None:
        return None
    s = np.linalg.svd(J, compute_uv=False)
    smax = float(s[0]); smin = float(s[-1])
    rank = int((s > 1e-6 * smax).sum())
    kappa = np.inf if smin <= 0 else smax / smin
    return kappa, K - rank


def bucket(pop, metas, c_all, mask, label):
    idx = np.where(mask)[0]
    ks, nonfin = [], 0
    for m in idx:
        k = kappa_of(pop, metas, c_all, m)
        if k is None:
            nonfin += 1
        else:
            ks.append(k)
    ks = np.array(ks)
    # 阈值: σmin/σmax < 1e-12 视作精确秩亏; fp32 ~7 位 → κ>~1e7 时 fp32 JtJ 已不可救
    exact = int(((np.isinf(ks)) | (ks > 1e12)).sum())
    near = int(((ks > 1e4) & (ks <= 1e12)).sum())
    well = int((ks <= 1e4).sum())
    fin = ks[np.isfinite(ks)]
    print(f"\n{label}  (n={int(mask.sum())}, J非有限/奇点 {nonfin})")
    print(f"  精确秩亏 κ>1e12        : {exact:4d}   [层次1 rank-reveal 即治]")
    print(f"  近奇异   1e4<κ≤1e12    : {near:4d}   [fp32 JtJ 平方毁掉 → 需 QR/fp64-JtJ] ←决定")
    print(f"  良态     κ≤1e4         : {well:4d}   [非秩亏问题; 若短=Jacobian 噪声]")
    if fin.size:
        print(f"  κ 中位={np.median(fin):.1e}  分位[25/50/75/95]="
              f"{np.percentile(fin,[25,50,75,95])}")


def main():
    path = runner.resolve_pop("preset:inner-const-heavy")
    pop = popio.load_pop_bin(path)
    metas = pop["metas"]
    Lstar = runner.get_oracle(path, pop, None, CACHE)["loss_star"]

    rawA, cA, _ = run_binary(BIN["A"], pop, reps=1)
    _, _, tBa, elig = classify(pop, cA, Lstar)
    _, cF9, _ = run_binary(BIN["F9"], pop, reps=1)
    _, _, tBf9, _ = classify(pop, cF9, Lstar)
    _, cF3, _ = run_binary(BIN["F3"], pop, reps=1)
    _, _, tBf3, _ = classify(pop, cF3, Lstar)

    afail = (rawA == 4) & elig & ~tBa          # 520 chol-miss
    f3_only = elig & tBf3 & ~tBa & ~afail      # F3 多救回, 但 A 里不是 chol-崩 = 近奇异候选
    f9gain = int((elig & tBf9 & ~tBa).sum())
    f3gain = int((elig & tBf3 & ~tBa).sum())
    print(f"chol-miss(A)={int(afail.sum())}  F9 救回={f9gain}  F3 救回={f3gain}  "
          f"(F3 多出的非chol桶 n={int(f3_only.sum())})")

    bucket(pop, metas, cA, afail, "AFAIL: A 的 520 chol-miss")
    bucket(pop, metas, cA, f3_only, "F3-only: F3救回-非chol (近奇异候选)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
