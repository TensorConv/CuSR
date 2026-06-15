"""诊断: kernel "失败"树到底返回了什么 c_final, 它们的 loss 离 tier 多远.
验证 best-finite 是否已隐式实现 (h_c = 最后接受值). 临时脚本, 跑完可删."""
from pathlib import Path
import numpy as np
from cusr.benchmark import popio, interp, runner

POP = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "pop.bin"
pop = popio.slice_pop(popio.load_pop_bin(POP), 300)
oracle = runner.get_oracle(POP, pop, 300, Path("_cache"))
Lstar = oracle["loss_star"]
K = pop["metas"][:, 3]

be = runner.make_backend("kernel")
res = be.fit_pop(pop)
Lb = interp.loss_pop(pop, res.c_final)
ta, tb, elig = runner.classify_tiers(Lb, Lstar, K)

print("kernel raw status_counts:", res.meta.get("status_counts"))
print("c_final 全有限?", bool(np.isfinite(res.c_final).all()),
      f"(NaN/Inf 个数={int((~np.isfinite(res.c_final)).sum())}/{res.c_final.size})")

st = res.status  # 0 conv, 1 maxiter, 2 fail, 3 k0
names = {0: "converged", 1: "maxiter", 2: "failed", 3: "k0"}
print(f"\n{'status':10s} {'n':>4s} {'finite-loss':>11s} {'tierB':>6s} {'loss中位(有限)':>14s}")
for s in (0, 1, 2, 3):
    msk = st == s
    n = int(msk.sum())
    if n == 0:
        continue
    fin = np.isfinite(Lb[msk])
    med = float(np.median(Lb[msk][fin])) if fin.any() else float("nan")
    print(f"{names[s]:10s} {n:4d} {int(fin.sum()):11d} {int(tb[msk].sum()):6d} {med:14.3e}")

# 重点: 失败树里, 有多少其实返回了有限且"接近"的 c_final
fail = st == 2
if fail.any():
    finL = Lb[fail]
    fin = np.isfinite(finL)
    print(f"\n失败树 {int(fail.sum())} 棵: c_final 有限 loss {int(fin.sum())} 棵, "
          f"非有限 {int((~fin).sum())} 棵")
    # 这些失败树离 tier-B 多远 (loss_b / max(1.05*L*, ...))
    thrB = np.maximum(1.05 * Lstar, Lstar + 1e-10)
    msk = fail & np.isfinite(Lb) & np.isfinite(Lstar)
    with np.errstate(all="ignore"):
        r = Lb[msk] / thrB[msk]
    if r.size:
        miss = r[r > 1]
        print(f"  失败但有限树 loss_b/tierB阈值: 已达 tier-B {int((r <= 1).sum())}/{r.size} 棵; "
              f"未达的 {miss.size} 棵 ratio 中位={np.median(miss):.1f} max={miss.max():.1f}")
        print("  → 未达标的是真·拟合质量不够 (离阈值远), 不是被 NaN/排除坑掉")
