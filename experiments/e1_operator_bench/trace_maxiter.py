"""trace_maxiter.py — 追 floor(F9) 在 W0 gate 数据 (008/data/pop.bin) 上把树搞差的机制.

背景: floor 把退化主元 s≤0 抬成 +ε 放行, 全 preset 验证看着能救 tier-B, 但 W0 逐树
parity gate 炸了 (一批树 fp64 真 loss 远差于 nofloor). best-seen 修法已被逻辑证伪对
MAXITER 是 no-op: MAXITER 树从没走收敛分支, 只走 accept-on-decrease(534) → h_c 末值
就是整条轨迹 fp32-loss 最低的迭代 = best-seen-by-fp32. 那 floor 为何更差?

advisor 两读法 (在 floor 的 c_final 同一点比 kernel-fp32 h_loss vs interp fp64):
  (A) fp32 撒谎 : floor_fp64 ≫ floor_fp32  (kernel 以为改进, 真 loss 差) → 杠杆=accept 用 fp64/Kahan
  (B) 更差盆地 : floor_fp64 ≈ floor_fp32 都高 (kernel 忠实最小化忠实指标, 但 floor 贪心
                 中步把树带进真更差局部最优; nofloor 升 λ 才是对的正则) → bank it
判别量 = floor_fp64 / floor_fp32 (同一 c_final). ≫1→A, ≈1→B.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import cusr.kernel
from cusr.benchmark import interp, popio

D008 = Path(cusr.kernel.__file__).parent.resolve()
POP = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "pop.bin"
OUT = {"nofloor": Path("_trace_nofloor").resolve(),
       "floor":   Path("_trace_floor").resolve()}
# 008 binary 把输出写在 008/<out_dir>; 这里 out_dir 是相对 008 跑的, 实际落在 008 下
OUT = {k: D008 / f"_trace_{k}" for k in OUT}


def load(tag):
    d = OUT[tag]
    st = np.fromfile(d / "status.bin", dtype=np.int32)
    c = np.fromfile(d / "c_final.bin", dtype=np.float32)
    lf = np.fromfile(d / "loss_final.bin", dtype=np.float32)  # kernel 自报 fp32 loss
    return st, c, lf


def main():
    pop = popio.load_pop_bin(POP)
    K = pop["metas"][:, 3]
    elig = K >= 1

    st_nf, c_nf, fp32_nf = load("nofloor")
    st_f, c_f, fp32_f = load("floor")

    L_nf = interp.loss_pop(pop, c_nf)   # fp64 真 loss (协议口径)
    L_f = interp.loss_pop(pop, c_f)

    fin = np.isfinite(L_nf) & np.isfinite(L_f)
    worse = elig & fin & (L_f > 1.05 * L_nf)
    nw = int(worse.sum())
    print(f"M={len(K)}  eligible(K≥1)={int(elig.sum())}  floor 比 nofloor 更差(>1.05×)= {nw} 棵")

    ratio = L_f[worse] / np.maximum(L_nf[worse], 1e-300)
    print(f"  更差倍数 (fp64 floor/nofloor): 中位={np.median(ratio):.1f}×  "
          f"分位[50/90/99]={np.percentile(ratio,[50,90,99]).round(1)}  max={ratio.max():.0f}×")

    SMAP = {0: "CONVERGED", 1: "MAXITER", 2: "FAIL_NAN", 3: "K0_SKIP", 4: "FAIL_CHOL"}
    dist = {SMAP.get(int(s), s): int((st_f[worse] == s).sum()) for s in np.unique(st_f[worse])}
    print(f"  这些更差树的 floor-side status: {dist}")
    nmax = int((st_f[worse] == 1).sum())
    print(f"  其中 MAXITER = {nmax}/{nw} ({100*nmax/max(nw,1):.0f}%)  ← best-seen 对这些是 no-op")

    # ---- 判别: floor 的 c_final 同一点, fp64 / fp32 ----
    f64 = L_f[worse]
    f32 = fp32_f[worse]
    disc = f64 / np.maximum(f32, 1e-300)
    print(f"\n[判别] 在 floor 的 c_final 同一点, floor_fp64 / floor_fp32:")
    print(f"  中位={np.median(disc):.2f}×  分位[10/50/90]={np.percentile(disc,[10,50,90]).round(2)}  max={disc.max():.1f}×")
    a_like = int((disc > 2.0).sum())
    b_like = int((disc <= 2.0).sum())
    print(f"  ≫1 (>2×, (A) fp32 撒谎)= {a_like}   ≈1 (≤2×, (B) 真更差盆地)= {b_like}")

    # 旁证: kernel 自报 fp32, floor 是否"以为自己不比 nofloor 差"?
    kernel_thinks_better = int((fp32_f[worse] <= 1.05 * fp32_nf[worse]).sum())
    print(f"\n  旁证: 更差树里 floor 的 fp32 h_loss ≤ 1.05× nofloor 的 fp32 h_loss "
          f"(kernel 自认不差)= {kernel_thinks_better}/{nw}")
    print(f"        若高 → kernel 自己的 fp32 账本就显示 floor 更差 (不是被 fp64 才看出) → 偏 (B)")

    # floor 是"破坏 nofloor 拟合好的树" 还是"没救起本就烂的树"?
    SMAP2 = {0: "CONV", 1: "MAXIT", 2: "NAN", 3: "K0", 4: "CHOL"}
    nf_dist = {SMAP2.get(int(s), s): int((st_nf[worse] == s).sum()) for s in np.unique(st_nf[worse])}
    nf_good = int((L_nf[worse] < 1e-3).sum())
    nf_chol = int((st_nf[worse] == 4).sum())
    print(f"\n  这些更差树的 nofloor-side status: {nf_dist}")
    print(f"    nofloor 把它们拟合到 fp64<1e-3(好) = {nf_good}/{nw}   nofloor 自己 chol-崩 = {nf_chol}/{nw}")
    print(f"    → 多为'好' = floor 在破坏 nofloor 本已拟合好的树 (不是没救起烂树)")

    # 抽 3 棵看具体数字
    idx = np.where(worse)[0]
    order = np.argsort(-ratio)[:3]
    print(f"\n  样本 (按 fp64 更差倍数 top3):")
    print(f"    {'tree':>5s} {'K':>3s} {'nf_fp64':>11s} {'f_fp64':>11s} {'f_fp32':>11s} {'f64/f32':>8s} {'status':>9s}")
    for o in order:
        m = idx[o]
        print(f"    {m:5d} {int(K[m]):3d} {L_nf[m]:11.3e} {L_f[m]:11.3e} {fp32_f[m]:11.3e} "
              f"{L_f[m]/max(fp32_f[m],1e-300):8.2f} {SMAP.get(int(st_f[m])):>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
