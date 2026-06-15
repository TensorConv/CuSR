"""gen_synth.py — 合成 workload 生成器 (W4 后半).

按 characterization.md 的真实快照边际分布 (nodes / K / 算子组占比) 校准的
随机树生成器, 产出 008 格式 pop.bin (popio.build_pop 构造入口). 用途:

* 给 W7 做 design-space 扫描 (M / N / 结构 preset 三轴), 不依赖跑 EvoGP;
* 与真实 preset 互补: 合成树的 y 由树自身在 c_true 处求值 + 噪声生成,
  即"可恢复"问题 (oracle 能到全局最优附近), 适合压性能与精度轴;
  真实 preset 的树多数拟合不上目标, 适合讲质量档位故事 — 两者口径不同,
  论文里要分开报.

三个 synth preset 镜像三个真实 preset 的边际 (校准结果见 PRESETS 注释):

  uv run python gen_synth.py --preset early-gen --M 4000 --N 1000 --seed 0
  uv run python gen_synth.py --all              # 三个 preset 默认尺寸全生成
  uv run python gen_synth.py --preset late-gen-bloated --M 200 --report-only

生成确定性: 同 (preset, M, N, seed, noise) → 逐字节相同的 .bin.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from cusr.benchmark import popio
from cusr.benchmark.interp import F, eval_tree_arrays

NTYPE_VAR, NTYPE_CONST, NTYPE_UFUNC, NTYPE_BFUNC = 0, 1, 2, 3

# 算子组内权重 = 全快照 opcode 频次 (characterization.md 全局表)
BINARY_OPS = [F.ADD, F.SUB, F.MUL, F.DIV]
BINARY_W = np.array([23.43, 19.28, 22.35, 13.85]); BINARY_W /= BINARY_W.sum()
UNARY_OPS = [F.SIN, F.COS, F.TAN]
UNARY_W = np.array([9.54, 7.93, 3.61]); UNARY_W /= UNARY_W.sum()

# 每 preset 的目标边际取自 characterization.md 对应真实快照行;
# f_trig / p_const / p_allvar / size 分布参数为数值校准结果 (M=4000 N=200 seed 0 验证,
# nodes_mean / K_mean / K0 / trig 均落在目标 ±5% 内). 两个已知残差, 不再追:
# * late 组 trig 4.5% vs 真实 2% — 偶数节点预算被迫含 1 个一元 (奇偶下限);
# * late 组 K2+ 35% vs 39% — 叶子二项模型 vs 真实 K 分布过散布形状不同,
#   K_mean / K0 已对齐, 三者不可兼得.
PRESETS = {
    # 镜像 I.18.12_pop4000 gen5: nodes m12.4/p50 11/p90 22, K m2.77 K0 10% K2+ 70%, 三角 30%
    "early-gen": dict(
        n_vars=3, size_kind="lognorm", size_mu=2.40, size_sigma=0.54,
        size_lo=3, size_hi=32, f_trig=0.30, p_const=0.53, p_allvar=0.07,
        target=dict(nodes_mean=12.4, K_mean=2.77, K0=0.10, K2=0.70, trig=0.30),
    ),
    # 镜像 I.12.1_pop4000 gen50: nodes m26.6/p50 29, K m1.23 K0 29% K2+ 39%, 三角 2%
    "late-gen-bloated": dict(
        n_vars=2, size_kind="normclip", size_mu=27.0, size_sigma=4.5,
        size_lo=5, size_hi=32, f_trig=0.0, p_const=0.094, p_allvar=0.06,
        target=dict(nodes_mean=26.6, K_mean=1.23, K0=0.29, K2=0.39, trig=0.02),
    ),
    # 镜像 I.6.2_pop4000 gen30: nodes m27.8/p50 31, K m6.88 K0 1% K2+ 97%, 三角 39%
    "inner-const-heavy": dict(
        n_vars=2, size_kind="normclip", size_mu=28.8, size_sigma=5.0,
        size_lo=7, size_hi=32, f_trig=0.39, p_const=0.62, p_allvar=0.01,
        target=dict(nodes_mean=27.8, K_mean=6.88, K0=0.01, K2=0.97, trig=0.39),
    ),
}


def _sample_size(rng, cfg) -> int:
    if cfg["size_kind"] == "lognorm":
        s = rng.lognormal(cfg["size_mu"], cfg["size_sigma"])
    else:
        s = rng.normal(cfg["size_mu"], cfg["size_sigma"])
    return int(np.clip(round(s), cfg["size_lo"], cfg["size_hi"]))


def _unary_budget(rng, n: int, f_trig: float) -> int:
    """n 节点树要 trig 占比 f, 反解一元个数 u.

    u 个一元 + b 个二元 → 节点 n = u+2b+1, op 数 = u+b → f = 2u/(n-1+u),
    解出 u = f(n-1)/(2-f). 约束: u ≡ (n-1) mod 2 (纯二元树节点数恒为奇数,
    偶 n 至少被迫 1 个一元), 0 ≤ u ≤ n-1."""
    u0 = f_trig * (n - 1) / (2.0 - f_trig)
    u = int(round(u0))
    if (u & 1) != ((n - 1) & 1):
        u += 1 if u < u0 else -1
    return int(np.clip(u, (n - 1) & 1, n - 1))


def _build_prefix(rng, n: int, u: int, cfg, allvar: bool, out: list) -> None:
    """n 节点 / u 一元 预算, prefix 序 (父→左→右) 追加 (ntype, nv) 到 out.
    不变量: u ≡ (n-1) mod 2, 0 ≤ u ≤ n-1."""
    if n == 1:
        if not allvar and rng.random() < cfg["p_const"]:
            out.append((NTYPE_CONST, 0.0))
        else:
            out.append((NTYPE_VAR, float(rng.integers(cfg["n_vars"]))))
        return
    # 选一元的概率 = u / 剩余 op 数, 使一元在树里均匀散布; u==n-1 时必为一元
    if u > 0 and rng.random() < u / ((n - 1 + u) / 2.0):
        out.append((NTYPE_UFUNC, float(rng.choice(UNARY_OPS, p=UNARY_W))))
        _build_prefix(rng, n - 1, u - 1, cfg, allvar, out)
        return
    out.append((NTYPE_BFUNC, float(rng.choice(BINARY_OPS, p=BINARY_W))))
    while True:  # 抽左子树大小 + 左侧一元配额, 直到奇偶/上限约束可满足
        ls = int(rng.integers(1, n - 1))  # 1..n-2
        rs = n - 1 - ls
        lo, hi = max((ls - 1) & 1, u - (rs - 1)), min(ls - 1, u)
        if (lo & 1) != ((ls - 1) & 1):
            lo += 1
        if lo <= hi:
            break
    ul = lo + 2 * int(rng.integers(0, (hi - lo) // 2 + 1))
    _build_prefix(rng, ls, ul, cfg, allvar, out)
    _build_prefix(rng, rs, u - ul, cfg, allvar, out)


def gen_pop(preset: str, M: int, N: int, seed: int, noise_rel: float = 1e-2) -> dict:
    """生成 M 棵树的合成 pop. y = f(x; c_true) + noise_rel·rms(y)·N(0,1),
    c_init = c_true·U(0.7,1.3) + N(0,0.02) — 与真实演化里"常数没调好"一致."""
    cfg = PRESETS[preset]
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.5, 2.5, size=(N, cfg["n_vars"])).astype(np.float32)
    trees, ys, c_trues = [], [], []
    n_reject = 0
    while len(trees) < M:
        nodes: list = []
        size = _sample_size(rng, cfg)
        _build_prefix(rng, size, _unary_budget(rng, size, cfg["f_trig"]),
                      cfg, rng.random() < cfg["p_allvar"], nodes)
        nt = np.array([t for t, _ in nodes], np.int32)
        nv = np.array([v for _, v in nodes], np.float32)
        ci = np.full(len(nt), -1, np.int32)
        ci[nt == NTYPE_CONST] = np.arange(int((nt == NTYPE_CONST).sum()), dtype=np.int32)
        K = int((nt == NTYPE_CONST).sum())
        c_true = (rng.uniform(0.5, 3.0, K) * rng.choice([-1.0, 1.0], K)).astype(np.float32)
        y = eval_tree_arrays(nt, nv, ci, xs, c_true)
        # 拒绝采样: 非有限 / 量级爆炸 (DIV 极点、tan 渐近线) 的树重抽
        if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > 1e6:
            n_reject += 1
            continue
        rms = max(float(np.sqrt(np.mean(y * y))), 1e-3)
        ys.append(y + noise_rel * rms * rng.standard_normal(N))
        c_init = c_true * rng.uniform(0.7, 1.3, K) + rng.normal(0.0, 0.02, K)
        trees.append((nt, nv, ci, c_init.astype(np.float32)))
        c_trues.append(c_true)  # 真常数, 噪声地板用; 仅接受的树 (拒绝的不进)
    if n_reject:
        print(f"  (拒绝采样: 重抽 {n_reject} 棵, 占比 {n_reject / (M + n_reject):.1%})",
              file=sys.stderr)
    pop = popio.build_pop(trees, xs, np.array(ys, np.float32))
    # c_true 与 c_init 同序同布局 (按 metas c_off 切片) → loss_pop(pop, c_true) 即噪声地板
    pop["c_true"] = (np.concatenate(c_trues).astype(np.float32)
                     if pop["total_c"] else np.zeros(0, np.float32))
    return pop


# ---------------------------------------------------------------- 边际统计

def _depth(nt) -> int:
    def walk(i):  # → (span, depth)
        t = int(nt[i])
        if t in (NTYPE_VAR, NTYPE_CONST):
            return 1, 1
        if t == NTYPE_UFUNC:
            s, d = walk(i + 1)
            return 1 + s, 1 + d
        ls, ld = walk(i + 1)
        rs, rd = walk(i + 1 + ls)
        return 1 + ls + rs, 1 + max(ld, rd)
    return walk(0)[1]


def pop_marginals(pop: dict) -> dict:
    M = pop["M"]
    nodes = pop["metas"][:, 1].astype(float)
    Ks = pop["metas"][:, 3].astype(float)
    depths, n_un, n_ops = [], 0, 0
    for m in range(M):
        off, n, _, _ = pop["metas"][m].tolist()
        nt = pop["nt"][off:off + n]
        depths.append(_depth(nt))
        n_un += int((nt == NTYPE_UFUNC).sum())
        n_ops += int(((nt == NTYPE_UFUNC) | (nt == NTYPE_BFUNC)).sum())
    return dict(nodes_mean=nodes.mean(), nodes_p50=float(np.median(nodes)),
                nodes_p90=float(np.percentile(nodes, 90)), nodes_max=int(nodes.max()),
                depth_mean=float(np.mean(depths)), depth_max=int(np.max(depths)),
                K_mean=Ks.mean(), K0=float((Ks == 0).mean()), K2=float((Ks >= 2).mean()),
                trig=n_un / max(n_ops, 1))


def report(preset: str, pop: dict) -> str:
    tg, got = PRESETS[preset]["target"], pop_marginals(pop)
    lines = [f"[{preset}] M={pop['M']} N={pop['N']} n_vars={pop['n_vars']} "
             f"K_max={pop['K_max']} max_stack={pop['max_stack']}",
             f"  nodes m/p50/p90/max = {got['nodes_mean']:.1f}/{got['nodes_p50']:.0f}/"
             f"{got['nodes_p90']:.0f}/{got['nodes_max']}  (目标 m={tg['nodes_mean']})",
             f"  depth m/max = {got['depth_mean']:.1f}/{got['depth_max']}"]
    for k in ("K_mean", "K0", "K2", "trig"):
        lines.append(f"  {k:7s} = {got[k]:.3f}  (目标 {tg[k]:.3f})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--all", action="store_true", help="三个 preset 全生成")
    ap.add_argument("--M", type=int, default=4000)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-rel", type=float, default=1e-2)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[3] / "data" / "workload" / "synth")
    ap.add_argument("--report-only", action="store_true", help="只打边际, 不落盘")
    args = ap.parse_args()
    names = sorted(PRESETS) if args.all else ([args.preset] if args.preset else [])
    if not names:
        ap.error("--preset 或 --all 二选一")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        pop = gen_pop(name, args.M, args.N, args.seed, args.noise_rel)
        print(report(name, pop))
        if not args.report_only:
            out = args.out_dir / f"synth_{name}_M{args.M}_N{args.N}_seed{args.seed}.bin"
            popio.save_pop_bin(pop, out)
            sha = hashlib.sha1(out.read_bytes()).hexdigest()[:16]
            print(f"  → {out}  sha1[:16]={sha}")
            sidecar = out.with_name(out.stem + ".ctrue.npy")
            np.save(sidecar, pop["c_true"])
            print(f"  → {sidecar.name}  (c_true {pop['c_true'].size} floats, 噪声地板用)")


if __name__ == "__main__":
    main()
