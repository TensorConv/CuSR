"""闭环: 真 EvoGP 配置下 stock vs kernel-CO, 在构造 inner-const 题库上.

回答两件事 (一张表):
  1. 富采样 stock 行不行? (real_p0)  —— 上一轮 partial 已强烈提示 0 (结构对/常数差一丝
     或结构都没锁住). 这里跑全.
  2. **CO 在真配置下救不救得回?** (real_p1 vs real_p0)  —— 这是闭环. demo_unlock 证过 CO
     在弱配置 (palette+DefaultSelection) 下解锁; 这里换成真配置 (rich sampling + tournament +
     tree64) 看 CO 还顶不顶得住. 顶得住 → demonstrator 在真配置下成立, 不是配置砍出来的.

三条臂, 除 CO 旋钮 / 常数池外其余 (run_bench.run_one) 全一致:
  real_p0    : sample_cnt=10000, co_probability=0  —— 真 EvoGP stock
  real_p1    : sample_cnt=10000, co_probability=1  —— 真配置 + kernel CO (闭环主角)
  palette_p0 : const_samples={0,+-1,+-2,0.5}, co=0  —— 隔离"池粒度" (6值表能算术拼精确常数)

backend = 真 kernel CudaKernelLM (008 batch_lm via bridge), 只在 real_p1 调用.
judge 已带 >50 节点守卫 (防 bloated 树卡死 sympy, 对所有臂对称).

跑: source scripts/env.sh
    uv run python experiments/009_sr_benchmark/closed_loop_constructed.py --smoke   # u_sin25×2seed×3臂
    uv run python experiments/009_sr_benchmark/closed_loop_constructed.py            # 全量 6题×5seed×3臂
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.co_backend import CudaKernelLM

sys.path.insert(0, str(HERE))  # sibling run-script
from run_bench import run_one  # noqa: E402

TARGETS = [
    dict(id="u_sin25",    expr="2.0*sin(2.5*x0)",          nv=1, rng=((0.0, 3.0),),            nc=2, ni=1),
    dict(id="u_sincos",   expr="sin(2.5*x0)*cos(1.3*x1)",  nv=2, rng=((0.0, 3.0), (0.0, 3.0)), nc=2, ni=2),
    dict(id="u_expdecay", expr="3.0*exp(-0.7*x0)",         nv=1, rng=((0.0, 3.0),),            nc=2, ni=1),
    dict(id="u_quad",     expr="x0*x0 + 3.5*x0",           nv=1, rng=((-3.0, 3.0),),           nc=1, ni=0),
    dict(id="u_sinphase", expr="sin(0.7*x0 + 1.3)",        nv=1, rng=((0.0, 6.0),),            nc=2, ni=2),
    dict(id="u_linear",   expr="1.7*x0 + 0.6",             nv=1, rng=((-3.0, 3.0),),           nc=2, ni=0),
]
PALETTE = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]
N_SEEDS = 5

# (臂名, co_probability, run_one 的常数池 kwargs)
ARMS = [
    ("real_p0",    0.0, dict(sample_cnt=10000)),
    ("real_p1",    1.0, dict(sample_cnt=10000)),
    ("palette_p0", 0.0, dict(const_samples=PALETTE)),
]


def make_prob(t):
    return sb.Problem(id=t["id"], source="constructed", true_expr=t["expr"], n_vars=t["nv"],
                      var_ranges=t["rng"], n_consts=t["nc"], n_inner_consts=t["ni"],
                      difficulty="demo", is_control=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="u_sin25 × 2 seed × 3 臂 (验 wiring)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--max-iter", type=int, default=50, help="kernel LM 迭代数")
    args = ap.parse_args()
    targets = [t for t in TARGETS if t["id"] == "u_sin25"] if args.smoke else TARGETS
    n_seeds = 2 if args.smoke else args.seeds
    fn = "closed_loop_smoke.jsonl" if args.smoke else "closed_loop_constructed.jsonl"

    backend = CudaKernelLM(max_iter=args.max_iter)  # 只 real_p1 用
    out = (HERE / "runs" / fn).open("w")
    arm_names = [a[0] for a in ARMS]
    print(f"闭环: 真配置 stock vs kernel-CO   seeds={n_seeds}" + ("  [SMOKE]" if args.smoke else ""))
    print(f"{'id':12}{'ni':>3}" + "".join(f"{a:>12}" for a in arm_names) + "   true_expr")

    tot = {a: 0 for a in arm_names}
    inner = {a: 0 for a in arm_names}
    n_inner = sum(1 for t in targets if t["ni"] > 0)
    for t in targets:
        prob = make_prob(t)
        cell = {}
        for aname, p, kw in ARMS:
            rec = 0
            for seed in range(n_seeds):
                if p > 0:
                    backend.cum_stats.clear()
                t0 = time.time()
                r = run_one(prob, backend, seed=seed, co_probability=p, **kw)
                r.update(arm=aname, seed=seed, p=p, wall=round(time.time() - t0, 1))
                if p > 0:
                    r["kernel_stats"] = dict(backend.cum_stats)
                out.write(json.dumps(r) + "\n"); out.flush()
                rec += int(r["lenient"])
            cell[aname] = rec
            tot[aname] += rec
            if t["ni"] > 0:
                inner[aname] += rec
        print(f"{t['id']:12}{t['ni']:>3}"
              + "".join(f"{f'{cell[a]}/{n_seeds}':>12}" for a in arm_names)
              + f"   {t['expr']}")

    denom, idenom = len(targets) * n_seeds, n_inner * n_seeds
    print(f"\n{'TOTAL':12}{'':>3}" + "".join(f"{f'{tot[a]}/{denom}':>12}" for a in arm_names))
    if idenom:
        print(f"{'inner ni>0':12}{'':>3}" + "".join(f"{f'{inner[a]}/{idenom}':>12}" for a in arm_names))
    out.close()
    print(f"\n  闭环判读: real_p1 > real_p0 → CO 在真配置下解锁 (demonstrator 成立)")
    print(f"            real_p0 ≈ 0 → claim-3 的 p=0 baseline 前提 (真配置版) 成立")
    print(f"\nwrote {HERE / 'runs' / fn}")


if __name__ == "__main__":
    main()
