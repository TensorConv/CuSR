"""真实 EvoGP 配置在构造 inner-const 题库上的 stock 表现 (p=0, 不开 CO).

回答: 富采样 (sample_cnt=10000) 的 stock 能不能恢复内部常数 (2.5/0.7/1.3...),
还是说就算给真配置也是堵死的真墙?

两个条件, 其余配置 (run_bench.run_one: TournamentSelection / tree_len=64 / layer=6 /
mutation=0.1 / 早停 R2>=0.9999) 全部一致, 只动常数池 —— 干净隔离"池粒度":
  real_10000  : const_range=[-5,5] x sample_cnt=10000  (真 EvoGP)
  palette_6   : const_samples={0,+-1,+-2,0.5}          (demo 用的 6 值表, 同真配置)

对照: demo_unlock.jsonl 的 stock 是 6 值表 BUT DefaultSelection/小树/mutation0.2 ——
所以 demo vs 这里的 palette_6 差的是"选择算子+树规模", real_10000 vs palette_6 差的是"池粒度".

注: run_bench.run_one 的 judge 已加节点守卫 (>50 节点跳过 sympy 判未恢复), 防 bloated 树
卡死 sympy (SIGALRM 挡不住 C 层卡死). 守卫对两个臂对称, 不偏比较.

跑: source scripts/env.sh
    uv run python experiments/009_sr_benchmark/real_evogp_constructed.py --smoke  # u_sin25×3seed, 验守卫
    uv run python experiments/009_sr_benchmark/real_evogp_constructed.py          # 全量
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.co_backend import TorchLM

sys.path.insert(0, str(HERE))  # sibling run-script
from run_bench import run_one  # noqa: E402

# 与 demo_unlock.py 完全一致的 6 道构造题 (关键常数全在 6 值表外)
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

CONDITIONS = [
    ("real_10000", dict(sample_cnt=10000)),
    ("palette_6",  dict(const_samples=PALETTE)),
]


def make_prob(t):
    return sb.Problem(id=t["id"], source="constructed", true_expr=t["expr"], n_vars=t["nv"],
                      var_ranges=t["rng"], n_consts=t["nc"], n_inner_consts=t["ni"],
                      difficulty="demo", is_control=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 u_sin25 × 3 seed (含上次卡死的 seed 2), 验证节点守卫挡住卡死")
    args = ap.parse_args()
    targets = [t for t in TARGETS if t["id"] == "u_sin25"] if args.smoke else TARGETS
    n_seeds = 3 if args.smoke else N_SEEDS
    fn = "real_evogp_smoke.jsonl" if args.smoke else "real_evogp_constructed.jsonl"

    backend = TorchLM(dtype="float32")  # p=0 -> 永不调用, run_one 只需要个对象
    out = (HERE / "runs" / fn).open("w")
    print(f"stock EvoGP (p=0) on constructed inner-const set   seeds={n_seeds}"
          + ("  [SMOKE]" if args.smoke else ""))
    print(f"{'id':12}{'n_in':>5}" + "".join(f"{c:>14}" for c, _ in CONDITIONS) + "   true_expr")
    tot = {c: 0 for c, _ in CONDITIONS}
    inner_tot = {c: 0 for c, _ in CONDITIONS}  # 仅 ni>0 子集 —— CO 该解锁的战场
    n_inner = sum(1 for t in targets if t["ni"] > 0)
    for t in targets:
        prob = make_prob(t)
        cell = {}
        for cname, kw in CONDITIONS:
            rec = 0
            for seed in range(n_seeds):
                r = run_one(prob, backend, seed=seed, co_probability=0.0, **kw)
                r["condition"] = cname
                out.write(json.dumps(r) + "\n"); out.flush()
                rec += int(r["lenient"])
            cell[cname] = rec
            tot[cname] += rec
            if t["ni"] > 0:
                inner_tot[cname] += rec
        print(f"{t['id']:12}{t['ni']:>5}"
              + "".join(f"{f'{cell[c]}/{n_seeds}':>14}" for c, _ in CONDITIONS)
              + f"   {t['expr']}")
    denom, inner_denom = len(targets) * n_seeds, n_inner * n_seeds
    print(f"\n{'TOTAL(all)':12}{'':>5}" + "".join(f"{f'{tot[c]}/{denom}':>14}" for c, _ in CONDITIONS))
    if inner_denom:
        print(f"{'inner(ni>0)':12}{'':>5}" + "".join(f"{f'{inner_tot[c]}/{inner_denom}':>14}" for c, _ in CONDITIONS))
    out.close()
    print(f"\nwrote {HERE / 'runs' / fn}")


if __name__ == "__main__":
    main()
