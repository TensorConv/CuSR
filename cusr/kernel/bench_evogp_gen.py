"""bench_evogp_gen.py — EvoGP 纯进化 (无 CO) 每代墙钟时间 (W0-6).

为什么测: 这是 in-loop CO overhead 比例的分母 —— batch_lm 一次全种群拟合
~0.7-0.8 s (1000 树), 占一代总时间的比例决定 p-sweep tradeoff 怎么讲.
001/007 从来没记录过这个数.

协议: 每个 pop size 跑 warmup 代 (JIT/分配预热, 不计) + 计时代;
每代前后 torch.cuda.synchronize() 取真实墙钟. 报告 mean/p50/p90 +
前 10 代 vs 后 10 代均值 (bloat 让树变大, 每代变慢, 漂移本身是 workload 信息).

用法:
    uv run python bench_evogp_gen.py [--pops 1000,4000] [--gens 50] [--warmup 3]

注意: 单独跑, 别和其它 GPU 任务并发; 记录跑在哪张卡上 (本机 RTX 5070 Ti
laptop 的数字投稿前要在 A100 重测, 见 RERUN_A100.md).
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

try:  # works both as package module and as a standalone script
    from .dump_evogp import _build_evogp, _load_feynman_problem, _sample_xy
except ImportError:
    from cusr.kernel.dump_evogp import _build_evogp, _load_feynman_problem, _sample_xy


def bench_one(pop: int, gens: int, warmup: int, dataset: str, N: int, seed: int) -> dict:
    prob = _load_feynman_problem(dataset)
    X_np, y_np = _sample_xy(prob, N, seed)
    X_t = torch.from_numpy(np.ascontiguousarray(X_np)).cuda()
    y_t = torch.from_numpy(np.ascontiguousarray(y_np.reshape(-1, 1))).cuda()
    algo, pipeline = _build_evogp(X_t, y_t, pop=pop, n_vars=X_np.shape[1], seed=seed)

    for _ in range(warmup):
        pipeline.step()
    torch.cuda.synchronize()

    per_gen = []
    for _ in range(gens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipeline.step()
        torch.cuda.synchronize()
        per_gen.append(time.perf_counter() - t0)

    per_gen_ms = [t * 1e3 for t in per_gen]
    head = statistics.mean(per_gen_ms[:10])
    tail = statistics.mean(per_gen_ms[-10:])
    return dict(
        pop=pop, gens=gens,
        mean_ms=statistics.mean(per_gen_ms),
        p50_ms=statistics.median(per_gen_ms),
        p90_ms=sorted(per_gen_ms)[int(0.9 * len(per_gen_ms))],
        head10_ms=head, tail10_ms=tail,
        total_s=sum(per_gen),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="feynman/I.18.12")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--pops", default="1000,4000")
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    print(f"[bench] GPU={gpu}  dataset={args.dataset}  N={args.N}  "
          f"gens={args.gens} (+{args.warmup} warmup)  seed={args.seed}\n")

    for pop in [int(p) for p in args.pops.split(",")]:
        r = bench_one(pop, args.gens, args.warmup, args.dataset, args.N, args.seed)
        print(f"pop={r['pop']:<6} per-gen: mean={r['mean_ms']:.1f}ms  p50={r['p50_ms']:.1f}ms  "
              f"p90={r['p90_ms']:.1f}ms  head10={r['head10_ms']:.1f}ms  tail10={r['tail10_ms']:.1f}ms  "
              f"total({r['gens']}gen)={r['total_s']:.2f}s")


if __name__ == "__main__":
    main()
