#!/usr/bin/env python
"""e5 — 强基线初步对比 (DRAFT): GPU CO kernel vs Operon CPU.

目的: de-risk OUTLINE §五A 的"加速 2.4–29×"。那串旧数字是本机粗测、基线待具名。
本脚本把 GPU kernel 和 Operon (具名强 CPU baseline) 放到**同一份 pop、同一批树**上比
(两后端共用同一 bytecode: OperonLM 逐树重建 Operon 树并校验系数映射; kernel 直接吃
popio pop dict), 故 apples-to-apples。

测的是**部署路径、warm、各自调好的工作点**:
  * kernel = in-process .so (co_inproc.get_inproc_co, CUDA ctx 一次性) — 不是子进程
    binary (那个每次 fork + 落盘 + ctx-init, 是 kernel 最差封装)。先 1 次不计时 warm,
    再 N_REP 次取中位。
  * Operon = 持久进程池 (warm: 先在小切片上付池启动, 再计时真跑)。
  * kernel max_iter=50 (调好的工作点); Operon max_iter=200 (它逐树早停, cap 罕中)。
    诊断: 同时报 kernel @50 vs @200 的 wall 与 med_loss — wall 近 4× = 批跑满 cap
    (无逐树回收, 50 是诚实工作点); loss 持平 = inner-const-heavy 的质量差是 fp32/条件数
    天花板, 那 1.8x 不算 speedup (必须等质量才算)。

诚实口径 (写死在报告里):
  * 时钟未锁 (210MHz idle / 1410 max, 锁频需 sudo 待批) → DRAFT, 有方差; 取中位。
  * 本机 256 核; 报对 1 核 / 对 PARALLEL_NPROC 核 两个加速比, 并投影满核 (乐观上界)。
    绝不把对单核的数字当头条, 也不把 256 核满机投影当唯一基线 (那是最对抗性的口径)。
  * loss 一律 interp.loss_pop (fp64) 统一重算; 报每后端中位 loss vs 噪声地板
    (loss_pop(pop, c_true)) — 等质量才比时间。

跑: CUDA_VISIBLE_DEVICES=0 uv run python experiments/e5_strong_baseline/prelim.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

# 子进程/库用哪块 GPU; 默认 0, 可被外部 env 覆盖
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from cusr.benchmark import interp  # noqa: E402
from cusr.benchmark import popio  # noqa: E402
from cusr.benchmark.backends import OperonLM  # noqa: E402
from cusr.benchmark.workload.gen_synth import gen_pop  # noqa: E402
from cusr.kernel.co_inproc import get_inproc_co  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# kernel 原始 status → 统一码 (同 backends.CudaKernelPop): 4=FAIL_CHOLESKY→failed
_KERNEL_STATUS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2}

ALL_PRESETS = ["early-gen", "late-gen-bloated", "inner-const-heavy"]
SEED = 0
N_REP = 3            # GPU 计时重复 (时钟未锁, 取中位)
OPERON_MAX_ITER = 200
KERNEL_MAX_ITER = 50   # kernel 调好的工作点 (批跑满 cap, 大 cap 只是白烧 GPU)
DEVICE_ID = 0          # CUDA_VISIBLE_DEVICES 下的相对 id


def _median_loss(pop, c_final):
    losses = interp.loss_pop(pop, np.asarray(c_final, np.float64))
    finite = losses[np.isfinite(losses)]
    return float(np.median(finite)) if finite.size else float("inf")


def _status_frac(status, code):
    return float(np.mean(status == code)) if len(status) else 0.0


def time_kernel(pop, max_iter):
    """In-process .so (部署路径), warm, 取 N_REP 次 wall 中位。这是每代真实 CO 成本。"""
    co = get_inproc_co(device_id=DEVICE_ID, variant="fd")
    co.optimize(pop, max_iter=max_iter)  # warm (per-shape device alloc / caches)
    walls, res = [], None
    for _ in range(N_REP):
        t = time.perf_counter()
        res = co.optimize(pop, max_iter=max_iter)
        walls.append(time.perf_counter() - t)
    status = np.array([_KERNEL_STATUS_MAP.get(int(s), 2) for s in res["status"]],
                      np.int32)
    return dict(
        backend=f"kernel-inproc-fd(it={max_iter})",
        wall=statistics.median(walls),
        wall_all=walls,
        med_loss=_median_loss(pop, res["c_final"]),
        frac_converged=_status_frac(status, 0),
        frac_failed=_status_frac(status, 2),
    )


def time_operon(pop, nproc):
    backend = OperonLM(nproc=nproc, max_iter=OPERON_MAX_ITER)
    # warmup: 付池启动 (spawn worker + import pyoperon) + 缓存热, 让计时段 warm
    backend.fit_pop(popio.slice_pop(pop, min(128, pop["M"])))
    res = backend.fit_pop(pop)
    return dict(
        backend=f"operon(nproc={nproc})",
        nproc=nproc,
        wall=res.wall_core,
        wall_e2e=res.wall_e2e,
        med_loss=_median_loss(pop, res.c_final),
        frac_converged=_status_frac(res.status, 0),
        frac_failed=_status_frac(res.status, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4000,
                    help="种群树数 (实际在环 1k–4k; 大 M 更利 GPU)")
    ap.add_argument("--N", type=int, default=1000, help="每题数据点")
    ap.add_argument("--parallel-nproc", type=int, default=64,
                    help="并行 CPU 点核数 (满机 256 ≈ 再 /4)")
    ap.add_argument("--presets", nargs="+", default=ALL_PRESETS, choices=ALL_PRESETS)
    ap.add_argument("--smoke", action="store_true", help="小 M/单 preset, 验证管线")
    ap.add_argument("--tag", default="prelim", help="输出文件名前缀")
    args = ap.parse_args()
    if args.smoke:
        args.M, args.N, args.parallel_nproc = 200, 300, 8
        args.presets = ["inner-const-heavy"]
        args.tag = "smoke"

    n_cpu = os.cpu_count()
    print(f"# e5 strong-baseline DRAFT  M={args.M} N={args.N} seed={SEED}  "
          f"GPU={os.environ['CUDA_VISIBLE_DEVICES']}  kernel=in-process libcusr_co_fd.so")
    print(f"# clocks UNLOCKED (draft); cpu cores={n_cpu}; "
          f"operon max_iter={OPERON_MAX_ITER} kernel max_iter={KERNEL_MAX_ITER}\n")
    report = dict(M=args.M, N=args.N, seed=SEED, kernel="in-process libcusr_co_fd.so",
                  clocks="UNLOCKED-draft", cpu_cores=n_cpu,
                  operon_max_iter=OPERON_MAX_ITER, kernel_max_iter=KERNEL_MAX_ITER,
                  parallel_nproc=args.parallel_nproc, presets={})
    for preset in args.presets:
        pop = gen_pop(preset, args.M, args.N, SEED)
        noise_floor = _median_loss(pop, pop["c_true"])
        K_mean = float(pop["metas"][:, 3].mean())
        print(f"## {preset}  K_mean={K_mean:.2f}  noise_floor(med loss)={noise_floor:.3e}")

        k = time_kernel(pop, KERNEL_MAX_ITER)
        k_hi = time_kernel(pop, 200)   # 诊断: wall 近 4×? loss 持平?
        op1 = time_operon(pop, 1)
        opP = time_operon(pop, args.parallel_nproc)

        sp_1 = op1["wall"] / k["wall"] if k["wall"] > 0 else float("nan")
        sp_P = opP["wall"] / k["wall"] if k["wall"] > 0 else float("nan")
        sp_full_proj = sp_1 / n_cpu  # 投影满核 (假设近线性, 乐观上界)
        # 等质量门: kernel med_loss ≤ operon med_loss(×1.05) 才认 speedup 干净
        quality_matched = k["med_loss"] <= op1["med_loss"] * 1.05

        for r in (k, k_hi, op1, opP):
            print(f"   {r['backend']:26s} wall={r['wall']:.3f}s "
                  f"med_loss={r['med_loss']:.3e} "
                  f"conv={r['frac_converged']:.0%} fail={r['frac_failed']:.0%}")
        print(f"   [diag] kernel wall 50→200: {k['wall']:.3f}→{k_hi['wall']:.3f}s "
              f"(×{k_hi['wall']/k['wall']:.1f}); med_loss 50→200: "
              f"{k['med_loss']:.3e}→{k_hi['med_loss']:.3e}")
        print(f"   speedup(kernel@50): vs 1-core={sp_1:.2f}x | "
              f"vs {args.parallel_nproc}-core={sp_P:.3f}x | "
              f"proj vs {n_cpu}-core(linear)={sp_full_proj:.3f}x | "
              f"quality_matched={quality_matched}\n")

        report["presets"][preset] = dict(
            K_mean=K_mean, noise_floor=noise_floor,
            kernel=k, kernel_maxiter200=k_hi, operon_1core=op1, operon_parallel=opP,
            speedup_vs_1core=sp_1, speedup_vs_parallel=sp_P,
            speedup_proj_full_node=sp_full_proj, quality_matched=quality_matched,
        )

    out_path = OUT / f"{args.tag}_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
