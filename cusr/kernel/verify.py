"""verify.py — batch_lm GPU 结果 vs scipy LM 比对 + 并行 scipy baseline.

输入:
- data/pop.bin     — dump_evogp.py 写的种群
- data/c_final.bin — batch_lm 输出的 c_final
- data/status.bin  — batch_lm 输出的 status

每棵树 (跨 worker 并行):
1. 从 pop.bin 取 (nt, nv, ci, c_init, xs, ym_per_tree)
2. 跑 scipy.least_squares(method='lm') 拟合 (跨 process pool)
3. 跟 batch_lm 的 c_final 对比: throughput / loss-降 rate / scipy envelope / c_rel_err

输出: data/verify_report.md

CLI:
    uv run python verify.py [--nproc 16] [--gpu-elapsed <gpu_wall_seconds>]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

try:  # works both as package module (cusr.kernel.verify) and as a standalone script
    from .tree_interpreter import eval_batch
except ImportError:
    from cusr.kernel.tree_interpreter import eval_batch


HERE = Path(__file__).resolve().parent
POP_MAGIC = 0x4D4C344D


def load_pop_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        header_bytes = f.read(64)
        header = struct.unpack("<16i", header_bytes)
        magic, version, M, total_n, total_c, N, n_vars, K_max, max_stack = header[0:9]
        if magic != POP_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x} (want 0x{POP_MAGIC:08x})")
        nt = np.frombuffer(f.read(total_n * 4), dtype=np.int32).copy()
        nv = np.frombuffer(f.read(total_n * 4), dtype=np.float32).copy()
        ci = np.frombuffer(f.read(total_n * 4), dtype=np.int32).copy()
        metas = np.frombuffer(f.read(M * 16), dtype=np.int32).reshape(M, 4).copy()
        c_init = np.frombuffer(f.read(total_c * 4), dtype=np.float32).copy()
        xs = np.frombuffer(f.read(N * n_vars * 4), dtype=np.float32).reshape(N, n_vars).copy()
        ym = np.frombuffer(f.read(M * N * 4), dtype=np.float32).reshape(M, N).copy()
    return dict(M=M, total_nodes=total_n, total_c=total_c, N=N, n_vars=n_vars,
                K_max=K_max, max_stack=max_stack,
                nt=nt, nv=nv, ci=ci, metas=metas, c_init=c_init, xs=xs, ym=ym)


def _scipy_worker(payload):
    """Multiprocessing worker — picklable top-level fn.

    payload = (m, tree_nt, tree_nv, xs, y_target, c_init)
    returns  (m, c_final, loss, status_str, n_iter, initial_loss)
    """
    m, tree_nt, tree_nv, xs, y_target, c_init = payload
    # initial loss at c_init — needed for "loss-降" diagnostic
    ss = np.array([len(tree_nt)], dtype=np.int32)
    try:
        y_init = eval_batch(tree_nt, tree_nv, ss, xs, c_init)
        if np.all(np.isfinite(y_init)):
            init_loss = 0.5 * float(np.sum((y_init - y_target) ** 2))
        else:
            init_loss = float("inf")
    except Exception:
        init_loss = float("inf")

    c_final, loss, status_str, n_iter = scipy_fit_one(tree_nt, tree_nv, xs, y_target, c_init)
    return m, c_final, loss, status_str, n_iter, init_loss


def scipy_fit_one(tree_nt, tree_nv, xs, y_target, c_init, max_nfev=200):
    """跑 scipy LM, 返回 (c_final, final_loss, status_str, n_iter).

    status_str ∈ {'ok', 'maxiter', 'fail'}.
    """
    ss = np.array([len(tree_nt)], dtype=np.int32)

    def residual(c):
        try:
            yp = eval_batch(tree_nt, tree_nv, ss, xs, c)
            r = yp - y_target
            # 若有 NaN/Inf, 给 scipy 一个大的有限值, 不要 raise — scipy 直接挂
            if not np.all(np.isfinite(r)):
                r = np.where(np.isfinite(r), r, 1e6)
            return r
        except Exception:
            return np.full_like(y_target, 1e6)

    try:
        res = least_squares(
            residual, c_init.astype(np.float64),
            method='lm', max_nfev=max_nfev, xtol=1e-10, ftol=1e-10,
        )
        c_final = np.asarray(res.x, dtype=np.float32)
        loss = 0.5 * float(np.sum(res.fun ** 2))
        if not np.all(np.isfinite(c_final)) or not np.isfinite(loss):
            return c_final, loss, "fail", int(res.nfev)
        # res.status: -1=input err, 0=maxiter, 1+=converged
        if res.status >= 1:
            return c_final, loss, "ok", int(res.nfev)
        else:
            return c_final, loss, "maxiter", int(res.nfev)
    except Exception as e:
        return None, float("inf"), "fail", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=Path, default=HERE.parents[1] / "data" / "fixtures" / "pop.bin")
    ap.add_argument("--c-final", type=Path, default=HERE / "data" / "c_final.bin")
    ap.add_argument("--status", type=Path, default=HERE / "data" / "status.bin")
    ap.add_argument("--report", type=Path, default=HERE / "data" / "verify_report.md")
    ap.add_argument("--limit", type=int, default=0, help="only verify first N trees (0=all)")
    ap.add_argument("--nproc", type=int, default=min(16, os.cpu_count() or 1),
                    help="number of scipy worker processes (default: min(16, ncpu))")
    ap.add_argument("--gpu-elapsed", type=float, default=None,
                    help="if provided, recorded as batch_lm GPU wall time in the throughput section")
    args = ap.parse_args()

    print(f"[verify] loading {args.pop}", flush=True)
    pop = load_pop_bin(args.pop)
    M = pop["M"]
    print(f"           M_prob={M} N={pop['N']} n_vars={pop['n_vars']} total_c={pop['total_c']}", flush=True)

    gpu_c = np.fromfile(args.c_final, dtype=np.float32)
    gpu_status = np.fromfile(args.status, dtype=np.int32)
    if len(gpu_status) != M:
        raise ValueError(f"status.bin has {len(gpu_status)} entries, want M_prob={M}")

    limit = args.limit if args.limit > 0 else M

    # --- Build payloads for K>0 trees (scipy work, dispatched in parallel) ---
    print(f"[verify] building payloads for {limit} trees ...", flush=True)
    payloads = []
    tree_meta = {}  # m -> (node_off, n_nodes, c_off, K, my_nt, my_nv, my_c_init, my_y)
    for m in range(limit):
        node_off, n_nodes, c_off, K = pop["metas"][m].tolist()
        my_nt = pop["nt"][node_off:node_off+n_nodes].copy()
        my_nv = pop["nv"][node_off:node_off+n_nodes].copy()
        my_c_init = pop["c_init"][c_off:c_off+K].copy()
        my_y = pop["ym"][m]
        tree_meta[m] = (node_off, n_nodes, c_off, K, my_nt, my_nv, my_c_init, my_y)
        if K > 0:
            payloads.append((m, my_nt, my_nv, pop["xs"], my_y, my_c_init))

    # --- Run scipy in parallel ---
    nproc = max(1, int(args.nproc))
    print(f"[verify] dispatching {len(payloads)} scipy fits to {nproc} workers ...", flush=True)
    scipy_results = {}  # m -> (c_final, loss, status_str, n_iter, init_loss)
    t_sc = time.time()
    with mp.Pool(processes=nproc) as pool:
        for i, (m_idx, sc_c, sc_loss, sc_status, sc_iter, init_loss) in enumerate(
            pool.imap_unordered(_scipy_worker, payloads, chunksize=4)
        ):
            scipy_results[m_idx] = (sc_c, sc_loss, sc_status, sc_iter, init_loss)
            if (i + 1) % 100 == 0:
                print(f"        {i+1}/{len(payloads)}  elapsed={time.time()-t_sc:.1f}s", flush=True)
    scipy_elapsed = time.time() - t_sc
    scipy_throughput = len(payloads) / scipy_elapsed if scipy_elapsed > 0 else float("nan")
    print(f"[verify] scipy done in {scipy_elapsed:.1f}s "
          f"({scipy_throughput:.1f} trees/s wall, nproc={nproc})", flush=True)

    # --- Assemble per-tree record ---
    rec = []
    t_rec = time.time()
    for m in range(limit):
        node_off, n_nodes, c_off, K, my_nt, my_nv, my_c_init, my_y = tree_meta[m]
        gpu_status_m = int(gpu_status[m])
        gpu_c_final = gpu_c[c_off:c_off+K].copy()

        # GPU final loss (Python recompute)
        ss = np.array([n_nodes], dtype=np.int32)
        gpu_loss = float("inf")
        try:
            if K == 0:
                y_pred = eval_batch(my_nt, my_nv, ss, pop["xs"], np.zeros(0))
            else:
                y_pred = eval_batch(my_nt, my_nv, ss, pop["xs"], gpu_c_final)
            if np.all(np.isfinite(y_pred)):
                gpu_loss = 0.5 * float(np.sum((y_pred - my_y) ** 2))
        except Exception:
            gpu_loss = float("inf")

        # GPU initial loss (at c_init), for "loss-降" diagnostic
        gpu_init_loss = float("inf")
        try:
            if K == 0:
                gpu_init_loss = gpu_loss  # K=0 has no params, init==final
            else:
                y_init = eval_batch(my_nt, my_nv, ss, pop["xs"], my_c_init)
                if np.all(np.isfinite(y_init)):
                    gpu_init_loss = 0.5 * float(np.sum((y_init - my_y) ** 2))
        except Exception:
            pass

        # scipy result (from parallel pool, K=0 trees we skipped)
        if K == 0:
            sc_status = "k0_skip"
            sc_loss = gpu_loss
            sc_c_final = None
            sc_iter = 0
            c_rel = 0.0
        else:
            sc_c_final, sc_loss, sc_status, sc_iter, _ = scipy_results[m]
            if sc_c_final is not None and np.all(np.isfinite(sc_c_final)) and np.all(np.isfinite(gpu_c_final)):
                denom = float(np.linalg.norm(sc_c_final))
                if denom > 1e-12:
                    c_rel = float(np.linalg.norm(gpu_c_final - sc_c_final) / denom)
                else:
                    c_rel = float(np.linalg.norm(gpu_c_final))
            else:
                c_rel = float("nan")

        rec.append(dict(
            m=m, K=K,
            gpu_status=gpu_status_m, gpu_loss=gpu_loss, gpu_init_loss=gpu_init_loss,
            sc_status=sc_status, sc_loss=sc_loss, sc_iter=sc_iter,
            c_rel=c_rel,
            gpu_c=gpu_c_final.tolist(), sc_c=(sc_c_final.tolist() if sc_c_final is not None else None),
            c_init=my_c_init.tolist(),
            n_nodes=n_nodes,
        ))

    elapsed = time.time() - t_sc
    print(f"[verify] total (scipy + recompute): {elapsed:.1f}s", flush=True)

    # ===== Aggregate =====
    # GPU status codes: 0=conv 1=maxiter 2=fail_nan 3=k0_skip 4=fail_cholesky
    # scipy status string: 'ok'/'maxiter'/'fail'/'k0_skip'
    gpu_conv = sum(1 for r in rec if r["gpu_status"] == 0)
    gpu_maxiter = sum(1 for r in rec if r["gpu_status"] == 1)
    gpu_fail_nan = sum(1 for r in rec if r["gpu_status"] == 2)
    gpu_k0 = sum(1 for r in rec if r["gpu_status"] == 3)
    gpu_fail_chol = sum(1 for r in rec if r["gpu_status"] == 4)
    sc_conv = sum(1 for r in rec if r["sc_status"] == "ok")
    sc_maxiter = sum(1 for r in rec if r["sc_status"] == "maxiter")
    sc_fail = sum(1 for r in rec if r["sc_status"] == "fail")
    sc_k0 = sum(1 for r in rec if r["sc_status"] == "k0_skip")

    # 联合 (排除 K=0). GPU 收敛 = status 0; GPU 没收敛 = 1/2/4.
    only_gpu = [r for r in rec if r["K"] > 0 and r["gpu_status"] == 0 and r["sc_status"] != "ok"]
    only_sc = [r for r in rec if r["K"] > 0 and r["sc_status"] == "ok" and r["gpu_status"] != 0]
    both = [r for r in rec if r["K"] > 0 and r["sc_status"] == "ok" and r["gpu_status"] == 0]
    both_fail = [r for r in rec if r["K"] > 0 and r["sc_status"] != "ok" and r["gpu_status"] != 0]

    if both:
        c_rels = np.array([r["c_rel"] for r in both if np.isfinite(r["c_rel"])], dtype=float)
        # 排除 nan 后 percentile
        c_rels.sort()
        n = len(c_rels)
        def pct(p): return float(c_rels[min(int(round(p/100*n)), n-1)]) if n > 0 else float("nan")
        p50 = pct(50); p90 = pct(90); p99 = pct(99); pmax = float(c_rels[-1]) if n > 0 else float("nan")
        # fitness rel err on the same subset
        # 用对称归一化 max(sc, gpu, eps) 而不是 sc_loss, 否则 sc_loss=1e-10 类 case 让分母爆.
        loss_rels = []
        for r in both:
            if np.isfinite(r["gpu_loss"]) and np.isfinite(r["sc_loss"]):
                denom = max(abs(r["sc_loss"]), abs(r["gpu_loss"]), 1e-10)
                loss_rels.append(abs(r["gpu_loss"] - r["sc_loss"]) / denom)
        loss_rels = np.sort(np.asarray(loss_rels, dtype=float))
        nl = len(loss_rels)
        def lpct(p): return float(loss_rels[min(int(round(p/100*nl)), nl-1)]) if nl > 0 else float("nan")
        lp50 = lpct(50); lp90 = lpct(90); lp99 = lpct(99)
        # **fitness-equivalent rate**: 主要 acceptance metric.
        # 定义: gpu_loss <= max(1.05*sc_loss, sc_loss + 1e-10).
        # 这个比 c_rel_err 公平 — degenerate identifiability (c0+c1-c1 类) 让 c_rel_err
        # 看着糟, 但其实 loss 等价.
        n_fit_eq = 0
        for r in both:
            if np.isfinite(r["gpu_loss"]) and np.isfinite(r["sc_loss"]):
                if r["gpu_loss"] <= max(1.05 * r["sc_loss"], r["sc_loss"] + 1e-10):
                    n_fit_eq += 1
        fit_eq_rate = n_fit_eq / len(both) if both else float("nan")
    else:
        p50 = p90 = p99 = pmax = float("nan")
        lp50 = lp90 = lp99 = float("nan")
        fit_eq_rate = float("nan")
        n_fit_eq = 0

    # Top 10 大 c_rel_err 样本 (both converged)
    both_sorted = sorted([r for r in both if np.isfinite(r["c_rel"])], key=lambda r: -r["c_rel"])[:10]

    # --- Diagnostic 1: loss-降 rate (GPU collectively drove loss down?) ---
    # 仅针对 K>0 的树, gpu_loss 和 gpu_init_loss 都有限 + gpu_loss < gpu_init_loss
    loss_down_eligible = [r for r in rec if r["K"] > 0
                          and np.isfinite(r["gpu_loss"]) and np.isfinite(r["gpu_init_loss"])]
    loss_down_count = sum(1 for r in loss_down_eligible if r["gpu_loss"] < r["gpu_init_loss"])
    loss_down_rate = (loss_down_count / len(loss_down_eligible)
                      if loss_down_eligible else float("nan"))

    # --- Diagnostic 2: scipy envelope at k× (gpu_loss <= k * sc_loss) on both-converged ---
    def envelope_rate(k, eps=1e-10):
        if not both:
            return float("nan"), 0
        n = sum(1 for r in both
                if np.isfinite(r["gpu_loss"]) and np.isfinite(r["sc_loss"])
                and r["gpu_loss"] <= max(k * r["sc_loss"], r["sc_loss"] + eps))
        return n / len(both), n
    env_1_05_rate, env_1_05_n = envelope_rate(1.05)  # ≡ fitness_equiv
    env_2x_rate,  env_2x_n  = envelope_rate(2.0)
    env_10x_rate, env_10x_n = envelope_rate(10.0)

    # --- Throughput ---
    gpu_elapsed = args.gpu_elapsed  # seconds (user passes via --gpu-elapsed)
    gpu_throughput = (limit / gpu_elapsed) if (gpu_elapsed and gpu_elapsed > 0) else None
    speedup = (scipy_elapsed / gpu_elapsed) if (gpu_elapsed and gpu_elapsed > 0) else None

    # ===== Write report =====
    lines = []
    lines.append("# batch-lm-sr — GPU batched LM for SR populations")
    lines.append("")
    lines.append(f"对 **{limit} 棵真实 EvoGP 候选树** 跑常数拟合 "
                 f"(`{args.pop.name}`, K ∈ [0, {pop['K_max']}], 每棵 N={pop['N']} 个数据点).")
    lines.append("")
    lines.append("## 吞吐对比 / Throughput")
    lines.append("")
    lines.append("| 后端 (backend) | 硬件 | 墙钟 (wall time) | 吞吐 (throughput) |")
    lines.append("|---|---|---:|---:|")
    if gpu_elapsed is not None:
        lines.append(f"| **batch-lm-sr** (GPU) | NVIDIA GPU | {gpu_elapsed:.3f} s | "
                     f"{gpu_throughput:.0f} trees/s |")
    else:
        lines.append("| **batch-lm-sr** (GPU) | NVIDIA GPU | _(重跑时传 `--gpu-elapsed S`)_ | — |")
    lines.append(f"| scipy LM (CPU 并行) | {nproc} workers | {scipy_elapsed:.1f} s | "
                 f"{scipy_throughput:.1f} trees/s |")
    if speedup is not None:
        lines.append(f"| **加速比 (speedup)** | — | — | **~{speedup:.0f}×** |")
    lines.append("")
    lines.append("**超参数注**: batch-lm-sr 用 xtol=1e-5 / max_iter=50; scipy 用 xtol=1e-10 / max_nfev=200. "
                 "这些都是 placeholder, 后续调参会改, 把这组数字看作 baseline, 不是稳态值。")
    lines.append("")
    lines.append("## 状态分布 / Status distribution")
    lines.append("")
    lines.append("| 状态 (status) | batch-lm-sr | scipy |")
    lines.append("|---|---:|---:|")
    lines.append(f"| converged (收敛) | {gpu_conv} | {sc_conv} |")
    lines.append(f"| maxiter (跑满迭代) | {gpu_maxiter} | {sc_maxiter} |")
    lines.append(f"| fail (NaN/Inf 在 eval) | {gpu_fail_nan} | {sc_fail} |")
    lines.append(f"| fail (Cholesky breakdown, JᵀJ 退化) | {gpu_fail_chol} | (scipy 不暴露此分类) |")
    lines.append(f"| K=0 skip (无常数可优化) | {gpu_k0} | {sc_k0} |")
    lines.append("")
    lines.append("**联合统计 (排除 K=0)**:")
    lines.append("")
    lines.append(f"- 两边都收敛: **{len(both)}**")
    lines.append(f"- 只 batch-lm-sr 收敛: {len(only_gpu)}")
    lines.append(f"- 只 scipy 收敛: {len(only_sc)}")
    lines.append(f"- 两边都没收敛: {len(both_fail)}")
    lines.append("")
    lines.append("## 正确性诊断 / Correctness diagnostics")
    lines.append("")
    lines.append("三个独立视角, **都是诊断指标, 不是 acceptance gate**. "
                 "超参数 (xtol, max_iter, FD eps, λ) 后续都会调, 这些数字会跟着动。")
    lines.append("")
    lines.append("### (1) loss 是不是真的降了 / Did the loss actually go down?")
    lines.append("")
    lines.append(f"K>0 的树里, `final_loss < initial_loss(c_init)` 的比例: "
                 f"**{loss_down_rate*100:.1f}%** ({loss_down_count}/{len(loss_down_eligible)}).")
    lines.append("")
    lines.append("这是最松的 sanity check — \"LM 到底跑没跑\". 这一条过, 说明求解器没在空转。")
    lines.append("")
    lines.append("### (2) 跟 scipy 比, 落在 k× envelope 内的比例")
    lines.append("")
    lines.append("两边都收敛 subset 上对比:")
    lines.append("")
    lines.append("| envelope | 落入比例 | count |")
    lines.append("|---|---:|---:|")
    lines.append(f"| `gpu_loss ≤ 1.05× sc_loss` (跟 scipy 几乎等价, ≡ \"fitness-equivalent\") | "
                 f"**{env_1_05_rate*100:.1f}%** | {env_1_05_n}/{len(both)} |")
    lines.append(f"| `gpu_loss ≤ 2× sc_loss` (差距 2 倍以内) | **{env_2x_rate*100:.1f}%** | "
                 f"{env_2x_n}/{len(both)} |")
    lines.append(f"| `gpu_loss ≤ 10× sc_loss` (无 blow-up) | **{env_10x_rate*100:.1f}%** | "
                 f"{env_10x_n}/{len(both)} |")
    lines.append("")
    lines.append("### (3) 两边都收敛子集的分布诊断")
    lines.append("")
    lines.append("`c_rel_err = ||gpu_c − sc_c|| / ||sc_c||` (常数向量相对误差):")
    lines.append("")
    lines.append(f"- p50 = {p50:.3e}, p90 = {p90:.3e}, p99 = {p99:.3e}, max = {pmax:.3e}")
    lines.append("")
    lines.append("`fitness_rel_err = |gpu_loss − sc_loss| / max(gpu_loss, sc_loss, 1e-10)` (loss 对称相对误差):")
    lines.append("")
    lines.append(f"- p50 = {lp50:.3e}, p90 = {lp90:.3e}, p99 = {lp99:.3e}")
    lines.append("")
    lines.append("**关于大 c_rel_err 但小 fitness_rel_err**: 进化产出的树常有"
                 "**identifiability 退化** (例如 `c0 + c1 − c1` 里 c1 不影响 loss), "
                 "导致 c 向量不唯一但 loss 等价。所以 c_rel_err 在这种树上不公正, "
                 "fitness_rel_err 才是公平指标。下面 Top divergences 表里能看到这些案例。")
    lines.append("")
    lines.append("## Top 10 c_rel_err 离群案例 / Top divergences (两边都收敛)")
    lines.append("")
    lines.append("| m | K | n_nodes | c_rel | gpu_loss | sc_loss | gpu_c | sc_c | c_init |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in both_sorted:
        gpuc = ",".join(f"{v:.3e}" for v in r["gpu_c"]) if r["gpu_c"] else ""
        scc = ",".join(f"{v:.3e}" for v in r["sc_c"]) if r["sc_c"] else ""
        ci_str = ",".join(f"{v:.3e}" for v in r["c_init"])
        lines.append(f"| {r['m']} | {r['K']} | {r['n_nodes']} | {r['c_rel']:.3e} | "
                     f"{r['gpu_loss']:.3e} | {r['sc_loss']:.3e} | "
                     f"{gpuc} | {scc} | {ci_str} |")
    lines.append("---")
    lines.append("")
    lines.append("_本报告无 acceptance gate, 三个指标并列展示。超参数 (xtol, max_iter, FD eps, λ) 后续会调, "
                 "这组数字视为 **baseline 测量**, 不是稳态值。_")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"[verify] wrote {args.report}")
    print(f"\n--- summary ---")
    if gpu_elapsed is not None:
        print(f"throughput: batch-lm-sr {gpu_elapsed:.3f}s ({gpu_throughput:.0f} trees/s) "
              f"vs scipy×{nproc} {scipy_elapsed:.1f}s ({scipy_throughput:.1f} trees/s) "
              f"→ ~{speedup:.0f}×")
    else:
        print(f"throughput: scipy×{nproc} {scipy_elapsed:.1f}s ({scipy_throughput:.1f} trees/s) "
              f"— pass --gpu-elapsed to record GPU side")
    print(f"status: both_conv={len(both)} only_gpu={len(only_gpu)} only_sc={len(only_sc)} both_fail={len(both_fail)}")
    print(f"loss-down rate (K>0): {loss_down_rate*100:.1f}%")
    print(f"within scipy envelope: 1.05×={env_1_05_rate*100:.1f}%  "
          f"2×={env_2x_rate*100:.1f}%  10×={env_10x_rate*100:.1f}%")


if __name__ == "__main__":
    main()
