"""runner.py — E1 harness 执行器: oracle 缓存 + tier 判定 + 结果表.

口径见 PROTOCOL.md (v1). 用法:

    source ../../scripts/env.sh
    uv run python runner.py --pop ../008_batch_lm_sr/data/pop.bin \
        --backends scipy,torch,kernel,kernel_devjac,kernel_fusedfd \
        --limit 300 --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import cusr.kernel
from . import backends
from . import interp
from . import popio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # cusr/benchmark/ -> cusr/ -> repo root
KERNEL_DIR = Path(cusr.kernel.__file__).parent
WORKLOAD_DIR = ROOT / "data" / "workload"


def resolve_pop(spec: str) -> Path:
    """--pop 取值: 文件路径, 或 preset:NAME (查 data/workload/presets.json).
    sha1 不匹配只警告 (oracle 缓存键本来就是实际文件 sha1, 不会串)."""
    if not spec.startswith("preset:"):
        return Path(spec)
    name = spec[len("preset:"):]
    table = json.loads((WORKLOAD_DIR / "presets.json").read_text())
    table.pop("_comment", None)
    if name not in table:
        raise SystemExit(f"未知 preset '{name}', 可选: {', '.join(sorted(table))}")
    entry = table[name]
    path = WORKLOAD_DIR / entry["file"]
    if not path.exists():
        hint = entry.get("gen_cmd", "workload/harvest.py (seed 0)")
        raise SystemExit(f"preset '{name}' 的文件不存在: {path}\n  复现: {hint}")
    sha = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    if sha != entry["sha1_16"]:
        print(f"[warn] preset '{name}' sha1[:16]={sha} != 钉档 {entry['sha1_16']} "
              f"(文件被重新生成过? 结果仍有效但与钉档版本不可直接比)", file=sys.stderr)
    return path
PROTOCOL_VERSION = 2
TIER_ABS_FLOOR = 1e-10   # 协议 §3
# tier-A 的 0.5% 容差覆盖 scipy 在多进程池下的实测抖动上界 (~2.3e-3, 见 PROTOCOL §3)
TIER_A_MULT, TIER_B_MULT = 1.005, 1.05


# ---------------------------------------------------------------- tier 数学

def classify_tiers(loss_b: np.ndarray, loss_star: np.ndarray, K: np.ndarray):
    """返回 (tier_a, tier_b, eligible). K=0 或 oracle 无效的树不进 tier 统计."""
    eligible = (np.asarray(K) > 0) & np.isfinite(loss_star)
    thr_a = np.maximum(TIER_A_MULT * loss_star, loss_star + TIER_ABS_FLOOR)
    thr_b = np.maximum(TIER_B_MULT * loss_star, loss_star + TIER_ABS_FLOOR)
    with np.errstate(invalid="ignore"):
        tier_a = eligible & (loss_b <= thr_a)
        tier_b = eligible & (loss_b <= thr_b)
    return tier_a, tier_b, eligible


def tier_rate(tier: np.ndarray, eligible: np.ndarray) -> float:
    n = int(eligible.sum())
    return float(tier.sum() / n) if n else float("nan")


def selection_fidelity(loss_b: np.ndarray, loss_star: np.ndarray, eligible: np.ndarray):
    """CO 服务"选择": 后端 loss 排序 vs oracle loss 排序有多一致 (协议 §3 面板).

    只看 eligible 且两侧都有限的树. Spearman = 全局排序一致性; top-K 重合 = 进化真正
    在意的"选最好那部分"的保真度 (oracle 选出 loss 最低的 K 棵 vs 后端选出的交集比).
    fp32 在 top-10% 会因 loss 差低于精度分辨而洗牌, top-25/50% 几乎不受影响.
    纯 numpy (rank 后 pearson), 不引 scipy.stats.
    """
    msk = np.asarray(eligible) & np.isfinite(loss_b) & np.isfinite(loss_star)
    idx = np.where(msk)[0]
    out = dict(n=int(idx.size), spearman=None, top10=None, top25=None, top50=None)
    if idx.size < 3:
        return out
    a, b = loss_b[idx], loss_star[idx]
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))  # 名次
    out["spearman"] = float(np.corrcoef(ra, rb)[0, 1])
    for frac, key in ((0.1, "top10"), (0.25, "top25"), (0.5, "top50")):
        k = max(1, int(frac * idx.size))
        o_top = set(idx[np.argsort(b)[:k]].tolist())
        k_top = set(idx[np.argsort(a)[:k]].tolist())
        out[key] = len(o_top & k_top) / k
    return out


def noise_floor_path(pop_path: Path) -> Path:
    """合成 preset 的 c_true sidecar (gen_synth 落盘约定): <stem>.ctrue.npy 同目录.
    真实 preset 无此文件 → 无噪声地板 (那些树没有'真常数')."""
    return pop_path.parent / (pop_path.stem + ".ctrue.npy")


# ---------------------------------------------------------------- oracle

def get_oracle(pop_path: Path, pop: dict, limit: int | None, cache_dir: Path) -> dict:
    """scipy-fp64 紧公差 oracle (协议 §3), 按 (文件 sha1, limit) 缓存."""
    sha = hashlib.sha1(pop_path.read_bytes()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"oracle_{sha}_L{limit or 'all'}_serial.npz"
    if cache.exists():
        z = np.load(cache)
        return dict(loss_star=z["loss_star"], c_star=z["c_star"],
                    status=z["status"], cached=True, wall=float(z["wall"]))
    t0 = time.perf_counter()
    # oracle 必须串行: 多进程池对 ~1% 退化树非确定 (PROTOCOL §3), 串行 bit 级可复现; 有缓存, 慢点无所谓
    res = backends.ScipyPop(nproc=1).fit_pop(pop)
    loss_star = interp.loss_pop(pop, res.c_final)
    wall = time.perf_counter() - t0
    np.savez(cache, loss_star=loss_star, c_star=res.c_final,
             status=res.status, wall=wall)
    return dict(loss_star=loss_star, c_star=res.c_final, status=res.status,
                cached=False, wall=wall)


# ---------------------------------------------------------------- 环境

def make_backend(name: str):
    if name == "scipy":
        return backends.ScipyPop()
    if name == "torch":
        return backends.TorchPop()
    if name == "operon":
        return backends.OperonLM()
    if name == "pysr":          # PySR 原生预算 (p=0.14 世界里的真实配置)
        return backends.PySRBFGS(iterations=8, nrestarts=2, name="pysr")
    if name == "pysr200":       # 放宽预算档 (E1 报两档, W3 冒烟: 8 iter 拟不回 sin 频率)
        return backends.PySRBFGS(iterations=200, nrestarts=2, name="pysr200")
    bin_map = {"kernel": "batch_lm", "kernel_devjac": "batch_lm_devjac",
               "kernel_fusedfd": "batch_lm_fusedfd"}
    if name in bin_map:
        return backends.CudaKernelPop(KERNEL_DIR / bin_map[name], name=name)
    raise ValueError(f"unknown backend {name!r} "
                     f"(want scipy/torch/operon/pysr/pysr200/kernel[_devjac|_fusedfd])")


def _nvidia_smi() -> str | None:
    """解析 nvidia-smi 路径: WSL2 下不在裸 PATH (在 /usr/lib/wsl/lib), env.sh 也不加它.
    返回可执行路径或 None (无 GPU 机器)."""
    from shutil import which
    return which("nvidia-smi") or next(
        (p for p in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi")
         if Path(p).exists()), None)


def gpu_compute_procs() -> list[str]:
    smi = _nvidia_smi()
    if not smi:
        return []
    try:
        r = subprocess.run([smi, "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader"], capture_output=True,
                           encoding="utf-8", timeout=10)
        return [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001 — nvidia-smi 异常时不拦 (沿用旧行为)
        return []


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, encoding="utf-8").stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def env_info() -> dict:
    gpu = "?"
    smi = _nvidia_smi()
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, encoding="utf-8", timeout=10)
            gpu = (r.stdout or "?").strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    return dict(hostname=socket.gethostname(), gpu=gpu,
                git_sha_main=_git_sha(ROOT),
                git_sha_008=_git_sha(KERNEL_DIR),
                timestamp=datetime.now().isoformat(timespec="seconds"),
                protocol_version=PROTOCOL_VERSION)


# ---------------------------------------------------------------- 主流程

def run_bench(pop_path: Path, backend_names: list[str], *, limit: int | None = None,
              repeats: int = 3, smoke: bool = False, allow_busy: bool = False,
              out_dir: Path | None = None, cache_dir: Path | None = None) -> dict:
    pop_path = Path(pop_path)
    out_dir = Path(out_dir or HERE / "_out")
    cache_dir = Path(cache_dir or HERE / "_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    pop = popio.load_pop_bin(pop_path)
    if limit:
        pop = popio.slice_pop(pop, limit)
    M = pop["M"]
    K = pop["metas"][:, 3]

    # 噪声地板 (合成 preset): c_true sidecar 在则算 — 真常数代回的 loss = 统计下界.
    # interp.loss_pop 与 oracle / 各后端走同一条 fp64 重算路径 → 口径一致.
    nf = None
    nf_path = noise_floor_path(pop_path)
    if nf_path.exists():
        c_true = np.load(nf_path)
        if c_true.shape[0] >= pop["total_c"]:
            nf = dict(L_noise=interp.loss_pop(pop, c_true[:pop["total_c"]]))
        else:
            print(f"[warn] c_true sidecar 长度 {c_true.shape[0]} < total_c "
                  f"{pop['total_c']}, 跳过噪声地板", file=sys.stderr)

    needs_gpu = any(n.startswith(("kernel", "torch")) for n in backend_names)
    if needs_gpu:
        procs = gpu_compute_procs()
        if procs:
            msg = f"GPU 非空闲, 计时不可信: {procs}"
            if not (smoke or allow_busy):
                raise RuntimeError(msg + " (--allow-busy 可强行跑)")
            print(f"[warn] {msg}", file=sys.stderr)

    print(f"[oracle] scipy-fp64 紧公差 ... (pop={pop_path.name} M={M})", flush=True)
    oracle = get_oracle(pop_path, pop, limit, cache_dir)
    if oracle["cached"]:
        print("[oracle] cache hit", flush=True)
    else:
        print(f"[oracle] computed in {oracle['wall']:.1f}s", flush=True)

    if nf is not None:  # 审计金标准: oracle 自身够到地板的比例 (<100% = 高K触 nfev 上限)
        oa, ob, oe = classify_tiers(oracle["loss_star"], nf["L_noise"], K)
        nf["summary"] = dict(
            median=float(np.median(nf["L_noise"][oe])) if oe.any() else float("nan"),
            n=int(oe.sum()), oracle_reach_a=tier_rate(oa, oe), oracle_reach_b=tier_rate(ob, oe))

    results: dict = dict(meta=env_info(), pop=dict(path=str(pop_path), M=M,
                         N=pop["N"], n_vars=pop["n_vars"], K_max=pop["K_max"],
                         limit=limit, smoke=smoke, repeats=repeats),
                         oracle=dict(cached=oracle["cached"], wall=oracle["wall"],
                                     n_eligible=int(((K > 0) & np.isfinite(oracle["loss_star"])).sum())),
                         backends={})

    for name in backend_names:
        be = make_backend(name)
        print(f"[{name}] warmup ...", flush=True)
        try:
            # 预热须含 ≥1 棵 K>0 树, 否则 pysr 的 fit_batch JIT 落进首次计时 run
            k_pos = int(np.argmax(K > 0)) if bool((K > 0).any()) else 0
            be.fit_pop(popio.slice_pop(pop, min(M, max(2, k_pos + 1))))
            walls, res = [], None
            for rep in range(repeats):
                r = be.fit_pop(pop)
                walls.append(r.wall_e2e)
                res = r   # 确定性后端, 取最后一次的解
                print(f"[{name}] run {rep + 1}/{repeats}: e2e={r.wall_e2e:.3f}s "
                      f"core={r.wall_core:.3f}s", flush=True)
        except backends.KernelRunError as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr, flush=True)
            results["backends"][name] = dict(error=str(e))
            continue
        loss_b = interp.loss_pop(pop, res.c_final)
        tier_a, tier_b, eligible = classify_tiers(loss_b, oracle["loss_star"], K)
        selfid = selection_fidelity(loss_b, oracle["loss_star"], eligible)
        wall = float(np.median(walls))
        n_elig = int(eligible.sum())
        results["backends"][name] = dict(
            wall_e2e=wall, wall_e2e_runs=walls, wall_core=res.wall_core,
            tier_a_n=int(tier_a.sum()), tier_b_n=int(tier_b.sum()), n_eligible=n_elig,
            tier_a_rate=tier_rate(tier_a, eligible), tier_b_rate=tier_rate(tier_b, eligible),
            tput_a=float(tier_a.sum() / wall), tput_b=float(tier_b.sum() / wall),
            selection_fidelity=selfid,
            status_counts={s: int((res.status == c).sum()) for s, c in
                           [("converged", 0), ("iter_limit", 1), ("failed", 2), ("k0", 3)]},
            n_iter_median=(int(np.median(res.n_iter[res.n_iter >= 0]))
                           if (res.n_iter >= 0).any() else None),
            meta=res.meta)
        if nf is not None:  # 后端够到地板 (绝对参照); oracle≈地板时 ≈ tier-B
            fa, fb, fe = classify_tiers(loss_b, nf["L_noise"], K)
            results["backends"][name]["floor_a_rate"] = tier_rate(fa, fe)
            results["backends"][name]["floor_b_rate"] = tier_rate(fb, fe)

    if nf is not None:
        results["noise_floor"] = nf["summary"]
    _write_outputs(results, pop_path, out_dir)
    return results


def _write_outputs(results: dict, pop_path: Path, out_dir: Path):
    host = results["meta"]["hostname"]
    limit = results["pop"]["limit"]
    stem = f"{pop_path.stem}_L{limit or 'all'}_{host}"
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2, default=str))

    title = (f"# E1 op-bench — {pop_path.name} (M={results['pop']['M']}, limit={limit}"
             + (", SMOKE" if results["pop"]["smoke"] else f", repeats={results['pop']['repeats']}")
             + ")")
    oracle_line = (f"* oracle: scipy-fp64 紧公差, eligible {results['oracle']['n_eligible']} 树"
                   + (" (cache)" if results["oracle"]["cached"]
                      else f" ({results['oracle']['wall']:.1f}s)"))
    lines = [title,
             "",
             f"* host {host} / GPU {results['meta']['gpu']}",
             f"* git main={results['meta']['git_sha_main']} 008={results['meta']['git_sha_008']} "
             f"protocol v{results['meta']['protocol_version']} @ {results['meta']['timestamp']}",
             oracle_line,
             "* note: 本机数字仅 sanity, 论文数字按 008/RERUN_A100.md 重测",
             "",
             "| backend | dtype | e2e (s) | core (s) | tier-A | tier-B | tput-A (t/s) | tput-B (t/s) | conv/lim/fail | iters(med) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for name, r in results["backends"].items():
        if "error" in r:
            lines.append(f"| {name} | — | FAILED: {r['error'][:60]} | | | | | | | |")
            continue
        sc = r["status_counts"]
        lines.append(
            f"| {name} | {r['meta'].get('dtype', '?')} | {r['wall_e2e']:.3f} | "
            f"{r['wall_core']:.3f} | {r['tier_a_rate'] * 100:.1f}% | "
            f"{r['tier_b_rate'] * 100:.1f}% | {r['tput_a']:.0f} | {r['tput_b']:.0f} | "
            f"{sc['converged']}/{sc['iter_limit']}/{sc['failed']} | "
            f"{r['n_iter_median'] if r['n_iter_median'] is not None else '—'} |")
    lines.append("")
    lines.append("conv/lim/fail 与 iters 按各后端原生语义 (pysr: improved/no-improve, "
                 "iters=f_calls; kernel 不导出迭代数) — 细节见 json 的 meta.criteria")

    # 选择保真度面板 (协议 §3): CO 服务"选择", 后端 loss 排序 vs oracle 排序的一致性
    lines += ["", "## 选择保真度 (后端 loss 排序 vs oracle)", "",
              "| backend | Spearman | top-10% | top-25% | top-50% |",
              "|---|---:|---:|---:|---:|"]
    for name, r in results["backends"].items():
        if "error" in r:
            continue
        sf = r.get("selection_fidelity") or {}
        def _p(x):
            return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "—"
        sp = f"{sf['spearman']:.3f}" if sf.get("spearman") is not None else "—"
        lines.append(f"| {name} | {sp} | {_p(sf.get('top10'))} | "
                     f"{_p(sf.get('top25'))} | {_p(sf.get('top50'))} |")
    lines.append("")
    lines.append("top-K% = oracle 选出 loss 最低的 K% 棵树, 后端也选出 K%, 交集比例 "
                 "(进化选择真正在意的粒度). top-10% 偏低多因 fp32 在最优档分辨不足.")

    nf = results.get("noise_floor")
    if nf:  # 合成 preset 才有; 主推 oracle 审计, 后端列作佐证 (oracle≈地板 ⇒ ≈ tier-B)
        lines += ["", "## 噪声地板 (仅合成 preset; c_true 代回的 loss = 注入噪声的能量尺度)", "",
                  f"* 地板 loss 中位 = {nf['median']:.3e}  (n={nf['n']} 棵 K>0; 注入 1% 噪声)",
                  f"* **oracle (scipy-fp64) 够到地板: ≤1.005× {nf['oracle_reach_a'] * 100:.1f}%, "
                  f"≤1.05× {nf['oracle_reach_b'] * 100:.1f}%** ← 审计金标准 (它确实把真常数找回);"
                  f" <100% = 高 K 树触 max_nfev=200 上限 (量化 §3 已知妥协, 非缺陷)",
                  "",
                  "| backend | 够到地板 ≤1.005× | 够到地板 ≤1.05× |",
                  "|---|---:|---:|"]
        for name, r in results["backends"].items():
            if "error" in r or "floor_b_rate" not in r:
                continue
            lines.append(f"| {name} | {r['floor_a_rate'] * 100:.1f}% | {r['floor_b_rate'] * 100:.1f}% |")
        lines += ["",
                  "MLE 最优在有限样本下约低于地板 ~K/N (σ²(N−K) vs σ²N), K≪N 时 <1% — "
                  "oracle ≈ 地板属健康, 远低于才是拟合噪声. 后端够到地板 ≈ 其 tier-B "
                  "(因 oracle≈地板), 故此节主推 oracle 审计, 后端列作佐证."]

    md = "\n".join(lines) + "\n"
    (out_dir / f"{stem}.md").write_text(md)
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True,
                    help="pop.bin 路径, 或 preset:NAME (见 workload/presets.json)")
    ap.add_argument("--backends", default="scipy,kernel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="1 次重复 + 放宽 GPU 占用检查")
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()
    run_bench(resolve_pop(args.pop),
              [b.strip() for b in args.backends.split(",") if b.strip()],
              limit=args.limit or None,
              repeats=1 if args.smoke else args.repeats,
              smoke=args.smoke, allow_busy=args.allow_busy)


if __name__ == "__main__":
    main()
