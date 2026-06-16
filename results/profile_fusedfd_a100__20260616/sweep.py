"""sweep.py — per-kernel CUDA-event profiling of batch_lm_fusedfd across M.

Runs the -DPROFILE build (cusr/kernel/batch_lm_fusedfd_prof), which emits a
PROFILE_JSON line splitting the LM loop into per-kernel GPU time + host residual,
and separates one-time process setup (CUDA ctx + device alloc + initial eval)
from the steady-state loop. Median of REPEATS timed runs (WARMUP discarded).

This is CHARACTERIZATION (where does CO time go), not optimization. No root.
Pin one GPU via CUDA_VISIBLE_DEVICES before running (gpu_guard precheck).

Outputs (data/): prof_M<M>.json, profile_summary.json, profile_breakdown.csv
Plots  (plots/): loop_breakdown_frac.png, wall_decomposition.png
"""
import json, subprocess, re, statistics, os, sys, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # results/<dir>/sweep.py -> repo root
BIN  = ROOT / "cusr/kernel/batch_lm_fusedfd_prof"
SYN  = ROOT / "data/workload/synth"
POP  = lambda M: SYN / f"synth_early-gen_M{M}_N1000_seed0.bin"
MS       = [1000, 4000, 16000, 64000, 256000]
REPEATS  = 3
WARMUP   = 1
SCRATCH  = Path("/tmp/cusr_prof_scratch")
PROF_RE  = re.compile(r"PROFILE_JSON (\{.*\})")
# loop-time category order (largest first) + a synthetic 'host_residual' bucket
CATS = ["fd_jacobian", "eval", "build_jtj", "solve", "residual", "loss",
        "memcpy_H2D", "memcpy_D2H"]

def run_once(M):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(BIN), str(POP(M)), str(SCRATCH), "--quiet"],
                       capture_output=True, text=True)
    m = PROF_RE.search(p.stdout)
    if not m:
        sys.stderr.write(p.stdout + p.stderr)
        raise RuntimeError(f"no PROFILE_JSON for M={M} (binary built? GPU free?)")
    return json.loads(m.group(1))

def main():
    if not BIN.exists():
        sys.exit(f"missing {BIN} — build it: make -C cusr/kernel batch_lm_fusedfd_prof")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)
    rows = []
    for M in MS:
        samples = []
        for r in range(WARMUP + REPEATS):
            d = run_once(M)
            tag = "warmup" if r < WARMUP else f"rep{r-WARMUP}"
            if r >= WARMUP:
                samples.append(d)
            print(f"  M={M:>6} {tag}: loop={d['loop_ms']:7.1f}ms setup={d['setup_ms']:6.1f}ms "
                  f"fd={d['cats']['fd_jacobian']['ms']:7.1f}ms host_resid={d['host_residual_ms']:5.1f}ms")
        med = lambda key: statistics.median(s[key] for s in samples)
        medc = lambda c: statistics.median(s["cats"][c]["ms"] for s in samples)
        agg = dict(M=M, N=samples[0]["N"], iters_run=samples[0]["iters_run"],
                   load_ms=med("load_ms"), setup_ms=med("setup_ms"),
                   loop_ms=med("loop_ms"), total_ms=med("total_ms"),
                   gpu_sum_ms=med("gpu_sum_ms"), host_residual_ms=med("host_residual_ms"),
                   cats={c: dict(ms=medc(c), launches=samples[0]["cats"][c]["launches"]) for c in CATS})
        rows.append(agg)
        (HERE / "data" / f"prof_M{M}.json").write_text(json.dumps(agg, indent=2))

    (HERE / "data" / "profile_summary.json").write_text(json.dumps(rows, indent=2))
    with open(HERE / "data" / "profile_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "iters", "loop_ms", "setup_ms", "total_ms", "gpu_sum_ms",
                    "host_residual_ms"] + [f"{c}_ms" for c in CATS] +
                   [f"{c}_pct_loop" for c in CATS] + ["host_residual_pct_loop"])
        for r in rows:
            lp = r["loop_ms"]
            w.writerow([r["M"], r["iters_run"], f"{r['loop_ms']:.3f}", f"{r['setup_ms']:.3f}",
                        f"{r['total_ms']:.3f}", f"{r['gpu_sum_ms']:.3f}", f"{r['host_residual_ms']:.3f}"]
                       + [f"{r['cats'][c]['ms']:.3f}" for c in CATS]
                       + [f"{100*r['cats'][c]['ms']/lp:.2f}" for c in CATS]
                       + [f"{100*r['host_residual_ms']/lp:.2f}"])

    # ---- plot 1: LM-loop time fraction (stacked %), the headline ----
    labels = [f"{m//1000}k" if m >= 1000 else str(m) for m in MS]
    stack_cats = CATS + ["host_residual"]
    colors = {"fd_jacobian": "#d1495b", "eval": "#edae49", "build_jtj": "#66a182",
              "solve": "#2e4057", "residual": "#8d96a3", "loss": "#c3cbd1",
              "memcpy_H2D": "#00798c", "memcpy_D2H": "#0db0c4", "host_residual": "#7d4f9c"}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bottom = [0.0] * len(MS)
    for c in stack_cats:
        vals = [(r["host_residual_ms"] if c == "host_residual" else r["cats"][c]["ms"])
                / r["loop_ms"] * 100 for r in rows]
        ax.bar(labels, vals, bottom=bottom, label=c, color=colors[c], edgecolor="white", linewidth=.4)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of LM-loop wall"); ax.set_xlabel("population M (trees)")
    ax.set_ylim(0, 100)
    ax.set_title("Where the LM loop spends time — fused-FD (A100, fp32)\n"
                 "fd_jacobian dominates; host/transfer is negligible")
    ax.legend(ncol=2, fontsize=8, loc="lower center", framealpha=.9)
    for i, r in enumerate(rows):  # annotate fd_jacobian share
        ax.text(i, r["cats"]["fd_jacobian"]["ms"] / r["loop_ms"] * 50,
                f"{100*r['cats']['fd_jacobian']['ms']/r['loop_ms']:.0f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "plots" / "loop_breakdown_frac.png", dpi=130)

    # ---- plot 2: wall decomposition setup vs loop (absolute, log-y) ----
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = range(len(MS))
    setup = [r["setup_ms"] for r in rows]; loop = [r["loop_ms"] for r in rows]
    ax.bar(x, setup, label="one-time setup (CUDA ctx + alloc + init eval)", color="#b0b7c0")
    ax.bar(x, loop, bottom=setup, label="LM loop (per-CO work)", color="#d1495b")
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("wall (ms, log)"); ax.set_xlabel("population M (trees)")
    ax.set_title("Process wall = one-time setup + LM loop\n"
                 "setup is ~constant => dominates small M, amortized at scale")
    for i, r in enumerate(rows):
        ax.text(i, setup[i] + loop[i], f"loop {loop[i]/ (setup[i]+loop[i])*100:.0f}%\nof wall",
                ha="center", va="bottom", fontsize=7.5)
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(HERE / "plots" / "wall_decomposition.png", dpi=130)

    # ---- console summary ----
    print("\n=== summary (median of", REPEATS, "runs) ===")
    print(f"{'M':>7} {'loop_ms':>8} {'setup_ms':>8} {'fd %loop':>8} {'eval %':>7} {'host %':>7} {'loop %wall':>10}")
    for r in rows:
        lp = r["loop_ms"]
        print(f"{r['M']:>7} {lp:>8.1f} {r['setup_ms']:>8.1f} "
              f"{100*r['cats']['fd_jacobian']['ms']/lp:>7.1f}% {100*r['cats']['eval']['ms']/lp:>6.1f}% "
              f"{100*r['host_residual_ms']/lp:>6.1f}% {100*lp/r['total_ms']:>9.1f}%")

if __name__ == "__main__":
    main()
