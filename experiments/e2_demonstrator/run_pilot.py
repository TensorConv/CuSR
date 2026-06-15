"""W5 E2 demonstrator pilot (<=6/16 checkpoint).

p in {0, 1.0} x ~5 inner-const problems x N seeds, kernel CO backend. Answers the
plan's three pilot questions at once:
  1. single-run cost (wall_s) -- scheduling basis. LAPTOP is a rough proxy only;
     paper-grade cost is A100 (RERUN on the cluster).
  2. is p=0 baseline recovery truly ~0?  -- claim-3's empirical premise.
  3. does p=1.0 (kernel CO) pull recovery apart from p=0?

Arm definition (the only difference between arms is the CO knob):
  p=0   -> honest stock EvoGP: co_probability=0, no CO call, selection on raw
           fitness; EvoGP's own discrete-const mutation still runs.
  p=1.0 -> kernel CO on every selected candidate (CudaKernelLM, tree->pop.bin->batch_lm).
Same EvoGP config, same seed, same backend object across both arms.

NOT auto-run. Full pilot = 5 problems x 2 arms x 10 seeds = 100 EvoGP runs.
  uv run python experiments/009_sr_benchmark/run_pilot.py --smoke   # 1 problem x 1 seed x 2 arms (wiring)
  uv run python experiments/009_sr_benchmark/run_pilot.py           # full pilot
  uv run python experiments/009_sr_benchmark/run_pilot.py --seeds 10 --problems korns_7 korns_12
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

from cusr.demonstrator import problems as P
from cusr.demonstrator.co_backend import CudaKernelLM

sys.path.insert(0, str(HERE))  # sibling run-script
from run_bench import run_one  # noqa: E402

ARMS = [0.0, 1.0]  # p=0 (stock) vs p=1.0 (kernel CO); intermediate p is the full sweep
N_PILOT_FEYNMAN = 3


def pilot_problems(cat, n_feyn=N_PILOT_FEYNMAN):
    """Korns-7/12 + the highest-n_inner Feynman (the 'high n_inner SRSD' set)."""
    by_id = {p.id: p for p in cat}
    picks = [by_id[i] for i in ("korns_7", "korns_12") if i in by_id]
    feyn = sorted((p for p in cat if p.source == "feynman" and p.n_inner_consts > 0),
                  key=lambda p: (-p.n_inner_consts, p.id))
    return picks + feyn[:n_feyn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 problem x 1 seed x 2 arms")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--max-iter", type=int, default=50, help="kernel LM iterations")
    args = ap.parse_args()

    cat = P.load_catalogue()
    by_id = {p.id: p for p in cat}
    if args.problems:
        probs = [by_id[i] for i in args.problems if i in by_id]
    else:
        probs = pilot_problems(cat)
    if args.smoke:
        probs, n_seeds = probs[:1], 1
    else:
        n_seeds = args.seeds

    backend = CudaKernelLM(max_iter=args.max_iter)
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    fn = out / ("pilot_smoke.jsonl" if args.smoke else "pilot_results.jsonl")
    sink = fn.open("w")

    print(f"E2 pilot  problems={[p.id for p in probs]}  arms p={ARMS}  seeds={n_seeds}")
    print(f"  (laptop = rough cost proxy; recovery is eval-budget axis, A100 for paper cost)\n")
    print(f"{'id':14}{'p':>5}{'seed':>5}{'n_in':>5}{'lenient':>8}{'gens':>5}"
          f"{'co_fits':>8}{'kern/fb':>9}{'wall':>7}")
    rows = []
    for prob in probs:
        for p in ARMS:
            for seed in range(n_seeds):
                backend.cum_stats.clear()
                t0 = time.time()
                r = run_one(prob, backend, seed=seed, co_probability=p)
                r["p"] = p
                r["seed"] = seed
                r["kernel_stats"] = dict(backend.cum_stats)
                rows.append(r)
                sink.write(json.dumps(r) + "\n"); sink.flush()
                ks = backend.cum_stats
                kfb = f"{ks.get('n_kernel', 0)}/{ks.get('n_fallback', 0)}" if p > 0 else "-"
                print(f"{r['id']:14}{p:>5.2f}{seed:>5}{r['n_inner']:>5}"
                      f"{str(r['lenient']):>8}{r['gens']:>5}{r['co_fits']:>8}{kfb:>9}"
                      f"{r['wall_s']:>6}s")
    sink.close()

    # --- the three pilot questions ---
    print("\n=== recovery (lenient) by arm ===")
    for p in ARMS:
        arm = [r for r in rows if r["p"] == p]
        rec = sum(int(r["lenient"]) for r in arm)
        wall = sum(r["wall_s"] for r in arm) / max(1, len(arm))
        label = "stock EvoGP (baseline)" if p == 0 else "kernel CO"
        print(f"  p={p:.2f} {label:24}: {rec}/{len(arm)} recovered   "
              f"avg {wall:.0f}s/run (laptop proxy)")
    base = sum(int(r['lenient']) for r in rows if r['p'] == 0)
    co = sum(int(r['lenient']) for r in rows if r['p'] == 1.0)
    print(f"\n  Q2 baseline (p=0) recovery ~0?  -> {base} recovered "
          f"({'YES, ~0' if base == 0 else 'NO -> claim-3 downgrades to amplification'})")
    print(f"  Q3 kernel CO pulls apart?       -> p=0:{base} vs p=1.0:{co} "
          f"({'separates' if co > base else 'flat -> retreat to E1+HPC'})")
    print(f"\nwrote {fn} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
