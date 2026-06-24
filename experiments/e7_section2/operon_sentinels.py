#!/usr/bin/env python
"""operon_sentinels.py — confirm the REUSED e6 Operon timing hasn't drifted.

Re-runs a few Operon configs that exist in e6 and compares wall_core + throughput
to the stored e6 values. Operon is CPU + deterministic compute on a byte-identical
pop, so a large gap would mean machine-state drift (which would invalidate reusing
the 162 e6 records). Decision (fixed UP FRONT): drift <15% = clean; 15-30% = note;
>30% = STOP-AND-REPORT (do not rely on the reused records).

Pop identity is asserted (gen_synth byte-reproducibility already verified in
pop_hash_verify.txt). Uses the SAME run_operon helper as the sweep (same metric).

Usage: uv run python experiments/e7_section2/operon_sentinels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e6_kernel_sweep"))
import sweep_e6 as e6   # noqa: E402

E6 = Path("experiments/e6_kernel_sweep/out/sweep_e6.jsonl")
OUT = Path("experiments/e7_section2/out/operon_sentinels.json")

# representative + fast-ish (CPU): vary preset / N / ncores; keep M small so the
# sentinel finishes quickly while still exercising the Operon timing path.
SENTINELS = [
    # low-core controls
    ("early-gen", 4000, 1000, 1),
    ("early-gen", 4000, 1000, 16),
    ("inner-const-heavy", 4000, 1000, 16),
    ("early-gen", 1000, 10000, 64),
    # high-core regime (nc=64/128) — where the crossover headline lives + the
    # config that flagged 46% drift; characterize whether it's systematic
    ("late-gen-bloated", 4000, 1000, 64),
    ("late-gen-bloated", 4000, 1000, 128),
    ("early-gen", 4000, 1000, 128),
    ("inner-const-heavy", 4000, 1000, 64),
    ("early-gen", 16000, 1000, 64),
    ("late-gen-bloated", 16000, 1000, 128),
    ("inner-const-heavy", 16000, 100, 128),
]


def main():
    recs = [json.loads(l) for l in E6.read_text().splitlines() if l.strip()]
    op = {(r["preset"], r["M"], r["N"], r["knob"]): r for r in recs
          if r.get("backend") == "operon" and r.get("status") == "ok"}
    results, worst = [], 0.0
    for preset, M, N, nc in SENTINELS:
        old = op.get((preset, M, N, nc))
        if not old:
            results.append(dict(config=[preset, M, N, nc], status="not_in_e6"))
            continue
        pop = e6.get_pop(preset, M, N, 0)
        o = e6.run_operon(pop, nc, max_iter=e6.OPERON_MAX_ITER, n_rep=3)
        e6.evict_pop(preset, M, N, 0)
        old_wc, new_wc = old["wall_core"], o["wall_core"]
        old_tp, new_tp = old["throughput"], o["throughput"]
        drift_wc = abs(new_wc - old_wc) / old_wc * 100 if old_wc else float("inf")
        drift_tp = abs(new_tp - old_tp) / old_tp * 100 if old_tp else float("inf")
        worst = max(worst, drift_wc, drift_tp)
        verdict = ("clean" if max(drift_wc, drift_tp) < 15
                   else "note" if max(drift_wc, drift_tp) < 30 else "STOP")
        row = dict(config=[preset, M, N, nc],
                   e6_wall_core=round(old_wc, 4), new_wall_core=round(new_wc, 4),
                   e6_throughput=round(old_tp, 1), new_throughput=round(new_tp, 1),
                   drift_wall_core_pct=round(drift_wc, 1),
                   drift_throughput_pct=round(drift_tp, 1), verdict=verdict)
        results.append(row)
        print(f"  {preset:18s} M={M} N={N} nc={nc}: e6 wc={old_wc:.3f}s -> "
              f"new {new_wc:.3f}s (drift {drift_wc:.1f}%) | tput {old_tp:.0f} -> "
              f"{new_tp:.0f} ({drift_tp:.1f}%)  [{verdict}]")
    overall = ("clean" if worst < 15 else "note" if worst < 30 else "STOP")
    OUT.write_text(json.dumps(dict(
        worst_drift_pct=round(worst, 1), overall=overall,
        tolerance="<15 clean / 15-30 note / >30 STOP", results=results),
        indent=2, default=float))
    print(f"\nWORST drift = {worst:.1f}%  -> OVERALL: {overall}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
