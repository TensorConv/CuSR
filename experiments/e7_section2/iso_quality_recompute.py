#!/usr/bin/env python
"""iso_quality_recompute.py — OFFLINE iso-quality gate + GPU-vs-Operon crossover.

WHY: reusing the e6 Operon records means Phase B is skipped, so the harness never
computes revad's iso-quality gate (and the copied Operon gates are the stale e6
ad/fusedfd-only values). This script recomputes the gate OFFLINE for all three GPU
variants (fusedfd / ad / revad) vs Operon {1,16,64,128} cores, purely from on-disk
artifacts (the .npy loss sidecars), then reports the iso-quality crossover speedup.

ANTI-FAKERY (task 05 + codex review + advisor):
  - Every loss is recomputed as median-of-finite from the .npy sidecar on disk,
    NOT trusted from the JSONL scalar. We additionally cross-check the recomputed
    GPU median against the record's stored med_loss_fp64 and FLAG any mismatch.
  - Pop identity at pairing: we assert all GPU variants present at a (preset,M,N)
    cell share an identical pop_sha256 (the kernel records store the hash of the
    pop the GPU actually saw). gen_synth byte-determinism vs that hash is verified
    separately in pop_hash_verify.txt (preflight) + codex full-overlap check.
  - iso-quality := GPU seed0 fp64 median loss <= Operon seed0 fp64 median * 1.05
    (QUALITY_TOL, ported from the harness). Non-finite either side => cannot
    certify => not iso (never silently passed).
  - Operon records are CPU; reused from e6 (sentinels confirm no timing drift).

Usage:
  uv run python experiments/e7_section2/iso_quality_recompute.py --out <dir>
  (default --out = experiments/e6_kernel_sweep/out, for the dry-run against e6)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

QUALITY_TOL = 1.05  # iso-quality envelope, identical to sweep_e6.quality_matched
VARIANTS = ["fusedfd", "ad", "revad"]
NCORES = [1, 16, 64, 128]


def med_finite(arr: np.ndarray) -> float:
    a = np.asarray(arr, np.float64)
    f = a[np.isfinite(a)]
    return float(np.median(f)) if f.size else float("inf")


def load_sidecar(loss_dir: Path, name: str):
    p = loss_dir / name
    return np.load(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default="experiments/e6_kernel_sweep/out",
                    help="dir holding sweep_e6.jsonl + sweep_e6_losses/")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    jsonl = out / "sweep_e6.jsonl"
    loss_dir = out / "sweep_e6_losses"

    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    kern = [r for r in recs if r.get("backend") == "kernel"
            and r.get("status") == "ok"]
    oper = [r for r in recs if r.get("backend") == "operon"
            and r.get("status") == "ok"]

    # index kernel by (variant,preset,M,N); operon by (preset,M,N,ncores)
    kidx = {(r["variant"], r["preset"], r["M"], r["knob"]): r for r in kern}
    oidx = {(r["preset"], r["M"], r["N"], r["knob"]): r for r in oper}

    cells = sorted({(r["preset"], r["M"], r["knob"]) for r in kern})
    rows = []                 # one per (preset,M,N,variant,ncores) iso pairing
    audit_flags = []
    pop_consistency = []

    for (preset, M, N) in cells:
        # ---- pop identity within the cell (all GPU variants share one pop) ----
        hashes = {v: kidx[(v, preset, M, N)]["pop_sha256"]
                  for v in VARIANTS if (v, preset, M, N) in kidx}
        uniq = set(hashes.values())
        if len(uniq) > 1:
            pop_consistency.append(dict(cell=[preset, M, N], hashes=hashes,
                                        ok=False))
        elif uniq:
            pop_consistency.append(dict(cell=[preset, M, N],
                                        pop_sha256=list(uniq)[0], ok=True))

        # ---- GPU fp64 median loss per variant, recomputed from sidecar ----
        gpu_med, gpu_tput = {}, {}
        for v in VARIANTS:
            rec = kidx.get((v, preset, M, N))
            if rec is None:
                continue
            side = load_sidecar(loss_dir, f"kernel_{v}_{preset}_{M}_{N}.npy")
            if side is None:
                audit_flags.append(f"MISSING kernel sidecar {v} {preset} M={M} N={N}")
                continue
            m = med_finite(side)
            gpu_med[v] = m
            gpu_tput[v] = rec["throughput"]
            stored = rec.get("med_loss_fp64")
            if (stored is not None and np.isfinite(stored) and np.isfinite(m)
                    and abs(m - stored) > 1e-6 * max(1.0, abs(stored))):
                audit_flags.append(
                    f"MED MISMATCH {v} {preset} M={M} N={N}: sidecar={m:.6g} "
                    f"record={stored:.6g}")

        # ---- Operon fp64 median + throughput per ncores, from sidecar ----
        op_med, op_tput = {}, {}
        for nc in NCORES:
            rec = oidx.get((preset, M, N, nc))
            if rec is None:
                continue
            side = load_sidecar(loss_dir, f"operon_N{N}_{preset}_{M}_{nc}.npy")
            if side is None:
                continue
            op_med[nc] = med_finite(side)
            op_tput[nc] = rec["throughput"]

        # ---- iso-quality pairing + crossover speedup ----
        for v in VARIANTS:
            if v not in gpu_med:
                continue
            for nc in NCORES:
                if nc not in op_med:
                    continue
                iso = bool(np.isfinite(gpu_med[v]) and np.isfinite(op_med[nc])
                           and gpu_med[v] <= op_med[nc] * QUALITY_TOL)
                speedup = (gpu_tput[v] / op_tput[nc]
                           if op_tput.get(nc) else float("nan"))
                rows.append(dict(
                    preset=preset, M=M, N=N, variant=v, ncores=nc,
                    gpu_med_loss=gpu_med[v], operon_med_loss=op_med[nc],
                    iso_quality=iso, gpu_tput=gpu_tput[v],
                    operon_tput=op_tput[nc],
                    speedup_gpu_over_operon=round(speedup, 3)))

    # ---- headline: iso-quality crossover speedups by (variant, ncores) ----
    headline = {}
    for v in VARIANTS:
        for nc in NCORES:
            iso_rows = [r for r in rows if r["variant"] == v
                        and r["ncores"] == nc and r["iso_quality"]]
            if iso_rows:
                sp = [r["speedup_gpu_over_operon"] for r in iso_rows
                      if np.isfinite(r["speedup_gpu_over_operon"])]
                headline[f"{v}_vs_{nc}c"] = dict(
                    n_iso=len(iso_rows), n_total_pairs=sum(
                        1 for r in rows if r["variant"] == v and r["ncores"] == nc),
                    speedup_median=round(st.median(sp), 2) if sp else None,
                    speedup_max=round(max(sp), 2) if sp else None,
                    speedup_min=round(min(sp), 2) if sp else None,
                    best_cell=max(
                        (r for r in iso_rows),
                        key=lambda r: r["speedup_gpu_over_operon"],
                        default=None) and {
                        k: max(iso_rows,
                               key=lambda r: r["speedup_gpu_over_operon"])[k]
                        for k in ("preset", "M", "N",
                                  "speedup_gpu_over_operon")})

    result = dict(
        source=str(jsonl), quality_tol=QUALITY_TOL,
        locked_clock_mhz=(kern[0].get("locked_clock_mhz") if kern else None),
        n_cells=len(cells), n_pairings=len(rows),
        n_iso=sum(1 for r in rows if r["iso_quality"]),
        pop_consistency_ok=all(p["ok"] for p in pop_consistency),
        pop_consistency_n=len(pop_consistency),
        audit_flags=audit_flags, headline=headline, rows=rows)

    (out / "iso_quality.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"cells={len(cells)} pairings={len(rows)} iso={result['n_iso']} "
          f"pop_consistency_ok={result['pop_consistency_ok']} "
          f"audit_flags={len(audit_flags)}")
    for k, v in sorted(headline.items()):
        print(f"  {k}: n_iso={v['n_iso']}/{v['n_total_pairs']} "
              f"speedup med={v['speedup_median']} "
              f"[{v['speedup_min']}..{v['speedup_max']}]")
    if audit_flags:
        print("AUDIT FLAGS:")
        for f in audit_flags[:20]:
            print("  !", f)
    print("wrote:", out / "iso_quality.json")


if __name__ == "__main__":
    main()
