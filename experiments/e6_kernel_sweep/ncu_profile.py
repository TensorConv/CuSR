#!/usr/bin/env python
"""ncu_profile.py — task 04: ncu SOL + warp-stall + instruction-roofline driver.

Profiles the hot CO kernels (Jacobian / build_jtj / eval) for fusedfd / ad / revad
with Nsight Compute, then parses the .ncu-rep into: Speed-of-Light utilisation
(compute% / memory% / issue%), the top warp-stall reasons, and an instruction
roofline point (GIPS vs instruction intensity). This is the PRIMARY bottleneck
diagnosis — the kernel runs at <1% of both FLOP and HBM peak, so a classic FLOP
roofline diagnoses nothing (see OPTIMIZATION_BACKLOG §0.2).

WHY exact kernel names (codex #7): a wide `eval` regex also captures
eval_loss_fp64_kernel. We pin the demangled bases:
  fd_jacobian_fused_kernel | ad_jacobian_kernel | rev_jacobian_kernel
  | build_jtj_jtr_kernel | eval_kernel_batched
ncu only profiles kernels that ACTUALLY LAUNCH, so the captured Jacobian kernel
also doubles as the variant proof (铁律 #4): ad_prof must show ad_jacobian_kernel,
revad_prof must show rev_jacobian_kernel (assert_device_jacobian alone can't tell
ad from revad — both report under the PROFILE 'fd_jacobian' category).

The pure parser (parse_ncu_csv / summarize_kernel) is unit-tested in
test_ncu_profile.py on a synthetic CSV — no GPU needed. The driver (run) needs
GPU + profiling sudo.

Usage (driver; quiet GPU only — run AFTER the sweep frees the cards):
  source scripts/env.sh
  PATH=/usr/local/cuda/bin:$PATH \
    uv run python experiments/e6_kernel_sweep/ncu_profile.py \
      --gpu 0 --out experiments/e7_section2/ncu
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parents[2] / "cusr" / "kernel"
KERNEL_REGEX = ("regex:fd_jacobian_fused_kernel|ad_jacobian_kernel|"
                "rev_jacobian_kernel|build_jtj_jtr_kernel|eval_kernel_batched")
VARIANT_BINARY = {
    "fusedfd": KERNEL_DIR / "batch_lm_fusedfd_prof",
    "ad": KERNEL_DIR / "batch_lm_ad_prof",
    "revad": KERNEL_DIR / "batch_lm_revad_prof",
}

# ncu metric names (Nsight Compute 2023+). SpeedOfLight section publishes the
# %-of-peak throughputs; WarpStateStats publishes per-stall warp-cycle averages;
# InstructionStats publishes inst_executed (for GIPS) + the issue active %.
SOL_METRICS = {
    "compute_sol_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_sol_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "inst_exec_pct": "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
    # fp32-FMA and fp64 pipe utilisation: the "<1% FLOP" classic-roofline
    # COUNTER-EVIDENCE (high composite SOL but near-zero FP pipe => not FLOP-bound)
    "fma_pipe_pct": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "fp64_pipe_pct": "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # achieved occupancy (warps resident per cycle / 64 max on sm_80)
    "warps_active_per_cycle": "smsp__warps_active.avg.per_cycle_active",
}
# inst + timing for the instruction-roofline point. gpu__time_duration.sum is in
# MICROSECONDS (verified). dram__bytes.sum is ABSENT from these reps (no
# InstructionStats DRAM byte count) -> instruction intensity left None unless a
# memory section is added; GIPS still computed.
INST_METRICS = {
    "inst_executed": "smsp__inst_executed.sum",
    "duration_us": "gpu__time_duration.sum",
    "dram_bytes": "dram__bytes.sum",
}
_INST_ALIASES = ["smsp__inst_executed.sum", "sm__inst_executed.sum"]
STALL_PREFIX = "smsp__average_warps_issue_stalled_"   # + <reason>_per_issue_active.ratio
STALL_SUFFIX = "_per_issue_active.ratio"


def _to_num(s: str):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return s


def _short_kernel(name: str) -> str:
    """ncu reports the full demangled signature; key on the bare function name."""
    return name.split("(")[0].strip() if name else name


def parse_ncu_csv(text: str):
    """ncu CSV -> {kernel_name: {metric_id: (value, unit)}}. Handles BOTH:
      * long  `--page details` (cols 'Metric Name' + 'Metric Value'), and
      * wide  `--page raw` (one row per kernel; metric IDs are the column headers,
        a units row follows the header, kernel rows have a non-empty Kernel Name).
    Robust to leading banner lines; kernels are keyed on the bare function name."""
    rows = list(csv.reader(io.StringIO(text)))
    # ---- long format (the unit-test fixture + --page details) ----
    for i, row in enumerate(rows):
        if "Metric Name" in row and "Metric Value" in row:
            kcol = row.index("Kernel Name") if "Kernel Name" in row else None
            mcol = row.index("Metric Name")
            ucol = row.index("Metric Unit") if "Metric Unit" in row else None
            vcol = row.index("Metric Value")
            out: dict = {}
            for r in rows[i + 1:]:
                if len(r) <= max(mcol, vcol):
                    continue
                kn = _short_kernel(r[kcol]) if kcol is not None and kcol < len(r) else "?"
                unit = r[ucol] if ucol is not None and ucol < len(r) else ""
                out.setdefault(kn, {})[r[mcol]] = (_to_num(r[vcol]), unit)
            return out
    # ---- wide format (--page raw): header row has 'Kernel Name' + metric IDs ----
    for i, row in enumerate(rows):
        if "Kernel Name" in row:
            hdr = row
            kcol = hdr.index("Kernel Name")
            units_row = rows[i + 1] if i + 1 < len(rows) else []
            out = {}
            for r in rows[i + 1:]:
                if len(r) <= kcol or not r[kcol].strip():
                    continue                       # skip the units row / blanks
                kn = _short_kernel(r[kcol])
                d = out.setdefault(kn, {})
                for j, metric in enumerate(hdr):
                    if j == kcol or j >= len(r) or r[j] == "":
                        continue
                    unit = units_row[j] if j < len(units_row) else ""
                    d[metric] = (_to_num(r[j]), unit)
            return out
    return {}


def _get(metrics: dict, name: str):
    v = metrics.get(name)
    return v[0] if isinstance(v, tuple) else None


def summarize_kernel(metrics: dict) -> dict:
    """Reduce one kernel's metric dict to SOL bars + top stalls + an instruction
    roofline point. All derived numbers come straight from ncu metrics."""
    sol = {k: _get(metrics, m) for k, m in SOL_METRICS.items()}
    # top warp-stall reasons (WarpStateStats): warps stalled per issue-active,
    # already a fraction-of-warps measure; rank descending.
    stalls = {}
    for m, (val, _u) in metrics.items():
        if m.startswith(STALL_PREFIX) and m.endswith(STALL_SUFFIX):
            reason = m[len(STALL_PREFIX):-len(STALL_SUFFIX)]
            if isinstance(val, (int, float)):
                stalls[reason] = val
    top_stalls = sorted(stalls.items(), key=lambda kv: kv[1], reverse=True)[:5]
    # issue-active fallback: if InstructionStats' issue_active% absent, use the
    # SOL inst-executed% as the issue-utilisation proxy.
    if sol.get("issue_active_pct") is None:
        sol["issue_active_pct"] = sol.get("inst_exec_pct")
    # instruction roofline point: GIPS = inst_executed / duration. duration is in
    # MICROSECONDS -> GIPS = inst / dur_us / 1e3. intensity = inst / dram_bytes
    # (None when dram_bytes absent from the rep).
    inst = next((_get(metrics, a) for a in _INST_ALIASES
                 if _get(metrics, a) is not None), None)
    # ncu reports gpu__time_duration.sum in ms OR us depending on the kernel —
    # MUST read the unit and normalize to microseconds (codex caught a 1000x bug).
    dur_raw = metrics.get(INST_METRICS["duration_us"])
    dur_val, dur_unit = (dur_raw if isinstance(dur_raw, tuple) else (None, ""))
    _to_us = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    dur_us = (dur_val * _to_us.get((dur_unit or "").strip(), 1.0)
              if dur_val else None)
    dram_b = _get(metrics, INST_METRICS["dram_bytes"])
    gips = (inst / dur_us / 1e3) if (inst and dur_us) else None
    inten = (inst / dram_b) if (inst and dram_b) else None     # inst/byte
    return dict(sol=sol, top_stalls=top_stalls,
                instruction_roofline=dict(gips=gips, inst_per_byte=inten,
                                          inst_executed=inst, duration_us=dur_us,
                                          dram_bytes=dram_b))


# ---------------------------------------------------------------------------
# DRIVER (GPU + sudo) — keep below the pure functions so imports stay GPU-free.
# ---------------------------------------------------------------------------
def _gen_pop(preset, M, N, seed, dst: Path):
    from cusr.benchmark.workload.gen_synth import gen_pop
    from cusr.benchmark import popio
    pop = gen_pop(preset, M, N, seed)
    popio.save_pop_bin(pop, dst)
    return pop


def run(gpu: int, out: Path, ncu_bin: str):
    out.mkdir(parents=True, exist_ok=True)
    pops = out / "pops"
    pops.mkdir(exist_ok=True)
    # two M (codex #8: small + one M≈64k to confirm the large-M claim), two
    # regimes (inner-const-heavy high-K + early-gen). N=1000 = the backlog
    # profiling regime.
    configs = [
        dict(regime="early-gen", M=2000, N=1000, sections="full"),
        dict(regime="inner-const-heavy", M=2000, N=1000, sections="full"),
        dict(regime="early-gen", M=64000, N=1000, sections="sol"),
        dict(regime="inner-const-heavy", M=64000, N=1000, sections="sol"),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    manifest = []
    for variant, binary in VARIANT_BINARY.items():
        for cfg in configs:
            tag = f"{variant}_{cfg['regime']}_M{cfg['M']}"
            pop_path = pops / f"{cfg['regime']}_M{cfg['M']}_N{cfg['N']}.bin"
            if not pop_path.exists():
                _gen_pop(cfg["regime"], cfg["M"], cfg["N"], 0, pop_path)
            rep = out / f"{tag}.ncu-rep"
            kout = out / f"{tag}_kout"
            kout.mkdir(parents=True, exist_ok=True)  # binary writes status.bin here
            sections = (["--section", "SpeedOfLight",
                         "--section", "WarpStateStats",
                         "--section", "SchedulerStats",
                         "--section", "InstructionStats"]
                        if cfg["sections"] == "full"
                        else ["--section", "SpeedOfLight"])
            cmd = (["sudo", "-n", ncu_bin, "--target-processes", "all",
                    "--kernel-name-base", "demangled",
                    "--kernel-name", KERNEL_REGEX, "-c", "2"]
                   + sections
                   + ["-f", "-o", str(rep.with_suffix("")),
                      str(binary), str(pop_path), str(kout),
                      "--max-iter", "2"])
            print(f"[NCU] {tag} -> {rep.name}")
            r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               timeout=1800)
            (out / f"{tag}.ncu.stdout").write_text(r.stdout + "\n---STDERR---\n"
                                                   + r.stderr)
            if r.returncode != 0 or not rep.exists():
                print(f"[NCU][FAIL] {tag} rc={r.returncode}: {r.stderr[-300:]}")
                manifest.append(dict(tag=tag, status="failed",
                                     rc=r.returncode, rep=None))
                continue
            # export csv + parse
            csv_path = out / f"{tag}.csv"
            ex = subprocess.run([ncu_bin, "--import", str(rep), "--csv",
                                 "--page", "raw"], capture_output=True,
                                text=True, timeout=600)
            csv_path.write_text(ex.stdout)
            parsed = parse_ncu_csv(ex.stdout)
            summ = {kn: summarize_kernel(m) for kn, m in parsed.items()}
            (out / f"{tag}.summary.json").write_text(
                json.dumps(summ, indent=2, default=float))
            manifest.append(dict(tag=tag, variant=variant, regime=cfg["regime"],
                                 M=cfg["M"], N=cfg["N"], status="ok",
                                 rep=str(rep), csv=str(csv_path),
                                 kernels=sorted(parsed.keys())))
            print(f"[NCU][OK] {tag}: kernels={sorted(parsed.keys())}")
    (out / "ncu_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                       default=float))
    print("wrote:", out / "ncu_manifest.json")
    return manifest


def reparse(out: Path, ncu_bin: str):
    """Re-export + re-summarize ALL existing .ncu-rep in `out` with the current
    parser/summarizer — no GPU, no re-profiling. Use after a parser fix."""
    reps = sorted(out.glob("*.ncu-rep"))
    print(f"[REPARSE] {len(reps)} rep(s) in {out}")
    for rep in reps:
        tag = rep.stem
        ex = subprocess.run([ncu_bin, "--import", str(rep), "--csv",
                             "--page", "raw"], capture_output=True, text=True,
                            timeout=600)
        (out / f"{tag}.csv").write_text(ex.stdout)
        parsed = parse_ncu_csv(ex.stdout)
        summ = {kn: summarize_kernel(m) for kn, m in parsed.items()}
        (out / f"{tag}.summary.json").write_text(
            json.dumps(summ, indent=2, default=float))
        print(f"  [REPARSE] {tag}: kernels={sorted(parsed.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="experiments/e7_section2/ncu")
    ap.add_argument("--ncu-bin", type=str, default="/usr/local/cuda/bin/ncu")
    ap.add_argument("--reparse", action="store_true",
                    help="re-summarize existing .ncu-rep (no GPU/profiling)")
    args = ap.parse_args()
    if args.reparse:
        reparse(Path(args.out).resolve(), args.ncu_bin)
    else:
        run(args.gpu, Path(args.out).resolve(), args.ncu_bin)


if __name__ == "__main__":
    main()
