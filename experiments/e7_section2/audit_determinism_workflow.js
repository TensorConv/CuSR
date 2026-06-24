export const meta = {
  name: 'audit-determinism',
  description: 'Adversarially audit the C2 determinism/clean-baseline round: re-derive every headline number from raw reps, then codex final review',
  phases: [
    { title: 'Audit', detail: 'parallel skeptics re-derive CV/crossover/isolation/floor from reps_raw.jsonl' },
    { title: 'Codex', detail: 'codex reviews design + code + results + anti-fraud over the whole round' },
  ],
}

const DIR = 'experiments/e7_section2/out/determinism'
const REPO = '/home/weish/hao/CuSR'

const FINDING_SCHEMA = {
  type: 'object',
  required: ['lens', 'verdict', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['clean', 'minor_issues', 'blocking_issues'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['claim', 'reproduced', 'severity', 'detail'],
        properties: {
          claim: { type: 'string', description: 'the specific number/claim audited' },
          reproduced: { type: 'boolean', description: 'did it reproduce from raw reps_raw.jsonl?' },
          recomputed_value: { type: 'string', description: 'what you independently got' },
          severity: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'critical'] },
          detail: { type: 'string' },
        },
      },
    },
  },
}

const COMMON = `
You are an ADVERSARIAL auditor of a GPU-vs-CPU symbolic-regression timing experiment (CuSR HPEC paper, Section 2 hardening). Repo: ${REPO}. Artifacts in ${DIR}/.

THE GROUND TRUTH is ${DIR}/reps_raw.jsonl (one JSON per measured rep, fsync'd at write). Every summary JSON (cv_gpu.json, cv_operon.json, crossover_clean.json, etc.) is a DERIVED claim. Your job: re-derive the claim INDEPENDENTLY from reps_raw.jsonl with your own python (use .venv/bin/python; the pure helpers in experiments/e7_section2/determinism_analysis.py are available but RE-IMPLEMENT the math yourself to cross-check, do not just call them blindly). A number in a summary JSON that does NOT reproduce from the raw reps is a FABRICATION — flag it critical.

Anti-fraud rules this experiment claims to follow (verify each where in scope): every number traceable to a disk rep; GPU SM locked @1410 + readback; Operon bound via taskset/sched_setaffinity with per-rep tenant-overlap snapshot; fp64 quality recomputed independently (never backend self-report); serial measurement. Be skeptical and concrete. Return ONLY the structured object.
`

phase('Audit')

const LENSES = [
  {
    key: 'gpu-determinism',
    prompt: `${COMMON}
LENS: GPU determinism (CV of loop_ms).
1. From reps_raw.jsonl phase=="G" records, for each (variant, cell) collect the raw loop_ms reps and recompute median + CV (std_ddof1/mean). Compare to cv_gpu.json. Do they match to ~3 sig figs?
2. Is the GPU-determinism claim honest (CV magnitude)? A sub-1% CV at the occupancy-saturated cells empirically refutes "0.38s runs are too unreliable for a paper".
3. Check each phase-G record carries gpu_id and locked_mhz=1410 (锁频 provenance). Flag any missing.
Falsify: any cv_gpu.json number not reproducible from raw reps; any determinism claim the spread contradicts.`,
  },
  {
    key: 'operon-crossover',
    prompt: `${COMMON}
LENS: Operon CV + clean crossover multipliers (the headline-tightening deliverable).
1. From reps_raw.jsonl phase=="O" recompute Operon wall_core median + CV per (cell, ncores). Compare to cv_operon.json.
2. Recompute crossover multiplier = GPU revad throughput_at_median / Operon throughput_at_median for each cell × {64c,128c}; compare to crossover_clean.json. Recompute throughput = (M - n_dropped)/(median_s) yourself from the medians.
3. Verify the iso-quality gate: is gpu revad fp64 loss <= operon fp64 loss * 1.05 at each cell? If iso=false, the multiplier is NOT a valid iso-quality crossover — flag if crossover_clean.json reports a multiplier as headline-grade where iso failed.
4. The original FINDINGS headline was 9.6x (early-gen M=16k N=1000 vs 128c) and 28.2x (late-gen M=64k N=100 vs 128c, ±30% noisy). Did the clean re-measure reproduce these, or move them? Report the delta honestly.
Falsify: any multiplier not reproducible; any iso-quality violation hidden; any cross-day-vs-within-session conflation.`,
  },
  {
    key: 'isolation-honesty',
    prompt: `${COMMON}
LENS: isolation / contention honesty (advisor gate-2 + gate-3).
1. For each phase-O record read tenant: {foreign_overlap_samples, n_samples, foreign_overlap_frac, foreign_cores_hit, affinity_inheritance_ok, my_workers_outside_mask}. Was the 64c (node1) run ACTUALLY isolated (foreign_overlap_samples==0)? Was affinity_inheritance_ok true (workers stayed on the bound mask)?
2. The 128c run uses the FULL machine and overlaps co-tenants on node0 BY CONSTRUCTION. Is crossover_clean.json honest that the 128c number is NOT isolated (isolated_verified=false, tenant_overlap_frac>0)? A "clean 128c" claim would be FALSE on a shared box — flag it.
3. Gate-3 honesty: the within-session CV must NOT be presented as capturing the cross-day ±30% drift. Check binding.txt + any writeup framing for this conflation.
Falsify: any "isolated/clean" claim the tenant snapshot contradicts; any affinity_inheritance_ok=false that undermines a 64c "bound" claim.`,
  },
  {
    key: 'floor-oversub-linearity',
    prompt: `${COMMON}
LENS: IPC-floor decomposition (Q1) + oversubscription verdict + linearity/intercept (gate-1).
1. Q1 floor: compare phase-F (homogeneous, cv_floor_homogeneous.json) CV to phase-O (heterogeneous) CV at the SAME cell/ncores. Is heterogeneous CV > homogeneous floor (=> genuine load-imbalance jitter is real and attributable)? Or does the floor EAT the excess (=> determinism claim must be DROPPED)? State which, from the raw reps.
2. Oversub (oversub_innerconst.json phase=="X"): recompute capped/uncapped wall_core ratio from raw reps. Is the verdict's interpretation correct? (ratio<0.85 => oversubscribed/inflated; ratio>1.15 => uncapped threads help, baseline is Operon's best/fair; ~1 => idle.)
3. Linearity gate (linearity_precheck.json, timing_window.json): did intercept regression get applied ONLY where loop_ms was linear in max_iter (no early-stop saturation)? If a plateau was present but the full range was fit anyway, the intercept is meaningless — flag it. Is intercept_frac small enough to claim "no systematic short-region bias"?
Falsify: a determinism claim the floor refutes; a wrong oversub interpretation; an intercept fit over a saturated regime.`,
  },
]

const audits = await parallel(LENSES.map(l => () =>
  agent(`${l.prompt}`, { label: `audit:${l.key}`, phase: 'Audit', schema: FINDING_SCHEMA })
))

const valid = audits.filter(Boolean)
const blocking = valid.filter(a => a.verdict === 'blocking_issues')
const allFindings = valid.flatMap(a => (a.findings || []).map(f => ({ lens: a.lens, ...f })))
const notReproduced = allFindings.filter(f => f.reproduced === false)
const highSev = allFindings.filter(f => ['high', 'critical'].includes(f.severity))

log(`Audit: ${valid.length}/4 lenses; ${blocking.length} blocking; ${notReproduced.length} numbers NOT reproduced; ${highSev.length} high/critical findings`)

phase('Codex')

const codexPrompt = `You are the FINAL reviewer (codex) of the CuSR HPEC paper's Section-2 "determinism / clean-baseline" hardening round. Repo: ${REPO}.

Review the WHOLE round end to end for correctness AND anti-fraud honesty:
- DESIGN: tasks/RUN_determinism_handoff.md (spec) — but note the red-team REPLACED the --repeat binary change with intercept regression (no binary change), and Q1 added the homogeneous IPC-floor control. Confirm the executed design matches that, not the stale --repeat text.
- CODE: experiments/e7_section2/determinism_run.py (serial driver, reuses audited sweep_e6 helpers) + determinism_analysis.py (pure, TDD'd in test_determinism_analysis.py). Read them. Check: the timed path adds nothing; affinity is set before pool spawn and pools are scrapped before the OMP-cap test; the tenant sampler identifies Operon workers by ppid; fp64 quality is recomputed independently.
- RESULTS: ${DIR}/*.json + reps_raw.jsonl + binding.txt + clock_lock_{pre,post}.txt + run.log. Spot-check that headline numbers trace to raw reps.
- The internal adversarial audit found: ${JSON.stringify({ blocking: blocking.length, notReproduced: notReproduced.length, highSev: highSev.map(f => `${f.lens}:${f.claim}`).slice(0, 8) })}.

Deliver: (1) any number/claim you can falsify or that is not honestly scoped; (2) whether the determinism contribution is claimable or must be dropped (per the floor decomposition); (3) whether the clean crossover multipliers honestly supersede the old ±30% (gate-3: within-session != cross-day); (4) a GO / GO-WITH-FIXES / NO-GO for folding these numbers into the paper, with the specific fixes. Be concrete and terse. Use .venv/bin/python to recompute anything you doubt.`

const codex = await agent(codexPrompt, { label: 'codex:final-review', phase: 'Codex', agentType: 'codex:codex-rescue' })

return {
  audit_lenses: valid.length,
  blocking_lenses: blocking.map(a => a.lens),
  numbers_not_reproduced: notReproduced,
  high_severity: highSev,
  all_findings: allFindings,
  codex_review: codex,
}
