export const meta = {
  name: 'section2-adversarial-audit',
  description: 'Independently falsify each section-2 headline number from raw on-disk artifacts',
  phases: [
    { title: 'Falsify', detail: '3 adversarial lenses per headline claim, in parallel' },
    { title: 'Synthesize', detail: 'majority-refute verdict per claim' },
  ],
}

// args.claims = [{ id, statement, artifacts: [paths], }]
let _args = args
if (typeof _args === 'string') {
  try { _args = JSON.parse(_args) } catch (e) { _args = {} }
}
const claims = (_args && _args.claims) || []
if (!claims.length) {
  log('No claims passed in args.claims — nothing to audit.')
  return { error: 'no claims' }
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean', description: 'true if the claim is NOT supported by the artifacts' },
    confidence: { type: 'number', description: '0..1' },
    recomputed_value: { type: 'string', description: 'the value you independently derived, or N/A' },
    evidence: { type: 'string', description: 'specific files/numbers you checked and what you found' },
  },
  required: ['refuted', 'confidence', 'evidence'],
}

const LENSES = [
  { key: 'rederive', instr:
    'RE-DERIVE the number yourself from the RAW on-disk artifacts (the sweep_e6.jsonl records and the .npy per-tree fp64 loss sidecars) with your own python. Do NOT trust any *summary*.json, the iso_quality.json headline, or any FINDINGS text — recompute from the primary records. Report your recomputed value and whether it matches the claim within rounding.' },
  { key: 'provenance', instr:
    'Attack PROVENANCE / anti-fakery. Verify: (a) variant is the claimed one — binary basename in the allowlist, and for ad-vs-revad the ncu manifest shows the right launched kernel (ad_jacobian_kernel vs rev_jacobian_kernel); (b) locked_clock_mhz is present (=1410) on EVERY kernel record behind the number; (c) pop_sha256 is consistent across variants at each cell; (d) the fp64 quality gate / iso-quality envelope was actually satisfied (recompute the median from the sidecar, do not trust the stored scalar); (e) NO skipped / crashed / NaN / tripwire-halt record was silently counted as a result. A crash reinterpreted as a finding = REFUTED.' },
  { key: 'framing', instr:
    'Attack FRAMING / UNITS / SCOPE. Does the stated number match what was actually measured? trees/s vs points/s; seed-median (over seeds 0,1,2) vs a single seed; which Operon ncores the iso-quality crossover is against and whether the gate truly held; DRAFT (unlocked) vs locked; and any silent truncation or cherry-pick (top-N only, dropped cells, best-case preset). REFUTE if the framing overclaims relative to the artifacts.' },
]

phase('Falsify')
const audited = await pipeline(
  claims,
  (claim) => parallel(LENSES.map((L) => () =>
    agent(
      `You are an ADVERSARIAL auditor for a CUDA symbolic-regression paper. Your job is to REFUTE, not confirm. Default to refuted=true unless the on-disk artifacts clearly prove the claim. Honesty over politeness.\n\n` +
      `REPO: /home/weish/hao/CuSR  (cd there; use Bash + python3/uv, and Read on the artifacts)\n` +
      `CLAIM #${claim.id}: ${claim.statement}\n` +
      `RELEVANT ARTIFACTS: ${JSON.stringify(claim.artifacts)}\n\n` +
      `LENS — ${L.key}:\n${L.instr}\n\n` +
      `Quote the exact numbers/files you inspected. Then return your verdict (refuted/confidence/recomputed_value/evidence). Do NOT modify any file.`,
      { label: `claim${claim.id}:${L.key}`, phase: 'Falsify', schema: VERDICT, agentType: 'general-purpose' },
    ).then((v) => ({ ...(v || { refuted: true, confidence: 0, evidence: 'agent returned null' }), lens: L.key })),
  )).then((vs) => ({ claim, verdicts: (vs || []).filter(Boolean) })),
)

phase('Synthesize')
const summary = audited.filter(Boolean).map((r) => {
  const refuted_votes = r.verdicts.filter((v) => v.refuted).length
  return {
    id: r.claim.id,
    statement: r.claim.statement,
    refuted_votes,
    total_votes: r.verdicts.length,
    flagged: refuted_votes >= 2, // majority of 3 lenses refute => do not trust as-is
    verdicts: r.verdicts.map((v) => ({ lens: v.lens, refuted: v.refuted, confidence: v.confidence, recomputed_value: v.recomputed_value, evidence: v.evidence })),
  }
})
const flagged = summary.filter((s) => s.flagged)
log(`Audited ${summary.length} claims; ${flagged.length} FLAGGED (majority-refuted).`)
return { n_claims: summary.length, n_flagged: flagged.length, summary }
