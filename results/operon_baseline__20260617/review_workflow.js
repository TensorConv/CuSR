export const meta = {
  name: 'baseline-adversarial-review',
  description: 'Adversarial multi-lens review of the Operon-vs-kernel CO baseline: setup errors, cheating, fake reports',
  phases: [
    { title: 'Review', detail: 'parallel reviewers, one focused lens each, reading code+data+report' },
    { title: 'Verify',  detail: 'adversarially verify each high/critical finding against the actual files' },
    { title: 'Synthesize', detail: 'dedupe, rank, verdict + fix list' },
  ],
}

const ROOT = '/home/weish/hao/CuSR'
const DIR = `${ROOT}/results/operon_baseline__20260617`
const ctx = `
You are reviewing a committed-quality SCIENTIFIC BASELINE experiment in the CuSR repo.
The experiment compares constant-optimization (CO) on byte-identical pre-CO GP trees by THREE engines:
  - Operon's CPU LMOptimizer (mature fp64 trust-region) -- external reference
  - the GPU kernel batch_lm_fusedfd (FD Jacobian, fp32 fast-math) -- established
  - the GPU kernel batch_lm_ad   (AD Jacobian, fp32 fast-math) -- candidate (adoption DEFERRED by the user)
Neutral arbiter: each engine returns optimized coefficients; all are re-scored by an identical fp64
0.5*sum((f-y)^2) (lib.half_sse). Two comparisons are kept separate: AD-vs-FD is CONTROLLED (same fp32 LM,
only the Jacobian differs); kernel-vs-Operon is an EXTERNAL YARDSTICK (different precision/LM/Jacobian).

KEY FILES (read them — do not assume):
  ${DIR}/lib.py            (fp64 neutral evaluator + helpers)
  ${DIR}/run_operon.py     (Operon CO side; regenerates corpus trees; Probe A/B gates)
  ${DIR}/run_kernel.py     (kernel side; runs binary; neutral-evals c_final)
  ${DIR}/analyze.py        (ALL aggregate metrics; the honesty core)
  ${DIR}/build_report.py   (report.json from aggregates — numbers must be sourced, not transcribed)
  ${DIR}/report.json + ${DIR}/data/report_aggregates.json   (the claims + the numbers)
  ${DIR}/probes/*.py       (the gate probes: same-start/objective/K, regenerated==committed)
  ${ROOT}/docs/kernel/OPERON_ADAPTER_SPEC.md (adapter semantics: postfix->prefix, K=CoefficientsCount, etc.)
  ${ROOT}/cusr/kernel/batch_lm_ad.cu / batch_lm_fusedfd.cu (loss = 0.5*sum(r^2); status codes; accept/reject)
You MAY run read-only commands (python to recompute a number from the JSONs, grep, etc.). Do NOT modify files.

Your job: find REAL problems a sharp reviewer or the user would catch. The user explicitly asked to
"严抓设置问题, 作弊问题, 虚假报告等ai常见错误" (strictly catch setup problems, cheating, fake reports, common AI errors).
Be concrete and adversarial. A finding must name the file + line/function + the exact defect + why it matters.
Do NOT invent problems; if the design is sound on your lens, say so and explain what you checked.`

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    summary: { type: 'string', description: 'one-line overall verdict for this lens' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit'] },
          title: { type: 'string' },
          location: { type: 'string', description: 'file:line or file:function' },
          defect: { type: 'string', description: 'the exact problem' },
          why: { type: 'string', description: 'why it matters / how it biases the result' },
          suggested_fix: { type: 'string' },
        },
        required: ['severity', 'title', 'location', 'defect', 'why'],
      },
    },
  },
  required: ['lens', 'summary', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'partly', 'refuted', 'not-a-defect'] },
    evidence: { type: 'string', description: 'what you checked in the actual files/data to decide' },
    corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit', 'none'] },
  },
  required: ['title', 'verdict', 'evidence', 'corrected_severity'],
}

const LENSES = [
  { key: 'anti-cheat', prompt: `LENS: CHEATING / statistical dishonesty. Are kernel FAILURES ever silently dropped from rate denominators? Are failed/diverged (inf) losses folded into means (poisoning)? Are quality comparisons restricted to favorable subsets without disclosing set composition? Is any "win" an artifact of excluding the cases where the engine looks worst? Check analyze.py denominators (opt=K>0), the quality masks, and report.json claims.` },
  { key: 'setup-fairness', prompt: `LENS: SETUP FAIRNESS. Do all three engines truly start from the SAME c_init (Probe A)? Same objective shape (0.5*SSE, no linear scaling)? Same K (no extra DOF)? Is the neutral fp64 arbiter applied IDENTICALLY to all engines (no fp32-vs-fp64 asymmetry that favors one)? Is the kernel mi=1000 vs Operon mi=500 asymmetry fair (Operon converges within 500)? Is the "Operon distribution = stress test for the kernel" disclosed?` },
  { key: 'fake-report', prompt: `LENS: FAKE REPORT. Trace EVERY quantitative claim in report.json back to data/report_aggregates.json (recompute a few with python). Any number that does not match? Any claim stronger than its evidence (e.g., "more robust" without the caveat, "no bug" overgeneralized)? Does the report respect that AD adoption is DEFERRED (no adoption thesis)? Bilingual zh/en consistency?` },
  { key: 'eval-correctness', prompt: `LENS: EVALUATOR CORRECTNESS. Is lib.eval_fp64 a faithful fp64 mirror of the kernel eval_tree_d and pop_ref.ref_eval (reverse-prefix stack, BFUNC [op,A,B] => A op B, A=left)? Is the per-tree const index ci LOCAL (0..K-1) and sliced correctly? K=0 handling? Does half_sse return inf for non-finite (not 0 or a bogus small number)? Recompute one tree by hand from a pop.bin if needed.` },
  { key: 'metric-defs', prompt: `LENS: METRIC DEFINITIONS in analyze.py. Is "meaningful improvement" (ratio<0.99) applied consistently to ALL engines? Is "converged-but-worsened" correct (status==0 & fp64>start)? Is the fp32-lie subset (kernel fp32<=start) defended (not overclaimed)? Are the efficacy median-ratios computed on the right masks (finite, start>0)? Any off-by-one or status-code mixup (0=conv,1=maxiter,2=NaN,3=K0,4=Cholesky)?` },
  { key: 'operon-side', prompt: `LENS: OPERON SIDE (run_operon.py). Is convert(otree) reading the OPTIMIZED coefficients in correct prefix-slot order (not Operon's GetCoefficients order)? Is "keeps best" assumed where it shouldn't be (Operon returns worse-than-start on 33 trees — is that handled, not asserted away)? Is the DispatchTable held (lifetime)? Is Probe B (regenerated==committed) actually enforced per cell? Could tree m on the Operon side mis-align with tree m in the kernel's pop.bin?` },
  { key: 'kernel-side', prompt: `LENS: KERNEL SIDE (run_kernel.py). Is c_final.bin sliced per-tree by metas (c_offset:c_offset+K) correctly? status.bin / loss_final.bin lengths checked? Is the kernel's own fp32 loss recorded for the lie-detector? Is the committed pop.bin (not a regenerated one) used so the kernel sees the SAME bytes Operon's trees produced? Any GPU nondeterminism that would make this irreproducible?` },
  { key: 'stats-validity', prompt: `LENS: STATISTICAL VALIDITY. Is pooling across 17 problems x 5 gens legitimate, or does it hide per-problem heterogeneity (Simpson)? Median vs mean choice defended? Are sample sizes (n) reported with every ratio? Is the within-1.05x tie band reasonable? Does the per-problem table reveal cases that contradict the pooled headline?` },
  { key: 'reproducibility', prompt: `LENS: REPRODUCIBILITY. Can the pipeline be regenerated from committed artifacts (the reproduce block)? Is determinism real (Operon threads=1; kernel GPU)? Are the right things committed (scripts, report, aggregates) vs gitignored (bulk .bin)? Are seeds/caps/gens pinned? Does report.json point to the probes as gates?` },
  { key: 'devils-advocate', prompt: `LENS: DEVIL'S ADVOCATE. Give the single STRONGEST argument that the headline conclusions are WRONG or confounded: e.g., "AD is more robust" is really a fast-math artifact; "Operon goes deeper on bloated trees" is a max_iter or objective-shape artifact; the fp32-lie count is a neutral-evaluator bug not a kernel bug. Then check the files/data to see if your strongest attack survives.` },
  { key: 'fast-math-fairness', prompt: `LENS: FAST-MATH FAIRNESS. The kernel uses --use_fast_math (fp32); the neutral arbiter is exact fp64; Operon optimizes fp64. Is it FAIR to score the kernel's fast-math-optimized coefficients with exact fp64 (penalizing fast-math divergence)? Or is this the only honest "true fit" measure? Is the caveat about this present and correct? Could the kernel-vs-Operon tail (p90~2.4) be ENTIRELY this artifact rather than a real LM-quality gap?` },
  { key: 'general-free', prompt: `LENS: GENERAL / FREE (自由). Anything else a careful scientist would flag: missing controls, unstated assumptions, figures that mislead, a better experiment that should have been run, internal contradictions across files, or claims that don't generalize. Also: is the EXPERIMENT FLOW itself (the user asked to review 实验流程和设置) sound end-to-end?` },
]

phase('Review')
const reviews = await parallel(LENSES.map(L => () =>
  agent(`${ctx}\n\n${L.prompt}`, { label: `review:${L.key}`, phase: 'Review', schema: FINDING_SCHEMA })
))
const valid = reviews.filter(Boolean)
const allFindings = valid.flatMap(r => (r.findings || []).map(f => ({ ...f, lens: r.lens })))
log(`collected ${allFindings.length} findings from ${valid.length}/${LENSES.length} reviewers`)

// verify only the consequential ones (high/critical); pass through the rest
const toVerify = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')
log(`adversarially verifying ${toVerify.length} high/critical findings`)
phase('Verify')
const verdicts = await parallel(toVerify.map(f => () =>
  agent(`${ctx}\n\nA reviewer raised this ${f.severity} finding. Adversarially VERIFY it against the ACTUAL files/data. ` +
        `Default to skeptical: only 'confirmed' if you reproduced the defect. If it's a misunderstanding of the design, say 'not-a-defect'.\n\n` +
        `TITLE: ${f.title}\nLOCATION: ${f.location}\nDEFECT: ${f.defect}\nWHY: ${f.why}`,
    { label: `verify:${(f.location||f.title).slice(0,40)}`, phase: 'Verify', schema: VERDICT_SCHEMA })
    .then(v => ({ ...f, verification: v }))
))
const verified = verdicts.filter(Boolean)
const confirmed = verified.filter(v => v.verification &&
  (v.verification.verdict === 'confirmed' || v.verification.verdict === 'partly'))

phase('Synthesize')
const synthesis = await agent(
  `${ctx}\n\nSynthesize this adversarial review into a verdict for the user. Below are all findings (with ` +
  `verification verdicts for the high/critical ones). Dedupe overlapping findings. Produce:\n` +
  `(1) an overall verdict: is this baseline SOUND, SOUND-WITH-FIXES, or HAS-SERIOUS-PROBLEMS?\n` +
  `(2) the confirmed must-fix items (ranked), each with file:line and the fix;\n` +
  `(3) what was checked and found CLEAN (so the user knows the coverage);\n` +
  `(4) any residual disagreement among reviewers.\n` +
  `Be specific and honest — the user wants AI-common-errors (cheating/fake-report/setup) caught, not flattery.\n\n` +
  `ALL FINDINGS:\n${JSON.stringify(allFindings, null, 1)}\n\n` +
  `VERIFICATION VERDICTS (high/critical):\n${JSON.stringify(verified.map(v => ({ title: v.title, sev: v.severity, loc: v.location, verdict: v.verification?.verdict, ev: v.verification?.evidence, corrected: v.verification?.corrected_severity })), null, 1)}`,
  { label: 'synthesis', phase: 'Synthesize' })

return {
  reviewers: valid.length,
  total_findings: allFindings.length,
  high_critical: toVerify.length,
  confirmed_after_verify: confirmed.length,
  confirmed_items: confirmed.map(c => ({ severity: c.verification?.corrected_severity || c.severity, title: c.title, location: c.location, lens: c.lens })),
  synthesis,
}
