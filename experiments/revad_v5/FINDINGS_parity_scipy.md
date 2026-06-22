# Reverse-AD v5 — parity divergence resolved by scipy fp64 oracle (2026-06-22)

## Question
The real-pop element-wise parity test (`test_revad_parity.cu`, inner-const M4000 N1000) reported
17048 `finite-mismatch` elements: **all** of the form `fwd=nan, rev=<finite>` (0 rel-failures;
9052 both-non-finite OK; max rel err where both finite = 5.14e-7). Is forward-AD or reverse-AD
correct at those points?

## Method
`_dump_jac_sample.cu` dumped the ACTUAL forward-AD (`eval_tree_jvp_d`, chunked) and reverse-AD
(`eval_tree_vjp_d`) gradients for 5 singular + 8 clean real-pop trees (+ a few points each).
`compare_scipy.py` recomputed each Jacobian with **`scipy.optimize._numdiff.approx_derivative`
(3-point, fp64)** — an independent interpreter reimplemented in numpy fp64 — and also recorded
whether the tree VALUE at each point is finite (so a finite gradient is meaningful).

## Result (scipy_compare_result.txt)
- AGREED elements (fwd & rev both finite): **n=300, rev~scipy=300, fwd~scipy=300** (sanity ✓).
- DISPUTED elements (fwd/rev differ in finiteness): **n=68**
  - rev matches scipy = **68/68**
  - fwd matches scipy = **0/68**
  - value FINITE & rev=scipy & fwd nonfinite (REVERSE correct) = **68**
  - value NON-finite (function undefined; fwd NaN defensible) = **0**
- Example: m=1247, value=`-0` (finite), true grad=`0` → fwd=`nan`, **rev=`0`=scipy**.

## Verdict
**Reverse-AD is MORE correct than forward-AD.** Forward-AD's chunked JVP NaN-contaminates the
whole tangent vector (all K columns) when any local partial is non-finite (`NaN*0=NaN`), poisoning
sibling columns even where the value and the true gradient are finite. Reverse-AD's per-node
adjoint isolates columns and matches the scipy fp64 ground truth. 0 disputed elements were at
genuinely-undefined (NaN-value) points, so forward's NaN is NOT defensible here.

## Implication
- Keep reverse-AD AS-IS (do NOT replicate forward's NaN-contamination bug — that would degrade
  correctness to match a defect).
- The `rev-vs-fwd` parity test's premise (forward = ground truth) is disproven for the
  `fwd-nan/rev-finite` direction. Correct criterion: that direction is forward's defect (accept,
  count separately); `rev-nan/fwd-finite` and finite rel-mismatch remain FAIL.
- Reframes the contribution: reverse-AD v5 is **faster AND fixes a forward-AD NaN-contamination
  bug** (scipy-validated), not merely a faster equivalent.
