# korns_corpus__20260617

**Korns inner-constant (frequency-class) trees harvested on the A100 pipeline.**
The realistic GP corpus for the regime where constant optimization (CO) is
*decisive* — true constants buried INSIDE cos/sin as frequencies, which linear
least-squares cannot fit. This is the "CO should shine" battlefield for the
planned end-to-end CO experiment; it is **not** that experiment.

- **Read `report.html`** for the full bilingual (中/EN) writeup (theme-adaptive).
- Why this corpus: the 17-problem Feynman+Nguyen corpus is essentially
  inner-constant-free (true constants are outer scales or absent → CO least
  useful). These two Korns targets carry genuine inner constants:
  - `korns/11`  `6.87 + 11*cos(7.23*x0**3)`  — 1 inner const (freq 7.23, cube)
  - `korns/12`  `2 - 2.1*cos(9.8*x0)*sin(1.3*x1)`  — 2 inner consts (freqs 9.8, 1.3)

## What's here
- `summarize.py` — loads the korns snapshot manifests, aggregates K/node drift,
  runs FD+AD kernels on a sample (convergence/fail vs gen), writes `report.json`
  + CSV + plots.
- `data/` — `summary.json`, `drift.csv`, `kernel_profile.json`.
- `plots/` — `workload_drift.png`, `kernel_vs_gen.png`.
- The corpus itself: `data/workload/snapshots/korns_{11,12}_pop4000_*` (24 cells /
  216 snapshots / 864k trees). The `.bin` are gitignored; each cell's
  `manifest.json` is the committed, regenerable record.

## Key findings (honest)
1. **Inner-const regime built** — 24 cells / 216 snapshots / 864k trees, 0 dropped,
   0 K-over. Matrix matches the Feynman corpus (noise{0,1%} × cap{32,64} ×
   seed{0,1,2} × gen{0..100}, pop=4000, N=1000).
2. **The workload difference is bloat TIMING, not endpoint** — both bloat to a
   similar high K by gen100 (cap64 ≈ 12.3–12.7). korns/11 (near-unsolvable
   cube+high-freq) bloats early (K≈14 by gen16); korns/12 stays sparse mid-gen
   (K≈1.9 at gen16) and only bloats late. The clean low-K operating point is
   korns/12 at early/mid gen.
3. **Kernel runs fine; rank-deficiency tracks bloat, not inner-consts per se** —
   FD & AD both crash-free, finite returns. Cholesky-fail is low early/mid (10–25%)
   and both collapse to rank-deficient at late-gen bloat (83–94%) — the SAME
   bloat→rank-deficiency mechanism as the Feynman corpus.

## Caveats
- Kernel `converged`/`fail` is a **workload descriptor, not a quality verdict**
  (operon_baseline proved status==0 ≠ genuine improvement). Quality needs neutral
  fp64 scoring — deferred to the end-to-end experiment.
- `korns/11`'s cube+high-freq target on [-5,5] is a known ceiling (near-unsolvable
  even with perfect CO); here it is a high-K + rank-deficiency stress workload, not
  a clean demo. `korns/12`'s early/mid-gen regime is the cleaner demo.
- GP funcset has no POW; korns/11's `x**3` is approximated by multiplication (a
  y-generator, not a GP constraint — same as the Nguyen log/sqrt entries).

Regenerate: see `report.json` → `reproduce`.
