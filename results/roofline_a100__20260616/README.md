# roofline_a100__20260616

**ANALYTICAL** roofline for `batch_lm_fusedfd` on the A100. ncu (measured) is blocked
(`ERR_NVGPUCTRPERM`, perf counters admin-locked, no sudo), so FLOPs + DRAM bytes are
modeled from the algorithm and combined with the measured Tier0 wall time.

- **Read `report.html`** for the full structured writeup.
- `roofline_model.py` — the model (op-FLOP table, per-iter FLOP/byte formula, A100 peaks).
- `roofline_data.json` — computed AI + per-point achieved FLOP/s.
- `plots/roofline.png` — the roofline.

**Headline:** AI ≈ 5.3 FLOP/byte (< A100 ridge 9.56), but achieved is only **0.18–0.53%
of fp32 peak** (16k→256k). The kernel sits ~2 orders of magnitude **below** both ceilings →
bound by **execution efficiency / host-orchestration overhead**, NOT compute or bandwidth.
Big optimization headroom (divergent interpreter, warp-per-tree, 1-thread Cholesky, K
redundant full-tree FD re-evals, per-iter H↔D copies).

**TODO (needs admin/root):** enable `NVreg_RestrictProfilingToAdminUsers=0` → run ncu for a
MEASURED roofline that separates per-kernel GPU compute from the host/launch overhead the
wall-time analytical version conflates.

Regenerate: `uv run python results/roofline_a100__20260616/roofline_model.py results/roofline_a100__20260616 && uv run python scripts/make_report.py results/roofline_a100__20260616/report.json`
