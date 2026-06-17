# v4 AD Jacobian — validation tooling (2026-06-17, A100)

Reproducibility backing for `docs/kernel/AD_JACOBIAN_V4.md` §9. **Only the scripts are
tracked** — raw `*.txt` outputs are gitignored (regenerate by running them; headline
numbers live in §9). Scripts hardcode `/tmp/...` output dirs; adjust paths to re-run.
Always `source scripts/env.sh` first; GPU work uses `CUDA_VISIBLE_DEVICES=0`.

| script | what it measures (→ output) |
|---|---|
| `run_compare.sh` | builds old `batch_lm_fusedfd`, runs AD + FD on `data/fixtures/pop.bin`, then `test_parity_gate.py` for each vs scipy-fp64. AD 80.8% / FD 92.4% within_1.05×. → `compare_output.txt` |
| `headtohead.py` | per-tree `loss_AD` vs `loss_FD` on the **common converged** set (no scipy). Decisive no-bug test: 95.3% agree within 1.05× both ways. |
| `setcontrolled.py` | within_1.05× vs scipy on the **stable** (AD&FD&scipy all converged) set. Env-parameterizable (`SETCTRL_POP/AD_OUT/FD_OUT`) → runs on any corpus. Post-fix: **pop.bin AD 91.1% ≈ FD 89.9%**, **high-K feyn AD 81.9% ≈ FD 82.3%** → gate gap is set composition, not quality. |
| `coverage_probe.py` | for trees one engine converged but the other did NOT, fp64-recompute BOTH losses (no scipy): real fit gap or just a CONVERGED-labeling gap? High-K feyn: **AD matches FD within 1.05× on 91.8%** of FD's extra trees → FD's higher converged COUNT is ~92% labeling, not better fits. |
| `status_trade.py` | convergence trade by status code (pre-fix snapshot; net +62 was inflated by fake convergence, see below). |
| `throughput_sweep.py` | AD vs FD end-to-end trees/s across M (synth early-gen 1k–256k). |
| `operon_sweep.py` | AD vs FD trees/s on the **Operon pre-CO benchmark corpus** (realistic high K). |

**Bottom line** (updated post stop-criterion fix, commit `8b9234b`): AD has no derivative
bug, and **delivered fit quality is AD ≈ FD on both corpora** (set-controlled pop.bin
91.1≈89.9, high-K feyn 81.9≈82.3). The earlier "**net more robust** (494>432)" was
**fake-convergence inflation** — the LM accept/reject committed uphill steps as CONVERGED;
now fixed in both engines (h_loss monotone non-increasing). Post-fix FD shows a higher
converged COUNT at high K (2051 vs 1393), but `coverage_probe.py` shows ~92% of that is a
**labeling artifact**, not better fits. AD's real edge: **~1.25–1.3× throughput + exact /
no-eps**. The within_1.05× raw gate is **not set-invariant** (penalizes converging more
hard trees) — judge by set-controlled + delivered loss, not converged-count. NOTE:
`docs/kernel/AD_JACOBIAN_V4.md` §9 predates this fix (old fake-inflated numbers) — update pending.
