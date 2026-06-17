# v4 AD Jacobian — validation tooling + measured logs (2026-06-17, A100)

Reproducibility backing for `docs/kernel/AD_JACOBIAN_V4.md` §9. These are the
exact scripts + raw logs that produced the v4 numbers. Scripts hardcode `/tmp/cmp/`
output dirs (binary outputs `ad_out/`, `fd_out/`); adjust paths to re-run.

| file | what it measures |
|---|---|
| `run_compare.sh` | builds old `batch_lm_fusedfd`, runs AD + FD binaries on `data/fixtures/pop.bin`, then `test_parity_gate.py` for each vs scipy-fp64. → `compare_output.txt` |
| `compare_output.txt` | AD vs FD parity gate output (AD 80.8% / FD 92.4% within_1.05×, etc.) |
| `headtohead.py` | per-tree `loss_AD` vs `loss_FD` on the **common converged** set (no scipy). Decisive no-bug test: 95.3% agree within 1.05× both ways. |
| `setcontrolled.py` | within_1.05× vs scipy on the **stable** (AD&FD&scipy all converged) set. → `setcontrolled_output.txt`: **AD 92.1% ≈ FD 91.8%** → regression is set composition, not quality. |
| `setcontrolled_output.txt` | set-controlled output |
| `status_trade.py` | the convergence trade by status code: AD gains 115 (45 ex-Cholesky-fail), loses 53 (50 just maxiter), net +62. |
| `profile_ad_vs_fd.txt` | rung-5 per-kernel profiling (-DPROFILE): J production 88.7→29.7 ms (**3.0× faster**), loop 131→77 ms (1.70×). |

Re-run: `source scripts/env.sh && uv run python cusr/kernel/tests/v4_validation/<script>.py`
(after `bash run_compare.sh` regenerates `/tmp/cmp/ad_out` + `fd_out`).

**Bottom line**: AD has no derivative bug; it is net more robust; the within_1.05×
parity gate is not set-invariant. See `docs/kernel/AD_JACOBIAN_V4.md` §9.
