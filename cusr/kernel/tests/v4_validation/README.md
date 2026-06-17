# v4 AD Jacobian — validation tooling (2026-06-17, A100)

Reproducibility backing for `docs/kernel/AD_JACOBIAN_V4.md` §9. **Only the scripts are
tracked** — raw `*.txt` outputs are gitignored (regenerate by running them; headline
numbers live in §9). Scripts hardcode `/tmp/...` output dirs; adjust paths to re-run.
Always `source scripts/env.sh` first; GPU work uses `CUDA_VISIBLE_DEVICES=0`.

| script | what it measures (→ output) |
|---|---|
| `run_compare.sh` | builds old `batch_lm_fusedfd`, runs AD + FD on `data/fixtures/pop.bin`, then `test_parity_gate.py` for each vs scipy-fp64. AD 80.8% / FD 92.4% within_1.05×. → `compare_output.txt` |
| `headtohead.py` | per-tree `loss_AD` vs `loss_FD` on the **common converged** set (no scipy). Decisive no-bug test: 95.3% agree within 1.05× both ways. |
| `setcontrolled.py` | within_1.05× vs scipy on the **stable** (AD&FD&scipy all converged) set: **AD 92.1% ≈ FD 91.8%** → the gate "regression" is set composition, not quality. → `setcontrolled_output.txt` |
| `status_trade.py` | convergence trade by status code: AD gains 115 (45 ex-Cholesky-fail), loses 53 (50 just maxiter), net +62. |
| `throughput_sweep.py` | AD vs FD end-to-end trees/s across M (synth early-gen 1k–256k). |
| `operon_sweep.py` | AD vs FD trees/s on the **Operon pre-CO benchmark corpus** (realistic high K). |

**Bottom line**: AD has no derivative bug; it is net more robust; end-to-end throughput is
a stable ~1.25–1.3× across scale (J-production speedup vs its loop-share trade off; high-K
Operon is a stress test, not the design workload). The within_1.05× parity gate is not
set-invariant. Full numbers + interpretation: `docs/kernel/AD_JACOBIAN_V4.md` §9.
