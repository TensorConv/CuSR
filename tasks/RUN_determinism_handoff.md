# RUN — Controlled determinism + clean-baseline experiment (C2 hardening)

Status: **DRAFT for review** (design red-team + codex) → then execute. Created 2026-06-24.
Owner: main agent (serial measurement). Repo: `/home/weish/hao/CuSR`.

## 0. Why this experiment
The e7 Section-2 audit (4 independent agents) found that **both crossover headlines
(revad-vs-128c: stable ~9.6× and noisy 28×) rest on Operon baseline cells that were never
directly sentineled, and sit in regimes with 27–43% cross-run drift.** That drift is
cross-run on a **shared** box → it mixes Operon's own variance with co-tenant contention and
**cannot be attributed**. Consequence: every crossover multiplier is currently "approximate"
only. The GPU side is solid (cross-seed CV 0.6% at M≥64k); the **CPU baseline is the shaky part**.

This experiment produces, under **controlled** conditions (locked clocks, pinned cores, single
session, no cross-day/cross-tenant contamination):
1. **A clean Operon baseline** at the crossover configs → tighten multipliers from "±30%" to a real number.
2. **GPU vs Operon latency determinism (CV)** → if Operon CV ≫ GPU CV *even when pinned*,
   "deterministic GPU latency vs CPU jitter on heterogeneous populations" is a clean, defensible
   contribution. If Operon CV collapses when pinned, the 43% was environment → say so, drop the claim.
3. **A timing-window systematic-bias check** (the standing concern: single-run `loop_ms` is short,
   ~20ms–sub-second; random jitter is provably small but that does NOT exclude a systematic short-region bias).

## 1. 铁律 (anti-fraud — non-negotiable)
- **不准编数**: every number traceable to a disk artifact (raw reps jsonl + computed CV json).
- **锁频**: lock GPU SM clock @ 1410 MHz; record locked MHz + GPU id per record; read back after.
- **绑核记录**: record the exact `taskset` mask, NUMA node, thread count, `CUDA_VISIBLE_DEVICES` per record.
- **租户快照**: snapshot `nvidia-smi` + other-user top CPU consumers **at measurement time** → `tenant_evidence.txt`.
  If a co-tenant lands on our bound cores/GPU mid-run, **PAUSE and disclose**.
- **fp64 独立重算**: never trust backend self-reported loss; recompute via the existing `interp.loss_pop` gate.
- **guard 不绕过**: the new `--repeat` flag must NOT bypass `assert_device_jacobian` / the PROFILE guards;
  the binary must still pass them per rep.
- **串行**: GPU and Operon measurement run **serially** (one process at a time on the bound resource).
  No fan-out across agents for measurement (Workflow is for review + audit only).
- **失败照实报**: if the long-window reveals bias, if Operon CV stays high, if the box gets contended,
  if a config OOMs — report honestly, do not reinterpret.

## 2. Hardware + binding
- 8×A100-SXM4-80GB (sm_80), lock SM @ 1410. Use a verified-idle GPU.
- CPU: 2× AMD EPYC 7763 (64 cores/socket, SMT2 = 256 threads). **NUMA node0 = CPU 0-63,128-191;
  node1 = 64-127,192-255** (numactl absent → use `taskset -c`).
- Other tenant: `luoq` (~3 CPU cores). Before binding, check which NUMA node luoq sits on; pin Operon
  to the **other** node where possible.
- **"64c" = one NUMA node's 64 physical cores** (e.g. `taskset -c 0-63`). **"128c" = both sockets' 128
  physical cores** (`taskset -c 0-127`) — deliberately spans NUMA to expose cross-socket variance.
  Set Operon thread count to match. Default: physical cores only (no SMT siblings); record the choice.

## 3. Experiment design
Configs (the crossover-relevant, paper-grade region): **preset = early-gen, N = 1000, M ∈ {16000, 64000}**.
- GPU variants: {revad (hero), ad, fusedfd}. Operon core counts: {64, 128}.
- **Reps: R = 20 measured + 1 warmup discarded, all in ONE session, same fixed population per M**
  (generate once, reuse across reps + variants → CV reflects *timing*, not workload).
- Per config report: median, min, max, **CV = stddev/mean** of `loop_ms` (GPU) / `wall_core` (Operon).

- **Part 1 — GPU determinism**: R reps per GPU config; CV of `loop_ms`.
- **Part 2 — Operon determinism + clean baseline + NUMA**: R reps per Operon config at 64c (1 node) and
  128c (2 sockets); CV of `wall_core` + clean median throughput. Compare 64c-vs-128c CV to test cross-NUMA as a variance source.
- **Part 3 — timing-window bias**: for {revad, fusedfd} × {16k, 64k}, run the LM loop with `--repeat K`
  (K so that K×loop ≥ ~1 s); per-gen throughput = work / (loop_ms / K). Compare to single-loop (K=1).
  **Bias disproven if they agree within ~1%.** Discard the first internal repeat to avoid a cold-cache confound.

## 4. Binary change (`--repeat`)
Add `--repeat K` (default 1) to the `-DPROFILE` kernel binaries: wrap the timed LM-loop region in
`for(r=0;r<K;r++)`, report `loop_ms` for the whole region (divide by K downstream). **TDD**: a test
asserting `loop_ms(K=4) ≈ 4×loop_ms(K=1)` within tolerance, and that `assert_device_jacobian` still passes.
Rebuild; re-verify the existing anti-fraud guards stay green.

## 5. Artifacts (`experiments/e7_section2/out/determinism/`)
`reps_raw.jsonl` (every rep: config, binding, locked_mhz, gpu_id, loop_ms/wall_core, tenant ref) ·
`cv_gpu.json`, `cv_operon.json` (per-config median/min/max/CV) · `timing_window.json` (K=1 vs K-large %diff) ·
`crossover_clean.json` (recomputed revad-vs-{64c,128c} multipliers from the clean baseline, CV-derived error bars) ·
`tenant_evidence.txt`, `clock_lock_{pre,post}.txt`, `binding.txt`.

## 6. Success criteria / what to report
- GPU CV and Operon CV per config; explicit verdict: Operon CV ≫ GPU CV under control (→ determinism
  contribution) or not (→ retract; 43% was environment).
- Clean crossover multipliers with error bars (supersede "approximate ±30%").
- Timing-window: bias present? (Y/N + %). If N → the short-`loop_ms` concern is empirically closed;
  if Y → re-derive all GPU throughputs.
- All null/ugly results reported as-is.

## 7. Orchestration
- **Design red-team (this review): Workflow**, parallel lenses + codex, BEFORE building.
- **Build + measure: main agent, serial** (铁律).
- **Results audit: Workflow**, adversarial agents falsify each headline CV/baseline number from `reps_raw.jsonl`.
- **Final review: codex (gpt-5.5 / xhigh)** over the whole round (design + results + anti-fraud).

## 8. Open questions for the red-team to resolve
- Is CV the right determinism statistic for R=20, or report a bootstrap CI / IQR instead?
- Is "128c spanning 2 sockets" the honest analogue of the paper's "128-core EPYC", or should both
  be single-socket? (The paper's e6 baseline used 128 cores — confirm how it was bound there.)
- Does `--repeat` with first-repeat-discarded truly isolate systematic bias, or leave a confound?
