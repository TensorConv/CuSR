# Kernel CO Optimization Backlog (e6-derived, 2026-06-21)

Status tags: **[measured]** real data · **[prior]** reasoned, unmeasured · **[gated:X]** needs
measurement X first · **[refuted]** tried & failed · **[deploy]** only pays off in the real EvoGP
loop, not the standalone benchmark.

All compute-side numbers are **DRAFT** (clocks unlocked) and gated on the two pending measurements
below. Don't quote them as final.

---

## ⚠️ MEASURED UPDATE 2026-06-21 (§0.1 DONE — REFUTES the host-bound prior this doc was built on)

The PROFILE breakdown is in and it OVERTURNS "small M = host-round-trip bound". Measured (N=1000,
PROFILE_JSON host_residual vs gpu_sum):

| config | loop_ms | host_residual | host% | dominant GPU cat |
|---|---|---|---|---|
| fusedfd early M=4000 | 222 | 5.2 | **2%** | fd_jacobian 66% |
| fusedfd early M=64000 | 1161 | 47 | **4%** | fd_jacobian 50% |
| fusedfd inner M=64000 | 3111 | 70 | **2%** | fd_jacobian 60% |
| ad inner M=64000 | 2451 | 72 | **3%** | fd_jacobian 49% |

The LM loop is **>95% GPU compute at every measured M**; the host round-trip (accept/reject + memcpy)
is only **2–5%**. Consequences (these SUPERSEDE §1–§3 below):
- **On-device loop (old §3.1 "primary, ~2×") is REFUTED → DEMOTED to a ~2–5% micro-opt.**
- The small-M throughput gap (M=4000 ≈ 43% of saturation) is **GPU under-occupancy** (per-tree GPU
  time ~2.4× higher at M=4000 than 64000), NOT host overhead → fix = run larger M (design choice), not
  remove the round-trip.
- **The real lever is the Jacobian kernel (60–66% of the loop) → reverse-mode AD (§3.2) is PROMOTED to
  #1**, esp. high-K inner-const (fd_jacobian 1870 of 3111 ms @ M=64k). eval (~15%) + build_jtj (~7–17%)
  are secondary. (Lesson in action: measured, and it refuted my own prior — that's why we measure.)
- Still pending: §0.2 ncu profiling (SOL + warp-stall; instruction roofline as the figure — the
  interpreter is issue/SFU-bound, NOT FLOP/BW-bound, so a classic FLOP roofline diagnoses nothing).
  Profiled at N=1000 only; N=100 (smaller kernels) may show higher host% but is the
  peak-throughput regime anyway.

---

## Framing: population size M is a DESIGN CHOICE, not a fixed "realistic 4000"

The "~4000" below is an **inherited CPU-era population size, NOT a constraint**. CPU GP uses small
populations because the CPU can't afford large ones (CO ∝ M). Our data inverts this: the GPU saturates
at M≈64k, advantage GROWS with M, while Operon caps ~10k t/s and barely scales past 16 cores (measured
M=256k N=100: nc 16→128 = 8.4k→9.9k). So the value prop is **"GPU unlocks the large-M regime CPU GP
can't afford"** — natural operating point ~64k (16× the CPU-era 4000), reachable via bigger populations
/ island models / parallel restarts / multi-problem batching (memory allows M up to ~1M @ N=1000 on one
A100). The small-M analysis below is still valid WHERE small M occurs (small islands, CPU-era configs),
but **"GPU starved at realistic M" is the wrong headline** — on a GPU you'd choose large M.
Two-layer honesty: kernel = large-M CO is cheap on GPU [measured]; "therefore large M helps SR" =
end-to-end, needs the demonstrator [deploy].

---

## 0. Pending measurements (the gate — do these FIRST)

1. **PROFILE breakdown** — ✅ **DONE 2026-06-21 (see MEASURED UPDATE above: host 2–5%, Jacobian 60–66%;
   on-device loop refuted, reverse-AD promoted).** [was gated: after Phase B; ran on free GPUs once
   Operon done]. Used the `_prof` binaries' `PROFILE_JSON` (`host_residual_ms`,
   `gpu_sum_ms`, per-category: fd_jacobian/build_jtj/solve/eval/residual/loss/memcpy_H2D/D2H) at
   **M=4000 and M=64000, both fusedfd & ad**. Decides: how much is host round-trip vs GPU compute
   at realistic M (→ #1 payoff), and which kernel dominates at large M per variant.
2. **ncu profiling — SOL + warp-stall reasons (NOT a classic FLOP roofline)** [gated: sudo/clock-lock].
   The kernel runs at <1% of BOTH fp32-FLOP and HBM-BW peak (measured 0.18–0.53% FLOP / 0.33–0.96% BW
   at M=16k–256k), so a classic FLOP roofline only shows a dot far under both roofs and diagnoses
   nothing — it's issue/latency-bound, not compute- or bandwidth-bound. PRIMARY = ncu Speed-of-Light +
   warp stall-reason breakdown on the hot kernels (Jacobian, build_jtj, eval): where the cycles go
   (expect MIO/SFU throttle from transcendentals + execution-dependency from the stack-machine chain +
   branch divergence). FIGURE = instruction roofline (GIPS vs instruction intensity; Ding & Williams
   2019), the roofline variant appropriate for an issue-bound kernel. Keep the classic FLOP roofline
   only as a one-line counter-evidence ("not bandwidth- or compute-bound" → justifies skipping the §4
   memory-layout work). Decides: whether memory-layout / double-buffer matter at all (prior says NO —
   see §4) and the realizable issue-rate ceiling.
3. **per-iteration active-tree histogram** (convergence front-loading) — decides compaction payoff
   (#5). Can be added as an additive diagnostic to the LM loop.

---

## 1. Bottleneck map (regime-dependent) [prior, to confirm by §0.1]

| regime | bottleneck | same for ad/fd? |
|---|---|---|
| small M (small islands / CPU-era ~4000) | **host orchestration + per-iter H2D/D2H sync** (GPU underfed; identical loop) | **YES** |
| large M (≥64k) / high N (saturated) | **Jacobian kernel** (the only differing part) + build_jtj; compute-bound | **NO** — fd: K re-eval passes; ad: ceil(K/W) tangent passes (heavier registers) |

Evidence: M=4000 runs at ~43% of saturated throughput (ad early N=1000: 30k vs 69k). The "rises then
saturates ~M=64k" shape = a fixed per-iter overhead being amortized. fp32 FLOP utilization at
saturation is ~0.1–0.2% of peak → **interpreter/instruction-bound, NOT FLOP- or bandwidth-bound**.

---

## 2. Theoretical headroom [prior]

- **Small M (~4000, e.g. small islands):** ~**2×** — closing the host-overhead gap to saturation.
  Measurable via §0.1. (On GPU you'd instead run large M near the ~64k operating point — see Framing.)
- **Large-M compute ceiling:** the kernel is at ~0.1% of FP32 FLOP-peak, but that's misleading — it's
  interpreter-bound, not FLOP-bound. Realizable headroom ≈ **few× to ~10×** via algorithm (§3.1, §3.4),
  exact bound **gated on §0.2 (ncu profiling)**. Do NOT promise raw FLOP-peak ratios.
- **End-to-end (deploy):** warm-start (§3.2) can cut iterations **5–10×** in the real loop.

---

## 3. Optimization directions (ranked by leverage × feasibility)

### Reliable near-term
1. **On-device per-iter accept/reject + λ update** — kill the host round-trip. Attacks the
   structurally-confirmed small-M bottleneck; helps **both variants equally**; well-scoped (per-tree
   arithmetic, no Cholesky; outer loop stays on host, all-done reduced every K iters not every iter).
   At large M it also deletes ~2.3 GB of PCIe traffic (delta D2H = M·K_max·4 = 33 MB/iter @ M=256k).
   **DEMOTED by §0.1 measurement** — host is only 2–5% of the loop, so this saves ~2–5%, NOT ~2×. Keep
   as a minor opt, NOT the primary pick. (The large-M PCIe-bytes point still holds, but the loop is
   GPU-bound there too, so the win is small.)

### High-leverage algorithmic bets
2. **Reverse-mode AD Jacobian (v5)** — per point the output is scalar, so one forward+backward gets
   ALL K partials at ~2× eval, **independent of K**, vs forward-mode's ∝K. ⟹ high-K trees
   (inner-const K~13) Jacobian **~6× cheaper**, and the Jacobian dominates at large M. Cost: store the
   forward tape (n_nodes intermediates/point) → register/shared pressure (SR trees small, n_nodes
   tens → feasible). The principled upgrade for the compute-bound high-K regime. **PROMOTED to #1 by
   §0.1: the Jacobian is 60–66% of the LM loop (measured) — this is THE lever, esp. high-K inner-const.**
3. **warm-start CO from inherited constants** [deploy] — offspring start from parent's optimized
   constants → converge in 1–3 iters not 50. **5–10× fewer iterations** in the real EvoGP loop.
   Biggest practical win; invisible in the standalone benchmark.
4. **codegen / specialize the interpreter** — compile each tree to straight-line code instead of a
   switch-per-node stack machine; kills branch divergence + interpreter overhead. **Raises the §2
   compute ceiling.** Biggest but hardest (GPU per-tree JIT is hard; group-by-structure / templated /
   offline codegen are the routes).

### Approximations (quality tradeoff, all quantifiable)
5. **Subsampled / stochastic Jacobian** — compute J on a random N/4–N/10 subset, full N only for
   loss/accept. Cheap where Jacobian dominates (high N); risk = noisier J → maybe a few more iters.
6. **Early-stop the convergence tail** — [measured] inner-const 50→25 iters = loss +14%; easy presets
   ~unchanged. Cheap, preset-dependent quality cost.
7. **Adaptive per-tree CO budget by fitness** [deploy] — doomed offspring get few iters, promising
   ones get full; cuts total CO work in the real loop.
8. **compaction** (lossless) — late iters have few active trees (frac_converged 0.60–0.81); pack the
   active set to restore occupancy. **Gated on §0.3** (only worth it if convergence is front-loaded).

### Cheap add-ons (low risk, opportunistic)
9. **N → multiple of 32** — **[refuted/measured 2026-06-22]** the "~25% tail waste" is NOT
   recoverable for a fixed problem. Per-launch GPU time tracks `ceil(N/32)` warp-iterations: a
   partial tail iteration costs full wall-time (masked lanes still issue) in this latency/
   instruction-bound kernel, so **rounding N *up* to a multiple of 32 is a pure no-op** — measured
   AD eval/launch N=100→128 = 0.0976→0.0974 ms (0.0%), jac −5.5% (slightly worse: 28 more real
   points). The only "win" is rounding *down* (100→96 = +20%) which just drops 4% of the data
   (smaller problem). 128B row alignment of d_J/d_ym is moot (measured <1% HBM peak). N is a
   dataset property, not a free knob. ⇒ **dead as a kernel opt.** Where the size IS a free choice —
   the subsampled/stochastic Jacobian subset (#5) — pick a multiple of 32 for free warp efficiency.
   (artifacts: data/workload/synth/synth_early-gen_M16000_N{96,100,128,992,1000,1024}_seed0.bin)
10. **Fuse residual+loss kernels** — one fewer launch/iter.

---

## 4. Wrong tools / low-leverage here (why) [prior, confirm via §0.2]

- **Double buffering / async transfer overlap** — the LM iteration is a strict serial dependency
  chain; per-iter H2D/D2H are ON the critical path (next compute needs the transfer result), so
  there's no independent work to hide them behind. And at small M the cost is sync *latency* (not
  bandwidth), at large M it's pure PCIe *bytes* — neither is hidden by double-buffering given the
  dependency. The fix is to **eliminate** the round-trip (§3.1), not hide it. (Async DOES apply to:
  intra-kernel software pipelining in the compute regime — §0.2-gated — and cross-generation pop
  prefetch in the in-process deploy path.)
- **Memory layout (coalescing / SoA / alignment)** — these are *bandwidth* tools, but the kernel is a
  branchy stack-interpreter + transcendentals = **compute/instruction-bound, not bandwidth-bound**.
  Existing layouts are already mostly coalesced (d_J/d_ym rows: lane-stride consecutive i; nodes:
  broadcast-read). Only real candidate: **d_xs AoS→SoA** (`[point][var]`→`[var][point]`), and only if
  n_vars>1 (SR usually small). **Low priority; gated on §0.2 — if ncu surprises us with memory-bound,
  do d_xs SoA first.**

---

## 5. Refuted / out of scope

- **Damping fixes for the inner-const quality ceiling — REFUTED, intrinsic.** See
  `docs/MIXED_PRECISION_PROBE.md`: fp64 K×K solve (+0 tier-B), pivot-floor (failed W0 parity gate,
  broke 123 healthy trees), QR/rank-reveal (healthy trees also rank-deficient, spectra overlap),
  best-seen (no-op). Root cause = structural rank deficiency in high-K GP trees; safe baseline is
  correct. `-DTRUST_REGION` (λI) is the one unvalidated candidate but predicted to fail the same way.
  **Do NOT re-propose damping.** This is a quality issue, not a speed lever, and the honest paper
  framing is an intrinsic ceiling.
- **fp16 eval** — conditioning-risky; we already hit an fp32 ceiling on inner-const.

---

## 6. fd vs ad (for reference)

ad leads for deploy/headline: median **1.25×** faster (range 0.84–1.97), exact Jacobian (no FD eps
wart). fd is NOT useless: independent secant-vs-analytic **parity oracle** (mutual correctness check),
wins throughput in some low-N/high-M shapes (ad's tangent lanes → register pressure), lower register
pressure / simpler. AD handles ALL operators incl non-smooth (max/min branch-select, comparisons→0
tangent, abs→sign) — fd has NO operator-generality edge (checked `ad_interp.cuh`). Report both in the
paper (agreement = correctness evidence; the throughput crossover is itself a finding).
