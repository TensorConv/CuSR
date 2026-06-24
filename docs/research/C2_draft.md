# C2 — GPU Batched Heterogeneous SR Constant-Optimization Primitive (paper draft)

> Paper sections this feeds: **§ GPU Primitive** (design, ~1.5 pp) + **§ Evaluation** (eval, ~1.75 pp).
> English draft of the actual paper prose. Every number cites its on-disk artifact; LaTeX
> assembly is deferred (no `main.tex` yet — content first, conversion is mechanical).
> Figures: `experiments/e7_section2/figs/` + `docs/kernel/{fig_architecture,lm_algorithm}.tex`.
>
> **Red lines** (`docs/research/contributions.md`): NOT first GPU CO; NOT a novel optimizer;
> NO "CO improves SR". The claim is a *custom CUDA second-order batched heterogeneous-tree CO
> primitive and its measured envelope*. Baselines we actually ran: Operon + scipy (no PySR/BFGS).
>
> Status: **Design section = first full draft (sample for voice/density/length).**
> Eval section = locked skeleton + two pending decisions (see end).

---

## § GPU Primitive (Design)

A symbolic-regression search produces, each generation, a forest of *M* candidate expression
trees. Before any tree can be scored, its internal constants must be fit to the data — for tree
*m* this is a small nonlinear least-squares (NLS) problem in its *K_m* free constants against one
shared dataset (*X* ∈ ℝ^{N×d}, *y* ∈ ℝ^N). The trees are *structurally heterogeneous* (different
operator mixes) and have *heterogeneous constant counts* *K_m*. We solve all *M* NLS problems in a
single CUDA launch with batched second-order Levenberg–Marquardt (LM). The contribution is systems
design, not a new optimizer: (i) a stack-machine tree encoding and warp-per-tree batching that
absorbs structural and *K* heterogeneity at zero intra-tree divergence; (ii) a per-tree Jacobian
built by reverse-mode automatic differentiation in a single pass, *independent of K* — the
dominant cost in the loop; and (iii) an fp64 honesty guard that makes "delivered loss ≤ initial
loss" an unconditional, per-tree guarantee.

### A. Tree encoding and heterogeneous batching

Each tree is stored as a linearized stack-machine program (`pop_format.h`): three parallel arrays
over the forest's total nodes — `node_type` (int32), `node_value` (float32), and `const_idx`
(int32, −1 unless the node is a constant) — with a per-tree record `(node_offset, n_nodes,
c_offset, K)` that slices them. The free constants are gathered into one flat vector **c**; tree
*m* optimizes the slice `c[c_offset : c_offset+K_m]`, and `const_idx` maps each constant node to
its position in **c**. The kernel therefore operates on *packed* constants, decoupled from how any
host engine lays them out internally.

Batching is **warp-per-tree**: 32 lanes cooperate on one tree, striding over the *N* data points
(lane *ℓ* handles points *ℓ, ℓ+32, …*); distinct trees occupy distinct warps. Because all 32 lanes
execute the *same* program, there is no warp divergence *within* a tree — structural heterogeneity
lives across warps, not inside them. Every tree independently carries its own *K_m*, damping
*λ_m*, and status, so one launch processes a structurally heterogeneous, heterogeneous-*K* batch.
Compile-time bounds (`MAX_K`=32, `MAX_STACK`=64, `MAX_NODES`=128) size the in-register/local
working set and fail loud rather than silently corrupt if exceeded (current workloads peak at
*K_max* ≤ 13 and `n_nodes` ≤ 32 — ≥ 4× headroom). *(Figure: `fig_architecture`.)*

### B. Batched Levenberg–Marquardt solve

The inner solver is standard; our work is doing it batched, in-register, and heterogeneous. Each
tree runs Gauss–Newton with Marquardt damping: per iteration we assemble the per-tree normal
equations from its Jacobian *J_m* ∈ ℝ^{N×K_m} and residual *r_m*, then solve *(J_mᵀJ_m*, diagonal
scaled by *1+λ_m)* *δ_m* = −*J_mᵀr_m* with an in-register *K×K* Cholesky (one thread per tree); a
non-SPD pivot returns `chol=fail`, *δ*=0. Accept/reject is per-tree and **improvement-first**: a
step is accepted only if the loss does not increase, *λ* is scaled by 1/10 on accept and ×10 on
reject, and a tree is declared *Converged* only *after* an accepted, non-uphill step with
‖*δ*‖ < x_tol(‖*c*‖+x_tol) — so a small *uphill* step is never mistaken for convergence. The host
drives the outer loop and per-tree scheduling; the named device kernels run the numerical hot
path. Five statuses (*Converged / MaxIter / Fail_NaN / Fail_Chol / K0_Skip*) are reported, not
hidden. *(Algorithm 1; `lm_algorithm.tex`.)*

### C. Per-tree Jacobian by reverse-mode AD (the main lever)

Profiling places the Jacobian at **60–66% of the LM loop** (Eval §B) — it is the single dominant
cost, so *how J is built* decides the kernel's speed. Each tree is a scalar map ℝ^{K_m}→ℝ (its
*K_m* constants to one output, per data point); its Jacobian is the *K_m* partials ∂ŷ/∂c_j at each
of the *N* points. We implement three builders behind one identical assemble/solve path:

- **Finite differences (FD):** *K_m* perturbed re-evaluations per tree — cost ∝ *K_m*, approximate.
- **Forward-mode AD:** a forward pass carrying a bundle of *W_A*=8 tangents, ⌈*K_m*/*W_A*⌉ passes —
  cost ∝ *K_m*, exact.
- **Reverse-mode AD (deployed):** one forward pass that records each node's local partials onto a
  tape, then one reverse pass whose adjoint stack mirrors the forward value stack — **all *K_m*
  columns in a single pass, cost independent of *K_m***. For a scalar output with many inputs this
  is the natural mode (it is backpropagation); the larger *K_m*, the more it wins, because forward
  passes grow with *K_m* and the reverse pass does not.

Mechanically (`revad_interp.cuh`): the forward sweep (nodes last→first) computes node values and
stores local partials (unary: *df/da*; binary: ∂o/∂l and ∂o/∂r); the reverse sweep (first→last)
seeds the output adjoint at 1 and accumulates ∂ŷ/∂c_j at each constant node. Reverse-mode also
resolves a numerical hazard that *bundled* forward-mode cannot: a single non-finite local partial
in a forward tangent bundle poisons the whole bundle — and its sibling columns — through
0×∞ = NaN; per-node adjoints isolate columns, and a guarded multiply (a true zero factor
annihilates the edge, so it returns 0) keeps a finite value finite where naive AD emits NaN. Thus
reverse-mode is both *K*-independent *and* strictly no less correct than forward-mode (verified
against a scipy fp64 oracle: reverse never worse — Eval §D). Its working set — a two-slot tape plus
value/adjoint stacks, ≈ 1.5 KB/lane — is *smaller* than the forward bundle's. *(The measured
speedups belong to Eval §C; here the point is only why reverse-mode is the right primitive.)*

### D. fp64 honesty guard

For throughput the loop runs in fp32 with `--use_fast_math`, which is unsafe to trust near
singularities. The primitive therefore closes with a **delivery audit in IEEE fp64**: for every
tree it recomputes the loss at the final *and* the initial constants in honest double precision and
reverts the tree to its initial constants whenever the final loss is non-finite or larger than the
initial. This makes **delivered loss ≤ initial loss an unconditional, per-tree guarantee** — a
fast-math fp32 "improvement" near a singularity can never be shipped — and it is *independent of
the kernel's own fp32 self-report*: correctness is judged only by the fp64 re-evaluation, never by
the backend's reported number. The guard is a system feature (cf. conditioning/honesty in prior
art), not an algorithmic claim, and it is precisely what lets a fast fp32 kernel be reported
honestly. *(Algorithm 1, delivery-guard line.)*

---

## § Evaluation (LOCKED SKELETON — to be drafted next)

Figures (triad — see decision 1): roofline + crossover + one scaling. Every number → FINDINGS
artifact under `experiments/e7_section2/`.

**A. Setup.** A100-SXM4-80GB, sm_80, clocks **locked @ 1410 MHz** (`clock_lock_{pre,post}.txt`).
Timing protocol: in-binary CUDA-event `loop_ms` (excludes ~280 ms CUDA-init+load), 3 reps ×
3 seeds → median of 9; headline cells `loop_ms` ≥ 150 ms (M≥16k@N≥1000 or any M≥64k), reproducibility
median 0.6–0.8%. **Workload scope boundary (MUST foreground — advisor):** the three presets'
*structure* is calibrated to 3 real Feynman snapshots (`docs/characterization.md`) = real; the
*targets* are synthetic and recoverable *by design*, for a clean perf+precision benchmark — NOT a
"fits real data" claim. Every point passes an independent fp64 quality gate (铁律: never trust the
backend's self-reported loss).

**B. Profiling (ncu) — the distinctive systems point.** Primary diagnosis, *not* a classic FLOP
roofline: at the working point the Jacobian kernels are **not FLOP-bound** (FMA ≤ 11%, fp64 = 0%)
and **not DRAM-bound** (DRAM ≤ 23%, mostly < 10%) → issue-/on-chip-bound; top warp stalls = `wait`
(execution dependency, the stack-machine serial chain) + `long_scoreboard` (memory latency).
**Instruction roofline figure** (`fig_instruction_roofline`): 146–333 GIPS = 24–55% of the issue
roof at < 1% of FLOP/HBM roofs. This is also the counter-evidence to "why not do memory-layout
optimization." Jacobian = 60–66% of the loop (the §C motivation). [`ncu_summary.md`,
`ncu_instruction_roofline.json`]

**C. Scaling envelope + Jacobian-method comparison.** Peak **361,334 trees/s** (revad early-gen
M=256k N=100, 3-seed median, locked; `gpu_phase_a.json`). One scaling figure (decision 1). Deployed
vs the other Jacobian builders: **revad/fd = 1.50× median**, revad/ad = 1.17× — and forward-AD is
**demoted to one line**: "we also implemented forward-mode AD; reverse-mode is 1.17× faster *on
median* (range dips to 0.86×), avoids tangent NaN-contamination, and is rank-identical (0.995), so
we deploy reverse-mode." [`gpu_phase_a.json`]

**D. Apples-to-apples vs Operon (iso-quality crossover) — baseline = one 64-core EPYC 7763.**
We compare one A100 against **one server CPU** (one EPYC 7763 = 64 cores) — the standard
one-accelerator-vs-one-CPU framing. The dual-socket **128-core data is dropped**: 128c spans the
co-tenanted second NUMA node, so its slowdown is contention, not Operon scaling (FINDINGS §Phase-6;
128c = 6,711 vs 64c = 10,802 t/s, a −38% contention drop). In-loop, iso-quality (in-process
per-generation CO cost; host round-trip measured 2–5%): **revad is ≈ 5.8× a 64-core EPYC in the
stable regime** (early-gen M=16k N=1000; `crossover_clean.json`, GPU CV 0.17% / Operon CV 5.2%), up
to **≈ 14.8×** in the bloated regime (higher variance). Reported as an **upper bound** — Operon ran
on a shared box (tenant_overlap=1.0), which biases the ratio *up*; a precise figure needs an idle
host, so we write "≈" and state the direction. (One-shot e2e, which charges per-call CUDA init that
the in-process C3 path avoids, is lower.) Ranking de-risking: **Spearman(revad, fd) = 0.959,
top-10% selection overlap 95.5%** (`ranking_revad_vs_fd.txt`) — the fp32/AD path does not move
selection. [`crossover_clean.json`, FINDINGS Phase 6]

**E. Limits.** High-*K* rank-deficient quality ceiling is *intrinsic* (damping/fp64/floor/QR all
refuted — `docs/MIXED_PRECISION_PROBE.md`); high-*N* collapse (win is N-dominated); multi-inner is
an open frontier (the study cannot test it).

---

## Two decisions to lock before drafting Eval

1. **Figure triad (BLOCKING — 5 figures won't fit 1.75 pp; ~3 fit).** Fixed: `instruction_roofline`
   + `crossover`. **Pick ONE scaling figure** from {`throughput_surface`, `n_collapse`, `large_m`};
   the other two get cut or merged into subpanels. (Leaning `n_collapse` — the N-sensitivity caveat
   is the distinctive honest point the OUTLINE stresses — but this is a story choice.)
2. ~~In-loop vs e2e crossover sentence~~ **RESOLVED (2026-06-25):** baseline switched to **one
   64-core EPYC 7763**; 128-core data dropped as contention-confounded (FINDINGS §Phase-6). Headline
   = **revad ≈ 5.8× a 64-core EPYC (stable), in-loop, iso-quality, upper bound**. See §D.
