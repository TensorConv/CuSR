"""Smoke test: call PySR's constant-optimization operator (Julia BFGS) standalone.

Goal: (expression tree w/ constants, init constants, X, y) -> optimized constants
+ final loss + eval count, WITHOUT running any evolution. This probes the
"PySR-BFGS" backend for the E1 operator-level benchmark.

Run:  uv run python experiments/012_op_bench/smoke_pysr_co.py

Verified against SymbolicRegression.jl v1.11.3 (pysr 1.5.10 pin), source at
~/.julia/packages/SymbolicRegression/L5TJa/src/ConstantOptimization.jl

防雷规则 (gotchas, read before reusing this as a backend):
1. First `import pysr.julia_import` starts Julia + loads packages: tens of
   seconds. First call of each Julia helper additionally JIT-compiles
   (seconds). E1 protocol: warm up once, time afterwards.
2. `optimize_constants(dataset, member, options)` lives in the NON-exported
   submodule `SymbolicRegression.ConstantOptimizationModule`. Signature:
       (dataset::Dataset{T,L}, member::PopMember{T,L}, options) -> (member, num_evals::Float64)
   num_evals = Optim f_calls summed over all (re)starts * dataset_fraction
   (fraction == 1.0 when batching=false, the default). Optim iteration counts
   are NOT returned — only f_calls.
3. ALGORITHM SWITCH: if the tree has exactly 1 constant, it silently uses
   Newton (+backtracking), NOT BFGS. >=2 constants -> options.optimizer_algorithm
   (BFGS default). Benchmark trees should have >=2 constants, or this must be
   documented per-tree.
4. Default `autodiff_backend=nothing` => Optim.BFGS computes gradients by
   FINITE DIFFERENCES (Optim/NLSolversBase default), exactly like stock PySR.
5. `optimizer_nrestarts=2` (default) adds 2 extra runs from randomly perturbed
   x0 (x .* (1 + 0.5*randn)) => stochastic. Seed Julia RNG (Random.seed!) or
   set nrestarts=0 for determinism. `optimizer_iterations` (default 8) is a
   plain Options kwarg — externally controllable per Options instance.
6. optimize_constants MUTATES the expression's constants in place (and
   restores x0 if no improvement over baseline). Re-parse or copy(ex) per
   call when looping. member.loss is only updated on improvement — recompute
   final loss via eval_loss() to be safe.
7. Data layout: Julia Dataset wants X::(nfeatures, n) Float64 matrix.
   PopMember(dataset, ex, options) REQUIRES `deterministic=` kwarg as a Bool
   (passing nothing throws).
8. Constants order from get_scalar_constants(tree) = depth-first (preorder,
   left child first) over the tree. Same refs order used by
   set_scalar_constants!. Confirmed empirically below.
9. Write constants as float literals in parsed strings and force
   node_type=Node{Float64}; use safe_pow/safe_log/safe_sqrt to match PySR's
   operator mapping (plain ^/log/sqrt throw DomainError on negatives).
10. juliacall does NOT release the Python GIL during Julia calls (measured
    below: python thread progresses at ~1% of free rate) — no Python-side
    thread parallelism across CO calls. Parallelism must live Julia-side
    (Threads.@threads inside one call; PYTHON_JULIACALL_THREADS=auto is set
    by pysr; nthreads()=24 here).
11. Julia helper signatures must accept loose types: a Python list crosses
    as PyList{Any} (NOT Vector{String}), numpy arrays as PyArray — over-typed
    methods throw MethodError. Convert inside the helper.
12. num_evals counts Optim's f_calls only. Under finite-difference gradients
    the FD probe evaluations go through NLSolversBase's separate df path and
    are NOT included => undercounts true residual evals. Don't use it as a
    cross-backend eval-count currency without checking g_calls.
"""

import time
import threading

import numpy as np

# ---------------------------------------------------------------- timed import
t0 = time.perf_counter()
from pysr.julia_import import jl  # noqa: E402  (starts Julia)

T_IMPORT = time.perf_counter() - t0
print(f"[import] pysr.julia_import (Julia start + pkg load): {T_IMPORT:.1f} s")

# ------------------------------------------------------------- Julia helpers
JULIA_HELPERS = r"""
module CoSmoke

using SymbolicRegression:
    Options, Dataset, PopMember, Node,
    parse_expression, string_tree,
    get_scalar_constants, set_scalar_constants!,
    eval_loss,
    safe_pow, safe_log, safe_sqrt
using SymbolicRegression.ConstantOptimizationModule: optimize_constants
import Random

# Full operator set for our benchmark family (PySR-faithful safe_* variants).
# neg == unary -, inv == 1/x, pow == safe_pow.
function make_options(; iterations::Integer=8, nrestarts::Integer=2)
    return Options(;
        binary_operators=[+, -, *, /, safe_pow],
        unary_operators=[sin, cos, tan, sinh, cosh, tanh,
                         safe_log, exp, -, abs, safe_sqrt, inv],
        optimizer_iterations=iterations,
        optimizer_nrestarts=nrestarts,
        deterministic=false,
    )
end

# String -> Expression{Float64} tree. Constants must be float literals.
# NOTE: accept AbstractVector, not Vector{String} — a Python list crosses the
# bridge as PyList{Any} and would not dispatch (防雷 #11).
function parse_tree(s::AbstractString, options, variable_names)
    vnames = String[string(v) for v in variable_names]
    return parse_expression(
        Meta.parse(s);
        operators=options.operators,
        variable_names=vnames,
        node_type=Node{Float64},
    )
end

seed!(s::Integer) = (Random.seed!(s); nothing)

# One standalone CO call. Copies `ex` so the caller's tree is not mutated.
function run_co(ex, Xpy, ypy, options)
    X = Matrix{Float64}(Xpy)            # (nfeatures, n)
    y = Vector{Float64}(ypy)
    dataset = Dataset(X, y)
    tree = copy(ex)
    member = PopMember(dataset, tree, options; deterministic=options.deterministic)
    x0, _ = get_scalar_constants(member.tree)
    loss0 = member.loss
    member, num_evals = optimize_constants(dataset, member, options)
    xopt, _ = get_scalar_constants(member.tree)
    final_loss = eval_loss(member.tree, dataset, options; regularization=false)
    return (x0=x0, xopt=xopt, loss0=loss0, loss=final_loss,
            num_evals=num_evals, tree_str=string_tree(member.tree))
end

# Timing loop kept Julia-side so per-call Python<->Julia overhead can be
# separated from the operator cost itself.
function run_co_loop(ex, Xpy, ypy, options, n::Integer)
    X = Matrix{Float64}(Xpy)
    y = Vector{Float64}(ypy)
    dataset = Dataset(X, y)
    times = Vector{Float64}(undef, n)
    evals = Vector{Float64}(undef, n)
    for i in 1:n
        tree = copy(ex)
        member = PopMember(dataset, tree, options; deterministic=options.deterministic)
        t = time_ns()
        member, ne = optimize_constants(dataset, member, options)
        times[i] = (time_ns() - t) / 1e9
        evals[i] = ne
    end
    return (times=times, evals=evals)
end

busy_wait(sec::Real) = (t = time(); while time() - t < sec end; nothing)

nthreads() = Threads.nthreads()

end # module
"""

t0 = time.perf_counter()
CoSmoke = jl.seval(JULIA_HELPERS)
print(f"[seval] helper module defined: {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
opts8 = CoSmoke.make_options(iterations=8, nrestarts=2)      # PySR defaults
opts200 = CoSmoke.make_options(iterations=200, nrestarts=2)  # relaxed budget
opts8_det = CoSmoke.make_options(iterations=8, nrestarts=0)  # deterministic
print(f"[options] 3 Options built (first incl. JIT): {time.perf_counter() - t0:.1f} s")

# ------------------------------------------------------------------- problem
# Target: y = 3.0*sin(2.0*x1) + 0.5*x1 ; start from constants * 1.5.
rng = np.random.default_rng(0)
N = 256
X = rng.uniform(-3.0, 3.0, size=(1, N))  # (nfeatures, n) Julia layout
y = 3.0 * np.sin(2.0 * X[0]) + 0.5 * X[0]

TRUE_EXPR = "3.0 * sin(2.0 * x1) + 0.5 * x1"
PERT_EXPR = "4.5 * sin(3.0 * x1) + 0.75 * x1"  # x1.5 perturbed constants
TRUE_CONSTS = np.array([3.0, 2.0, 0.5])        # preorder DFS order (verified below)

# --------------------------------------------------------------- (a) parsing
t0 = time.perf_counter()
ex = CoSmoke.parse_tree(PERT_EXPR, opts8, ["x1"])
t_parse_first = time.perf_counter() - t0
print(f"\n[a] parse_expression OK ({t_parse_first:.2f} s first call)")
print(f"    string_tree: {jl.SymbolicRegression.string_tree(ex)}")
x0_check = np.asarray(CoSmoke.run_co(ex, X, y, opts8_det).x0)
print(f"    get_scalar_constants order (expect preorder 4.5, 3.0, 0.75): {x0_check}")

# ------------------------------------------------- (b)+(c) single CO call
jl.seval("CoSmoke.seed!(0)")
t0 = time.perf_counter()
res = CoSmoke.run_co(ex, X, y, opts8)
t_first_co = time.perf_counter() - t0  # includes Optim/BFGS JIT
print(f"\n[b] optimize_constants first call (incl. JIT): {t_first_co:.1f} s")
print(f"    x0        = {np.asarray(res.x0)}")
print(f"    xopt      = {np.asarray(res.xopt)}")
print(f"    loss0     = {res.loss0:.6g}")
print(f"    loss(fin) = {res.loss:.6g}")
print(f"    num_evals = {res.num_evals}")
print(f"    tree      = {res.tree_str}")

# -------------------------------------------------------------- (e) accuracy
print("\n[e] recovery: y = 3.0*sin(2.0*x1) + 0.5*x1, init = 1.5x constants")
for label, o in [("iters=8, nrestarts=2 (PySR default)", opts8),
                 ("iters=8, nrestarts=0", opts8_det),
                 ("iters=200, nrestarts=2", opts200)]:
    jl.seval("CoSmoke.seed!(0)")
    r = CoSmoke.run_co(ex, X, y, o)
    xopt = np.asarray(r.xopt)
    err = np.abs(xopt - TRUE_CONSTS)
    print(f"    [{label}]")
    print(f"      xopt={np.round(xopt, 6)}  abs_err={np.round(err, 6)}  "
          f"max_abs_err={err.max():.3g}  loss={r.loss:.3g}  num_evals={r.num_evals}")

# ---------------------------------------------------------------- (d) timing
print("\n[d] cost (warm, Julia-side timer excludes Python<->Julia crossing):")
for label, o in [("default iters=8 nrestarts=2", opts8),
                 ("iters=200 nrestarts=2", opts200)]:
    jl.seval("CoSmoke.seed!(0)")
    loop = CoSmoke.run_co_loop(ex, X, y, o, 100)
    times = np.asarray(loop.times)
    evals = np.asarray(loop.evals)
    print(f"    [{label}] 100 calls: mean={times.mean()*1e3:.2f} ms  "
          f"median={np.median(times)*1e3:.2f} ms  p95={np.percentile(times,95)*1e3:.2f} ms  "
          f"mean_num_evals={evals.mean():.1f}")

# per-call overhead from Python side (1 full run_co incl. crossing + Dataset build)
jl.seval("CoSmoke.seed!(0)")
t0 = time.perf_counter()
n_py = 20
for _ in range(n_py):
    CoSmoke.run_co(ex, X, y, opts8)
t_py = (time.perf_counter() - t0) / n_py
print(f"    python->julia full run_co (incl. crossing+Dataset): {t_py*1e3:.2f} ms/call")

# GIL check: does a Python thread make progress while Julia busy-waits?
counter = [0]
stop = [False]


def spin():
    while not stop[0]:
        counter[0] += 1


th = threading.Thread(target=spin)
th.start()
time.sleep(0.5)
base = counter[0]
time.sleep(1.0)
rate_free = counter[0] - base          # increments during 1 s of pure Python
base = counter[0]
CoSmoke.busy_wait(1.0)                 # 1 s busy-wait inside Julia
rate_julia = counter[0] - base
stop[0] = True
th.join()
gil_released = rate_julia > 0.05 * rate_free
print(f"    GIL during Julia call: python-thread progress = "
      f"{rate_julia}/{rate_free} ({100*rate_julia/max(rate_free,1):.1f}% of free rate) "
      f"=> GIL {'RELEASED' if gil_released else 'HELD'}")
print(f"    Julia Threads.nthreads() = {CoSmoke.nthreads()}")

print(f"\n[summary] import={T_IMPORT:.0f}s  first_co={t_first_co:.1f}s (JIT, exclude per E1 protocol)")
print("done.")
