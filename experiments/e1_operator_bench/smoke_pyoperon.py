"""W2 day-1 smoke: operator-level LM constant optimization with pyoperon.

Question: can pyoperon optimize constants of a *given* tree (no evolution loop)?
Target expr: y = 3.0*sin(2.0*x1) + 0.5*x1, 256 points, x in [0.1, 5].

Covers:
  (a) tree construction: InfixParser.Parse vs manual postfix Node list
  (b) LM with controlled iterations + hand-set initial constants
  (c) readback: fitted constants, final cost, iteration count
  (d) per-tree overhead over ~100 trees, serial vs ThreadPoolExecutor (GIL probe)

Run: uv run python experiments/012_op_bench/smoke_pyoperon.py
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyoperon as op

# ---------------------------------------------------------------- data
N = 256
TRUE = np.array([0.5, 2.0, 3.0])  # postfix coefficient order: slope, freq, amp
x = np.linspace(0.1, 5.0, N)
y = 3.0 * np.sin(2.0 * x) + 0.5 * x

ds = op.Dataset(np.asfortranarray(np.column_stack([x, y])))
# NOTE: ds.VariableNames() is broken in the pypi wheel (nanobind can't convert
# list[std::string]); iterate ds.Variables instead.
vmap = {v.Name: v.Hash for v in ds.Variables}  # {'X1': ..., 'X2': ...}; X2 = y
target = max(ds.Variables, key=lambda v: v.Index)

problem = op.Problem(ds)
problem.TrainingRange = op.Range(0, N)
problem.TestRange = op.Range(N - 1, N)  # unused but must be set
problem.Target = target
problem.InputHashes = [vmap["X1"]]

dtable = op.DispatchTable()
rng = op.RandomGenerator(42)


def clone(tree: op.Tree) -> op.Tree:
    """Deep-copy a tree via its node list. GOTCHA: the bound copy ctor
    op.Tree(tree) SEGFAULTS in the pypi 0.6.1 wheel — do not use it."""
    return op.Tree(tree.Nodes).UpdateNodes()


def freeze_variable_weights(tree: op.Tree) -> op.Tree:
    """Operon variable nodes carry an optimizable weight (w*X). Freeze them so
    LM only sees the explicit constants. GetCoefficients/SetCoefficients
    respect the per-node Optimize flag."""
    nodes = tree.Nodes  # list of copies
    for n in nodes:
        if n.IsVariable:
            n.Optimize = False
    return op.Tree(nodes).UpdateNodes()


# ---------------------------------------------------------- (a) construction
# Route 1: parse from infix string. Variable map: name -> dataset hash.
t_parsed = freeze_variable_weights(
    op.InfixParser.Parse("3.0 * sin(2.0 * X1) + 0.5 * X1", {"X1": vmap["X1"]})
)

# Route 2: manual postfix node list (children first, then parent).
def var_node(h: int) -> op.Node:
    n = op.Node(op.NodeType.Variable)
    n.Value = 1.0          # variable weight w in (w * X1)
    n.HashValue = h        # which dataset column
    n.Optimize = False     # freeze the weight
    return n

h1 = vmap["X1"]
t_manual = op.Tree(
    [
        # 3.0 * sin(2.0 * X1)
        var_node(h1), op.Node.Constant(2.0), op.Node.Mul(),
        op.Node.Sin(),
        op.Node.Constant(3.0), op.Node.Mul(),
        # + 0.5 * X1
        var_node(h1), op.Node.Constant(0.5), op.Node.Mul(),
        op.Node.Add(),
    ]
).UpdateNodes()

for label, t in [("parsed", t_parsed), ("manual", t_manual)]:
    pred = np.asarray(op.Evaluate(t, ds, op.Range(0, N)))
    err = np.abs(pred - y).max()
    print(f"(a) {label:6s} len={t.Length} coeffs={t.GetCoefficients()} "
          f"infix={op.InfixFormatter.Format(t, ds, 3)} max|pred-y|={err:.2e}")
    assert err < 1e-4, f"{label} tree does not reproduce y"

# ----------------------------------------------------- (b)(c) perturb + LM
# GetCoefficients order is postfix order: [0.5 (slope), 2.0 (freq), 3.0 (amp)].
# Perturbation: x1.5 on slope/amplitude; x1.1 on the sin frequency.
# (x1.5 on the frequency lands outside the LM basin -> local minimum; shown below.)
PERTURB = np.array([1.5, 1.1, 1.5])

def run_lm(tree: op.Tree, init: np.ndarray, max_iter: int = 100):
    t = clone(tree)  # CoefficientOptimizer does not mutate its input
    t.SetCoefficients(init.astype(np.float32))
    # GOTCHA: CoefficientOptimizer stores a raw pointer to the optimizer.
    # op.CoefficientOptimizer(op.LMOptimizer(...)) with a temporary is a
    # use-after-free -> nondeterministic segfault. Keep `lm` referenced.
    lm = op.LMOptimizer(dtable, problem, max_iter=max_iter, batch_size=N)
    co = op.CoefficientOptimizer(lm)
    t_opt, summary = co(rng, t)
    return t_opt, summary

t_opt, s = run_lm(t_parsed, TRUE * PERTURB)
final = np.array(t_opt.GetCoefficients())
rel = np.abs(final - TRUE) / np.abs(TRUE)
print(f"(b) init={list(TRUE * PERTURB)}")
print(f"(c) success={s.Success} iters={s.Iterations} fevals={s.FunctionEvaluations} "
      f"jevals={s.JacobianEvaluations}")
print(f"(c) cost {s.InitialCost:.4f} -> {s.FinalCost:.4g}")
print(f"(c) params {s.InitialParameters} -> {s.FinalParameters}")
print(f"(c) recovered={final} rel_err={rel}")
assert s.Success
assert rel.max() < 1e-2, f"constant recovery failed: rel_err={rel}"

# cost semantics: ceres-style 0.5 * SSE over the training range
t_chk = clone(t_parsed)
t_chk.SetCoefficients((TRUE * PERTURB).astype(np.float32))
pred_p = np.asarray(op.Evaluate(t_chk, ds, op.Range(0, N)))
print(f"(c) cost check: InitialCost={s.InitialCost:.4f} vs 0.5*SSE={0.5*np.sum((pred_p-y)**2):.4f}")

# iteration cap: reported Iterations == max_iter + 2 (Eigen LM bookkeeping
# steps); the cap does bound the work and cost decreases with larger cap.
# max_iter=0 is a no-op (0 evals, Success=False).
for mi in (1, 2, 3):
    _, s_cap = run_lm(t_parsed, TRUE * PERTURB, max_iter=mi)
    print(f"(b) max_iter={mi} -> iters={s_cap.Iterations} "
          f"fevals={s_cap.FunctionEvaluations} cost={s_cap.FinalCost:.4g}")
    assert s_cap.Iterations <= mi + 2

# known failure mode: x1.5 on the sin frequency -> local minimum (not a bug)
_, s_bad = run_lm(t_parsed, TRUE * 1.5)
print(f"(b) all-x1.5 init: success={s_bad.Success} final_cost={s_bad.FinalCost:.4g} "
      f"(stuck != 0 -> frequency outside basin, expected)")

# ------------------------------------------------------------- (d) batching
M = 100
rs = np.random.default_rng(0)
inits = TRUE * rs.uniform(0.9, 1.1, size=(M, 3))  # within-basin jitter
trees = []
for i in range(M):
    t = clone(t_parsed)
    t.SetCoefficients(inits[i].astype(np.float32))
    trees.append(t)

def make_co(max_iter=20):
    # per-thread optimizer stack; problem/dataset shared (read-only at opt time).
    # Return (lm, co) so the LMOptimizer outlives the CoefficientOptimizer
    # (raw-pointer lifetime gotcha, see run_lm).
    lm = op.LMOptimizer(dtable, problem, max_iter=max_iter, batch_size=N)
    return lm, op.CoefficientOptimizer(lm)

# serial
_lm1, co1 = make_co()
rng1 = op.RandomGenerator(1)
t0 = time.perf_counter()
res_serial = [co1(rng1, t) for t in trees]
dt_serial = time.perf_counter() - t0
n_ok = sum(s.FinalCost < 1e-3 for _, s in res_serial)
iters = [s.Iterations for _, s in res_serial]
print(f"(d) serial: {M} trees in {dt_serial*1e3:.1f} ms "
      f"-> {dt_serial/M*1e6:.0f} us/tree (iters min/med/max {min(iters)}/"
      f"{int(np.median(iters))}/{max(iters)}, {n_ok}/{M} converged <1e-3)")

# binding overhead floor: max_iter=0 is a no-op call (0 evals)
_lm0, co0 = make_co(max_iter=0)
t0 = time.perf_counter()
for t in trees:
    co0(rng1, t)
dt0 = time.perf_counter() - t0
print(f"(d) max_iter=0 (no-op) calls: {dt0/M*1e6:.1f} us/tree (binding floor)")

# GIL probe: ThreadPoolExecutor with per-thread optimizers.
# CONCLUSION (also verified with a 200k-row dataset, 11.8 ms/call: speedup
# 0.98-1.01x at 2/4/8 threads): CoefficientOptimizer.__call__ does NOT release
# the GIL -> no thread-level parallelism. There is also no native batched
# optimize API (EvaluateTrees(nthread=...) exists for inference only; the GA
# parallelizes per-individual CO internally via taskflow, not exposed at
# operator level). Multi-core CPU-LM baseline => multiprocessing.
for nthreads in (2, 4, 8):
    stacks = [make_co() for _ in range(nthreads)]
    cos = [co for _, co in stacks]
    rngs = [op.RandomGenerator(100 + i) for i in range(nthreads)]
    chunks = [list(range(i, M, nthreads)) for i in range(nthreads)]

    def work(k):
        return [cos[k](rngs[k], trees[j]) for j in chunks[k]]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(nthreads) as ex:
        out = list(ex.map(work, range(nthreads)))
    dt_par = time.perf_counter() - t0
    print(f"(d) threads={nthreads}: {dt_par*1e3:.1f} ms "
          f"-> speedup {dt_serial/dt_par:.2f}x")

print("SMOKE PASSED")
