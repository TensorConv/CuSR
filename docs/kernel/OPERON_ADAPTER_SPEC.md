# Operon → pop.bin adapter — authoritative spec

Status: **core algorithm validated end-to-end** (6/6 round-trips exact against Operon's
own evaluator, incl. non-commutative SUB/DIV, nested, weighted variables; K matches
`CoefficientsCount` on all). This document is the single source of truth — every
implementation/test/review task must follow it exactly. Do **not** re-derive any of these
facts; they were established empirically against pyoperon 0.6.1 in `/home/weish/hao/operon-venv`.

Goal: convert an Operon `Tree` (a GP individual) into the `pop.bin` format consumed by the
CuSR batched-LM kernel, so that (a) Operon pre-CO populations become a workload corpus
comparable to the evogp corpus, and (b) Operon's own `LMOptimizer` can serve as a CPU
baseline on the *identical* trees. The conversion must be **exact**: the converted tree,
evaluated by the kernel/our reference interpreter, must produce the same values as Operon's
interpreter on the same data.

---

## 1. pop.bin byte layout (see `cusr/kernel/pop_format.h`, `loader.c`)

Little-endian, all int32/float32, no padding. Order on disk:

```
[PopHeader 64 B]   magic=0x4D4C344D, version=1, M_prob, total_nodes, total_c,
                   N, n_vars, K_max, max_stack, reserved[7]
[nt_all   total_nodes * int32]   node types
[nv_all   total_nodes * float32] node values
[ci_all   total_nodes * int32]   const index (-1 if not CONST else 0..K-1)
[metas    M_prob * 16 B]         TreeMeta{node_offset, n_nodes, c_offset, K}
[c_init   total_c * float32]     initial constants (forward-prefix order across the tree)
[xs       N * n_vars * float32]  xs[i*n_vars + k]
[ym       M_prob * N * float32]  ym[m*N + i]   (all trees share one y → replicate M times)
```

`metas[m] = (node_offset into nt/nv/ci, n_nodes, c_offset into c_init, K)`. Header invariants
(checked by loader + inspect): `sum(K)==total_c`, `sum(n_nodes)==total_nodes`,
`K_max==max(K)`, `max_stack==max stack depth`. **Reuse the existing writer** — factor the
pop.bin writer out of `dump_evogp.py::_write_pop_bin` into a shared module so the byte format
is produced in exactly one place. Run `cusr/kernel/inspect` on every produced `pop.bin`; it
must PASS.

## 2. pop.bin semantics — PREFIX storage, reverse evaluation

Trees are stored **prefix** (root at index 0). The kernel `eval_tree_d`
(`batch_lm.cu:63`) evaluates by iterating **i = n_nodes-1 → 0** with a stack:

```
for i in n-1..0:
    VAR  (nt=0): push xs[(int)nv[i]]          # nv = column index
    CONST(nt=1): push c[ci[i]]                # value from c_init via ci
    UFUNC(nt=2): a=pop;            push f(a)   # nv = Func id
    BFUNC(nt=3): l=pop; rv=pop;    push (l OP rv)   # l popped first, rv second
return stack[0]
```

Consequence for a binary node stored prefix as `[op, A, B]` (A then B follow the op):
reverse iteration processes B's subtree first (B value lands on stack), then A's (A on top),
then the op pops `l=A, rv=B` → computes **A OP B**. So **A is the left/first operand, B the
right/second** (numerator for DIV, minuend for SUB). Emit children in left→right order after
the op.

Type enum `NType`: `VAR=0, CONST=1, UFUNC=2, BFUNC=3` (TFUNC=4 unused — Operon never produces it).
Func id enum (subset we use): `ADD=1, SUB=2, MUL=3, DIV=4, SIN=14, COS=15, TAN=16`
(full enum in `inspect.c`: POW=6, LOG=20, EXP=22, INV=23, NEG=25, ABS=26, SQRT=27).

`ci`: within each tree, the CONST nodes get index `0,1,..,K-1` in the order they appear in the
prefix array (forward-prefix rank); non-CONST nodes get `-1`. `c_init` lists the K constant
values in that same order. `max_stack` = simulated max stack depth over the reverse walk
(VAR/CONST +1; UFUNC net 0; BFUNC net −1). Compute it the same way `dump_evogp` does.

## 3. Operon tree model (validated)

- `Tree.Nodes` is stored **postfix** (children before parent). For a node at index `i` with
  `Node.Length = L` (subtree size − 1) and `Node.Arity = a`, its children are the `a`
  contiguous postfix subtrees ending at `i-1`. **The child nearest the parent (ending at
  `i-1`) is operand[0] (left); walking backward gives operand[0], operand[1], …** (verified:
  `X1 - X2` stores `[X2, X1, Sub]` and evaluates to `x0 - x1`, i.e. the node at `i-1` is the
  minuend).
- **Every leaf is an optimizable coefficient.** A `Variable` node represents `Value * x_col`
  (an optimizable weight, default 1.0); a `Constant` node represents `Value`. Internal
  function nodes have `Optimize=False` and an unused `Value`.
- `Tree.CoefficientsCount` == number of leaves == K for our purposes. `Tree.GetCoefficients()`
  returns the coefficient values in node order. **Use these only as an assertion** — read each
  leaf's own `.Value` directly (Guard 1: order-independent by construction).
- A `Variable` node's dataset column is found via `Node.HashValue` → `Dataset` variable index.
  **Never assume positional.** Build `hash2idx = {v.Hash: v.Index for v in ds.Variables}`.

## 4. The mapping (the crux — reviewed heavily)

For each Operon tree, recurse the postfix array into prefix output, expanding leaves:

- `Constant(v)` → one `CONST` node; its `c_init` slot = `v`.
- `Variable(weight=w, col=c)` → **expand to `MUL(CONST=w, VAR=c)`** = prefix `[MUL, CONST, VAR]`
  (3 nodes, 1 coefficient slot). `VAR.nv = c = hash2idx[node.HashValue]`. The CONST slot = `w`.
  (MUL is commutative so operand order is irrelevant here, but keep `[MUL, CONST, VAR]`.)
- function node (arity `a`) → our `(NType, Func)` from the table in §6; then its children in
  left→right order (operand[0] first).

This makes K = `CoefficientsCount` (= #leaves). Operon trees are therefore ~2.7× more
coefficient-dense than evogp trees (where only explicit constant-leaves count) — this is a
**representation difference to report, not a bug** (see §9).

`c_init` is collected from each `CONST` node's value **in prefix emission order** (so `ci`
prefix-rank is consistent). Assert `len(c_init) == tree.CoefficientsCount` and
`multiset(c_init) == multiset(GetCoefficients())`.

Filters (count + log each, never silently drop): K > MAX_K (=32; structurally rare, see §10);
any unsupported NodeType (must be 0 under the restricted grammar — surface loudly if not);
non-finite c_init.

### Proven reference converter (golden seed — harden + test it, don't re-derive)

```python
N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_MUL = 3
# OP2FUNC maps int(NodeType) -> (NType, Func); see §6

def convert(tree, hash2idx):
    """Operon postfix Tree -> our prefix arrays (nt, nv, ci, c_init)."""
    nodes = list(tree.Nodes)            # `tree` MUST stay referenced (lifetime)
    def conv(i):                        # returns (tokens, subtree_start_index)
        nd = nodes[i]
        if nd.IsConstant:
            return [(N_CONST, 0.0, float(nd.Value))], i
        if nd.IsVariable:
            col = hash2idx[int(nd.HashValue)]
            return [(N_BFUNC, float(F_MUL), None),
                    (N_CONST, 0.0, float(nd.Value)),     # weight
                    (N_VAR, float(col), None)], i
        nty, fid = OP2FUNC[int(nd.Type)]
        kids = []; j = i - 1
        for _ in range(nd.Arity):
            toks, start = conv(j); kids.append(toks); j = start - 1
        # NO reverse: child nearest parent (i-1) is operand[0] (left)
        out = [(nty, float(fid), None)]
        for k in kids: out += k
        return out, j + 1
    toks, _ = conv(len(nodes) - 1)
    # ... pack toks into nt[], nv[], ci[] (prefix-rank CONSTs), c_init[]
```

### Proven reference interpreter (host, mirrors the kernel exactly — for tests)

```python
def ref_eval(nt, nv, ci, c_init, X):    # X: (N, n_vars) float32
    st = []
    for i in range(len(nt) - 1, -1, -1):
        t = int(nt[i]); v = nv[i]
        if   t == N_VAR:   st.append(X[:, int(v)].astype(np.float32))
        elif t == N_CONST: st.append(np.full(X.shape[0], c_init[ci[i]], np.float32))
        elif t == N_UFUNC: a = st.pop();  st.append({14:np.sin,15:np.cos,16:np.tan}[int(v)](a))
        elif t == N_BFUNC:
            l = st.pop(); rv = st.pop()
            st.append({1:l+rv, 2:l-rv, 3:l*rv, 4:l/rv}[int(v)])
    return st[0]
```

## 5. The killer test (TDD)

Round-trip: build known Operon trees via `InfixParser.Parse(expr, {name:hash})` (incl.
`X1 - X2`, `X1 / X2`, weighted `3*X1 - 2*X2`, nested `(X1-X2)/(X1+X3)`, unary `sin(X1)*X2`);
evaluate truth with `op.EvaluateTrees([tree], ds, op.Range(0,N), out_f32, 1)`; `convert`;
`ref_eval`; assert `max|truth − ours| < 1e-3` on finite entries **and** `len(c_init)==CoefficientsCount`.
This single assertion covers order, opcode, expansion, and coefficient mapping. Also test:
per-leaf c_init values, ci prefix order, determinism (same tree → identical bytes), the filters,
and edge cases (single Constant; single Variable; K=0 impossible since every leaf counts; K
near MAX_K). `inspect` must PASS on produced pop.bin.

## 6. NodeType → (NType, Func) map

Restricted grammar (mirrors evogp `USING_FUNCS` {+,−,*,/,sin,cos,tan}):

| Operon NodeType | int value   | NType | Func |
|-----------------|-------------|-------|------|
| Add             | 1           | BFUNC | 1 ADD |
| Mul             | 2           | BFUNC | 3 MUL |
| Sub             | 4           | BFUNC | 2 SUB |
| Div             | 8           | BFUNC | 4 DIV |
| Sin             | 4194304     | UFUNC | 14 SIN |
| Cos             | 32768       | UFUNC | 15 COS |
| Tan             | 67108864    | UFUNC | 16 TAN |
| Constant        | 1073741824  | —     | leaf → CONST |
| Variable        | 2147483648  | —     | leaf → MUL(CONST,VAR) |

Any other NodeType under this grammar is an error (log + filter the tree, count it). Do not
silently map. (If the grammar is later widened: Exp→22, Log→20, Sqrt→27, Pow→6 BFUNC;
`Square(x)`→`x*x`; `Aq` is analytic-quotient `a/sqrt(1+b²)` and must be disabled, not mapped.)

## 7. Operon API gotchas (cause real segfaults — obey exactly)

- **LIFETIME (the #1 crash):** hold a live Python reference to *every* operator for the GP's
  whole lifetime — dataset, problem, primitive set, config, creator, tree-initializer,
  coefficient-initializer, dispatch table, evaluator, `LMOptimizer`, `CoefficientOptimizer`,
  crossover, **each** sub-mutation, the `MultiMutation`, selector(s), generator, reinserter,
  the GP. Creating any of these in a temporary (e.g. a list comprehension passed to
  `MultiMutation.Add`) lets Python GC it → dangling C++ pointer → segfault in `Run`. Assign to
  named variables / a retained container.
- `gp.Run(rng, callback, threads)`: `callback` **must be callable** — `None` segfaults; use
  `def cb(): pass` (or a real per-generation snapshot callback).
- `gp.Generation` is a **property** (no parens).
- `gp.Individuals` has `population_size + pool_size` entries; the population is the first
  `population_size` (slice it). `gp.Parents` / `gp.Offspring` return a `std::span` that does
  **not** convert to Python — do not use them.
- Grammar: `problem.ConfigurePrimitiveSet(combined_int)` where
  `combined_int = int(NT.Add)|int(NT.Sub)|...|int(NT.Variable)`; then `pset = problem.PrimitiveSet`.
  (`PrimitiveSet.SetConfig` rejects a combined value; `IsEnabled(int)` treats the int as a node
  hash, not a type.)
- No inline CO: `GeneticAlgorithmConfig(..., local_iterations=0, p_local=0.0)`.
- `problem.Target` wants a `Variable` (e.g. `ds.Variables[-1]`), not a hash; `problem.InputHashes`
  wants a list of hashes.
- `Tree.SetCoefficients` wants a float32 ndarray (we only *read* coefficients, via each node's
  `.Value`).
- Reading population tree `.Nodes` is safe **once lifetime is correct** (the earlier ".Nodes
  crash" was a misdiagnosis — `Run` was dying first from the GC'd mutations).

## 8. Comparability with the evogp corpus (must hold)

- Same problems: the Feynman + Nguyen set already in `dump_evogp.py::PROBLEMS`. Factor
  `PROBLEMS` + the sampler into a shared `sr_problems.py` (pure numpy+sympy, **no evogp/torch
  import** so it loads in operon-venv) and have both paths use it. Identical `(X, y)` for the
  same `(problem, seed, N, noise)` — same `np.random.default_rng(seed)`, same ranges, same
  sympy skeleton + ground-truth constants, same RMS-relative Gaussian noise model as
  `_sample_xy`. Add a test asserting `sr_problems.PROBLEMS` matches `dump_evogp.PROBLEMS`
  (parse the latter's source to avoid importing evogp).
- Same pop=4000, N=1000, seeds {0,1,2}, caps {32,64}, checkpoint gens {0,1,2,4,8,16,32,64,100}.
- fp32 throughout (kernel design target). Restricted grammar = evogp funcset.
- Operon length cap maps to evogp `max_tree_len` (both = node count): set the tree-initializer
  max length and the crossover/`ReplaceSubtreeMutation` length limits to the cap.

## 9. Anti-cheat / honesty requirements (user ask #5; advisor Guard 2)

- **Rank-deficiency is NOT mutual confirmation.** evogp shows rank-deficient Cholesky from
  *bloat-driven redundant constants*; Operon will show it from *built-in per-leaf weight
  over-parameterization* (`2.5*X1*X2` carries 3 coefficients for 1 real DOF). Same symptom,
  different mechanism. The cross-validation must distinguish them, not claim Operon "confirms"
  the evogp finding.
- Report Operon's higher coefficient density (K≈#leaves) explicitly as a representation
  difference.
- For any loss/quality comparison, **independently recompute the loss in fp64 from the dumped
  constants** — never trust an engine's self-reported fitness (fp32 kernel vs fp64 Ceres/Eigen).
- Disclose + control Operon thread count (the box is 2× EPYC 7763 = 256 threads; "N cores ≈ 1
  A100"). Pin threads for determinism.
- Report **all** problems; never cherry-pick. `log()` every filtered/dropped tree count. A
  bounded run must say what it bounded.

## 10. Empirical facts (already measured — don't re-measure to "discover" them)

- K-distribution (I.18.12, pop=4000, no inline CO, restricted grammar), `K = CoefficientsCount`:
  cap=32 → maxK 15 at every gen; cap=64 → maxK 28→31 by gen100. **`K>32` is 0.0% at every
  checkpoint/cap → MAX_K=32 is safe, no kernel recompile.** Still count K-over per cell and
  verify it stays ~0 across all problems/seeds.
- Density: cap64/gen100 meanK≈22 over ≈51 nodes vs evogp's ≈8 over ≈53 → Operon ≈2.7× denser.
- GP loop is reliable (0 crashes over many trials) once §7 lifetime is obeyed.

## 11. Paths / how to run

- Operon env: `/home/weish/hao/operon-venv/bin/python` (pyoperon 0.6.1, numpy, sympy, scipy,
  sklearn, pytest 9.1). Repo: `/home/weish/hao/CuSR`.
- New code under `cusr/benchmark/workload/`: `sr_problems.py`, `operon_adapter.py`,
  `operon_harvest.py`, and a host reference interpreter (may live in `operon_adapter.py` or a
  `pop_ref.py`). Tests under `cusr/benchmark/workload/tests/` run with the operon-venv python.
- pop.bin tooling: `cusr/kernel/{pop_format.h, loader.c, inspect.c, dump_evogp.py}`.
- Operon `.bin` are gitignored (like evogp's); commit the per-cell `manifest.json` and the
  results archive. Snapshots root mirrors evogp: `data/workload/snapshots/operon_*`.
