"""operon_adapter.py — Operon ``Tree`` -> CuSR pop.bin prefix arrays.

Single source of truth: ``docs/kernel/OPERON_ADAPTER_SPEC.md``. This module
converts an Operon GP individual (postfix ``Tree.Nodes``) into the prefix
``(nt, nv, ci, c_init)`` representation consumed by the batched-LM kernel
(``cusr/kernel/batch_lm.cu``), so that (a) Operon pre-CO populations become a
workload corpus comparable to the evogp corpus and (b) Operon's own
``LMOptimizer`` can serve as a CPU baseline on the *identical* trees.

The conversion is **exact**: the converted tree, evaluated by the host reference
interpreter ``pop_ref.ref_eval`` (which mirrors the kernel), reproduces Operon's
own evaluator on the same data (round-trip test, spec §5).

Key facts baked in (do NOT re-derive — proven against pyoperon 0.6.1):
  * ``Tree.Nodes`` is POSTFIX (children before parent). For a function node at
    index ``i`` with arity ``a``, its operands are the ``a`` contiguous postfix
    subtrees ending at ``i-1``; the child nearest the parent (ending at ``i-1``)
    is operand[0] (left). **No children.reverse()** — walking backward from
    ``i-1`` already yields operand[0], operand[1], …
  * EVERY Operon leaf is an optimizable coefficient. A ``Constant(v)`` -> one
    ``CONST`` node (c_init slot = v). A ``Variable(weight=w, col=c)`` ->
    ``MUL(CONST=w, VAR=c)`` = prefix ``[MUL, CONST, VAR]`` (1 coefficient slot).
    Hence ``K == tree.CoefficientsCount == #leaves``.
  * A ``Variable``'s dataset column is found via ``Node.HashValue`` ->
    ``hash2idx`` (NEVER positional).
  * ``c_init`` is read from each leaf's own ``.Value`` in prefix-emission order
    (order-independent by construction); ``GetCoefficients()`` is used only as an
    assertion (multiset + count), never as the value source.

The pop.bin byte format is produced in exactly one place (``pop_io.write_pop_bin``).
``write_operon_pop_bin`` takes the raw Operon trees (+ ``hash2idx``), calls
``convert`` per-tree inside a ``try/except UnsupportedNodeType`` and applies the
spec §4 filters (unsupported NodeType, K > MAX_K, non-finite c_init) with per-reason
counts **returned and logged** — the offending tree is counted+logged+filtered, never
silently dropped and never aborting the whole dump (honesty requirement, spec §9).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyoperon as op

from .pop_io import MAX_K, Func, NType, write_pop_bin

log = logging.getLogger(__name__)

# NType tags (spec §2) — mirror pop_ref / pop_io so the host interpreter agrees.
N_VAR = int(NType.VAR)      # 0
N_CONST = int(NType.CONST)  # 1
N_UFUNC = int(NType.UFUNC)  # 2
N_BFUNC = int(NType.BFUNC)  # 3

F_MUL = int(Func.MUL)       # 3 — the synthetic MUL wrapping a weighted Variable

# Operon NodeType -> our (NType, Func) under the restricted grammar (spec §6).
# Keyed by the *runtime* int(op.NodeType.X) (a bit-flag) rather than the literal
# constants, so a transcription typo in the flag values cannot silently mis-map.
OP2FUNC: dict[int, tuple[int, int]] = {
    int(op.NodeType.Add): (N_BFUNC, int(Func.ADD)),  # 1
    int(op.NodeType.Sub): (N_BFUNC, int(Func.SUB)),  # 2
    int(op.NodeType.Mul): (N_BFUNC, int(Func.MUL)),  # 3
    int(op.NodeType.Div): (N_BFUNC, int(Func.DIV)),  # 4
    int(op.NodeType.Sin): (N_UFUNC, int(Func.SIN)),  # 14
    int(op.NodeType.Cos): (N_UFUNC, int(Func.COS)),  # 15
    int(op.NodeType.Tan): (N_UFUNC, int(Func.TAN)),  # 16
}


class UnsupportedNodeType(ValueError):
    """A NodeType outside the restricted grammar reached ``convert``.

    Raised either by an Operon function NodeType not in ``OP2FUNC`` or by a
    ``Variable`` whose ``HashValue`` is absent from ``hash2idx`` (the latter can
    arise after a ``ChangeVariableMutation``). Under the restricted grammar with a
    matching dataset this must never happen; surface it loudly (spec §6 — "Do not
    silently map"). ``write_operon_pop_bin`` catches this around its per-tree
    ``convert`` call to count+log+filter the offending tree (the
    ``dropped_unsupported`` reason) rather than aborting the whole dump.
    """


def build_hash2idx(ds) -> dict[int, int]:
    """Build ``{variable_hash: dataset_column_index}`` from an Operon ``Dataset``.

    A ``Variable`` node's dataset column is resolved via its ``HashValue``, never
    positionally (spec §3). Matches the form used throughout the tests:
    ``{int(v.Hash): int(v.Index) for v in ds.Variables}``.
    """
    return {int(v.Hash): int(v.Index) for v in ds.Variables}


def convert(tree, hash2idx):
    """Operon postfix ``Tree`` -> our prefix arrays ``(nt, nv, ci, c_init)``.

    Recurse the postfix node array into prefix output, expanding leaves:
      * ``Constant(v)``            -> ``[CONST]``           ; c_init slot = v
      * ``Variable(w, col=c)``     -> ``[MUL, CONST, VAR]`` ; CONST = w, VAR.nv = c
      * function node (arity a)    -> ``[Func]`` then operand[0..a-1] left->right

    Returns ``(nt, nv, ci, c_init)`` as numpy arrays with the kernel dtypes
    (nt int32, nv float32, ci int32, c_init float32). ``ci`` is the forward-prefix
    rank of each CONST node (-1 for non-CONST); ``c_init`` lists the CONST values
    in that same order.

    Asserts (spec §4): ``len(c_init) == tree.CoefficientsCount`` and
    ``multiset(c_init) == multiset(tree.GetCoefficients())`` (compared in fp64,
    before the float32 cast, so it won't spuriously fire on evolved trees).

    Raises ``UnsupportedNodeType`` if a NodeType outside the restricted grammar
    is encountered (caller may catch to filter+count the tree).
    """
    nodes = list(tree.Nodes)  # `tree` MUST stay referenced for the call's lifetime

    def conv(i):
        """Return (tokens, subtree_start_index) for the postfix subtree ending at i.

        ``tokens`` is a list of ``(nty, nv, const_value_or_None)`` in prefix order.
        """
        nd = nodes[i]
        if nd.IsConstant:
            return [(N_CONST, 0.0, float(nd.Value))], i
        if nd.IsVariable:
            h = int(nd.HashValue)
            if h not in hash2idx:
                raise UnsupportedNodeType(
                    f"Variable hash {h} not in hash2idx (unknown dataset column)"
                )
            col = hash2idx[h]
            return (
                [(N_BFUNC, float(F_MUL), None),     # MUL
                 (N_CONST, 0.0, float(nd.Value)),   # weight (the coefficient)
                 (N_VAR, float(col), None)],        # VAR -> dataset column
                i,
            )
        key = int(nd.Type)
        if key not in OP2FUNC:
            raise UnsupportedNodeType(
                f"NodeType {key} unsupported under the restricted grammar "
                f"(arity={nd.Arity}) at postfix index {i}"
            )
        nty, fid = OP2FUNC[key]
        # Operands are the `arity` contiguous postfix subtrees ending at i-1.
        # Child nearest the parent (i-1) is operand[0]; walk backward. NO reverse.
        kids = []
        j = i - 1
        for _ in range(nd.Arity):
            toks, start = conv(j)
            kids.append(toks)
            j = start - 1
        out = [(nty, float(fid), None)]
        for k in kids:
            out += k
        return out, j + 1

    toks, _ = conv(len(nodes) - 1)

    # Pack tokens into nt/nv/ci/c_init; CONST nodes get forward-prefix rank in ci.
    n = len(toks)
    nt = np.empty(n, dtype=np.int32)
    nv = np.empty(n, dtype=np.float32)
    ci = np.full(n, -1, dtype=np.int32)
    c_vals: list[float] = []
    for idx, (nty, val, cval) in enumerate(toks):
        nt[idx] = nty
        nv[idx] = val
        if nty == N_CONST:
            ci[idx] = len(c_vals)
            c_vals.append(cval)

    # --- assertions (spec §4): count + multiset against Operon's own view ------
    K = len(c_vals)
    cc = int(tree.CoefficientsCount)
    assert K == cc, f"K={K} != tree.CoefficientsCount={cc}"
    # Compare the multiset in fp64, BEFORE the float32 cast, so float rounding on
    # evolved trees cannot make a correct mapping look wrong.
    got = sorted(float(x) for x in tree.GetCoefficients())
    mine = sorted(c_vals)
    assert len(got) == len(mine) and np.allclose(got, mine, rtol=1e-6, atol=1e-9), (
        f"c_init multiset {mine} != GetCoefficients() {got}"
    )

    c_init = np.asarray(c_vals, dtype=np.float32)
    return nt, nv, ci, c_init


def write_operon_pop_bin(trees, hash2idx, X, y, out):
    """Convert raw Operon trees and write a pop.bin via ``pop_io.write_pop_bin``.

    ``trees`` is a list of Operon ``Tree`` objects (GP individuals). Each is run
    through ``convert(tree, hash2idx)`` *here*, inside a ``try/except
    UnsupportedNodeType`` so that a tree referencing a NodeType outside the
    restricted grammar — or a ``Variable`` whose hash is absent from ``hash2idx``
    (e.g. after a ``ChangeVariableMutation``) — is counted+logged+filtered rather
    than crashing the whole dump (spec §9 honesty requirement; matches the
    ``UnsupportedNodeType`` docstring). Then the spec §4 array filters apply,
    **counting + logging** every dropped tree (never silently dropped):

      * ``UnsupportedNodeType`` from ``convert`` -> ``dropped_unsupported``
      * ``K = len(c_init) > MAX_K`` (=32; structurally ~0%, still counted)
      * unsupported NType in the converted array: any ``nt`` outside ``{0,1,2,3}``
        -> ``dropped_bad_type``. ``convert`` already raises ``UnsupportedNodeType``
        on unmapped Operon NodeTypes, so under the restricted grammar this
        array-level count must stay 0; it is defence-in-depth.
      * non-finite ``c_init`` (NaN/Inf) -> ``dropped_nonfinite``

    Note ``convert``'s ``K == CoefficientsCount`` / multiset assertions are NOT
    caught: they are correctness invariants and an ``AssertionError`` signals a real
    conversion bug, which must surface loudly rather than be silently dropped.

    Returns the ``pop_io.write_pop_bin`` stats dict augmented with filter counts:
    ``n_in``, ``n_kept``, ``dropped_unsupported``, ``dropped_k_over``,
    ``dropped_bad_type``, ``dropped_nonfinite``. Writes only the surviving trees.
    """
    n_in = len(trees)
    kept = []
    dropped_unsupported = 0
    dropped_k_over = 0
    dropped_bad_type = 0
    dropped_nonfinite = 0

    valid_types = {N_VAR, N_CONST, N_UFUNC, N_BFUNC}

    for m, tree in enumerate(trees):
        try:
            nt, nv, ci, c_init = convert(tree, hash2idx)
        except UnsupportedNodeType as e:
            dropped_unsupported += 1
            log.warning("DROP tree %d: unsupported NodeType -> %s", m, e)
            continue
        nt = np.asarray(nt)
        c_init = np.asarray(c_init, dtype=np.float64)
        K = len(c_init)
        if K > MAX_K:
            dropped_k_over += 1
            log.warning("DROP tree %d: K=%d > MAX_K=%d", m, K, MAX_K)
            continue
        bad = {int(t) for t in np.unique(nt)} - valid_types
        if bad:
            dropped_bad_type += 1
            log.warning("DROP tree %d: unsupported NType(s) %s (must be empty)", m, sorted(bad))
            continue
        if not np.all(np.isfinite(c_init)):
            dropped_nonfinite += 1
            log.warning("DROP tree %d: non-finite c_init %s", m, c_init.tolist())
            continue
        kept.append((nt, nv, ci, np.asarray(c_init, dtype=np.float32)))

    n_dropped = dropped_unsupported + dropped_k_over + dropped_bad_type + dropped_nonfinite
    if n_dropped:
        log.warning(
            "write_operon_pop_bin(%s): kept %d/%d (dropped unsupported=%d k_over=%d "
            "bad_type=%d nonfinite=%d)",
            Path(out).name, len(kept), n_in, dropped_unsupported, dropped_k_over,
            dropped_bad_type, dropped_nonfinite,
        )

    stats = write_pop_bin(out, kept, X, y)
    stats.update(
        n_in=n_in,
        n_kept=len(kept),
        dropped_unsupported=dropped_unsupported,
        dropped_k_over=dropped_k_over,
        dropped_bad_type=dropped_bad_type,
        dropped_nonfinite=dropped_nonfinite,
    )
    return stats
