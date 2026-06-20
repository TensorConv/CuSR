"""Robust symbolic-recovery judge with a SIGKILL hard timeout.

Kept in its OWN module — importing NOTHING heavy (no evogp / torch / pipeline) —
so the `spawn` subprocess that resolves the target function does NOT re-import
the EvoGP harness and therefore does NOT init a CUDA context just to run sympy.

Why not `judge.recovery_columns_timeout`: it uses `terminate()` (SIGTERM) then a
no-timeout `join()`. When `sympy.simplify` is stuck in C-level code that ignores
SIGTERM (observed: cpu_every on a damped multi-inner best tree hung ~20 min), the
`join()` blocks forever. SIGKILL cannot be ignored.

(The numeric-proxy fallback inside `judge.recovery_columns` DOES import evogp
lazily, but only on sympy-opaque cases — rare, and bounded by the timeout here.)
"""
from __future__ import annotations

import multiprocessing as _mp
import os as _os
import signal as _signal

_REC_FALSE = {"strict": False, "lenient": False, "srbench": False}


def _child(q, candidate, true, X):
    try:
        from cusr.demonstrator import judge as _j
        q.put(_j.recovery_columns(candidate, true, X=X))
    except Exception as e:  # noqa: BLE001
        q.put({**_REC_FALSE, "_err": str(e)})


def recovery_hard(candidate, true, X, *, timeout: float = 15.0) -> dict:
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, str(candidate), str(true), X), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            _os.kill(p.pid, _signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        p.join(5)
        return {**_REC_FALSE, "_timeout": True}
    try:
        return q.get(timeout=2)
    except Exception:  # noqa: BLE001
        return {**_REC_FALSE, "_err": "no result"}
