"""co_inproc.py — Python ctypes wrapper for the in-process CO drop-in (P1).

Loads libcusr_co_{fd,ad}.so, calls co_init ONCE (CUDA primary-context init paid
once per process), and exposes .optimize(pop_dict, max_iter) returning the same
blobs the standalone writes. See docs/kernel/INPROCESS_CO_PLAN.md.

MUST-FIX #1: a wrapper instance is pinned to ONE variant ('fd' or 'ad'); the
caller compares its output only against the SAME-variant golden.

MUST-FIX #5: the pop dict (popio.build_pop / load_pop_bin) has key 'M' and NO
magic/version — this wrapper synthesises the full 64-B PopHeader (magic
0x4D4C344D, version 1, M_prob<-pop['M'], ...) so a naive pass-through can't
KeyError or under-size the struct -> OOB device reads.

All input arrays are coerced to C-contiguous, exact dtype (int32 / float32) and
held on `self`/locals so the GC can't free them under ctypes (the .so reads the
raw pointers synchronously inside co_optimize, so locals living to the return is
sufficient; we keep them in scope explicitly).
"""
from __future__ import annotations

import atexit
import ctypes
import threading
from pathlib import Path

import numpy as np

POP_MAGIC = 0x4D4C344D
POP_VERSION = 1

# co_lib.h return codes
CO_OK = 0
CO_ERR_BOUNDS = 2
CO_ERR_CUDA = -1
CO_ERR_SUM_C = -2
CO_ERR_SUM_NODES = -3
CO_ERR_NULLARG = -4
CO_ERR_OOM = -5

_ERR_MSG = {
    CO_ERR_BOUNDS: "K_max > MAX_K(32) or max_stack > MAX_STACK(64) (caller must pre-filter)",
    CO_ERR_CUDA: "CUDA error inside co_optimize",
    CO_ERR_SUM_C: "sum(metas.K) != header.total_c (pop inconsistent)",
    CO_ERR_SUM_NODES: "sum(metas.n_nodes) != header.total_nodes (pop inconsistent)",
    CO_ERR_NULLARG: "null pointer / bad argument",
    CO_ERR_OOM: "out of memory growing a device buffer",
}


class COError(RuntimeError):
    """Raised when co_init / co_optimize returns a non-OK code."""

    def __init__(self, code: int, where: str):
        self.code = code
        msg = _ERR_MSG.get(code, f"unknown code {code}")
        super().__init__(f"{where}: {msg} (code={code})")


class PopHeader(ctypes.Structure):
    """Mirror of pop_format.h PopHeader — 16 x int32 = 64 B (incl. reserved[7])."""
    _fields_ = [
        ("magic", ctypes.c_int32),
        ("version", ctypes.c_int32),
        ("M_prob", ctypes.c_int32),
        ("total_nodes", ctypes.c_int32),
        ("total_c", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("n_vars", ctypes.c_int32),
        ("K_max", ctypes.c_int32),
        ("max_stack", ctypes.c_int32),
        ("reserved", ctypes.c_int32 * 7),
    ]


class TreeMeta(ctypes.Structure):
    """Mirror of pop_format.h TreeMeta — 4 x int32 = 16 B."""
    _fields_ = [
        ("node_offset", ctypes.c_int32),
        ("n_nodes", ctypes.c_int32),
        ("c_offset", ctypes.c_int32),
        ("K", ctypes.c_int32),
    ]


assert ctypes.sizeof(PopHeader) == 64, ctypes.sizeof(PopHeader)
assert ctypes.sizeof(TreeMeta) == 16, ctypes.sizeof(TreeMeta)

_F32 = ctypes.POINTER(ctypes.c_float)
_I32 = ctypes.POINTER(ctypes.c_int32)


def _c_ptr(arr, dtype):
    """C-contiguous, exact-dtype view -> (ctypes ptr, the backing ndarray).

    The ndarray must be kept alive by the caller until the .so returns."""
    a = np.ascontiguousarray(arr, dtype=dtype)
    return a.ctypes.data_as(ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))), a


class InProcessCO:
    """One persistent CO context over one variant's .so. Call .teardown() (or use
    as a context manager) to free; CUDA primary-context init is paid in __init__."""

    def __init__(self, device_id: int, lib_path: str, variant: str = "ad"):
        self.variant = variant
        self.device_id = int(device_id)
        self._lib = ctypes.CDLL(str(Path(lib_path).resolve()))
        self._handle = ctypes.c_void_p()
        self._torn = False

        # signatures
        self._lib.co_init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self._lib.co_init.restype = ctypes.c_int
        self._lib.co_optimize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PopHeader),
            _I32, _F32, _I32,                 # nt, nv, ci
            ctypes.POINTER(TreeMeta),          # metas
            _F32, _F32, _F32,                  # c_init, xs, ym
            ctypes.c_int,                      # max_iter
            _F32, _I32,                        # c_final_out, status_out
            _F32, _F32,                        # loss_init_out, loss_final_out
        ]
        self._lib.co_optimize.restype = ctypes.c_int
        self._lib.co_teardown.argtypes = [ctypes.c_void_p]
        self._lib.co_teardown.restype = None
        # co_last_split (-DPROFILE builds only; present-but-sentinel otherwise).
        # Bound unconditionally — the symbol always exists (returns 1 + -1.0 in the
        # non-prof .so). Used by the amortization suite for the setup/loop split.
        if hasattr(self._lib, "co_last_split"):
            self._lib.co_last_split.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ]
            self._lib.co_last_split.restype = ctypes.c_int

        rc = self._lib.co_init(self.device_id, ctypes.byref(self._handle))
        if rc != CO_OK:
            raise COError(rc, "co_init")

    # -- context manager sugar --
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.teardown()
        return False

    def _build_header(self, pop: dict) -> PopHeader:
        """MUST-FIX #5: synthesise the full header from a pop dict (key 'M', no
        magic/version). Field-for-field, no pass-through."""
        h = PopHeader()
        h.magic = POP_MAGIC
        h.version = POP_VERSION
        h.M_prob = int(pop["M"])
        h.total_nodes = int(pop["total_nodes"])
        h.total_c = int(pop["total_c"])
        h.N = int(pop["N"])
        h.n_vars = int(pop["n_vars"])
        h.K_max = int(pop["K_max"])
        h.max_stack = int(pop["max_stack"])
        for i in range(7):
            h.reserved[i] = 0
        return h

    def optimize(self, pop: dict, max_iter: int = 50):
        """Run CO on one pop dict; return {'c_final','status'[,'loss_init',
        'loss_final']} as np arrays sliced to current M/total_c. Raises COError on
        a non-OK return code."""
        if self._torn:
            raise RuntimeError("InProcessCO used after teardown()")
        M = int(pop["M"])
        total_c = int(pop["total_c"])

        hdr = self._build_header(pop)

        # inputs: keep every ndarray alive until co_optimize returns.
        nt_p, _nt = _c_ptr(pop["nt"], np.int32)
        nv_p, _nv = _c_ptr(pop["nv"], np.float32)
        ci_p, _ci = _c_ptr(pop["ci"], np.int32)
        c_init_p, _ci0 = _c_ptr(pop["c_init"], np.float32)
        xs_p, _xs = _c_ptr(pop["xs"], np.float32)
        ym_p, _ym = _c_ptr(pop["ym"], np.float32)

        # metas: pop['metas'] is int32 (M,4) -> TreeMeta[M]
        metas_np = np.ascontiguousarray(pop["metas"], dtype=np.int32).reshape(M, 4)
        metas_buf = (TreeMeta * M)()
        ctypes.memmove(metas_buf, metas_np.ctypes.data, M * 16)

        # caller-allocated outputs
        c_final = np.zeros(total_c, dtype=np.float32)
        status = np.zeros(M, dtype=np.int32)
        loss_init = np.zeros(M, dtype=np.float32)
        loss_final = np.zeros(M, dtype=np.float32)

        rc = self._lib.co_optimize(
            self._handle, ctypes.byref(hdr),
            nt_p, nv_p, ci_p, metas_buf,
            c_init_p, xs_p, ym_p,
            int(max_iter),
            c_final.ctypes.data_as(_F32),
            status.ctypes.data_as(_I32),
            loss_init.ctypes.data_as(_F32),
            loss_final.ctypes.data_as(_F32),
        )
        # keep references alive across the call (defensive; locals already do)
        _keep = (_nt, _nv, _ci, _ci0, _xs, _ym, metas_np, metas_buf)
        del _keep
        if rc != CO_OK:
            raise COError(rc, "co_optimize")

        out = {"c_final": c_final, "status": status}
        if self.variant == "ad":
            out["loss_init"] = loss_init
            out["loss_final"] = loss_final
        return out

    def last_split(self):
        """(setup_ms, loop_ms) of the LAST optimize() call, or (None, None) if the
        loaded .so was not built with -DPROFILE. The amortization suite uses the
        *_prof.so to read this split; the headline wall is timed in Python around
        the non-prof optimize()."""
        if not hasattr(self._lib, "co_last_split"):
            return (None, None)
        s = ctypes.c_double(-1.0)
        l = ctypes.c_double(-1.0)
        rc = self._lib.co_last_split(ctypes.byref(s), ctypes.byref(l))
        if rc != 0:                      # non-prof build -> no data
            return (None, None)
        return (s.value, l.value)

    def teardown(self):
        if self._torn:
            return
        if self._handle:
            self._lib.co_teardown(self._handle)
            self._handle = ctypes.c_void_p()
        self._torn = True

    def __del__(self):
        try:
            self.teardown()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# get_inproc_co — process-wide singleton (the P1 payoff: CUDA primary-context
# init paid ONCE across all generations). Keyed by (device_id, variant) so a
# second variant on the same device does not return the wrong cached handle.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SINGLETONS: dict[tuple[int, str], "InProcessCO"] = {}
_SINGLETON_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _so_path(variant: str) -> Path:
    """libcusr_co_{fd,ad}.so next to this module (the build target dir)."""
    return _HERE / f"libcusr_co_{variant}.so"


def _teardown_all() -> None:
    for co in list(_SINGLETONS.values()):
        try:
            co.teardown()
        except Exception:  # noqa: BLE001
            pass
    _SINGLETONS.clear()


def get_inproc_co(device_id: int = 0, variant: str = "fd") -> "InProcessCO":
    """Process-wide singleton InProcessCO for (device_id, variant).

    The CUDA primary context is created on the FIRST call (per key) and reused
    across every later .optimize — that is the one-time-setup amortization P1 buys.
    `atexit` tears every singleton down so the interpreter exit is clean (a .so must
    never exit() mid-evolution; see co_lib.cu). Distinct variants/devices get
    distinct contexts (never alias)."""
    global _ATEXIT_REGISTERED
    key = (int(device_id), str(variant))
    with _SINGLETON_LOCK:
        co = _SINGLETONS.get(key)
        if co is None or co._torn:
            so = _so_path(variant)
            if not so.exists():
                raise FileNotFoundError(
                    f"{so.name} not built — run `make libcusr_co_{variant}.so` in "
                    f"{_HERE}"
                )
            co = InProcessCO(device_id=device_id, lib_path=str(so), variant=variant)
            _SINGLETONS[key] = co
            if not _ATEXIT_REGISTERED:
                atexit.register(_teardown_all)
                _ATEXIT_REGISTERED = True
        return co
