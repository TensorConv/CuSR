"""GPU clock read / lock / verify helpers (A100 experiment hygiene).

Why: every CuSR throughput number so far is "clocks unlocked, DRAFT". Before a
paper-grade sweep we pin the SM/graphics clock so timings are reproducible. The
actual lock needs `sudo nvidia-smi -lgc` (a side effect); the parse / select /
verify logic here is what the test suite covers. Mirrors the clock-read in
results/tier0_a100__20260616/sweep.py.

CLI:
    python -m cusr.benchmark.gpu_clocks --gpu 0            # show current + supported
    python -m cusr.benchmark.gpu_clocks --lock --gpu 0     # lock to max supported (or --mhz N)
    python -m cusr.benchmark.gpu_clocks --unlock --gpu 0
"""
from __future__ import annotations

import argparse
import subprocess


def _run(args, timeout: int = 15):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _parse_clock(stdout: str) -> int:
    """First non-empty line of an `--query-gpu=clocks.sm` reply, as int MHz."""
    return int(stdout.strip().splitlines()[0])


def _parse_supported(stdout: str) -> list[int]:
    """`--query-supported-clocks=gr` reply -> list of int MHz (order preserved)."""
    return [int(line) for line in stdout.strip().splitlines() if line.strip()]


def query_sm_clock(gpu: int = 0) -> int:
    """Current SM clock (MHz) of GPU `gpu`."""
    r = _run(["nvidia-smi", "-i", str(gpu), "--query-gpu=clocks.sm",
              "--format=csv,noheader,nounits"])
    return _parse_clock(r.stdout)


def supported_graphics_clocks(gpu: int = 0) -> list[int]:
    """Supported graphics clocks (MHz) reported by nvidia-smi (descending)."""
    r = _run(["nvidia-smi", "-i", str(gpu), "--query-supported-clocks=gr",
              "--format=csv,noheader,nounits"])
    return _parse_supported(r.stdout)


def lock_sm_clock(gpu: int, mhz: int) -> None:
    """Pin the SM/graphics clock to `mhz` (needs sudo). Release with unlock_sm_clock."""
    _run(["sudo", "nvidia-smi", "-i", str(gpu), "-lgc", str(mhz)])


def unlock_sm_clock(gpu: int) -> None:
    """Release a previously-locked clock (needs sudo)."""
    _run(["sudo", "nvidia-smi", "-i", str(gpu), "-rgc"])


def verify_locked(gpu: int, target_mhz: int, tol_mhz: int = 15) -> bool:
    """True iff the current SM clock is within `tol_mhz` of `target_mhz`."""
    return abs(query_sm_clock(gpu) - target_mhz) <= tol_mhz


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU clock read/lock/verify (A100 hygiene)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lock", action="store_true", help="lock the SM clock")
    ap.add_argument("--unlock", action="store_true", help="release the lock")
    ap.add_argument("--mhz", type=int, default=None,
                    help="clock to lock to (default: max supported)")
    a = ap.parse_args()

    if a.unlock:
        unlock_sm_clock(a.gpu)
        print(f"[gpu_clocks] GPU {a.gpu} unlocked; SM clock now {query_sm_clock(a.gpu)} MHz")
        return 0
    if a.lock:
        mhz = a.mhz if a.mhz is not None else max(supported_graphics_clocks(a.gpu))
        lock_sm_clock(a.gpu, mhz)
        ok = verify_locked(a.gpu, mhz)
        print(f"[gpu_clocks] GPU {a.gpu} lock -> {mhz} MHz; "
              f"now {query_sm_clock(a.gpu)} MHz; verified={ok}")
        return 0 if ok else 1
    print(f"[gpu_clocks] GPU {a.gpu} SM clock = {query_sm_clock(a.gpu)} MHz")
    print(f"[gpu_clocks] supported graphics clocks = {supported_graphics_clocks(a.gpu)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
