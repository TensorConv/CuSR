"""conftest.py — workload test fixtures + path setup.

Must run under the operon-venv python (pyoperon 0.6.1 + numpy + sympy + pytest).
Ensures the repo root is importable so ``import cusr.benchmark.workload...``
resolves regardless of pytest's invocation cwd, and exposes a couple of shared
constants.
"""
from __future__ import annotations

import sys
from pathlib import Path

# repo root = .../CuSR (tests/ -> workload -> benchmark -> cusr -> root)
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# C inspector built via `make -C cusr/kernel inspect`
INSPECT_BIN = REPO_ROOT / "cusr" / "kernel" / "inspect"
DUMP_EVOGP_SRC = REPO_ROOT / "cusr" / "kernel" / "dump_evogp.py"
