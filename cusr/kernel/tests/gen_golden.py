#!/usr/bin/env python3
"""gen_golden.py — regenerate the parity golden references for P1 in-process CO.

Goldens are the byte-for-byte parity reference for every later step (single-call,
parity-under-reuse, status-branch span, end-to-end). They are PER-VARIANT (FD/AD)
and per-pop. This script runs each standalone once at the pinned MAX_ITER and
writes its output blobs under tests/golden/<variant>/<pop>/, plus a manifest.json
with sha256 + the pop header stats for provenance.

Determinism (run-to-run byte-parity) is asserted separately by
test_inproc_parity.py::test_standalone_determinism. Run that FIRST.

Usage (GPU pinned, device idle):
    source scripts/env.sh
    cd cusr/kernel && make batch_lm batch_lm_ad
    export CUDA_VISIBLE_DEVICES=2
    python cusr/kernel/tests/gen_golden.py
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the single source of truth for variants/pops/MAX_ITER.
from test_inproc_parity import GOLDEN, MAX_ITER, POPS, VARIANTS  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_header(pop: Path) -> dict:
    # pop_format.h PopHeader: 16 x int32 (64 B). magic=0x4D4C344D, version=1,
    # then M_prob, total_nodes, total_c, N, n_vars, K_max, max_stack, ...
    with open(pop, "rb") as f:
        raw = f.read(64)
    fields = struct.unpack("<16i", raw)
    return {
        "magic": hex(fields[0] & 0xFFFFFFFF),
        "version": fields[1],
        "M_prob": fields[2],
        "total_nodes": fields[3],
        "total_c": fields[4],
        "N": fields[5],
        "n_vars": fields[6],
        "K_max": fields[7],
        "max_stack": fields[8],
    }


def run_standalone(binary: Path, pop: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(binary), str(pop), str(out_dir), "--quiet", "--max-iter", str(MAX_ITER)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError(f"{binary.name} exit={r.returncode}")


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "max_iter": MAX_ITER,
        "nvcc_flags": "-O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math",
        "note": "byte-parity reference for P1 in-process CO (docs/kernel/INPROCESS_CO_PLAN.md)",
        "variants": {},
    }
    for variant, spec in VARIANTS.items():
        binary = spec["binary"]
        if not binary.exists():
            raise SystemExit(f"{binary} not built — run `make batch_lm batch_lm_ad`")
        manifest["variants"][variant] = {}
        for pop_name, pop_path in POPS.items():
            gdir = GOLDEN / variant / pop_name
            run_standalone(binary, pop_path, gdir)
            # keep ONLY the blobs this variant defines; drop any extras for tidiness
            keep = set(spec["outputs"])
            for f in gdir.iterdir():
                if f.is_file() and f.name not in keep:
                    f.unlink()
            entry = {
                "pop": str(pop_path.relative_to(GOLDEN.parents[3])),
                "header": read_header(pop_path),
                "blobs": {n: {"bytes": (gdir / n).stat().st_size,
                              "sha256": sha256(gdir / n)} for n in spec["outputs"]},
            }
            manifest["variants"][variant][pop_name] = entry
            print(f"[golden] {variant}/{pop_name}: "
                  + ", ".join(f"{n}={entry['blobs'][n]['sha256'][:12]}" for n in spec["outputs"]))

    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[golden] wrote manifest.json ({len(POPS)} pops x {len(VARIANTS)} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
