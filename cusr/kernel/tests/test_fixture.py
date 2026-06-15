"""test_fixture.py — 3-archetype 算法基线 fixture.

跑 ./batch_lm pop_fixture.bin, 读 status.bin + c_final.bin, 验:
1. 3 棵 archetype (powerlaw_K2 / quadratic_K3 / expsat_K3) 都 STATUS_CONVERGED (status==0)
2. c_final 跟 c_true 的相对误差 < 5e-3
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BATCH_LM = PROJ / "batch_lm"
FIXTURE = HERE / "pop_fixture.bin"
TRUTH = HERE / "pop_fixture_truth.json"


def run_batch_lm(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(BATCH_LM), str(FIXTURE), str(out_dir)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("batch_lm stdout:")
        print(r.stdout)
        print("batch_lm stderr:")
        print(r.stderr)
        raise RuntimeError(f"batch_lm exit={r.returncode}")
    print(r.stdout)
    status = np.fromfile(out_dir / "status.bin", dtype=np.int32)
    c_final = np.fromfile(out_dir / "c_final.bin", dtype=np.float32)
    return status, c_final


def main():
    if not BATCH_LM.exists():
        print(f"batch_lm not built. Run `make batch_lm` in {PROJ}.")
        return 1
    if not FIXTURE.exists():
        print(f"fixture not found. Run pop_fixture_gen.py first.")
        return 1

    truth = json.loads(TRUTH.read_text())
    fixtures = truth["fixtures"]

    out_dir = HERE / "_out"
    status, c_final = run_batch_lm(out_dir)

    print(f"\n== 算法基线 fixture 验证 ==")
    n_pass = 0
    n_total = len(fixtures)
    for fx in fixtures:
        name = fx["name"]
        K = fx["K"]
        c_off = fx["c_offset"]
        c_true = np.asarray(fx["c_true"], dtype=np.float32)
        c_got = c_final[c_off:c_off + K]
        denom = float(np.linalg.norm(c_true))
        c_rel = float(np.linalg.norm(c_got - c_true) / denom) if denom > 0 else float("nan")
        st = int(status[fixtures.index(fx)])

        ok = (st == 0) and np.isfinite(c_rel) and (c_rel < 5e-3)
        if ok: n_pass += 1
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name:16s} K={K} status={st} c_rel={c_rel:.3e}  "
              f"c_final={c_got}  c_true={c_true}")

    print(f"\nresult: {n_pass}/{n_total} PASS")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
