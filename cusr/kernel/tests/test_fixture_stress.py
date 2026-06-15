"""test_fixture_stress.py — 边界情况 stress fixture.

3 棵, 验:
- K=0 tree status==3 (K0_SKIP, 无常数可优化)
- healthy tree status==0 (CONVERGED)
- NaN tree status==2 (FAIL_NAN, 中间算出 Inf/NaN)
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
FIXTURE = HERE / "pop_fixture_stress.bin"
TRUTH = HERE / "pop_fixture_stress_truth.json"


def main():
    if not BATCH_LM.exists(): print("batch_lm not built"); return 1
    if not FIXTURE.exists(): print("fixture not built; run pop_fixture_stress_gen.py"); return 1

    truth = json.loads(TRUTH.read_text())
    out_dir = HERE / "_out_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(BATCH_LM), str(FIXTURE), str(out_dir)],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        return r.returncode

    status = np.fromfile(out_dir / "status.bin", dtype=np.int32)

    print(f"\n== 边界情况 stress fixture 验证 ==")
    n_pass = 0
    for i, fx in enumerate(truth["fixtures"]):
        name = fx["name"]
        expect = fx["expect_status"]
        got = int(status[i])
        ok = (got == expect)
        if ok: n_pass += 1
        meanings = {0: "CONV", 1: "MAXITER", 2: "FAIL_NAN", 3: "K0_SKIP", 4: "FAIL_CHOLESKY"}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:14s} expect={expect}({meanings[expect]}) got={got}({meanings.get(got,'?')})")

    print(f"\nresult: {n_pass}/{len(truth['fixtures'])} PASS")
    return 0 if n_pass == len(truth["fixtures"]) else 1


if __name__ == "__main__":
    sys.exit(main())
