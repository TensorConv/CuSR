"""test_fixture_op.py — op 集 + multi-variable fixture.

跑 batch_lm 在 pop_fixture_op.bin (8 棵, 覆盖 TAN/SIN/COS/EXP/LOG/SQRT/...) ,
验每棵都收敛 + c_rel < 5e-3.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
# BATCH_LM=../batch_lm_ad 可指定变体二进制 (默认 baseline), 与 test_fixture_scale.py 一致
BATCH_LM = Path(os.environ.get("BATCH_LM", PROJ / "batch_lm"))
FIXTURE = HERE / "pop_fixture_op.bin"
TRUTH = HERE / "pop_fixture_op_truth.json"


def main():
    if not BATCH_LM.exists(): print("batch_lm not built"); return 1
    if not FIXTURE.exists(): print("fixture not built"); return 1

    truth = json.loads(TRUTH.read_text())
    fixtures = truth["fixtures"]

    out_dir = HERE / "_out_op"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(BATCH_LM), str(FIXTURE), str(out_dir)],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        return r.returncode

    status = np.fromfile(out_dir / "status.bin", dtype=np.int32)
    c_final = np.fromfile(out_dir / "c_final.bin", dtype=np.float32)

    print(f"\n== op 集 / multi-var fixture 验证 ==")
    n_pass = 0
    for i, fx in enumerate(fixtures):
        name = fx["name"]; K = fx["K"]; c_off = fx["c_offset"]
        c_true = np.asarray(fx["c_true"], dtype=np.float32)
        c_got = c_final[c_off:c_off+K]
        denom = float(np.linalg.norm(c_true))
        c_rel = float(np.linalg.norm(c_got - c_true) / denom) if denom > 0 else float("nan")
        st = int(status[i])
        ok = (st == 0) and np.isfinite(c_rel) and (c_rel < 5e-3)
        if ok: n_pass += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} K={K} status={st} c_rel={c_rel:.3e}  c_got={c_got}  c_true={c_true}")

    print(f"\nresult: {n_pass}/{len(fixtures)} PASS")
    return 0 if n_pass == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
