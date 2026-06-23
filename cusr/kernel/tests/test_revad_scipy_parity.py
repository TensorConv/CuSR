"""test_revad_scipy_parity.py — reverse-AD Jacobian vs scipy fp64 oracle.

GPU-gated regression for the reverse-AD kernel. Re-dumps the ACTUAL forward +
reverse AD gradients on the canonical fixture pop (via the prebuilt
_dump_jac_sample) and compares them, element by element, to an independent scipy
fp64 oracle — reusing classify() from experiments/revad_v5/compare_scipy.py.

Asserted (the ROBUST, unambiguous properties):
  - On AGREED elements (both AD modes finite), reverse-AD matches scipy 100%.
  - Reverse-AD is finite-safe: NO element where forward is finite but reverse is
    NaN (disp_fwd_fin_rev_nan == 0). Reverse being the finite one is the v5 win.

NOT asserted: exact scipy-match on DISPUTED (forward-NaN) singular points. There,
reverse-AD's 0-annihilating NaN-safety returns 0 even where scipy's finite
difference reports a nonzero slope (across a kink). That is a deliberate
speed/stability tradeoff; its end-to-end cost is measured by the loss-level gate
(test_parity_gate.py with BATCH_LM=batch_lm_revad), not at the gradient level here.

Run:
  uv run python -m pytest cusr/kernel/tests/test_revad_scipy_parity.py -q
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parent                       # cusr/kernel
ROOT = HERE.parents[2]                     # repo root
DUMP = KERNEL / "_dump_jac_sample"
POP = ROOT / "data" / "fixtures" / "pop.bin"
COMPARE = ROOT / "experiments" / "revad_v5" / "compare_scipy.py"


def _load_classify():
    spec = importlib.util.spec_from_file_location("compare_scipy", COMPARE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="no GPU (nvidia-smi absent)")
def test_revad_jacobian_matches_scipy(tmp_path):
    if not DUMP.exists():
        pytest.skip(f"{DUMP.name} not built (compile _dump_jac_sample.cu first)")
    if not POP.exists():
        pytest.skip(f"fixture pop missing: {POP}")

    sample = tmp_path / "jac.jsonl"
    r = subprocess.run([str(DUMP), str(POP)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"_dump_jac_sample failed: {r.stderr[:400]}"
    sample.write_text(r.stdout)

    res = _load_classify()(str(sample))

    # clean signal: where both AD modes are finite, reverse matches the fp64 oracle
    assert res["agreed_n"] > 0, "no agreed (both-finite) elements were dumped"
    assert res["agreed_rev_ok"] == res["agreed_n"], (
        f"reverse-AD disagrees with scipy on "
        f"{res['agreed_n'] - res['agreed_rev_ok']}/{res['agreed_n']} well-defined elements")
    # finite-safety: reverse must never be the one that NaNs where forward is finite
    assert res["disp_fwd_fin_rev_nan"] == 0, (
        f"reverse-AD introduced {res['disp_fwd_fin_rev_nan']} NaNs where forward was finite")
