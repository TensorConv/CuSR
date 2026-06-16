"""test_operon_harvest.py — the Operon/CPU harvest DRIVER (operon_harvest.py).

Covers the four pieces the spec asks for, kept FAST (tiny pop/N, 2 gens):
  1. cell-dir naming parity with harvest.py (the ``operon_`` prefix + noise/cap/seed tags);
  2. ``enumerate_cells`` / dry-run cell count == |problems x noises x caps x seeds|;
  3. resume-skip logic (a cell whose ``manifest.json`` exists is skipped);
  4. a tiny end-to-end: 2 problems x 1 seed x 1 cap x gens "0,2" produces per-cell
     ``manifest.json`` + ``pop_gen{0000,0002}.bin`` that ``inspect`` reports PASS on.

The two-run determinism check is deliberately NOT a pytest test (too slow for the
suite) — it is asserted once via the standalone ``operon_harvest`` smoke instead.

Runs under the operon-venv python (pyoperon present); the e2e spawns real
``operon_dump`` subprocesses (which need pyoperon) using ``sys.executable``.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cusr.benchmark.workload import operon_harvest as oh
from cusr.benchmark.workload.harvest import ALL_PROBLEMS

from .conftest import INSPECT_BIN


def _cell_dirs(root):
    """Cell directories under ``root`` — directories only, so the sibling
    ``operon_harvest_log.txt`` (same ``operon_`` prefix) is never mistaken for a cell."""
    return sorted(p for p in root.glob("operon_*") if p.is_dir())


# ---- 1. cell-dir naming -----------------------------------------------------

def test_cell_dir_naming_and_operon_prefix(tmp_path):
    """Cell dir = harvest.py's scheme with an ``operon_`` prefix; '/' -> '_';
    noise 0.0 -> tag '0', noise>0 -> the literal float string."""
    d0 = oh.cell_dir(tmp_path, "feynman/I.18.12", 4000, 0.0, 32, 1)
    assert d0.name == "operon_feynman_I.18.12_pop4000_noise0_len32_seed1"
    assert d0.parent == tmp_path

    dn = oh.cell_dir(tmp_path, "nguyen/5", 4000, 0.01, 64, 2)
    assert dn.name == "operon_nguyen_5_pop4000_noise0.01_len64_seed2"

    # Parallels harvest.cell_dir exactly modulo the prefix + root.
    from cusr.benchmark.workload.harvest import cell_dir as evogp_cell_dir
    evo = evogp_cell_dir("feynman/I.18.12", 4000, 0.0, 32, 1).name
    assert d0.name == f"operon_{evo}"


# ---- 2. enumerate_cells / dry-run cell count --------------------------------

def test_enumerate_cells_count_and_order():
    cells = oh.enumerate_cells(["a", "b"], [0.0, 0.01], [32, 64], [0, 1, 2])
    assert len(cells) == 2 * 2 * 2 * 3
    assert cells[0] == ("a", 0.0, 32, 0)            # cartesian, problem-major
    assert len(set(cells)) == len(cells)            # no dups


def test_default_matrix_is_204_cells():
    """Defaults == harvest.py parity: 17 problems x 2 noises x 2 caps x 3 seeds."""
    cells = oh.enumerate_cells(ALL_PROBLEMS, [0.0, 0.01], [32, 64], [0, 1, 2])
    assert len(ALL_PROBLEMS) == 17
    assert len(cells) == 204


def test_dry_run_prints_matrix_count(tmp_path, capsys):
    """--dry-run prints the cell-count line and does NO work (no cell dirs)."""
    rc = oh.main([
        "--dry-run", "--out-root", str(tmp_path),
        "--problems", "nguyen/1,nguyen/2", "--noises", "0.0",
        "--caps", "16", "--seeds", "0", "--gens", "0,2",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # 2 problems x 1 noise x 1 cap x 1 seed = 2 cells
    assert "= 2 cells" in out
    assert "snapshots/cell" in out and "est storage" in out
    # dry-run must not create any cell directory
    assert not _cell_dirs(tmp_path)


def test_dry_run_thread_budget_disclosed(tmp_path, capsys):
    rc = oh.main(["--dry-run", "--out-root", str(tmp_path),
                  "--problems", "nguyen/1", "--noises", "0.0", "--caps", "16",
                  "--seeds", "0", "--procs", "4", "--threads", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "concurrency: procs=4 x threads=1 = 4 threads" in out
    assert f"budget {oh.MAX_THREADS}" in out


# ---- 3. resume-skip logic ---------------------------------------------------

def test_resume_skip_counts_existing_manifest(tmp_path):
    """A cell whose manifest.json exists is counted done; run_cell returns 'skip'
    without invoking any subprocess."""
    ds, noise, cap, seed = "nguyen/1", 0.0, 16, 0
    d = oh.cell_dir(tmp_path, ds, 4000, noise, cap, seed)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")

    class Args:
        out_root = tmp_path
        pop = 4000
        N = 1000
        gens = "0,2"
        threads = 1

    import threading
    logf = tmp_path / oh.LOG_NAME
    status = oh.run_cell((ds, noise, cap, seed), Args(), logf,
                         threading.Lock(), sys.executable)
    assert status == "skip"
    # No new files beyond the manifest we planted (no .bin written on skip).
    assert sorted(p.name for p in d.iterdir()) == ["manifest.json"]


def test_thread_budget_guard_refuses_oversubscription(tmp_path):
    """procs*threads > MAX_THREADS aborts (honesty §9 — never oversubscribe)."""
    with pytest.raises(SystemExit):
        oh.main(["--out-root", str(tmp_path), "--problems", "nguyen/1",
                 "--noises", "0.0", "--caps", "16", "--seeds", "0", "--gens", "0,2",
                 "--procs", str(oh.MAX_THREADS + 1), "--threads", "1"])


# ---- 4. tiny end-to-end: manifests + inspect PASS ---------------------------

@pytest.mark.skipif(not INSPECT_BIN.exists(),
                    reason=f"inspect not built: run `make -C cusr/kernel inspect` ({INSPECT_BIN})")
def test_end_to_end_two_cells_inspect_pass(tmp_path):
    """2 problems x 1 seed x 1 cap x gens '0,2' (tiny pop/N) -> 2 cells, each with a
    manifest.json + pop_gen0000/0002.bin that `inspect` reports PASS on."""
    rc = oh.main([
        "--out-root", str(tmp_path),
        "--problems", "nguyen/1,nguyen/2", "--noises", "0.0",
        "--caps", "16", "--seeds", "0", "--gens", "0,2",
        "--pop", "24", "--N", "32", "--procs", "2", "--threads", "1",
    ])
    assert rc == 0, "driver returned nonzero (a cell failed) — see operon_harvest_log.txt"

    cell_dirs = _cell_dirs(tmp_path)
    assert len(cell_dirs) == 2, f"expected 2 cell dirs, got {[d.name for d in cell_dirs]}"

    # Honesty (spec §9): the driver must surface corpus-wide drop totals incl. K-over.
    log_txt = (tmp_path / oh.LOG_NAME).read_text()
    assert "drop totals over 2 cells" in log_txt
    assert "K-over=" in log_txt, "driver must log the corpus-wide K-over count (anti-cheat)"

    for d in cell_dirs:
        manifest = d / "manifest.json"
        assert manifest.exists(), f"missing manifest in {d.name}"
        m = json.loads(manifest.read_text())
        # manifest schema sanity (engine identity + the honesty fields)
        assert m["engine"] == "operon" and m["inline_co"] is False
        assert m["threads"] == 1
        assert m["checkpoint_gens"] == [0, 2]
        assert m["pop"] == 24 and m["N"] == 32
        assert len(m["snapshots"]) == 2
        assert m["operon_config"]["threads"] == 1
        # top-level schema parity with the evogp manifest (checkpoint_every present)
        assert m["checkpoint_every"] == 0, "missing top-level checkpoint_every (evogp schema parity)"
        # every evolution-affecting param is recorded in operon_config so the population is
        # reproducible from (manifest + operon_dump.py @ git_sha) — guard against silent drift.
        oc = m["operon_config"]
        for key in ("tree_init_len_min", "tree_init_len_max", "tree_max_depth",
                    "tree_min_depth", "crossover_max_depth", "mutation_max_depth",
                    "max_evaluations", "budget", "mutations"):
            assert key in oc, f"operon_config missing reproducibility field {key}"
        assert [mm["name"] for mm in oc["mutations"]] == [
            "NormalOnePointMutation", "ChangeVariableMutation",
            "ChangeFunctionMutation", "ReplaceSubtreeMutation"]
        # git_dirty flag recorded (honesty: an uncommitted runner edit must be visible)
        assert "git_dirty" in m, "manifest must record the git_dirty flag"
        # every snapshot records the adapter drop counts (anti-cheat §9) AND the evogp-
        # compatible aliases (n_kover == dropped_k_over, n_tfunc_skip) so shared consumers
        # read both corpora by the same keys (spec: SAME schema as the evogp manifest).
        for snap in m["snapshots"]:
            for key in ("n_in", "n_kept", "dropped_unsupported", "dropped_k_over",
                        "dropped_bad_type", "dropped_nonfinite"):
                assert key in snap, f"snapshot missing drop field {key}"
            assert snap["n_kover"] == snap["dropped_k_over"], "n_kover alias must equal dropped_k_over"
            assert snap["n_tfunc_skip"] == 0, "operon never produces TFUNC; n_tfunc_skip must be 0"

        for g in (0, 2):
            binf = d / f"pop_gen{g:04d}.bin"
            assert binf.exists(), f"missing {binf.name} in {d.name}"
            res = subprocess.run([str(INSPECT_BIN), str(binf)],
                                 capture_output=True, text=True)
            assert res.returncode == 0, f"inspect rc={res.returncode} on {binf.name}: {res.stderr}"
            assert "PASS" in res.stdout, f"inspect did not PASS on {binf.name}: {res.stdout}"


@pytest.mark.skipif(not INSPECT_BIN.exists(),
                    reason="inspect not built")
def test_end_to_end_resume_second_call_skips(tmp_path):
    """A second driver invocation over the same out-root skips both cells (resume),
    leaving the already-written manifests untouched."""
    common = ["--out-root", str(tmp_path), "--problems", "nguyen/1,nguyen/2",
              "--noises", "0.0", "--caps", "16", "--seeds", "0", "--gens", "0,2",
              "--pop", "24", "--N", "32", "--procs", "2", "--threads", "1"]
    assert oh.main(common) == 0
    mtimes = {d.name: (d / "manifest.json").stat().st_mtime_ns
              for d in _cell_dirs(tmp_path)}
    assert len(mtimes) == 2

    # second run: every cell is skipped, manifests unchanged (idempotent resume).
    assert oh.main(common) == 0
    for d in _cell_dirs(tmp_path):
        assert (d / "manifest.json").stat().st_mtime_ns == mtimes[d.name], \
            f"{d.name} manifest was rewritten on resume (should have been skipped)"
