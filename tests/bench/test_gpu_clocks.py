"""Unit tests for cusr.benchmark.gpu_clocks — GPU clock read/lock/verify helpers.

No GPU needed: subprocess.run is monkeypatched with canned nvidia-smi output.
The actual `-lgc` lock is a sudo side effect exercised on the box, not here — we
only test the parse / select / verify / command-build logic.
"""
from __future__ import annotations

import pytest

from cusr.benchmark import gpu_clocks as gc


class _FakeProc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_parse_clock_takes_first_line_as_int():
    assert gc._parse_clock("1410\n") == 1410
    assert gc._parse_clock("1320\n1320\n") == 1320  # one value per GPU; take first


def test_parse_supported_clocks_returns_int_list():
    assert gc._parse_supported("1410\n1395\n1380\n1005\n") == [1410, 1395, 1380, 1005]
    assert gc._parse_supported("") == []


def test_query_sm_clock_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gc.subprocess, "run", lambda *a, **k: _FakeProc(stdout="1275\n"))
    assert gc.query_sm_clock(0) == 1275


def test_verify_locked_true_within_tolerance(monkeypatch):
    monkeypatch.setattr(gc.subprocess, "run", lambda *a, **k: _FakeProc(stdout="1407\n"))
    assert gc.verify_locked(0, 1410, tol_mhz=15) is True


def test_verify_locked_false_when_far(monkeypatch):
    monkeypatch.setattr(gc.subprocess, "run", lambda *a, **k: _FakeProc(stdout="1200\n"))
    assert gc.verify_locked(0, 1410, tol_mhz=15) is False


def test_lock_sm_clock_builds_lgc_command(monkeypatch):
    calls = {}
    def _capture(cmd, *a, **k):
        calls["cmd"] = cmd
        return _FakeProc()
    monkeypatch.setattr(gc.subprocess, "run", _capture)
    gc.lock_sm_clock(3, 1410)
    cmd = [str(x) for x in calls["cmd"]]
    assert "nvidia-smi" in cmd and "-lgc" in cmd and "1410" in cmd and "3" in cmd


def test_unlock_sm_clock_builds_rgc_command(monkeypatch):
    calls = {}
    def _capture(cmd, *a, **k):
        calls["cmd"] = cmd
        return _FakeProc()
    monkeypatch.setattr(gc.subprocess, "run", _capture)
    gc.unlock_sm_clock(3)
    cmd = [str(x) for x in calls["cmd"]]
    assert "-rgc" in cmd and "3" in cmd
