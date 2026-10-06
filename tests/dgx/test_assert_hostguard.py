"""Tests for gate leg 4's venv check. Each names the mutation it would catch.

The failures guarded: a venv recreated without its hostguard.pth (every smoke test and
ad-hoc script on it then runs uncapped -- the 2026-10-06 13:51 hang), and a box whose
deployed guard is an older copy than preflight's (Turing runs an rsynced copy).
"""

from __future__ import annotations

import hashlib

import assert_hostguard as mod


def _guard(tmp_path, text="GUARD v2\n"):
    p = tmp_path / "hostguard.py"
    p.write_text(text)
    return p, hashlib.sha256(text.encode()).hexdigest()


def test_loaded_and_identical_passes(tmp_path):
    p, sha = _guard(tmp_path)
    ok, _ = mod.verdict(str(p), sha)
    assert ok


def test_not_loaded_at_startup_is_refused(tmp_path):
    """Mutation: treat a missing module as fine -> a venv without the .pth passes."""
    _, sha = _guard(tmp_path)
    ok, msg = mod.verdict(None, sha)
    assert not ok
    assert "install_pth.py" in msg


def test_a_different_copy_is_refused(tmp_path):
    """Mutation: skip the digest -> a stale deployed guard on the box passes."""
    p, _ = _guard(tmp_path, "GUARD v1\n")
    sha_v2 = hashlib.sha256(b"GUARD v2\n").hexdigest()
    ok, msg = mod.verdict(str(p), sha_v2)
    assert not ok
    assert "stale" in msg


def test_main_reads_the_module_this_interpreter_loaded(tmp_path, monkeypatch):
    """Mutation: main ignores sys.modules -> it cannot see what startup loaded."""
    p, sha = _guard(tmp_path)
    fake = type("M", (), {"__file__": str(p)})()
    monkeypatch.setitem(mod.sys.modules, "hostguard", fake)
    assert mod.main(["--expect-sha", sha]) == 0
    monkeypatch.delitem(mod.sys.modules, "hostguard")
    assert mod.main(["--expect-sha", sha]) == 1
