"""Tests for the venv .pth installer. Each names the mutation it would catch."""

from __future__ import annotations

import install_pth as mod


def _venv(tmp_path):
    sp = tmp_path / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    return tmp_path, sp / mod.PTH_NAME


def test_install_then_check_passes(tmp_path):
    """Mutation: write a different line than check expects -> check fails after install."""
    venv, pth = _venv(tmp_path)
    assert mod.main([str(venv)]) == 0
    assert pth.read_text() == mod.pth_line()
    assert mod.main(["--check", str(venv)]) == 0


def test_check_reports_a_missing_guard(tmp_path):
    """Mutation: --check exits 0 regardless -> an unguarded venv goes unnoticed."""
    venv, pth = _venv(tmp_path)
    assert mod.main(["--check", str(venv)]) == 1
    assert not pth.exists()


def test_the_line_imports_hostguard_from_this_directory():
    """Mutation: path or import dropped -> the .pth loads nothing."""
    line = mod.pth_line()
    assert line.startswith("import ")
    assert str(mod.HERE) in line
    assert "import hostguard" in line
