"""Tests for the earlyoom settings installer. Each names the mutation it would catch."""

from __future__ import annotations

import re
import shlex

import install_earlyoom as mod


def test_install_then_check_is_clean(tmp_path):
    """Mutation: install writes elsewhere than check reads -> check never passes."""
    mod.install(tmp_path)
    assert mod.differing(tmp_path) == []
    assert mod.main(["--check", "--root", str(tmp_path)]) == 0


def test_a_hand_edit_in_etc_is_reported(tmp_path):
    """Mutation: check compares existence only -> an edited /etc file passes as tracked."""
    mod.install(tmp_path)
    edited = tmp_path / mod.TARGETS["earlyoom.default"]
    edited.write_text(edited.read_text().replace("-M 8388608,4194304", "-m 10"))
    assert mod.differing(tmp_path) == [str(edited)]
    assert mod.main(["--check", "--root", str(tmp_path)]) == 1


def test_a_missing_file_is_reported(tmp_path):
    """Mutation: skip absent targets -> a box with no drop-in passes."""
    assert len(mod.differing(tmp_path)) == len(mod.TARGETS)


def test_tracked_settings_ignore_swap_and_use_fixed_sizes():
    """Mutation: revert to the shipped thresholds -> the 13:51 failure mode returns."""
    text = (mod.HERE / "earlyoom.default").read_text()
    args = next(line for line in text.splitlines() if line.startswith("EARLYOOM_ARGS="))
    assert "-s 100,100" in args
    assert "-M 8388608,4194304" in args
    assert " -m " not in args
    words = shlex.split(args.split("=", 1)[1].strip('"'))
    prefer = re.compile(words[words.index("--prefer") + 1])
    # the kernel's 15-character process names, as earlyoom reads them (comm)
    assert all(prefer.search(name) for name in ("python3", "python3.12", "llama-server", "llama-bench", "llama-perplexit"))
    assert not any(prefer.search(name) for name in ("ollama", "claude", "sshd"))
