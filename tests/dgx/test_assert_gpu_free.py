"""Tests for the DGX GPU-occupancy gate.

Each test names the mutation it would catch. The default must stay exactly
today's rule -- one job at a time -- because raising it silently is how a box
whose CUDA pool IS system RAM gets taken down.
"""

from __future__ import annotations

import assert_gpu_free as mod
import pytest
from types import SimpleNamespace

MY_UID = 1000
OTHER_UID = 2000

#: The real function, captured before the autouse fixture below stubs it out --
#: the one test that exercises ``running_jobs`` itself needs the real one back.
_REAL_RUNNING_JOBS = mod.running_jobs
_REAL_COMPUTE_APPS = mod.compute_apps


@pytest.fixture(autouse=True)
def _quiet_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: a card with nothing on it and a uid we control."""
    monkeypatch.setattr(mod.os, "getuid", lambda: MY_UID)
    monkeypatch.setattr(mod, "running_jobs", lambda *_: [])
    monkeypatch.setattr(mod, "compute_apps", list)
    monkeypatch.setattr(mod, "owner_of", lambda _pid: MY_UID)


def run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Invoke the gate's CLI with ``argv``."""
    monkeypatch.setattr("sys.argv", ["assert_gpu_free.py", *argv])
    return mod.main()


def test_default_budget_is_one_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: defaulting --max-jobs above 1.

    The standing rule is one job at a time; the default must encode it.
    """
    monkeypatch.setattr(mod, "running_jobs", lambda *_: [(42, "python -m train")])
    assert run(monkeypatch) == 1


def test_a_raised_budget_permits_a_second_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: ignoring --max-jobs and always refusing an occupied card.

    Deliberate, measured sharing has to be expressible or it happens unguarded.
    """
    monkeypatch.setattr(mod, "running_jobs", lambda *_: [(42, "python -m train")])
    assert run(monkeypatch, "--max-jobs", "2") == 0
    assert run(monkeypatch, "--max-jobs", "1") == 1


def test_another_user_is_refused_whatever_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: folding other users into the same count as our own jobs.

    Sharing with yourself is a scheduling choice; sharing with a stranger is the
    case that wedged the sharing daemon for 9 h 44 m.
    """
    monkeypatch.setattr(mod, "compute_apps", lambda: [(99, "31518 MiB, python")])
    monkeypatch.setattr(mod, "owner_of", lambda _pid: OTHER_UID)
    assert run(monkeypatch, "--max-jobs", "8") == 1


def test_another_user_may_be_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: making --allow-other-users inert."""
    monkeypatch.setattr(mod, "compute_apps", lambda: [(99, "31518 MiB, python")])
    monkeypatch.setattr(mod, "owner_of", lambda _pid: OTHER_UID)
    assert run(monkeypatch, "--allow-other-users", "measured, agreed with them") == 0


def test_unknown_ownership_counts_as_someone_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: treating an unreadable owner as our own process.

    Guessing 'probably mine' is how a guard talks itself into sharing a card.
    """
    monkeypatch.setattr(mod, "compute_apps", lambda: [(99, "python")])
    monkeypatch.setattr(mod, "owner_of", lambda _pid: None)
    assert run(monkeypatch) == 1


def test_an_unmeasurable_device_refuses_launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unavailable device query must not permit another GPU allocation."""
    monkeypatch.setattr(mod, "compute_apps", lambda: None)
    assert run(monkeypatch) == 1
    assert "occupancy is unknown" in capsys.readouterr().err


@pytest.mark.parametrize("output", ["driver warning", "unknown, N/A, python"])
def test_malformed_gpu_output_does_not_mean_free(monkeypatch, output):
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=output))
    # The fixture replaces this function; exercise its actual implementation.
    assert _REAL_COMPUTE_APPS() is None


def test_a_nonsense_budget_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: accepting --max-jobs 0, which would permit nothing coherently."""
    assert run(monkeypatch, "--max-jobs", "0") == 2


def test_our_budget_counts_only_our_own_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Catches the ownership filter being dropped from ``running_jobs``.

    A SHARED box means another user runs the same module we do, and without
    this filter their job is counted against OUR budget -- measured on the
    Turing DGX on 2026-09-21, where a colleague's
    ``python3 -m autocast.scripts.train.encoder_processor_decoder`` made the
    gate refuse with "1 job(s) of ours already running" while we had none,
    which also makes ``--allow-other-users`` impossible to honour.
    """
    module = mod.DEFAULT_MODULES[0]
    entries = []
    for pid in (1111, 2222):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(
            b"\x00".join([b"/usr/bin/python3", b"-m", module.encode()]) + b"\x00"
        )
        entries.append(directory)
    monkeypatch.setattr(mod, "running_jobs", _REAL_RUNNING_JOBS)
    monkeypatch.setattr(mod, "ancestors", lambda _pid: set())
    monkeypatch.setattr(mod, "Path", lambda *_a, **_k: _FakeProc(entries))
    monkeypatch.setattr(
        mod, "owner_of", lambda pid: MY_UID if pid == 1111 else OTHER_UID
    )
    found = {pid for pid, _ in mod.running_jobs()}
    assert found == {1111}, (
        "running_jobs must count only OUR processes: the other user's job "
        "belongs to the 'another user is on the card' leg, not to our budget."
    )


class _FakeProc:
    """Stands in for ``Path("/proc")`` -- and for ``Path(argv[0])``."""

    def __init__(self, entries) -> None:
        self._entries = entries
        self.name = "python3"

    def iterdir(self):
        """The fake process directories, in order."""
        return iter(self._entries)


RESIDENT_APP = (7, "20934 MiB, /home/me/llama.cpp/build/bin/llama-server")


def test_a_declared_resident_of_ours_is_not_counted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: making --resident inert, or dropping its printed declaration.

    Without it a queued grid's wait for a free card never succeeds on a box
    where the owner's LLM server holds the GPU permanently.
    """
    monkeypatch.setattr(mod, "compute_apps", lambda: [RESIDENT_APP])
    assert run(monkeypatch) == 1
    assert run(monkeypatch, "--resident", "llama-server=owner's resident LLM") == 0
    assert "DECLARED RESIDENT" in capsys.readouterr().out


def test_a_resident_excuses_only_the_named_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: excusing every process once any resident is declared."""
    monkeypatch.setattr(
        mod, "compute_apps", lambda: [RESIDENT_APP, (8, "4000 MiB, python")]
    )
    assert run(monkeypatch, "--resident", "llama-server=owner's resident LLM") == 1


def test_a_resident_never_excuses_another_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: matching the name without checking the owner.

    A stranger's process of the same name is someone else on the card.
    """
    monkeypatch.setattr(mod, "compute_apps", lambda: [RESIDENT_APP])
    monkeypatch.setattr(mod, "owner_of", lambda _pid: OTHER_UID)
    assert run(monkeypatch, "--resident", "llama-server=owner's resident LLM") == 1


def test_a_resident_without_a_reason_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: accepting a bare name, which would be an undocumented hole."""
    assert run(monkeypatch, "--resident", "llama-server") == 2
    assert run(monkeypatch, "--resident", "llama-server=") == 2
