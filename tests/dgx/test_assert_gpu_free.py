"""Tests for the DGX GPU-occupancy gate.

Each test names the mutation it would catch. The default must stay exactly
today's rule -- one job at a time -- because raising it silently is how a box
whose CUDA pool IS system RAM gets taken down.
"""

from __future__ import annotations

import assert_gpu_free as mod
import pytest

MY_UID = 1000
OTHER_UID = 2000


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


def test_an_unmeasurable_device_warns_and_does_not_claim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: treating 'nvidia-smi did not answer' as 'the card is free'."""
    monkeypatch.setattr(mod, "compute_apps", lambda: None)
    assert run(monkeypatch) == 0
    assert "did NOT run" in capsys.readouterr().out


def test_a_nonsense_budget_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: accepting --max-jobs 0, which would permit nothing coherently."""
    assert run(monkeypatch, "--max-jobs", "0") == 2
