"""Tests for the unified-memory host guard.

Each test names the mutation it would catch. The failure being guarded: on a
unified-memory box (GB10) a GPU allocation larger than what the host can spare
does not fail, it makes the kernel evict everything else and the machine
thrashes until it is power-cycled (gpu-node-1, 2026-10-06 13:51: the cap was
total - 12 GiB = 109.7 GiB while ~86 GiB was available and a 21 GiB server
already held part of the pool).
"""

from __future__ import annotations

import types
from pathlib import Path

import hostguard as hg
import pytest

GIB = 2**30
TOTAL = int(121.7 * GIB)
LAUNCHER_MARKED = 1000  # oom_score_adj a launcher may already have set


class FakeBackend:
    def __init__(self, total: int):
        self._total = total
        self.fraction: float | None = None

    def is_available(self) -> bool:
        return True

    def get_device_properties(self, _idx: int):
        return types.SimpleNamespace(total_memory=self._total)

    def set_per_process_memory_fraction(self, fraction: float) -> None:
        self.fraction = fraction


def _meminfo(tmp_path: Path, total: int, available: int) -> Path:
    p = tmp_path / "meminfo"
    p.write_text(
        f"MemTotal:       {total // 1024} kB\n"
        f"MemFree:        {available // 4 // 1024} kB\n"
        f"MemAvailable:   {available // 1024} kB\n"
    )
    return p


def _oom(tmp_path: Path, value: int = 0) -> Path:
    p = tmp_path / "oom_score_adj"
    p.write_text(f"{value}\n")
    return p


# --- the cap ------------------------------------------------------------------


def test_cap_follows_what_is_available_not_the_pool_total():
    """Mutation: cap from device total (the 2026-10-06 defect) -> 109.7, not 73.7."""
    cap = hg.cap_bytes(TOTAL, int(85.7 * GIB), 12 * GIB)
    assert cap == pytest.approx(73.7 * GIB, rel=1e-6)


def test_cap_never_exceeds_total_minus_reserve():
    """Mutation: drop the total bound -> an over-reported MemAvailable lifts the cap."""
    cap = hg.cap_bytes(TOTAL, TOTAL + GIB, 12 * GIB)
    assert cap == TOTAL - 12 * GIB


def test_too_little_available_gives_the_floor_not_zero_or_negative():
    """Mutation: no floor -> a non-positive fraction, which torch rejects or ignores."""
    assert hg.cap_bytes(TOTAL, 5 * GIB, 12 * GIB) == hg.FLOOR_BYTES


# --- applying it --------------------------------------------------------------


def _torch(backend: FakeBackend):
    return types.SimpleNamespace(cuda=backend, xpu=None)


def test_apply_sets_the_available_based_fraction(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("AUTOCAST_HOST_MEMORY_RESERVE_GIB", raising=False)
    be = FakeBackend(TOTAL)
    hg.apply(_torch(be), _meminfo(tmp_path, TOTAL, int(85.7 * GIB)), _oom(tmp_path))
    assert be.fraction == pytest.approx(73.7 * GIB / TOTAL, rel=1e-6)


def test_apply_marks_the_process_as_the_preferred_oom_victim(tmp_path, monkeypatch):
    """Mutation: skip the mark -> under pressure the killer may pick the terminal."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    oom = _oom(tmp_path, 0)
    hg.apply(_torch(FakeBackend(TOTAL)), _meminfo(tmp_path, TOTAL, 80 * GIB), oom)
    assert int(oom.read_text()) == hg.OOM_SCORE_ADJ


def test_apply_never_lowers_an_existing_higher_oom_score(tmp_path, monkeypatch):
    """Mutation: unconditional write -> a job marked 1000 by its launcher is protected."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    oom = _oom(tmp_path, LAUNCHER_MARKED)
    hg.apply(_torch(FakeBackend(TOTAL)), _meminfo(tmp_path, TOTAL, 80 * GIB), oom)
    assert int(oom.read_text()) == LAUNCHER_MARKED


def test_discrete_gpu_is_left_alone(tmp_path, monkeypatch):
    """Mutation: drop the unified check -> a 24 GiB card gets a host-sized cap."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    be, oom = FakeBackend(24 * GIB), _oom(tmp_path, 0)
    hg.apply(_torch(be), _meminfo(tmp_path, TOTAL, 80 * GIB), oom)
    assert be.fraction is None
    assert int(oom.read_text()) == 0


def test_slurm_jobs_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    be = FakeBackend(TOTAL)
    hg.apply(_torch(be), _meminfo(tmp_path, TOTAL, 80 * GIB), _oom(tmp_path))
    assert be.fraction is None


# --- chaining the sitecustomize it displaces ----------------------------------


def test_chain_skips_hostguard_shims_and_runs_the_real_one(tmp_path, monkeypatch):
    """Mutation: no shim check -> the chain re-runs a shim and never reaches apport's."""
    shim, real, ran = tmp_path / "a", tmp_path / "b", tmp_path / "ran"
    shim.mkdir()
    real.mkdir()
    (shim / "sitecustomize.py").write_text(f"# {hg.SHIM_MARKER}\nraise SystemExit\n")
    (real / "sitecustomize.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    monkeypatch.setattr(hg.sys, "path", [str(shim), str(real)])
    hg.chain_shadowed_sitecustomize()
    assert ran.exists()


# --- the fixed per-job budget (shared boxes) ------------------------------------


def test_budget_lowers_the_cap_below_what_is_available():
    """Caught by: ignoring the budget -- the 2026-10-07 Turing shape, where a job
    that started with ~117 GiB free kept a cap far above its agreed share."""
    cap = hg.cap_bytes(TOTAL, int(117 * GIB), 12 * GIB, budget=60 * GIB)
    assert cap == 60 * GIB


def test_budget_never_raises_the_cap_above_what_is_available():
    """Caught by: max() instead of min(), which would hand a job more than the
    host can spare whenever its budget is generous."""
    cap = hg.cap_bytes(TOTAL, int(50 * GIB), 12 * GIB, budget=60 * GIB)
    assert cap == 38 * GIB


def test_no_budget_leaves_the_cap_unchanged():
    """Caught by: a default budget, which would silently shrink every job."""
    assert hg.cap_bytes(TOTAL, int(85.7 * GIB), 12 * GIB) == hg.cap_bytes(
        TOTAL, int(85.7 * GIB), 12 * GIB, budget=None
    )


@pytest.mark.parametrize("raw", ["", "0", "-5", "sixty"])
def test_unset_zero_negative_or_garbled_budget_means_none(monkeypatch, raw):
    """Caught by: float() raising on a typo (killing the import) or a 0 budget
    becoming a 0-byte cap."""
    monkeypatch.setenv("AUTOCAST_HOSTGUARD_BUDGET_GIB", raw)
    assert hg.read_budget_gib() is None


def test_apply_honours_the_budget_from_the_environment(tmp_path, monkeypatch):
    """Caught by: reading the budget but never passing it to the cap."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("AUTOCAST_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setenv("AUTOCAST_HOSTGUARD_BUDGET_GIB", "60")
    be = FakeBackend(TOTAL)
    hg.apply(_torch(be), _meminfo(tmp_path, TOTAL, int(117 * GIB)), _oom(tmp_path))
    assert be.fraction == pytest.approx(60 * GIB / TOTAL, rel=1e-6)
