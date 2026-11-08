"""Make `slurm/*.py` importable as plain modules from here (standalone drivers, not a package)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

SLURM: Path = Path(__file__).resolve().parents[2] / "slurm"
if str(SLURM) not in sys.path:
    sys.path.insert(0, str(SLURM))
# Every tool reads a site file on import: the tests read the fixture below, never the machine's.
os.environ["GPUJOBS_SITE"] = str(SLURM.parent / "tests" / "slurm" / "fixtures" / "site.toml")


@pytest.fixture
def make_cgroup(tmp_path: Path) -> Callable[..., Path]:
    """A fake job cgroup under tmp_path: 4.35 GB current, 0.6 GB page cache of which 0.05 GB shmem (3.80 GB held),
    no memory limit, pids 100 and 101."""

    def make(name: str = "job", *, current: int = 4_350_000_000, cache: int = 600_000_000, shmem: int = 50_000_000,
             limit: str = "max", pids: str = "100\n101\n") -> Path:
        cgroup = tmp_path / name
        cgroup.mkdir(parents=True)
        (cgroup / "memory.current").write_text(f"{current}\n")
        (cgroup / "memory.stat").write_text(f"anon 1\nfile {cache}\nshmem {shmem}\n")
        (cgroup / "memory.max").write_text(f"{limit}\n")
        (cgroup / "cgroup.procs").write_text(pids)
        return cgroup

    return make
