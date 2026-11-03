"""sharing.py: the choice of a level from measured runs, and a level's pinned copies."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import gpu_jobs
import pytest
import sharing
import site_config


def result(level: int, slowest_s: float, *, code: int = 0, used_gb: float = 10.0) -> sharing.LevelResult:
    return sharing.LevelResult(level, (slowest_s,), (code,), 1, used_gb, 100.0, (0.0,))


TWO, THREE, FOUR = 2, 3, 4
SLEEP_S = 0.2
FAIL = 3
SOLO = [result(1, 100.0), result(1, 102.0)]  # t1 = 101 s; the two runs differ by 2%


def test_choose_level_takes_the_smallest_level_within_the_noise_of_the_best() -> None:
    noise = sharing.noise_of([100.0, 102.0])
    # S(2) = 2 x 101 / 110 = 1.84, S(3) = 3 x 101 / 165 = 1.84, S(4) = 4 x 101 / 200 = 2.02: 4 is the best, 2 and 3 are not within 2%
    assert sharing.choose_level([*SOLO, result(2, 110.0), result(3, 165.0), result(4, 200.0)], noise)[0] == FOUR
    # S(3) = 2.00 is 1% below the best S(4) = 2.02, inside the 2% noise: the smaller wins
    assert sharing.choose_level([*SOLO, result(2, 110.0), result(3, 151.5), result(4, 200.0)], noise)[0] == THREE
    # no gain beyond the noise: no sharing
    assert sharing.choose_level([*SOLO, result(2, 200.0)], noise)[0] == 1
    # the best level exited non-zero, and the next best used too much memory: both ignored
    levels = [*SOLO, result(2, 110.0), result(3, 100.0, code=1), result(4, 100.0, used_gb=90.0)]
    assert sharing.choose_level(levels, noise, fraction=0.5)[0] == TWO


def test_probe_pins_copies_to_disjoint_cores_and_records_a_failing_copy(tmp_path: Path) -> None:
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < TWO:
        pytest.skip("needs two cores")
    cores = cores[:TWO]
    code = (f"FAIL = {FAIL}; import os, sys, time; c = sorted(os.sched_getaffinity(0)); "
            f"open(sys.argv[1] + '/' + str(c[0]), 'w').write(repr(c)); time.sleep({SLEEP_S}); sys.exit(FAIL if c[0] == int(sys.argv[2]) else 0)")
    probe = sharing.SharingProbe([sys.executable, "-c", code, str(tmp_path), str(cores[1])], cores, sample_s=0.05)
    level = probe.run(2)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(str(c) for c in cores)
    assert all((tmp_path / str(c)).read_text() == repr([c]) for c in cores)
    assert level.exit_codes == (0, FAIL) and not level.clean
    assert level.cores_per_copy == 1 and all(wall >= SLEEP_S for wall in level.walls_s)


def test_the_record_run_stores_is_the_one_the_planner_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10", shards=12)
    node = site_config.Node("n", 20, 100_000, (gpu,), cores_per_gpu=12)
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(gpu_jobs, "load_site", lambda: SimpleNamespace(nodes=(node,), sharing_memory_fraction=None))
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    (tmp_path / "jobs").mkdir()
    leg = {"peak_gpu_mib": 1024, "peak_mem_bytes": 0, "ended_at": 1, "exit": 0, "signals": []}
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "n", "legs": [leg]}))  # an earlier clean run of the job
    (tmp_path / "jobs" / "9.json").write_text(json.dumps({"job_id": "9", "name": "n"}))  # this probe's own sidecar

    class Probe:
        def __init__(self, *_a: object) -> None: ...

        def run(self, level: int) -> sharing.LevelResult:
            return result(level, {1: 100.0, 2: 110.0}[level])  # two copies in 110 s: S = 1.84, beyond the noise

    monkeypatch.setattr(sharing, "SharingProbe", Probe)
    arguments = SimpleNamespace(command=["--", "probe"], levels="1,2")
    assert sharing.run(arguments) == 0
    record = gpu_jobs.measured_sharing("n")
    assert record is not None and record["peak_bytes"] == 1024 * 1024**2
    packing = gpu_jobs.plan_packing(gpu=gpu, node=node, record=record, peak_bytes=record["peak_bytes"], fraction=None,
                                    cpus_asked=None, explicit_level=None, margin=0.1)
    assert (packing.level, packing.shards, packing.cores) == (TWO, 6, 6)  # the keys the planner matches on all agree
    assert gpu_jobs.measured_peak_bytes("n") == 1024 * 1024**2  # the probe's own sidecar never sizes a real job
