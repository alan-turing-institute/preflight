"""stop_on_low_free: stop only after a LASTING low, only while the job's process runs; evict_from_cache walks safely."""

from __future__ import annotations

import os

import evict_from_cache
import stop_on_low_free as mod

GIB = mod.KIB_PER_GIB
#: Separate dips below the floor in the gpu-node-1 trace below.
SPARK_DIPS = 2


def run(free_gib, *, running=True, active_for=None, floor_gib=4.0, hold_s=10.0,
        driver_error_at=None, window_s=2400.0):
    """Drive watch() one simulated second per sample of ``free_gib``; return (exit, stops, log)."""
    clock = {"t": 0.0}
    samples = iter(free_gib)
    stops, log = [], []

    def active():
        return active_for is None or clock["t"] < active_for

    def free():
        try:
            return int(next(samples) * GIB)
        except StopIteration:
            clock["t"] = 1e9  # out of samples: end the unit
            return int(100 * GIB)

    def driver_error():
        return clock["t"] >= driver_error_at

    code = mod.watch(free, active, lambda: running, lambda: stops.append(clock["t"]),
                     floor_kib=int(floor_gib * GIB), hold_s=hold_s, interval=1.0,
                     clock=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                     log=log.append, driver_error=driver_error if driver_error_at is not None else None,
                     driver_window_s=window_s)
    return code, stops, log


def test_short_dips_like_spark_ad27_do_not_stop():
    # 2026-10-12 on gpu-node-1: free fell to 0.9-1.3 GiB for 1-3 s at a time and the box was fine
    code, stops, log = run([8.8, 1.3, 0.9, 2.4, 4.6, 5.0, 1.0, 1.8, 5.0] + [9.0] * 5, active_for=20)
    assert code is mod.Exit.UNIT_ENDED
    assert stops == []
    assert [r["event"] for r in log].count("recovered") == SPARK_DIPS


def test_a_lasting_low_stops_once_at_the_hold():
    code, stops, log = run([20.0, 3.0] + [2.0] * 30)
    assert code is mod.Exit.STOPPED
    assert stops == [11.0]  # low from t=1, stopped once 10 s later
    assert log[-1]["event"] == "stopping"


def test_low_at_the_floor_exactly_is_not_low():
    code, stops, _ = run([4.0] * 30, active_for=25)
    assert (code, stops) == (mod.Exit.UNIT_ENDED, [])


def test_no_stop_while_the_job_process_is_absent():
    # a unit still waiting for its turn must never be stopped for someone else's squeeze
    code, stops, _ = run([1.0] * 30, running=False, active_for=25)
    assert (code, stops) == (mod.Exit.UNIT_ENDED, [])


def test_a_recovery_resets_the_clock():
    code, stops, _ = run([2.0] * 9 + [6.0] + [2.0] * 9 + [6.0] + [9.0] * 3, active_for=22)
    assert (code, stops) == (mod.Exit.UNIT_ENDED, [])


def test_evict_counts_files_and_skips_symlinks(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.ckpt").write_bytes(b"0" * 5000)
    (tmp_path / "y.json").write_bytes(b"1" * 300)
    os.symlink("/etc/hostname", tmp_path / "link")
    assert evict_from_cache.evict([str(tmp_path)]) == {"files": 2, "bytes": 5300, "skipped": 1}
    assert evict_from_cache.evict([str(tmp_path / "y.json")])["files"] == 1


def test_a_driver_error_at_start_up_stops_at_the_next_check():
    # 2026-10-10 on Turing: NV_ERR_NO_MEMORY 7 s into the job, box frozen ~80 s later
    code, stops, log = run([50.0] * 40, driver_error_at=7.0)
    assert code is mod.Exit.DRIVER_ERROR
    assert stops == [10.0]  # first check after the error (checks every CHECK_EVERY_S = 5 s)
    assert log[-1]["event"] == "driver_error_stopping"


def test_a_driver_error_after_the_start_up_window_is_left_alone():
    code, stops, _ = run([50.0] * 60, driver_error_at=30.0, window_s=20.0, active_for=55)
    assert (code, stops) == (mod.Exit.UNIT_ENDED, [])


def test_no_driver_error_stop_while_the_job_process_is_absent():
    code, stops, _ = run([50.0] * 30, running=False, driver_error_at=0.0, active_for=25)
    assert (code, stops) == (mod.Exit.UNIT_ENDED, [])
