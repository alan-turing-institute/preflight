"""Tests for the flight recorder. Each names the mutation it would catch."""

from __future__ import annotations

import datetime as dt
import json
import sys

import flight_recorder as mod

MEMINFO = """MemTotal:       127600000 kB
MemFree:         1048576 kB
MemAvailable:   59768832 kB
Cached:         58195968 kB
SwapFree:       16776324 kB
"""
PSI = """some avg10=12.50 avg60=3.00 avg300=1.00 total=1
full avg10=7.25 avg60=2.00 avg300=0.50 total=1
"""
FREE_MIB = 1024.0
AVAIL_MIB = 58368.0
PSI_SOME, PSI_FULL = 12.5, 7.25
SCAN_RATE, ALLOCSTALL_RATE = 50.0, 3.0
KEEP_DAYS = 14


def test_meminfo_keeps_free_and_available_apart():
    """Mutation: record only MemAvailable -> the GB10 low-free state is invisible."""
    out = mod.parse_meminfo(MEMINFO)
    assert out["free"] == FREE_MIB
    assert out["avail"] == AVAIL_MIB
    assert "MemTotal" not in out and "total" not in out


def test_psi_reads_both_lines():
    """Mutation: parse only the first line -> 'full' stalls (nothing runs) are lost."""
    assert mod.parse_psi(PSI, "psi_memory") == {
        "psi_memory_some": PSI_SOME,
        "psi_memory_full": PSI_FULL,
    }


def test_rates_are_per_second_and_allocstall_sums_zones():
    """Mutation: report raw counters or one zone -> rates meaningless across intervals."""
    before = mod.parse_vmstat("pgscan_kswapd 100\nallocstall_normal 1\nallocstall_movable 1\n")
    after = mod.parse_vmstat("pgscan_kswapd 200\nallocstall_normal 3\nallocstall_movable 5\n")
    out = mod.rates(before, after, seconds=2.0)
    assert out["scan_kswapd_s"] == SCAN_RATE
    assert out["allocstall_s"] == ALLOCSTALL_RATE


def test_each_line_is_fsynced_and_old_days_pruned(tmp_path, monkeypatch):
    """Mutation: drop fsync -> the last seconds before a freeze are lost; never prune."""
    synced = []
    monkeypatch.setattr(mod.os, "fsync", lambda fd: synced.append(fd))
    today = dt.date(2026, 10, 12)
    old = tmp_path / f"flight-{(today - dt.timedelta(days=KEEP_DAYS + 1)).isoformat()}.jsonl"
    kept = tmp_path / f"flight-{(today - dt.timedelta(days=KEEP_DAYS)).isoformat()}.jsonl"
    old.write_text("{}\n")
    kept.write_text("{}\n")
    sink = mod.DayFile(tmp_path, KEEP_DAYS)
    sink.write(json.dumps({"t": 1}), today)
    assert not old.exists() and kept.exists()
    sink.write(json.dumps({"t": 2}), today + dt.timedelta(days=1))
    assert len(synced) == 2
    assert (tmp_path / "flight-2026-10-13.jsonl").read_text() == '{"t": 2}\n'


def test_a_stuck_gpu_query_is_never_stacked():
    """Mutation: launch a new query each poll -> stuck nvidia-smi processes pile up."""
    poller = mod.GpuPoller(every=1, timeout=0.2, command=[sys.executable, "-c", "import time; time.sleep(30)"])
    poller.poll()
    assert poller.result == "timeout"
    first = poller.stuck
    first.send_signal(19)  # SIGSTOP: stays alive like a query stuck in the driver
    try:
        poller.poll()
        assert poller.result == "previous query still stuck"
        assert poller.stuck is first
    finally:
        first.kill()
        first.send_signal(18)
        first.wait()


def test_gpu_parse_tolerates_unavailable_memory():
    """Mutation: float() every field -> one '[N/A]' kills every GPU reading."""
    assert mod.parse_gpu("123, 51234\n456, [N/A]\n") == {"123": 51234.0, "456": None}


def test_unit_points_at_this_script_and_restarts(tmp_path):
    """Mutation: a relative or %h path -> the unit breaks on the Turing deployed copy."""
    script = tmp_path / "flight_recorder.py"
    text = mod.unit_text(script, tmp_path / "out", python="/usr/bin/python3")
    assert f"ExecStart=/usr/bin/python3 {script} --out {tmp_path / 'out'}" in text
    assert "Restart=always" in text and "WantedBy=default.target" in text
