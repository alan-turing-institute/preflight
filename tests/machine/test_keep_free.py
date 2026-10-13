"""keep_free: parse real /proc text, act only below the free-memory floor, log before/after."""

from __future__ import annotations

import json
import os

import keep_free as mod
import pytest

GIB = mod.KIB_PER_GIB
#: Real lines from gpu-node-1 (2026-10-13, training running): orders 0..13.
BUDDY = (
    "Node 0, zone      DMA   1822   1332   1075    551    227     74     14      6      3"
    "      2      3      2      2     13 \n"
    "Node 0, zone   Normal 824110 858246 520351 251402  82865  32610  10899   2978    773"
    "   1396    175     26      0      0 \n"
)


def meminfo_text(free_gib: float, cached_gib: float) -> str:
    return (
        "MemTotal:       127600748 kB\n"
        f"MemFree:        {int(free_gib * GIB)} kB\n"
        "MemAvailable:   51002000 kB\nBuffers:          172436 kB\n"
        f"Cached:         {int(cached_gib * GIB)} kB\nSwapCached:        43632 kB\n"
    )


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A fake /proc: meminfo, buddyinfo and a drop_caches file that records writes."""
    mem, buddy, drop = tmp_path / "meminfo", tmp_path / "buddyinfo", tmp_path / "drop"
    buddy.write_text(BUDDY)
    drop.write_text("")
    monkeypatch.setattr(mod, "MEMINFO", str(mem))
    monkeypatch.setattr(mod, "BUDDYINFO", str(buddy))
    monkeypatch.setattr(mod, "DROP_CACHES", str(drop))
    return mem, drop


def test_meminfo_reads_free_and_cached(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(meminfo_text(5.0, 20.0))
    fd = os.open(path, os.O_RDONLY)
    assert mod.meminfo(fd) == (5 * GIB, 20 * GIB)


#: DMA 3+2+2+13 = 20, Normal 175+26+0+0 = 201 (the 201 measured on the box).
BUDDY_LARGE = 221


def test_large_blocks_counts_order_ten_and_up_in_every_zone():
    assert mod.large_blocks(BUDDY) == BUDDY_LARGE


@pytest.mark.parametrize(
    ("free", "cached", "expected"),
    [(15.9, 20, True), (16.0, 20, False), (2, 3.9, False), (2, 4.0, True)],
)
def test_acts_only_below_the_floor_and_with_cache_to_drop(free, cached, expected):
    assert mod.needs_action(int(free * GIB), int(cached * GIB), 16 * GIB, 4 * GIB) is expected


def test_once_below_floor_drops_cache_and_logs_before_after(box, capsys):
    mem, drop = box
    mem.write_text(meminfo_text(3.0, 30.0))
    assert mod.main(["--once"]) == 0
    assert drop.read_text() == "1"  # pagecache only: never 3, never a sync
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["event"] == "cache_dropped"
    assert record["before"] == {"free_gib": 3.0, "cached_gib": 30.0, "large_blocks": BUDDY_LARGE}
    assert "drop_s" in record and "after" in record


def test_once_above_floor_touches_nothing(box, capsys):
    mem, drop = box
    mem.write_text(meminfo_text(40.0, 30.0))
    mod.main(["--once"])
    assert drop.read_text() == ""
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "no_action"


def test_a_failed_drop_is_logged_not_fatal(box, capsys):
    mem, drop = box
    mem.write_text(meminfo_text(3.0, 30.0))
    drop.unlink()
    drop.mkdir()  # opening a directory for writing raises OSError
    assert mod.main(["--once"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "action_failed"
