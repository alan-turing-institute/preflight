"""heartbeat: report while healthy, stop reporting only after a sustained memory stall."""

from __future__ import annotations

import json
import os
import socket

import heartbeat as mod
import pytest

#: Real /proc/pressure/memory from gpu-node-1, 2026-10-13.
PSI = (
    "some avg10=0.00 avg60=0.00 avg300=0.00 total=365249245\n"
    "full avg10=0.00 avg60=0.00 avg300=0.00 total={full}\n"
)


FULL_TOTAL = 344641117
HOLD_S = 20


def test_reads_the_full_line_total(tmp_path):
    path = tmp_path / "psi"
    path.write_text(PSI.format(full=FULL_TOTAL))
    assert mod.full_total_us(os.open(path, os.O_RDONLY)) == FULL_TOTAL


def test_stall_needs_every_interval_over_threshold_for_the_whole_hold():
    stall = mod.Stall(threshold_pct=10, hold_s=30)
    assert not stall.update(1_000_000, 10)  # 10 % for 10 s
    assert not stall.update(1_000_000, 10)  # 20 s
    assert stall.update(1_000_000, 10)  # 30 s: held


def test_one_quiet_interval_resets_the_count():
    stall = mod.Stall(threshold_pct=10, hold_s=30)
    stall.update(5_000_000, 10)
    stall.update(5_000_000, 10)
    assert not stall.update(999_999, 10)  # 9.99999 %: below threshold
    assert stall.seconds == 0
    assert not stall.update(5_000_000, 10)


class Stop(BaseException):
    """Ends main()'s loop from inside time.sleep (BaseException: the loop catches Exception)."""


def run_main(tmp_path, monkeypatch, fulls, argv):
    """Run main() under a fake systemd notify socket, feeding one PSI total per tick."""
    psi, address = tmp_path / "psi", str(tmp_path / "notify")
    psi.write_text(PSI.format(full=fulls[0]))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(address)
    server.setblocking(False)
    monkeypatch.setenv("NOTIFY_SOCKET", address)
    monkeypatch.setattr(mod, "PSI", str(psi))
    monkeypatch.setattr(mod, "STATE", tmp_path / "state" / "withheld.json")
    clock = iter(range(0, 10_000, 10))
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(clock))
    ticks = iter(fulls[1:])

    def sleep(_):
        try:
            psi.write_text(PSI.format(full=next(ticks)))
        except StopIteration:
            raise Stop from None

    monkeypatch.setattr(mod.time, "sleep", sleep)
    with pytest.raises(Stop):
        mod.main(argv)
    messages = []
    while True:
        try:
            messages.append(server.recv(64))
        except BlockingIOError:
            return messages


def test_healthy_box_reports_every_tick(tmp_path, monkeypatch):
    messages = run_main(tmp_path, monkeypatch, [0, 0, 0, 0], ["--interval", "10"])
    assert messages == [b"READY=1"] + [b"WATCHDOG=1"] * 3


def test_sustained_stall_stops_reports_and_records_why(tmp_path, monkeypatch, capsys):
    # each tick adds 5 s of "full" stall in 10 s (50 %); hold 20 s -> withheld from tick 2
    fulls = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000]
    messages = run_main(tmp_path, monkeypatch, fulls, ["--hold", str(HOLD_S)])
    assert messages == [b"READY=1", b"WATCHDOG=1"]
    record = json.loads((tmp_path / "state" / "withheld.json").read_text())
    assert record["stalled_s"] == HOLD_S
    assert '"heartbeat_withheld"' in capsys.readouterr().out


def test_reports_resume_when_the_stall_clears(tmp_path, monkeypatch):
    fulls = [0, 5_000_000, 10_000_000, 10_000_000, 10_000_000]
    messages = run_main(tmp_path, monkeypatch, fulls, ["--hold", str(HOLD_S)])
    assert messages == [b"READY=1", b"WATCHDOG=1", b"WATCHDOG=1", b"WATCHDOG=1"]


def test_unreadable_psi_still_reports(tmp_path, monkeypatch, capsys):
    """A broken gauge must never reboot the box: it logs and keeps reporting."""
    address = str(tmp_path / "notify")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(address)
    server.setblocking(False)
    monkeypatch.setenv("NOTIFY_SOCKET", address)
    monkeypatch.setattr(mod, "PSI", str(tmp_path / "missing"))
    ticks = iter(range(3))

    def sleep(_):
        if next(ticks, None) is None:
            raise Stop

    monkeypatch.setattr(mod.time, "sleep", sleep)
    with pytest.raises(Stop):
        mod.main([])
    assert [server.recv(64) for _ in range(4)] == [b"READY=1"] + [b"WATCHDOG=1"] * 3
    out = capsys.readouterr().out
    assert '"psi_unavailable"' in out and '"loop_error"' in out
