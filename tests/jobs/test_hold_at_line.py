"""hold_at_line: freeze a real process at a log line, after its awaited child starts; kill only the named child."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import hold_at_line as mod
import pytest

#: A stand-in queue: on "go" it logs the trigger line, starts a scoring child, then a card-check child.
QUEUE = textwrap.dedent(
    """
    import subprocess, sys, time
    log, go = sys.argv[1], sys.argv[2]
    while not __import__('os').path.exists(go):
        time.sleep(0.01)
    open(log, 'a').write('cell_a: exited 0\\n')
    time.sleep(0.3)  # the real queue's gap is milliseconds; widened so a hold that skips the wait is caught
    sleep = [sys.executable, '-c', 'import time; time.sleep(60)']
    kids = [subprocess.Popen(sleep + ['score_swe_rollout.py'])]
    time.sleep(0.2)
    kids.append(subprocess.Popen(sleep + ['assert_gpu_free.py']))
    time.sleep(60)
    """
)


def state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rindex(")") + 2:].split()[0]


@pytest.fixture
def queue(tmp_path):
    log, go = tmp_path / "queue.log", tmp_path / "go"
    log.write_text("cell_a: training\n")
    proc = subprocess.Popen([sys.executable, "-c", QUEUE, str(log), str(go)])
    yield proc, log, go
    for pid in mod.children(proc.pid):
        subprocess.run(["kill", "-9", str(pid)], check=False)
    proc.kill()
    proc.wait()


def hold(proc, log, go, *extra, await_child="score_swe_rollout.py"):
    result = {}
    thread = threading.Thread(target=lambda: result.update(code=mod.main([
        "--pid", str(proc.pid), "--log", str(log), "--line", "cell_a: exited",
        "--await-child", await_child, *extra])))
    thread.start()
    time.sleep(0.3)  # armed, at the log's end
    go.touch()
    thread.join(timeout=10)
    return result["code"]


def test_holds_after_the_awaited_child_and_kills_only_the_named_one(queue):
    proc, log, go = queue
    code = hold(proc, log, go, "--kill-child", "assert_gpu_free.py", "--settle", "2")
    assert code == mod.Exit.HELD
    assert state(proc.pid) == "T"  # the queue is frozen before it can start its card check
    kids = mod.children(proc.pid)
    assert mod.matching(kids, ["score_swe_rollout.py"])  # scoring started and still runs
    assert not mod.matching(kids, ["assert_gpu_free.py"])


def test_a_card_check_already_in_flight_is_killed_and_scoring_survives(queue):
    proc, log, go = queue
    code = hold(proc, log, go, "--kill-child", "assert_gpu_free.py", await_child="assert_gpu_free.py")
    assert code == mod.Exit.HELD
    assert state(proc.pid) == "T"
    time.sleep(0.2)  # the SIGKILLed child is reaped by nobody (parent frozen): a zombie, not a live child
    kids = mod.children(proc.pid)
    assert mod.matching(kids, ["score_swe_rollout.py"])
    assert not mod.matching(kids, ["assert_gpu_free.py"])


def test_an_old_line_does_not_trigger(tmp_path):
    log = tmp_path / "queue.log"
    log.write_text("cell_a: exited 0\n")  # already in the log before arming
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(line=mod.wait_for_line(log, "cell_a: exited", 0.01)), daemon=True)
    thread.start()
    time.sleep(0.2)
    assert "line" not in result
    with log.open("a") as stream:
        stream.write("cell_a: exited 0 (new)\n")
    thread.join(timeout=5)
    assert result["line"] == "cell_a: exited 0 (new)"


def test_a_started_next_job_is_reported_late(queue):
    proc, log, go = queue
    code = hold(proc, log, go, "--late-child", "score_swe_rollout.py")
    assert code == mod.Exit.LATE
    assert state(proc.pid) == "T"
