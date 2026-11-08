"""The memory guard (D44): which job it stops, the stop request's protocol, one pass, and what a pending stop needs."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

import gpu_jobs
import memory_guard as mg
from gpu_jobs import JobMemory, MemoryStopEnd, QueuedJob

GB = 10**9
NEXT_RUN = 2  # a requeued run of a job: a new slurm start time


def memory(job_id: str, held_gb: float, ask_gb: float, card_gb: float = 0.0, start: int = 1) -> JobMemory:
    return JobMemory(job_id, "u", "n", start, int(held_gb * GB), int(card_gb * GB), int(ask_gb * GB))


def test_policy_stops_the_largest_excess_and_never_a_job_within_its_ask() -> None:
    policy = mg.StopPolicy()
    big, carded, within, noise = memory("7", 20, 10), memory("8", 15, 10, card_gb=3), memory("9", 30, 40), memory("10", 10.5, 10)
    assert policy.choose([noise, within, carded, big], set()) == big  # 10 GB over beats 8 GB over (host + card)
    assert policy.choose([noise, within], set()) is None  # within its ask, or less than MIN_EXCESS over: never
    holds_more, holds_less = memory("12", 12, 2), memory("11", 11, 1)  # both 10 GB over
    assert policy.choose([holds_less, holds_more], set()) == holds_more
    assert policy.choose([memory("21", 11, 1), memory("3", 11, 1)], set()).job_id == "3"  # then the lower id
    assert policy.choose([big, carded], {("7", 1)}) == carded  # a run already being stopped is skipped ...
    assert policy.choose([memory("7", 20, 10, start=NEXT_RUN), carded], {("7", 1)}).start == NEXT_RUN  # ... its next run is not
    assert policy.choose([big, carded], set(), only="8") == carded


def test_a_stop_request_is_trusted_only_from_the_agent_and_belongs_to_one_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = mg.StopRequest("5", 77, 1000.0, 9 * GB, 2 * GB, 8 * GB, 3 * GB)
    request.write(tmp_path)
    logged: list[str] = []
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", "somebody-else")  # the file's owner is neither slurm's user nor root
    assert mg.StopRequest.read("5", tmp_path, logged.append) is None and "ignoring" in logged[0]
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
    assert mg.StopRequest.read("5", tmp_path) == request
    assert mg.StopRequest.read("6", tmp_path, logged.append) is None and len(logged) == 1  # no file: nothing to say
    assert request.for_leg(999.0) and not request.for_leg(1001.0)  # older than the wrapper's leg: an earlier run's
    assert request.acked({"stop_ack": request.ack_fields(1001.0)["stop_ack"]}) and not request.acked(None)
    assert not request.acked({"stop_ack": {"request_at": 900.0, "at": 901.0}})  # an earlier request's acknowledgement
    assert not request.escalation_due(1001.9, acked=False) and request.escalation_due(1002.0, acked=False)
    assert not request.escalation_due(1014.9, acked=True) and request.escalation_due(1015.0, acked=True)
    running = QueuedJob("5", "gpu", "RUNNING", "", "n", start=77)
    assert request.same_run(running) and not request.same_run(dataclasses.replace(running, start=78))  # requeued
    assert not request.same_run(dataclasses.replace(running, state="SUSPENDED"))


class Node:
    """A fake node: a meminfo file, two running jobs' cgroups, a card, a clock, and every call the guard makes."""

    def __init__(self, tmp_path: Path, make_cgroup: Callable[..., Path]) -> None:
        self.state, self.meminfo, self.now = tmp_path / "state", tmp_path / "meminfo", 10_000.0
        self.calls: list[list[object]] = []
        self.lines: list[str] = []
        self.usage: dict[int, float] | None = {100: 2048.0}  # MiB: 2.1 GB on the card for job 7's pid
        self.actuator = mg.Actuator(self.state, scancel=lambda argv: self.calls.append(argv) or 0)
        over = make_cgroup("over", current=20 * GB, cache=0, shmem=0, limit=str(10 * GB), pids="100\n")
        within = make_cgroup("within", current=5 * GB, cache=0, shmem=0, limit=str(10 * GB), pids="200\n")
        paused = make_cgroup("paused", current=40 * GB, cache=0, shmem=0, limit=str(10 * GB), pids="300\n")
        self.jobs = [(QueuedJob("7", "gpu", "RUNNING", "", "big", user="u", start=1), over),
                     (QueuedJob("8", "gpu", "RUNNING", "", "small", user="u", start=2), within),
                     (QueuedJob("9", "cpu", "SUSPENDED", "", "paused", user="u", start=3), paused)]  # never chosen (D20)
        self.guard = self.new_guard()

    def new_guard(self) -> mg.MemoryGuard:  # what an agent restart builds
        guard = mg.MemoryGuard(mg.StopPolicy(), self.actuator, meminfo_path=self.meminfo, gpu_usage=self.gpu,
                               clock=lambda: self.now, log=self.lines.append)
        guard.update(self.jobs, level=16 * GB, jobs_memory=87 * GB, last_usage={100: 1024.0})
        return guard

    def gpu(self, timeout: float) -> dict[int, float] | None:
        self.calls.append(["nvidia-smi", timeout])
        return self.usage

    def free(self, gb: float) -> None:
        self.meminfo.write_text(f"MemTotal: 130000000 kB\nMemAvailable: {round(gb * GB / 1024)} kB\n")


def test_one_pass_requests_one_stop_holds_off_and_logs_every_decision(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_cgroup: Callable[..., Path]) -> None:
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
    node = Node(tmp_path, make_cgroup)
    assert mg.MemoryGuard(mg.StopPolicy(), node.actuator, meminfo_path=node.meminfo).tick() is None  # nothing handed over
    node.free(20)
    assert node.guard.tick() is None and node.calls == []  # not short: not even nvidia-smi
    node.free(15)
    assert node.guard.tick()["action"] == mg.GuardAction.STOP
    assert node.calls == [["nvidia-smi", mg.GPU_TIMEOUT_S]]  # a request is written; nothing is sent to slurm
    request = mg.StopRequest.read("7", node.state)
    free = gpu_jobs.meminfo(node.meminfo)["MemAvailable"]
    assert (request.start, request.held, request.card, request.ask, request.free) == (1, 20 * GB, 2048 * 2**20, 10 * GB, free)
    assert node.lines[-1] == "[agent] memory guard: free 15.0 GB below 16.0 GB: over their ask: [7 +12.1 GB]; stopping 7 (run 1)"
    node.now += 5
    node.guard = node.new_guard()  # the hold-off is read from the request files, so a restarted agent keeps it
    assert node.guard.tick()["action"] == mg.GuardAction.HOLD and "holding off" in node.lines[-1]
    node.now += 6
    node.usage = None  # nvidia-smi does not answer: the agent's last sample (1 GB on the card) is used, and said
    assert node.guard.tick()["action"] == mg.GuardAction.NONE
    assert "did not answer" in node.lines[-2] and "[7 +11.1 GB], none to stop now" in node.lines[-1]
    node.jobs.pop(0)
    node.guard = node.new_guard()
    node.guard.tick()
    assert node.lines[-1].endswith("none over their ask; leaving it to the health check and earlyoom")
    assert node.guard.heartbeat() == {"level_mb": 16_000, "free_mb": 15_000, "last_action": "none", "last_at": node.now}
    for method in (mg.MemoryGuard.__init__, mg.MemoryGuard.update, mg.MemoryGuard.tick):  # the agent cannot aim it
        assert "only" not in inspect.signature(method).parameters


def test_a_pending_stop_escalates_without_an_ack_waits_with_one_and_is_recorded_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_cgroup: Callable[..., Path]) -> None:
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
    node = Node(tmp_path, make_cgroup)
    request = node.actuator.request(memory("7", 20, 10), 15 * GB, node.now)
    queue = [node.jobs[0][0]]
    sidecars: dict[str, dict[str, object]] = {}

    def steps(at: float, jobs: list[QueuedJob]) -> list[mg.StopStep]:
        node.now = request.at + at
        return [step for _, step in node.guard.pending(jobs, lambda job_id, _user: sidecars.get(job_id))]

    assert steps(1.9, queue) == [] and steps(2.0, queue) == [mg.StopStep.ESCALATE]  # no acknowledgement in 2 s
    sidecars["7"] = request.ack_fields(request.at + 0.5)
    assert steps(14.9, queue) == [] and steps(15.0, queue) == [mg.StopStep.ESCALATE]  # acknowledged: 15 s to end
    sent = request.updated(node.state, escalated_at=request.at + 15, ended=MemoryStopEnd.BY_SCANCEL)
    assert steps(29.9, queue) == [] and steps(30.0, queue) == [mg.StopStep.STUCK]  # still there 15 s after scancel
    sent.updated(node.state, ended=MemoryStopEnd.NOT_ENDED)
    assert steps(31.0, queue) == [] and steps(31.0, []) == []  # recorded: nothing more, even once it is gone
    request.write(node.state)  # a fresh request whose run ended by itself, or was requeued
    assert steps(3.0, []) == [mg.StopStep.ENDED] and steps(3.0, [dataclasses.replace(queue[0], start=2)]) == [mg.StopStep.ENDED]
    recorded = request.updated(node.state, ended=MemoryStopEnd.BY_WRAPPER)
    node.now = request.at + mg.HOLD_S - 1
    assert not node.guard.prunable(recorded, None)  # kept for the hold-off
    node.now = request.at + mg.HOLD_S
    assert node.guard.prunable(recorded, None) and not node.guard.prunable(recorded, queue[0])
    assert node.actuator.escalate(request) and node.calls[-1] == ["scancel", "--signal=KILL", "7"]
