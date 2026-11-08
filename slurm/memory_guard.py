#!/usr/bin/env python3
"""The memory guard of a unified-memory node (D44): when the node's free memory falls below its close level, stop the
running job furthest over its ask (the memory its cgroup holds plus its own processes' card memory), through the job's
own wrapper, with `scancel --signal=KILL` when the wrapper does not act. It runs as a thread in the agent and watches
the agent's own node; the installer refuses a unified-memory site with a GPU node elsewhere.

    python3 slurm/memory_guard.py --once --level-gb N --only-job J   # one pass with a chosen level, for a check

Run as slurm's user: the stop request goes into the agent's own folder, which no job's user can write.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import socket
import sys
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import gpu_jobs
from gpu_jobs import JobMemory, MemoryStopEnd, QueuedJob, State, size_text

#: Bytes over its ask before a job counts as over it: above the sampling noise.
MIN_EXCESS = 10**9
#: After a stop, a further one waits this long for the stopped job's memory to come back.
HOLD_S = 10.0
#: A wrapper that has not acknowledged a stop by then is not there (a plain sbatch job) or hung: scancel.
ACK_S = 2.0
#: An acknowledged stop's processes must be gone by then (unmapping tens of GB takes seconds), or scancel anyway.
STUCK_S = 15.0
#: How often the guard reads free memory.
TICK_S = 1.0
#: How long the guard waits for nvidia-smi while memory is short; then it uses the agent's last sample.
GPU_TIMEOUT_S = 3.0
PREFIX = "[agent] memory guard:"

Log = Callable[[str], None]


def stops_dir(state_dir: Path | None = None) -> Path:
    """Where stop requests live: in the agent-only folder, so no job's user can forge one."""
    return (state_dir or gpu_jobs.STATE_DIR) / "agent" / "stops"


def stop_path(job_id: str, state_dir: Path | None = None) -> Path:
    return stops_dir(state_dir) / f"job_{job_id}.json"


class StopStep(StrEnum):
    """What a pending stop needs from the agent's step (`MemoryGuard.pending`)."""

    ENDED = "ended"  # the run is gone with nothing recorded yet: its wrapper ended it (`by_wrapper`)
    ESCALATE = "escalate"  # no acknowledgement in ACK_S, or acknowledged and still running after STUCK_S: scancel
    STUCK = "stuck"  # scancel was sent STUCK_S ago and the same run is still there (a process stuck in the driver)


class GuardAction(StrEnum):
    """What one pass did, for the log and the heartbeat."""

    HOLD = "hold"  # a stop was made less than HOLD_S ago
    NONE = "none"  # nothing over its ask (or all of it already being stopped)
    STOP = "stop"


@dataclass(frozen=True)
class StopRequest:
    """A stop of one run (job id + slurm start time), and the whole protocol around it: the agent writes it, the
    job's wrapper acknowledges and carries it out, the agent escalates when it does not."""

    job: str
    start: int
    at: float
    held: int
    card: int
    ask: int
    free: int
    escalated_at: float | None = None  # when the agent sent scancel
    ended: str | None = None  # the `MemoryStopEnd` of the newest memory_stop event written for this request

    def write(self, state_dir: Path | None = None) -> None:
        gpu_jobs.write_json(stop_path(self.job, state_dir), dataclasses.asdict(self))

    def updated(self, state_dir: Path | None = None, **changes: object) -> StopRequest:
        """This request with `changes`, written back."""
        request = dataclasses.replace(self, **changes)
        request.write(state_dir)
        return request

    @classmethod
    def read(cls, job_id: str, state_dir: Path | None = None, log: Log = print) -> StopRequest | None:
        """The request for `job_id`, trusted only when slurm's user or root wrote it; None otherwise (logged)."""
        path = stop_path(job_id, state_dir)
        data = gpu_jobs.read_json(path, gpu_jobs.SLURM_USER) or gpu_jobs.read_json(path, "root")
        request = cls._parse(data) if data is not None else None
        if request is None and path.is_file():
            log(f"{PREFIX} ignoring {path}: not a request written by {gpu_jobs.SLURM_USER} or root")
        return request

    @classmethod
    def _parse(cls, data: Mapping[str, object]) -> StopRequest | None:
        try:
            request = cls(**{f.name: data[f.name] for f in dataclasses.fields(cls) if f.name in data})
        except TypeError:  # a field missing
            return None
        numbers = (request.start, request.held, request.card, request.ask, request.free)
        if not (isinstance(request.job, str) and isinstance(request.at, int | float)
                and all(isinstance(n, int) for n in numbers)):
            return None
        return request

    def for_leg(self, leg_started: float) -> bool:
        """A request older than the wrapper's leg belongs to an earlier run of the job."""
        return self.at >= leg_started

    def ack_fields(self, now: float) -> dict[str, object]:
        """What the wrapper merges into its job record before it kills the command."""
        return {"stop_ack": {"request_at": self.at, "at": now}}

    def acked(self, sidecar: Mapping[str, object] | None) -> bool:
        ack = (sidecar or {}).get("stop_ack")
        return isinstance(ack, dict) and ack.get("request_at") == self.at

    def same_run(self, job: QueuedJob) -> bool:
        return job.job_id == self.job and job.start == self.start and job.state == State.RUNNING

    def escalation_due(self, now: float, acked: bool) -> bool:
        return now - self.at >= (STUCK_S if acked else ACK_S)

    def kill_description(self, pid: int, now: float) -> dict[str, object]:
        """The `killed_by` the wrapper records, in the shape `job_wrapper.after_kill` prints."""
        return {"job": self.job, "pid": pid, "signal": "SIGKILL", "guard": "memory stop-gap", "at": now}

    def event_fields(self, ended: MemoryStopEnd, message: str) -> dict[str, object]:
        """The fields of the memory_stop event for this request (`gpu_jobs.memory_kill` reads `total`)."""
        return {"message": message, "start": self.start, "held": self.held, "card": self.card, "ask": self.ask,
                "total": self.held + self.card, "free": self.free, "ended": str(ended)}


class StopPolicy:
    """Which job to stop: the largest excess of at least `min_excess`; ties to the larger total, then the lower id."""

    def __init__(self, min_excess: int = MIN_EXCESS) -> None:
        self.min_excess = min_excess

    def over(self, jobs: Sequence[JobMemory]) -> list[JobMemory]:
        """The jobs over their ask, the first to stop first."""
        # job ids are digits: shorter is lower
        return sorted((j for j in jobs if j.excess >= self.min_excess),
                      key=lambda j: (-j.excess, -j.total, len(j.job_id), j.job_id))

    def choose(self, jobs: Sequence[JobMemory], stopping: Collection[tuple[str, int]],
               only: str | None = None) -> JobMemory | None:
        return next((j for j in self.over(jobs)
                     if (j.job_id, j.start) not in stopping and (only is None or j.job_id == only)), None)


class Actuator:
    """Writes stop requests and sends the escalation."""

    def __init__(self, state_dir: Path | None = None, scancel: Callable[[list[str]], int] = gpu_jobs.run_slurm) -> None:
        self.state_dir, self.scancel = state_dir, scancel

    def request(self, jm: JobMemory, free: int, now: float) -> StopRequest:
        request = StopRequest(jm.job_id, jm.start, now, jm.held, jm.card, jm.ask, free)
        request.write(self.state_dir)
        return request

    def escalate(self, request: StopRequest) -> bool:
        """SIGKILL, the signal scancel sends to the batch step by default; True when scancel succeeded."""
        return self.scancel(["scancel", "--signal=KILL", request.job]) == 0

    def requests(self, log: Log = print) -> list[StopRequest]:
        folder = stops_dir(self.state_dir)
        names = sorted(p.name for p in folder.glob("job_*.json")) if folder.is_dir() else []
        found = (StopRequest.read(name.removeprefix("job_").removesuffix(".json"), self.state_dir, log) for name in names)
        return [r for r in found if r is not None]


def one_pass(policy: StopPolicy, actuator: Actuator, jobs: Sequence[tuple[QueuedJob, Path]], *, free: int, level: int,
             jobs_memory: int, gpu_usage: Callable[..., Mapping[int, float] | None], last_usage: Mapping[int, float],
             now: float, log: Log, only: str | None = None) -> dict[str, object]:
    """One decision while free memory is below the level: hold off, stop nothing, or request one stop."""
    head = f"{PREFIX} free {size_text(free)} below {size_text(level)}"
    requests = actuator.requests(log)
    record: dict[str, object] = {"at": now, "free": free, "level": level}
    newest = max((r.at for r in requests), default=None)
    if newest is not None and now - newest < HOLD_S:
        log(f"{head}: holding off, the last stop was {now - newest:.0f} s ago (a further one waits {HOLD_S:.0f} s)")
        return {**record, "action": GuardAction.HOLD}
    usage = gpu_usage(timeout=GPU_TIMEOUT_S)
    if usage is None:
        log(f"{PREFIX} nvidia-smi did not answer in {GPU_TIMEOUT_S:.0f} s; using the agent's last card sample")
        usage = last_usage
    measured = [m for job, cgroup in jobs
                if job.state == State.RUNNING and (m := JobMemory.measure(job, cgroup, usage, jobs_memory)) is not None]
    over = policy.over(measured)
    listing = ", ".join(f"{m.job_id} +{size_text(m.excess)}" for m in over)
    chosen = policy.choose(measured, {(r.job, r.start) for r in requests}, only)
    if chosen is None:
        why = f"over their ask: [{listing}], none to stop now" if over else "none over their ask"
        log(f"{head}: {why}; leaving it to the health check and earlyoom")
        return {**record, "action": GuardAction.NONE}
    actuator.request(chosen, free, now)
    log(f"{head}: over their ask: [{listing}]; stopping {chosen.job_id} (run {chosen.start})")
    return {**record, "action": GuardAction.STOP, "job": chosen.job_id, "start": chosen.start}


class MemoryGuard:
    """Reads free memory every TICK_S in its own thread and stops one job at a time while it is below the level. The
    agent's step hands it the running jobs, their cgroups, the level and its card sample (`update`), so the 1 s path
    never asks slurm anything."""

    def __init__(self, policy: StopPolicy, actuator: Actuator, *, meminfo_path: Path = gpu_jobs.MEMINFO,
                 gpu_usage: Callable[..., Mapping[int, float] | None] = gpu_jobs.gpu_usage,
                 clock: Callable[[], float] = time.time, log: Log = print) -> None:
        self.policy, self.actuator, self.meminfo_path = policy, actuator, meminfo_path
        self.gpu_usage, self.clock, self.log = gpu_usage, clock, log
        self._lock = threading.Lock()
        self._jobs: list[tuple[QueuedJob, Path]] = []
        self._level: int | None = None  # None until the agent's first step
        self._jobs_memory = 0
        self._last_usage: Mapping[int, float] = {}
        self._free: int | None = None
        self._last: dict[str, object] = {}
        self._stopped = threading.Event()

    def update(self, jobs: Sequence[tuple[QueuedJob, Path]], level: int, jobs_memory: int,
               last_usage: Mapping[int, float]) -> None:
        with self._lock:
            self._jobs, self._level, self._jobs_memory, self._last_usage = list(jobs), level, jobs_memory, dict(last_usage)

    def tick(self) -> dict[str, object] | None:
        with self._lock:  # a snapshot: the pass may wait GPU_TIMEOUT_S, and the agent's step must not wait with it
            jobs, level, jobs_memory, last_usage = self._jobs, self._level, self._jobs_memory, self._last_usage
        if level is None:
            return None  # the agent's first step has not handed anything over yet
        free = gpu_jobs.meminfo(self.meminfo_path)["MemAvailable"]
        self._free = free
        if free >= level:
            return None
        self._last = one_pass(self.policy, self.actuator, jobs, free=free, level=level, jobs_memory=jobs_memory,
                              gpu_usage=self.gpu_usage, last_usage=last_usage, now=self.clock(), log=self.log)
        return self._last

    def pending(self, queue: Sequence[QueuedJob],
                read_sidecar: Callable[..., Mapping[str, object] | None] = gpu_jobs.read_sidecar,
                ) -> list[tuple[StopRequest, StopStep]]:
        """What each stop request needs now; the agent acts on it and records the outcome in the request."""
        now, by_id = self.clock(), {job.job_id: job for job in queue}
        steps: list[tuple[StopRequest, StopStep]] = []
        for request in self.actuator.requests(self.log):
            job = by_id.get(request.job)
            if job is None or not request.same_run(job):
                if request.ended is None:
                    steps.append((request, StopStep.ENDED))
            elif request.escalated_at is None:
                if request.escalation_due(now, request.acked(read_sidecar(job.job_id, job.user))):
                    steps.append((request, StopStep.ESCALATE))
            elif now - request.escalated_at >= STUCK_S and request.ended != MemoryStopEnd.NOT_ENDED:
                steps.append((request, StopStep.STUCK))
        return steps

    def prunable(self, request: StopRequest, job: QueuedJob | None) -> bool:
        """A request can go once its run is gone and recorded, and HOLD_S has passed (the hold-off reads it)."""
        gone = job is None or not request.same_run(job)
        return gone and request.ended is not None and self.clock() - request.at >= HOLD_S

    def start(self) -> None:
        threading.Thread(target=self._run, name="memory-guard", daemon=True).start()

    def stop(self) -> None:
        self._stopped.set()

    def _run(self) -> None:
        while not self._stopped.wait(TICK_S):
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - one bad pass must never end the guard
                self.log(f"{PREFIX} a pass failed: {error!r}")

    def heartbeat(self) -> dict[str, object]:
        level, free = self._level, self._free
        return {"level_mb": None if level is None else level // gpu_jobs.MB,
                "free_mb": None if free is None else free // gpu_jobs.MB,
                "last_action": str(self._last.get("action", "")), "last_at": self._last.get("at")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One pass of the memory guard on this node, with a level and one "
                                                 "job chosen for a check. The agent runs the guard itself.")
    parser.add_argument("--once", action="store_true", required=True, help="one pass, then exit (the only mode)")
    parser.add_argument("--level-gb", type=float, required=True, help="act as if this were the close level (decimal GB)")
    parser.add_argument("--only-job", required=True, help="the one job id the pass may stop")
    args = parser.parse_args(argv)
    queue = gpu_jobs.read_queue()
    if queue is None:
        print("squeue did not answer", file=sys.stderr)
        return 1
    here = socket.gethostname().split(".")[0]
    node = next((n for n in gpu_jobs.load_site().nodes if n.name == here), None)
    if node is None:
        print(f"this host ({here}) is not a node of the site", file=sys.stderr)
        return 1
    jobs = [(job, cgroup) for job in queue
            if job.node == here and job.state == State.RUNNING and (cgroup := gpu_jobs.job_cgroup(job)) is not None]
    free, level = gpu_jobs.meminfo()["MemAvailable"], round(args.level_gb * gpu_jobs.GB)
    if free >= level:
        print(f"{PREFIX} free {size_text(free)} is not below {size_text(level)}: nothing to do")
        return 0
    record = one_pass(StopPolicy(), Actuator(), jobs, free=free, level=level, jobs_memory=node.jobs_mb * gpu_jobs.SLURM_MB,
                      gpu_usage=gpu_jobs.gpu_usage, last_usage={}, now=time.time(), log=print, only=args.only_job)
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
