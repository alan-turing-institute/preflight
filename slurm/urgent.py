#!/usr/bin/env python3
"""Run urgent work on gpu-node-1 now: show what is in its way and what a pause loses, then pre-empt and submit.

Design 3.6 (`design notes`).

    slurm/urgent.py --project P --what '...' --time 2:00:00 [--mem M | --whole-box] [--gpu] \
        [--wait] [--requeue-steady] [--dry-run] -- <command...>

1. THE PLAN, printed before acting, classifies everything in the urgent job's way:
   pausable at once (`normal` jobs: they lose at most the time since their last checkpoint; service-a),
   pausable with a delay (service-b, at its next job boundary), wait-only (a `steady` job, another urgent
   job, any other lease holder such as service-c's queue, unregistered GPU work).
2. With a wait-only blocker it stops and submits nothing, unless `--wait` (accept the wait) or, for a
   `steady` job, `--requeue-steady` (requeue it, losing its progress so far, which the plan states).
3. Otherwise it submits to `urgent` with the required `--time`. Default size `--whole-box`: the GPU, all
   memory and all cores. Slurm pre-empts and requeues the `normal` jobs in the way; their dependents wait;
   the services step aside because an urgent job is keeper demand, with or without `--gpu`.
4. ORDER AFTERWARDS: a requeued job cannot start for the requeue wait, so an urgent job shorter than that
   would hand the card to another queued job first. Other pending tier-1 GPU jobs are held and recorded in
   `restore_order.json`; the keeper releases them once every pre-empted job is running again or has left.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit  # noqa: E402 - path set just above
import gpu_jobs  # noqa: E402
import submit  # noqa: E402

#: Background services in the lease protocol that yield to a waiting lease, and how fast (design section 2).
YIELDING_SERVICES = {
    "service-a": "yields within about 1 min (30-s poll, then its stop)",
    "service-b": "yields at its next job boundary, up to ~15 min (an inference runs up to 900 s)",
}
#: slurm's own delay from an urgent submission to its start, measured in the trial (build notes).
SLURM_START_S = (31, 46)
RESTORE_ORDER = "restore_order.json"


def elapsed(job_id: str) -> str:
    done = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%M"], capture_output=True, text=True, check=False)
    return done.stdout.strip() or "?"


def plan(
    queue: list[gpu_jobs.QueuedJob], *, gpu: bool, whole_box: bool, leases: list[tuple[str, bool]],
    unregistered: list[tuple[int, float, str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """(lines to print, normal victims, steady blockers, other wait-only blockers)."""
    lines, victims, steady, wait_only = [], [], [], []
    for job in queue:
        record = gpu_jobs.read_sidecar(job.job_id) or {}
        in_way = whole_box or (gpu and bool(record.get("gpu")))
        if not in_way or not (job.active or job.partition == gpu_jobs.Partition.URGENT):
            continue
        what = f"{job.job_id} {job.name} ({record.get('what', 'no sidecar')}), running {elapsed(job.job_id)}"
        if job.partition == gpu_jobs.Partition.NORMAL:
            victims.append(job.job_id)
            lines.append(f"[urgent] PAUSABLE  {what}: requeued; loses at most the time since its last checkpoint")
        elif job.partition == gpu_jobs.Partition.STEADY:
            steady.append(job.job_id)
            lines.append(f"[urgent] WAIT-ONLY {what}: cannot resume; requeueing it loses all of that run")
        else:
            wait_only.append(job.job_id)
            lines.append(f"[urgent] WAIT-ONLY {what}: another urgent job; this one waits behind it")
    for project, waiting in leases:
        if project == gpu_jobs.LEASE_PROJECT:
            continue
        if project in YIELDING_SERVICES:
            lines.append(f"[urgent] PAUSABLE  service {project}: {YIELDING_SERVICES[project]}")
        else:
            wait_only.append(f"lease:{project}")
            state = "waiting for the card" if waiting else "holds the card"
            lines.append(f"[urgent] WAIT-ONLY {project} {state} and does not yield to a request")
    for pid, mib, command in unregistered:
        wait_only.append(f"pid:{pid}")
        lines.append(f"[urgent] WAIT-ONLY unregistered GPU work, pid {pid} ({mib:.0f} MiB): {command[:120]}")
    if not lines:
        lines.append("[urgent] nothing in the way")
    return lines, victims, steady, wait_only


def other_leases() -> list[tuple[str, bool]]:
    found = []
    for path in gpu_jobs.LEASE_DIR.glob("*.json"):
        record = gpu_jobs.read_json(path) or {}
        if gpu_jobs.pid_alive(record.get("pid")):
            found.append((str(record.get("project") or path.stem), record.get("requested_at") is not None))
    return found


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", required=True)
    p.add_argument("--what", required=True)
    p.add_argument("--time", required=True, help="how long the urgent job may run (slurm time, e.g. 2:00:00)")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--mem", default=None, help="memory it needs; default: the whole box")
    size.add_argument("--whole-box", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--wait", action="store_true", help="accept waiting behind what cannot be paused")
    p.add_argument("--requeue-steady", action="store_true", help="requeue steady jobs in the way (loses them)")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--name", default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--workdir", type=Path, default=Path.cwd())
    p.add_argument("--wait-limit", default="1h")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the job script; change nothing")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def run(argv: list[str]) -> int:
    arguments = parser().parse_args(argv)
    arguments.command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not arguments.command:
        print("[urgent] REFUSED: no command after --", file=sys.stderr)
        return 1
    whole_box = arguments.whole_box or arguments.mem is None
    arguments.gpu = arguments.gpu or whole_box
    arguments.name = arguments.name or Path(arguments.command[0]).name
    arguments.job_class, arguments.after, arguments.max_jobs = "urgent", [], 1
    queue = gpu_jobs.read_queue()
    if queue is None:
        print("[urgent] REFUSED: slurm does not answer (controller down); see slurm/status.py", file=sys.stderr)
        return 1
    lines, victims, steady, wait_only = plan(
        queue, gpu=arguments.gpu, whole_box=whole_box, leases=other_leases(), unregistered=audit.unregistered()
    )
    print("\n".join(lines))
    print(f"[urgent] expected start: slurm's own {SLURM_START_S[0]}-{SLURM_START_S[1]} s after the pre-empted jobs "
          "end (trial), plus the slowest yield above")
    blocked = wait_only + ([] if arguments.requeue_steady else steady)
    if blocked and not arguments.wait:
        print(f"[urgent] STOPPED, nothing submitted: {blocked} cannot be paused. Pass --wait to queue behind them"
              + (", or --requeue-steady to requeue the steady jobs (they lose their progress)" if steady else ""),
              file=sys.stderr)
        return 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = arguments.log or gpu_jobs.STATE_DIR / "logs" / f"{arguments.name}-{stamp}.log"
    extra = [f"--time={arguments.time}"] + (["--exclusive"] if whole_box else [])
    mem = "0" if whole_box else arguments.mem  # --mem=0: all of the node's memory
    script = submit.sbatch_script(
        arguments, mem=mem, log=log, manifest=None, partition=gpu_jobs.Partition.URGENT, extra=tuple(extra),
        needs_lease=True,
    )
    if arguments.dry_run:
        print(script)
        return 0
    if not submit.keeper_running():
        print(f"[urgent] REFUSED: the lease keeper ({submit.KEEPER_UNIT}) is not running", file=sys.stderr)
        return 1
    if arguments.requeue_steady:
        for job_id in steady:
            subprocess.run(["scontrol", "requeue", job_id], check=True)
            print(f"[urgent] requeued steady job {job_id} (its run so far is lost)")
    held = [
        job.job_id for job in queue
        if job.waiting and job.partition != gpu_jobs.Partition.URGENT and job.job_id not in victims
        and (gpu_jobs.read_sidecar(job.job_id) or {}).get("gpu")
    ]
    if victims and held:
        subprocess.run(["scontrol", "hold", ",".join(held)], check=True)
        gpu_jobs.write_json(gpu_jobs.STATE_DIR / RESTORE_ORDER, {"victims": {v: False for v in victims}, "held": held})
        print(f"[urgent] held {held} until the pre-empted jobs {victims} run again")
    path = gpu_jobs.STATE_DIR / "scripts" / f"{arguments.name}-{stamp}.sbatch"
    path.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    done = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        print(f"[urgent] REFUSED by slurm: {done.stderr.strip()}", file=sys.stderr)
        if victims and held:
            subprocess.run(["scontrol", "release", ",".join(held)], check=False)
        return 1
    job_id = done.stdout.strip().split(";")[0]
    gpu_jobs.update_sidecar(
        job_id, project=arguments.project, what=arguments.what, job_class="urgent", name=arguments.name,
        gpu=arguments.gpu, mem=mem, command=arguments.command, log=str(log), script=str(path), after=[],
        preempts=victims, submitted_at=int(time.time()),
    )
    print(f"[urgent] job {job_id} submitted; log {log}")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
