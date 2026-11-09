#!/usr/bin/env python3
"""Run urgent work now: show what is in its way and what a pause loses, then pre-empt and submit.

Design 3.6 (`design notes`).

    slurm/urgent.py --project P --what '...' --time 2:00:00 [--mem M|auto | --whole-box] [--gpu | --gpus N] \
        [--entry script.py --repo DIR [--search DIR]...] [--wait] [--requeue-steady] [--dry-run] -- <command...>

1. THE PLAN, printed before acting, classifies everything in the urgent job's way:
   pausable at once (`normal` jobs: they lose at most the time since their last checkpoint; service-a),
   pausable with a delay (service-b, at its next job boundary), wait-only (a `steady` job, another urgent
   job, any other lease holder such as service-c's queue, unregistered GPU work).
2. With a wait-only blocker it stops and submits nothing, unless `--wait` (accept the wait) or, for a
   `steady` job, `--requeue-steady` (requeue it, losing its progress so far, which the plan states). The
   requeue happens only once slurm has accepted the urgent job: a refused submission must not cost a run.
3. Otherwise it submits to `urgent` with the required `--time`. Default size `--whole-box`: the GPU, all
   memory and all cores. Slurm pre-empts and requeues the `normal` jobs in the way; their dependents wait;
   the services step aside because an urgent job is keeper demand, with or without `--gpu`.
   `--mem auto` asks for the highest peak of the job name's 3 newest clean runs plus margin, raised after a run killed
   for memory (`submit.resolve_mem`), the whole box when there is none. `--entry` (with `--repo`, `--search`) runs the DGX gate as `submit.py` does, before anything is held or
   submitted: a refusal changes nothing, and the job wrapper re-checks the code when the job starts.
4. ORDER AFTERWARDS: a requeued job cannot start for the requeue wait, so an urgent job shorter than that
   would hand the card to another queued job first. Other pending tier-1 GPU jobs are held and recorded in
   `requests/restore_order-<ns>.json`; the agent releases them once every pre-empted job is running again or has left.
"""

from __future__ import annotations

import argparse
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit  # noqa: E402 - path set just above
import cores  # noqa: E402
import gpu_jobs  # noqa: E402
import jobs_view  # noqa: E402
import site_config  # noqa: E402
import submit  # noqa: E402
from gpu_jobs import mem_mb  # noqa: E402

#: slurm's own delay from an urgent submission to its start, measured in the trial (build notes).
SLURM_START_S = (31, 46)
ADMIN_REMEDY = "sudo slurm/install/install.py admin add USER"


def elapsed(job_id: str) -> str:
    done = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%M"], capture_output=True, text=True, check=False)
    return done.stdout.strip() or "?"


def free_on_nodes() -> tuple[int, int] | None:
    """(memory MB, cores) slurm has not allocated on the nodes an urgent job can start on (in the `urgent` partition and
    not down or draining: a lent machine that is off is not room), or None if slurm does not answer or none can."""
    total = gpu_jobs.capacity(partition=gpu_jobs.Partition.URGENT)
    if not total.get("nodes"):
        return None
    return total["mem_mb"] - total["mem_used_mb"], total["cpus"] - total["cpus_used"]


def free_fast_cores(queue: list[gpu_jobs.QueuedJob], node: site_config.Node) -> int:
    """The fast cores (D46) of `node` no running job holds: what a job bound to them can start on."""
    fast = f"{node.name}:{node.fast_socket}:"
    held = sum(1 for job in queue if job.state == gpu_jobs.State.RUNNING for core in job.cores if core.startswith(fast))
    return node.fast_cores - held


def _asked_mb(job: gpu_jobs.QueuedJob) -> int:
    """The memory a job holds (slurm's request, else its sidecar's); 0 when unknown, the whole box or unreadable, which
    counts as freeing nothing, so a CPU job is requeued rather than the urgent job left waiting."""
    return jobs_view.requested_mem_mb(job, gpu_jobs.read_sidecar(job.job_id, job.user) or {}) or 0


def _describe(job: gpu_jobs.QueuedJob) -> str:
    record = gpu_jobs.read_sidecar(job.job_id, job.user) or {}
    return f"{job.job_id} {job.name} ({record.get('what', 'no sidecar')}), running {elapsed(job.job_id)}"


def plan(
    queue: list[gpu_jobs.QueuedJob], *, gpu: bool, whole_box: bool, leases: list[tuple[str, bool]],
    unregistered: list[audit.Stray], services: dict[str, str], memory_short: bool = False,
    cores_short: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """(lines to print, normal victims, steady blockers, other wait-only blockers).

    `memory_short` / `cores_short`: the urgent job's memory / cores do not fit in what slurm has left, so every
    running job is in its way (slurm picks which `normal` ones to pre-empt; a job without GPUs needs cores from a
    GPU job whenever the GPU jobs and the `cpu` partition hold them all). `services`: service name -> yield text, the site's
    `[lease] services`. A GPU urgent job puts every running `normal` job holding GPUs in the way, on any node.
    """
    lines, victims, steady, wait_only = [], [], [], []
    for job in queue:
        in_way = whole_box or memory_short or cores_short or (gpu and job.uses_card)
        if not in_way or not (job.active or job.partition == gpu_jobs.Partition.URGENT):
            continue
        if job.partition == gpu_jobs.Partition.PART_TIME:
            continue  # on another machine: it holds none of the cores, memory or card this urgent job needs
        what = _describe(job)
        if job.partition in {gpu_jobs.Partition.NORMAL, gpu_jobs.Partition.CPU_FILL}:
            victims.append(job.job_id)
            lines.append(f"[urgent] PAUSABLE  {what}: may be pre-empted (slurm picks as many as the urgent job needs); "
                         "loses at most the time since its last checkpoint")
        elif job.partition == gpu_jobs.Partition.SERVICE:
            victims.append(job.job_id)
            lines.append(f"[urgent] PAUSABLE  {what}: a service's CPU stand-in; may be requeued by slurm")
        elif job.partition == gpu_jobs.Partition.CPU and not job.restarts:  # slurm never pre-empts `cpu`: urgent.py does
            victims.append(job.job_id)
            lines.append(f"[urgent] PAUSABLE  {what}: requeued once the urgent job is accepted"
                         + ("" if whole_box else ", if pre-empting the normal jobs does not free enough room")
                         + "; loses at most the time since its last checkpoint")
        elif job.restarts:
            steady.append(job.job_id)
            lines.append(f"[urgent] WAIT-ONLY {what}: cannot resume; requeueing it loses all of that run")
        else:
            wait_only.append(job.job_id)
            lines.append(f"[urgent] WAIT-ONLY {what}: another urgent job; this one waits behind it")
    requeued = [job.job_id for job in queue if job.job_id in victims and job.partition == gpu_jobs.Partition.NORMAL]
    if requeued:  # a requeued `normal` job restarts at a higher tier than `cpu_fill`, so it takes cores back from its jobs,
        # including one that starts in the gap, which no plan made now can see: the warning does not depend on what runs now
        running = [job.job_id for job in queue
                   if job.active and job.partition == gpu_jobs.Partition.CPU_FILL and job.job_id not in victims]
        lines.append(f"[urgent] KNOCK-ON  normal job(s) {','.join(requeued)} it pre-empts restart later at a higher tier than "
                     "cpu_fill and may pre-empt cpu_fill jobs for their cores, including ones that start meanwhile"
                     + (f" (running now, not otherwise touched: {','.join(running)})" if running else ""))
    for project, waiting in leases:
        if project == gpu_jobs.LEASE_PROJECT:
            continue
        if project in services:
            lines.append(f"[urgent] PAUSABLE  service {project}: {services[project]}")
        else:
            wait_only.append(f"lease:{project}")
            state = "waiting for the card" if waiting else "holds the card"
            lines.append(f"[urgent] WAIT-ONLY {project} {state} and does not yield to a request")
    for stray in unregistered:
        wait_only.append(f"pid:{stray.pid}")
        lines.append(f"[urgent] WAIT-ONLY unregistered GPU work, pid {stray.pid} ({stray.mib:.0f} MiB): {stray.command[:120]}"
                     + (f" ({stray.why})" if stray.why else ""))
    if not lines:
        lines.append("[urgent] nothing in the way")
    return lines, victims, steady, wait_only


def other_leases() -> list[tuple[str, bool]]:
    found = []
    for path in gpu_jobs.LEASE_DIR.glob("*.json") if gpu_jobs.LEASE_DIR else []:
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
    size.add_argument("--mem", default=None,
                      help="memory it needs; `auto`: the last measured run of --name plus 10%%, else the whole box; "
                           "default: the whole box")
    size.add_argument("--whole-box", action="store_true")
    count = p.add_mutually_exclusive_group()
    count.add_argument("--gpu", action="store_true", help="one GPU (same as --gpus 1)")
    count.add_argument("--gpus", type=int, default=None, metavar="N", help="N GPUs, N >= 1")
    p.add_argument("--wait", action="store_true", help="accept waiting behind what cannot be paused")
    p.add_argument("--requeue-steady", action="store_true", help="requeue steady jobs in the way (loses them)")
    p.add_argument("--entry", default=None, help="the Python entry script, relative to --repo (runs the gate)")
    p.add_argument("--repo", type=Path, default=None)
    p.add_argument("--search", action="append", default=[], metavar="DIR",
                   help="an import root the gate's code-parity leg also walks, relative to --repo (repeatable)")
    p.add_argument("--cpus", type=int, default=None, help="default: as submit.py (a GPU job gets the site's cores per GPU)")
    p.add_argument("--name", default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--workdir", type=Path, default=Path.cwd())
    p.add_argument("--wait-limit", default="1h")
    p.add_argument(
        "--max-jobs", type=int, default=1,
        help="GPU jobs allowed on the card, this one included; above 1 only against a measured sharing result",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and the job script; change nothing")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def run(argv: list[str]) -> int:
    arguments = parser().parse_args(argv)
    arguments.command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not arguments.command:
        print("[urgent] REFUSED: no command after --", file=sys.stderr)
        return 1
    if arguments.gpus is not None and arguments.gpus < 1:
        print("[urgent] REFUSED: --gpus must be an integer >= 1", file=sys.stderr)
        return 1
    if arguments.entry and arguments.repo is None:
        print("[urgent] REFUSED: --entry needs --repo (the project checkout)", file=sys.stderr)
        return 1
    site = gpu_jobs.load_site()
    if not site_config.is_admin(site, pwd.getpwuid(os.getuid()).pw_name):
        print(f"[urgent] REFUSED: only members of {site.admin_group} may submit urgent work; "
              f"an admin adds you with: {ADMIN_REMEDY}", file=sys.stderr)
        return 1
    services = site.lease.services if site.lease else {}
    named = arguments.name
    if arguments.mem == "auto":  # a job with no measured run (the whole box) is sized as with no --mem
        resolved = submit.resolve_mem(arguments.mem, named, arguments.gpu or arguments.gpus is not None)
        arguments.mem = None if resolved in (None, submit.WHOLE_BOX) else resolved
    whole_box = arguments.whole_box or arguments.mem is None
    arguments.gpu = arguments.gpu or arguments.gpus is not None or whole_box
    arguments.gpus = arguments.gpus or int(arguments.gpu)
    submit.warn_memory(arguments.mem, named, arguments.gpu)
    arguments.name = named or Path(arguments.command[0]).name
    arguments.cpus = submit.resolve_cpus(arguments.cpus, arguments.gpus, site, speeds=cores.CoreSpeeds.read())
    arguments.job_class, arguments.after = "urgent", []
    queue = gpu_jobs.read_queue()
    if queue is None:
        print("[urgent] REFUSED: slurm does not answer (controller down); see slurm/status.py", file=sys.stderr)
        return 1
    free = None if whole_box else free_on_nodes()
    bound = () if whole_box else submit.binding(arguments.cpus, arguments.gpus, 0, site, said="[urgent]")
    if bound and free is not None:  # held to the fast cores: the slow ones free are no room for it
        free = (free[0], min(free[1], free_fast_cores(queue, site_config.gpu_node(site))))
    memory_short = free is not None and mem_mb(arguments.mem) > free[0]
    cores_short = free is not None and arguments.cpus > free[1]
    lines, victims, steady, wait_only = plan(
        queue, gpu=arguments.gpu, whole_box=whole_box, leases=other_leases(), unregistered=audit.unregistered(),
        memory_short=memory_short, cores_short=cores_short, services=services,
    )
    # slurm pre-empts `normal` victims; a `cpu` one urgent.py requeues itself, and it stays out of the restore-order
    # request (whose held GPU jobs would otherwise wait until a requeued CPU job got its capped cores back)
    by_id = {job.job_id: job for job in queue}
    cpu_victims = [v for v in victims if by_id[v].partition == gpu_jobs.Partition.CPU]
    victims = [v for v in victims if v not in cpu_victims]
    if cpu_victims and free is not None:  # short: only if the normal ones cannot free enough
        enough_memory = not memory_short or sum(_asked_mb(by_id[v]) for v in victims) >= mem_mb(arguments.mem) - free[0]
        enough_cores = not cores_short or sum(by_id[v].cpus for v in victims) >= arguments.cpus - free[1]
        if enough_memory and enough_cores:
            cpu_victims = []
    if free is not None:
        print(f"[urgent] memory: needs {mem_mb(arguments.mem)} MB, slurm has {free[0]} MB unallocated"
              + ("; jobs will be pre-empted to make room" if memory_short else "; it fits beside what runs"))
        print(f"[urgent] cores: needs {arguments.cpus}{' fast' if bound else ''}, slurm has {free[1]} unallocated"
              + ("; jobs will be pre-empted to make room" if cores_short else "; they fit beside what runs"))
    print("\n".join(map(gpu_jobs.printable, lines)))  # descriptions and lease names are what other users wrote
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
    extra = [f"--time={arguments.time}"] + (["--exclusive"] if whole_box else []) + list(bound)
    mem = "0" if whole_box else arguments.mem  # --mem=0: all of the node's memory
    gated = bool(arguments.entry) and site.dgx_gate
    manifest = gpu_jobs.STATE_DIR / "manifests" / f"{arguments.name}-{stamp}.json" if gated else None
    script = submit.sbatch_script(
        arguments, mem=mem, log=log, manifest=manifest, partition=gpu_jobs.Partition.URGENT, extra=tuple(extra),
        needs_lease=True,
    )
    if arguments.dry_run:
        print(script)
        return 0
    try:
        submit.require_keeper(site)
        if manifest:  # before anything is held or requeued: a refusal changes nothing
            submit.run_gate(arguments.repo.resolve(), arguments.entry, arguments.command, manifest, arguments.search)
    except submit.Refused as refused:
        print(f"[urgent] REFUSED: {refused}", file=sys.stderr)
        return 1
    # Written before anything is held or requeued, so a refusal here leaves every other job as it was.
    path = gpu_jobs.STATE_DIR / "scripts" / f"{arguments.name}-{stamp}.sbatch"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = gpu_jobs.write_new(path, script)
    except FileExistsError as error:
        print(f"[urgent] REFUSED: the batch script {error.strerror}", file=sys.stderr)
        return 1
    held = [
        job.job_id for job in queue
        if job.waiting and job.partition != gpu_jobs.Partition.URGENT and job.job_id not in victims
        and job.uses_card
    ]
    if victims and held and gpu_jobs.run_slurm(["scontrol", "hold", ",".join(held)]):
        # slurm holds the ids it can and refuses the rest (one that started meanwhile): release them all
        subprocess.run(["scontrol", "release", ",".join(held)], capture_output=True, check=False)
        print(f"[urgent] REFUSED, nothing submitted: slurm would not hold {held} (its message is above)", file=sys.stderr)
        return 1
    done = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        print(f"[urgent] REFUSED by slurm: {done.stderr.strip()}", file=sys.stderr)
        if victims and held:
            subprocess.run(["scontrol", "release", ",".join(held)], check=False)
        return 1
    job_id = done.stdout.strip().split(";")[0]
    if victims and held:  # written only now: a refused submission leaves no request and no hold in the jobs' history
        requests = gpu_jobs.STATE_DIR / "requests"
        try:  # first: the request is what releases the held jobs later
            requests.mkdir(parents=True, exist_ok=True)
            gpu_jobs.write_json(
                requests / f"restore_order-{time.time_ns()}.json", {"urgent": job_id, "victims": victims, "held": held}
            )
        except OSError as error:
            subprocess.run(["scontrol", "release", ",".join(held)], capture_output=True, check=False)
            print(f"[urgent] could not record when to release {held} ({error}): released them now, so they may "
                  f"start before the pre-empted jobs {victims}", file=sys.stderr)
        else:
            print(f"[urgent] held {held} until the pre-empted jobs {victims} run again")
            for held_id in held:
                try:
                    gpu_jobs.write_event(held_id, gpu_jobs.EventAction.URGENT_HOLD, urgent=f"{arguments.name}: {arguments.what}")
                except OSError as error:  # the history only: the request above is what matters
                    print(f"[urgent] warning: job {held_id}'s history misses this hold ({error})", file=sys.stderr)
    for victim in cpu_victims:  # a `cpu` job resumes but slurm never pre-empts it (PreemptMode=OFF): requeue it here
        if gpu_jobs.run_slurm(["scontrol", "requeue", victim]):
            print(f"[urgent] could not requeue job {victim} (slurm's message is above); job {job_id} waits behind it",
                  file=sys.stderr)
    if arguments.requeue_steady:  # only now: requeueing loses the run, which a refused submission must not cost
        for steady_id in steady:
            if gpu_jobs.run_slurm(["scontrol", "requeue", steady_id]):
                print(f"[urgent] could not requeue steady job {steady_id} (slurm's message is above); "
                      f"job {job_id} waits behind it", file=sys.stderr)
            else:
                print(f"[urgent] requeued steady job {steady_id} (its run so far is lost)")
    gpu_jobs.record_queued_job(
        job_id, "urgent", project=arguments.project, what=arguments.what, job_class="urgent", name=arguments.name,
        gpu=arguments.gpu, gpus=arguments.gpus, mem=mem, command=arguments.command, log=str(log), script=str(path), entry=arguments.entry,
        manifest=str(manifest) if manifest else None, after=[], preempts=victims, submitted_at=int(time.time()),
    )
    print(f"[urgent] job {job_id} submitted; log {log}")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
