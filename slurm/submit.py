#!/usr/bin/env python3
"""Submit one job to slurm: the one command every project uses for GPU work on this machine.

Design 3.3 (`design notes`).

    slurm/submit.py --project P --what 'one line' --class resume|restart [--mem 24G|auto] [--gpu|--gpus N] \
        [--entry script.py --repo DIR] [--after JOBID]... [--cpus N] [--name NAME] [--log FILE] \
        [--workdir DIR] [--wait-limit 1h] [--resume-ok] [--level low|normal|high] [--service] [--anywhere] -- <command...>

ANYWHERE: `--anywhere` lets a CPU-only `resume` job start on a part-time machine (someone's own, lent to the cluster
while idle) as well as on the main node: it is submitted to `cpu,part_time`. Only with `--class resume` and no GPUs,
because the health check requeues such a job whenever the machine's owner needs the memory back, and the job then
continues from its checkpoint elsewhere. Refused on a site with no part-time node.

SERVICE (D20, sites with `[service]`): `--service` runs a background service's CPU stand-in in the `service` partition
(`--mem` required). When cores are short the agent suspends CPU-only jobs for it within seconds (they keep their memory,
lose nothing and are resumed when it has gone); it never touches a GPU job, never waits for the card, and waits when
memory is short. Which jobs it may suspend is its owner's consent (D9): `--suspend J[,J]` only those, `--no-suspend`
none, neither any. With `--dry-run` the first line is `[submit] service room: {json}`: `starts` (`now`,
`after_suspending` or `waits`), `suspends` ([{job, user, name, what}]) and `why`. Only the site's service group and
admins may use it.

MEMORY: `--mem` is optional. Without it (or with `auto`) a job given `--name` asks for that name's newest measured
run plus 10%; with no measured run (or no `--name`) a GPU job asks for the whole box, a CPU job for slurm's default.

LEVEL (D18): where the job waits among the others, `normal` by default; `high` goes ahead of normal and low, `low`
behind them. Open to everyone. It orders the waiting line only: a running job is never stopped for a higher level.
The job is submitted with that level as its QOS; it waits behind every placed job until the agent gives it its
place, at most 5 s later.

REFUSES: a class outside
the closed set; a `resume` job without `--resume-ok` (the caller's statement that running the command
again continues from its latest checkpoint); `--after` on a job slurm no longer knows; `--gpu` together with `--gpus`; any submission
while the lease keeper is not running (checked only when the site file declares a lease).

AT SUBMISSION, before anything is queued (a fault must surface before the queue wait, not after it):
with `--entry` (and a site with `dgx_gate`), the DGX gate in check-only mode against this box (code parity, the entry script's
import, the host memory guard), its GPU leg deferred to launch, where the job wrapper runs it; the
manifest it writes is copied per job, so two queued jobs with one entry script cannot overwrite each
other's. Then `audit.py`, and the job's sidecar record.

Every job gets `--requeue` (a `steady` job is protected by its partition's `PreemptMode=OFF`, not by
`--no-requeue`, which would also stop recovery after a node failure) and `--open-mode=append` (a
requeued job keeps its first leg's log). The job name is `P:class:NAME`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pwd
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit  # noqa: E402 - path set just above
import gpu_jobs  # noqa: E402
import site_config

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "dgx" / "dgx_submit.sh"
KEEPER_UNIT = "slurm-gpu-lease-keeper"
#: The keeper rewrites its state every 5 s (`gpu_lease_keeper.py --interval`); a heartbeat older than this means it is down.
KEEPER_STALE_S = 60
#: Longest `--comment` we write: our choice (it keeps `squeue` output readable), not a slurm limit.
COMMENT_MAX = 200
#: --mem auto: the measured peak, plus this share, rounded up to whole GiB.
AUTO_MARGIN = 0.10
#: --mem with no measured run, for a GPU job: slurm's "all of the memory on each node".
WHOLE_BOX = "0"
DEFAULT_CPUS = 4  # a CPU-only job, or any job on a site whose nodes have no GPUs
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)([smh]?)$")
_UNIT_S = {"": 1, "s": 1, "m": 60, "h": 3600}


class Refused(Exception):
    """A submission this tool will not make; the message says why."""


def seconds(text: str) -> float:
    match = _DURATION.match(text.strip())
    if not match:
        msg = f"not a duration: {text!r} (use 90, 30m or 2h)"
        raise Refused(msg)
    return float(match.group(1)) * _UNIT_S[match.group(2)]


def resolve_mem(mem: str | None, name: str | None, gpu: bool) -> str | None:
    """The `--mem` to ask slurm for. A stated figure is used as given; none (or `auto`) is the newest measured run of
    `name` plus `AUTO_MARGIN`, only for a job given `--name` (the default name, the command's first word such as
    `python`, is shared by unrelated jobs); with no measured run, the whole box for a GPU job (`0`: slurm grants
    every byte it hands out) and slurm's own default for a CPU job. The wrapper measures every run, so only a
    named job's first run takes the whole box."""
    if mem not in (None, "auto"):
        return mem
    peak = None if name is None else gpu_jobs.measured_peak_bytes(name)
    if peak is not None:
        return f"{math.ceil(peak * (1 + AUTO_MARGIN) / 1024**3)}G"
    return WHOLE_BOX if gpu else None


def warn(text: str) -> None:
    print(f"[submit] warning: {text}", file=sys.stderr)


def _gb(megabytes: float) -> str:
    return f"{megabytes * 1024**2 / 1e9:.1f} GB"


def warn_memory(mem: str | None, name: str | None, gpu: bool) -> None:
    """C1: a job named `name` with a measured footprint that states `--mem` by hand warns when it is below the peak (the
    memory floor may kill it) or `0` (the whole box: it waits for every other job). Warnings only, never a refusal."""
    peak = None if name is None or mem in (None, "auto") else gpu_jobs.measured_peak_bytes(name)
    if peak is None:
        return
    try:
        asked_mb = gpu_jobs.mem_mb(mem)
    except (ValueError, IndexError):
        return  # not slurm notation: sbatch reports it
    if asked_mb == 0:
        auto = resolve_mem(None, name, gpu)
        warn(f"--mem 0 asks for the whole box, so the job waits for every other job; {name!r} peaked at "
             f"{peak / 1e9:.1f} GB, and --mem auto would ask for {_gb(gpu_jobs.mem_mb(auto))}")
    elif asked_mb * 1024**2 < peak:
        warn(f"--mem {mem} ({_gb(asked_mb)}) is below the {peak / 1e9:.1f} GB {name!r} peaked at on its last clean "
             "run; the memory floor may kill the job")


def warn_device(arguments: argparse.Namespace) -> None:
    """C4: a CPU-only job told `--device cpu` while no GPU job runs or waits: the GPU is free and may finish it far sooner."""
    command = arguments.command
    on_cpu = "--device=cpu" in command or any(a == "--device" and b == "cpu" for a, b in itertools.pairwise(command))
    if arguments.gpus or not on_cpu:
        return
    queue = gpu_jobs.read_queue()
    if queue is None:
        return
    reserved = any(j.gpus > 0 and j.state in (gpu_jobs.State.RUNNING, gpu_jobs.State.PENDING) for j in queue)
    if not reserved:
        warn("the command says --device cpu but no GPU job is running or waiting: the GPU is free, and the work may "
             "finish far sooner there (resubmit with --gpu)")


def comment(what: str) -> str:
    """The `--what` text as a one-line slurm comment, cut to `COMMENT_MAX` characters."""
    return what.replace("\r", " ").replace("\n", " ")[:COMMENT_MAX]


def keeper_running(now: float | None = None) -> bool:
    """Whether the keeper's heartbeat is fresh. Read from its state file, not `systemctl --user`: the keeper is a user
    unit of the site's keeper user, which another submitter's systemd cannot see (found in the container cluster, F1)."""
    state = gpu_jobs.read_json(gpu_jobs.keeper_state_path()) or {}
    updated = state.get("updated_at")
    if not isinstance(updated, (int, float)):
        return False
    return (time.time() if now is None else now) - updated < KEEPER_STALE_S


def require_keeper(site: site_config.Site) -> None:
    """Refuse when the site has a lease and the keeper's heartbeat is stale."""
    if site.lease and not keeper_running():
        msg = f"the lease keeper ({KEEPER_UNIT}) is not running; start it before submitting"
        raise Refused(msg)


def known_to_slurm(job_id: str) -> bool:
    done = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True, check=False)
    return done.returncode == 0


def run_gate(repo: Path, entry: str, command: list[str], manifest_copy: Path, searches: list[str] = ()) -> None:
    env = dict(os.environ, AUTOCAST_DGX_GATE_CHECK_ONLY="1", AUTOCAST_DGX_HOST="localhost", AUTOCAST_DGX_ROOT=str(repo))
    done = subprocess.run(
        [
            "bash", str(GATE), "--repo", str(repo), "--entry", entry,
            *(arg for root in searches for arg in ("--search", root)),
            "--gpu-leg-at-launch", "slurm: the job wrapper checks the card when the job starts",
            "--", *command,
        ],
        env=env,
        check=False,
    )
    if done.returncode != 0:
        msg = f"the DGX gate refused (exit {done.returncode}); nothing was queued"
        raise Refused(msg)
    manifest = repo / "outputs" / "dgx_gate" / Path(entry).stem / "manifest.json"
    manifest_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, manifest_copy)


def sbatch_script(
    arguments: argparse.Namespace,
    *,
    mem: str | None,
    log: Path,
    manifest: Path | None,
    partition: str | None = None,
    extra: tuple[str, ...] = (),
    needs_lease: bool | None = None,
    node_python: str | None = None,
) -> str:
    """The job script: directives, then one line running the wrapper. `urgent.py` passes its own partition and no
    level (its partition goes first whatever the level). A job that may land in `part_time` runs the wrapper with
    `node_python` (the site's interpreter there), never the submitter's own, whose venv path that machine lacks."""
    label = arguments.job_class
    level = getattr(arguments, "level", None) or gpu_jobs.Level.NORMAL
    partition = partition or gpu_jobs.partition_for(
        gpu_jobs.JobClass(label), arguments.gpus, service=getattr(arguments, "service", False),
        anywhere=getattr(arguments, "anywhere", False),
    )
    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --job-name={arguments.project}:{label}:{arguments.name}",
        f"#SBATCH --cpus-per-task={arguments.cpus}",
        f"#SBATCH --qos={level}",
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
        f"#SBATCH --output={log}",
        f"#SBATCH --chdir={arguments.workdir}",
    ]
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    gpus = arguments.gpus or int(arguments.gpu)
    if gpus:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if getattr(arguments, "what", None):
        lines.append(f"#SBATCH --comment={shlex.quote(comment(arguments.what))}")
    if arguments.after:
        lines.append(f"#SBATCH --dependency=afterok:{':'.join(arguments.after)}")
    lines += [f"#SBATCH {directive}" for directive in extra]
    on_part_time = gpu_jobs.Partition.PART_TIME in partition.split(",")
    wrapper = [node_python if on_part_time and node_python else sys.executable, str(HERE / "job_wrapper.py"), "--wait-limit", str(seconds(arguments.wait_limit))]
    if arguments.gpu if needs_lease is None else needs_lease:
        wrapper.append("--needs-lease")
    if arguments.gpu:
        wrapper.append("--needs-card")
    if arguments.max_jobs != 1:
        wrapper += ["--max-jobs", str(arguments.max_jobs)]
    if manifest:
        wrapper += ["--manifest", str(manifest), "--repo", str(arguments.repo)]
    lines.append(f"exec {shlex.join([*wrapper, '--', *arguments.command])}")
    return "\n".join(lines) + "\n"


def resolve_cpus(asked: int | None, gpus: int, site: site_config.Site) -> int:
    """`--cpus`, else for a GPU job the cores the site holds for its GPUs (`cores_per_gpu`) or, unset, its GPUs'
    share of the GPU node's cores, else DEFAULT_CPUS.

    A GPU job given the old flat 4 cores was CPU-bound: the same inference timing ran 2-14x slower than with all 20
    (slurm jobs 254 and 258, 2026-10-19), while the other cores sat idle behind the one GPU.
    """
    if asked is not None:
        return asked
    node = next((n for n in site.nodes if n.gpus), None)
    if not gpus or node is None:
        return min(DEFAULT_CPUS, site.cpu_job_cap or DEFAULT_CPUS)  # a CPU-only job's default fits under the cap
    if node.cores_per_gpu:
        return node.cores_per_gpu * min(gpus, len(node.gpus))
    return max(1, node.cpus * min(gpus, len(node.gpus)) // len(node.gpus))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", required=True)
    p.add_argument("--what", required=True, help="one line: what the job is")
    p.add_argument("--class", dest="job_class", required=True, choices=[c.value for c in gpu_jobs.JobClass])
    p.add_argument("--mem", default=None,
                   help="e.g. 24G; default auto: the last measured run of --name plus 10%%, else the whole box (GPU)")
    gpu = p.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", action="store_true", help="one GPU (same as --gpus 1)")
    gpu.add_argument("--gpus", type=int, default=None, metavar="N", help="N GPUs, N >= 1")
    p.add_argument("--entry", default=None, help="the Python entry script, relative to --repo (runs the gate)")
    p.add_argument("--repo", type=Path, default=None)
    p.add_argument("--search", action="append", default=[], metavar="DIR",
                   help="an import root the gate's code-parity leg also walks, relative to --repo (repeatable)")
    p.add_argument("--after", action="append", default=[], metavar="JOBID")
    p.add_argument("--cpus", type=int, default=None,
                   help=f"default: a GPU job gets the cores the site holds per GPU (cores_per_gpu), else its GPUs' share "
                        f"of the GPU node's cores; other jobs {DEFAULT_CPUS}")
    p.add_argument("--name", default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--workdir", type=Path, default=Path.cwd())
    p.add_argument("--wait-limit", default="1h", help="how long the job waits in-job for the card")
    p.add_argument("--resume-ok", action="store_true", help="running the command again resumes it")
    p.add_argument("--level", choices=[level.value for level in gpu_jobs.Level], default=gpu_jobs.Level.NORMAL.value,
                   help="where it waits among the others (default normal); never stops a running job")
    p.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="GPU jobs allowed on the card, this one included; above 1 only against a measured sharing result",
    )
    p.add_argument("--service", action="store_true",
                   help="a background service's CPU stand-in: starts at once by suspending CPU-only jobs (D20)")
    consent = p.add_mutually_exclusive_group()
    consent.add_argument("--suspend", default=None, metavar="J[,J]",
                         help="with --service: the only jobs it may suspend (their owner's consent, D9)")
    consent.add_argument("--no-suspend", action="store_true", help="with --service: suspend nothing; wait instead")
    p.add_argument("--anywhere", action="store_true",
                   help="a CPU-only resume job may also start on a part-time machine (cpu,part_time)")
    p.add_argument("--dry-run", action="store_true", help="print the sbatch script and submit nothing")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def submit(argv: list[str]) -> str:
    arguments = parser().parse_args(argv)
    arguments.command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not arguments.command:
        msg = "no command after --"
        raise Refused(msg)
    if ":" in arguments.project:
        # the job name is project:class:name, and tools read the class back from it (QueuedJob.job_class): a ':' here
        # could make a restart job read as one that resumes, and urgent.py would then requeue it and lose its run
        msg = f"--project {arguments.project!r}: no ':' in a project name"
        raise Refused(msg)
    if arguments.gpus is not None and arguments.gpus < 1:
        msg = f"--gpus {arguments.gpus}: ask for at least one GPU, or leave it out for a CPU job"
        raise Refused(msg)
    arguments.gpus = 1 if arguments.gpu else arguments.gpus or 0
    arguments.gpu = arguments.gpus > 0
    site = gpu_jobs.load_site()
    named = arguments.name
    arguments.name = arguments.name or Path(arguments.command[0]).name
    if arguments.job_class == gpu_jobs.JobClass.RESUME and not arguments.resume_ok:
        msg = "a resume job needs --resume-ok: running its command again must continue from its latest checkpoint"
        raise Refused(msg)
    if arguments.entry and arguments.repo is None:
        msg = "--entry needs --repo (the project checkout)"
        raise Refused(msg)
    seconds(arguments.wait_limit)
    mem = resolve_mem(arguments.mem, named, arguments.gpu)
    warn_memory(arguments.mem, named, arguments.gpu)
    named_cpus = arguments.cpus
    arguments.cpus = resolve_cpus(arguments.cpus, arguments.gpus, site)
    if arguments.anywhere:
        if arguments.job_class != gpu_jobs.JobClass.RESUME or arguments.gpus or arguments.service:
            msg = ("--anywhere needs --class resume with no GPUs and no --service: the owner of a part-time machine "
                   "can take it back at any moment, which requeues the job, and only a job that continues from its "
                   "checkpoint loses nothing")
            raise Refused(msg)
        if not site.part_time_nodes:
            msg = "--anywhere: this site has no part-time node (part_time = true on a node in the site file)"
            raise Refused(msg)
    if arguments.service:
        if arguments.gpus:
            msg = "--service runs a CPU stand-in: no --gpu or --gpus"
            raise Refused(msg)
        if site.service is None:
            msg = "--service: this site has no service partition (no [service] table in the site file)"
            raise Refused(msg)
        if arguments.mem in (None, "auto"):
            msg = "--service needs --mem: whether it fits beside what runs is judged by it (suspending frees no memory)"
            raise Refused(msg)
        if not site_config.may_submit_service(site, pwd.getpwuid(os.getuid()).pw_name):
            msg = (f"--service: only members of {site.service.group} may use it (slurm would hold the job forever); "
                   "an admin adds you with: sudo slurm/install/install.py service add USER")
            raise Refused(msg)
        if named_cpus is None:  # the default fits under the cap, as a CPU-only job's does
            arguments.cpus = min(DEFAULT_CPUS, site.service.cores)
        if arguments.cpus > site.service.cores:
            msg = f"--cpus {arguments.cpus}: a service job may use at most {site.service.cores} cores here"
            raise Refused(msg)
    elif arguments.suspend is not None or arguments.no_suspend:
        msg = "--suspend and --no-suspend go with --service only"
        raise Refused(msg)
    elif not arguments.gpus and site.cpu_job_cap and arguments.cpus > site.cpu_job_cap:
        msg = (f"--cpus {arguments.cpus}: a job without GPUs may use at most {site.cpu_job_cap} cores here; the rest "
               "are held for GPU jobs (cores_per_gpu in the site file)")
        raise Refused(msg)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = arguments.log or gpu_jobs.STATE_DIR / "logs" / f"{arguments.name}-{stamp}.log"
    gated = bool(arguments.entry) and site.dgx_gate
    manifest = gpu_jobs.STATE_DIR / "manifests" / f"{arguments.name}-{stamp}.json" if gated else None
    consent = may_suspend(arguments) if arguments.service else None
    script = sbatch_script(arguments, mem=mem, log=log, manifest=manifest, node_python=site.node_python)
    warn_device(arguments)
    if arguments.dry_run:
        if arguments.service:
            print(f"[submit] service room: {json.dumps(service_room(site, arguments.cpus, gpu_jobs.mem_mb(mem), consent))}")
        return script
    require_keeper(site)
    for job_id in arguments.after:
        if not known_to_slurm(job_id):
            msg = f"--after {job_id}: slurm no longer knows that job (finished over a day ago, or never existed)"
            raise Refused(msg)
    if gated:
        run_gate(arguments.repo.resolve(), arguments.entry, arguments.command, manifest, arguments.search)
    print("\n".join(audit.report(audit.unregistered())))
    path = gpu_jobs.STATE_DIR / "scripts" / f"{arguments.name}-{stamp}.sbatch"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = gpu_jobs.write_new(path, script)
    except FileExistsError as error:
        raise Refused(f"the batch script {error.strerror}") from None
    done = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        msg = f"sbatch refused: {done.stderr.strip()}"
        raise Refused(msg)
    job_id = done.stdout.strip().split(";")[0]
    gpu_jobs.record_queued_job(
        job_id,
        "submit",
        project=arguments.project,
        what=arguments.what,
        job_class=arguments.job_class,
        name=arguments.name,
        gpu=arguments.gpu,
        gpus=arguments.gpus,
        mem=mem,
        command=arguments.command,
        log=str(log),
        script=str(path),
        entry=arguments.entry,
        manifest=str(manifest) if manifest else None,
        after=arguments.after,
        max_jobs=arguments.max_jobs,
        submitted_at=int(time.time()),
        **({"may_suspend": None if consent is None else sorted(consent)} if arguments.service else {}),
    )
    partition = gpu_jobs.partition_for(
        gpu_jobs.JobClass(arguments.job_class), arguments.gpus, service=arguments.service, anywhere=arguments.anywhere
    )
    pauses = {
        gpu_jobs.ANYWHERE_PARTITIONS: "requeued for urgent work or when a part-time machine's owner needs it back; "
                                      "resumed from its checkpoint",
        gpu_jobs.Partition.NORMAL: "pre-empted and resumed from its checkpoint",
        gpu_jobs.Partition.STEADY: "never pre-empted",
        gpu_jobs.Partition.CPU_FILL: "fills idle cores; requeued for GPU or urgent work and resumed from its checkpoint",
        gpu_jobs.Partition.SERVICE: "the agent suspends CPU-only jobs it may to make room; requeued for urgent work",
    }.get(partition) or (
        "requeued for urgent work and resumed from its checkpoint" if arguments.job_class == gpu_jobs.JobClass.RESUME
        else "never pre-empted"
    )
    print(f"[submit] job {job_id} ({partition}: {pauses}); log {log}")
    return job_id


def may_suspend(arguments: argparse.Namespace) -> frozenset[str] | None:
    """The jobs a service job may suspend: `--suspend`'s, none for `--no-suspend`, any (None) without either."""
    if arguments.no_suspend:
        return frozenset()
    if arguments.suspend is None:
        return None
    ids = [j.strip() for j in arguments.suspend.split(",") if j.strip()]
    if not ids or not all(j.isdigit() for j in ids):
        msg = f"--suspend {arguments.suspend!r}: job ids, comma-separated"
        raise Refused(msg)
    return frozenset(ids)


#: The id `service_room` gives the job it is asked about: past any real one, so it is judged last among equals.
_ASKING = "999999999"


def service_room(site: site_config.Site, cpus: int, memory_mb: int, consent: frozenset[str] | None) -> dict[str, object]:
    """What a `--service` job submitted now would meet (D20; the agent decides with the same `gpu_jobs.plan_room`).
    The queue moves on: a submission made later may meet another, which `consent` bounds."""
    queue = gpu_jobs.read_queue()
    if queue is None:
        return {"starts": "waits", "suspends": [], "why": "slurm does not answer"}
    node, service = site.nodes[0], site.service
    asking = gpu_jobs.QueuedJob(_ASKING, gpu_jobs.Partition.SERVICE, gpu_jobs.State.PENDING, gpu_jobs.Reason.RESOURCES,
                                "asking", priority=2**32, cpus=cpus, mem_mb=memory_mb)
    room = gpu_jobs.plan_room([*queue, asking], set(), node.cpus, node.real_memory_mb, service.cores, {_ASKING: consent})
    by_id = {job.job_id: job for job in queue}
    if room.helping == _ASKING:
        return {"starts": "after_suspending", "why": "cores are short; these cpu jobs are suspended until it has gone",
                "suspends": [{"job": j, "user": by_id[j].user, "name": by_id[j].name,
                              "what": str((gpu_jobs.read_sidecar(j, by_id[j].user) or {}).get("what") or "")}
                             for j in room.suspend]}
    free_cores, free_mb = gpu_jobs.free_room(queue, node.cpus, node.real_memory_mb)
    service_free = service.cores - sum(j.cpus for j in queue
                                       if j.partition == gpu_jobs.Partition.SERVICE and j.state in gpu_jobs.ON_CORES)
    if free_mb is None:
        why = "a job holds the whole machine's memory (or one of unknown size runs); suspending frees no memory"
    elif memory_mb > free_mb:
        why = f"memory: it needs {memory_mb} MB and {free_mb} MB are free; suspending frees no memory"
    elif cpus > service_free:
        why = f"other service jobs use {service.cores - service_free} of the service partition's {service.cores} cores"
    elif cpus > free_cores:
        why = f"cores: {free_cores} free, and the cpu jobs it may suspend cannot free the other {cpus - free_cores}"
    else:
        return {"starts": "now", "suspends": [], "why": "it fits beside what runs"}
    return {"starts": "waits", "suspends": [], "why": why}


def main() -> int:
    try:
        out = submit(sys.argv[1:])
    except Refused as refusal:
        print(f"[submit] REFUSED: {refusal}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
