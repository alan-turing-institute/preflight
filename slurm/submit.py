#!/usr/bin/env python3
"""Submit one job to slurm: the one command every project uses for GPU work on this machine.

Design 3.3 (`design notes`).

    slurm/submit.py --project P --what 'one line' --class resume|restart [--mem 24G|auto] [--gpu|--gpus N] \
        [--entry script.py --repo DIR] [--after JOBID]... [--cpus N] [--name NAME] [--log FILE] \
        [--workdir DIR] [--wait-limit 1h] [--resume-ok] [--level low|normal|high] [--service] [--anywhere|--no-anywhere] -- <command...>

PART-TIME MACHINES (D25): a CPU-only `resume` job (no GPUs, not `--service`) is submitted to `cpu_fill,part_time`, so it
may start on a part-time machine (someone's own, lent to the cluster while idle) as well as on the main node, when a
part-time node passes the checks below. Only `resume`, because the health check requeues such a job whenever the
machine's owner needs the memory back, and the job then continues from its checkpoint elsewhere. Per node, in order:
size (the cores and memory it lends), ssh reachability, then the named paths (`portability.py`): the working folder on
storage both machines see, the program built for the node's architecture, every other named path shared or a system
file. A node that fails is left out with `--exclude`, and each decision is printed (`[submit] part_time NODE: included`
or `excluded: why`, with `--dry-run` too) and kept in the job's record. At launch the wrapper checks the same things
again on the node and requeues the job elsewhere if they no longer hold. `--no-anywhere` keeps the job off them.
`--anywhere` is the submitter's word that the job is fit for them: the path checks are skipped (the size check
is not), and it is refused if no part-time node can take the job. A node with `default_jobs = false` takes only such jobs.

SERVICE (D20, sites with `[service]`): `--service` runs a background service's CPU stand-in in the `service` partition
(`--mem` required). When cores are short the agent suspends CPU-only jobs for it within seconds (they keep their memory,
lose nothing and are resumed when it has gone); it never touches a GPU job, never waits for the card, and waits when
memory is short. Which jobs it may suspend is its owner's consent (D9): `--suspend J[,J]` only those, `--no-suspend`
none, neither any. With `--dry-run` the first line is `[submit] service room: {json}`: `starts` (`now`,
`after_suspending` or `waits`), `suspends` ([{job, user, name, what}]) and `why`. Only the site's service group and
admins may use it.

MEMORY: `--mem` is optional. Without it (or with `auto`) a job given `--name` asks for the highest peak of that name's
3 newest clean runs plus 10%; with no measured run (or no `--name`) a GPU job asks for the whole box, a CPU job for
slurm's default. After a run of the name was killed for memory (slurm's OUT_OF_MEMORY or the host's OOM guard) and no
clean run has ended since, it asks for at least 1.2 times what that run was given, and 105 MB more at least (D35, the
Vertical Pod Autoscaler's rule), and prints `[submit] memory: ...` saying so.

CORES (D35, site field `cpus_from_history`): a CPU-only job given `--name` and no `--cpus` is offered the cores its
3 newest completed runs of the same `project:class:name` used: the highest average (core time over wall time) times
1.25, rounded up, within the cores the job may have. The line `[submit] cores: ...` shows it; `recommend` (the
default) only prints it, `enforce` asks for it. A `--cpus` above twice it warns.

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
import cores
import gpu_jobs  # noqa: E402
import portability
import remote
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
#: --mem auto with no clean run: the largest peak of the crashed or pre-empted runs (a lower bound), plus this share.
UNCLEAN_MARGIN = 0.25
#: After a run killed for memory (D35): the next ask is at least this many times what it was given ...
OOM_BUMP_RATIO = 1.2
#: ... and at least this many slurm MB (2^20 bytes) more: the Vertical Pod Autoscaler's OOMBumpUpRatio and OOMMinBumpUp.
OOM_MIN_BUMP_MB = 100
#: --mem with no measured run, for a GPU job: slurm's "all of the memory on each node".
WHOLE_BOX = "0"
DEFAULT_CPUS = 4  # a CPU-only job, or any job on a site whose nodes have no GPUs
#: Cores from history (D35): the highest average in use over the newest completed runs, times this, rounded up.
CORES_MARGIN = 1.25
#: A hand-set --cpus more than this many times the cores from history warns (never a refusal, as for memory).
CORES_OVER_ASK = 2
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
    """The `--mem` to ask slurm for. A stated figure is used as given; none (or `auto`) is the highest peak of the 3
    newest clean runs of `name` plus `AUTO_MARGIN`, only for a job given `--name` (the default name, the command's
    first word such as `python`, is shared by unrelated jobs); with no clean run, the largest peak of its crashed or
    pre-empted runs plus `UNCLEAN_MARGIN`; with no measured run at all, the whole box for a GPU job (`0`: slurm grants
    every byte it hands out) and slurm's own default for a CPU job. The wrapper measures every run, so only a
    named job's first run takes the whole box.

    After a run of the name was killed for memory, with no clean run since (`gpu_jobs.memory_kill`), the ask is at
    least `OOM_BUMP_RATIO` times what that run was given and `OOM_MIN_BUMP_MB` more, never below the history's ask,
    and the whole box when that is more than any node gives jobs (D35)."""
    if mem not in (None, "auto"):
        return mem
    if name is None:
        return WHOLE_BOX if gpu else None
    history = None  # GiB
    for clean, margin in ((True, AUTO_MARGIN), (False, UNCLEAN_MARGIN)):
        if (peak := gpu_jobs.measured_peak_bytes(name, clean=clean)) is not None:
            history = math.ceil(peak * (1 + margin) / 1024**3)
            break
    if (kill := gpu_jobs.memory_kill(name)) is not None:
        said = f"run {kill.job_id} of {name!r} was killed by {kill.by}"
        if kill.asked_mb is None:
            warn(f"{said}, but what it was given is unknown (no --mem, and accounting does not say); sized from its "
                 "history instead")
        elif kill.asked_mb == 0:
            warn(f"{said} with the whole box given; it asks for the whole box again")
            return WHOLE_BOX
        else:
            bumped_mb = max(kill.asked_mb + OOM_MIN_BUMP_MB, kill.asked_mb * OOM_BUMP_RATIO)
            ask = max(math.ceil(bumped_mb / 1024), history or 0)
            most_mb = max((node.jobs_mb for node in gpu_jobs.load_site().nodes), default=None)
            said += f" with {_gb(kill.asked_mb)} given"
            if most_mb is not None and ask * 1024 > most_mb:
                warn(f"{said}; {OOM_BUMP_RATIO} times that is more than any node gives jobs, so it asks for the whole box")
                return WHOLE_BOX
            print(f"[submit] memory: {said}; asking {ask}G, at least {OOM_BUMP_RATIO} times that (D35)")
            return f"{ask}G"
    if history is not None:
        return f"{history}G"
    if gpu:
        warn(f"no measured run called {name!r}: it asks for the whole box and waits for every other job; "
             "pass --mem from a similar job's footprint (slurm/footprint.py NAME) to let it start beside them")
        return WHOLE_BOX
    return None


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
        warn(f"--mem {mem} ({_gb(asked_mb)}) is below the {peak / 1e9:.1f} GB {name!r} peaked at on one of its last "
             f"{gpu_jobs.HISTORY_RUNS} clean runs; the memory floor may kill the job")


def warn_device(arguments: argparse.Namespace) -> None:
    """C4: a CPU-only job told `--device cpu` while no GPU job runs or waits: the GPU is free and may finish it far sooner."""
    command = arguments.command
    on_cpu = "--device=cpu" in command or any(a == "--device" and b == "cpu" for a, b in itertools.pairwise(command))
    if arguments.gpus or not on_cpu:
        return
    queue = gpu_jobs.read_queue()
    if queue is None:
        return
    reserved = any(j.uses_card and j.state in (gpu_jobs.State.RUNNING, gpu_jobs.State.PENDING) for j in queue)
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
    requirements: tuple[tuple[str, str, str], ...] = (),
    shards: int = 0,
) -> str:
    """The job script: directives, then one line running the wrapper. `shards` > 0 asks for that many shards of a
    shared card instead of the GPU (the wrapper then tells it is a shard job from slurm's environment). `urgent.py` passes its own partition and no
    level (its partition goes first whatever the level). A job that may land in `part_time` runs the wrapper with
    `node_python` (the site's interpreter there), never the submitter's own, whose venv path that machine lacks; it is also handed `requirements` (what
    `portability.decide` checked) to check again when it starts there."""
    label = arguments.job_class
    level = getattr(arguments, "level", None) or gpu_jobs.Level.NORMAL
    partition = partition or gpu_jobs.partition_for(
        gpu_jobs.JobClass(label), arguments.gpus, service=getattr(arguments, "service", False)
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
    if shards:
        lines.append(f"#SBATCH --gres=shard:{shards}")
    elif gpus:
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
    if arguments.max_jobs != 1 and not shards:
        wrapper += ["--max-jobs", str(arguments.max_jobs)]
    if on_part_time and requirements:
        wrapper += ["--require", json.dumps(requirements)]
    if manifest:
        wrapper += ["--manifest", str(manifest), "--repo", str(arguments.repo)]
    lines.append(f"exec {shlex.join([*wrapper, '--', *arguments.command])}")
    return "\n".join(lines) + "\n"


def place_on_part_time(arguments: argparse.Namespace, site: site_config.Site, mem: str | None) -> portability.Decision | None:
    """The part-time nodes this job may use and why not the others (printed, one line each); None when the job is not
    offered to any (not an eligible job, or the site has no node that takes unasked jobs, or `--mem` is not slurm's
    notation, which sbatch reports). `--anywhere` is a vouching: size is still checked, the paths are not."""
    if not gpu_jobs.part_time_eligible(gpu_jobs.JobClass(arguments.job_class), arguments.gpus, service=arguments.service,
                                       no_anywhere=arguments.no_anywhere):
        return None
    nodes = [n for n in site.part_time_nodes if arguments.anywhere or n.default_jobs]
    try:
        memory_mb = None if mem is None else gpu_jobs.mem_mb(mem)
    except (ValueError, IndexError):
        return None
    if not nodes:
        return None
    decision = portability.decide(
        nodes, cpus=arguments.cpus, memory_mb=memory_mb, check_paths=not arguments.anywhere, python=site.node_python,
        needs=[*portability.named_paths(arguments.command, arguments.workdir, arguments.repo),
               portability.Need(portability.Role.TOOLS, str(HERE.resolve()))],  # the script runs the wrapper from here
    )
    print("\n".join(v.line() for v in decision.verdicts))
    if arguments.anywhere:
        print("[submit] part_time: the submitter vouches for this job, so the path checks were skipped (size was checked)")
        if not decision.included:
            why = "; ".join(f"{v.node}: {v.reason}" for v in decision.verdicts)
            msg = f"no part-time machine can take this job ({why}); ask for what the machine lends, or leave the request out"
            raise Refused(msg)
    return decision


def resolve_packing(arguments: argparse.Namespace, named: str | None, site: site_config.Site) -> gpu_jobs.Packing:
    """Share the card with other copies of a named job, as far as its measurements allow (D27-D30). Only one card of a
    site whose card has shards is ever shared (`--gpus` above 1 never is), and only a job given `--name`: the default
    name is shared by unrelated jobs. Prints the decision, and a warning when the job has outgrown its measurement."""
    node = next((n for n in site.nodes if n.gpus), None)
    if arguments.gpus != 1 or node is None or not node.gpus[0].shards:
        return gpu_jobs.Packing()
    asked, arguments.max_jobs = arguments.max_jobs, 1  # the level is settled here: the wrapper's declared-sharing flag is for sites without shards
    if named is None:
        if asked and asked > 1:
            warn(f"--max-jobs {asked} needs --name (the default name is shared by unrelated jobs): the whole card")
        return gpu_jobs.Packing()
    record = gpu_jobs.measured_sharing(named)
    peak = gpu_jobs.measured_peak_bytes(named) or gpu_jobs.measured_peak_bytes(named, clean=False)
    packing = gpu_jobs.plan_packing(
        gpu=node.gpus[0], node=node, record=record, peak_bytes=peak, fraction=site.sharing_memory_fraction,
        cpus_asked=arguments.cpus, explicit_level=asked, margin=AUTO_MARGIN,
    )
    if packing.line:
        print(f"[submit] sharing {named}: {packing.line}")
    measured = int((record or {}).get("peak_bytes") or 0)
    if peak and measured and peak > measured * (1 + AUTO_MARGIN):
        warn(f"{named!r} peaked at {peak / 1e9:.1f} GB on its last clean run, above the {measured / 1e9:.1f} GB its "
             "sharing level was measured at: re-measure (slurm/sharing.py measure)")
    return packing


def resolve_cpus(asked: int | None, gpus: int, site: site_config.Site, shared_cores: int = 0,
                 speeds: cores.CoreSpeeds | None = None) -> int:
    """`--cpus`, else for a GPU job the cores the site holds for its GPUs (`cores_per_gpu`) or, unset, its GPUs'
    share of the GPU node's cores, else DEFAULT_CPUS. On a host whose cores differ (`speeds`), a whole-card GPU job
    is raised to the fewest cores that always include a fast one.

    A GPU job given the old flat 4 cores was CPU-bound: the same inference timing ran 2-14x slower than with all 20
    (slurm jobs 254 and 258, 2026-10-19), while the other cores sat idle behind the one GPU.
    """
    cpus = asked if asked is not None else _default_cpus(gpus, site, shared_cores)
    node = next((n for n in site.nodes if n.gpus), None)
    if not gpus or shared_cores or node is None or speeds is None or len(speeds.khz) != node.cpus:
        return cpus  # no GPU, a shard job (its share of the held cores is fixed), or speeds that are not this node's
    floor = min(speeds.fast_core_floor(), node.cpus)
    if cpus < floor:
        warn(f"a GPU job with {cpus} cores can run every one of them on this host's {node.cpus - floor + 1} slower cores, "
             f"which starves the GPU (measured 2026-11-04: 4 slow cores kept it 45% busy, 12 cores 86-92%); using {floor}, "
             "the fewest that always include a fast core")
        return floor
    return cpus


def _default_cpus(gpus: int, site: site_config.Site, shared_cores: int) -> int:
    if shared_cores:
        return shared_cores  # a shard job: its share of the cores held for the card
    node = next((n for n in site.nodes if n.gpus), None)
    if not gpus or node is None:
        return min(DEFAULT_CPUS, site.cpu_job_cap or DEFAULT_CPUS)  # a CPU-only job's default fits under the cap
    if node.cores_per_gpu:
        return node.cores_per_gpu * min(gpus, len(node.gpus))
    return max(1, node.cpus * min(gpus, len(node.gpus)) // len(node.gpus))


def cores_from_history(arguments: argparse.Namespace, named: str | None, hand_set: bool,
                       site: site_config.Site) -> dict[str, object] | None:
    """D35: the cores a CPU-only job given `--name` would ask for from its own completed runs (`gpu_jobs.measured_cores`
    of `project:class:name`): the highest average in use times `CORES_MARGIN`, rounded up, from 1 to the cores the job
    may have. Prints the line; under `enforce` a job without `--cpus` asks for them; a `--cpus` more than
    `CORES_OVER_ASK` times them warns. Returns what is kept in the job's record (for comparing proposals with use
    before `enforce`), None when nothing was proposed (`off`, a GPU job, no `--name`, no completed run)."""
    mode = site.cpus_from_history
    if mode is site_config.CoresFromHistory.OFF or arguments.gpus or named is None:
        return None
    measured = gpu_jobs.measured_cores(f"{arguments.project}:{arguments.job_class}:{named}")
    if measured is None:
        return None
    average, runs = measured
    cap = site.service.cores if arguments.service else site.cpu_job_cap or max((n.cpus for n in site.nodes), default=None)
    proposal = max(1, math.ceil(average * CORES_MARGIN))
    proposal = min(proposal, cap) if cap else proposal
    asking = arguments.cpus
    applied = mode is site_config.CoresFromHistory.ENFORCE and not hand_set
    if applied:
        arguments.cpus = proposal
    said = f"[submit] cores: measured {average:.1f} avg over {runs} completed runs; "
    print(said + (f"asking {proposal} (was {asking})" if applied else
                  f"would ask {proposal} (asking {asking} now{', set by --cpus' if hand_set else ''})"))
    if hand_set and asking > CORES_OVER_ASK * proposal:
        warn(f"--cpus {asking} is more than {CORES_OVER_ASK} times the {proposal} cores {named!r} would ask for from its "
             f"last {runs} completed runs ({average:.1f} in use on average at most); the idle cores are held from other jobs")
    return {"mode": str(mode), "average": round(average, 2), "runs": runs, "proposal": proposal, "asked": arguments.cpus}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", required=True)
    p.add_argument("--what", required=True, help="one line: what the job is")
    p.add_argument("--class", dest="job_class", required=True, choices=[c.value for c in gpu_jobs.JobClass])
    p.add_argument("--mem", default=None,
                   help="e.g. 24G; default auto: the highest peak of the 3 newest clean runs of --name plus 10%%, "
                        "else the whole box (GPU)")
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
        default=None,
        help="GPU jobs allowed on the card, this one included: the level to share it at. Default: the level "
             "`sharing.py measure` recorded for --name, else 1 (the whole card)",
    )
    p.add_argument("--service", action="store_true",
                   help="a background service's CPU stand-in: starts at once by suspending CPU-only jobs (D20)")
    consent = p.add_mutually_exclusive_group()
    consent.add_argument("--suspend", default=None, metavar="J[,J]",
                         help="with --service: the only jobs it may suspend (their owner's consent, D9)")
    consent.add_argument("--no-suspend", action="store_true", help="with --service: suspend nothing; wait instead")
    anywhere = p.add_mutually_exclusive_group()
    anywhere.add_argument("--anywhere", action="store_true",
                          help="vouch that this CPU-only resume job may run on a part-time machine: skips the path checks")
    anywhere.add_argument("--no-anywhere", action="store_true",
                          help="keep this job off the part-time machines, which a portable CPU-only resume job otherwise uses")
    p.add_argument("--site", default=None, metavar="NAME",
                   help="send the job to another site's slurm ([remote.NAME] in the site file): syncs --entry's import "
                        "closure and --data by checksum, then submits there with that site's own submit.py")
    p.add_argument("--data", action="append", default=[], metavar="DIR",
                   help="with --site: a folder inside --repo to sync by checksum (repeatable)")
    p.add_argument("--data-base", type=Path, default=None, metavar="DIR",
                   help="with --site: the folder an outside-the-repo --data path is taken relative to, so "
                        "DIR/gpe/x lands at data_root/gpe/x there (default: its own name)")
    p.add_argument("--data-env", default=None, metavar="NAME",
                   help="with --site: run the command with NAME set to the site's data_root (e.g. AUTOCAST_DATASETS)")
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
    if (arguments.data or arguments.data_base or arguments.data_env) and not arguments.site:
        msg = "--data, --data-base and --data-env go with --site (they name data to sync to that site)"
        raise Refused(msg)
    if arguments.site:
        arguments.name = arguments.name or Path(arguments.command[0]).name
        try:
            return remote.submit(arguments, gpu_jobs.load_site())
        except remote.RemoteRefused as error:
            raise Refused(str(error)) from None
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
    packing = resolve_packing(arguments, named, site)
    arguments.max_jobs = arguments.max_jobs or 1  # from here on: the level asked for, 1 when none
    arguments.cpus = resolve_cpus(arguments.cpus, arguments.gpus, site, packing.cores, cores.CoreSpeeds.read())
    if arguments.anywhere:
        if not gpu_jobs.part_time_eligible(gpu_jobs.JobClass(arguments.job_class), arguments.gpus, service=arguments.service):
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
    history_cores = cores_from_history(arguments, named, named_cpus is not None, site)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = arguments.log or gpu_jobs.STATE_DIR / "logs" / f"{arguments.name}-{stamp}.log"
    gated = bool(arguments.entry) and site.dgx_gate
    manifest = gpu_jobs.STATE_DIR / "manifests" / f"{arguments.name}-{stamp}.json" if gated else None
    consent = may_suspend(arguments) if arguments.service else None
    placed = place_on_part_time(arguments, site, mem)
    included = bool(placed and placed.included)
    partition = gpu_jobs.partition_for(
        gpu_jobs.JobClass(arguments.job_class), arguments.gpus, service=arguments.service, part_time=included
    )
    barred = (f"--exclude={','.join(placed.excluded)}",) if included and placed.excluded else ()
    script = sbatch_script(
        arguments, mem=mem, log=log, manifest=manifest, node_python=site.node_python, partition=partition, extra=barred,
        requirements=placed.requirements if included else (), shards=packing.shards,
    )
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
        **({"part_time": {v.node: v.reason or "included" for v in placed.verdicts}} if placed else {}),
        **({"cores_from_history": history_cores} if history_cores else {}),
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
