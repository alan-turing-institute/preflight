#!/usr/bin/env python3
"""Submit one job to slurm: the one command every project uses for GPU work on this machine.

Design 3.3 (`design notes`).

    slurm/submit.py --project P --what 'one line' --class resume|restart [--mem 24G|auto] [--gpu|--gpus N] \
        [--entry script.py --repo DIR] [--after JOBID]... [--cpus N] [--name NAME] [--log FILE] \
        [--workdir DIR] [--wait-limit 1h] [--resume-ok] -- <command...>

MEMORY: `--mem` is optional. Without it (or with `auto`) a job given `--name` asks for that name's newest measured
run plus 10%; with no measured run (or no `--name`) a GPU job asks for the whole box, a CPU job for slurm's default.

REFUSES: a class outside
the closed set; a `resume` job without `--resume-ok` (the caller's statement that running the command
again continues from its latest checkpoint); `--after` on a job slurm no longer knows; `--gpu` together with `--gpus`; any submission
while the lease keeper is not running (checked only when the site file declares a lease, or there is no site file).

AT SUBMISSION, before anything is queued (a fault must surface before the queue wait, not after it):
with `--entry` (and a site with `dgx_gate`, or no site file), the DGX gate in check-only mode against this box (code parity, the entry script's
import, the host memory guard), its GPU leg deferred to launch, where the job wrapper runs it; the
manifest it writes is copied per job, so two queued jobs with one entry script cannot overwrite each
other's. Then `audit.py`, and the job's sidecar record.

Every job gets `--requeue` (a `steady` job is protected by its partition's `PreemptMode=OFF`, not by
`--no-requeue`, which would also stop recovery after a node failure) and `--open-mode=append` (a
requeued job keeps its first leg's log). The job name is `P:class:NAME`.
"""

from __future__ import annotations

import argparse
import math
import os
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
#: Longest `--comment` we write: our choice (it keeps `squeue` output readable), not a slurm limit.
COMMENT_MAX = 200
#: --mem auto: the measured peak, plus this share, rounded up to whole GiB.
AUTO_MARGIN = 0.10
#: --mem with no measured run, for a GPU job: slurm's "all of the memory on each node".
WHOLE_BOX = "0"
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


def comment(what: str) -> str:
    """The `--what` text as a one-line slurm comment, cut to `COMMENT_MAX` characters."""
    return what.replace("\r", " ").replace("\n", " ")[:COMMENT_MAX]


def keeper_running() -> bool:
    done = subprocess.run(["systemctl", "--user", "is-active", "-q", KEEPER_UNIT], check=False)
    return done.returncode == 0


def known_to_slurm(job_id: str) -> bool:
    done = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True, check=False)
    return done.returncode == 0


def run_gate(repo: Path, entry: str, command: list[str], manifest_copy: Path) -> None:
    env = dict(os.environ, AUTOCAST_DGX_GATE_CHECK_ONLY="1", AUTOCAST_DGX_HOST="localhost", AUTOCAST_DGX_ROOT=str(repo))
    done = subprocess.run(
        [
            "bash", str(GATE), "--repo", str(repo), "--entry", entry,
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
) -> str:
    """The job script: directives, then one line running the wrapper. `urgent.py` passes its own partition."""
    label = arguments.job_class
    partition = partition or gpu_jobs.PARTITION_OF[gpu_jobs.JobClass(label)]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --job-name={arguments.project}:{label}:{arguments.name}",
        f"#SBATCH --cpus-per-task={arguments.cpus}",
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
    wrapper = [sys.executable, str(HERE / "job_wrapper.py"), "--wait-limit", str(seconds(arguments.wait_limit))]
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
    p.add_argument("--after", action="append", default=[], metavar="JOBID")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--name", default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--workdir", type=Path, default=Path.cwd())
    p.add_argument("--wait-limit", default="1h", help="how long the job waits in-job for the card")
    p.add_argument("--resume-ok", action="store_true", help="running the command again resumes it")
    p.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="GPU jobs allowed on the card, this one included; above 1 only against a measured sharing result",
    )
    p.add_argument("--dry-run", action="store_true", help="print the sbatch script and submit nothing")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def submit(argv: list[str]) -> str:
    arguments = parser().parse_args(argv)
    arguments.command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not arguments.command:
        msg = "no command after --"
        raise Refused(msg)
    if arguments.gpus is not None and arguments.gpus < 1:
        msg = f"--gpus {arguments.gpus}: ask for at least one GPU, or leave it out for a CPU job"
        raise Refused(msg)
    arguments.gpus = 1 if arguments.gpu else arguments.gpus or 0
    arguments.gpu = arguments.gpus > 0
    site = site_config.load() if site_config.SITE_FILE.exists() else None
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
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = arguments.log or gpu_jobs.STATE_DIR / "logs" / f"{arguments.name}-{stamp}.log"
    gated = bool(arguments.entry) and (site is None or site.dgx_gate)
    manifest = gpu_jobs.STATE_DIR / "manifests" / f"{arguments.name}-{stamp}.json" if gated else None
    script = sbatch_script(arguments, mem=mem, log=log, manifest=manifest)
    if arguments.dry_run:
        return script
    if (site is None or site.lease) and not keeper_running():
        msg = f"the lease keeper ({KEEPER_UNIT}) is not running; start it before submitting"
        raise Refused(msg)
    for job_id in arguments.after:
        if not known_to_slurm(job_id):
            msg = f"--after {job_id}: slurm no longer knows that job (finished over a day ago, or never existed)"
            raise Refused(msg)
    if gated:
        run_gate(arguments.repo.resolve(), arguments.entry, arguments.command, manifest)
    print("\n".join(audit.report(audit.unregistered())))
    path = gpu_jobs.STATE_DIR / "scripts" / f"{arguments.name}-{stamp}.sbatch"
    path.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    done = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        msg = f"sbatch refused: {done.stderr.strip()}"
        raise Refused(msg)
    job_id = done.stdout.strip().split(";")[0]
    gpu_jobs.update_sidecar(
        job_id,
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
    )
    partition = gpu_jobs.PARTITION_OF[gpu_jobs.JobClass(arguments.job_class)]
    pauses = "pre-empted and resumed from its checkpoint" if partition == gpu_jobs.Partition.NORMAL else "never pre-empted"
    print(f"[submit] job {job_id} ({partition}: {pauses}); log {log}")
    return job_id


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
