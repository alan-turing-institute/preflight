"""Send a job to another site's slurm (`submit.py --site NAME`): sync its code and data, then submit there.

Nothing is queued unless every step before it passed: (1) the import closure of `--entry` is built by
`dgx/assert_code_parity.py` (an import that lives in the repo but sits on no search root is a refusal); (2) the closure's
files and the named `--data` folders are copied with rsync by checksum, so what the other site already holds is skipped;
(3) the closure's checksums are read back from there and must equal this clone's; (4) that site's own `submit.py` is run
over ssh, so its job wrapper, log and jobs page handle the job. This site does not mirror the job (D2: the other site's
own tools show it; `status.py --site NAME` runs its status command).
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
from pathlib import Path

import site_config

PARITY = Path(__file__).resolve().parent.parent / "dgx" / "assert_code_parity.py"
SSH = ["ssh", "-o", "BatchMode=yes"]
#: `submit.py` options that mean something only on the site that owns the queue, so they are refused with --site.
LOCAL_ONLY = {"service": "--service", "anywhere": "--anywhere", "suspend": "--suspend", "no_suspend": "--no-suspend", "log": "--log"}


class RemoteRefused(Exception):
    """A step before the job was queued failed; nothing was queued."""


def _parity():
    spec = importlib.util.spec_from_file_location("assert_code_parity", PARITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(argv: list[str], what: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RemoteRefused(f"{what} failed (exit {done.returncode}): {(done.stderr or done.stdout).strip()[:400]}")
    return done.stdout


def pick(site: site_config.Site, name: str) -> site_config.Remote:
    try:
        return site.remotes[name]
    except KeyError:
        known = ", ".join(sorted(site.remotes)) or "none"
        raise RemoteRefused(f"--site {name}: this site's file names no such remote (known: {known}; add [remote.{name}])") from None


def plan(arguments: argparse.Namespace, remote: site_config.Remote) -> tuple[Path, list[str], list[str], list[tuple[Path, str]]]:
    """(repo, closure files, data folders inside the repo, data folders outside it as (parent, name)); the first two lists
    are relative to the repo. Refuses what cannot be shipped."""
    for dest, flag in LOCAL_ONLY.items():
        if getattr(arguments, dest, None):
            raise RemoteRefused(f"{flag} belongs to the site that owns the queue; it cannot go with --site")
    if not arguments.entry or arguments.repo is None:
        raise RemoteRefused("--site needs --repo and --entry: the entry script's imports decide which files are sent")
    repo = arguments.repo.resolve()
    parity = _parity()
    found = parity.plan_comparison(
        argparse.Namespace(entry=repo / arguments.entry, search=arguments.search, allow_diff=[], also=[]), repo
    )
    if found is None:
        raise RemoteRefused("the code closure could not be built (see [parity] above)")
    data, outside = [], []
    base = arguments.data_base.resolve() if arguments.data_base else None
    for given in arguments.data:
        path = (repo / given).resolve()
        if not path.is_dir() or path == repo:
            raise RemoteRefused(f"--data {given}: not a folder")
        if path.is_relative_to(repo):
            data.append(str(path.relative_to(repo)))
        elif base is not None and path.is_relative_to(base) and path != base:
            outside.append((base, str(path.relative_to(base))))
        elif base is None:
            outside.append((path.parent, path.name))
        else:
            raise RemoteRefused(f"--data {given}: outside --data-base {base}")
    if outside and not remote.data_root:
        raise RemoteRefused(f"--data outside {repo} needs data_root in [remote.{arguments.site}] of the site file")
    if arguments.data_env and not remote.data_root:
        raise RemoteRefused(f"--data-env needs data_root in [remote.{arguments.site}] of the site file")
    return repo, found[1], data, outside


def project_root(remote: site_config.Remote, project: str) -> str:
    return f"{remote.root.rstrip('/')}/{project}"


def sync(
    remote: site_config.Remote, root: str, repo: Path, files: list[str], data: list[str], outside: list[tuple[Path, str]] = ()
) -> int:
    """Copy the files and data folders under `root` there by checksum; returns how many items rsync changed."""
    _run([*SSH, remote.host, f"mkdir -p {shlex.quote(root)}"], f"creating {root} on {remote.host}")
    sources = [f"{repo}/./{name}" for name in files] + [f"{repo}/./{name}/" for name in data]
    out = _run(
        ["rsync", "-a", "--checksum", "--relative", "--itemize-changes", "-e", " ".join(SSH), *sources, f"{remote.host}:{root}/"],
        "rsync",
    )
    changed = sum(1 for line in out.splitlines() if line.startswith((">f", "cd")))
    if outside:
        _run([*SSH, remote.host, f"mkdir -p {shlex.quote(remote.data_root)}"], f"creating {remote.data_root} on {remote.host}")
        for parent, name in outside:  # a folder already there, byte for byte, costs a read on each side and no transfer
            above = shlex.quote(str(Path(remote.data_root) / Path(name).parent))
            _run([*SSH, remote.host, f"mkdir -p {above}"], f"creating {above} on {remote.host}")
            out = _run(  # --no-implied-dirs: the shared folders above the dataset keep their owner, group and mode there
                ["rsync", "-a", "--checksum", "--relative", "--no-implied-dirs", "--itemize-changes", "-e", " ".join(SSH),
                 f"{parent}/./{name}/",
                 f"{remote.host}:{remote.data_root}/"],
                "rsync of data",
            )
            changed += sum(1 for line in out.splitlines() if line.startswith((">f", "cd")))
    return changed


def verify(remote: site_config.Remote, root: str, repo: Path, files: list[str]) -> None:
    parity = _parity()
    there = parity.remote_digests(remote.host, root, files)
    bad = [name for name in files if there.get(name) != parity.digest(repo / name)]
    if bad:
        raise RemoteRefused(f"after the copy, {remote.host} still differs on: {', '.join(bad)}")


def forwarded(arguments: argparse.Namespace) -> list[str]:
    """The options the other site's `submit.py` takes, from the parsed ones (the sync options stay here)."""
    out = ["--project", arguments.project, "--what", arguments.what, "--class", arguments.job_class,
           "--name", arguments.name, "--wait-limit", arguments.wait_limit, "--level", str(arguments.level)]
    if arguments.max_jobs:  # unset: that site picks the level itself (its measured one, else the whole card)
        out += ["--max-jobs", str(arguments.max_jobs)]
    if arguments.mem:
        out += ["--mem", arguments.mem]
    if arguments.gpus:
        out += ["--gpus", str(arguments.gpus)]
    if arguments.cpus:
        out += ["--cpus", str(arguments.cpus)]
    for job in arguments.after:  # ids on the other site, where the job queues
        out += ["--after", job]
    if arguments.resume_ok:
        out.append("--resume-ok")
    return out


def submit(arguments: argparse.Namespace, site: site_config.Site) -> str:
    remote = pick(site, arguments.site)
    repo, files, data, outside = plan(arguments, remote)
    root = project_root(remote, arguments.project)
    workdir = arguments.workdir.resolve()
    if not workdir.is_relative_to(repo):
        raise RemoteRefused(f"--workdir {workdir} is outside --repo {repo}: the command runs from the same place there")
    inside = f"{root}/{workdir.relative_to(repo)}"
    job = ["env", f"{arguments.data_env}={remote.data_root}", *arguments.command] if arguments.data_env else arguments.command
    line = " ".join(shlex.quote(part) for part in [remote.submit_command, *forwarded(arguments), "--", *job])
    command = f"cd {shlex.quote(inside)} && {line}"
    print(f"[remote] {len(files)} code file(s) and {len(data)} data folder(s) -> {remote.host}:{root}"
          + (f"; {len(outside)} outside it -> {remote.data_root}" if outside else ""))
    if arguments.dry_run:
        return f"[remote] dry run: nothing copied or queued; would run on {remote.host}: {command}"
    changed = sync(remote, root, repo, files, data, outside)
    verify(remote, root, repo, files)
    print(f"[remote] copied {changed} changed item(s); code on {remote.host} matches this clone")
    return _run([*SSH, remote.host, command], f"{remote.submit_command} on {remote.host}").strip()


def status(site: site_config.Site, name: str) -> str:
    remote = pick(site, name)
    return _run([*SSH, remote.host, remote.status_command], f"{remote.status_command} on {remote.host}").rstrip()
