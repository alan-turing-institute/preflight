"""Send a job to another site's slurm (`submit.py --site NAME`): sync its code and data, then submit there.

Nothing is queued unless every step before it passed: (1) the import closure of `--entry` is built by
`dgx/assert_code_parity.py` (an import that lives in the repo but sits on no search root is a refusal); (2) the closure's
files and the named `--data` folders are copied with rsync by checksum, so what the other site already holds is skipped;
(3) the closure's checksums are read back from there and must equal this clone's; (4) Python there, from the job's own
folder, must load every first-party name of the closure from the synced folder (`remote_import_check.py`); (5) that
site's own `submit.py` is run over ssh, so its job wrapper, log and jobs page handle the job, and with `--check` it runs
the check there (D47), on the files and data the job will see. A symlink in a `--data` folder that would dangle there
is refused in (1). This site does not mirror the job (D2: the other site's own tools show it; `status.py --site NAME`
runs its status command).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import site_config

PARITY = Path(__file__).resolve().parent.parent / "dgx" / "assert_code_parity.py"
IMPORT_CHECK = Path(__file__).resolve().parent / "remote_import_check.py"
#: Python names the interpreter by its file name: `python`, `python3`, `python3.12`.
INTERPRETER_PREFIX = "python"
#: How many symlinks one refusal lists.
LISTED_LINKS = 3
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


def _run(argv: list[str], what: str, stdin: str | None = None) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, check=False, input=stdin)
    if done.returncode != 0:
        raise RemoteRefused(f"{what} failed (exit {done.returncode}): {(done.stderr or done.stdout).strip()[:800]}")
    return done.stdout


def pick(site: site_config.Site, name: str) -> site_config.Remote:
    try:
        return site.remotes[name]
    except KeyError:
        known = ", ".join(sorted(site.remotes)) or "none"
        raise RemoteRefused(f"--site {name}: this site's file names no such remote (known: {known}; add [remote.{name}])") from None


@dataclass(frozen=True)
class Plan:
    """What `plan` decided to ship; paths are relative to `repo` unless said otherwise."""

    repo: Path
    files: list[str]  # the code closure
    data: list[str]  # --data folders inside the repo
    outside: list[tuple[Path, str]]  # --data folders outside it, as (parent, name)
    roots: list[str]  # import roots inside the repo, in the order Python meets them
    names: list[str]  # top-level names the closure's files import as


def stray_links(folder: Path) -> list[str]:
    """Symlinks under `folder` that would not resolve inside it once it is copied, as `path -> text`: an absolute one
    (its text is copied, and it names a path of this machine) or a relative one that leaves the folder. rsync copies a
    link, never its target."""
    base = folder.resolve()
    found = []
    for here, dirs, names in os.walk(folder):
        for name in (*dirs, *names):
            path = Path(here, name)
            if path.is_symlink() and (os.path.isabs(os.readlink(path)) or not path.resolve().is_relative_to(base)):
                found.append(f"{path.relative_to(folder)} -> {os.readlink(path)}")
    return found


def refuse_stray_links(folder: Path, given: str) -> None:
    """Refuse a `--data` folder holding a link that would dangle there (2026-11-15: a job died in seconds on a link
    copied without its 487 MB target). Copying each target in would copy it once per link and put it where the project
    never looks for it, so the target is to be named instead, where it belongs."""
    stray = stray_links(folder)
    if stray:
        more = f"; and {len(stray) - LISTED_LINKS} more" if len(stray) > LISTED_LINKS else ""
        raise RemoteRefused(
            f"--data {given}: a symlink would dangle on the other site, because only the link is copied, never what it "
            f"points at ({'; '.join(stray[:LISTED_LINKS])}{more}). Name the folder it points at with --data (with "
            "--data-base and --data-env when that is outside the repo), or put a real copy of the file in the folder"
        )


def plan(arguments: argparse.Namespace, remote: site_config.Remote) -> Plan:
    """What to ship, from the arguments. Refuses what cannot be shipped."""
    for dest, flag in LOCAL_ONLY.items():
        if getattr(arguments, dest, None):
            raise RemoteRefused(f"{flag} belongs to the site that owns the queue; it cannot go with --site")
    if not arguments.entry or arguments.repo is None:
        raise RemoteRefused("--site needs --repo and --entry: the entry script's imports decide which files are sent")
    repo = arguments.repo.resolve()
    parity = _parity()
    entry = repo / arguments.entry
    found = parity.plan_comparison(argparse.Namespace(entry=entry, search=arguments.search, allow_diff=[], also=[]), repo)
    if found is None:
        raise RemoteRefused("the code closure could not be built (see [parity] above)")
    roots = list(dict.fromkeys(root for root in parity.search_roots(entry.resolve(), repo, arguments.search) if root.is_relative_to(repo)))
    data, outside, folders = [], [], []
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
        folders.append((path, given))
    if outside and not remote.data_root:
        raise RemoteRefused(f"--data outside {repo} needs data_root in [remote.{arguments.site}] of the site file")
    if arguments.data_env and not remote.data_root:
        raise RemoteRefused(f"--data-env needs data_root in [remote.{arguments.site}] of the site file")
    for path, given in folders:  # the walk reads every folder, so after the refusals that cost nothing
        refuse_stray_links(path, given)
    return Plan(repo, found[1], data, outside, [str(root.relative_to(repo)) for root in roots],
                parity.top_level_names(found[1], repo, roots))


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


def interpreter(arguments: argparse.Namespace) -> str:
    """The Python the job runs there, for the import check: `--remote-python`, else the command's own first word when it
    is one, relative to the job's folder as the command is. Anything else (`uv run`, a shell script) is refused,
    because guessing another interpreter would check the wrong Python."""
    named = getattr(arguments, "remote_python", None)
    if named:
        return named
    first = arguments.command[0]
    if Path(first).name.startswith(INTERPRETER_PREFIX):
        return first
    raise RemoteRefused(
        f"the command starts with {first!r}, not a Python: --remote-python PATH names the Python the job runs there "
        "(relative to --workdir, e.g. .venv/bin/python), so the import check uses the same one"
    )


def check_imports(remote: site_config.Remote, inside: str, plan_: Plan, python: str, to_root: str) -> None:
    """Refuse when Python there, from the job's folder, would load a first-party name from outside the synced folder
    (`remote_import_check.py`, piped so that site cannot hold a stale copy). `to_root` is the project folder as seen
    from `inside`."""
    spec = json.dumps({"project": to_root, "roots": plan_.roots, "names": plan_.names})
    _run([*SSH, remote.host, f"cd {shlex.quote(inside)} && {python} - {shlex.quote(spec)}"],
         f"the import check on {remote.host}", IMPORT_CHECK.read_text(encoding="utf-8"))


def require_remote_checks(remote: site_config.Remote) -> None:
    """Refuse when the other site's `submit.py` has no `--check`: it would reject the flag, or (an older script) never
    run the check, and a check that does not run must not look as if it passed."""
    out = _run([*SSH, remote.host, f"{remote.submit_command} --help"], f"{remote.submit_command} --help on {remote.host}")
    if "--check" not in out:
        raise RemoteRefused(
            f"--check / --no-check: {remote.submit_command} on {remote.host} predates submit-time checks (D47) and would "
            "not run it. Update the preflight checkout on that site to a commit that has `--check` (6950607 added it; "
            "7d2c902 fixed its signal and start-up handling), then submit again"
        )


def forwarded(arguments: argparse.Namespace, env: list[str] = ()) -> list[str]:
    """The options the other site's `submit.py` takes, from the parsed ones (the sync options stay here).
    `env` (the job's `env NAME=path` words) prefixes a check so it sees the same data path as the job."""
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
    if getattr(arguments, "size", None):  # sized there, from that site's own runs (D48)
        out += ["--size", *arguments.size]
    if getattr(arguments, "check", None):  # run there (D47), after the copy, on the files and data the job will see
        out += ["--check", f"{shlex.join(env)} {arguments.check}" if env else arguments.check]
    if getattr(arguments, "no_check", None):
        out += ["--no-check", arguments.no_check]
    return out


def submit(arguments: argparse.Namespace, site: site_config.Site) -> str:
    remote = pick(site, arguments.site)
    shipped = plan(arguments, remote)
    repo, files, data, outside = shipped.repo, shipped.files, shipped.data, shipped.outside
    root = project_root(remote, arguments.project)
    workdir = arguments.workdir.resolve()
    if not workdir.is_relative_to(repo):
        raise RemoteRefused(f"--workdir {workdir} is outside --repo {repo}: the command runs from the same place there")
    inside = f"{root}/{workdir.relative_to(repo)}"
    python = interpreter(arguments)
    env = ["env", f"{arguments.data_env}={remote.data_root}"] if arguments.data_env else []
    line = " ".join(shlex.quote(part) for part in [remote.submit_command, *forwarded(arguments, env), "--", *env, *arguments.command])
    command = f"cd {shlex.quote(inside)} && {line}"
    print(f"[remote] {len(files)} code file(s) and {len(data)} data folder(s) -> {remote.host}:{root}"
          + (f"; {len(outside)} outside it -> {remote.data_root}" if outside else ""))
    if arguments.dry_run:
        return f"[remote] dry run: nothing copied or queued; would run on {remote.host}: {command}"
    if getattr(arguments, "check", None) or getattr(arguments, "no_check", None):
        require_remote_checks(remote)  # before anything is copied
    changed = sync(remote, root, repo, files, data, outside)
    verify(remote, root, repo, files)
    print(f"[remote] copied {changed} changed item(s); code on {remote.host} matches this clone")
    check_imports(remote, inside, shipped, python, os.path.relpath(repo, workdir))
    print(f"[remote] Python on {remote.host} loads the code's {len(shipped.names)} top-level name(s) from {root}")
    return _run([*SSH, remote.host, command], f"{remote.submit_command} on {remote.host}").strip()


def status(site: site_config.Site, name: str) -> str:
    remote = pick(site, name)
    return _run([*SSH, remote.host, remote.status_command], f"{remote.status_command} on {remote.host}").rstrip()
