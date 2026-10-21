"""Install, check and administer the GPU job layer from the site file.

    install.py detect                           # print a draft site file for this machine
    install.py render --out DIR                 # write every enabled component's files under DIR
    install.py check                            # exit 1 if the installed files differ from the rendered ones
    sudo install.py apply [--component NAME ...]
    sudo install.py user add|remove USER          # let USER submit (slurm accounting: the site's account)
    sudo install.py admin add|remove USER
    sudo install.py migrate-state --from DIR    # copy job records from an older state directory

`--site PATH` names the site file (default /etc/gpujobs/site.toml or $GPUJOBS_SITE); `--root PATH` is the filesystem
root the files are written to (default /, for tests).
"""

from __future__ import annotations

import argparse
import grp
import os
import platform
import pwd
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parents[1] / "machine" / "earlyoom"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import components  # noqa: E402 - paths set just above
import detect  # noqa: E402
import render  # noqa: E402
import site_config  # noqa: E402
from install_earlyoom import differing, install  # noqa: E402

MIGRATED = ("jobs", "footprints", "events")
OPERATOR = "Operator"
NO_LEVEL = "None"
Chown = Callable[[Path, str], None]
#: The kernel settings the shared, world-writable state directories rely on (`man 5 proc_sys_fs`): with any of
#: them 0, a file or link another user leaves under a predictable name (a job's log, a record) is written through.
KERNEL_PROTECTIONS = Path("/proc/sys/fs")
PROTECTIONS = ("protected_regular", "protected_symlinks", "protected_hardlinks")
#: The oldest slurm this build is written for (README, Requirements); Ubuntu 22.04's apt offers 21.08.
MIN_SLURM = (23, 11)
#: Where the build has been tested (README, Requirements): elsewhere an apply warns, and goes ahead.
TESTED_OS = ("ubuntu", "24.04")
TESTED_ARCH = "aarch64"
OS_RELEASE = Path("/etc/os-release")


class Action(StrEnum):
    ADD = "add"
    REMOVE = "remove"


def _chown(path: Path, owner: str) -> None:
    group = "root" if owner == "root" else grp.getgrgid(pwd.getpwnam(owner).pw_gid).gr_name
    shutil.chown(path, owner, group)


def _under(root: Path, target: str | Path) -> Path:
    return root / str(target).lstrip("/")


def _all(modules: list, site: site_config.Site, what: str) -> dict:
    out: dict = {}
    for module in modules:
        part = getattr(module, what, None)
        if part:
            out.update(part(site))
    return out


def _stage(files: dict[str, render.FileSpec], stage: Path) -> dict[str, str]:
    """Write contents under `stage` as 0, 1, ...; return the name -> relative target map the earlyoom helpers take."""
    targets = {}
    for i, (target, spec) in enumerate(files.items()):
        (stage / str(i)).write_text(spec.content)
        targets[str(i)] = target.lstrip("/")
    return targets


def write_files(root: Path, files: dict[str, render.FileSpec], chown: Chown | None) -> None:
    """Install `files` under `root` with their modes; owners too when `chown` is given."""
    for target, spec in files.items():
        dest = _under(root, target)
        created = [p for p in (dest.parent, *dest.parent.parents) if not p.exists()]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if chown and spec.owner != "root":
            for directory in created:  # a user's ~/.config/... must not end up root-owned
                chown(directory, spec.owner)
        if not dest.exists():  # a secret is never briefly world-readable
            os.close(os.open(dest, os.O_WRONLY | os.O_CREAT, spec.mode))
    with tempfile.TemporaryDirectory() as tmp:
        install(root, Path(tmp), _stage(files, Path(tmp)))
    for target, spec in files.items():
        dest = _under(root, target)
        dest.chmod(spec.mode)
        if chown:
            chown(dest, spec.owner)


def _run_all(commands: list) -> None:
    """Run each argv in turn, stopping at the first failure; a function in the list gives its argvs when reached."""
    for argv in commands:
        if callable(argv):
            _run_all(argv())
            continue
        done = components.run(argv)
        if done.returncode != 0:
            raise SystemExit(f"FAILED ({done.returncode}): {' '.join(argv)}\n{done.stderr or ''}".rstrip())


def _warn_portal_secrets(site: site_config.Site, modules: list) -> None:
    """Without root the portal's root-only secrets cannot be read, and the files built from them are left out."""
    if os.geteuid() != 0 and site.portal.mode is site_config.PortalMode.INSTALL and any(m.NAME == "portal" for m in modules):
        print("warning: not root, so files that need the portal's root-only secrets are left out "
              "(/etc/ood/config/ood_portal.yml); run with sudo for the full set", file=sys.stderr)


def cmd_render(site, args, modules) -> int:
    _warn_portal_secrets(site, modules)
    files = _all(modules, site, "files")
    write_files(args.out, files, None)
    print(f"rendered {len(files)} file(s) under {args.out}")
    return 0


def _wrong_mode_or_owner(root: Path, files: dict[str, render.FileSpec], *, check_owner: bool) -> list[str]:
    """Installed files whose mode, or owner (only checkable as root, where apply sets it), is not the rendered one.

    Content alone is not enough: slurmdbd refuses a config readable by others, and a shim must stay executable.
    """
    out = []
    for target, spec in files.items():
        dest = _under(root, target)
        if not dest.exists():
            continue  # already reported by the content check
        info = dest.stat()
        if stat.S_IMODE(info.st_mode) != spec.mode:
            out.append(f"{dest} (mode {stat.S_IMODE(info.st_mode):04o}, want {spec.mode:04o})")
        elif check_owner and pwd.getpwuid(info.st_uid).pw_name != spec.owner:
            out.append(f"{dest} (owner {pwd.getpwuid(info.st_uid).pw_name}, want {spec.owner})")
    return out


def kernel_refusals() -> list[str]:
    problems = []
    for name in PROTECTIONS:
        try:
            value = int((KERNEL_PROTECTIONS / name).read_text())
        except (OSError, ValueError):
            value = 0  # absent: a kernel without the protection
        if value < 1:
            problems.append(f"fs.{name} is {value}; the shared state directories need it at least 1 "
                            f"(sysctl -w fs.{name}=1, and a file in /etc/sysctl.d to keep it)")
    return problems


def platform_refusals(site, modules) -> list[str]:
    """What would make an apply fail on this machine, found before anything is touched: no apt to install the
    packages with, or an apt whose slurm is older than this build supports."""
    if not apt_install_commands(modules, site):
        return []
    if shutil.which("apt-get") is None:
        return ["this installer installs its packages with apt (Debian or Ubuntu); this machine has no apt-get"]
    if not any(m.NAME == "slurm" for m in modules):
        return []
    policy = components.run(["apt-cache", "policy", "slurm-wlm"]).stdout or ""
    version = re.search(r"Candidate:\s*(?:\d+:)?(\d+)\.(\d+)", policy)
    if not version:
        return ["apt offers no slurm-wlm package here (`apt-cache policy slurm-wlm` names no candidate)"]
    if (int(version[1]), int(version[2])) < MIN_SLURM:
        return [f"apt's slurm-wlm is {version[1]}.{version[2]}; this build needs slurm "
                f"{MIN_SLURM[0]}.{MIN_SLURM[1]} or newer (README, Requirements)"]
    return []


def platform_warnings(site, os_release: Path = OS_RELEASE) -> list[str]:
    """Where this machine differs from what the build was tested on: worth knowing, not a reason to stop."""
    text = os_release.read_text(encoding="utf-8", errors="replace") if os_release.exists() else ""
    fields = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text, re.MULTILINE))
    out = []
    if (fields.get("ID"), fields.get("VERSION_ID")) != TESTED_OS:
        out.append(f"untested operating system {fields.get('PRETTY_NAME') or 'unknown'}: the build is tested on "
                   "Ubuntu 24.04 only")
    if platform.machine() != TESTED_ARCH:
        out.append(f"untested architecture {platform.machine()}: the build is tested on {TESTED_ARCH} only")
    if len(site.nodes) > 1:
        out.append(f"the site file lists {len(site.nodes)} nodes: only one-machine sites have been tested")
    return out


def cmd_check(site, args, modules) -> int:
    _warn_portal_secrets(site, modules)
    for warning in platform_warnings(site):
        print(f"WARNING  {warning}")
    for problem in [*kernel_refusals(), *platform_refusals(site, modules)]:
        print(f"REFUSED  {problem}")
    files = _all(modules, site, "files")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            diff = differing(args.root, Path(tmp), _stage(files, Path(tmp)))
        except PermissionError as error:  # slurmdbd.conf is 0600 slurm, as slurmdbd requires
            print(f"cannot read {error.filename}: run the check with sudo", file=sys.stderr)
            return 2
    diff += _wrong_mode_or_owner(args.root, files, check_owner=os.geteuid() == 0)
    for d in diff:
        print(f"DIFFERS  {d}")
    if not diff:
        print("ok       installed files match the rendered ones")
    return 1 if diff or kernel_refusals() else 0


def apt_install_commands(modules, site) -> list[list[str]]:
    """One `apt-get install` per distinct set of flags, in component order.

    A component may ask for flags with an optional `apt_flags(site)` (the metrics component wants
    `--no-install-recommends`, so `prometheus` does not pull in a second node exporter); the others keep apt's defaults.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for m in modules:
        if packages := m.packages(site):
            groups.setdefault(tuple(getattr(m, "apt_flags", lambda _site: [])(site)), []).extend(packages)
    return [["apt-get", "install", "-y", *flags, *packages] for flags, packages in groups.items()]


def cmd_apply(site, args, modules, chown: Chown | None = _chown) -> int:
    for warning in platform_warnings(site):
        print(f"WARNING: {warning}", file=sys.stderr)
    refusals = [*kernel_refusals(), *platform_refusals(site, modules), *(r for m in modules for r in m.refusals(site))]
    if refusals:
        for r in refusals:
            print(f"REFUSED: {r}", file=sys.stderr)
        return 1
    sources = _all(modules, site, "apt_sources")
    if sources:
        write_files(args.root, sources, chown)
        _run_all([["apt-get", "update"]])
    _run_all(apt_install_commands(modules, site))
    files = _all(modules, site, "files")
    write_files(args.root, files, chown)
    _run_all([argv for m in modules for argv in m.post_install(site)])
    print(f"installed {len(files)} file(s) for: {', '.join(m.NAME for m in modules)}")
    return 0


def cmd_user(site, args, modules) -> int:
    if args.action is Action.ADD:
        _run_all(components.accounting_adds(site, args.user))
    else:
        _run_all([["sacctmgr", "-i", "delete", "user", f"name={args.user}", f"account={site.account}",
                   f"cluster={site.cluster_name}"]])
    print(f"{args.user}: {args.action} user (slurm account {site.account} on cluster {site.cluster_name})")
    return 0


def cmd_admin(site, args, modules) -> int:
    level = OPERATOR if args.action is Action.ADD else NO_LEVEL
    membership = (
        ["usermod", "-aG", site.admin_group, args.user]
        if args.action is Action.ADD
        else ["gpasswd", "-d", args.user, site.admin_group]
    )
    joins = components.accounting_adds(site, args.user) if args.action is Action.ADD else []  # an admin is a user too
    _run_all([*joins, membership,
              ["sacctmgr", "-i", "modify", "user", f"name={args.user}", "set", f"adminlevel={level}"]])
    print(f"{args.user}: {args.action} admin ({site.admin_group}, slurm AdminLevel {level})")
    return 0


def _copy_keeping(src: Path, dest: Path) -> bool:
    """Copy a regular file keeping its mode, times and owner; False (nothing copied) for anything else.

    Root copies from a directory a user owns, so the source is opened without following a symlink (a file swapped
    for one after the walk is refused, never read through) and checked to be a regular file on the open descriptor.
    """
    try:
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as target:
            shutil.copyfileobj(source, target)
    dest.chmod(stat.S_IMODE(info.st_mode))
    os.utime(dest, ns=(info.st_atime_ns, info.st_mtime_ns))
    try:
        os.chown(dest, info.st_uid, info.st_gid)
    except PermissionError:  # not root: the copy is ours, and the owner is the thing that cannot be kept
        print(f"warning: could not give {dest} its owner (uid {info.st_uid}); run with sudo", file=sys.stderr)
    return True


def cmd_migrate(site, args, modules) -> int:
    copied = skipped = 0
    for name in MIGRATED:
        source = args.source / name
        if source.is_symlink() or not source.is_dir():
            continue
        # os.walk does not descend into symlinked directories; _copy_keeping refuses anything but a regular file.
        for src in sorted(Path(d) / f for d, _dirs, files in os.walk(source) for f in files):
            dest = _under(args.root, site.state_dir) / name / src.relative_to(source)
            if dest.exists():
                skipped += 1
                continue
            if _copy_keeping(src, dest):
                copied += 1
            else:
                skipped += 1
    print(f"copied {copied} file(s), left {skipped} alone (already there, or not a plain file)")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, default=site_config.SITE_FILE)
    ap.add_argument("--root", type=Path, default=Path("/"))
    # --site and --root are accepted before or after the subcommand
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--site", type=Path, default=argparse.SUPPRESS)
    shared.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("detect", parents=[shared])
    sub.add_parser("check", parents=[shared])
    sub.add_parser("render", parents=[shared]).add_argument("--out", type=Path, required=True)
    apply = sub.add_parser("apply", parents=[shared])
    apply.add_argument("--component", action="append", choices=components.ORDER, dest="components")
    apply.add_argument("--replace-existing-slurm", action="store_true",
                       help="take over a slurm this installer did not set up (its settings files are replaced)")
    for name in ("user", "admin"):
        who = sub.add_parser(name, parents=[shared])
        who.add_argument("action", type=Action, choices=list(Action))
        who.add_argument("user")
    sub.add_parser("migrate-state", parents=[shared]).add_argument("--from", type=Path, required=True, dest="source")
    return ap


COMMANDS = {
    "render": cmd_render,
    "check": cmd_check,
    "user": cmd_user,
    "admin": cmd_admin,
    "migrate-state": cmd_migrate,
}


def main(argv: list[str] | None = None, chown: Chown | None = _chown) -> int:
    args = parser().parse_args(argv)
    if args.command == "detect":
        return detect.main([])
    try:
        site = site_config.load(args.site)
    except site_config.SiteError as error:
        print(error, file=sys.stderr)
        return 2
    components.SITE_PATH = args.site
    components.ROOT = getattr(args, "root", Path("/"))
    components.REPLACE_EXISTING_SLURM = getattr(args, "replace_existing_slurm", False)
    modules = components.load(site, getattr(args, "components", None))
    if args.command == "apply":
        return cmd_apply(site, args, modules, chown)
    return COMMANDS[args.command](site, args, modules)


if __name__ == "__main__":
    sys.exit(main())
