"""Install, check and administer the GPU job layer from the site file.

    install.py detect                           # print a draft site file for this machine
    install.py render --out DIR                 # write every enabled component's files under DIR
    install.py check                            # exit 1 if the installed files differ from the rendered ones
    sudo install.py apply [--component NAME ...]
    sudo install.py part-time add NODE          # on the controller: check user ids, copy the munge key to a part-time machine
    sudo install.py apply --part-time NODE --controller ADDR --slurm-uid U --slurm-gid G --i-accept-cluster-trust
    sudo install.py upgrade-slurm               # step the running slurm up to the site file's [slurm] release
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
import socket
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from enum import StrEnum
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parents[1] / "machine" / "earlyoom"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import components  # noqa: E402 - paths set just above
import cores  # noqa: E402
import components.agent as agent_component  # noqa: E402
import components.part_time as part_time  # noqa: E402
import components.slurm as slurm_component  # noqa: E402
import detect  # noqa: E402
import render  # noqa: E402
import site_config  # noqa: E402
from site_config import UpgradeExit  # noqa: E402
import slurm_build  # noqa: E402
from install_earlyoom import differing, install  # noqa: E402

#: slurmdbd's output while it converts the accounting database, kept beside the backup
CONVERSION_LOG = "slurmdbd-conversion.log"
MIGRATED = ("jobs", "footprints", "events")
OPERATOR = "Operator"
#: slurmctld caches who is in a partition's AllowGroups for GroupUpdateTime (600 s by default); a reconfigure reloads it
#: (measured 2026-10-23, harness: a user just added stayed refused until it).
RELOAD_GROUPS = ["scontrol", "reconfigure"]
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


def package_manager(site) -> str:
    """`apt-get`, or `dnf` on a machine that has no apt-get when the site builds its own slurm (`[slurm]`)."""
    if getattr(site, "slurm_build", None) and shutil.which("apt-get") is None and shutil.which("dnf"):
        return "dnf"
    return "apt-get"


def _this_host() -> str:
    return socket.gethostname().split(".")[0]


def _slurmd_reads_layouts() -> bool | None:
    """Whether this machine's slurmd is built with hwloc, the only way it reads a layout file (`HWLOC_XMLFILE`); without
    hwloc it takes its cores from /proc/cpuinfo and the GPU would be bound to the wrong ones. None: no slurmd yet."""
    slurmd = shutil.which("slurmd")
    return None if slurmd is None else "libhwloc" in (components.run(["ldd", slurmd]).stdout or "")


def core_kind_refusals(site, speeds: cores.CoreSpeeds | None = None, here: str | None = None) -> list[str]:
    """`fast_cpus` (D46) must describe this machine (the layout file is read by the slurmd it is installed beside),
    name cores every one of which is faster than every core outside it, and meet a slurmd built with hwloc. Prints
    each check that holds."""
    here = here or _this_host()
    out = []
    for node in (n for n in site.full_nodes if n.fast_cpus):
        if node.name != here:
            out.append(f"site file: fast_cpus is set on node {node.name}, but this is {here}: the layout file is installed for "
                       "this machine's slurmd, so set fast_cpus only on this machine's own node entry")
            continue
        khz = (speeds or cores.CoreSpeeds.read()).khz
        fast, others = node.fast_cpu_ids, [cpu for cpu in range(node.cpus) if cpu not in node.fast_cpu_ids]
        known = all(cpu in khz for cpu in range(node.cpus))
        if not known or min(khz[cpu] for cpu in fast) <= max(khz[cpu] for cpu in others):
            groups = cores.CoreSpeeds(khz).groups()
            seen = "; ".join(f"{mhz} MHz: {cores.cpu_list(g)}" for mhz, g in groups.items()) or "no core reports a top speed"
            out.append(f"site file: {node.name}'s fast_cpus = {node.fast_cpus!r} are not all faster than its other cores "
                       f"({seen}; `slurm/cores.py`): correct fast_cpus, or leave it out")
            continue
        print(f"ok       fast_cpus {node.fast_cpus} are this host's fastest cores")
        reads = _slurmd_reads_layouts()
        if reads is False:
            out.append("site file: fast_cpus needs a slurmd built with hwloc to read its layout file, and this machine's slurmd "
                       "is not (`ldd $(command -v slurmd)` names no libhwloc): leave fast_cpus out")
        else:
            print("ok       slurmd is built with hwloc" if reads else
                  "note     no slurmd yet: run `install.py check` after apply to confirm it is built with hwloc")
    return out


def core_kind_warnings(site, speeds: cores.CoreSpeeds | None = None, here: str | None = None) -> list[str]:
    """A GPU node whose cores come in two equal kinds and has no `fast_cpus`: its GPU jobs are raised to the fewest
    cores that always include a fast one (D33) instead of holding only what they ask for (D46)."""
    node = next((n for n in site.full_nodes if n.gpus and not n.fast_cpus and n.name == (here or _this_host())), None)
    if node is None:
        return []
    kinds = list((speeds or cores.CoreSpeeds.read()).groups().values())
    if len(kinds) != site_config.CORE_KINDS or len(kinds[0]) != len(kinds[1]) or len(kinds[0]) * site_config.CORE_KINDS != node.cpus:
        return []
    return [f"{node.name}'s cores come in two kinds; with fast_cpus = \"{cores.cpu_list(kinds[-1])}\" in its site file entry, "
            "a GPU job holds only the cores it asks for, all fast ones (D46, README \"Fast and slow cores\")"]


def platform_refusals(site, modules) -> list[str]:
    """What would make an apply fail on this machine, found before anything is touched: a `fast_cpus` that is not
    this host's fastest cores (asked only when the slurm component, which installs the layout file, is applied: a
    part-time machine takes only the tools), no package manager to install the packages with, or an apt whose slurm is
    older than this build supports (not asked when the site builds its own slurm: `[slurm]`)."""
    lays_out = any(m.NAME == slurm_component.NAME for m in modules)
    return [*(core_kind_refusals(site) if lays_out else []), *_package_refusals(site, modules)]


def _package_refusals(site, modules) -> list[str]:
    if not apt_install_commands(modules, site):
        return []
    built = getattr(site, "slurm_build", None)
    if shutil.which(package_manager(site)) is None:
        return ["this installer needs apt-get or dnf to install its packages; this machine has neither"] if built else [
            "this installer installs its packages with apt (Debian or Ubuntu); this machine has no apt-get"]
    if built or not any(m.NAME == "slurm" for m in modules):
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
    return out + core_kind_warnings(site)


def cmd_check(site, args, modules) -> int:
    _warn_portal_secrets(site, modules)
    for warning in platform_warnings(site):
        print(f"WARNING  {warning}")
    refused = [*kernel_refusals(), *platform_refusals(site, modules)]
    for problem in refused:
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
    return 1 if diff or refused else 0


def apt_install_commands(modules, site) -> list[list[str]]:
    """One `apt-get install` per distinct set of flags, in component order.

    A component may ask for flags with an optional `apt_flags(site)` (the metrics component wants
    `--no-install-recommends`, so `prometheus` does not pull in a second node exporter); the others keep apt's defaults.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for m in modules:
        if packages := m.packages(site):
            groups.setdefault(tuple(getattr(m, "apt_flags", lambda _site: [])(site)), []).extend(packages)
    if package_manager(site) == "dnf":  # apt's flags mean nothing to it
        return [["dnf", "install", "-y", *dict.fromkeys(p for pkgs in groups.values() for p in pkgs)]] if groups else []
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
    # a slurm package starts its daemons when installed, on settings that are not written yet: not while we install
    with slurm_component.daemons_held(package_manager(site)) if any(m.NAME == "slurm" for m in modules) else nullcontext():
        _run_all(apt_install_commands(modules, site))
        _run_all([step for m in modules for step in getattr(m, "install_steps", lambda _site: [])(site)])
    files = _all(modules, site, "files")
    write_files(args.root, files, chown)
    _run_all([argv for m in modules for argv in m.post_install(site)])
    print(f"installed {len(files)} file(s) for: {', '.join(m.NAME for m in modules)}")
    return 0


def cmd_part_time_apply(site, args, chown: Chown | None = _chown) -> int:
    """Join this machine to the cluster as the site file's part-time node `args.part_time` (plan B5)."""
    if not args.accept_cluster_trust:
        print(f"REFUSED: {part_time.TRUST_WARNING}", file=sys.stderr)
        return 1
    if problems := part_time.refusals(site, args):
        for r in problems:
            print(f"REFUSED: {r}", file=sys.stderr)
        return 1
    manager = package_manager(site)
    if shutil.which(manager) is None:
        print(f"REFUSED: {manager} is not on this machine", file=sys.stderr)
        return 1
    tools = components.load(site, ["tools"])

    def install_tools() -> None:
        if cmd_apply(site, args, tools, chown):
            raise SystemExit("installing the tools component failed")

    def install_site_copy() -> None:
        write_files(components.ROOT, agent_component.site_copy(site), chown)

    _run_all(part_time.steps(site, args, manager, install_tools, install_site_copy))
    print(f"{args.part_time}: joined as a part-time node; now run `install.py part-time add {args.part_time}` on the controller")
    return 0


def cmd_part_time(site, args, modules) -> int:
    return part_time.cmd_add(site, args)


BUILD_ERRORS = (slurm_build.SlurmBuildError, subprocess.CalledProcessError, OSError)


def cmd_upgrade(site, args, modules, chown: Chown | None = _chown) -> int:
    """Step a running slurm up to the site's `[slurm]` release, through each of its `steps` first (plan A5, A6.1).

    Builds every release first (a failed build leaves slurm untouched), closes the partitions and checks the queue
    again, stops the controller and nodes, backs up, then for each release installs it (no daemon starts by itself),
    converts the accounting database with a foreground slurmdbd and checks the history is intact. A failure before any
    package changed puts slurm back as it was; later ones print the commands that restore the backup (a downgrade is
    not supported once a newer daemon has started). Ends with an `apply`, so the settings files are the release's.
    """
    previous, running = slurm_component.running_version()
    if refusals := slurm_component.upgrade_refusals(site, running):
        for r in refusals:
            print(f"REFUSED: {r}", file=sys.stderr)
        return UpgradeExit.REFUSED
    manager = package_manager(site)
    try:
        built = slurm_component.build_all(site, running)
    except BUILD_ERRORS as error:
        print(f"slurm build failed: {error}\nslurm was not touched.", file=sys.stderr)
        return UpgradeExit.REFUSED
    state_save, user = slurm_component.config_value("StateSaveLocation"), slurm_component.config_value("SlurmUser")
    window = slurm_component.history_window(datetime.now())
    before = slurm_component.history(window)
    states = slurm_component.partition_states()
    stopped = installing = False
    version, folder = "", None
    try:
        if failed := slurm_component.set_partitions(dict.fromkeys(states, slurm_component.INACTIVE)):  # nothing new can be queued
            raise SystemExit("could not close every partition; run by hand:\n" + "\n".join(f"  {c}" for c in failed))
        if slurm_component.queue_busy():
            raise SystemExit("jobs were queued after the first check; nothing was changed")
        _run_all(slurm_component.stop_daemons()[:1])
        stopped = True
        folder = slurm_component.backup(state_save)
        _run_all(slurm_component.stop_daemons()[1:])  # slurmdbd only after the dump
        installing = True
        for index, version in enumerate(built):
            print(f"upgrading slurm to {version}")
            if index:
                _run_all(slurm_component.stop_daemons())
            with slurm_component.daemons_held(manager):
                _run_all([lambda p=built[version]: slurm_component.install_commands(p)])
            slurm_component.convert_accounting(user, folder / CONVERSION_LOG)
            _run_all([["systemctl", "start", "slurmdbd"], ["systemctl", "start", "slurmctld"], ["systemctl", "start", "slurmd"]])
            slurm_component.await_answers(["sinfo", "-h"])
            if lost := before - slurm_component.history(window):
                raise SystemExit(f"sacct no longer lists {len(lost)} job(s) it listed before: {', '.join(sorted(lost)[:5])}")
    except slurm_component.ConversionUnfinished as error:
        reopen = "\n".join(f"  scontrol update PartitionName={name} State={state}" for name, state in states.items())
        print(f"UPGRADE STOPPED at slurm {version}: {error}\nThe backup is {folder}. Do not stop that slurmdbd or restore "
              f"while it runs. Once it answers, stop it (kill {error.pid}), then as root:\n  systemctl start slurmdbd slurmctld "
              f"slurmd\nand reopen the partitions, which the upgrade closed:\n{reopen}", file=sys.stderr)
        return UpgradeExit.CONVERSION_RUNNING
    except (SystemExit, *BUILD_ERRORS) as error:
        if not installing:
            print(f"UPGRADE REFUSED: {error}", file=sys.stderr)
            try:
                slurm_component.resume_unchanged(states, stopped)
            except SystemExit as again:
                print(again, file=sys.stderr)
            else:
                print("No package was changed; slurm is running with its partitions as they were.", file=sys.stderr)
            return UpgradeExit.REFUSED
        print(f"UPGRADE FAILED at slurm {version}: {error}", file=sys.stderr)
        print(f"Nothing further was done. To go back, as root:\n"
              f"{slurm_component.restore_text(folder, state_save, previous, manager, states)}", file=sys.stderr)
        return UpgradeExit.FAILED_AFTER_CHANGE
    if failed := slurm_component.set_partitions(states):
        print("could not reopen the partitions; run by hand:\n" + "\n".join(f"  {c}" for c in failed), file=sys.stderr)
        return UpgradeExit.FAILED_AFTER_CHANGE
    return UpgradeExit.DONE if cmd_apply(site, args, modules, chown) == 0 else UpgradeExit.FAILED_AFTER_CHANGE


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
              ["sacctmgr", "-i", "modify", "user", f"name={args.user}", "set", f"adminlevel={level}"], RELOAD_GROUPS])
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


def cmd_service(site, args, modules) -> int:
    """Who may submit to the service partition (D20): membership of its Unix group, which slurm's AllowGroups reads."""
    if site.service is None:
        print("REFUSED: the site file has no [service] table, so there is no service partition", file=sys.stderr)
        return 1
    if args.action is Action.ADD:
        _run_all([*components.accounting_adds(site, args.user), ["usermod", "-aG", site.service.group, args.user],
                  RELOAD_GROUPS])
    else:
        _run_all([["gpasswd", "-d", args.user, site.service.group], RELOAD_GROUPS])
    print(f"{args.user}: {args.action} service ({site.service.group})")
    return 0


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
    sub.add_parser("upgrade-slurm", parents=[shared])
    sub.add_parser("render", parents=[shared]).add_argument("--out", type=Path, required=True)
    apply = sub.add_parser("apply", parents=[shared])
    apply.add_argument("--component", action="append", choices=components.ORDER, dest="components")
    apply.add_argument("--replace-existing-slurm", action="store_true",
                       help="take over a slurm this installer did not set up (its settings files are replaced)")
    apply.add_argument("--part-time", metavar="NODE", help="join this machine as that part-time node of the site file")
    apply.add_argument("--i-accept-cluster-trust", action="store_true", dest="accept_cluster_trust")
    apply.add_argument("--controller", help="with --part-time: the controller's address (slurmd's --conf-server, the firewall)")
    apply.add_argument("--slurm-uid", type=int, help="with --part-time: the controller's uid for the slurm user")
    apply.add_argument("--slurm-gid", type=int, help="with --part-time: the controller's gid for the slurm group")
    away = sub.add_parser("part-time", parents=[shared])
    away.add_argument("action", type=part_time.PartTimeAction, choices=list(part_time.PartTimeAction))
    away.add_argument("node")
    for name in ("user", "admin", "service"):
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
    "service": cmd_service,
    "part-time": cmd_part_time,
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
    if args.command == "apply" and args.part_time:
        return cmd_part_time_apply(site, args, chown)
    if args.command == "apply":
        return cmd_apply(site, args, modules, chown)
    if args.command == "upgrade-slurm":
        return cmd_upgrade(site, args, modules, chown)
    return COMMANDS[args.command](site, args, modules)


if __name__ == "__main__":
    sys.exit(main())
