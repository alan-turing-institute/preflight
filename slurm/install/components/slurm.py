"""Slurm itself: daemons, accounting database, their settings files and the restart-protect drop-ins."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import render
import slurm_build

import components

NAME = "slurm"
PACKAGES = ["slurm-wlm", "slurmdbd", "mariadb-server", "munge"]
#: What the site file's `[slurm]` table replaces: slurm itself comes from `slurm_build`, these two from the distro.
DISTRO_ONLY = ["mariadb-server", "munge"]
#: Ubuntu's slurm packages, removed (when installed) before SchedMD's, which conflict with them.
DISTRO_SLURM = ["slurm-wlm", "slurmctld", "slurmd", "slurmdbd", "slurm-client"]
#: A `dpkg-query -f "${db:Status-Abbrev} ${Package}"` line: its status and the package.
DPKG_FIELDS = 2
RESTARTS = [["mariadb", "munge"], ["slurmdbd"], ["slurmctld", "slurmd"]]
# slurmctld's StateSaveLocation (templates/slurm/slurm.conf). The packages do not make it, and slurmctld, running as
# SlurmUser, cannot make it in /var/spool itself (Turing box, 2026-10-21: "mkdir(/var/spool/slurmctld): Permission denied").
STATE_SAVE = "/var/spool/slurmctld"
# slurmdbd logs in to MariaDB as the OS user `slurm` by unix_socket (templates/slurm/slurmdbd.conf), so no password
# exists anywhere. A fresh MariaDB has no such login: it is made here, before slurmdbd first starts.
DB_LOGIN = ("CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED VIA unix_socket; "
            "GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';")


def enabled(site) -> bool:
    return True


def files(site) -> dict[str, render.FileSpec]:
    return render.render(site, NAME, extra={"slurm_conf": components.slurm_conf_extra(site)})


def packages(site) -> list[str]:
    return list(DISTRO_ONLY if site.slurm_build else PACKAGES)


def install_steps(site) -> list:
    """With `[slurm]` set, build that release (cached) and install its packages in place of the distro's slurm; the
    build runs when its turn comes, after the distro packages, so a refused or dry run never builds."""
    return [lambda: _install_built(site)] if site.slurm_build else []


def _install_built(site) -> list[list[str]]:
    return install_release(*site.slurm_build)


def install_release(version: str, sha256: str) -> list[list[str]]:
    """The commands that put the release's packages (built first when not cached) in place of what is installed."""
    return install_commands(slurm_build.build(version, sha256))


def lock(manager: str, names: list[str], locked: bool) -> list[str]:
    """Keep the package manager from replacing our slurm packages (`locked`), or let it (only around our own installs).
    SchedMD's upgrade guide: keep slurm out of routine updates. Measured 2026-11-08: a Fedora 44 -> 45 upgrade put the
    distribution's slurm of the same version (built with hwloc) in place of ours, which then saw 12 of a hybrid CPU's 22
    CPUs, and the controller marked the node invalid. A dnf versionlock holds through a release upgrade (container
    test 2026-11-09, `distro-sync --releasever`); apt-mark hold is apt's equivalent."""
    if manager == "apt-get":
        return ["apt-mark", "hold" if locked else "unhold", *names]
    return ["dnf", "versionlock", "add" if locked else "delete", *names]


def _installed_debs(names: list[str]) -> list[str]:
    status = components.run(["dpkg-query", "-W", "-f=${db:Status-Abbrev} ${Package}\\n", *names]).stdout or ""
    # the second letter is the state: `ii` installed, `hi` installed and held (as `lock` leaves ours), `rc` removed
    return [fields[1] for line in status.splitlines() if len(fields := line.split()) == DPKG_FIELDS and fields[0][1:] == "i"]


def install_commands(built: list[Path]) -> list[list[str]]:
    """Install the allowlisted packages among `built` (`slurm_build.installable`), after removing the distribution's
    slurm packages when installed, and `lock` them. Run them under `daemons_held`."""
    wanted = [str(p) for p in slurm_build.installable(built)]
    if not wanted:
        raise slurm_build.SlurmBuildError("the build left none of the packages that get installed (slurm_build.DEB_INSTALL, RPM_INSTALL)")
    names = [slurm_build.package_name(Path(p)) for p in wanted]
    if not wanted[0].endswith(".deb"):
        # A build has the name, version and release of the distribution's own package of that version (Fedora ships
        # slurm under the same names), and `dnf install` then keeps the installed files and reports success (container
        # test 2026-11-09); `reinstall` puts ours in. `dnf-command(versionlock)` is dnf5 itself, or dnf4's plugin.
        return [["dnf", "install", "-y", "dnf-command(versionlock)"], lock("dnf", names, False),
                ["dnf", "install", "-y", *wanted], ["dnf", "reinstall", "-y", *wanted], lock("dnf", names, True)]
    installed = _installed_debs([*DISTRO_SLURM, *names])
    distro, ours = [n for n in installed if n in DISTRO_SLURM], [n for n in installed if n in names]
    return [*([["apt-get", "remove", "-y", *distro]] if distro else []),
            *([lock("apt-get", ours, False)] if ours else []),  # apt-mark refuses a package it does not know
            ["apt-get", "install", "-y", *wanted], lock("apt-get", names, True)]


#: Packages start their daemons when installed (SchedMD's debs, Ubuntu's, and the rpms'): a new slurmdbd started that
#: way converts the accounting database unattended, and a slurmctld on a half-written config. Debian-like systems ask
#: `policy-rc.d`, whose exit status 101 means "do not start"; others get the units masked until the install is over.
POLICY_RC_D = "usr/sbin/policy-rc.d"
POLICY_RC_D_SAVED = POLICY_RC_D + ".gpujobs-saved"
NO_START = "#!/bin/sh\nexit 101\n"
DAEMONS = ["slurmdbd", "slurmctld", "slurmd"]
EXECUTABLE = 0o755


@contextmanager
def daemons_held(manager: str) -> Iterator[None]:
    """Package installs and removals inside the block start no slurm daemon (`manager` is `apt-get` or `dnf`)."""
    with (_policy_rc_d() if manager == "apt-get" else _units_masked()):
        yield


@contextmanager
def _units_masked() -> Iterator[None]:
    done = components.run(["systemctl", "mask", "--runtime", *DAEMONS])
    if done.returncode != 0:
        raise SystemExit(f"cannot keep the slurm daemons from starting during the install: systemctl mask failed: {done.stderr or ''}".rstrip())
    try:
        yield
    finally:
        if components.run(["systemctl", "unmask", "--runtime", *DAEMONS]).returncode != 0:
            print(f"WARNING: run `systemctl unmask --runtime {' '.join(DAEMONS)}` by hand", file=sys.stderr)


@contextmanager
def _policy_rc_d() -> Iterator[None]:
    path, saved = components.ROOT / POLICY_RC_D, components.ROOT / POLICY_RC_D_SAVED
    if saved.exists() or saved.is_symlink():
        raise SystemExit(f"{saved} is left over from an interrupted run: put it back as {path} (or delete it), then run again")
    if path.exists() or path.is_symlink():
        os.replace(path, saved)  # someone's own policy (Docker images ship one) comes back afterwards
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(NO_START, encoding="utf-8")
        path.chmod(EXECUTABLE)
        yield
    finally:
        path.unlink(missing_ok=True)
        if saved.exists() or saved.is_symlink():
            os.replace(saved, path)


#: squeue's own listing without the waiting jobs: a restart of the daemons kills or requeues a job that is running, but a
#: job still waiting is kept in slurmctld's saved state (backed up, and converted by an upgrade) and keeps its place.
ACTIVE_JOBS = ["squeue", "-h", "-t", "RUNNING,SUSPENDED,COMPLETING,CONFIGURING"]


def refusals(site, upgrading: bool = False) -> list[str]:
    """A slurm someone else set up is never replaced unasked (its slurm.conf lacks `render.RENDERED_MARK`), `apply`
    never changes the slurm release (`upgrade-slurm` does, with `upgrading`), and a restart requeues or kills running
    jobs, so none may be running (an absent `squeue` means no slurm yet); jobs that are only waiting do not count."""
    out = []
    conf = components.ROOT / "etc/slurm/slurm.conf"
    try:
        first = conf.read_text(encoding="utf-8", errors="replace").partition("\n")[0]
    except FileNotFoundError:
        first = render.RENDERED_MARK  # no slurm set up here yet
    if render.RENDERED_MARK not in first and not components.REPLACE_EXISTING_SLURM:
        out.append(f"{conf} was not written by this installer: this machine runs a slurm someone else set up, and "
                   "apply would replace its settings. Leave it, or back it up and pass --replace-existing-slurm")
    if site.slurm_build and not upgrading and (installed := _installed_version()) not in ("", site.slurm_build[0]):
        hint = "use install.py upgrade-slurm"
        if site.slurm_release and version_tuple(installed) > version_tuple(site.slurm_build[0]):
            hint = (f"if a maintenance window installed it (slurm_release.py apply), pin the site file's [slurm] to "
                    f'version = "{installed}" with the sha256 in {site.slurm_release.records}/{installed}.json, and no steps')
        out.append(f"slurm {installed} is installed and the site file's [slurm] version is {site.slurm_build[0]}: "
                   f"apply never changes the release; {hint}")
    try:
        done = components.run(ACTIVE_JOBS)
    except FileNotFoundError:
        return out
    if done.returncode != 0 and _never_ran():
        return out  # a first apply that stopped before starting the controller can be run again
    if done.returncode != 0 or done.stdout.strip():
        out.append("slurm has running jobs (or squeue failed); restarting its daemons now would requeue or kill them")
    return out


def _installed_version() -> str:
    """The installed slurm's release as `sinfo -V` prints it ("slurm 23.11.4"), or "" when there is none."""
    try:
        done = components.run(["sinfo", "-V"])
    except FileNotFoundError:
        return ""
    found = re.search(r"\d+\.\d+\.\d+", done.stdout or "")
    return found[0] if done.returncode == 0 and found else ""


def _never_ran() -> bool:
    """slurmctld is down and has never saved job state, so no job exists that a restart could requeue or kill."""
    if (components.ROOT / STATE_SAVE.lstrip("/") / "job_state").exists():
        return False
    return components.run(["systemctl", "is-active", "--quiet", "slurmctld"]).returncode != 0


def post_install(site) -> list:
    """Restart the database side, give slurmdbd its database login, give accounting the cluster and its account, then
    start the controller.

    The records go in before slurmctld starts, the order the test cluster boots in, so slurmctld registers with a
    cluster accounting already knows.
    """
    database, accounting, controller = RESTARTS
    return [
        ["systemctl", "daemon-reload"],
        ["systemctl", "restart", *database],
        ["mariadb", "-e", DB_LOGIN],
        ["systemctl", "restart", *accounting],
        lambda: _records(site),
        ["install", "-d", "-m", "0755", "-o", "slurm", "-g", "slurm", STATE_SAVE],
        ["systemctl", "restart", *controller],
    ]


def _records(site) -> list[list[str]]:
    components.wait_for_accounting()
    return components.accounting_adds(site)


# -- upgrade-slurm ----------------------------------------------------------------------------------------------

BACKUP_ROOT = Path("/var/backups/gpujobs")
#: slurmdbd converts the accounting database on its first start of a newer release, "tens of minutes" (plan, facts)
CONVERSION_WAIT_S = 60 * 60
CONVERSION_POLL_S = 5
ANSWER_WAIT_S = 120
BYTES_PER_MB = 1_000_000
DB_DUMP, STATE_TAR, CONF_TAR = "slurm_acct_db.sql", "state_save.tar.gz", "etc_slurm.tar.gz"
sleep = time.sleep


def running_version() -> tuple[str, tuple[int, ...]]:
    """The installed slurm as `sinfo -V` names it ("slurm 23.11.4", "slurm-wlm 23.11.4") and its version."""
    done = components.run(["sinfo", "-V"])
    text = (done.stdout or "").strip()
    found = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if done.returncode != 0 or not found:
        raise SystemExit(f"cannot read the running slurm version from `sinfo -V`: {text} {done.stderr or ''}".strip())
    return text, version_tuple(found[0])


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(map(int, version.split(".")))


def steps_after(site, running: tuple[int, ...]) -> list[tuple[str, str]]:
    """The releases still to install: the site's steps then its own version, each newer than the running one."""
    return [(v, h) for v, h in (*site.slurm_steps, site.slurm_build) if version_tuple(v) > running]


def build_all(site, running: tuple[int, ...]) -> dict[str, list[Path]]:
    """Every release `upgrade-slurm` will install, built (or found built) before anything is stopped."""
    return {version: slurm_build.build(version, sha256) for version, sha256 in steps_after(site, running)}


def upgrade_refusals(site, running: tuple[int, ...]) -> list[str]:
    if not site.slurm_build:
        return ["the site file has no [slurm] table: there is no release to upgrade to"]
    if version_tuple(site.slurm_build[0]) <= running:
        return [f"slurm {'.'.join(map(str, running))} is already at or after the site file's {site.slurm_build[0]}"]
    return refusals(site, upgrading=True)


def config_value(key: str) -> str:
    done = components.run(["scontrol", "show", "config"])
    found = re.search(rf"^{key}\s*=\s*(\S+)", done.stdout or "", re.MULTILINE)
    if done.returncode != 0 or not found:
        raise SystemExit(f"`scontrol show config` gives no {key}")
    return found[1].split("(")[0]  # SlurmUser prints as "slurm(64030)"


def backup(state_save: str, stamp: datetime | None = None) -> Path:
    """Dump the accounting database and tar the state directory and /etc/slurm under a new folder; print each size."""
    folder = BACKUP_ROOT / f"slurm-{stamp or datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    jobs = [(["mariadb-dump", f"--result-file={folder / DB_DUMP}", "slurm_acct_db"], folder / DB_DUMP),
            (["tar", "-czf", str(folder / STATE_TAR), "-C", "/", state_save.lstrip("/")], folder / STATE_TAR),
            (["tar", "-czf", str(folder / CONF_TAR), "-C", "/", "etc/slurm"], folder / CONF_TAR)]
    for argv, target in jobs:
        done = components.run(argv)
        if done.returncode != 0 or not target.exists():
            raise SystemExit(f"FAILED ({done.returncode}): {' '.join(argv)}\n{done.stderr or ''}".rstrip())
        print(f"backup {target}: {target.stat().st_size / BYTES_PER_MB:.1f} MB")
    return folder


def stop_daemons() -> list[list[str]]:
    return [["systemctl", "stop", "slurmctld", "slurmd"], ["systemctl", "stop", "slurmdbd"]]


PARTITION = re.compile(r"^PartitionName=(\S+)\s.*?\bState=(\S+)", re.MULTILINE)
INACTIVE = "INACTIVE"


def partition_states() -> dict[str, str]:
    """Each partition's state (UP, DOWN, DRAIN or INACTIVE), to put back when the upgrade is over."""
    done = components.run(["scontrol", "show", "partition", "-o"])
    states = dict(PARTITION.findall(done.stdout or ""))
    if done.returncode != 0 or not states:
        raise SystemExit(f"`scontrol show partition` lists no partitions: {done.stdout or ''} {done.stderr or ''}".strip())
    return states


def set_partitions(states: dict[str, str]) -> list[str]:
    """Give each partition its state; the commands that failed (to be run by hand) come back."""
    failed = []
    for name, state in states.items():
        argv = ["scontrol", "update", f"PartitionName={name}", f"State={state}"]
        if components.run(argv).returncode != 0:
            failed.append(" ".join(argv))
    return failed


def queue_busy() -> bool:
    done = components.run(ACTIVE_JOBS)
    return done.returncode != 0 or bool((done.stdout or "").strip())


def resume_unchanged(states: dict[str, str], stopped: bool) -> None:
    """Back from a refusal or a failure before any package changed: start what was stopped (`stopped`), reopen the
    partitions."""
    if stopped:
        for unit in DAEMONS:
            components.run(["systemctl", "start", unit])
        await_answers(["sinfo", "-h"])
    if failed := set_partitions(states):
        raise SystemExit("could not reopen the partitions; run by hand:\n" + "\n".join(f"  {c}" for c in failed))


class ConversionUnfinished(Exception):
    """The foreground slurmdbd is still running (converting the accounting database) and is left alone."""

    def __init__(self, pid: int, log: Path):
        super().__init__(f"slurmdbd (pid {pid}) is still converting the accounting database and was left running; "
                         f"watch it with `tail -f {log}` and `sacctmgr show cluster` (that answers once it is done)")
        self.pid, self.log = pid, log


def convert_accounting(user: str, log: Path) -> None:
    """Run the new slurmdbd in the foreground until it answers (its first start converts the database), then stop it
    and hand over to the service. A slurmdbd that exits first is a failure; one that is still running when the wait
    ends or breaks is never signalled (stopping a converting database server risks the database): it raises
    `ConversionUnfinished`. It runs in its own session, writing `log`, so closing the terminal does not end it."""
    state = components.run(["systemctl", "is-active", "slurmdbd"])
    if (state.stdout or "").strip() != "inactive":
        raise SystemExit(f"the slurmdbd service is {(state.stdout or '').strip() or 'not inactive'} before the conversion: "
                         "it must be stopped, or two slurmdbd would work on one database")
    with log.open("ab") as out:
        proc = components.popen(["sudo", "-u", user, "slurmdbd", "-D", "-vvv"], stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        components.wait_for_accounting(sleep, CONVERSION_WAIT_S, CONVERSION_POLL_S, lambda: proc.poll() is None)
    except BaseException as error:
        if proc.poll() is None:
            raise ConversionUnfinished(proc.pid, log) from error
        raise
    proc.terminate()  # it answers, so the conversion is finished
    proc.wait()


def await_answers(command: list[str]) -> None:
    for _ in range(ANSWER_WAIT_S):
        if components.run(command).returncode == 0:
            return
        sleep(1)
    raise SystemExit(f"{' '.join(command)} did not succeed within {ANSWER_WAIT_S} s")


HISTORY_DAYS = 7
SACCT_TIME = "%Y-%m-%dT%H:%M:%S"


def history_window(now: datetime) -> tuple[str, str]:
    """The seven days up to `now` as absolute `sacct` times, fixed once so every listing covers the same jobs."""
    return (now - timedelta(days=HISTORY_DAYS)).strftime(SACCT_TIME), now.strftime(SACCT_TIME)


def history(window: tuple[str, str]) -> set[str]:
    start, end = window
    done = components.run(["sacct", "-X", "-n", "-P", "-S", start, "-E", end, "-o", "JobID"])
    if done.returncode != 0:
        raise SystemExit(f"sacct failed ({done.returncode}): {done.stderr or ''}".rstrip())
    return set((done.stdout or "").split())


def restore_text(folder: Path, state_save: str, previous: str, manager: str, partitions: dict[str, str]) -> str:
    """The commands that put the machine back as the backup found it, printed when a step fails. `previous` is what
    `sinfo -V` printed before the upgrade: the distribution's slurm is reinstalled from the repositories, SchedMD's
    from the build cache. The packages go in under the same no-daemon guard as the upgrade's, and the daemons are
    stopped again before the files come back; `partitions` are the states they had before the upgrade closed them."""
    cached = slurm_build.DEFAULT_CACHE / re.search(r"\d+\.\d+\.\d+", previous)[0]  # as printed: the directory is 25.05.9
    apt = manager == "apt-get"
    ours = slurm_build.DEB_INSTALL if apt else slurm_build.RPM_INSTALL
    unlock = f"{' '.join(lock(manager, ours, False))} || true"  # the upgrade locked the new release's packages
    if previous.startswith("slurm-wlm"):
        reinstall = [f"{manager} install -y {'--allow-downgrades ' if apt else ''}slurm-wlm slurmdbd"]
    else:
        built = cached / slurm_build.build_dir()  # this system release's build, never an older release's
        files = [f"{built}/{n}_*.deb" if apt else f"{built}/rpmbuild/RPMS/*/{n}-[0-9]*.rpm" for n in ours]  # as the upgrade installs
        reinstall = [f"{manager} install -y {' '.join(files)}", " ".join(lock(manager, ours, True))]
    policy = f"/{POLICY_RC_D}"
    hold, release = ([f"mv -n {policy} {policy}.gpujobs-saved 2>/dev/null; printf '#!/bin/sh\\nexit 101\\n' > {policy}; chmod 755 {policy}"],
                     [f"rm -f {policy}; mv {policy}.gpujobs-saved {policy} 2>/dev/null; true"]) if apt else (
        [f"systemctl mask --runtime {' '.join(DAEMONS)}"], [f"systemctl unmask --runtime {' '.join(DAEMONS)}"])
    stop = "systemctl stop slurmctld slurmd slurmdbd"
    lines = [stop,
             *hold,
             unlock,
             f"{manager} remove -y 'slurm-smd*'" if apt else "dnf remove -y 'slurm*'",
             *reinstall,
             *release,
             stop,  # the packages' scripts may have started one
             "mariadb -e 'DROP DATABASE IF EXISTS slurm_acct_db; CREATE DATABASE slurm_acct_db'",
             f"mariadb slurm_acct_db < {folder / DB_DUMP}",
             f"rm -rf {state_save}/* /etc/slurm/*",
             f"tar -xzf {folder / STATE_TAR} -C /",
             f"tar -xzf {folder / CONF_TAR} -C /",
             "systemctl restart slurmdbd slurmctld slurmd",
             *(f"scontrol update PartitionName={name} State={state}" for name, state in partitions.items())]
    return "\n".join(f"  {line}" for line in lines)
