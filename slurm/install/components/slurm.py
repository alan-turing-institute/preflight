"""Slurm itself: daemons, accounting database, their settings files and the restart-protect drop-ins."""

from __future__ import annotations

import re
import time
from datetime import datetime
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
    built = [str(p) for p in slurm_build.build(version, sha256) if "-dbgsym" not in p.name]
    if not built[0].endswith(".deb"):
        return [["dnf", "install", "-y", *built]]
    status = components.run(["dpkg-query", "-W", "-f=${db:Status-Abbrev} ${Package}\\n", *DISTRO_SLURM]).stdout or ""
    installed = [line[3:] for line in status.splitlines() if line.startswith("ii ")]
    return [*([["apt-get", "remove", "-y", *installed]] if installed else []), ["apt-get", "install", "-y", *built]]


def refusals(site) -> list[str]:
    """A slurm someone else set up is never replaced unasked (its slurm.conf lacks `render.RENDERED_MARK`), and a
    restart requeues or kills running jobs, so the queue must be empty (an absent `squeue` means no slurm yet)."""
    out = []
    conf = components.ROOT / "etc/slurm/slurm.conf"
    try:
        first = conf.read_text(encoding="utf-8", errors="replace").partition("\n")[0]
    except FileNotFoundError:
        first = render.RENDERED_MARK  # no slurm set up here yet
    if render.RENDERED_MARK not in first and not components.REPLACE_EXISTING_SLURM:
        out.append(f"{conf} was not written by this installer: this machine runs a slurm someone else set up, and "
                   "apply would replace its settings. Leave it, or back it up and pass --replace-existing-slurm")
    try:
        done = components.run(["squeue", "-h"])
    except FileNotFoundError:
        return out
    if done.returncode != 0 and _never_ran():
        return out  # a first apply that stopped before starting the controller can be run again
    if done.returncode != 0 or done.stdout.strip():
        out.append("slurm has jobs (or squeue failed); restarting its daemons now would requeue or kill them")
    return out


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


def upgrade_refusals(site, running: tuple[int, ...]) -> list[str]:
    if not site.slurm_build:
        return ["the site file has no [slurm] table: there is no release to upgrade to"]
    if version_tuple(site.slurm_build[0]) <= running:
        return [f"slurm {'.'.join(map(str, running))} is already at or after the site file's {site.slurm_build[0]}"]
    return refusals(site)


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


def convert_accounting(user: str) -> None:
    """Run the new slurmdbd in the foreground until it answers (its first start converts the database), then stop it
    and hand over to the service; a slurmdbd that exits first is a failure."""
    proc = components.popen(["sudo", "-u", user, "slurmdbd", "-D", "-vvv"])
    try:
        components.wait_for_accounting(sleep, CONVERSION_WAIT_S, CONVERSION_POLL_S, lambda: proc.poll() is None)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def await_answers(command: list[str]) -> None:
    for _ in range(ANSWER_WAIT_S):
        if components.run(command).returncode == 0:
            return
        sleep(1)
    raise SystemExit(f"{' '.join(command)} did not succeed within {ANSWER_WAIT_S} s")


def history() -> set[str]:
    done = components.run(["sacct", "-X", "-n", "-P", "-S", "now-7days", "-o", "JobID"])
    if done.returncode != 0:
        raise SystemExit(f"sacct failed ({done.returncode}): {done.stderr or ''}".rstrip())
    return set((done.stdout or "").split())


def restore_text(folder: Path, state_save: str, previous: str, manager: str) -> str:
    """The commands that put the machine back as the backup found it, printed when a step fails. `previous` is what
    `sinfo -V` printed before the upgrade: the distribution's slurm is reinstalled from the repositories, SchedMD's
    from the build cache."""
    cached = slurm_build.DEFAULT_CACHE / ".".join(map(str, version_tuple(re.search(r"\d+\.\d+\.\d+", previous)[0])))
    apt = manager == "apt-get"
    if previous.startswith("slurm-wlm"):
        reinstall = f"{manager} install -y {'--allow-downgrades ' if apt else ''}slurm-wlm slurmdbd"
    else:
        reinstall = f"{manager} install -y {cached}/*/slurm-smd*.deb" if apt else f"dnf install -y {cached}/*/rpmbuild/RPMS/*/*.rpm"
    lines = ["systemctl stop slurmctld slurmd slurmdbd",
             f"{manager} remove -y 'slurm-smd*'" if apt else "dnf remove -y 'slurm*'",
             reinstall,
             "mariadb -e 'DROP DATABASE IF EXISTS slurm_acct_db; CREATE DATABASE slurm_acct_db'",
             f"mariadb slurm_acct_db < {folder / DB_DUMP}",
             f"rm -rf {state_save}/* /etc/slurm/*",
             f"tar -xzf {folder / STATE_TAR} -C /",
             f"tar -xzf {folder / CONF_TAR} -C /",
             "systemctl start slurmdbd slurmctld slurmd"]
    return "\n".join(f"  {line}" for line in lines)
