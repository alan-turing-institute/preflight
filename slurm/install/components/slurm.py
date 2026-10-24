"""Slurm itself: daemons, accounting database, their settings files and the restart-protect drop-ins."""

from __future__ import annotations

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
    version, sha256 = site.slurm_build
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
