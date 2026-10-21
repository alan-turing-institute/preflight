"""Slurm itself: daemons, accounting database, their settings files and the restart-protect drop-ins."""

from __future__ import annotations

import render

import components

NAME = "slurm"
PACKAGES = ["slurm-wlm", "slurmdbd", "mariadb-server", "munge"]
RESTARTS = [["mariadb", "munge"], ["slurmdbd"], ["slurmctld", "slurmd"]]
# slurmdbd logs in to MariaDB as the OS user `slurm` by unix_socket (templates/slurm/slurmdbd.conf), so no password
# exists anywhere. A fresh MariaDB has no such login: it is made here, before slurmdbd first starts.
DB_LOGIN = ("CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED VIA unix_socket; "
            "GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';")


def enabled(site) -> bool:
    return True


def files(site) -> dict[str, render.FileSpec]:
    return render.render(site, NAME, extra={"slurm_conf": components.slurm_conf_extra(site)})


def packages(site) -> list[str]:
    return list(PACKAGES)


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
    """slurmctld is down and has never saved job state (StateSaveLocation in templates/slurm/slurm.conf), so no job
    exists that a restart could requeue or kill."""
    if (components.ROOT / "var/spool/slurmctld/job_state").exists():
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
        ["systemctl", "restart", *controller],
    ]


def _records(site) -> list[list[str]]:
    components.wait_for_accounting()
    return components.accounting_adds(site)
