"""Slurm itself: daemons, accounting database, their settings files and the restart-protect drop-ins."""

from __future__ import annotations

import render

import components

NAME = "slurm"
PACKAGES = ["slurm-wlm", "slurmdbd", "mariadb-server", "munge"]
RESTARTS = [["mariadb", "munge", "slurmdbd"], ["slurmctld", "slurmd"]]


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
    if done.returncode != 0 or done.stdout.strip():
        out.append("slurm has jobs (or squeue failed); restarting its daemons now would requeue or kill them")
    return out


def post_install(site) -> list:
    """Restart the database side, give accounting the cluster and its account, then start the controller.

    The records go in before slurmctld starts, the order the test cluster boots in, so slurmctld registers with a
    cluster accounting already knows.
    """
    database, controller = RESTARTS
    return [
        ["systemctl", "daemon-reload"],
        ["systemctl", "restart", *database],
        lambda: _records(site),
        ["systemctl", "restart", *controller],
    ]


def _records(site) -> list[list[str]]:
    components.wait_for_accounting()
    return components.accounting_adds(site)
