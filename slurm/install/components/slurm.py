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
    """A restart requeues or kills running jobs, so the queue must be empty (an absent `squeue` means no slurm yet)."""
    try:
        done = components.run(["squeue", "-h"])
    except FileNotFoundError:
        return []
    if done.returncode != 0 or done.stdout.strip():
        return ["slurm has jobs (or squeue failed); restarting its daemons now would requeue or kill them"]
    return []


def post_install(site) -> list[list[str]]:
    return [["systemctl", "daemon-reload"], *(["systemctl", "restart", *group] for group in RESTARTS)]
