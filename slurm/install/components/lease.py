"""The GPU lease keeper: a user unit of site.lease.keeper_user, written into that user's home."""

from __future__ import annotations

import pwd
from pathlib import Path
from string import Template

import render

import components

NAME = "lease"
UNIT = "slurm-gpu-lease-keeper.service"
CLIENT_CHECK = "from localmodels.lease import GpuLeaseClient"
FILE_MODE = 0o644


def enabled(site) -> bool:
    return site.lease is not None


def _home(user: str) -> Path:
    try:
        return Path(pwd.getpwnam(user).pw_dir)
    except KeyError:
        return Path("/home") / user


def files(site) -> dict[str, render.FileSpec]:
    user = site.lease.keeper_user
    unit = Template((render.TEMPLATES / NAME / UNIT).read_text()).substitute(
        keeper_user=user, python=site.lease.python, install_dir=site.install_dir
    )
    return {str(_home(user) / ".config/systemd/user" / UNIT): render.FileSpec(unit, FILE_MODE, user)}


def packages(site) -> list[str]:
    return []


def refusals(site) -> list[str]:
    python = str(site.lease.python)
    try:
        done = components.run([python, "-c", CLIENT_CHECK])
    except FileNotFoundError:
        return [f"lease.python {python} does not exist"]
    if done.returncode != 0:
        return [f"{python} cannot run `{CLIENT_CHECK}`; point lease.python at an interpreter that has localmodels"]
    return []


def post_install(site) -> list[list[str]]:
    # `systemctl --user --machine=USER@.host` reaches that user's bus (systemctl(1), -M/--machine); it needs the
    # user's systemd instance running (linger on, or a login session).
    user = ["systemctl", "--user", f"--machine={site.lease.keeper_user}@.host"]
    return [[*user, "daemon-reload"], [*user, "enable", "--now", UNIT], [*user, "restart", UNIT]]
