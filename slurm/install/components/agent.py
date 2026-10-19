"""The agent service, the state directories it and the job tools share, and the admin group."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

import render

import components

NAME = "agent"
SITE_COPY = "/etc/gpujobs/site.toml"
UNIT = "gpujobs-agent.service"
STICKY = ("jobs", "events", "footprints", "metrics", "requests")  # plan 3.3: each file owned by its writer
FILE_MODE = 0o644


def enabled(site) -> bool:
    return True


def slurm_user(site) -> str:
    """SlurmUser from the rendered slurm.conf (default slurm)."""
    conf = render.render(site, "slurm")["/etc/slurm/slurm.conf"].content
    match = re.search(r"^SlurmUser=(\S+)", conf, re.MULTILINE)
    return match.group(1) if match else "slurm"


def files(site) -> dict[str, render.FileSpec]:
    if components.SITE_PATH is None:
        raise RuntimeError("components.SITE_PATH is not set")
    unit = Template((render.TEMPLATES / NAME / UNIT).read_text()).substitute(
        slurm_user=slurm_user(site), site_file=SITE_COPY, install_dir=site.install_dir
    )
    return {
        SITE_COPY: render.FileSpec(components.SITE_PATH.read_text(), FILE_MODE, "root"),
        f"/etc/systemd/system/{UNIT}": render.FileSpec(unit, FILE_MODE, "root"),
    }


def packages(site) -> list[str]:
    return []


def refusals(site) -> list[str]:
    return []


def post_install(site) -> list[list[str]]:
    state = Path(site.state_dir)
    dirs = [["install", "-d", "-m", "0755", "-o", "root", "-g", "root", str(state)]]
    dirs += [["install", "-d", "-m", "1777", str(state / name)] for name in STICKY]
    user = slurm_user(site)
    dirs.append(["install", "-d", "-m", "0755", "-o", user, "-g", user, str(state / "agent")])
    return [
        ["groupadd", "-f", site.admin_group],
        *dirs,
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", "--now", UNIT],
    ]
