"""The agent service, the state directories it and the job tools share, and the admin group."""

from __future__ import annotations

from pathlib import Path
from string import Template

import render

import components

NAME = "agent"
SITE_COPY = "/etc/gpujobs/site.toml"
UNIT = "gpujobs-agent.service"
# plan 3.3: each file owned by its writer; logs, scripts and manifests are where submit.py and urgent.py write a job's
# log, batch script and gate manifest when the caller gives no --log
STICKY = ("jobs", "events", "footprints", "requests", "logs", "scripts", "manifests")
# the agent writes these two itself: agent/ is its heartbeat, metrics/ the per-job node-exporter textfiles (a file any user
# could write there could forge a series); readers (the exporter) need only read access
AGENT_ONLY = ("agent", "metrics")


def enabled(site) -> bool:
    return True


def files(site) -> dict[str, render.FileSpec]:
    if components.SITE_PATH is None:
        raise RuntimeError("components.SITE_PATH is not set")
    unit = Template((render.TEMPLATES / NAME / UNIT).read_text()).substitute(
        slurm_user=render.SLURM_USER, site_file=SITE_COPY, install_dir=site.install_dir
    )
    return {
        SITE_COPY: render.FileSpec(components.SITE_PATH.read_text(), render.FILE_MODE, "root"),
        f"/etc/systemd/system/{UNIT}": render.FileSpec(unit, render.FILE_MODE, "root"),
    }


def packages(site) -> list[str]:
    return []


def refusals(site) -> list[str]:
    return []


def post_install(site) -> list[list[str]]:
    state = Path(site.state_dir)
    dirs = [["install", "-d", "-m", "0755", "-o", "root", "-g", "root", str(state)]]
    user = render.SLURM_USER
    # requests/ belongs to the agent's user: in a sticky directory only a file's owner or the directory's owner may
    # delete it, and the agent must delete the restore-order requests it has carried out.
    dirs += [["install", "-d", "-m", "1777", *(["-o", user, "-g", user] if name == "requests" else []), str(state / name)]
             for name in STICKY]
    dirs += [["install", "-d", "-m", "0755", "-o", user, "-g", user, str(state / name)] for name in AGENT_ONLY]
    return [
        ["groupadd", "-f", site.admin_group],
        *dirs,
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", "--now", UNIT],
    ]
