"""The agent service, the state directories it and the job tools share, and the admin group."""

from __future__ import annotations

import socket
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
# the agent writes these itself: agent/ is its heartbeat, metrics/ the per-job node-exporter textfiles (a file any user
# could write there could forge a series; readers, the exporter, need only read access), remote/ each part-time machine's
# copied-back job state (B5b: a file any user could write there could forge a record)
AGENT_ONLY = ("agent", "metrics", "remote")


def enabled(site) -> bool:
    return True


def site_copy(site) -> dict[str, render.FileSpec]:
    """The site file as the tools read it (`/etc/gpujobs/site.toml`); a part-time machine's wrapper needs it too."""
    if components.SITE_PATH is None:
        raise RuntimeError("components.SITE_PATH is not set")
    return {SITE_COPY: render.FileSpec(components.SITE_PATH.read_text(), render.FILE_MODE, "root")}


def files(site) -> dict[str, render.FileSpec]:
    unit = Template((render.TEMPLATES / NAME / UNIT).read_text()).substitute(
        slurm_user=render.SLURM_USER, site_file=SITE_COPY, install_dir=site.install_dir
    )
    render_py = Path(render.__file__)  # the agent rewrites part_time_nodes.conf with the installer's own line format
    return {
        str(Path(site.install_dir) / "slurm" / "install" / render_py.name): render.FileSpec(
            render_py.read_text(), render.FILE_MODE, "root"),
        **site_copy(site),
        f"/etc/systemd/system/{UNIT}": render.FileSpec(unit, render.FILE_MODE, "root"),
    }


def packages(site) -> list[str]:
    return ["rsync"] if site.part_time_nodes and not site.shared_state else []  # the copy-back of job state


def refusals(site, hostname: str | None = None) -> list[str]:
    """On a unified-memory site the agent's memory guard watches only the agent's own node (D44), so a GPU node
    elsewhere would run unguarded: refused at install rather than installed half-watched."""
    if not site.unified_memory:
        return []
    here = (hostname or socket.gethostname()).split(".")[0]
    return [f"node {node.name} has GPUs on a unified-memory site but is not this machine: the memory guard watches only "
            "the agent's own node (several GPU nodes are a later change)" for node in site.nodes if node.gpus and node.name != here]


def state_dir_commands(site) -> list[list[str]]:
    """Make the state directory and its subdirectories with their owners and modes (also on a part-time machine, where
    a job's wrapper writes its records)."""
    state = Path(site.state_dir)
    dirs = [["install", "-d", "-m", "0755", "-o", "root", "-g", "root", str(state)]]
    user = render.SLURM_USER
    # requests/ belongs to the agent's user: in a sticky directory only a file's owner or the directory's owner may
    # delete it, and the agent must delete the restore-order requests it has carried out.
    dirs += [["install", "-d", "-m", "1777", *(["-o", user, "-g", user] if name == "requests" else []), str(state / name)]
             for name in STICKY]
    return dirs + [["install", "-d", "-m", "0755", "-o", user, "-g", user, str(state / name)] for name in AGENT_ONLY]


def post_install(site) -> list[list[str]]:
    return [
        ["groupadd", "-f", site.admin_group],
        *([["groupadd", "-f", site.service.group]] if site.service else []),
        *state_dir_commands(site),
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", UNIT],
        # restart, not `enable --now`: an upgrade must swap the running agent for the new code (2026-10-21: the
        # old agent kept running after an apply until it was restarted by hand); a stopped one is started
        ["systemctl", "restart", UNIT],
    ]
