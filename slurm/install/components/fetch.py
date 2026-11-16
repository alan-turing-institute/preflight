"""Fetch-back (D52): a user timer that runs `fetch_back.py run`, on a site that sends jobs to another (`[remote.*]`).

The units go to /etc/systemd/user/ (root's) like the lease keeper's; each user enables the timer for themselves
(`gpujobs-fetch-back enable`), since the fetch runs with that user's own ssh login and writes into their folders.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

import render

NAME = "fetch"
UNITS = ("gpujobs-fetch.service", "gpujobs-fetch.timer")
UNIT_DIR = Path("/etc/systemd/user")


def enabled(site) -> bool:
    return bool(site.remotes)


def files(site) -> dict[str, render.FileSpec]:
    values = {"install_dir": site.install_dir, "every_s": site.tuning.fetch_every_s}
    return {
        str(UNIT_DIR / unit): render.FileSpec(
            Template((render.TEMPLATES / NAME / unit).read_text()).substitute(values), render.FILE_MODE, "root"
        )
        for unit in UNITS
    }


def packages(site) -> list[str]:
    return ["rsync"]


def refusals(site) -> list[str]:
    return []


def post_install(site) -> list[list[str]]:
    print("[install] fetch-back: each user who sends jobs to another site runs `gpujobs-fetch-back enable` once")
    return []
