"""The installer's components: one module each, loaded by name in a fixed order.

A component module defines `NAME`, `enabled(site)`, `files(site)`, `packages(site)`, `refusals(site)` and
`post_install(site)`; optionally `apt_sources(site)` and `slurm_conf_lines(site)`. Every external command any
component runs goes through `run`, so a test swaps the one runner and nothing real is touched.
"""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from site_config import Site

ORDER = ["slurm", "tools", "agent", "lease", "metrics", "portal"]
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]

#: the site file as given on the command line; the agent component installs a copy of it
SITE_PATH: Path | None = None


def _real_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


_runner: Runner = _real_run


def set_runner(runner: Runner | None) -> None:
    """Replace the command runner (None restores the real one)."""
    global _runner
    _runner = runner or _real_run


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return _runner(argv)


def load(site: Site, names: list[str] | None = None) -> list[ModuleType]:
    """The enabled components among `names` (default all), in ORDER; a module not written yet is skipped."""
    out = []
    for name in ORDER:
        if names is not None and name not in names:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
            continue
        if module.enabled(site):
            out.append(module)
    return out


def slurm_conf_extra(site: Site) -> str:
    """The lines enabled components contribute to slurm.conf, joined."""
    parts = (getattr(m, "slurm_conf_lines", lambda _s: "")(site) for m in load(site))
    return "".join(p if p.endswith("\n") else p + "\n" for p in parts if p)
