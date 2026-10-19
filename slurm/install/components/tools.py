"""The command-line tools: the repo's slurm/ tree copied under install_dir, plus a shim per tool in /usr/local/bin."""

from __future__ import annotations

from pathlib import Path

import render

NAME = "tools"
SLURM_DIR = Path(__file__).resolve().parents[2]
SHIM_DIR = "/usr/local/bin"
TOOLS = ["submit", "status", "pause", "resume", "kill", "urgent", "wait", "footprint", "audit"]
SHIM = '#!/bin/sh\nexec /usr/bin/python3 {install_dir}/slurm/{tool}.py "$@"\n'
FILE_MODE = 0o644
SHIM_MODE = 0o755


def enabled(site) -> bool:
    return True


def _tree() -> list[Path]:
    """Every .py directly in slurm/, and everything under slurm/ood_app/; never tests or caches."""
    found = sorted(SLURM_DIR.glob("*.py"))
    ood = SLURM_DIR / "ood_app"
    if ood.is_dir():
        found += sorted(p for p in ood.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return found


def _mode(path: Path) -> int:
    """Executable in the repository (the app's bin/python launcher) stays executable; everything else is 0644."""
    return SHIM_MODE if path.stat().st_mode & 0o100 else FILE_MODE


def files(site) -> dict[str, render.FileSpec]:
    dest = site.install_dir / "slurm"
    out = {
        str(dest / p.relative_to(SLURM_DIR)): render.FileSpec(p.read_text(), _mode(p), "root") for p in _tree()
    }
    for tool in TOOLS:
        shim = SHIM.format(install_dir=site.install_dir, tool=tool)
        out[f"{SHIM_DIR}/gpujobs-{tool}"] = render.FileSpec(shim, SHIM_MODE, "root")
    return out


def packages(site) -> list[str]:
    return []


def refusals(site) -> list[str]:
    return []


def post_install(site) -> list[list[str]]:
    return []
