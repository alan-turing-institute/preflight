"""The command-line tools: the repo's slurm/ tree copied under install_dir, plus a shim per tool in /usr/local/bin.

hostguard and the DGX gate go beside it when the site turns them on: the job wrapper and `submit.py` find them
next to their own `slurm/` directory.
"""

from __future__ import annotations

from pathlib import Path

import render

NAME = "tools"
SLURM_DIR = Path(__file__).resolve().parents[2]
REPO = SLURM_DIR.parent
SHIM_DIR = "/usr/local/bin"
TOOLS = ["submit", "status", "pause", "resume", "kill", "urgent", "wait", "footprint", "audit"]
SHIM = '#!/bin/sh\nexec /usr/bin/python3 {install_dir}/slurm/{tool}.py "$@"\n'


def enabled(site) -> bool:
    return True


def _tree(site) -> list[Path]:
    """Every .py directly in slurm/, and everything under slurm/ood_app/; never tests or caches. hostguard when the
    site uses it or the DGX gate (whose leg 4 reads it), and dgx/ with the gate."""
    found = sorted(SLURM_DIR.glob("*.py"))
    ood = SLURM_DIR / "ood_app"
    if ood.is_dir():
        found += sorted(p for p in ood.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if site.hostguard or site.dgx_gate:
        found.append(REPO / "hostguard" / "hostguard.py")
    if site.dgx_gate:
        found += sorted(p for p in (REPO / "dgx").iterdir() if p.is_file())
    return found


def _mode(path: Path) -> int:
    """Executable in the repository (the app's bin/python launcher) stays executable; everything else is 0644."""
    return render.EXEC_MODE if path.stat().st_mode & 0o100 else render.FILE_MODE


def files(site) -> dict[str, render.FileSpec]:
    out = {
        str(site.install_dir / p.relative_to(REPO)): render.FileSpec(p.read_text(), _mode(p), "root")
        for p in _tree(site)
    }
    for tool in TOOLS:
        shim = SHIM.format(install_dir=site.install_dir, tool=tool)
        out[f"{SHIM_DIR}/gpujobs-{tool}"] = render.FileSpec(shim, render.EXEC_MODE, "root")
    return out


def packages(site) -> list[str]:
    return []


def refusals(site) -> list[str]:
    return []


def post_install(site) -> list[list[str]]:
    return []
