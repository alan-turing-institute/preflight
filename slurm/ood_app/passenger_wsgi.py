"""Passenger entry point for the GPU jobs app (Open OnDemand runs it as the viewing user).

The portal installs this app as a symlink to `<install_dir>/slurm/ood_app`, so the tools it imports
(`gpu_jobs`, `jobs_view`, `site_config`) are the installed `slurm/` directory, this directory's parent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import application

__all__ = ["application"]
