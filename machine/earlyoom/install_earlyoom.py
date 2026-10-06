"""Install or check this machine's earlyoom settings from the tracked copies beside this file.

The settings decide whether a runaway GPU job on a unified-memory box is killed or
thrashes the machine until it is power-cycled (gpu-node-1, 2026-10-06 13:51). They
used to live only in /etc, unreviewed and unrecoverable; these tracked copies are the
source of truth and /etc is an install target.

    python3 install_earlyoom.py --check        # exit 1 if /etc differs from the tracked copies
    sudo python3 install_earlyoom.py           # install, daemon-reload, restart earlyoom

Machine-specific: only for boxes where this account has root (not the Turing DGX).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: tracked file -> where it is installed, relative to the filesystem root
TARGETS: dict[str, str] = {
    "earlyoom.default": "etc/default/earlyoom",
    "10-self-protect.conf": "etc/systemd/system/earlyoom.service.d/10-self-protect.conf",
}


def differing(root: Path, src_dir: Path = HERE) -> list[str]:
    """Installed paths whose content differs from (or lacks) the tracked copy."""
    out = []
    for name, rel in TARGETS.items():
        dest = root / rel
        if not dest.exists() or dest.read_bytes() != (src_dir / name).read_bytes():
            out.append(str(dest))
    return out


def install(root: Path, src_dir: Path = HERE) -> None:
    for name, rel in TARGETS.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_dir / name, dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", type=Path, default=Path("/"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    diff = differing(args.root)
    if args.check:
        for d in diff:
            print(f"DIFFERS  {d}")
        if not diff:
            print("ok       installed earlyoom settings match the tracked copies")
        return 1 if diff else 0
    install(args.root)
    if args.root == Path("/"):
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "restart", "earlyoom"], check=True)
        subprocess.run(["journalctl", "-u", "earlyoom", "-n", "4", "--no-pager", "-o", "cat"],
                       check=False)
    print(f"installed {len(TARGETS)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
