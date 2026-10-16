"""Install or check this box's slurm settings from the tracked copies beside this file.

Design: design notes (revision 2). The copies here are the source of truth;
/etc is an install target, as for earlyoom (machine/earlyoom/install_earlyoom.py, whose copy-and-compare
this reuses).

    python3 install_slurm.py --check      # exit 1 if /etc or the user unit differs from the tracked copies
    sudo python3 install_slurm.py         # install, daemon-reload, restart slurmctld and slurmd
    python3 install_slurm.py --user       # install and (re)start the lease keeper's user unit

A restart is safe only with no slurm job running: `squeue` must be empty, or the install refuses.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "earlyoom"))

from install_earlyoom import differing, install  # noqa: E402

DROPIN = "10-restart-protect.conf"
#: tracked file -> where it is installed, relative to the filesystem root
TARGETS: dict[str, str] = {
    "slurm.conf": "etc/slurm/slurm.conf",
    "cgroup.conf": "etc/slurm/cgroup.conf",
    "gres.conf": "etc/slurm/gres.conf",
    "slurmdbd.conf": "etc/slurm/slurmdbd.conf",
    "mariadb-slurm.cnf": "etc/mysql/mariadb.conf.d/90-slurm.cnf",
    **{
        f"{DROPIN}@{unit}": f"etc/systemd/system/{unit}.service.d/{DROPIN}"
        for unit in ("slurmctld", "slurmd", "munge", "slurmdbd")
    },
}
KEEPER = "slurm-gpu-lease-keeper.service"


def sources() -> Path:
    """A directory where every TARGETS key names a file (the drop-in once per unit)."""
    staged = Path("/tmp") / f"install_slurm_{Path.home().name}"
    staged.mkdir(exist_ok=True)
    for name in TARGETS:
        (staged / name).write_bytes((HERE / name.split("@")[0]).read_bytes())
    return staged


def queue_empty() -> bool:
    done = subprocess.run(["squeue", "-h"], capture_output=True, text=True, check=False)
    return done.returncode == 0 and not done.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--user", action="store_true", help="install and start the lease keeper's user unit")
    args = ap.parse_args(argv)
    # The keeper is the invoking user's unit, also when the check runs under sudo.
    home = Path(os.path.expanduser(f"~{os.environ['SUDO_USER']}")) if os.environ.get("SUDO_USER") else Path.home()
    user_unit = {KEEPER: f"{home.relative_to('/')}/.config/systemd/user/{KEEPER}"}
    if args.check:
        try:
            diff = differing(Path("/"), sources(), TARGETS) + differing(Path("/"), HERE, user_unit)
        except PermissionError as error:  # slurmdbd.conf is 0600 slurm, as slurmdbd requires
            print(f"cannot read {error.filename}: run the check with sudo", file=sys.stderr)
            return 2
        for d in diff:
            print(f"DIFFERS  {d}")
        if not diff:
            print("ok       installed slurm settings match the tracked copies")
        return 1 if diff else 0
    if args.user:
        install(Path("/"), HERE, user_unit)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", KEEPER], check=True)
        subprocess.run(["systemctl", "--user", "restart", KEEPER], check=True)
        print(f"installed and started {KEEPER}")
        return 0
    if not queue_empty():
        print("REFUSED: slurm has jobs; restarting its daemons now would requeue or kill them", file=sys.stderr)
        return 1
    install(Path("/"), sources(), TARGETS)
    dbd = Path("/etc/slurm/slurmdbd.conf")
    shutil.chown(dbd, "slurm", "slurm")
    dbd.chmod(0o600)  # slurmdbd refuses a config readable by others
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "restart", "mariadb", "munge", "slurmdbd"], check=True)
    subprocess.run(["systemctl", "restart", "slurmctld", "slurmd"], check=True)
    subprocess.run(["scontrol", "show", "config"], capture_output=True, check=True)
    print(f"installed {len(TARGETS)} file(s); munge, slurmctld and slurmd restarted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
