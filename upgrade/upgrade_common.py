"""Pieces shared by `torch_upgrade.py` (one PyTorch upgrade, by hand) and `weekly.py` (every project, on a timer).

Nothing here decides anything: running a command, listing a venv, putting hostguard into a venv, running a
script under two venvs in ONE slurm job and comparing what it wrote. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PREFLIGHT = Path(__file__).resolve().parents[1]
HOSTGUARD = PREFLIGHT / "hostguard"
SLURM = PREFLIGHT / "slurm"


def run(cmd: list[str], **kwargs) -> str:
    """Run a command, print it, return its stdout; a non-zero exit raises CalledProcessError."""
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd, check=True, text=True, capture_output=True, **kwargs
    ).stdout


def freeze(venv: Path) -> dict[str, str]:
    """name -> pinned line, editable installs keyed by their path."""
    lines = run(
        ["uv", "pip", "freeze", "--python", str(venv / "bin/python")]
    ).splitlines()
    return {
        line.split("==")[0].lower() if "==" in line else line: line
        for line in lines
        if line
    }


def has_hostguard(venv: Path) -> bool:
    return next(venv.glob("lib/python*/site-packages/hostguard.pth"), None) is not None


def install_hostguard(venv: Path) -> None:
    run([sys.executable, str(HOSTGUARD / "install_pth.py"), str(venv)])


def submit_and_wait(
    project: str,
    what: str,
    name: str,
    command: str,
    workdir: Path,
    log: Path,
    *,
    gpu: bool,
    mem: str,
    cpus: int | None = None,
) -> str:
    """Submit `bash -c command` as one slurm job (class restart: never pre-empted), wait, return the job id.

    The caller reads the files the command wrote: a job's exit state alone says nothing about what it measured.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(SLURM / "submit.py"),
        "--project",
        project,
        "--what",
        what,
        "--class",
        "restart",
        "--name",
        name,
        *(["--gpu"] if gpu else []),
        "--mem",
        mem,
        *(["--cpus", str(cpus)] if cpus else []),
        "--workdir",
        str(workdir),
        "--log",
        str(log),
        "--",
        "bash",
        "-c",
        command,
    ]
    job = run(argv).split()[-1]
    subprocess.run(
        [sys.executable, str(SLURM / "wait.py"), job, "--poll", "5"], check=False
    )
    return job


def probe_legs(old: Path, new: Path, probe: str, out: Path) -> str:
    """The shell command that runs `<venv>/bin/python PROBE --out FILE` under both venvs, old first."""
    legs = [
        f"{venv}/bin/python {probe} --out {out}/{tag}.json"
        for tag, venv in (("old", old), ("new", new))
    ]
    return f"export PYTHONPATH={HOSTGUARD}; " + " && ".join(legs)


def read_probes(out: Path) -> tuple[dict, dict] | None:
    """The old and new probe results, or None when either leg wrote nothing."""
    paths = [out / "old.json", out / "new.json"]
    if not all(path.exists() for path in paths):
        return None
    old, new = (json.loads(path.read_text()) for path in paths)
    return old, new


def largest_relative_difference(a: list[float], b: list[float]) -> float:
    """0.0 exactly when the two lists are identical; raises ValueError when their lengths differ."""
    return max(
        (abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(a, b, strict=True)),
        default=0.0,
    )


def timed(out: Path, cwd: Path, command: list[str]) -> int:
    """Run `command` (an argv) in `cwd`, output to OUT/output.log, {"returncode", "seconds"} to OUT/result.json.

    This is what a slurm job runs for a tests check: the job's own exit state says nothing about how long it took.
    """
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (out / "output.log").open("w") as log:
        code = subprocess.run(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode
    seconds = time.monotonic() - start
    (out / "result.json").write_text(
        json.dumps({"returncode": code, "seconds": seconds}) + "\n"
    )
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=timed.__doc__)
    ap.add_argument("verb", choices=["timed"])
    ap.add_argument("out", type=Path)
    ap.add_argument("cwd", type=Path)
    ap.add_argument("command", nargs="+")
    parsed = ap.parse_args()
    sys.exit(timed(parsed.out, parsed.cwd, parsed.command))
