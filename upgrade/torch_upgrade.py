#!/usr/bin/env python3
"""Upgrade a project's PyTorch in two cheap steps: build a side venv, then check it on the GPU.

Policy and history: ~/.claude/software-upgrade-policy.md ("A project upgrade").

    torch_upgrade.py build OLD_VENV NEW_VENV --torch 2.15.0 --cuda cu132
        NEW_VENV = OLD_VENV's packages with only the PyTorch stack replaced, in ONE resolve with
        exact pins (installing in two steps let an old `setuptools` pin silently pull torch back
        to 2.12.1, 2026-10-29). Refuses if any other package changed version.
    torch_upgrade.py check OLD_VENV NEW_VENV --project P --workdir DIR -- PROBE [ARGS...]
        One slurm GPU job runs `<venv>/bin/python PROBE ARGS --out FILE` under both venvs; prints
        each version's step time and the largest relative difference of the probe's values.
        The probe writes JSON {"seconds_per_step": float, "values": [float, ...]} and takes seconds
        (for example, a short training run that times its steps).

A reboot is never needed: the PyTorch wheel brings its own CUDA libraries. Only a wheel built for
a CUDA newer than the driver supports can fail, and `check` shows that (CUDA 13.2 wheels ran on
driver 580.178.04, a CUDA 13.0 driver, 2026-10-29).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import upgrade_common as common
from upgrade_common import freeze, run

PYPI = "https://pypi.org/simple"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{cuda}"
# Replaced together with torch: the wheel pins its own CUDA libraries and Triton; companions are
# re-resolved against the new torch; setuptools because torch 2.15 needs >= 77.
STACK = re.compile(
    r"^(torch|torchvision|torchaudio|triton|nvidia-|cuda-|setuptools)", re.IGNORECASE
)
COMPANIONS = ("torchvision", "torchaudio")


def build(old: Path, new: Path, torch: str, cuda: str) -> None:
    if new.exists():
        sys.exit(f"REFUSED: {new} exists; remove it or choose another path")
    before = freeze(old)
    keep = [
        line
        for name, line in before.items()
        if not STACK.match(name) and not line.startswith("-e ")
    ]
    editables = [
        line.removeprefix("-e ") for line in before.values() if line.startswith("-e ")
    ]
    companions = [name for name in COMPANIONS if name in before]
    reqs = new.parent / f"{new.name}.requirements.txt"
    new.parent.mkdir(parents=True, exist_ok=True)
    reqs.write_text("\n".join(keep) + "\n")
    python = str(new / "bin/python")
    version = re.search(
        r"Python (\S+)", run([str(old / "bin/python"), "--version"])
    ).group(1)
    run(["uv", "venv", str(new), "--python", version, "-q"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-sources",
            "--index-url",
            PYPI,
            "--extra-index-url",
            PYTORCH_INDEX.format(cuda=cuda),
            "--index-strategy",
            "unsafe-best-match",
            "-r",
            str(reqs),
            f"torch=={torch}+{cuda}",
            *companions,
            "setuptools",
        ]
    )
    for path in editables:
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-sources",
                "--no-deps",
                "-e",
                path,
            ]
        )
    if common.has_hostguard(old):
        common.install_hostguard(new)

    after = freeze(new)
    moved = [
        f"{before[n]} -> {after.get(n, 'missing')}"
        for n in before
        if not STACK.match(n) and before[n] != after.get(n)
    ]
    if moved:
        sys.exit(
            "REFUSED: packages outside the PyTorch stack changed:\n  "
            + "\n  ".join(moved)
        )
    if after.get("torch") != f"torch=={torch}+{cuda}":
        sys.exit(
            f"REFUSED: installed {after.get('torch')}, asked for torch=={torch}+{cuda}"
        )
    for name in (
        "torch",
        *companions,
        "triton",
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
        "cuda-toolkit",
    ):
        if name in before or name in after:
            print(f"{name}: {before.get(name, '-')} -> {after.get(name, '-')}")
    size = run(["du", "-sh", str(new)]).split()[0]
    print(f"built {new} ({size}); give it a DISK.md line in the project")


def check(old: Path, new: Path, project: str, workdir: Path, probe: list[str]) -> None:
    out = new.parent / f"{new.name}.check"
    out.mkdir(parents=True, exist_ok=True)
    common.submit_and_wait(
        project,
        f"PyTorch upgrade check: {new.name} against {old}",
        "torch-upgrade-check",
        common.probe_legs(old, new, " ".join(probe), out),
        workdir,
        out / "job.log",
        gpu=True,
        mem="16G",
    )
    results = common.read_probes(out)
    if results is None:
        sys.exit(f"FAILED: a leg wrote nothing; read {out / 'job.log'}")
    a, b = results
    diff = common.largest_relative_difference(a["values"], b["values"])
    print(
        f"seconds per step: old {a['seconds_per_step']:.4f}, new {b['seconds_per_step']:.4f} "
        f"(new/old {b['seconds_per_step'] / a['seconds_per_step']:.2f})"
    )
    print(
        "values: identical"
        if diff == 0
        else f"values: largest relative difference {diff:.2e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("old", type=Path)
    b.add_argument("new", type=Path)
    b.add_argument("--torch", required=True, help="e.g. 2.15.0")
    b.add_argument("--cuda", required=True, help="PyTorch index tag, e.g. cu132")
    c = sub.add_parser("check")
    c.add_argument("old", type=Path)
    c.add_argument("new", type=Path)
    c.add_argument("--project", required=True)
    c.add_argument("--workdir", type=Path, required=True)
    c.add_argument(
        "probe", nargs="+", help="after --: the probe script and its arguments"
    )
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.old.resolve(), args.new.resolve(), args.torch, args.cuda)
    else:
        check(
            args.old.resolve(),
            args.new.resolve(),
            args.project,
            args.workdir.resolve(),
            args.probe,
        )


if __name__ == "__main__":
    main()
