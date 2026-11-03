"""Print a draft site file (TOML) for this machine, from read-only probes.

Usage: python3 slurm/install/detect.py [--reserve-mb N]
"""

import argparse
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # slurm/, a standalone driver like footprint.py

import cores  # noqa: E402 - path set just above

DEFAULT_RESERVE_MB = 12288  # the host memory guard's 12 GiB
NO_VALUE = "[N/A]"
GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"]

Runner = Callable[[list[str]], str | None]


def run_command(argv: list[str]) -> str | None:
    """stdout of a read-only command, or None when it is missing or fails."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def parse_mem_total_mb(meminfo: str) -> int:
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    if not match:
        raise ValueError("no MemTotal line in meminfo")
    return int(match.group(1)) // 1024


def parse_gpus(csv_text: str) -> list[dict]:
    """Rows of `index, uuid, name, memory.total`; memory is kept as text."""
    gpus = []
    for line in csv_text.strip().splitlines():
        index, uuid, name, memory = (field.strip() for field in line.split(",", 3))
        gpus.append({
            "index": int(index),
            "uuid": uuid,
            "device": f"/dev/nvidia{index}",
            "type": name.lower().replace(" ", "-"),
            "memory": memory,
        })
    return gpus


def is_unified_memory(gpus: list[dict]) -> bool:
    return bool(gpus) and all(gpu["memory"] == NO_VALUE for gpu in gpus)


def parse_os_name(os_release: str) -> str:
    match = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', os_release, re.MULTILINE)
    return match.group(1) if match else "unknown"


def probe_gpus(run: Runner) -> list[dict]:
    out = run(GPU_QUERY)
    return parse_gpus(out) if out else []


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(
    *, hostname: str, cpus: int, real_memory_mb: int, gpus: list[dict], slurm_version: str, os_name: str,
    core_groups: Mapping[int, list[int]] | None = None,
) -> str:
    lines = [
        f"# Draft site file for {hostname}, from detect.py. Review before installing.",
        f"# OS: {os_name}",
        f"# Slurm: {slurm_version}",
        f"name = {quote(hostname)}",
        'state_dir = "/var/lib/gpujobs"',
        'install_dir = "/opt/gpujobs"',
        'admin_group = "gpujobs-admin"',
        f"unified_memory = {str(is_unified_memory(gpus)).lower()}",
        "hostguard = false",
        "dgx_gate = false",
        "notify_command = []",
        "",
        "[[nodes]]",
        f"name = {quote(hostname)}",
        f"cpus = {cpus}",
        f"real_memory_mb = {real_memory_mb}",
    ]
    for mhz, group in (core_groups or {}).items():  # only given when the cores differ
        lines.append(f"# top speed {mhz} MHz: {len(group)} cores ({cores.cpu_list(group)})")
    if core_groups:
        lines.append("# slurm hands cores out without telling these apart; `slurm/cores.py` lists them, each run records which it got")
    if not gpus:
        lines.append("# no GPUs detected (nvidia-smi missing or listed none)")
    for gpu in gpus:
        lines += [
            "",
            "[[nodes.gpus]]",
            f"index = {gpu['index']}",
            f"uuid = {quote(gpu['uuid'])}",
            f"device = {quote(gpu['device'])}",
            f"type = {quote(gpu['type'])}",
        ]
    lines += ["", "[metrics]", 'mode = "off"', "", "[portal]", 'mode = "off"',
              'auth = "dex-local"', 'server_name = ""', ""]
    return "\n".join(lines)


def detect(
    run: Runner = run_command,
    meminfo_path: Path = Path("/proc/meminfo"),
    os_release_path: Path = Path("/etc/os-release"),
    hostname: str | None = None,
    cpus: int | None = None,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    cpu_root: Path = cores.CPU_ROOT,
) -> str:
    speeds = cores.CoreSpeeds.read(cpu_root)
    os_text = os_release_path.read_text() if os_release_path.exists() else ""
    return render(
        hostname=hostname or socket.gethostname(),
        cpus=cpus or os.cpu_count() or 1,
        real_memory_mb=parse_mem_total_mb(meminfo_path.read_text()) - reserve_mb,
        gpus=probe_gpus(run),
        slurm_version=(run(["sinfo", "-V"]) or "not found").strip(),
        os_name=parse_os_name(os_text),
        core_groups=speeds.groups() if speeds.mixed else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-mb", type=int, default=DEFAULT_RESERVE_MB,
                        help="memory left to the host, subtracted from MemTotal (default 12288)")
    args = parser.parse_args(argv)
    sys.stdout.write(detect(reserve_mb=args.reserve_mb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
