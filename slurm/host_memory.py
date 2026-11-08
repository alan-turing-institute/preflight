"""This host's memory as `/proc/meminfo` reports it, in bytes.

Kept apart from `gpu_jobs`, which reads the site file when it is imported: the installer's `detect` parses meminfo on a
machine that has no site file yet. `gpu_jobs` re-exports these names for the tools.
"""

from __future__ import annotations

from pathlib import Path

KB_BYTES = 1024  # meminfo's "kB" is 2^10 bytes
SLURM_MB = 2**20  # slurm's "MB" (RealMemory, `--mem`, memory limits) is 2^20 bytes
MEMINFO = Path("/proc/meminfo")


def meminfo_text(text: str) -> dict[str, int]:
    """Every `Key: N kB` line of `/proc/meminfo` text, in bytes. Raises ValueError when MemTotal or MemAvailable is
    missing (an unreadable host is unknown, never zero)."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        number, _, unit = rest.strip().partition(" ")
        if unit.strip() == "kB" and number.isdigit():
            found[key] = int(number) * KB_BYTES
    if not {"MemTotal", "MemAvailable"} <= found.keys():
        raise ValueError("meminfo has no MemTotal or no MemAvailable line")
    return found


def meminfo(path: Path = MEMINFO) -> dict[str, int]:
    """`meminfo_text` of a meminfo file."""
    return meminfo_text(path.read_text(encoding="ascii"))
