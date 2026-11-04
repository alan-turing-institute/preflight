#!/usr/bin/env python3
"""Which of this host's cores run faster than others (hybrid CPUs: Arm big.LITTLE, Intel performance/efficiency cores).

    slurm/cores.py          # one line per top speed: how many cores, which ones

Slurm hands a job the lowest free cores without knowing they differ. Measured 2026-11-03 on an Arm host: one thread of a small
torch fit took 6-8 s on a Cortex-X925 core (3.9 GHz) and 27-43 s on a Cortex-A725 core (2.8 GHz), so the same job was 4 to 6
times slower depending on which cores it was given. This module only reads the facts (each core's top speed, Linux cpufreq
`cpuinfo_max_freq`, which separates the two kinds cleanly; `cpu_capacity` does not) so that a run can record what it received.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

CPU_ROOT = Path("/sys/devices/system/cpu")
KHZ_PER_MHZ = 1000


def cpu_list(cpus: Iterable[int]) -> str:
    """CPU numbers as ranges, the way `taskset -c` and slurm print them: {0, 1, 2, 8, 9} -> "0-2,8-9"."""
    ranges: list[list[int]] = []
    for cpu in sorted(set(cpus)):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(low) if low == high else f"{low}-{high}" for low, high in ranges)


@dataclass(frozen=True)
class CoreSpeeds:
    """The top speed (kHz) of each OS CPU number this host reports one for."""

    khz: Mapping[int, int]

    @classmethod
    def read(cls, root: Path = CPU_ROOT) -> CoreSpeeds:
        found: dict[int, int] = {}
        for path in root.glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
            with contextlib.suppress(OSError, ValueError):
                found[int(path.parent.parent.name.removeprefix("cpu"))] = int(path.read_text())
        return cls(found)

    @property
    def mixed(self) -> bool:
        """Whether the cores that report a speed do not all have the same one."""
        return len(set(self.khz.values())) > 1

    def groups(self, cpus: Iterable[int] | None = None) -> dict[int, list[int]]:
        """Top speed in MHz -> the CPUs (of `cpus`, else all that report one) that have it, slowest group first."""
        wanted = self.khz if cpus is None else set(cpus) & set(self.khz)
        out: dict[int, list[int]] = {}
        for cpu in sorted(wanted):
            out.setdefault(self.khz[cpu] // KHZ_PER_MHZ, []).append(cpu)
        return dict(sorted(out.items()))

    def fast_core_floor(self) -> int:
        """The fewest cores that cannot all be slow: one more than the cores that are not of the fastest kind, so any
        allocation of this size or more holds a fast core whichever cores slurm hands out. 0 when the cores are alike."""
        if not self.mixed:
            return 0
        return len(self.khz) - len(next(reversed(self.groups().values()))) + 1

    def received(self, cpus: Iterable[int]) -> dict[str, object] | None:
        """What a run allowed `cpus` was given, for its record: {"cpus": "0-3,8-11", "cores_by_max_mhz": {"2808": 6, "3900": 2}}.
        None on a host whose cores are all alike (or report no speed): there is nothing to tell apart."""
        if not self.mixed:
            return None
        return {"cpus": cpu_list(cpus),
                "cores_by_max_mhz": {str(mhz): len(group) for mhz, group in self.groups(cpus).items()}}


def main() -> int:
    speeds = CoreSpeeds.read()
    if not speeds.khz:
        print("no core reports a top speed (no cpufreq under /sys/devices/system/cpu)")
        return 1
    for mhz, group in speeds.groups().items():
        print(f"{len(group):>3} cores at {mhz} MHz: {cpu_list(group)}")
    print("all cores alike" if not speeds.mixed else "mixed: slurm does not tell these apart when it hands out cores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
