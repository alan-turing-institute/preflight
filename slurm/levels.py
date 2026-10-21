"""The levels a waiting job can have, and the priority each gives it (D18,
`plan notes`).

Its own module, stdlib only and with nothing done on import, so the installer (which creates the levels in slurm's
accounting) can import it where `gpu_jobs`, which reads the site file on import, cannot be.
"""

from __future__ import annotations

from enum import StrEnum


class Level(StrEnum):
    """Where a job waits among the others: a slurm QOS name, open to everyone. It orders the waiting line only and never
    stops a running job; `urgent` (its own partition) is the one push-out."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


#: The highest priority `scontrol update` takes (`parse_uint32` refuses NO_VAL, 0xfffffffe, and above).
_HIGHEST = 0xFFFF_FFFD
#: Priorities each level spans. All three bands sit above slurm's TOP_PRIORITY (0xffff0000), the top of every number
#: priority/basic gives by itself, so a job the agent has not placed (just submitted, or released, which ranks it
#: afresh) waits behind every placed one, whatever its owner does.
SPAN = 21_000
BAND = {Level.HIGH: _HIGHEST, Level.NORMAL: _HIGHEST - SPAN, Level.LOW: _HIGHEST - 2 * SPAN}


def level_of(qos: str) -> Level:
    """A job's level from its QOS; an admin's own QOS counts as normal."""
    return Level(qos) if qos in set(Level) else Level.NORMAL


def priority_for(level: Level, job_id: str, base: int) -> int:
    """The priority the agent gives a waiting job: by level, then by submission (job id, as `squeue --json` gives it),
    counted from `base`, a job id at or below every live one. A job more than SPAN ids past it shares the level's
    lowest priority (slurm then goes by submission time)."""
    return BAND[level] - min(max(int(job_id) - base, 0), SPAN - 1)


def rebase(base: int | None, job_ids: list[str]) -> int | None:
    """`base` while every live job id fits in a level's span from it, else the lowest live id. Moving it re-places
    every waiting job, so it moves only when it must."""
    ids = [int(job_id) for job_id in job_ids]
    if not ids:
        return base
    if base is None or min(ids) < base or max(ids) - base >= SPAN:
        return min(ids)
    return base
