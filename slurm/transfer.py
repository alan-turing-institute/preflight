"""Facts about rsync and slurm that more than one tool reads: one definition each.

The part-time copy-back (agent.py) and the remote fetch-back (fetch_back.py) both pull with rsync and judge its exit
value; the fetch-back also reads a remote job's end from `sacct`. These are facts of those programs, not settings.
"""

from __future__ import annotations

from enum import StrEnum

#: rsync gives up a stalled transfer after this many seconds (`--timeout`; exit 30).
COPY_IO_TIMEOUT_S = 20
#: rsync(1) EXIT VALUES: 23 partial transfer due to error, 24 source files vanished, 30 timeout in data send/receive,
#: 255 the remote shell (ssh) failed.
RSYNC_PARTIAL_ERROR = 23
RSYNC_VANISHED = 24
#: rsync's "some files could not be transferred" and "some files vanished": the rest was copied.
RSYNC_PARTIAL = frozenset({RSYNC_PARTIAL_ERROR, RSYNC_VANISHED})
RSYNC_IO_TIMEOUT = 30
RSYNC_SSH_FAILED = 255
#: rsync's stderr when the remote folder does not exist; the exit value is 23, the same as a partial transfer
#: (measured with rsync 3.2.7: "change_dir ... failed: No such file or directory").
MISSING_SOURCE = "change_dir"


class EndState(StrEnum):
    """The states in which a slurm job has finished (`sacct --helpstate`; REVOKED and SPECIAL_EXIT as the GPU-hour
    ledger treats them). Any other state, including one not listed here, means not finished yet."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    PREEMPTED = "PREEMPTED"
    REVOKED = "REVOKED"
    SPECIAL_EXIT = "SPECIAL_EXIT"


def end_state(sacct_state: str) -> EndState | None:
    """The end state in a `sacct` State field ("CANCELLED by 1002" is CANCELLED); None while the job has not finished."""
    words = sacct_state.split()
    try:
        return EndState(words[0]) if words else None
    except ValueError:
        return None
