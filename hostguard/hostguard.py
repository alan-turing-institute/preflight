"""Stop a GPU process from eating the memory its own host is living in.

WHY THIS EXISTS
---------------
On a unified-memory machine the GPU pool IS system RAM, so there is no device
limit to hit first: a process that keeps allocating keeps succeeding until the
WHOLE MACHINE is out. Two measured outcomes on spark-ad27 (NVIDIA GB10):

* 2026-09-01: the kernel OOM killer took an eval sharing a service with the
  terminal sessions (`shpool.service: Failed with result 'oom-kill'`), and every
  session on the box died.
* 2026-10-06 13:51: a smoke test asked for more than the host could spare. The
  driver logged `NV_ERR_NO_MEMORY`, available memory fell from ~86 GiB to ~20 GiB
  in 30 s, kswapd livelocked, no killer fired, and the box had to be power-cycled.
  This guard WAS loaded -- but it capped torch at total - 12 GiB = 109.7 GiB while
  only ~86 GiB was available and a 21 GiB server already held part of the pool.

WHAT IT DOES (when torch is imported, on a unified pool only)
------------------------------------------------------------
1. Caps torch's allocator at min(total, MemAvailable) - reserve, measured at
   import, so torch raises its own out-of-memory error instead of starving the
   host. On the GB10 `mem_get_info` reports only MemFree (page cache excluded),
   so MemAvailable -- free plus reclaimable -- is the honest "can spare" figure.
2. Raises the process's oom_score_adj to OOM_SCORE_ADJ (never lowers it), so if
   memory still runs out the kernel or earlyoom kills the GPU job, not a terminal.

HOW IT STAYS GENERAL
--------------------
Nothing here names a machine. It tests the PROPERTY that creates the hazard --
device total ~= MemTotal -- so a second unified box is covered automatically and
no discrete-GPU machine ever is. CUDA and Intel XPU both expose
`set_per_process_memory_fraction` (verified on torch 2.10.0). Under SLURM it does
nothing: the scheduler bounds the job and a cluster GPU is not host memory.

INSTALL: put this directory on PYTHONPATH (its sitecustomize.py imports this), or
add a `.pth` line to a virtualenv: see README.md.

Env:
    AUTOCAST_HOST_MEMORY_RESERVE_GIB   GiB kept for the host (default 12; 0 = off)
    AUTOCAST_HOSTGUARD_VERBOSE=1       also report the cases where it declines
"""

from __future__ import annotations

import os
import sys

GIB = 2**30

#: GiB kept for the host. ABSOLUTE, NOT A FRACTION: the host's need does not scale
#: with pool size (measured 6.6 GiB at rest on a 121.7 GiB box). A first attempt
#: used a 0.4 fraction and came within 2 GiB of killing a two-hour job.
DEFAULT_RESERVE_GIB = 12.0

#: The smallest cap ever set. When less than the reserve is available the job
#: cannot run safely; a 1 GiB cap makes its first real allocation fail loudly,
#: with the message below explaining why, instead of thrashing the box.
FLOOR_BYTES = 1 * GIB

#: oom_score_adj given to a guarded GPU process (range -1000..1000). High enough
#: to be chosen before ordinary processes, below 1000 so a launcher can still
#: mark one job as the very first to go.
OOM_SCORE_ADJ = 800

#: How close device total must be to MemTotal to call the pool unified. On the
#: GB10 the two are byte-identical (ratio 1.000000).
UNIFIED_TOLERANCE = 0.02

#: Any sitecustomize.py carrying this string is a hostguard shim, never chained.
SHIM_MARKER = "HOSTGUARD-SHIM"


def _say(msg: str, force: bool = False) -> None:
    if force or os.environ.get("AUTOCAST_HOSTGUARD_VERBOSE"):
        print(f"[hostguard] {msg}", file=sys.stderr, flush=True)


def read_meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    """MemTotal / MemAvailable in bytes; empty if unreadable."""
    out: dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    out[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    return out


def cap_bytes(device_total: int, mem_available: int, reserve: int) -> int:
    """The most torch may hold: what the host can spare now, less the reserve."""
    return max(min(device_total, mem_available) - reserve, FLOOR_BYTES)


def mark_preferred_victim(path: str = "/proc/self/oom_score_adj") -> None:
    """Raise oom_score_adj to OOM_SCORE_ADJ; never lower one already higher."""
    try:
        with open(path) as fh:
            current = int(fh.read().strip() or 0)
        if current < OOM_SCORE_ADJ:
            with open(path, "w") as fh:
                fh.write(f"{OOM_SCORE_ADJ}\n")
    except (OSError, ValueError) as exc:
        _say(f"could not set oom_score_adj ({exc})")


def apply(torch, meminfo_path: str = "/proc/meminfo",
          oom_path: str = "/proc/self/oom_score_adj") -> None:
    raw = os.environ.get("AUTOCAST_HOST_MEMORY_RESERVE_GIB")
    reserve_gib = DEFAULT_RESERVE_GIB if raw is None else float(raw)
    if reserve_gib <= 0:
        _say("disabled by AUTOCAST_HOST_MEMORY_RESERVE_GIB=0")
        return
    if os.environ.get("SLURM_JOB_ID"):
        _say("skipped: under SLURM")
        return
    mem = read_meminfo(meminfo_path)
    if "MemTotal" not in mem or "MemAvailable" not in mem:
        _say("skipped: /proc/meminfo unreadable")
        return
    reserve = int(reserve_gib * GIB)

    for name in ("cuda", "xpu"):
        backend = getattr(torch, name, None)
        if backend is None or not backend.is_available():
            continue
        if not hasattr(backend, "set_per_process_memory_fraction"):
            _say(f"skipped {name}: no set_per_process_memory_fraction in this torch")
            continue
        try:
            total = backend.get_device_properties(0).total_memory
        except Exception as exc:  # noqa: BLE001 -- never break an import
            _say(f"skipped {name}: cannot read device total ({exc})")
            continue
        if abs(total - mem["MemTotal"]) / total >= UNIFIED_TOLERANCE:
            _say(f"skipped {name}: discrete ({total / GIB:.1f} GiB), host memory is separate")
            continue
        cap = cap_bytes(total, mem["MemAvailable"], reserve)
        backend.set_per_process_memory_fraction(cap / total)
        mark_preferred_victim(oom_path)
        floor_note = " -- TOO LITTLE FREE: this job will fail its first large allocation" \
            if cap == FLOOR_BYTES else ""
        _say(
            f"{name}: unified pool {total / GIB:.1f} GiB, {mem['MemAvailable'] / GIB:.1f} GiB "
            f"available now, reserving {reserve_gib:.0f} GiB for the host -> cap "
            f"{cap / GIB:.1f} GiB; oom_score_adj >= {OOM_SCORE_ADJ}{floor_note}",
            force=True,
        )


def install() -> None:
    """Apply the cap the moment torch is imported, not at interpreter start.

    torch is usually imported AFTER this runs, so applying eagerly would do
    nothing. A meta-path hook firing on torch's own import is the only placement
    that is both automatic and correctly ordered. Idempotent.
    """
    import importlib.abc
    import importlib.machinery

    if any(type(f).__name__ == "_TorchWatcher" for f in sys.meta_path):
        return
    if "torch" in sys.modules:
        apply(sys.modules["torch"])
        return

    class _TorchWatcher(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "torch":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                inner = spec.loader.exec_module

                def exec_module(module, _inner=inner):
                    _inner(module)
                    try:
                        apply(module)
                    except Exception as exc:  # noqa: BLE001 -- never break an import
                        _say(f"FAILED: {type(exc).__name__}: {exc}", force=True)

                spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _TorchWatcher())


def chain_shadowed_sitecustomize() -> None:
    """Run the first real sitecustomize on sys.path (Ubuntu's installs apport's).

    A hostguard sitecustomize shadows it; skip every file carrying SHIM_MARKER.
    """
    import importlib.util

    for entry in sys.path:
        cand = os.path.join(entry or ".", "sitecustomize.py")
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand) as fh:
                if SHIM_MARKER in fh.read():
                    continue
            spec = importlib.util.spec_from_file_location("_shadowed_sitecustomize", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001, S110 -- a broken foreign hook must not break us
            pass
        return


try:
    install()
except Exception as exc:  # noqa: BLE001
    _say(f"hook install FAILED: {type(exc).__name__}: {exc}", force=True)
