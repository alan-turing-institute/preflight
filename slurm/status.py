#!/usr/bin/env python3
"""What every GPU job is, how it pauses, and who holds the card (design 3.3, 3.8)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402 - path set just above
import audit  # noqa: E402
import fetch_back
import gpu_jobs  # noqa: E402
import jobs_view  # noqa: E402
import remote  # noqa: E402
from wait import all_sidecars

#: How a job in each partition pauses, for the table.
PAUSES = {
    gpu_jobs.Partition.NORMAL: "pre-empted, resumes from its checkpoint",
    gpu_jobs.Partition.STEADY: "never pre-empted (a requeue redoes it)",
    gpu_jobs.Partition.CPU_FILL: "fills idle cores; requeued for GPU or urgent work, resumes from its checkpoint",
    gpu_jobs.Partition.URGENT: "urgent",
    gpu_jobs.Partition.SERVICE: "a service's CPU stand-in; requeued for urgent work",
    gpu_jobs.Partition.PART_TIME: "a part-time machine; requeued when its owner needs it, resumes from its checkpoint",
}
#: A job waiting in both (`gpu_jobs.ANYWHERE_PARTITIONS`; once running it is shown in the one it got).
ANYWHERE_PAUSE = "fills idle cores here or on a part-time machine; requeued for GPU or urgent work or when that machine's owner needs it, resumes from its checkpoint"


def _pauses(job: gpu_jobs.QueuedJob) -> str:
    """`PAUSES` by partition; a `cpu` job by its class (urgent.py requeues one that resumes, never one that restarts)."""
    if job.partition == gpu_jobs.Partition.CPU:
        return PAUSES[gpu_jobs.Partition.STEADY if job.restarts else gpu_jobs.Partition.NORMAL]
    if set(job.partition.split(",")) == {gpu_jobs.Partition.CPU_FILL, gpu_jobs.Partition.PART_TIME}:  # waiting: squeue lists both
        return ANYWHERE_PAUSE
    return PAUSES.get(job.partition, job.partition)


def _memory(job: gpu_jobs.QueuedJob, record: dict) -> str:
    """slurm's request as it stands (an `scontrol update MinMemoryNode=` shows), else what submit.py recorded."""
    mb = jobs_view.requested_mem_mb(job, record)
    return "default mem" if mb is None else f"{jobs_view.memory_text(mb)} mem"


def lines(queue: list[gpu_jobs.QueuedJob] | None) -> list[str]:
    out = [f"[status] node: {gpu_jobs.node_state()}"]
    if gpu_jobs.LEASE_DIR is not None:  # a site without a [lease] table has no keeper to report
        keeper = gpu_jobs.read_json(gpu_jobs.keeper_state_path()) or {}
        out.append(
            f"[status] keeper: {keeper.get('phase', 'no state')}; blockers {keeper.get('blockers', [])};"
            f" updated {keeper.get('updated_at', '?')} {keeper.get('note', '')}".rstrip()
        )
    if queue is None:
        out.append("[status] squeue did not answer: the slurm controller is down")
        return out
    if not queue:
        out.append("[status] no jobs in slurm")
    records = all_sidecars({job.job_id: job.user for job in queue})
    by_id = {job.job_id: job for job in queue}
    events = gpu_jobs.events()
    rooms = jobs_view.rooms_now(by_id, gpu_jobs.load_site())
    for job in queue:
        record = records.get(job.job_id, {})
        why = jobs_view.waiting_reason(job, by_id, records, events, rooms)
        out.append(gpu_jobs.printable(  # names and descriptions are what their submitters typed
            f"[status] {job.job_id} {job.state:<10} {job.user or '?':<10} {job.level:<6} {f'{job.shards} shards' if job.shards else f'{job.gpus} GPU'} {job.name}"
            f"{f' on {job.node}' if job.node else ''}"
            f" | {_memory(job, record)}, {_pauses(job)}"
            f" | {record.get('what', '(no sidecar)')}{f' | {why}' if why else ''}"
        ))
    out += [f"[idle] {line}" for line in agent.idle_summary(agent.read_idle())]
    out += gpu_lines(queue)
    out += audit.leftover_containers(gpu_jobs.known_containers(), queue, gpu_jobs.this_host(), time.time())
    return out


def gpu_lines(queue: list[gpu_jobs.QueuedJob]) -> list[str]:
    """The card verdict and the audit of GPU work outside slurm, from one `nvidia-smi` reading. A machine without
    `nvidia-smi` has no card: no verdict, and the audit finds no GPU work."""
    has_card = shutil.which("nvidia-smi") is not None
    usage = gpu_jobs.gpu_usage() if has_card else {}
    strays = audit.unregistered(usage) if usage is not None else []
    out = []
    if has_card:
        leases = gpu_jobs.lease_pids(gpu_jobs.LEASE_DIR, holding=True) if gpu_jobs.LEASE_DIR else {}
        out += card_lines(queue, gpu_jobs.this_host(), usage, gpu_jobs.gpu_numbers("power.draw"), leases, strays)
    return out + (audit.report(strays) if usage is not None else [])  # unknown is said once, by the card line


def card_lines(queue: list[gpu_jobs.QueuedJob], here: str, usage: dict[int, float] | None, watts: list[float] | None,
               leases: dict[int, str], strays: list[audit.Stray]) -> list[str]:
    """This machine's card, by `gpu_jobs.card_state` (the page's rule too), and who holds it: the slurm jobs here that
    hold a GPU or shards, the services holding a GPU lease (not waiting for one; the keeper's own lease stands for
    slurm's jobs), the compute processes outside both. The power draw and the compute processes are the evidence;
    nvidia-smi's GPU-Util never is (it reads about 96% on an idle GB10)."""
    jobs = [job for job in queue if job.uses_card and job.node == here and job.state in gpu_jobs.ACTIVE_STATES]
    services = sorted(set(leases.values()) - {gpu_jobs.LEASE_PROJECT})
    reading = None if usage is None else gpu_jobs.CardReading(len(usage), len(services), sum(watts) if watts else None)
    state = gpu_jobs.card_state(reading, len(jobs))
    if usage is None:
        evidence = "nvidia-smi did not answer (inside a slurm job without a GPU the card is hidden)"
    else:
        power = f"{sum(watts):.0f} W" if watts else "power unknown"
        card = gpu_jobs.size_text(gpu_jobs.card_bytes(usage, usage))
        evidence = f"{power}, {len(usage)} compute process(es) holding {card} of card memory; GPU-Util is not a test of this"
    if state == gpu_jobs.CardState.UNKNOWN:
        return [f"[card] {state.upper()}: {evidence}; do not treat the card as free"]
    if state == gpu_jobs.CardState.HELD:
        holders = [f"slurm job {job.job_id} ({job.name})" for job in jobs] + [f"the GPU lease of {name}" for name in services]
        holders += [f"pid {stray.pid}, outside slurm and the lease protocol" for stray in strays] or (
            [] if holders else [f"{len(usage or {})} compute process(es)"])
        return [gpu_jobs.printable(f"[card] {state.upper()} by {'; '.join(holders)} ({evidence})")]
    waiting = [job.job_id for job in queue if job.uses_card and job.state == gpu_jobs.State.PENDING]
    after = f"; GPU jobs waiting for it: {', '.join(waiting)}" if waiting else ""
    return [f"[card] {state.upper()}: no slurm job, GPU lease or compute process holds it ({evidence}){after}"]


#: `--reserve` (D44): the share of the one-minute samples the reserve must cover, and the least history to judge from.
RESERVE_PERCENTILE = 99
RESERVE_MIN_DAYS = 7


def reserve_lines(site: gpu_jobs.site_config.Site, folder: Path, here: str, mem_total: int) -> list[str]:
    """What each node used outside its jobs (the agent's one-minute record on a unified-memory node, D44) and, for the
    node this runs on, the `jobs_memory_gb` that keeps it free: its memory less the 99th percentile of that use, less
    the node's close level, so the health check and the memory guard are not reached in normal use."""
    samples: dict[str, list[tuple[int, int]]] = {}
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        for text in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                line = json.loads(text)
            except ValueError:
                continue
            if isinstance(line, dict) and isinstance(line.get("node"), str) and all(
                    type(line.get(key)) is int for key in ("t", "outside")):
                samples.setdefault(line["node"], []).append((line["t"], line["outside"]))
    if not samples:
        return ["[reserve] no record of memory used outside jobs yet (the agent writes one on a unified-memory node)"]
    nodes, size, out = {node.name: node for node in site.nodes}, gpu_jobs.size_text, []
    for name, found in sorted(samples.items()):
        days = len({time.strftime("%Y-%m-%d", time.localtime(t)) for t, _ in found})
        if days < RESERVE_MIN_DAYS:
            out.append(f"[reserve] {name}: not enough data ({days} days; at least {RESERVE_MIN_DAYS})")
            continue
        outside = statistics.quantiles([o for _, o in found], n=100, method="inclusive")[RESERVE_PERCENTILE - 1]
        out.append(f"[reserve] {name}: {len(found)} samples over {days} days; used outside jobs: {size(outside)} at the "
                   f"{RESERVE_PERCENTILE}th percentile")
        if name != here or name not in nodes:
            out.append(f"[reserve] {name}: run this on {name} for its jobs_memory_gb (it needs that machine's memory)")
            continue
        level = nodes[name].close_below_mb * gpu_jobs.MB
        out.append(f"[reserve] {name}: jobs_memory_gb = {math.floor((mem_total - outside - level) / gpu_jobs.GB)} "
                   f"(memory {size(mem_total)} less {size(outside)} used outside jobs less the {size(level)} close level)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=None, metavar="NAME", help="show another site's queue ([remote.NAME] in the site file)")
    parser.add_argument("--reserve", action="store_true",
                        help="the memory each node used outside its jobs, and the jobs_memory_gb that keeps it (D44)")
    arguments = parser.parse_args()
    if arguments.reserve:
        print("\n".join(reserve_lines(gpu_jobs.load_site(), gpu_jobs.STATE_DIR / "agent" / "outside",
                                      gpu_jobs.this_host(), gpu_jobs.meminfo()["MemTotal"])))
        return 0
    if arguments.site:
        try:
            print(remote.status(gpu_jobs.load_site(), arguments.site))
            fetched = fetch_back.summary_line(arguments.site)  # this user's records for that site (D52)
            if fetched:
                print(fetched)
        except remote.RemoteRefused as error:
            print(f"[status] REFUSED: {error}", file=sys.stderr)
            return 1
        return 0
    print("\n".join(lines(gpu_jobs.read_queue())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
