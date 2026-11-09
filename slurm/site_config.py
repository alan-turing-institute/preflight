"""The site file: every difference between one machine and another, read once and typed.

The TOML layout maps one-to-one onto the dataclasses below (`[[nodes]]`, `[[nodes.gpus]]`, `[lease]`,
`[lease.services]`, `[remote.NAME]`, `[slurm]`, `[metrics]`, `[portal]`, `[portal.auth_settings]`). Unknown keys are an error at every
table level, so a typo cannot silently fall back to a default. `site.example.toml` documents each field.
"""

from __future__ import annotations

import grp
import ipaddress
import math
import os
import pwd
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

SITE_FILE = Path(os.environ.get("GPUJOBS_SITE", "/etc/gpujobs/site.toml"))
MB_PER_GB = 1000  # site-file sizes are decimal gigabytes
SLURM_MB_PER_GB = 1e9 / 2**20  # slurm's "MB" (`real_memory_mb`, MemSpecLimit) is 2^20 bytes: a decimal GB is 953.67 of them
#: Health-check thresholds in GB when a node names none, by `part_time`: (close below, push back below).
_DEFAULT_THRESHOLDS_GB = {False: (16, 10), True: (6, 2)}
_CPU_SPEC_LIST = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")
#: Core kinds on a node with `fast_cpus`: slurm is shown one socket per kind (D46), and CoresPerSocket is one number.
CORE_KINDS = 2


def cpu_ids(spec: str) -> list[int]:
    """CPU numbers from a slurm/taskset list such as "0-4,10-14", sorted (the inverse of `cores.cpu_list`)."""
    ids: set[int] = set()
    for part in spec.split(","):
        low, _, high = part.partition("-")
        ids.update(range(int(low), int(high or low) + 1))
    return sorted(ids)


class MetricsMode(StrEnum):
    LOCAL = "local"
    EXTERNAL = "external"
    OFF = "off"


class PortalMode(StrEnum):
    INSTALL = "install"
    EXISTING = "existing"
    OFF = "off"


class AuthMode(StrEnum):
    OIDC = "oidc"
    DEX_LDAP = "dex-ldap"
    DEX_LOCAL = "dex-local"
    PROXY = "proxy"  # no sign-in: a trusted reverse proxy vouches for every request (see portal.py)
    SSH = "ssh"  # no sign-in page: each person's own ssh forward to the relay names them (D19, see portal.py)


class CoresFromHistory(StrEnum):
    """`cpus_from_history` (D35): what `submit.py` does with the cores a CPU-only job used on its last completed runs."""

    OFF = "off"  # nothing: no accounting query, no line
    RECOMMEND = "recommend"  # print the cores it would ask for; ask as before
    ENFORCE = "enforce"  # ask for them when --cpus is not given


SHARDS_FROM_CORES = "cores"  # `shards = "cores"`: one shard per core the node holds for the card
MAX_SHARDS = 1024  # slurm gres.conf(5): a shard Count is limited in size (currently 1024)


class SiteError(Exception):
    """The site file is missing or wrong; the message says what to do."""


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    device: str  # e.g. "/dev/nvidia0"
    type: str
    shards: int = 0  # slices of the card slurm may give to different jobs at once (gres/shard); 0: the card is one token


@dataclass(frozen=True)
class Node:
    name: str
    cpus: int
    real_memory_mb: int
    gpus: tuple[Gpu, ...]
    cores_per_gpu: int | None = None  # held for GPU jobs; CPU-only jobs never use them (slurm's RestrictedCoresPerGPU)
    part_time: bool = False  # someone's own machine, lent to the cluster when idle; the health check opens and closes it
    address: str | None = None  # slurm NodeAddr: where the controller reaches a part-time node (required with part_time)
    keep_cpus: str | None = None  # CpuSpecList: OS CPUs slurm never hands to jobs on a part-time node, e.g. "0-11"
    jobs_memory_gb: float | None = None  # memory jobs may use; MemSpecLimit is the real memory less this (full node: measured,
    # `status.py --reserve`)
    mains_only: bool | None = None  # health check: close on battery; None = decided at run time (true with a battery)
    close_below_gb: float | None = None  # health check: close the node below this much free memory
    push_back_below_gb: float | None = None  # health check: requeue a resume job below this much free memory
    cores_per_socket: int | None = None  # CoresPerSocket as `slurmd -C` prints it, when that is not `cpus` (hybrid Intel CPUs); None: cpus
    default_jobs: bool = True  # part-time: false keeps the node opt-in (`submit.py --anywhere`); true lets portable jobs go there unasked
    fast_cpus: str | None = None  # GPU node with two equal core kinds: OS CPUs of the fast kind, e.g. "5-9,15-19" (D46)

    @property
    def gpu_cores(self) -> int:
        """Cores held for GPU jobs on this node: 0 when `cores_per_gpu` is unset."""
        return (self.cores_per_gpu or 0) * len(self.gpus)

    @property
    def shards(self) -> int:
        """Shards of all this node's cards together (slurm's `gres/shard` count for the node); 0: no card is shared."""
        return sum(gpu.shards for gpu in self.gpus)

    @property
    def close_below_mb(self) -> int:
        gb = self.close_below_gb if self.close_below_gb is not None else _DEFAULT_THRESHOLDS_GB[self.part_time][0]
        return round(gb * MB_PER_GB)

    @property
    def push_back_below_mb(self) -> int:
        gb = self.push_back_below_gb if self.push_back_below_gb is not None else _DEFAULT_THRESHOLDS_GB[self.part_time][1]
        return round(gb * MB_PER_GB)

    @property
    def mem_spec_limit_mb(self) -> int | None:
        """MemSpecLimit: memory slurm keeps from jobs; None unless `jobs_memory_gb` is set."""
        if self.jobs_memory_gb is None:
            return None
        return self.real_memory_mb - round(self.jobs_memory_gb * SLURM_MB_PER_GB)

    @property
    def cores_lent(self) -> int:
        """Cores slurm hands to jobs: its abstract cores (`cores_per_socket`, else `cpus`) less the ids in `keep_cpus`."""
        return (self.cores_per_socket or self.cpus) - (len(cpu_ids(self.keep_cpus)) if self.keep_cpus else 0)

    @property
    def sockets(self) -> tuple[int, int]:
        """(sockets, cores per socket) as slurmd sees the node: one socket per core kind with `fast_cpus`, whose layout
        file shows the kinds as packages (D46), else one socket of `cores_per_socket` or `cpus`."""
        if self.fast_cpus:
            return CORE_KINDS, self.cpus // CORE_KINDS
        return 1, self.cores_per_socket or self.cpus

    @property
    def fast_cpu_ids(self) -> list[int]:
        """The OS CPUs of the fast kind (`fast_cpus`); empty when unset."""
        return cpu_ids(self.fast_cpus) if self.fast_cpus else []

    @property
    def fast_cores(self) -> int:
        """Cores of the fast kind: the most a GPU job may ask for and still be held to them (D46); 0 when unset."""
        return len(self.fast_cpu_ids)

    @property
    def fast_socket(self) -> int:
        """The socket slurm numbers the fast kind under the layout file: hwloc orders packages by their lowest CPU."""
        slow = [cpu for cpu in range(self.cpus) if cpu not in self.fast_cpu_ids]
        return 0 if self.fast_cpu_ids and min(self.fast_cpu_ids) < min(slow) else 1

    @property
    def fast_socket_cores(self) -> str:
        """Slurm's own (abstract) numbers of the fast socket's cores, as gres.conf's `Cores=` wants them: "10-19"."""
        first = self.fast_socket * self.fast_cores
        return f"{first}-{first + self.fast_cores - 1}"

    @property
    def jobs_mb(self) -> int:
        """Memory slurm hands to jobs, in slurm MB: the real memory less `mem_spec_limit_mb`."""
        return self.real_memory_mb - (self.mem_spec_limit_mb or 0)


@dataclass(frozen=True)
class Maintenance:
    """`[maintenance]`: a restart slot run outside these tools (the site's own) writes `status_file`, a JSON status of
    the updates waiting for a restart, which the jobs page shows its admins; its buttons leave requests in
    `request_dir`, which that slot reads and decides on."""

    status_file: str
    request_dir: str


@dataclass(frozen=True)
class HealthCheck:
    """`[health_check]`: slurmd runs `node_check.py` on every node each `interval_s` seconds (slurm's HealthCheckInterval)."""

    interval_s: int


@dataclass(frozen=True)
class Lease:
    lease_dir: Path
    keeper_user: str
    python: Path
    services: dict[str, str]  # service name -> how it yields


@dataclass(frozen=True)
class Service:
    """The `service` partition (D20): background services' CPU stand-ins start at once by suspending `cpu` jobs."""

    cores: int  # per node, the partition's MaxCPUsPerNode
    group: str  # Unix group allowed to submit (slurm AllowGroups, with the admin group)


@dataclass(frozen=True)
class Metrics:
    mode: MetricsMode
    prometheus_url: str
    grafana_url: str
    grafana_port: int
    retention_days: int
    allow_from: tuple[str, ...] = ()  # networks that may reach Grafana and the exporters; empty = any


@dataclass(frozen=True)
class Portal:
    mode: PortalMode
    auth: AuthMode
    server_name: str
    tls_cert: Path | None
    tls_key: Path | None
    auth_settings: dict[str, str]
    session_days: int = 0  # how long one sign-in lasts, across browser restarts and reboots; 0 = OOD's 8 hours
    allow_from: tuple[str, ...] = ()  # networks that may reach the portal; empty = any


@dataclass(frozen=True)
class Remote:
    """Another site this one can send jobs to (`submit.py --site NAME`): its own slurm, reached over ssh."""

    host: str  # ssh destination (an ~/.ssh/config alias works)
    root: str  # folder there, relative to the login home or absolute; one subfolder per project holds its code and data
    submit_command: str = "gpujobs-submit"  # that site's own submit.py
    status_command: str = "gpujobs-status"
    data_root: str = ""  # absolute folder there for --data folders from outside the repo; empty: those are refused


@dataclass(frozen=True)
class Site:
    name: str
    cluster_name: str  # slurm's ClusterName; Open OnDemand names its cluster file after it
    account: str  # the slurm account every user is added to (`install.py user add`)
    state_dir: Path
    install_dir: Path
    admin_group: str
    nodes: tuple[Node, ...]
    unified_memory: bool
    lease: Lease | None
    service: Service | None
    hostguard: bool
    dgx_gate: bool
    metrics: Metrics
    portal: Portal
    notify_command: tuple[str, ...]  # argv; empty means no notification
    slurm_build: tuple[str, str] | None = None  # ([slurm] version, sha256): build that release, not the distro's slurm
    slurm_steps: tuple[tuple[str, str], ...] = ()  # ([slurm] steps) the releases `upgrade-slurm` passes through, in order
    health_check: HealthCheck | None = None  # None: no health-check lines and no node_check.json
    maintenance: Maintenance | None = None  # None: no maintenance banner on the jobs page
    slurm_release: SlurmRelease | None = None  # None: no release watch
    constrain_devices: bool = True  # cgroup.conf ConstrainDevices: a job sees only the GPUs it holds
    shared_state: bool = False  # the part-time nodes see this state_dir (a shared filesystem): nothing is copied back
    node_python: str = "/usr/bin/python3"  # the interpreter a part-time job's wrapper runs with, there
    remotes: dict[str, Remote] = field(default_factory=dict)  # `[remote.NAME]` tables
    requeue_delay_s: int = 60  # slurm.conf AuthInfo=cred_expire: how long a requeued job waits before it may start again
    sharing_memory_fraction: float | None = None  # a site's own tightening of the share of a node's jobs memory that co-running shard jobs may fill
    cpus_from_history: CoresFromHistory = CoresFromHistory.RECOMMEND  # cores for a named CPU-only job without --cpus (D35)

    @property
    def cpu_job_cap(self) -> int | None:
        return _cpu_job_cap(self.nodes)

    @property
    def sharing(self) -> bool:
        """Whether any card is shared (declares shards)."""
        return any(node.shards for node in self.nodes)

    @property
    def part_time_nodes(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.part_time)

    @property
    def full_nodes(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if not n.part_time)


def gpu_node(site: Site) -> Node | None:
    """The node holding the GPUs (one-GPU-node sites, D44); None without GPUs."""
    return next((n for n in site.nodes if n.gpus), None)


def _cpu_job_cap(nodes: tuple[Node, ...]) -> int | None:
    """Cores a CPU-only job may use per node (its partitions' MaxCPUsPerNode): every node's cores less those held
    for its GPUs, at the tightest node; None when no node holds cores. slurm 23.11 has no per-node form, so a
    CPU-only node in a mixed site is capped too."""
    held = [n.cpus - n.gpu_cores for n in nodes if n.gpu_cores]
    return min(held) if held else None


_HINT = "run `slurm/install/install.py detect` to write one for this machine"


def _table(raw: Any, where: str, required: set[str], optional: dict[str, Any]) -> dict[str, Any]:
    """Check one table's keys: refuse unknown ones, demand the required ones, fill the defaults."""
    if not isinstance(raw, dict):
        raise SiteError(f"site file: [{where}] must be a table; {_HINT}")
    unknown = sorted(set(raw) - required - set(optional))
    if unknown:
        raise SiteError(f"site file: unknown key(s) in [{where}]: {', '.join(unknown)}")
    missing = sorted(required - set(raw))
    if missing:
        raise SiteError(f"site file: [{where}] is missing {', '.join(missing)}; {_HINT}")
    return {**optional, **raw}


def _enum[E: StrEnum](cls: type[E], value: str, where: str) -> E:
    try:
        return cls(value)
    except ValueError:
        allowed = ", ".join(m.value for m in cls)
        raise SiteError(f"site file: {where} = {value!r}; expected one of {allowed}") from None


def _networks(value: Any, where: str) -> tuple[str, ...]:
    """IP networks (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`, or one address), as systemd's IPAddressAllow= takes."""
    if not isinstance(value, list):
        raise SiteError(f"site file: {where} must be a list of IP networks")
    for entry in value:
        try:
            ipaddress.ip_network(entry, strict=False)
        except (TypeError, ValueError):
            raise SiteError(f"site file: {where} entry {entry!r} is not an IP network such as 100.64.0.0/10") from None
    return tuple(value)


def _opt_path(value: str) -> Path | None:
    return Path(value) if value else None


def _parse_gpu(raw: Any, where: str, cores_per_gpu: Any) -> Gpu:
    """One `[[nodes.gpus]]` table; `shards` is a whole number of shards, or "cores" for one per core held for the card."""
    t = _table(raw, where, {"index", "uuid", "device", "type"}, {"shards": 0})
    shards = t["shards"]
    if shards == SHARDS_FROM_CORES:
        if type(cores_per_gpu) is not int:
            raise SiteError(f"site file: [{where}] shards = \"{SHARDS_FROM_CORES}\" needs the node's cores_per_gpu: one shard per core held for the card")
        shards = cores_per_gpu
    if type(shards) is not int or not 0 <= shards <= MAX_SHARDS:
        raise SiteError(f"site file: [{where}] shards = {t['shards']!r}; expected \"{SHARDS_FROM_CORES}\" or a whole number from 0 (not shared) "
                        f"to {MAX_SHARDS}")
    return Gpu(**{**t, "shards": shards})


def _parse_node(raw: Any, i: int) -> Node:
    where = f"nodes[{i}]"
    t = _table(raw, where, {"name", "cpus", "real_memory_mb"}, {
        "gpus": [], "cores_per_gpu": None, "part_time": False, "address": None, "keep_cpus": None,
        "jobs_memory_gb": None, "mains_only": None, "close_below_gb": None, "push_back_below_gb": None,
        "cores_per_socket": None, "default_jobs": True, "fast_cpus": None,
    })
    gpus = tuple(_parse_gpu(g, f"{where}.gpus[{j}]", t["cores_per_gpu"]) for j, g in enumerate(t["gpus"]))
    node = Node(t["name"], t["cpus"], t["real_memory_mb"], gpus, t["cores_per_gpu"], t["part_time"], t["address"],
                t["keep_cpus"], t["jobs_memory_gb"], t["mains_only"], t["close_below_gb"], t["push_back_below_gb"],
                t["cores_per_socket"], t["default_jobs"], t["fast_cpus"])
    _check_part_time(node, where)
    _check_fast_cpus(node, where)
    if node.cores_per_socket is not None and not (type(node.cores_per_socket) is int and 1 <= node.cores_per_socket <= node.cpus):
        raise SiteError(f"site file: [{where}] cores_per_socket = {node.cores_per_socket!r}; expected a whole number from 1 to cpus ({node.cpus})")
    if node.cores_per_gpu is not None and not (
        type(node.cores_per_gpu) is int and gpus and node.cores_per_gpu >= 1 and node.gpu_cores < node.cpus
    ):
        raise SiteError(f"site file: [{where}] cores_per_gpu must be a whole number of at least 1, needs GPUs, and must leave CPU-only jobs a core "
                        f"({node.cores_per_gpu} x {len(gpus)} GPU(s) of {node.cpus} cores)")
    return node


def _check_fast_cpus(node: Node, where: str) -> None:
    if node.fast_cpus is None:
        return
    if not (isinstance(node.fast_cpus, str) and _CPU_SPEC_LIST.fullmatch(node.fast_cpus)):
        raise SiteError(f"site file: [{where}] fast_cpus = {node.fast_cpus!r}; expected CPU numbers and ranges such as \"5-9,15-19\"")
    if not node.gpus or node.part_time or node.keep_cpus is not None or node.cores_per_socket is not None:
        raise SiteError(f"site file: [{where}] fast_cpus binds the GPU to the fast cores: it needs a full node with GPUs, "
                        "and no keep_cpus or cores_per_socket")
    ids = node.fast_cpu_ids
    if ids[-1] >= node.cpus or len(ids) * CORE_KINDS != node.cpus:
        raise SiteError(f"site file: [{where}] fast_cpus = {node.fast_cpus!r}; expected exactly half of the node's {node.cpus} CPUs, "
                        "each below cpus (slurm gives every socket the same number of cores); `slurm/cores.py` lists the kinds, "
                        "or leave fast_cpus out")


def _check_part_time(node: Node, where: str) -> None:
    if type(node.part_time) is not bool or node.mains_only not in (None, True, False) or type(node.default_jobs) is not bool:
        raise SiteError(f"site file: [{where}] part_time, mains_only and default_jobs must be true or false")
    for key in ("close_below_gb", "push_back_below_gb"):
        value = getattr(node, key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < math.inf):
            raise SiteError(f"site file: [{where}] {key} = {value!r}; expected a positive number of GB")
    if node.push_back_below_mb > node.close_below_mb:
        raise SiteError(f"site file: [{where}] push_back_below_gb must not exceed close_below_gb: "
                        "a node pushes jobs back only after it would already be closed")
    jobs = node.jobs_memory_gb
    if jobs is not None and (isinstance(jobs, bool) or not isinstance(jobs, int | float) or not math.isfinite(jobs)
                             or not 0 < round(jobs * SLURM_MB_PER_GB) < node.real_memory_mb):
        raise SiteError(f"site file: [{where}] jobs_memory_gb = {jobs!r}; expected more than 0 and less than the "
                        f"node's memory ({node.real_memory_mb / SLURM_MB_PER_GB:.1f} GB)")
    own = {"address": node.address, "keep_cpus": node.keep_cpus}
    if not node.part_time:
        stray = sorted(k for k, v in own.items() if v is not None)
        if stray:
            raise SiteError(f"site file: [{where}] {', '.join(stray)} apply only with part_time = true")
        return
    if not isinstance(node.address, str) or not node.address:
        raise SiteError(f"site file: [{where}] a part_time node needs address, where the controller reaches it")
    if node.gpus:
        raise SiteError(f"site file: [{where}] a part_time node cannot have GPUs")
    if node.keep_cpus is not None and not (isinstance(node.keep_cpus, str) and _CPU_SPEC_LIST.fullmatch(node.keep_cpus)):
        raise SiteError(f"site file: [{where}] keep_cpus = {node.keep_cpus!r}; expected CPU numbers and ranges "
                        "such as \"0-11\" or \"0,2,4-7\"")


#: The shortest `[health_check] interval_s`: slurm's HealthCheckTimeout must stay below it.
MIN_HEALTH_CHECK_S = 10


def _parse_maintenance(raw: Any) -> Maintenance:
    t = _table(raw, "maintenance", {"status_file", "request_dir"}, {})
    for key in ("status_file", "request_dir"):
        if not (isinstance(t[key], str) and t[key].startswith("/")):
            raise SiteError(f"site file: maintenance.{key} = {t[key]!r}; expected an absolute path")
    return Maintenance(t["status_file"], t["request_dir"])


def _parse_health_check(raw: Any) -> HealthCheck:
    t = _table(raw, "health_check", set(), {"interval_s": 30})
    if type(t["interval_s"]) is not int or t["interval_s"] < MIN_HEALTH_CHECK_S:
        raise SiteError(f"site file: health_check.interval_s = {t['interval_s']!r}; expected a whole number of seconds, {MIN_HEALTH_CHECK_S} or more "
                        "(slurm's HealthCheckTimeout must stay below it)")
    return HealthCheck(t["interval_s"])


def _parse_lease(raw: Any) -> Lease:
    t = _table(raw, "lease", {"lease_dir", "keeper_user", "python"}, {"services": {}})
    return Lease(Path(t["lease_dir"]), t["keeper_user"], Path(t["python"]), dict(t["services"]))


def _parse_remotes(raw: Any) -> dict[str, Remote]:
    if not isinstance(raw, dict):
        raise SiteError(f"site file: [remote] must hold one [remote.NAME] table per site; {_HINT}")
    remotes = {}
    for name, table in raw.items():
        t = _table(table, f"remote.{name}", {"host", "root"}, {"submit_command": "gpujobs-submit", "status_command": "gpujobs-status", "data_root": ""})
        for key, value in t.items():
            if key == "data_root" and value == "":
                continue
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise SiteError(f"site file: remote.{name}.{key} = {value!r}; expected a non-empty text")
        if ".." in Path(t["root"]).parts:
            raise SiteError(f"site file: remote.{name}.root = {t['root']!r}; no '..' in it")
        if t["data_root"] and not t["data_root"].startswith("/"):
            raise SiteError(f"site file: remote.{name}.data_root = {t['data_root']!r}; expected an absolute path")
        remotes[name] = Remote(t["host"], t["root"], t["submit_command"], t["status_command"], t["data_root"])
    return remotes


def _parse_service(raw: Any, nodes: tuple[Node, ...]) -> Service:
    t = _table(raw, "service", {"cores"}, {"group": "gpujobs-service"})
    if len(nodes) != 1:  # the agent frees cores by suspending cpu jobs, and it counts one machine's cores
        raise SiteError("site file: [service] is built for a site of one node; leave the table out with several")
    cores, limit = t["cores"], _cpu_job_cap(nodes) or min(n.cpus for n in nodes)
    if type(cores) is not int or not 1 <= cores <= limit:
        raise SiteError(f"site file: service.cores = {cores!r}; expected a whole number from 1 to {limit}, the cores "
                        "CPU-only jobs may use, so suspending them can always make room")
    return Service(cores, t["group"])


#: A slurm release as SchedMD numbers it, and its tarball's digest: the `[slurm]` pin and the release tool check both.
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+")
RELEASE_SHA256 = re.compile(r"[0-9a-f]{64}")


class UpgradeExit(IntEnum):
    """`install.py upgrade-slurm`'s exit status, for a caller that must know whether slurm was changed (a maintenance
    window reboots only after DONE, and alerts a person on FAILED_AFTER_CHANGE or CONVERSION_RUNNING)."""

    DONE = 0
    REFUSED = 1  # nothing was changed: slurm runs as before
    FAILED_AFTER_CHANGE = 2  # a package changed and a later step failed: the restore commands were printed
    CONVERSION_RUNNING = 3  # slurmdbd is still converting the accounting database: it must be left running


class Follow(StrEnum):
    """`[slurm_release] follow`: which newer releases the watch proposes."""

    PATCH = "patch"  # newer releases of the running series only
    SERIES = "series"  # also the next series, once it has its first maintenance release (x.y.1)


@dataclass(frozen=True)
class SlurmRelease:
    """`[slurm_release]`: watch SchedMD's releases and prove a newer one in the container before a maintenance window
    installs it (`slurm_release.py`)."""

    follow: Follow
    checkout: str  # a checkout of these tools holding tests/integration and the installer
    records: str  # the folder of test results, owned by test_user (or root), writable by nobody else
    test_user: str  # who runs the test (an admin: the window trusts only their records)
    project: str = "gpujobs"  # the project the test job is submitted under; its size is the test cluster's (compose.yml)


def _parse_slurm_release(raw: Any) -> SlurmRelease:
    t = _table(raw, "slurm_release", {"checkout", "records", "test_user"}, {"follow": "series", "project": "gpujobs"})
    for key in ("checkout", "records"):
        if not (isinstance(t[key], str) and t[key].startswith("/")):
            raise SiteError(f"site file: slurm_release.{key} = {t[key]!r}; expected an absolute path")
    if not (isinstance(t["test_user"], str) and t["test_user"]) or not isinstance(t["project"], str):
        raise SiteError("site file: slurm_release.test_user and project must be names")
    return SlurmRelease(_enum(Follow, t["follow"], "slurm_release.follow"), t["checkout"], t["records"], t["test_user"],
                        t["project"])


def with_slurm_pin(text: str, version: str, sha256: str, steps: Sequence[tuple[str, str]] = ()) -> str:
    """A site file's text with its `[slurm]` table pinned to one release, through `steps` (none: straight to it); every
    other line kept byte for byte, and the table appended when the file has none."""
    pin = [f'version = "{version}"\n', f'sha256 = "{sha256}"\n']
    if steps:
        pin.append("steps = [" + ", ".join(f'"{v}:{d}"' for v, d in steps) + "]\n")
    out: list[str] = []
    inside = seen = skipping = False  # skipping: inside an old `steps = [...]` spread over lines
    for line in text.splitlines(keepends=True):
        if header := re.match(r"\s*\[(\[?)([^\]]+)\]", line):
            inside = header[2].strip() == "slurm" and not header[1]
            if inside:
                seen = True
                out += [line, *pin]
                continue
        if inside and (skipping or re.match(r"\s*(version|sha256|steps)\s*=", line)):
            skipping = re.match(r"\s*steps\s*=", line) is not None or skipping
            skipping = skipping and "]" not in line
            continue
        out.append(line)
    if not seen:
        out.append(("\n" if text and not text.endswith("\n") else "") + "\n[slurm]\n" + "".join(pin))
    return "".join(out)


def _release(version: Any, sha256: Any, where: str) -> tuple[str, str]:
    if not isinstance(version, str) or not RELEASE_VERSION.fullmatch(version):
        raise SiteError(f"site file: {where}version = {version!r}; expected a release such as 26.05.5")
    if not isinstance(sha256, str) or not RELEASE_SHA256.fullmatch(sha256):
        raise SiteError(f"site file: {where}sha256 must be 64 lowercase hex digits, the release tarball's digest")
    return version, sha256


def _parse_slurm(raw: Any) -> tuple[tuple[str, str] | None, tuple[tuple[str, str], ...]]:
    """([slurm] version and sha256 or None, the `steps` entries `"<version>:<sha256>"` as pairs)."""
    t = _table(raw, "slurm", set(), {"version": "", "sha256": "", "steps": []})
    version, sha256, steps = t["version"], t["sha256"], t["steps"]
    if not version and not sha256:
        if steps:
            raise SiteError("site file: slurm.steps needs version and sha256, the release the last step leads to")
        return None, ()
    if not (version and sha256):
        raise SiteError("site file: [slurm] needs both version and sha256, or neither")
    if not isinstance(steps, list):
        raise SiteError("site file: slurm.steps must be a list of \"<version>:<sha256>\" strings")
    pairs = []
    for entry in steps:
        if not isinstance(entry, str) or entry.count(":") != 1:
            raise SiteError(f"site file: slurm.steps entry {entry!r}; expected \"<version>:<sha256>\"")
        pairs.append(_release(*entry.split(":"), "slurm.steps entry "))
    target = _release(version, sha256, "slurm.")
    order = [tuple(map(int, v.split("."))) for v, _ in (*pairs, target)]
    if any(a >= b for a, b in zip(order, order[1:])):
        raise SiteError("site file: slurm.steps must be in ascending order of release, each older than slurm.version "
                        f"(got {', '.join(v for v, _ in (*pairs, target))})")
    return target, tuple(pairs)


def parse(raw: dict[str, Any]) -> Site:
    top = _table(
        raw,
        "top level",
        {"name", "nodes"},
        {
            "cluster_name": None,
            "account": "gpujobs",
            "state_dir": "/var/lib/gpujobs",
            "install_dir": "/opt/gpujobs",
            "admin_group": "gpujobs-admin",
            "unified_memory": False,
            "lease": None,
            "service": None,
            "remote": {},
            "slurm": None,
            "health_check": None,
            "maintenance": None,
            "slurm_release": None,
            "shared_state": False,
            "node_python": "/usr/bin/python3",
            "requeue_delay_s": 60,
            "sharing_memory_fraction": None,
            "cpus_from_history": CoresFromHistory.RECOMMEND.value,
            "hostguard": False,
            "dgx_gate": False,
            "constrain_devices": True,
            "metrics": {},
            "portal": {},
            "notify_command": [],
        },
    )
    m = _table(
        top["metrics"],
        "metrics",
        set(),
        {"mode": "off", "prometheus_url": "", "grafana_url": "", "grafana_port": 3000, "retention_days": 365,
         "allow_from": []},
    )
    p = _table(
        top["portal"],
        "portal",
        set(),
        {
            "mode": "off",
            "auth": "dex-local",
            "server_name": "",
            "tls_cert": "",
            "tls_key": "",
            "auth_settings": {},
            "session_days": 0,
            "allow_from": [],
        },
    )
    if not isinstance(p["session_days"], int) or isinstance(p["session_days"], bool) or p["session_days"] < 0:
        raise SiteError(f"site file: portal.session_days = {p['session_days']!r}; expected a whole number of days, 0 or more")
    if type(top["shared_state"]) is not bool:
        raise SiteError("site file: shared_state must be true or false")
    if not isinstance(top["node_python"], str) or not top["node_python"].startswith("/"):
        raise SiteError(f"site file: node_python = {top['node_python']!r}; expected an absolute path")
    if type(top["requeue_delay_s"]) is not int or top["requeue_delay_s"] < 1:
        raise SiteError(f"site file: requeue_delay_s = {top['requeue_delay_s']!r}; expected a whole number of seconds, 1 or more")
    fraction = top["sharing_memory_fraction"]
    if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction <= 1):
        raise SiteError(f"site file: sharing_memory_fraction = {fraction!r}; expected a number above 0 and at most 1")
    nodes = tuple(_parse_node(n, i) for i, n in enumerate(top["nodes"]))
    if nodes and not any(not n.part_time for n in nodes):
        raise SiteError("site file: every node is part_time; slurmctld runs on a full node, so list at least one")
    slurm = _parse_slurm(top["slurm"]) if top["slurm"] is not None else (None, ())
    return Site(
        name=top["name"],
        cluster_name=top["cluster_name"] or top["name"],
        account=top["account"],
        state_dir=Path(top["state_dir"]),
        install_dir=Path(top["install_dir"]),
        admin_group=top["admin_group"],
        nodes=nodes,
        unified_memory=top["unified_memory"],
        lease=_parse_lease(top["lease"]) if top["lease"] is not None else None,
        service=_parse_service(top["service"], tuple(n for n in nodes if not n.part_time)) if top["service"] is not None else None,
        hostguard=top["hostguard"],
        dgx_gate=top["dgx_gate"],
        metrics=Metrics(
            _enum(MetricsMode, m["mode"], "metrics.mode"),
            m["prometheus_url"],
            m["grafana_url"],
            m["grafana_port"],
            m["retention_days"],
            _networks(m["allow_from"], "metrics.allow_from"),
        ),
        portal=Portal(
            _enum(PortalMode, p["mode"], "portal.mode"),
            _enum(AuthMode, p["auth"], "portal.auth"),
            p["server_name"],
            _opt_path(p["tls_cert"]),
            _opt_path(p["tls_key"]),
            dict(p["auth_settings"]),
            p["session_days"],
            _networks(p["allow_from"], "portal.allow_from"),
        ),
        notify_command=tuple(top["notify_command"]),
        slurm_build=slurm[0],
        slurm_steps=slurm[1],
        health_check=_parse_health_check(top["health_check"]) if top["health_check"] is not None else None,
        maintenance=_parse_maintenance(top["maintenance"]) if top["maintenance"] is not None else None,
        slurm_release=_parse_slurm_release(top["slurm_release"]) if top["slurm_release"] is not None else None,
        constrain_devices=top["constrain_devices"],
        shared_state=top["shared_state"],
        node_python=top["node_python"],
        remotes=_parse_remotes(top["remote"]),
        requeue_delay_s=top["requeue_delay_s"],
        sharing_memory_fraction=None if fraction is None else float(fraction),
        cpus_from_history=_enum(CoresFromHistory, top["cpus_from_history"], "cpus_from_history"),
    )


def load(path: Path | None = None) -> Site:
    path = path or SITE_FILE
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise SiteError(f"no site file at {path}; {_HINT}") from None
    except tomllib.TOMLDecodeError as exc:
        raise SiteError(f"site file {path} is not valid TOML: {exc}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise SiteError(f"cannot read site file {path}: {exc}") from None
    try:
        return parse(raw)
    except (TypeError, ValueError) as exc:  # a value of the wrong kind, say a number where a path belongs
        raise SiteError(f"site file {path} has a value of the wrong kind: {exc}") from None


def is_admin(site: Site, user: str) -> bool:
    """True when `user` is in the admin group, by listed membership or as the primary group."""
    return in_group(site.admin_group, user)


def may_submit_service(site: Site, user: str) -> bool:
    """True when `user` may use the service partition (D20): its group or the admin group, as its AllowGroups."""
    return site.service is not None and (in_group(site.service.group, user) or is_admin(site, user))


def in_group(name: str, user: str) -> bool:
    try:
        group = grp.getgrnam(name)
        primary = pwd.getpwnam(user).pw_gid
    except KeyError:
        return False
    return user in group.gr_mem or primary == group.gr_gid
