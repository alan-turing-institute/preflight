"""The site file: every difference between one machine and another, read once and typed.

The TOML layout maps one-to-one onto the dataclasses below (`[[nodes]]`, `[[nodes.gpus]]`, `[lease]`,
`[lease.services]`, `[slurm]`, `[metrics]`, `[portal]`, `[portal.auth_settings]`). Unknown keys are an error at every
table level, so a typo cannot silently fall back to a default. `site.example.toml` documents each field.
"""

from __future__ import annotations

import grp
import ipaddress
import os
import pwd
import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SITE_FILE = Path(os.environ.get("GPUJOBS_SITE", "/etc/gpujobs/site.toml"))


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


class SiteError(Exception):
    """The site file is missing or wrong; the message says what to do."""


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    device: str  # e.g. "/dev/nvidia0"
    type: str


@dataclass(frozen=True)
class Node:
    name: str
    cpus: int
    real_memory_mb: int
    gpus: tuple[Gpu, ...]
    cores_per_gpu: int | None = None  # held for GPU jobs; CPU-only jobs never use them (slurm's RestrictedCoresPerGPU)

    @property
    def gpu_cores(self) -> int:
        """Cores held for GPU jobs on this node: 0 when `cores_per_gpu` is unset."""
        return (self.cores_per_gpu or 0) * len(self.gpus)


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

    @property
    def cpu_job_cap(self) -> int | None:
        return _cpu_job_cap(self.nodes)


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


def _parse_node(raw: Any, i: int) -> Node:
    where = f"nodes[{i}]"
    t = _table(raw, where, {"name", "cpus", "real_memory_mb"}, {"gpus": [], "cores_per_gpu": None})
    gpus = tuple(
        Gpu(**_table(g, f"{where}.gpus[{j}]", {"index", "uuid", "device", "type"}, {}))
        for j, g in enumerate(t["gpus"])
    )
    node = Node(t["name"], t["cpus"], t["real_memory_mb"], gpus, t["cores_per_gpu"])
    if node.cores_per_gpu is not None and not (
        type(node.cores_per_gpu) is int and gpus and node.cores_per_gpu >= 1 and node.gpu_cores < node.cpus
    ):
        raise SiteError(f"site file: [{where}] cores_per_gpu must be a whole number of at least 1, needs GPUs, and must leave CPU-only jobs a core "
                        f"({node.cores_per_gpu} x {len(gpus)} GPU(s) of {node.cpus} cores)")
    return node


def _parse_lease(raw: Any) -> Lease:
    t = _table(raw, "lease", {"lease_dir", "keeper_user", "python"}, {"services": {}})
    return Lease(Path(t["lease_dir"]), t["keeper_user"], Path(t["python"]), dict(t["services"]))


def _parse_service(raw: Any, nodes: tuple[Node, ...]) -> Service:
    t = _table(raw, "service", {"cores"}, {"group": "gpujobs-service"})
    if len(nodes) != 1:  # the agent frees cores by suspending cpu jobs, and it counts one machine's cores
        raise SiteError("site file: [service] is built for a site of one node; leave the table out with several")
    cores, limit = t["cores"], _cpu_job_cap(nodes) or min(n.cpus for n in nodes)
    if type(cores) is not int or not 1 <= cores <= limit:
        raise SiteError(f"site file: service.cores = {cores!r}; expected a whole number from 1 to {limit}, the cores "
                        "CPU-only jobs may use, so suspending them can always make room")
    return Service(cores, t["group"])


def _release(version: Any, sha256: Any, where: str) -> tuple[str, str]:
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SiteError(f"site file: {where}version = {version!r}; expected a release such as 26.05.5")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
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
    return _release(version, sha256, "slurm."), tuple(pairs)


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
            "slurm": None,
            "hostguard": False,
            "dgx_gate": False,
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
    nodes = tuple(_parse_node(n, i) for i, n in enumerate(top["nodes"]))
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
        service=_parse_service(top["service"], nodes) if top["service"] is not None else None,
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
