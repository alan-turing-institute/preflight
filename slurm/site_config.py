"""The site file: every difference between one machine and another, read once and typed.

The TOML layout maps one-to-one onto the dataclasses below (`[[nodes]]`, `[[nodes.gpus]]`, `[lease]`,
`[lease.services]`, `[metrics]`, `[portal]`, `[portal.auth_settings]`). Unknown keys are an error at every
table level, so a typo cannot silently fall back to a default. `site.example.toml` documents each field.
"""

from __future__ import annotations

import grp
import ipaddress
import os
import pwd
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


@dataclass(frozen=True)
class Lease:
    lease_dir: Path
    keeper_user: str
    python: Path
    services: dict[str, str]  # service name -> how it yields


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
    hostguard: bool
    dgx_gate: bool
    metrics: Metrics
    portal: Portal
    notify_command: tuple[str, ...]  # argv; empty means no notification


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
    t = _table(raw, where, {"name", "cpus", "real_memory_mb"}, {"gpus": []})
    gpus = tuple(
        Gpu(**_table(g, f"{where}.gpus[{j}]", {"index", "uuid", "device", "type"}, {}))
        for j, g in enumerate(t["gpus"])
    )
    return Node(t["name"], t["cpus"], t["real_memory_mb"], gpus)


def _parse_lease(raw: Any) -> Lease:
    t = _table(raw, "lease", {"lease_dir", "keeper_user", "python"}, {"services": {}})
    return Lease(Path(t["lease_dir"]), t["keeper_user"], Path(t["python"]), dict(t["services"]))


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
    return Site(
        name=top["name"],
        cluster_name=top["cluster_name"] or top["name"],
        account=top["account"],
        state_dir=Path(top["state_dir"]),
        install_dir=Path(top["install_dir"]),
        admin_group=top["admin_group"],
        nodes=tuple(_parse_node(n, i) for i, n in enumerate(top["nodes"])),
        unified_memory=top["unified_memory"],
        lease=_parse_lease(top["lease"]) if top["lease"] is not None else None,
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
    try:
        group = grp.getgrnam(site.admin_group)
        primary = pwd.getpwnam(user).pw_gid
    except KeyError:
        return False
    return user in group.gr_mem or primary == group.gr_gid
