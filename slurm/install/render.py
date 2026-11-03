"""Render the installed settings files for one component from the site file.

Templates are `string.Template` files under `templates/<component>/`; `render` returns every file the component
installs, keyed by its absolute target path. Substitution is strict: a placeholder with no value is an error.
Import as `import render` with `slurm/install/` on sys.path (standalone drivers, not a package).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from site_config import Node, Site

TEMPLATES = Path(__file__).resolve().parent / "templates"
UNITS = ("slurmctld", "slurmd", "munge", "slurmdbd")
DROPIN = "10-restart-protect.conf"
FILE_MODE = 0o644
#: In the first line of every slurm.conf this installer writes: a slurm.conf without it was set up by someone else.
RENDERED_MARK = "Rendered from the site file by slurm/install/render.py"
EXEC_MODE = 0o755
SECRET_MODE = 0o600
#: slurm's SlurmUser; `templates/slurm/slurm.conf` sets the same name.
SLURM_USER = "slurm"
#: The part-time nodes' NodeName lines live in their own file, which slurm.conf includes (no path, so configless ships it):
#: the agent rewrites it when an admin changes a machine's memory or kept cores on the page (agent.py, B4).
PART_TIME_CONF = "part_time_nodes.conf"
PART_TIME_CONF_PATH = f"/etc/slurm/{PART_TIME_CONF}"
#: SlurmUser's ssh key (under the state dir) for the read-only copy of a part-time machine's job state (B5b).
COPY_KEY = "agent/ssh/id_ed25519"
#: Slurm's default memory for a job that names none, capped below at the smallest node (bigger refuses such jobs).
DEF_MEM_MB = 16384


@dataclass(frozen=True)
class FileSpec:
    content: str
    mode: int
    owner: str


def ip_filter_lines(networks: tuple[str, ...]) -> str:
    """systemd lines that drop every packet of a service except to and from localhost and `networks` (empty: none).

    `man systemd.resource-control`, IPAddressAllow=: the allow list is checked first, then IPAddressDeny=. It filters
    both directions, so a service that connects out (Apache to its own server name) needs that address listed too.
    """
    if not networks:
        return ""
    return f"IPAddressDeny=any\nIPAddressAllow=localhost {' '.join(networks)}\n"


def _template(component: str, name: str) -> Template:
    return Template((TEMPLATES / component / name).read_text())


def _node_line(node: Node) -> str:
    # gres.conf(5) / slurm.conf(5) typed form: Gres=gpu:<type>:<count> (slurm.conf(5) example "Gres=gpu:tesla:1").
    gres = f" Gres=gpu:{node.gpus[0].type}:{len(node.gpus)}" if node.gpus else ""
    gres += f",shard:{node.shards}" if node.shards else ""  # the node's shards, summed over its sharing cards
    address = f" NodeAddr={node.address}" if node.part_time else ""
    spec = f" CpuSpecList={node.keep_cpus}" if node.keep_cpus else ""
    spec += f" MemSpecLimit={node.mem_spec_limit_mb}" if node.mem_spec_limit_mb is not None else ""
    return (
        f"NodeName={node.name}{address} CPUs={node.cpus} Boards=1 SocketsPerBoard=1 CoresPerSocket={node.cores_per_socket or node.cpus} "
        f"ThreadsPerCore=1 RealMemory={node.real_memory_mb}{spec}{gres} State=UNKNOWN\n"
    )


def _gres_lines(site: Site) -> str:
    """gres.conf lines: one per GPU with its Type (gres.conf(5) example: "Name=gpu Type=tesla File=/dev/nvidia1 COREs=2,3"),
    and for a sharing card a `Name=shard Count=<its shards>` line on the same device file (slurm copies Type and Cores)."""
    lines = ""
    for n in site.nodes:
        for g in n.gpus:
            lines += f"NodeName={n.name} Name=gpu Type={g.type} File={g.device}\n"
            if g.shards:
                lines += f"NodeName={n.name} Name=shard Count={g.shards} File={g.device}\n"
    return lines


def part_time_nodes_conf(site: Site) -> str:
    """The part-time nodes' NodeName lines: the content of `part_time_nodes.conf` ("" when the site has none)."""
    if not site.part_time_nodes:
        return ""
    return ("# The part-time nodes: written by slurm/install/render.py, and by the agent when the page changes a\n"
            "# machine's memory or kept cores.\n" + "".join(_node_line(n) for n in site.part_time_nodes))


def _part_time_include(site: Site) -> str:
    return f"# The part-time nodes are in their own file.\nInclude {PART_TIME_CONF}\n" if site.part_time_nodes else ""


def _health_lines(site: Site) -> str:
    """HealthCheck* settings when the site has `[health_check]`, and configless mode (the part-time nodes read
    slurm.conf from the controller) when it has a part-time node."""
    lines = ""
    if site.health_check:
        lines += (
            "# Every node runs the health check as root each interval; it closes and reopens the node itself.\n"
            f"HealthCheckProgram={site.install_dir}/slurm/node_check.py\n"
            f"HealthCheckInterval={site.health_check.interval_s}\n"
        )
        if site.slurm_build:  # the distribution's 23.11 does not know the key and refuses to read the file with it
            # slurm's default timeout (60 s) must stay below the interval or slurmd logs an error at every start
            lines += f"HealthCheckTimeout={min(60, site.health_check.interval_s - 5)}\n"
        lines += "HealthCheckNodeState=ANY\n"
    if site.part_time_nodes:
        lines += "# Part-time nodes fetch their settings from slurmctld (slurmd --conf-server).\nSlurmctldParameters=enable_configless\n"
    return lines


def _part_time_partition(site: Site) -> str:
    """The `part_time` partition: only the part-time nodes, never pre-empted by slurm, no per-node CPU cap."""
    if not site.part_time_nodes:
        return ""
    names = ",".join(n.name for n in site.part_time_nodes)
    return ("# part_time: other people's machines, lent when idle (submit.py --anywhere); node_check.py closes them.\n"
            f"PartitionName=part_time Nodes={names} PriorityTier=1 PreemptMode=OFF MaxTime=INFINITE State=UP\n")


def node_check_json(site: Site) -> str:
    """`/etc/gpujobs/node_check.json`: each node's health-check settings, keyed by node name (the render system
    writes one file per path, so the nodes share one; node_check.py reads its own hostname's entry)."""
    return json.dumps({
        n.name: {
            "node": n.name,
            "close_below_mb": n.close_below_mb,
            "push_back_below_mb": n.push_back_below_mb,
            "mains_only": n.mains_only,
            "meminfo_path": "/proc/meminfo",
            "power_supply_dir": "/sys/class/power_supply",
            "state_path": "/var/lib/gpujobs/node_check/state.json",
            "log_path": "/var/log/gpujobs/node_check.log",
        } for n in site.nodes
    }, indent=2) + "\n"


def _service_partition(site: Site, names: list[str]) -> str:
    """The service partition (D20), when the site has one: background services' CPU stand-ins. The agent suspends
    CPU-only jobs to make room (agent.py, step 3b). Urgent work requeues it. PriorityTier=2, the tier of `normal`: slurm
    pre-empts only strictly lower tiers, so a GPU job never requeues a service job (C3 trial, D23)."""
    if site.service is None:
        return ""
    return (
        "# service: background services' CPU stand-ins (submit.py --service); the agent suspends cpu jobs to make room.\n"
        f"PartitionName=service Nodes={','.join(names)} PriorityTier=2 PreemptMode=REQUEUE MaxTime=INFINITE State=UP "
        f"MaxCPUsPerNode={site.service.cores} AllowGroups={site.service.group},{site.admin_group}\n"
    )


def _slurm(site: Site, extra: dict[str, str]) -> dict[str, FileSpec]:
    names = [n.name for n in site.full_nodes]  # slurmctld and the partitions' Nodes: full nodes; part-time ones have theirs
    extra_conf = extra.get("slurm_conf", "")
    values = {
        "site_name": site.name,
        "head_node": names[0],
        "cluster_name": site.cluster_name,
        "partition_nodes": ",".join(names),
        # slurm.conf(5) MaxCPUsPerNode: a partition limited to a subset of the cores keeps the rest for GPU jobs
        "cpu_job_limit": f" MaxCPUsPerNode={site.cpu_job_cap}" if site.cpu_job_cap else "",
        "node_lines": "".join(_node_line(n) for n in site.full_nodes),
        "part_time_include": _part_time_include(site),
        "gres_lines": _gres_lines(site),
        "gres_types": "gpu,shard" if site.sharing else "gpu",
        "accounting_tres": "gres/gpu,gres/shard" if site.sharing else "gres/gpu",
        "admin_group": site.admin_group,
        "constrain_devices": "yes" if site.constrain_devices else "no",
        "requeue_delay_s": site.requeue_delay_s,
        "service_partition": _service_partition(site, names),
        "def_mem_mb": min(DEF_MEM_MB, *(n.real_memory_mb - (n.mem_spec_limit_mb or 0) for n in site.nodes)),
        "health_lines": _health_lines(site),
        "part_time_partition": _part_time_partition(site),
        "extra_slurm_conf": extra_conf if not extra_conf or extra_conf.endswith("\n") else extra_conf + "\n",
    }
    out = {
        "/etc/slurm/slurm.conf": FileSpec(_template("slurm", "slurm.conf").substitute(values), FILE_MODE, "root"),
        "/etc/slurm/cgroup.conf": FileSpec(_template("slurm", "cgroup.conf").substitute(values), FILE_MODE, "root"),
        "/etc/slurm/gres.conf": FileSpec(_template("slurm", "gres.conf").substitute(values), FILE_MODE, "root"),
        "/etc/slurm/slurmdbd.conf": FileSpec(
            _template("slurm", "slurmdbd.conf").substitute(values), SECRET_MODE, SLURM_USER
        ),
        "/etc/mysql/mariadb.conf.d/90-slurm.cnf": FileSpec(
            _template("slurm", "mariadb-slurm.cnf").substitute(values), FILE_MODE, "root"
        ),
    }
    if site.part_time_nodes:  # owned by SlurmUser: the agent, which runs as it, rewrites the file in place
        out[PART_TIME_CONF_PATH] = FileSpec(part_time_nodes_conf(site), FILE_MODE, SLURM_USER)
    dropin = _template("slurm", DROPIN).substitute(values)
    for unit in UNITS:
        out[f"/etc/systemd/system/{unit}.service.d/{DROPIN}"] = FileSpec(dropin, FILE_MODE, "root")
    return out


_COMPONENTS = {"slurm": _slurm}


def render(site: Site, component: str, extra: dict[str, str] | None = None) -> dict[str, FileSpec]:
    """Every file `component` installs for `site`, keyed by absolute target path."""
    try:
        build = _COMPONENTS[component]
    except KeyError:
        raise ValueError(f"unknown component {component!r}; known: {', '.join(_COMPONENTS)}") from None
    return build(site, extra or {})
