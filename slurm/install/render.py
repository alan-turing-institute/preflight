"""Render the installed settings files for one component from the site file.

Templates are `string.Template` files under `templates/<component>/`; `render` returns every file the component
installs, keyed by its absolute target path. Substitution is strict: a placeholder with no value is an error.
Import as `import render` with `slurm/install/` on sys.path (standalone drivers, not a package).
"""

from __future__ import annotations

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
    return (
        f"NodeName={node.name} CPUs={node.cpus} Boards=1 SocketsPerBoard=1 CoresPerSocket={node.cpus} "
        f"ThreadsPerCore=1 RealMemory={node.real_memory_mb}{gres} State=UNKNOWN\n"
    )


def _service_partition(site: Site, names: list[str]) -> str:
    """The service partition (D20), when the site has one: background services' CPU stand-ins. Slurm itself never
    pre-empts for it; the agent suspends CPU-only jobs to make room (agent.py, step 3b). Urgent work requeues it."""
    if site.service is None:
        return ""
    return (
        "# service: background services' CPU stand-ins (submit.py --service); the agent suspends cpu jobs to make room.\n"
        f"PartitionName=service Nodes={','.join(names)} PriorityTier=1 PreemptMode=REQUEUE MaxTime=INFINITE State=UP "
        f"MaxCPUsPerNode={site.service.cores} AllowGroups={site.service.group},{site.admin_group}\n"
    )


def _slurm(site: Site, extra: dict[str, str]) -> dict[str, FileSpec]:
    names = [n.name for n in site.nodes]
    # One line per GPU with its Type (gres.conf(5) example: "Name=gpu Type=tesla File=/dev/nvidia1 COREs=2,3").
    gres_lines = "".join(
        f"NodeName={n.name} Name=gpu Type={g.type} File={g.device}\n" for n in site.nodes for g in n.gpus
    )
    extra_conf = extra.get("slurm_conf", "")
    values = {
        "site_name": site.name,
        "head_node": names[0],
        "cluster_name": site.cluster_name,
        "partition_nodes": ",".join(names),
        # slurm.conf(5) MaxCPUsPerNode: a partition limited to a subset of the cores keeps the rest for GPU jobs
        "cpu_job_limit": f" MaxCPUsPerNode={site.cpu_job_cap}" if site.cpu_job_cap else "",
        "node_lines": "".join(_node_line(n) for n in site.nodes),
        "gres_lines": gres_lines,
        "admin_group": site.admin_group,
        "service_partition": _service_partition(site, names),
        "def_mem_mb": min(DEF_MEM_MB, *(n.real_memory_mb for n in site.nodes)),
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
