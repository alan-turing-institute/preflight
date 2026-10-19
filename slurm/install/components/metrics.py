"""Metrics component: Princeton Jobstats (per-job CPU, memory and GPU history) plus our own job-memory panel.

Installs, as Jobstats' documentation specifies (https://princetonuniversity.github.io/jobstats/, `docs/setup/*.md` at
the pinned commit, cited as `<file>:<line>`):
  - the node, cgroup (plazonic/cgroup_exporter) and GPU (plazonic/nvidia_gpu_prometheus_exporter) exporters
    (`docs/setup/overview.md:6`; `node.md:3`; `cgroups.md:16,71-74`; `gpu_node_scripts.md:3`);
  - the slurm prolog and epilog that write `/run/gpustat/<gpu>` (`gpu_node_scripts.md:23-44`) and the slurmctld epilog
    that stores each job's summary (`summaries.md:5-12`): our own short scripts in `files()` doing what the docs
    describe, so they exist before slurm is restarted and none of Jobstats' GPL scripts is copied or fetched;
  - the `jobstats` command and its `config.py` (`jobstats_command.md:31-56,69-81`);
  - mode "local": Prometheus (`prometheus.md:3-55`) and Grafana with the Jobstats dashboard (`grafana.md:5`);
  - the Open OnDemand helper app (`ood.md:3`), only when the portal is on.
Jobstats is GPL-2.0: none of its files are copied into this repository, each is fetched at install from the pinned
commit and checked against the sha256 recorded in `DOWNLOADS` below. Install method per piece, in order of
preference: distro package (Prometheus, Python libraries), vendor apt repository (Grafana), upstream release binary
(node, cgroup and, on x86_64, GPU exporter), build from the pinned source (GPU exporter on aarch64, for which upstream
publishes no binary).

Interface (the installer loads it by name): NAME, enabled, files, packages, refusals, post_install, apt_sources,
apt_flags, slurm_conf_lines. Import as `import metrics` with `slurm/install/` and `slurm/` on sys.path.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import PurePosixPath
from string import Template
from typing import TYPE_CHECKING

import render
from site_config import MetricsMode, PortalMode

if TYPE_CHECKING:
    from site_config import Site

NAME = "metrics"
TEMPLATES = render.TEMPLATES / NAME

SCRAPE_INTERVAL_S = 30  # `prometheus.md:18`; jobstats' SAMPLING_PERIOD must match (`jobstats_command.md:90`)


class Port(IntEnum):
    # Ports are the exporters' defaults (cgroup 9306 and GPU 9445: `prometheus.md:57`), except the node exporter:
    # this component runs its own instance, so it takes 9101 and never collides with another one on 9100.
    NODE = 9101
    CGROUP = 9306
    GPU = 9445
    PROMETHEUS = 9090


class Arch(StrEnum):
    X86_64 = "x86_64"
    AARCH64 = "aarch64"


# Release-asset architecture names of the upstream tarballs.
RELEASE_ARCH = {Arch.X86_64: "amd64", Arch.AARCH64: "arm64"}

CACHE_DIR = PurePosixPath("/var/cache/gpujobs/metrics")
BIN_DIR = PurePosixPath("/usr/local/lib/gpujobs")
JOBSTATS_DIR = PurePosixPath("/usr/local/jobstats")  # `jobstats_command.md:36`
JOBSTATS_LINK = PurePosixPath("/usr/local/bin/jobstats")  # `jobstats_command.md:53`
SLURMCTLD_EPILOG = PurePosixPath("/usr/local/sbin/slurmctldepilog.sh")  # `summaries.md:9`
PROLOG_DIR = PurePosixPath("/etc/slurm/prolog.d")  # `gpu_node_scripts.md:30`
EPILOG_DIR = PurePosixPath("/etc/slurm/epilog.d")
TARGETS_FILE = PurePosixPath("/etc/gpujobs/prometheus_targets.json")
PROMETHEUS_CONFIG = PurePosixPath("/etc/prometheus/gpujobs.yml")
GRAFANA_DASHBOARDS = PurePosixPath("/etc/grafana/gpujobs-dashboards")
GRAFANA_ENV_FILE = PurePosixPath("/etc/grafana/gpujobs.env")  # holds the first admin password, mode 0600
GRAFANA_KEY = PurePosixPath("/etc/apt/keyrings/grafana.asc")  # Grafana install docs: apt repository section
# The helper README says "drop in /var/www/ood/apps/sys" (`ood-jobstats-helper/README.md:28,33`); the directory is
# named jobstats so the helper answers at /pun/sys/jobstats/<jobid>, as in Jobstats' README, where our app links.
OOD_HELPER_DIR = PurePosixPath("/var/www/ood/apps/sys/jobstats")
DATASOURCE_NAME = "gpujobs-prometheus"
JOBSTATS_DASHBOARD_FILE = "jobstats-single-job.json"

SERVICES = ("gpujobs-node-exporter", "gpujobs-cgroup-exporter", "gpujobs-gpu-exporter")

# OOD's `custom.grafana` block minus `host` (the portal component adds it; OOD docs, customizations.rst "Grafana
# support", lines 1331-1356). The dashboard is Jobstats' (`grafana/Single_Job_Stats.json` at the pinned commit): "uid",
# slug `name` of its title, and its graph panels "Job CPU Utilization" id 7, "Job CPU Memory Utilization" id 9,
# "GPU Utilization" id 2, "GPU Memory utilization" id 3 (those two sit in a collapsed row; on a unified-memory GPU the
# memory one has no data). `labels` map OOD's cluster/host/jobid to the dashboard's variables (it has `JobID` and
# `node`, no cluster). Grafana's default organisation has id 1.
GRAFANA_DASHBOARD = {
    "orgId": 1,
    "dashboard": {
        "name": "single-job-stats",
        "uid": "HRLkiLS7k",
        "panels": {"cpu": 7, "memory": 9, "gpu-util": 2, "gpu-memory": 3},
    },
    "labels": {"cluster": "cluster", "host": "node", "jobid": "JobID"},
}
OWN_DASHBOARD = {"uid": "gpujobs-job-memory", "title": "GPU memory by job"}


@dataclass(frozen=True)
class Download:
    name: str
    version: str
    url: str
    sha256: str
    filename: str


_JOBSTATS_VERSION = "v1.1.0"
_JOBSTATS_COMMIT = "fe7b54fba174355b088c384b43aea9246fd000f8"
_GPU_EXPORTER_VERSION = "v0.2.4"
_GPU_EXPORTER_COMMIT = "45c2ce0517e80247956da4f7e94b18831ffdc2fe"
_CGROUP_EXPORTER_VERSION = "v0.3.4"
_NODE_EXPORTER_VERSION = "1.12.1"


def _jobstats_raw(name: str, path: str, sha256: str) -> Download:
    # Pinned by commit (a tag can move).
    url = f"https://raw.githubusercontent.com/PrincetonUniversity/jobstats/{_JOBSTATS_COMMIT}/{path}"
    return Download(name, _JOBSTATS_VERSION, url, sha256, name)


def _node_exporter(arch: Arch, sha256: str) -> Download:
    base = f"node_exporter-{_NODE_EXPORTER_VERSION}.linux-{RELEASE_ARCH[arch]}.tar.gz"
    url = f"https://github.com/prometheus/node_exporter/releases/download/v{_NODE_EXPORTER_VERSION}/{base}"
    return Download(f"node-exporter-{arch}", _NODE_EXPORTER_VERSION, url, sha256, base)


# Every download this component makes. post_install fetches only entries of this table, each checked with sha256sum.
DOWNLOADS: tuple[Download, ...] = (
    _jobstats_raw("jobstats", "jobstats", "4dd0b5e4673c22c09fd7ac2ed5e4c53b12fc25eaadc745447f7622fd2ee9e32e"),
    _jobstats_raw("jobstats.py", "jobstats.py", "89a2edfaf14d206a20526aae726eecde15a2a396d2074381071e1546a6173fb7"),
    _jobstats_raw(
        "output_formatters.py",
        "output_formatters.py",
        "262e9d5d4000332e858ff0281c766b0fbdb0f284b2431f6949fb4eb0eba4f3c7",
    ),
    _jobstats_raw(
        "Single_Job_Stats.json",
        "grafana/Single_Job_Stats.json",
        "c5df98c9e965d492caa5dcdae29ccd004f5cfe39bcada968a27ec2601430b08d",
    ),
    Download(
        "ood-jobstats-helper",
        _JOBSTATS_VERSION,
        f"https://github.com/PrincetonUniversity/jobstats/archive/{_JOBSTATS_COMMIT}.tar.gz",
        "19d2c8ff41bcd5cbfac07f5bbbf521087f7f61d5df888658149fc346fd03f081",
        f"jobstats-{_JOBSTATS_COMMIT}.tar.gz",
    ),
    _node_exporter(Arch.X86_64, "b51d8a76aa2a9156a55d501aca6276fae09e262259a5e4e831d2c2222f084e63"),
    _node_exporter(Arch.AARCH64, "ad35b605f9954b9f1ffddf5ba054bdc5a98d790b9eae5291e1eeb83f1ecbd0e7"),
    Download(
        "cgroup-exporter",
        _CGROUP_EXPORTER_VERSION,
        f"https://github.com/plazonic/cgroup_exporter/releases/download/{_CGROUP_EXPORTER_VERSION}/"
        f"cgroup_exporter-{_CGROUP_EXPORTER_VERSION}-binaries.tar.gz",
        "f30fe3da883b91626ad57b8bc25904431c0e4c6ca6eea565eed75661769e9fe8",
        f"cgroup_exporter-{_CGROUP_EXPORTER_VERSION}-binaries.tar.gz",
    ),
    Download(
        "gpu-exporter-x86_64",
        _GPU_EXPORTER_VERSION,
        "https://github.com/plazonic/nvidia_gpu_prometheus_exporter/releases/download/"
        f"{_GPU_EXPORTER_VERSION}/nvml_exporter-{_GPU_EXPORTER_VERSION}-x86_64.tar.gz",
        "c6fa39ab6ff8b950ab0f25fb26c5c6f3579f53f1512a8f3e2323169cbed3e5b6",
        f"nvml_exporter-{_GPU_EXPORTER_VERSION}-x86_64.tar.gz",
    ),
    # No aarch64 release asset exists upstream (releases list x86_64 only): built from this source.
    Download(
        "gpu-exporter-source",
        _GPU_EXPORTER_VERSION,
        f"https://github.com/plazonic/nvidia_gpu_prometheus_exporter/archive/{_GPU_EXPORTER_COMMIT}.tar.gz",
        "50f2d76c1ebdaa5b00235432921da7451a42c2626bd9423ad4c2c2eaddb52fec",
        f"nvidia_gpu_prometheus_exporter-{_GPU_EXPORTER_COMMIT}.tar.gz",
    ),
)
_BY_NAME = {d.name: d for d in DOWNLOADS}


def _arch() -> Arch | None:
    try:
        return Arch(platform.machine())
    except ValueError:
        return None


def _is_local(site: Site) -> bool:
    return site.metrics.mode == MetricsMode.LOCAL


def _has_gpus(site: Site) -> bool:
    return any(node.gpus for node in site.nodes)


def _portal_on(site: Site) -> bool:
    return site.portal.mode != PortalMode.OFF


def _prometheus_server(site: Site) -> str:
    return f"http://127.0.0.1:{Port.PROMETHEUS}" if _is_local(site) else site.metrics.prometheus_url


def _template(name: str, **values: object) -> str:
    return Template((TEMPLATES / name).read_text()).substitute(values)


def _spec(content: str, mode: int = render.FILE_MODE) -> render.FileSpec:
    return render.FileSpec(content, mode, "root")


def enabled(site: Site) -> bool:
    return site.metrics.mode != MetricsMode.OFF


def refusals(site: Site) -> list[str]:
    if not enabled(site):
        return []
    out = []
    if _arch() is None:
        out.append(
            f"metrics: unsupported architecture {platform.machine()!r}; the exporters are available for "
            f"{', '.join(Arch)} only"
        )
    if site.metrics.mode == MetricsMode.EXTERNAL:
        for field in ("prometheus_url", "grafana_url"):
            if not getattr(site.metrics, field):
                out.append(f"metrics: mode = external needs [metrics] {field} in the site file")
    if _is_local(site) and site.metrics.grafana_port in set(Port):
        out.append(f"metrics: grafana_port {site.metrics.grafana_port} is used by the metrics stack itself; pick another")
    if site.metrics.retention_days < 1:
        out.append("metrics: [metrics] retention_days must be at least 1")
    return out


def packages(site: Site) -> list[str]:
    # python3-requests and python3-blessed: the `jobstats` command's requirements (`jobstats_command.md:14`).
    pkgs = ["ca-certificates", "curl", "python3-requests", "python3-blessed"]
    if _is_local(site):
        pkgs += ["prometheus", "grafana"]
    if _has_gpus(site) and _arch() == Arch.AARCH64:
        pkgs += ["golang-go", "gcc"]  # builds the GPU exporter (cgo, NVML bindings)
    return pkgs


def apt_flags(site: Site) -> list[str]:
    # The Ubuntu `prometheus` package Recommends prometheus-node-exporter, which would start on port 9100 (this
    # component runs its own node exporter elsewhere) and can fail where something already listens there.
    return ["--no-install-recommends"]


def apt_sources(site: Site) -> dict[str, render.FileSpec]:
    # Grafana's own apt repository, key and one-line source exactly as its install guide gives them (key saved as
    # /etc/apt/keyrings/grafana.asc, `deb [signed-by=...] https://apt.grafana.com stable main`). The key is
    # https://apt.grafana.com/gpg-full.key; its current signing key, B53A E77B ADB6 30A6 8304 6005 963F A277 1045 8545,
    # verifies the repository's InRelease.
    if not _is_local(site):
        return {}
    return {
        str(GRAFANA_KEY): _spec((TEMPLATES / "grafana.asc").read_text()),
        "/etc/apt/sources.list.d/grafana.list": _spec(f"deb [signed-by={GRAFANA_KEY}] https://apt.grafana.com stable main\n"),
    }


def slurm_conf_lines(site: Site) -> str:
    if not enabled(site):
        return ""
    lines = ["# Jobstats: job summaries at job end (docs/setup/summaries.md:9)."]
    if _has_gpus(site):
        lines = [
            "# Jobstats: which job holds which GPU (docs/setup/gpu_node_scripts.md:33-34).",
            f"Prolog={PROLOG_DIR}/*.sh",
            f"Epilog={EPILOG_DIR}/*.sh",
            *lines,
        ]
    return "\n".join([*lines, f"EpilogSlurmctld={SLURMCTLD_EPILOG}"]) + "\n"


def _targets(site: Site) -> str:
    gpu = [f"{n.name}:{Port.GPU}" for n in site.nodes if n.gpus]
    groups = [
        {
            # Jobstats selects every series by `cluster=` (jobstats.py), which must equal slurm's ClusterName.
            "labels": {"cluster": site.cluster_name},
            "targets": [f"{n.name}:{port}" for n in site.nodes for port in (Port.NODE, Port.CGROUP)] + gpu,
        }
    ]
    return json.dumps(groups, indent=2) + "\n"


def _anonymous_lines(site: Site) -> str:
    # Open OnDemand's Active Jobs embeds Grafana panels, which needs anonymous read access and embedding allowed
    # (OOD docs, customizations.rst "Grafana support": [auth.anonymous] enabled, org_role = Viewer;
    # [security] allow_embedding = true). Only when the portal is on; otherwise Grafana asks for the admin login.
    if not _portal_on(site):
        return ""
    settings = ("GF_AUTH_ANONYMOUS_ENABLED=true", "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer", "GF_SECURITY_ALLOW_EMBEDDING=true")
    return "".join(f"Environment={s}\n" for s in settings)


def files(site: Site) -> dict[str, render.FileSpec]:
    if not enabled(site):
        return {}
    textfile_dir = site.state_dir / "metrics"
    unit_values = {
        "bin_dir": BIN_DIR,
        "textfile_dir": textfile_dir,
        "node_port": int(Port.NODE),
        "cgroup_port": int(Port.CGROUP),
        "gpu_port": int(Port.GPU),
    }
    out = {
        str(TARGETS_FILE): _spec(_targets(site)),
        str(JOBSTATS_DIR / "config.py"): _spec(
            _template(
                "jobstats_config.py.in",
                site_name=site.name,
                prometheus_server=_prometheus_server(site),
                retention_days=site.metrics.retention_days,
                scrape_interval=SCRAPE_INTERVAL_S,
            )
        ),
        "/etc/systemd/system/gpujobs-node-exporter.service": _spec(_template("gpujobs-node-exporter.service", **unit_values)),
        "/etc/systemd/system/gpujobs-cgroup-exporter.service": _spec(
            _template("gpujobs-cgroup-exporter.service", **unit_values)
        ),
        str(BIN_DIR / "run-cgroup-exporter"): _spec((TEMPLATES / "run-cgroup-exporter.sh").read_text(), 0o755),
        str(SLURMCTLD_EPILOG): _spec((TEMPLATES / "slurmctld-epilog.sh").read_text(), 0o755),
    }
    if _has_gpus(site):
        out[str(PROLOG_DIR / "gpustats_helper.sh")] = _spec((TEMPLATES / "gpustat-prolog.sh").read_text(), 0o755)
        out[str(EPILOG_DIR / "gpustats_helper.sh")] = _spec((TEMPLATES / "gpustat-epilog.sh").read_text(), 0o755)
        out["/etc/systemd/system/gpujobs-gpu-exporter.service"] = _spec(
            _template("gpujobs-gpu-exporter.service", **unit_values)
        )
    if _is_local(site):
        out |= _local_files(site)
    return out


def _local_files(site: Site) -> dict[str, render.FileSpec]:
    own = _template(
        "gpujobs-job-memory.dashboard.json",
        dashboard_uid=OWN_DASHBOARD["uid"],
        dashboard_title=OWN_DASHBOARD["title"],
        datasource_name=DATASOURCE_NAME,
    )
    return {
        str(PROMETHEUS_CONFIG): _spec(
            _template("prometheus.yml", targets_file=TARGETS_FILE, scrape_interval=SCRAPE_INTERVAL_S)
        ),
        "/etc/systemd/system/prometheus.service.d/10-gpujobs.conf": _spec(
            _template(
                "prometheus.service.dropin.conf",
                config_file=PROMETHEUS_CONFIG,
                retention_days=site.metrics.retention_days,
                prometheus_port=int(Port.PROMETHEUS),
            )
        ),
        "/etc/systemd/system/grafana-server.service.d/10-gpujobs.conf": _spec(
            _template(
                "grafana.service.dropin.conf",
                grafana_port=site.metrics.grafana_port,
                anonymous_lines=_anonymous_lines(site),
                admin_env_file=GRAFANA_ENV_FILE,
            )
        ),
        "/etc/grafana/provisioning/datasources/gpujobs.yaml": _spec(
            _template(
                "grafana-datasource.yaml",
                datasource_name=DATASOURCE_NAME,
                prometheus_url=_prometheus_server(site),
                scrape_interval=SCRAPE_INTERVAL_S,
            )
        ),
        "/etc/grafana/provisioning/dashboards/gpujobs.yaml": _spec(
            _template("grafana-dashboards.yaml", dashboards_dir=GRAFANA_DASHBOARDS)
        ),
        str(GRAFANA_DASHBOARDS / "gpujobs-job-memory.json"): _spec(own),
    }


def _cached(download: Download) -> PurePosixPath:
    return CACHE_DIR / download.filename


def _fetch(download: Download) -> list[str]:
    """Fetch into the cache and check the pinned sha256; a cached file that already checks out is kept."""
    script = (
        'if [ -f "$1" ] && echo "$3  $1" | sha256sum -c --status -; then exit 0; fi; '
        'curl -fsSL -o "$1.part" "$2" && echo "$3  $1.part" | sha256sum -c - && mv "$1.part" "$1"'
    )
    return ["sh", "-ec", script, "fetch", str(_cached(download)), download.url, download.sha256]


def _install(src: PurePosixPath | str, dst: PurePosixPath | str, mode: str) -> list[str]:
    return ["install", "-D", "-m", mode, str(src), str(dst)]


def _extract(download: Download, *members: str, into: PurePosixPath, strip: int = 0) -> list[str]:
    cmd = ["tar", "-xzf", str(_cached(download)), "--no-same-owner", "-C", str(into)]
    return [*cmd, f"--strip-components={strip}", *members] if strip else [*cmd, *members]


def _exporter_steps(site: Site, arch: Arch) -> list[list[str]]:
    stage = CACHE_DIR / "stage"
    node = _BY_NAME[f"node-exporter-{arch}"]
    cgroup = _BY_NAME["cgroup-exporter"]
    steps = [
        ["rm", "-rf", str(stage)],
        ["install", "-d", "-m", "0755", str(stage)],
        _extract(node, f"node_exporter-{_NODE_EXPORTER_VERSION}.linux-{RELEASE_ARCH[arch]}/node_exporter", into=stage, strip=1),
        _install(stage / "node_exporter", BIN_DIR / "node_exporter", "0755"),
        _extract(cgroup, f"cgroup_exporter-{RELEASE_ARCH[arch]}", into=stage),
        _install(stage / f"cgroup_exporter-{RELEASE_ARCH[arch]}", BIN_DIR / "cgroup_exporter", "0755"),
    ]
    if not _has_gpus(site):
        return steps
    if arch == Arch.X86_64:
        gpu = _BY_NAME["gpu-exporter-x86_64"]
        return [
            *steps,
            _extract(gpu, f"nvml_exporter-{_GPU_EXPORTER_VERSION}-x86_64", into=stage),
            _install(stage / f"nvml_exporter-{_GPU_EXPORTER_VERSION}-x86_64", BIN_DIR / "nvml_exporter", "0755"),
        ]
    source = _BY_NAME["gpu-exporter-source"]
    build = CACHE_DIR / "gpu-exporter-build"
    # GOTOOLCHAIN=local: go.mod names a newer toolchain than Ubuntu's golang-go; the build works with the packaged one
    # (go 1.22.2 on Ubuntu 24.04, aarch64), and local keeps go from downloading a toolchain at install.
    script = (
        'cd "$1/$2" && GOTOOLCHAIN=local GOPATH="$3/gopath" GOCACHE="$3/gocache" GOFLAGS=-modcacherw '
        'go build -trimpath -o "$4" .'
    )
    return [
        *steps,
        ["rm", "-rf", str(build)],
        ["install", "-d", "-m", "0755", str(build)],
        _extract(source, into=build),
        ["sh", "-ec", script, "build", str(build), source.filename.removesuffix(".tar.gz"), str(CACHE_DIR), str(stage / "nvml_exporter")],
        _install(stage / "nvml_exporter", BIN_DIR / "nvml_exporter", "0755"),
    ]


def _jobstats_steps(site: Site) -> list[list[str]]:
    steps = [
        _install(_cached(_BY_NAME["jobstats"]), JOBSTATS_DIR / "jobstats", "0755"),
        _install(_cached(_BY_NAME["jobstats.py"]), JOBSTATS_DIR / "jobstats.py", "0644"),
        _install(_cached(_BY_NAME["output_formatters.py"]), JOBSTATS_DIR / "output_formatters.py", "0644"),
        ["ln", "-sf", str(JOBSTATS_DIR / "jobstats"), str(JOBSTATS_LINK)],
    ]
    return steps


def _grafana_steps() -> list[list[str]]:
    dashboard = _cached(_BY_NAME["Single_Job_Stats.json"])
    # The exported dashboard names its data source `${DS_DEFAULT_LOCAL PROMETHEUS}`; provisioning needs the real name.
    rewrite = (
        'sed "s/\\${DS_DEFAULT_LOCAL PROMETHEUS}/$2/g" "$1" '
        '| install -D -m 0644 /dev/stdin "$3"'
    )
    # Grafana applies admin_password on its first start only; generate it once, readable by root alone.
    password = (
        'umask 077; test -s "$1" || printf "GF_SECURITY_ADMIN_PASSWORD=%s\\n" '
        '"$(head -c 24 /dev/urandom | base64 | tr -d "/+=\\n")" > "$1"'
    )
    return [
        ["sh", "-ec", rewrite, "rewrite", str(dashboard), DATASOURCE_NAME, str(GRAFANA_DASHBOARDS / JOBSTATS_DASHBOARD_FILE)],
        ["sh", "-ec", password, "password", str(GRAFANA_ENV_FILE)],
    ]


def _ood_helper_steps() -> list[list[str]]:
    archive = _BY_NAME["ood-jobstats-helper"]
    return [
        ["rm", "-rf", str(OOD_HELPER_DIR)],
        ["install", "-d", "-m", "0755", str(OOD_HELPER_DIR)],
        _extract(archive, f"jobstats-{_JOBSTATS_COMMIT}/ood-jobstats-helper", into=OOD_HELPER_DIR, strip=2),
    ]


def post_install(site: Site) -> list[list[str]]:
    arch = _arch()
    if not enabled(site) or arch is None:
        return []
    wanted = ["jobstats", "jobstats.py", "output_formatters.py", f"node-exporter-{arch}", "cgroup-exporter"]
    if _has_gpus(site):
        wanted.append("gpu-exporter-x86_64" if arch == Arch.X86_64 else "gpu-exporter-source")
    if _is_local(site):
        wanted.append("Single_Job_Stats.json")
    if _portal_on(site):
        wanted.append("ood-jobstats-helper")
    units = [s for s in SERVICES if s != "gpujobs-gpu-exporter" or _has_gpus(site)]
    steps = [
        ["install", "-d", "-m", "0755", str(CACHE_DIR), str(BIN_DIR)],
        *(_fetch(_BY_NAME[name]) for name in wanted),
        *_exporter_steps(site, arch),
        *_jobstats_steps(site),
    ]
    if _is_local(site):
        steps += _grafana_steps()
        units += ["prometheus", "grafana-server"]
    if _portal_on(site):
        steps += _ood_helper_steps()
    return [*steps, ["systemctl", "daemon-reload"], ["systemctl", "enable", *units], ["systemctl", "restart", *units]]
