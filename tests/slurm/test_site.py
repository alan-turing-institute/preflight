"""site_config.py: the shipped site files load, defaults hold, typos and a missing file are refused."""

from __future__ import annotations

import grp
import pwd
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import site_config as gpujobs_site

ROOT = Path(__file__).resolve().parents[2]


MINIMAL = """
name = "x"
[[nodes]]
name = "n"
cpus = 2
real_memory_mb = 1000
"""
SHA = "ab" * 32
GPU = "[[nodes.gpus]]\nindex = 0\nuuid = 'u'\ndevice = 'd'\ntype = 't'\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "site.toml"
    path.write_text(text)
    return path


def test_this_box_site_file_loads() -> None:
    site = gpujobs_site.load(ROOT / "machine/slurm/site.toml")
    assert (site.name, site.cluster_name, site.account) == ("gpu-node-1", "spark", "local") and site.unified_memory and site.hostguard and site.dgx_gate
    gpu = site.nodes[0].gpus[0]
    assert (site.nodes[0].real_memory_mb, gpu.device, gpu.type) == (100000, "/dev/nvidia0", "gb10")
    assert set(site.lease.services) == {"service-a", "service-b", "service-c-app"}
    assert site.service == gpujobs_site.Service(4, "gpujobs-service")
    assert site.lease.lease_dir == Path("/home/user1/.local/state/gpu-lease")
    assert (site.metrics.mode, site.metrics.grafana_port) == (gpujobs_site.MetricsMode.LOCAL, 3001)
    assert site.metrics.allow_from == ("100.64.0.0/10", "fd7a:115c:a1e0::/48", "192.0.2.8/32")  # Tailscale + NAS Caddy
    assert site.metrics.grafana_url == "https://jobs-grafana.home.example.com"
    assert (site.portal.mode, site.portal.auth) == (gpujobs_site.PortalMode.INSTALL, gpujobs_site.AuthMode.PROXY)
    assert site.portal.server_name == "jobs.home.example.com"  # a *.home name, like every service
    assert site.portal.auth_settings == {"user": "user1", "key_file": "/etc/gpujobs/portal/proxy_key"}  # no login
    assert site.portal.allow_from == ("192.0.2.8/32",)  # only the NAS Caddy
    assert site.portal.tls_cert is None and site.notify_command == ()


PART_TIME = "[[nodes]]\nname = 'p'\ncpus = 4\nreal_memory_mb = 8000\npart_time = true\n"


def test_example_loads_and_minimal_file_gets_every_default(tmp_path: Path) -> None:
    example = gpujobs_site.load(ROOT / "slurm/site.example.toml")
    assert example.nodes[0].gpus[0].index == 0 and example.lease is not None
    site = gpujobs_site.load(write(tmp_path, MINIMAL))
    assert (site.state_dir, site.install_dir, site.admin_group) == (
        Path("/var/lib/gpujobs"),
        Path("/opt/gpujobs"),
        "gpujobs-admin",
    )
    assert site.lease is None and not site.hostguard and not site.dgx_gate
    assert site.metrics.mode is gpujobs_site.MetricsMode.OFF and site.portal.mode is gpujobs_site.PortalMode.OFF
    assert site.metrics.allow_from == ()
    assert site.nodes[0].gpus == ()
    assert site.cluster_name == site.name and site.account == "gpujobs"  # defaults
    assert site.constrain_devices  # default: jobs are confined to their GPUs
    assert (site.shared_state, site.node_python) == (False, "/usr/bin/python3")  # part-time job state is copied back
    shared = gpujobs_site.load(write(tmp_path, 'shared_state = true\nnode_python = "/opt/py/bin/python"\n' + MINIMAL))
    assert (shared.shared_state, shared.node_python) == (True, "/opt/py/bin/python")
    assert site.service is None  # no [service] table: no service partition
    service = gpujobs_site.load(write(tmp_path, MINIMAL + "[service]\ncores = 2\n")).service
    assert service == gpujobs_site.Service(2, "gpujobs-service")  # no cores held for GPUs: up to the node's cores
    assert site.slurm_build is None and gpujobs_site.load(write(tmp_path, MINIMAL + "[slurm]\n")).slurm_build is None
    built = gpujobs_site.load(write(tmp_path, MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\n"))
    assert built.slurm_build == ("26.05.5", SHA) and built.slurm_steps == ()
    stepped = gpujobs_site.load(write(tmp_path, MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['25.05.9:{SHA}']\n"))
    assert stepped.slurm_steps == (("25.05.9", SHA),)
    # part-time nodes: their own limits, and the health check only when its table is there
    mixed = gpujobs_site.load(write(tmp_path, MINIMAL + PART_TIME + "address = 'a'\njobs_memory_gb = 5\nkeep_cpus = '0,2-3'\n"
                                   "[health_check]\n"))
    full, part = mixed.nodes
    assert (mixed.full_nodes, mixed.part_time_nodes) == ((full,), (part,)) and mixed.health_check == gpujobs_site.HealthCheck(30)
    assert (full.close_below_mb, full.push_back_below_mb, part.close_below_mb, part.push_back_below_mb) == (16000, 10000, 6000, 2000)
    assert (part.mem_spec_limit_mb, full.mem_spec_limit_mb, part.mains_only) == (8000 - 4768, None, None)  # 5 decimal GB is 4768 of slurm's 2^20-byte MB
    assert site.health_check is None


@pytest.mark.parametrize(
    "text, fragment",
    [
        (MINIMAL + PART_TIME, "a part_time node needs address"),
        (MINIMAL + PART_TIME + "address = 'a'\n" + GPU, "a part_time node cannot have GPUs"),
        (MINIMAL + PART_TIME + "address = 'a'\njobs_memory_gb = 9\n", "jobs_memory_gb = 9"),
        (MINIMAL + PART_TIME + "address = 'a'\njobs_memory_gb = nan\n", "jobs_memory_gb = nan"),
        (MINIMAL + PART_TIME + "address = 'a'\njobs_memory_gb = inf\n", "jobs_memory_gb = inf"),
        (MINIMAL + "close_below_gb = nan\n", "close_below_gb = nan"),
        (MINIMAL + "close_below_gb = inf\n", "close_below_gb = inf"),
        (MINIMAL + PART_TIME + "address = 'a'\nkeep_cpus = '0-'\n", "keep_cpus = '0-'"),
        (MINIMAL + "address = 'a'\n", "apply only with part_time = true"),
        (MINIMAL + "close_below_gb = 4\npush_back_below_gb = 5\n", "push_back_below_gb must not exceed"),
        (PART_TIME.replace("[[nodes]]", 'name = "x"\n[[nodes]]', 1) + "address = 'a'\n", "every node is part_time"),
        (MINIMAL + "[health_check]\ninterval_s = 0\n", "health_check.interval_s = 0"),
        (MINIMAL + "[health_check]\ninterval_s = 9\n", "health_check.interval_s = 9"),
        (MINIMAL + "[health_check]\nintervals = 5\n", "unknown key(s) in [health_check]"),
        ("shared_state = 'yes'\n" + MINIMAL, "shared_state must be true or false"),
        ("node_python = 'python3'\n" + MINIMAL, "node_python = 'python3'; expected an absolute path"),
        ("hostgaurd = true\n" + MINIMAL, "unknown key(s) in [top level]: hostgaurd"),
        (MINIMAL + "[metrics]\nmodee = 'off'\n", "unknown key(s) in [metrics]"),
        (MINIMAL + GPU + "extra = 1\n", "unknown key(s) in [nodes[0].gpus[0]]"),
        (MINIMAL + "[metrics]\nmode = 'sideways'\n", "metrics.mode"),
        (MINIMAL + "[metrics]\nallow_from = ['tailnet']\n", "metrics.allow_from entry 'tailnet' is not an IP network"),
        (MINIMAL + "[metrics]\nallow_from = '100.64.0.0/10'\n", "metrics.allow_from must be a list"),
        (MINIMAL + "[portal]\nsession_days = -1\n", "portal.session_days = -1"),
        (MINIMAL + "[portal]\nallow_from = ['ts']\n", "portal.allow_from entry 'ts'"),
        ('name = "x"\n', "missing nodes"),
        (MINIMAL + "cores_per_gpu = 1\n", "cores_per_gpu must be"),  # no GPUs: nothing to hold cores for
        (MINIMAL + "cores_per_gpu = 1.5\n" + GPU, "cores_per_gpu must be"),  # would render MaxCPUsPerNode=0.5
        (MINIMAL + "cores_per_gpu = 2\n" + GPU, "must leave CPU-only jobs a core"),  # 2 of 2: CPU jobs could never run
        ("state_dir = 5\n" + MINIMAL, "has a value of the wrong kind"),
        (MINIMAL + "cores_per_gpu = 1\n" + GPU + "[service]\ncores = 2\n", "service.cores = 2"),  # beyond the CPU pool
        (MINIMAL + "[service]\ncores = 0\n", "service.cores = 0"),
        (MINIMAL + "[[nodes]]\nname = 'm'\ncpus = 2\nreal_memory_mb = 1000\n[service]\ncores = 1\n", "a site of one node"),
        (MINIMAL + "[service]\ncores = 1\ngroups = 'g'\n", "unknown key(s) in [service]: groups"),
        (MINIMAL + "[slurm]\nversion = '26.05.5'\n", "needs both version and sha256"),
        (MINIMAL + f"[slurm]\nsha256 = '{SHA}'\n", "needs both version and sha256"),
        (MINIMAL + f"[slurm]\nversion = '26.05'\nsha256 = '{SHA}'\n", "slurm.version = '26.05'"),
        (MINIMAL + "[slurm]\nversion = '26.05.5'\nsha256 = 'abc'\n", "slurm.sha256 must be 64"),
        (MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['25.05.9']\n", "expected \"<version>:<sha256>\""),
        (MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['25.5:{SHA}']\n", "slurm.steps entry version = '25.5'"),
        (MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['25.05.9:abc']\n", "slurm.steps entry sha256 must be 64"),
        (MINIMAL + f"[slurm]\nsteps = ['25.05.9:{SHA}']\n", "steps needs version and sha256"),
        # steps run oldest first and each is older than version: out of order, repeated or not below it are refused
        (MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['25.05.9:{SHA}', '24.11.1:{SHA}']\n", "ascending order"),
        (MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['25.05.9:{SHA}', '25.05.9:{SHA}']\n", "ascending order"),
        (MINIMAL + f"[slurm]\nversion = '26.05.5'\nsha256 = '{SHA}'\nsteps = ['26.05.5:{SHA}']\n", "ascending order"),
    ],
)
def test_bad_site_files_are_refused(tmp_path: Path, text: str, fragment: str) -> None:
    with pytest.raises(gpujobs_site.SiteError, match=re.escape(fragment)):
        gpujobs_site.load(write(tmp_path, text))


def test_missing_file_names_the_detect_command(tmp_path: Path) -> None:
    with pytest.raises(gpujobs_site.SiteError, match="slurm/install/install.py detect"):
        gpujobs_site.load(tmp_path / "absent.toml")
    (tmp_path / "latin1.toml").write_bytes(b'name = "caf\xe9"\n')
    for unreadable in (tmp_path, tmp_path / "latin1.toml"):  # a SiteError, which every tool turns into one line
        with pytest.raises(gpujobs_site.SiteError, match="cannot read site file"):
            gpujobs_site.load(unreadable)


def test_is_admin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    site = gpujobs_site.load(write(tmp_path, MINIMAL))
    gids = {"listed": 1001, "primary": 2000, "other": 1002}
    monkeypatch.setattr(grp, "getgrnam", lambda _n: SimpleNamespace(gr_mem=["listed"], gr_gid=2000))
    monkeypatch.setattr(pwd, "getpwnam", lambda u: SimpleNamespace(pw_gid=gids[u]))  # KeyError for unknown
    assert gpujobs_site.is_admin(site, "listed") and gpujobs_site.is_admin(site, "primary")
    assert not gpujobs_site.is_admin(site, "other") and not gpujobs_site.is_admin(site, "ghost")

    def no_group(name: str) -> None:
        raise KeyError(name)

    monkeypatch.setattr(grp, "getgrnam", no_group)
    assert not gpujobs_site.is_admin(site, "listed")
