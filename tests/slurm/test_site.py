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
SHARING_NODE = MINIMAL.replace("cpus = 2", "cpus = 8") + "cores_per_gpu = 6\n" + GPU  # a node holding 6 cores for its one card


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "site.toml"
    path.write_text(text)
    return path


LOCAL_SITE_FILES = sorted((ROOT / "machine" / "slurm").glob("site*.toml"))  # this machine's own, untracked


@pytest.mark.skipif(not LOCAL_SITE_FILES, reason="no machine-local site files in this checkout")
@pytest.mark.parametrize("path", LOCAL_SITE_FILES, ids=lambda p: p.name)
def test_each_local_site_file_still_loads(path: Path) -> None:
    """The machine's real site files are git-ignored now; a typo in one must still fail here, not at install."""
    assert gpujobs_site.load(path).nodes


def test_fixture_site_file_loads() -> None:
    site = gpujobs_site.load(ROOT / "tests/slurm/fixtures/site.toml")
    assert (site.name, site.cluster_name, site.account) == ("gpu-node-1", "spark", "local") and site.unified_memory and site.hostguard and site.dgx_gate
    gpu = site.nodes[0].gpus[0]
    assert (site.nodes[0].real_memory_mb, gpu.device, gpu.type) == (100000, "/dev/nvidia0", "gb10")
    assert set(site.lease.services) == {"service-a", "service-b", "service-c-app"}
    assert site.service == gpujobs_site.Service(4, "gpujobs-service")
    assert site.remotes == {"turing": gpujobs_site.Remote("site-b", "gpujobs-sync", data_root="/data/site/user1/datasets")}
    fedora = site.part_time_nodes[0]  # what a job may ask of it: 16 abstract cores less keep_cpus "8-15"; 12 GB of 31.5
    assert (fedora.cores_lent, fedora.jobs_mb, fedora.default_jobs) == (8, 11444, True)
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
    # a full node may hold a measured reserve too (D44): 0.5 decimal GB is 477 of slurm's 2^20-byte MB
    assert gpujobs_site.load(write(tmp_path, MINIMAL + "jobs_memory_gb = 0.5\n")).nodes[0].mem_spec_limit_mb == 1000 - 477
    # card sharing (D28): absent = the card is one token; "cores" = one shard per core held for the card; a number as given
    def shared(extra: str, head: str = "") -> gpujobs_site.Site:
        return gpujobs_site.load(write(tmp_path, head + SHARING_NODE + extra))

    cores = shared("").nodes[0].cores_per_gpu
    assert (shared("").sharing, shared("").nodes[0].shards, site.sharing_memory_fraction) == (False, 0, None)
    assert (shared("shards = 'cores'\n").nodes[0].shards, shared("shards = 'cores'\n").sharing) == (cores, True)
    assert (shared("shards = 4\n").nodes[0].gpus[0].shards, shared("shards = 0\n").sharing) == (4, False)
    assert shared("", "sharing_memory_fraction = 0.5\n").sharing_memory_fraction == pytest.approx(0.5)
    # cores from history (D35): recommend unless the site says otherwise
    assert (site.cpus_from_history, shared("", "cpus_from_history = 'enforce'\n").cpus_from_history) == (
        gpujobs_site.CoresFromHistory.RECOMMEND, gpujobs_site.CoresFromHistory.ENFORCE)


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
        (MINIMAL + "jobs_memory_gb = 9\n", "jobs_memory_gb = 9"),  # a full node's reserve is checked as a part-time one's
        (MINIMAL + "close_below_gb = 4\npush_back_below_gb = 5\n", "push_back_below_gb must not exceed"),
        (PART_TIME.replace("[[nodes]]", 'name = "x"\n[[nodes]]', 1) + "address = 'a'\n", "every node is part_time"),
        (MINIMAL + "[health_check]\ninterval_s = 0\n", "health_check.interval_s = 0"),
        (MINIMAL + "[maintenance]\nstatus_file = 's.json'\nrequest_dir = '/r'\n", "maintenance.status_file = 's.json'"),
        (MINIMAL + "[maintenance]\nstatus_file = '/s.json'\n", "[maintenance] is missing request_dir"),
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
        (MINIMAL + PART_TIME + "address = 'a'\ndefault_jobs = 'no'\n", "default_jobs must be true or false"),
        (MINIMAL + "cores_per_gpu = 1\n", "cores_per_gpu must be"),  # no GPUs: nothing to hold cores for
        (MINIMAL + "cores_per_gpu = 1.5\n" + GPU, "cores_per_gpu must be"),  # would render MaxCPUsPerNode=0.5
        (MINIMAL + "cores_per_gpu = 2\n" + GPU, "must leave CPU-only jobs a core"),  # 2 of 2: CPU jobs could never run
        ("state_dir = 5\n" + MINIMAL, "has a value of the wrong kind"),
        (MINIMAL + "cores_per_gpu = 1\n" + GPU + "[service]\ncores = 2\n", "service.cores = 2"),  # beyond the CPU pool
        (SHARING_NODE + "shards = 'all'\n", "shards = 'all'; expected \"cores\""),
        (SHARING_NODE + "shards = 1025\n", "shards = 1025"),  # slurm's limit on a shard Count
        (SHARING_NODE + "shards = -1\n", "shards = -1"),
        (SHARING_NODE + "shards = true\n", "shards = True"),  # a flag is not a count
        (MINIMAL + GPU + "shards = 'cores'\n", "needs the node's cores_per_gpu"),  # nothing says how many cores the card holds
        ("sharing_memory_fraction = 1.5\n" + MINIMAL, "sharing_memory_fraction = 1.5"),
        ("sharing_memory_fraction = 0\n" + MINIMAL, "sharing_memory_fraction = 0"),
        ("sharing_memory_fraction = true\n" + MINIMAL, "sharing_memory_fraction = True"),
        ("cpus_from_history = 'always'\n" + MINIMAL, "cpus_from_history"),
        (MINIMAL + "[service]\ncores = 0\n", "service.cores = 0"),
        (MINIMAL + "[remote.t]\nhost = 'h'\n", "[remote.t] is missing root"),
        (MINIMAL + "[remote.t]\nhost = 'h'\nroot = '../x'\n", "no '..'"),
        (MINIMAL + "[remote.t]\nhost = ' '\nroot = 'r'\n", "remote.t.host = ' '"),
        (MINIMAL + "[remote.t]\nhost = 'h'\nroot = 'r'\ndata_root = 'd'\n", "expected an absolute path"),
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
