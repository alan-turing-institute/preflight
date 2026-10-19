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
    assert set(site.lease.services) == {"service-a", "service-b"}
    assert site.lease.lease_dir == Path("/home/user1/.local/state/gpu-lease")
    assert (site.metrics.mode, site.metrics.grafana_port) == (gpujobs_site.MetricsMode.LOCAL, 3001)
    assert site.metrics.allow_from == ("100.64.0.0/10", "fd7a:115c:a1e0::/48", "192.0.2.8/32")  # Tailscale + NAS Caddy
    assert site.metrics.grafana_url == "https://jobs-grafana.home.example.com"
    assert (site.portal.mode, site.portal.auth) == (gpujobs_site.PortalMode.INSTALL, gpujobs_site.AuthMode.DEX_LOCAL)
    assert (site.portal.server_name, site.portal.session_days) == ("jobs.home.example.com", 365)  # a *.home name, like every service
    assert site.portal.allow_from == ("192.0.2.8/32", "100.64.0.3/32")  # only the NAS Caddy
    assert site.portal.tls_cert is None and site.notify_command == ()


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


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("hostgaurd = true\n" + MINIMAL, "unknown key(s) in [top level]: hostgaurd"),
        (MINIMAL + "[metrics]\nmodee = 'off'\n", "unknown key(s) in [metrics]"),
        (MINIMAL + GPU + "extra = 1\n", "unknown key(s) in [nodes[0].gpus[0]]"),
        (MINIMAL + "[metrics]\nmode = 'sideways'\n", "metrics.mode"),
        (MINIMAL + "[metrics]\nallow_from = ['tailnet']\n", "metrics.allow_from entry 'tailnet' is not an IP network"),
        (MINIMAL + "[metrics]\nallow_from = '100.64.0.0/10'\n", "metrics.allow_from must be a list"),
        (MINIMAL + "[portal]\nsession_days = -1\n", "portal.session_days = -1"),
        (MINIMAL + "[portal]\nallow_from = ['ts']\n", "portal.allow_from entry 'ts'"),
        ('name = "x"\n', "missing nodes"),
    ],
)
def test_bad_site_files_are_refused(tmp_path: Path, text: str, fragment: str) -> None:
    with pytest.raises(gpujobs_site.SiteError, match=re.escape(fragment)):
        gpujobs_site.load(write(tmp_path, text))


def test_missing_file_names_the_detect_command(tmp_path: Path) -> None:
    with pytest.raises(gpujobs_site.SiteError, match="slurm/install/install.py detect"):
        gpujobs_site.load(tmp_path / "absent.toml")


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
