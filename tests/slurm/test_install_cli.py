import getpass
import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpujobs_install", ROOT / "slurm" / "install" / "install.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
components = cli.components

USER = getpass.getuser()
SITE = f"""
name = "t"
[[nodes]]
name = "n1"
cpus = 4
real_memory_mb = 1000
[lease]
lease_dir = "/x"
keeper_user = "{USER}"
python = "{sys.executable}"
"""


class Fake:
    """Records every argv; `queue` is squeue's output and `client_ok` the lease-client import's success."""

    def __init__(self, queue="", client_ok=True):
        self.calls, self.queue, self.client_ok = [], queue, client_ok

    def __call__(self, argv):
        self.calls.append(argv)
        rc = 0
        if argv[:1] == ["squeue"]:
            return subprocess.CompletedProcess(argv, 0, self.queue, "")
        if argv[0] == sys.executable:
            rc = 0 if self.client_ok else 1
        return subprocess.CompletedProcess(argv, rc, "", "")


@pytest.fixture
def env(tmp_path):
    site = tmp_path / "site.toml"
    site.write_text(SITE)
    fake = Fake()
    components.set_runner(fake)
    yield tmp_path, site, fake
    components.set_runner(None)


def run(site, root, *args, chown=None):
    return cli.main([*args[:1], "--site", str(site), "--root", str(root), *args[1:]], chown=chown)


def test_apply_then_check_against_a_temp_root(env):
    tmp, site, fake = env
    root = tmp / "root"
    owned = []
    assert run(site, root, "check") == 1  # nothing installed yet
    assert run(site, root, "apply", chown=lambda p, o: owned.append((p, o))) == 0
    assert run(site, root, "check") == 0
    assert stat.S_IMODE((root / "etc/slurm/slurmdbd.conf").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "usr/local/bin/gpujobs-submit").stat().st_mode) == 0o755
    assert ((root / "etc/slurm/slurmdbd.conf"), "slurm") in owned
    assert (root / "etc/gpujobs/site.toml", "root") in owned
    (root / "etc/slurm/slurmdbd.conf").chmod(0o644)  # same content, but readable by everyone
    assert run(site, root, "check") == 1
    (root / "etc/slurm/slurmdbd.conf").chmod(0o600)
    (root / "etc/slurm/gres.conf").write_text("edited\n")
    assert run(site, root, "check") == 1
    # the argv order: refusals' probes, packages, then post-install
    names = [a[0] if a[0] != "systemctl" else " ".join(a[:3]) for a in fake.calls]
    assert names.index("apt-get") < names.index("groupadd")
    assert ["apt-get", "install", "-y", "slurm-wlm", "slurmdbd", "mariadb-server", "munge"] in fake.calls
    assert ["systemctl", "restart", "slurmctld", "slurmd"] in fake.calls
    assert ["systemctl", "enable", "--now", "gpujobs-agent.service"] in fake.calls
    assert ["systemctl", "--user", f"--machine={USER}@.host", "restart", "slurm-gpu-lease-keeper.service"] in fake.calls
    assert any(a[:3] == ["install", "-d", "-m"] and a[-1].endswith("/requests") and a[3] == "1777" for a in fake.calls)


def test_apply_refuses_with_a_busy_queue_or_without_the_lease_client(env):
    tmp, site, fake = env
    fake.queue = "123 gpu job user R\n"
    assert run(site, tmp / "root", "apply") == 1
    assert not (tmp / "root").exists()
    assert not any(a[0] in ("apt-get", "systemctl") for a in fake.calls)
    fake.queue, fake.client_ok = "", False
    assert run(site, tmp / "root", "apply") == 1
    assert not (tmp / "root").exists()


def test_admin_argv(env):
    tmp, site, fake = env
    assert run(site, tmp, "admin", "add", "alice") == 0
    assert run(site, tmp, "admin", "remove", "alice") == 0
    assert fake.calls == [
        ["usermod", "-aG", "gpujobs-admin", "alice"],
        ["sacctmgr", "-i", "modify", "user", "name=alice", "set", "adminlevel=Operator"],
        ["gpasswd", "-d", "alice", "gpujobs-admin"],
        ["sacctmgr", "-i", "modify", "user", "name=alice", "set", "adminlevel=None"],
    ]


def test_migrate_keeps_owner_and_mode_and_never_overwrites(env):
    tmp, site, _ = env
    old, root = tmp / "old", tmp / "root"
    (old / "jobs").mkdir(parents=True)
    (old / "jobs/1.json").write_text("new")
    (old / "jobs/1.json").chmod(0o640)
    (old / "events").mkdir()
    (old / "events/e.json").write_text("e")
    (old / "events/e.json").chmod(0o640)
    existing = root / "var/lib/gpujobs/jobs/1.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep")
    assert run(site, root, "migrate-state", "--from", str(old)) == 0
    assert existing.read_text() == "keep"
    moved = root / "var/lib/gpujobs/events/e.json"
    assert moved.read_text() == "e"
    assert stat.S_IMODE(moved.stat().st_mode) == stat.S_IMODE((old / "events/e.json").stat().st_mode)
    assert moved.stat().st_uid == os.getuid()
