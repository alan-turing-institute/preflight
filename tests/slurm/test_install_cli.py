import getpass
import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpujobs_install", ROOT / "slurm" / "install" / "install.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
components = cli.components
slurm_component = importlib.import_module("components.slurm")
RENDERED_MARK = cli.render.RENDERED_MARK

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
    """Records every argv; `queue` is squeue's output, `client_ok` the lease-client import's success, `known` the
    `sacctmgr show` queries (after `show`) that find a record, and `policy` what `apt-cache policy` says."""

    def __init__(self, queue="", client_ok=True):
        self.calls, self.queue, self.client_ok, self.known = [], queue, client_ok, set()
        self.squeue_rc, self.ctld_rc = 0, 0  # squeue's exit status; `systemctl is-active slurmctld`'s (0: running)
        self.policy = "slurm-wlm:\n  Installed: (none)\n  Candidate: 23.11.4-1.2ubuntu5\n"
        self.distro_slurm = ""  # what `dpkg-query -W` reports installed, e.g. "ii slurm-wlm\nii slurmd\n"

    def __call__(self, argv):
        self.calls.append(argv)
        rc = 0
        if argv[:1] == ["squeue"]:
            return subprocess.CompletedProcess(argv, self.squeue_rc, self.queue, "")
        if argv[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(argv, self.ctld_rc, "", "")
        if argv[:1] == ["dpkg-query"]:
            return subprocess.CompletedProcess(argv, 1, self.distro_slurm, "")
        if argv[:2] == ["apt-cache", "policy"]:
            return subprocess.CompletedProcess(argv, 0, self.policy, "")
        if argv[:4] == ["sacctmgr", "-n", "-P", "show"]:
            return subprocess.CompletedProcess(argv, 0, "row\n" if " ".join(argv[4:]) in self.known else "", "")
        if argv[0] == sys.executable:
            rc = 0 if self.client_ok else 1
        return subprocess.CompletedProcess(argv, rc, "", "")


@pytest.fixture
def env(tmp_path, monkeypatch):
    site = tmp_path / "site.toml"
    site.write_text(SITE)
    fake = Fake()
    components.set_runner(fake)
    # the keeper user's home, where an old copy of the unit would override the installed one
    monkeypatch.setattr(importlib.import_module("components.lease"), "_home", lambda _user: tmp_path / "home")
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
    assert stat.S_IMODE((root / "opt/gpujobs/slurm/ood_app/bin/python").stat().st_mode) == 0o755  # OOD runs it
    # The installed wrapper runs a job with only what the install put there (this site leaves hostguard off).
    assert not (root / "opt/gpujobs/hostguard").exists()
    job = dict(os.environ, GPUJOBS_SITE=str(site), SLURM_GPU_STATE_DIR=str(tmp / "state"), SLURM_JOB_ID="1")
    wrapper = [sys.executable, str(root / "opt/gpujobs/slurm/job_wrapper.py"), "--", "true"]
    assert subprocess.run(wrapper, env=job, check=False).returncode == 0
    gated = SimpleNamespace(install_dir=Path("/opt/gpujobs"), hostguard=False, dgx_gate=True, health_check=None)
    shipped = components.load(gated, ["tools"])[0].files(gated)
    assert {"/opt/gpujobs/hostguard/hostguard.py", "/opt/gpujobs/dgx/dgx_submit.sh"} <= shipped.keys()
    assert ((root / "etc/slurm/slurmdbd.conf"), "slurm") in owned
    assert (root / "etc/gpujobs/site.toml", "root") in owned
    (root / "etc/slurm/slurmdbd.conf").chmod(0o644)  # same content, but readable by everyone
    assert run(site, root, "check") == 1
    (root / "etc/slurm/slurmdbd.conf").chmod(0o600)
    # the owner is only checkable as root, so the test asks directly: a file no real user owns is reported
    ghost = {"/etc/slurm/gres.conf": cli.render.FileSpec("x", (root / "etc/slurm/gres.conf").stat().st_mode & 0o7777, "no-such-user-gpujobs")}
    assert cli._wrong_mode_or_owner(root, ghost, check_owner=True)
    assert not cli._wrong_mode_or_owner(root, ghost, check_owner=False)
    (root / "etc/slurm/gres.conf").write_text("edited\n")
    assert run(site, root, "check") == 1
    # the argv order: refusals' probes, packages, then post-install
    names = [a[0] if a[0] != "systemctl" else " ".join(a[:3]) for a in fake.calls]
    assert names.index("apt-get") < names.index("groupadd")
    assert ["apt-get", "install", "-y", "slurm-wlm", "slurmdbd", "mariadb-server", "munge"] in fake.calls
    assert ["systemctl", "restart", "slurmctld", "slurmd"] in fake.calls
    # slurm's accounting gets the cluster and the account after slurmdbd is back and before slurmctld starts
    db, dbd, ctld = (fake.calls.index(["systemctl", "restart", *g]) for g in (["mariadb", "munge"], ["slurmdbd"], ["slurmctld", "slurmd"]))
    # a fresh MariaDB has no login for slurmdbd (Turing box, 2026-10-21): it is made between the two restarts
    assert db < fake.calls.index(["mariadb", "-e", slurm_component.DB_LOGIN]) < dbd
    # nor does it have slurmctld's state folder, which slurmctld cannot make itself (Turing box, 2026-10-21)
    state = ["install", "-d", "-m", "0755", "-o", "slurm", "-g", "slurm", slurm_component.STATE_SAVE]
    assert fake.calls.index(state) < ctld
    adds = [a for a in fake.calls[db:ctld] if a[2:3] == ["add"]]
    assert adds == [["sacctmgr", "-i", "add", "cluster", "t"],
                    ["sacctmgr", "-i", "add", "account", "gpujobs", "Cluster=t", "Description=gpujobs", "Organization=gpujobs"],
                    *(["sacctmgr", "-i", "add", "qos", level] for level in ("low", "normal", "high"))]  # D18's levels
    # an upgrade swaps the running agent for the new code, not only enables it
    assert fake.calls.index(["systemctl", "restart", "gpujobs-agent.service"]) > fake.calls.index(["systemctl", "enable", "gpujobs-agent.service"])
    assert ["systemctl", "--user", f"--machine={USER}@.host", "restart", "slurm-gpu-lease-keeper.service"] in fake.calls
    # plan 3.3, every state directory's (mode, owner): the shared ones sticky and open, requests/ the agent's own
    dirs = {a[-1]: (a[3], a[a.index("-o") + 1] if "-o" in a else None) for a in fake.calls if a[:2] == ["install", "-d"]}
    state = "/var/lib/gpujobs"
    sticky = ("jobs", "events", "footprints", "logs", "scripts", "manifests")  # submit.py writes logs, scripts
    assert {d: m for d, m in dirs.items() if d.startswith(state)} == {
        state: ("0755", "root"),
        f"{state}/agent": ("0755", "slurm"),
        f"{state}/metrics": ("0755", "slurm"),  # only the agent writes the per-job textfiles
        f"{state}/keeper": ("0755", USER),  # its heartbeat's home
        f"{state}/requests": ("1777", "slurm"),
        **{f"{state}/{name}": ("1777", None) for name in sticky},
    }
def test_apply_refuses_with_a_busy_queue_or_without_the_lease_client(env, monkeypatch):
    tmp, site, fake = env
    fake.queue = "123 gpu job user R\n"
    assert run(site, tmp / "root", "apply") == 1
    assert not (tmp / "root").exists()
    assert not any(a[0] in ("apt-get", "systemctl") for a in fake.calls)
    # squeue failing: refused while slurmctld runs or has saved jobs; a first apply that stopped before starting
    # slurmctld (Turing box, 2026-10-21) runs again
    fake.queue, fake.squeue_rc = "", 1
    assert run(site, tmp / "root", "apply") == 1 and not (tmp / "root").exists()
    fake.ctld_rc = 3
    saved = tmp / "saved"
    (saved / "var/spool/slurmctld").mkdir(parents=True)
    (saved / "var/spool/slurmctld/job_state").write_text("")
    assert run(site, saved, "apply") == 1 and not (saved / "etc").exists()
    assert run(site, tmp / "fresh", "apply") == 0
    fake.squeue_rc = fake.ctld_rc = 0
    fake.client_ok = False
    assert run(site, tmp / "root", "apply") == 1
    assert not (tmp / "root").exists()
    fake.client_ok = True
    old = tmp / "home/.config/systemd/user/slurm-gpu-lease-keeper.service"
    old.parent.mkdir(parents=True)
    old.symlink_to("/etc/ld.so.preload")  # a user's own unit (or link) would override the one root installs
    assert run(site, tmp / "root", "apply") == 1
    assert not (tmp / "root").exists()
    old.unlink()
    for policy in ("  Candidate: 21.08.5-2ubuntu1\n", "  Candidate: (none)\n"):  # Ubuntu 22.04's slurm; none at all
        fake.policy = policy
        assert run(site, tmp / "root", "apply") == 1 and not (tmp / "root").exists()
    fake.policy = Fake().policy
    # untested platforms warn and go ahead; the tested one says nothing
    release = tmp / "os-release"
    release.write_text('PRETTY_NAME="Debian GNU/Linux 12"\nID=debian\nVERSION_ID="12"\n')
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    warnings = cli.platform_warnings(SimpleNamespace(nodes=[1, 2]), release)
    assert [w.split()[:2] for w in warnings] == [["untested", "operating"], ["untested", "architecture"], ["the", "site"]]
    release.write_text('PRETTY_NAME="Ubuntu 24.04.3 LTS"\nID=ubuntu\nVERSION_ID="24.04"\n')
    monkeypatch.setattr(cli.platform, "machine", lambda: "aarch64")
    assert cli.platform_warnings(SimpleNamespace(nodes=[1]), release) == []
    taken = tmp / "taken"  # a machine whose slurm someone else set up: never replaced unasked
    (taken / "etc/slurm").mkdir(parents=True)
    (taken / "etc/slurm/slurm.conf").write_text("ClusterName=theirs\n")
    assert run(site, taken, "apply") == 1 and (taken / "etc/slurm/slurm.conf").read_text() == "ClusterName=theirs\n"
    assert run(site, taken, "apply", "--replace-existing-slurm") == 0
    assert RENDERED_MARK in (taken / "etc/slurm/slurm.conf").read_text().splitlines()[0]
    assert run(site, taken, "apply") == 0  # once ours, an apply goes ahead without the flag
    proc = tmp / "proc"
    proc.mkdir()
    for name, value in [("protected_regular", "2"), ("protected_symlinks", "0")]:  # protected_hardlinks absent
        (proc / name).write_text(value + "\n")
    monkeypatch.setattr(cli, "KERNEL_PROTECTIONS", proc)
    assert [p.split()[0] for p in cli.kernel_refusals()] == ["fs.protected_symlinks", "fs.protected_hardlinks"]
    assert run(site, tmp / "root", "apply") == 1  # the shared directories would be unsafe: nothing is installed
    assert not (tmp / "root").exists()


def test_user_and_admin_argv(env):
    tmp, site, fake = env
    fake.known = {"cluster where cluster=t format=cluster", "assoc where cluster=t account=gpujobs format=account",
                  "assoc where cluster=t account=gpujobs user=bob format=user", "user where name=erin format=user",
                  "qos where name=normal format=name", "qos where name=high format=name"}  # every database has normal
    assert run(site, tmp, "user", "add", "bob") == 0  # already there: nothing added
    assert run(site, tmp, "user", "add", "erin") == 0  # known to accounting elsewhere: keeps her default account
    assert run(site, tmp, "admin", "add", "alice") == 0  # not a user yet: made one first
    assert run(site, tmp, "admin", "remove", "alice") == 0
    assert run(site, tmp, "user", "remove", "alice") == 0
    assert run(site, tmp, "service", "add", "alice") == 1  # no [service] table: refused, nothing run
    low = ["sacctmgr", "-i", "add", "qos", "low"]  # the missing level, each time (this fake never learns it)
    assert [a for a in fake.calls if "show" not in a] == [
        low,
        low, ["sacctmgr", "-i", "add", "user", "erin", "Account=gpujobs", "Cluster=t"],
        low, ["sacctmgr", "-i", "add", "user", "alice", "Account=gpujobs", "DefaultAccount=gpujobs", "Cluster=t"],
        ["usermod", "-aG", "gpujobs-admin", "alice"],
        ["sacctmgr", "-i", "modify", "user", "name=alice", "set", "adminlevel=Operator"],
        ["scontrol", "reconfigure"],  # slurmctld's cached AllowGroups list, else up to 600 s before urgent admits her
        ["gpasswd", "-d", "alice", "gpujobs-admin"],
        ["sacctmgr", "-i", "modify", "user", "name=alice", "set", "adminlevel=None"],
        ["scontrol", "reconfigure"],
        ["sacctmgr", "-i", "delete", "user", "name=alice", "account=gpujobs", "cluster=t"],
    ]


def test_migrate_keeps_owner_and_mode_and_never_overwrites(env, monkeypatch):
    tmp, site, _ = env
    chowned = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: chowned.append((Path(path).name, uid, gid)))  # not root here
    old, root = tmp / "old", tmp / "root"
    (old / "jobs").mkdir(parents=True)
    (old / "jobs/1.json").write_text("new")
    (old / "jobs/1.json").chmod(0o640)
    (old / "events").mkdir()
    (old / "events/e.json").write_text("e")
    (old / "events/e.json").chmod(0o640)
    secret = tmp / "secret"
    secret.write_text("not the user's to copy")
    (old / "events/link.json").symlink_to(secret)  # root copies from a user's directory: never through a link
    (old / "footprints").mkdir()
    (old / "footprints/x.jsonl").write_text("f")
    (old / "metrics").symlink_to(tmp)  # a linked directory is not walked, whatever it is called
    existing = root / "var/lib/gpujobs/jobs/1.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep")
    assert run(site, root, "migrate-state", "--from", str(old)) == 0
    assert existing.read_text() == "keep"
    assert not (root / "var/lib/gpujobs/events/link.json").exists()
    assert not (root / "var/lib/gpujobs/metrics").exists()
    assert (root / "var/lib/gpujobs/footprints/x.jsonl").read_text() == "f"
    moved = root / "var/lib/gpujobs/events/e.json"
    assert moved.read_text() == "e"
    assert stat.S_IMODE(moved.stat().st_mode) == stat.S_IMODE((old / "events/e.json").stat().st_mode)
    source = (old / "events/e.json").stat()
    assert ("e.json", source.st_uid, source.st_gid) in chowned  # root keeps each file's writer as its owner


def test_apt_flags_apply_only_to_the_component_that_asks():
    plain = SimpleNamespace(packages=lambda _s: ["slurm-wlm"])
    lean = SimpleNamespace(packages=lambda _s: ["prometheus"], apt_flags=lambda _s: ["--no-install-recommends"])
    none = SimpleNamespace(packages=lambda _s: [])
    assert cli.apt_install_commands([plain, lean, none], site=None) == [
        ["apt-get", "install", "-y", "slurm-wlm"],
        ["apt-get", "install", "-y", "--no-install-recommends", "prometheus"],
    ]



def test_slurm_table_installs_the_built_packages_in_place_of_the_distros(env, monkeypatch):
    tmp, site, fake = env
    site.write_text(SITE + "[slurm]\nversion = '26.05.5'\nsha256 = '" + "ab" * 32 + "'\n")
    built = []
    debs = [tmp / "slurm-smd_26.05.5_arm64.deb", tmp / "slurm-smd-dbgsym_26.05.5_arm64.deb"]
    monkeypatch.setattr(slurm_component.slurm_build, "build", lambda v, s: built.append((v, s)) or debs)
    fake.distro_slurm = "ii slurm-wlm\nii slurmd\n"
    fake.policy = "  Candidate: 21.08.5-2ubuntu1\n"  # an old distro slurm no longer matters: it is not installed
    root = tmp / "root"
    assert run(site, root, "apply") == 0
    assert built == [("26.05.5", "ab" * 32)]
    installs = [c for c in fake.calls if c[:1] == ["apt-get"]]
    assert installs == [["apt-get", "install", "-y", "mariadb-server", "munge"],
                        ["apt-get", "remove", "-y", "slurm-wlm", "slurmd"],  # only what is installed
                        ["apt-get", "install", "-y", str(debs[0])]]  # not the debug symbols
    # without apt-get, dnf stands in when the site builds slurm, and is refused when it does not
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/dnf" if name == "dnf" else None)
    modules = components.load(cli.site_config.load(site), ["slurm"])
    assert cli.apt_install_commands(modules, cli.site_config.load(site)) == [["dnf", "install", "-y", "mariadb-server", "munge"]]
    assert cli.platform_refusals(cli.site_config.load(site), modules) == []
    assert cli.platform_refusals(SimpleNamespace(slurm_build=None), modules)


class UpgradeFake(Fake):
    """A running slurm 23.11.4 with job 7 in its history; `fail` is an argv prefix whose command exits 1."""

    def __init__(self, fail=()):
        super().__init__()
        self.version, self.fail, self.dead_dbd, self.lose_history = "slurm-wlm 23.11.4", list(fail), False, False

    def __call__(self, argv):
        installed = any(c[:2] == ["apt-get", "install"] for c in self.calls)
        out = {"sinfo -V": self.version, "sacct": "" if self.lose_history and installed else "7\n", "scontrol show": "StateSaveLocation = /var/spool/slurmctld\nSlurmUser = slurm(64030)\n"}
        text = " ".join(argv)
        if self.fail and text.startswith(" ".join(self.fail)):
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        for prefix in ("mariadb-dump", "tar"):  # the backup files exist once the command has run
            if argv[0] == prefix:
                Path(next(a.split("=")[-1] for a in argv if a.startswith("/") or "=/" in a)).write_text("x" * 1_500_000)
        if argv[:4] == ["sacctmgr", "-n", "show", "cluster"] and self.dead_dbd:
            return subprocess.CompletedProcess(argv, 1, "", "Connection refused")
        done = super().__call__(argv)
        match = next((v for k, v in out.items() if text.startswith(k)), None)
        return done if match is None else subprocess.CompletedProcess(argv, 0, match, "")


class FakeDbd:
    """slurmdbd in the foreground: runs until terminated, or has already exited with `dead`."""

    def __init__(self, calls, dead=False):
        self.dead, self.calls = dead, calls

    def poll(self):
        return 1 if self.dead else None

    def terminate(self):
        self.dead = True
        self.calls.append(["terminated"])

    def wait(self):
        pass


@pytest.fixture
def upgrade(env, monkeypatch, capsys):
    tmp, site, fake = env
    site.write_text(SITE + f"[slurm]\nversion = '26.05.5'\nsha256 = '{'cd' * 32}'\nsteps = ['25.05.9:{'ab' * 32}']\n")
    fake = UpgradeFake()
    components.set_runner(fake)
    monkeypatch.setattr(slurm_component, "BACKUP_ROOT", tmp / "backups")
    monkeypatch.setattr(slurm_component, "sleep", lambda _s: None)
    monkeypatch.setattr(slurm_component.slurm_build, "build", lambda v, s: [tmp / f"slurm-smd_{v}.deb"])
    monkeypatch.setattr(components, "popen", lambda argv: fake.calls.append(argv) or FakeDbd(fake.calls, fake.dead_dbd))
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "cmd_apply", lambda *a: fake.calls.append(["apply"]) or 0)
    return tmp, site, fake


def test_upgrade_slurm_backs_up_then_steps_through_each_release(upgrade, capsys):
    tmp, site, fake = upgrade
    assert run(site, tmp / "root", "upgrade-slurm") == 0
    calls = [" ".join(c) for c in fake.calls]
    saved = str(next((tmp / "backups").iterdir()))
    assert [c for c in calls if c.startswith(("mariadb-dump", "tar"))] == [
        f"mariadb-dump --result-file={saved}/slurm_acct_db.sql slurm_acct_db",
        f"tar -czf {saved}/state_save.tar.gz -C / var/spool/slurmctld", f"tar -czf {saved}/etc_slurm.tar.gz -C / etc/slurm"]
    after_backup = calls[calls.index(f"tar -czf {saved}/etc_slurm.tar.gz -C / etc/slurm"):]
    one = ["systemctl stop slurmctld slurmd", "systemctl stop slurmdbd", "apt-get install -y {deb}",
           "sudo -u slurm slurmdbd -D -vvv", "terminated", "systemctl start slurmdbd", "systemctl start slurmctld",
           "systemctl start slurmd"]
    assert [c for c in after_backup if c.startswith(("systemctl", "apt-get", "sudo", "terminated"))] == [
        c.format(deb=f"{tmp}/slurm-smd_{v}.deb") for v in ("25.05.9", "26.05.5") for c in one]
    assert calls[-1] == "apply"
    assert "state_save.tar.gz: 1.5 MB" in capsys.readouterr().out  # each backup file's size, decimal


def test_upgrade_slurm_refuses_before_touching_anything(upgrade, capsys):
    tmp, site, fake = upgrade
    fake.queue = "12 jobs\n"
    assert run(site, tmp / "root", "upgrade-slurm") == 1 and "slurm has jobs" in capsys.readouterr().err
    fake.queue, fake.version = "", "slurm 26.05.5"
    assert run(site, tmp / "root", "upgrade-slurm") == 1 and "already at or after" in capsys.readouterr().err
    assert not any(c[0] in ("mariadb-dump", "tar") for c in fake.calls) and not (tmp / "backups").exists()


# a failed install, a slurmdbd that dies converting, and a conversion that loses job history (data loss)
@pytest.mark.parametrize("fail, dead, lose", [(("apt-get", "install", "-y", "/"), False, False), ((), True, False), ((), False, True)])
def test_a_failed_step_prints_the_restore_commands_and_goes_no_further(upgrade, capsys, fail, dead, lose):
    tmp, site, fake = upgrade
    fake.fail, fake.dead_dbd, fake.lose_history = list(fail), dead, lose
    assert run(site, tmp / "root", "upgrade-slurm") == 1
    err = capsys.readouterr().err
    assert "UPGRADE FAILED at slurm 25.05.9" in err and "slurm_acct_db <" in err and "apt-get install -y --allow-downgrades slurm-wlm" in err
    assert not any("26.05.5" in " ".join(c) for c in fake.calls) and ["apply"] not in fake.calls
    assert lose or ["systemctl", "start", "slurmctld"] not in fake.calls  # a dead slurmdbd never reaches the controller
    assert not lose or "sacct no longer lists 1 job(s)" in err
