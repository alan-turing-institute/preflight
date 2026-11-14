import getpass
import importlib.util
import os
import shlex
import stat
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta
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
        self.distro_slurm = ""  # what `dpkg-query -W` reports installed, as dpkg prints it, e.g. "ii  slurm-wlm\nii  slurmd\n" (the abbreviation has its own trailing space)
        self.sinfo = ""  # what `sinfo -V` prints ("" = no slurm installed)
        self.dbd_active = False  # `systemctl is-active slurmdbd`
        self.held = []  # (argv, whether policy-rc.d stood) for each package install or removal
        self.rpms = ""  # what `rpm -qa --qf '%{NAME}\n'` lists, one name a line

    def __call__(self, argv):
        self.calls.append(argv)
        rc = 0
        if argv[:1] in (["apt-get"], ["dnf"]) and argv[1] in ("install", "remove"):
            self.held.append((argv, (components.ROOT / slurm_component.POLICY_RC_D).exists()))
        if argv[:1] == ["squeue"]:
            return subprocess.CompletedProcess(argv, self.squeue_rc, self.queue, "")
        if argv[:1] == ["sinfo"] and self.sinfo:
            return subprocess.CompletedProcess(argv, 0, self.sinfo, "")
        if argv[:3] == ["systemctl", "is-active", "slurmdbd"]:
            return subprocess.CompletedProcess(argv, 0 if self.dbd_active else 3, "active" if self.dbd_active else "inactive", "")
        if argv[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(argv, self.ctld_rc, "", "")
        if argv[:2] == ["rpm", "-qa"]:
            return subprocess.CompletedProcess(argv, 0, self.rpms, "")
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
    assert stat.S_IMODE((root / "etc/slurm/slurmdbd.conf").stat().st_mode) == cli.render.SECRET_MODE
    assert stat.S_IMODE((root / "usr/local/bin/gpujobs-submit").stat().st_mode) == cli.render.EXEC_MODE
    assert stat.S_IMODE((root / "opt/gpujobs/slurm/ood_app/bin/python").stat().st_mode) == cli.render.EXEC_MODE  # OOD runs it
    # The installed wrapper runs a job with only what the install put there (this site leaves hostguard off).
    assert not (root / "opt/gpujobs/hostguard").exists()
    job = dict(os.environ, GPUJOBS_SITE=str(site), SLURM_GPU_STATE_DIR=str(tmp / "state"), SLURM_JOB_ID="1")
    wrapper = [sys.executable, str(root / "opt/gpujobs/slurm/job_wrapper.py"), "--", "true"]
    assert subprocess.run(wrapper, env=job, check=False).returncode == 0
    gated = SimpleNamespace(install_dir=Path("/opt/gpujobs"), hostguard=False, dgx_gate=True, health_check=None)
    shipped = components.load(gated, ["tools"])[0].files(gated)
    assert {"/opt/gpujobs/hostguard/hostguard.py", "/opt/gpujobs/dgx/dgx_submit.sh"} <= shipped.keys()
    assert not any(k.endswith(".local.env") for k in shipped)  # a checkout's own dgx.local.env is never installed
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
        f"{state}/remote": ("0755", "slurm"),  # and the copied-back job state of the part-time machines
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
    warnings = cli.platform_warnings(SimpleNamespace(nodes=[1, 2], full_nodes=()), release)
    assert [w.split()[:2] for w in warnings] == [["untested", "operating"], ["untested", "architecture"], ["the", "site"]]
    release.write_text('PRETTY_NAME="Ubuntu 24.04.3 LTS"\nID=ubuntu\nVERSION_ID="24.04"\n')
    monkeypatch.setattr(cli.platform, "machine", lambda: "aarch64")
    assert cli.platform_warnings(SimpleNamespace(nodes=[1], full_nodes=()), release) == []
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



def test_slurm_table_installs_the_built_packages_in_place_of_the_distros(env, monkeypatch, capsys):
    tmp, site, fake = env
    site.write_text(SITE + "[slurm]\nversion = '26.05.5'\nsha256 = '" + "ab" * 32 + "'\n")
    built = []
    names = ["slurm-smd-dbgsym", "slurm-smd-dev", "slurm-smd-slurmdbd", "slurm-smd-doc", "slurm-smd", "slurm-smd-sview"]
    debs = [tmp / f"{n}_26.05.5_arm64.deb" for n in names]
    monkeypatch.setattr(slurm_component.slurm_build, "build", lambda v, s: built.append((v, s)) or debs)
    fake.distro_slurm = "ii  slurm-wlm\nii  slurmd\nrc  slurmdbd\nhi  slurm-smd\n"  # and one of ours, held by an earlier install
    fake.policy = "  Candidate: 21.08.5-2ubuntu1\n"  # an old distro slurm no longer matters: it is not installed
    root = tmp / "root"
    policy = root / "usr/sbin/policy-rc.d"
    policy.parent.mkdir(parents=True)
    policy.write_text("someone's own\n")  # kept, and put back afterwards
    fake.sinfo = "slurm-wlm 23.11.4"  # apply never changes the release: that is upgrade-slurm's
    assert run(site, root, "apply") == 1 and built == [] and "use install.py upgrade-slurm" in capsys.readouterr().err
    site.write_text(site.read_text() + "[slurm_release]\ncheckout = '/c'\nrecords = '/r'\ntest_user = 'u'\n")
    fake.sinfo = "slurm 26.11.1"  # newer than the pin: a maintenance window installed it; the refusal names its record
    assert run(site, root, "apply") == 1 and 'version = "26.11.1" with the sha256 in /r/26.11.1.json' in capsys.readouterr().err
    fake.sinfo = "slurm 26.05.5"
    assert run(site, root, "apply") == 0
    assert built == [("26.05.5", "ab" * 32)]
    installs = [c for c in fake.calls if c[:1] == ["apt-get"]]
    assert installs == [["apt-get", "install", "-y", "mariadb-server", "munge"],
                        ["apt-get", "remove", "-y", "slurm-wlm", "slurmd"],  # only what is installed
                        # only the daemons, the client and the base package, not -dev, -doc, sview or debug symbols
                        ["apt-get", "install", "-y", str(debs[4]), str(debs[2])]]
    # ours are locked against the distribution's after the install; unlocked first only where installed (apt-mark
    # refuses a package it does not know)
    marks = [c for c in fake.calls if c[:1] == ["apt-mark"]]
    assert marks == [["apt-mark", "unhold", "slurm-smd"], ["apt-mark", "hold", "slurm-smd", "slurm-smd-slurmdbd"]]
    assert fake.calls.index(marks[0]) < fake.calls.index(installs[-1]) < fake.calls.index(marks[1])
    # rpms: Fedora's own slurm has the names, version and release of a build made there, which `dnf install` leaves in place
    # and Fedora's `slurm-libs`, which SchedMD's build does not make, goes first (its libslurm is in ours: a file conflict)
    rpms = [tmp / "slurm-26.05.5-1.fc45.x86_64.rpm", tmp / "slurm-slurmd-26.05.5-1.fc45.x86_64.rpm",
            tmp / "slurm-devel-26.05.5-1.fc45.x86_64.rpm"]  # made, not installed, and not foreign when installed
    files = [str(r) for r in rpms[:2]]
    fake.rpms = "slurm\nslurm-libs\nslurm-slurmd\nslurm-devel\n"
    assert slurm_component.install_commands(rpms) == [
        ["dnf", "install", "-y", "dnf-command(versionlock)"], ["dnf", "versionlock", "delete", "slurm", "slurm-slurmd"],
        ["dnf", "remove", "-y", "--setopt=clean_requirements_on_remove=False", "slurm-libs"],
        ["dnf", "install", "-y", *files], ["dnf", "reinstall", "-y", *files], ["dnf", "versionlock", "add", "slurm", "slurm-slurmd"]]
    # no package command ran without policy-rc.d, which keeps the packages' scripts from starting a daemon
    assert [on for _, on in fake.held] == [True, True, True]
    assert policy.read_text() == "someone's own\n" and not policy.with_name("policy-rc.d.gpujobs-saved").exists()
    # without apt-get, dnf stands in when the site builds slurm, and is refused when it does not
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/dnf" if name == "dnf" else None)
    modules = components.load(cli.site_config.load(site), ["slurm"])
    assert cli.apt_install_commands(modules, cli.site_config.load(site)) == [["dnf", "install", "-y", "mariadb-server", "munge"]]
    assert cli.platform_refusals(cli.site_config.load(site), modules) == []
    assert cli.platform_refusals(SimpleNamespace(slurm_build=None, full_nodes=()), modules)
    fake.calls.clear()  # dnf systems mask the daemons around the install, and unmask them
    with slurm_component.daemons_held("dnf"):
        fake.calls.append(["install"])
    assert fake.calls == [["systemctl", "mask", "--runtime", "slurmdbd", "slurmctld", "slurmd"], ["install"],
                          ["systemctl", "unmask", "--runtime", "slurmdbd", "slurmctld", "slurmd"]]


PARTITIONS = ("PartitionName=gpu AllowGroups=ALL Default=YES MaxTime=UNLIMITED PreemptMode=REQUEUE State=UP TotalCPUs=20\n"
              "PartitionName=cpu AllowGroups=ALL Default=NO MaxTime=UNLIMITED PreemptMode=OFF State=DRAIN TotalCPUs=20\n")


class UpgradeFake(Fake):
    """A running slurm 23.11.4 with job 7 in its history and two partitions; `fail` is an argv prefix whose command
    exits 1, `late_job` makes the second `squeue` find a job, `slow_dbd` a slurmdbd that never answers."""

    def __init__(self, fail=()):
        super().__init__()
        self.version, self.fail, self.dead_dbd, self.lose_history = "slurm-wlm 23.11.4", list(fail), False, False
        self.slow_dbd = self.late_job = False
        self.squeues = 0

    def __call__(self, argv):
        installed = any(c[:2] == ["apt-get", "install"] for c in self.calls)
        out = {"sinfo -V": self.version, "sacct": "" if self.lose_history and installed else "7\n",
               "scontrol show config": "StateSaveLocation = /var/spool/slurmctld\nSlurmUser = slurm(64030)\n",
               "scontrol show partition": PARTITIONS}
        text = " ".join(argv)
        if self.fail and text.startswith(" ".join(self.fail)):
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        for prefix in ("mariadb-dump", "tar"):  # the backup files exist once the command has run
            if argv[0] == prefix:
                Path(next(a.split("=")[-1] for a in argv if a.startswith("/") or "=/" in a)).write_text("x" * 1_500_000)
        if argv[:4] == ["sacctmgr", "-n", "show", "cluster"] and (self.dead_dbd or self.slow_dbd):
            return subprocess.CompletedProcess(argv, 1, "", "Connection refused")
        if argv[:2] == ["squeue", "-h"]:
            self.squeues += 1
            if self.late_job:
                self.queue = "9 jobs\n" if self.squeues > 1 else ""
        done = super().__call__(argv)
        match = next((v for k, v in out.items() if text.startswith(k)), None)
        return done if match is None else subprocess.CompletedProcess(argv, 0, match, "")


class FakeDbd:
    """slurmdbd in the foreground: runs until terminated, or has already exited with `dead`."""

    pid = 4242

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
    monkeypatch.setattr(slurm_component.slurm_build, "build", lambda v, s: fake.calls.append(["build", v]) or [tmp / f"slurm-smd_{v}.deb"])
    monkeypatch.setattr(components, "popen", lambda argv, **kw: fake.calls.append(argv) or FakeDbd(fake.calls, fake.dead_dbd))
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "cmd_apply", lambda *a: fake.calls.append(["apply"]) or 0)
    return tmp, site, fake


def test_upgrade_slurm_backs_up_then_steps_through_each_release(upgrade, capsys):
    tmp, site, fake = upgrade
    assert run(site, tmp / "root", "upgrade-slurm") == 0
    calls = [" ".join(c) for c in fake.calls]
    saved = str(next((tmp / "backups").iterdir()))
    dump = f"mariadb-dump --result-file={saved}/slurm_acct_db.sql slurm_acct_db"
    state = f"tar -czf {saved}/state_save.tar.gz -C / var/spool/slurmctld"
    conf = f"tar -czf {saved}/etc_slurm.tar.gz -C / etc/slurm"
    close = [f"scontrol update PartitionName={n} State=INACTIVE" for n in ("gpu", "cpu")]
    reopen = ["scontrol update PartitionName=gpu State=UP", "scontrol update PartitionName=cpu State=DRAIN"]  # as they were
    step = ["apt-get install -y {deb}", "systemctl is-active slurmdbd", "sudo -u slurm slurmdbd -D -vvv", "terminated",
            "systemctl start slurmdbd", "systemctl start slurmctld", "systemctl start slurmd"]
    interesting = ("build", "systemctl", "apt-get", "sudo", "terminated", "scontrol update", "squeue", "mariadb-dump", "tar", "apply")
    assert [c for c in calls if c.startswith(interesting)] == [
        " ".join(slurm_component.ACTIVE_JOBS),  # the first no-running-jobs check, then both releases are built, all before anything stops
        "build 25.05.9", "build 26.05.5",
        *close, " ".join(slurm_component.ACTIVE_JOBS),  # no partition takes a job, then the jobs are checked again
        "systemctl stop slurmctld slurmd", dump, state, conf, "systemctl stop slurmdbd",  # slurmdbd only after the dump
        *(c.format(deb=f"{tmp}/slurm-smd_25.05.9.deb") for c in step),
        "systemctl stop slurmctld slurmd", "systemctl stop slurmdbd",
        *(c.format(deb=f"{tmp}/slurm-smd_26.05.5.deb") for c in step),
        *reopen, "apply"]
    # every sacct listing covers the same absolute seven days
    windows = {tuple(c[c.index("-S") + 1:c.index("-S") + 2] + c[c.index("-E") + 1:c.index("-E") + 2]) for c in fake.calls if c[0] == "sacct"}
    (start, end), = windows
    stamp = "%Y-%m-%dT%H:%M:%S"
    assert datetime.strptime(end, stamp) - datetime.strptime(start, stamp) == timedelta(days=7)
    # package installs ran under policy-rc.d, which is gone again, and slurmdbd's output has a log beside the backup
    assert [on for _, on in fake.held] == [True, True] and not (tmp / "root/usr/sbin/policy-rc.d").exists()
    assert (Path(saved) / "slurmdbd-conversion.log").exists()
    assert "state_save.tar.gz: 1.5 MB" in capsys.readouterr().out  # each backup file's size, decimal


def test_upgrade_slurm_refuses_before_touching_anything(upgrade, capsys, monkeypatch):
    tmp, site, fake = upgrade
    fake.queue = "12 jobs\n"
    assert run(site, tmp / "root", "upgrade-slurm") == 1 and "slurm has running jobs" in capsys.readouterr().err
    fake.queue, fake.version = "", "slurm 26.05.5"
    assert run(site, tmp / "root", "upgrade-slurm") == 1 and "already at or after" in capsys.readouterr().err
    fake.version = "slurm-wlm 23.11.4"

    def build(version, sha256):  # the second release cannot be built: stop before the first is installed
        if version == "26.05.5":
            raise slurm_component.slurm_build.SlurmBuildError("no network")
        return [tmp / f"slurm-smd_{version}.deb"]

    monkeypatch.setattr(slurm_component.slurm_build, "build", build)
    assert run(site, tmp / "root", "upgrade-slurm") == cli.site_config.UpgradeExit.REFUSED
    err = capsys.readouterr().err
    assert "no network" in err and "slurm was not touched" in err
    touched = ("systemctl", "mariadb-dump", "tar", "apt-get", "sudo")
    assert not any(c[0] in touched or c[:2] == ["scontrol", "update"] for c in fake.calls) and not (tmp / "backups").exists()


# a failed install, a slurmdbd that dies converting, a conversion that loses job history (data loss), and a slurmdbd
# service already running when the foreground one would start (two on one database)
@pytest.mark.parametrize("what", [{"fail": ["apt-get", "install", "-y", "/"]}, {"dead_dbd": True}, {"lose_history": True}, {"dbd_active": True}])
def test_a_failed_step_prints_the_restore_commands_and_goes_no_further(upgrade, capsys, what):
    tmp, site, fake = upgrade
    vars(fake).update(what)
    assert run(site, tmp / "root", "upgrade-slurm") == cli.site_config.UpgradeExit.FAILED_AFTER_CHANGE  # the window must not reboot
    err = capsys.readouterr().err
    assert "UPGRADE FAILED at slurm 25.05.9" in err and "apt-get install -y --allow-downgrades slurm-wlm" in err
    assert not any("26.05.5" in " ".join(c) for c in fake.calls if c[0] != "build") and ["apply"] not in fake.calls
    assert "lose_history" in what or ["systemctl", "start", "slurmctld"] not in fake.calls  # a dead slurmdbd never reaches the controller
    assert "lose_history" not in what or "sacct no longer lists 1 job(s)" in err
    assert not (tmp / "root/usr/sbin/policy-rc.d").exists()  # the guard comes off even when the install failed
    # the restore, in order: stop all; reinstall the old packages with no daemon starting; stop again (a script may have
    # started one); drop and reload the database, state and settings; restart (not start); reopen the partitions
    at = -1
    for needle in ("systemctl stop slurmctld slurmd slurmdbd", "exit 101", "--allow-downgrades slurm-wlm", "rm -f /usr/sbin/policy-rc.d",
                   "systemctl stop slurmctld slurmd slurmdbd", "DROP DATABASE", "tar -xzf", "systemctl restart slurmdbd slurmctld slurmd",
                   "scontrol update PartitionName=gpu State=UP"):
        at = err.index(needle, at + 1)
    # SchedMD's packages come back from this system release's build in the cache, only the ones the upgrade installs,
    # with the lock the upgrade put on the new ones lifted first and put back on the old ones after
    here = slurm_component.slurm_build.build_dir()
    smd = slurm_component.restore_text(Path("/b"), "/s", "slurm 25.05.9", "apt-get", {})
    assert f"/slurm/25.05.9/{here}/slurm-smd-slurmdbd_*.deb" in smd and "slurm-smd*.deb" not in smd
    assert smd.index("apt-mark unhold slurm-smd ") < smd.index("apt-get remove") < smd.index("apt-mark hold slurm-smd ")
    rpm = slurm_component.restore_text(Path("/b"), "/s", "slurm 25.05.9", "dnf", {})
    assert f"/slurm/25.05.9/{here}/rpmbuild/RPMS/*/slurm-slurmdbd-[0-9]*.rpm" in rpm
    assert rpm.index("dnf versionlock delete slurm ") < rpm.index("dnf install") < rpm.index("dnf versionlock add slurm ")


def test_a_slurmdbd_still_converting_is_left_running_and_nothing_is_restored(upgrade, capsys):
    tmp, site, fake = upgrade
    fake.slow_dbd = True
    assert run(site, tmp / "root", "upgrade-slurm") == cli.site_config.UpgradeExit.CONVERSION_RUNNING
    err = capsys.readouterr().err
    assert "still converting" in err and "pid 4242" in err and "tail -f" in err and "slurmdbd-conversion.log" in err
    assert ["terminated"] not in fake.calls  # never signalled
    assert "DROP DATABASE" not in err and "UPGRADE FAILED" not in err and "Do not stop that slurmdbd" in err
    assert ["systemctl", "start", "slurmctld"] not in fake.calls and ["apply"] not in fake.calls


# a job slips in after the first check, and a backup that fails: no package has changed, so slurm is put back
@pytest.mark.parametrize("what, reopened", [({"late_job": True}, False), ({"fail": ["tar", "-czf"]}, True)])
def test_a_failure_before_any_package_changes_puts_slurm_back(upgrade, capsys, what, reopened):
    tmp, site, fake = upgrade
    vars(fake).update(what)
    assert run(site, tmp / "root", "upgrade-slurm") == cli.site_config.UpgradeExit.REFUSED  # nothing changed
    err = capsys.readouterr().err
    assert "UPGRADE REFUSED" in err and "UPGRADE FAILED" not in err and "DROP DATABASE" not in err
    calls = [" ".join(c) for c in fake.calls]
    assert not any(c.startswith("apt-get") for c in calls) and "apply" not in calls
    assert calls.index("scontrol update PartitionName=gpu State=INACTIVE") < calls.index("scontrol update PartitionName=gpu State=UP")
    assert calls[-2:] == ["scontrol update PartitionName=gpu State=UP", "scontrol update PartitionName=cpu State=DRAIN"]
    assert ("systemctl start slurmctld" in calls) is reopened  # stopped only when the queue check passed


PART_TIME_SITE = SITE + """
[[nodes]]
name = "laptop"
cpus = 4
real_memory_mb = 1000
part_time = true
address = "100.1.1.1"
[slurm]
version = '26.05.5'
sha256 = '""" + "ab" * 32 + "'\n"
HOST_KEY = "# Host 100.1.1.1 found: line 3\n100.1.1.1 ssh-ed25519 AAAAHOST\n"
UNIT = "# /usr/lib/systemd/system/slurmd.service\n[Service]\nExecStart=/usr/sbin/slurmd --systemd $SLURMD_OPTIONS\n"


class PartTimeFake(Fake):
    """`local` and `remote` map a user to its uid for `getent passwd` here and over ssh; `users` is what accounting lists."""

    def __init__(self, local, remote, users, firewalld=True, unit=UNIT, known_host=HOST_KEY):
        super().__init__()
        self.local, self.remote, self.users, self.firewalld, self.unit = local, remote, users, firewalld, unit
        self.known_host = known_host  # what `ssh-keygen -F` finds in the admin's known_hosts: "" for no entry

    def __call__(self, argv):
        if argv[:1] == ["ssh"] and argv[-3:-1] == ["getent", "passwd"] or argv[:2] == ["getent", "passwd"]:
            table = self.remote if argv[:1] == ["ssh"] else self.local
            self.calls.append(argv)
            name = shlex.split(argv[-1])[0] if argv[:1] == ["ssh"] else argv[-1]  # the node's shell unquotes it
            uid = table.get(name)
            line = f"{name}:x:{uid}:{uid}::/home:/bin/sh\n" if uid is not None else ""
            return subprocess.CompletedProcess(argv, 0 if uid is not None else 2, line, "")
        if argv[:5] == ["sacctmgr", "-n", "-P", "show", "assoc"]:
            return subprocess.CompletedProcess(argv, 0, "\n".join(self.users) + "\n", "")
        if argv[:2] == ["ssh-keygen", "-F"] or argv[:1] == ["sudo"] and argv[3:5] == ["ssh-keygen", "-F"]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0 if self.known_host else 1, self.known_host, "")
        if argv[:3] == ["systemctl", "is-active", "firewalld"]:
            return subprocess.CompletedProcess(argv, 0 if self.firewalld else 3, "", "")
        if argv[:3] == ["systemctl", "cat", "slurmd"]:
            return subprocess.CompletedProcess(argv, 0, self.unit, "")
        if argv[:3] == ["getent", "group", "slurm"]:
            return subprocess.CompletedProcess(argv, 2, "", "")
        return super().__call__(argv)


def test_a_controller_restart_tells_the_part_time_nodes_to_fetch_the_settings(monkeypatch, capsys):
    # a configless slurmd keeps the settings it took at its start; only a reconfigure makes it fetch them again
    class Listing(Fake):
        """`rows`: what each `sinfo -N` lists in turn (the last repeats); `reconfigure_rc`: scontrol's exit status."""

        def __init__(self, rows, reconfigure_rc=0):
            super().__init__()
            self.rows, self.reconfigure_rc = list(rows), reconfigure_rc

        def __call__(self, argv):
            if argv[:2] == ["sinfo", "-h"] and "-N" in argv:
                self.calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, self.rows.pop(0) if len(self.rows) > 1 else self.rows[0], "")
            if argv == slurm_component.RECONFIGURE:
                self.calls.append(argv)
                return subprocess.CompletedProcess(argv, self.reconfigure_rc, "", "down")
            return super().__call__(argv)

    monkeypatch.setattr(slurm_component, "sleep", lambda _s: None)
    two = SimpleNamespace(part_time_nodes=(SimpleNamespace(name="a"), SimpleNamespace(name="b")))

    def told(fake):
        components.set_runner(fake)
        try:
            assert slurm_component.refresh_part_time(two, wait_s=3) == []  # it runs the reconfigure itself, never stops
        finally:
            components.set_runner(None)
        return fake.calls, capsys.readouterr()

    # just after the restart slurm still marks a node not responding from its saved state: wait for it, then tell
    calls, out = told(Listing(["a down*\nb idle*\n", "a idle\nb idle\n"]))
    assert calls[-1] == slurm_component.RECONFIGURE and len(calls) == 3 and "a, b: told to fetch" in out.out and not out.err
    # one asleep (marked) and one missing from the listing: both named, after the wait; the answering one is told
    calls, out = told(Listing(["a idle\nb idle*\n"]))
    assert len(calls) == 5 and "node(s) a: told" in out.out and "a idle" not in out.err
    calls, out = told(Listing(["a idle\n"]))
    assert "part-time node(s) b not answering" in out.err
    calls, out = told(Listing(["a idle\nb idle\n"], reconfigure_rc=1))  # a failed reconfigure is a warning naming it
    assert "`scontrol reconfigure` failed (1)" in out.err and "told" not in out.out
    plain = slurm_component.post_install(cli.site_config.parse(tomllib.loads(SITE)))
    assert len(slurm_component.post_install(cli.site_config.parse(tomllib.loads(PART_TIME_SITE)))) == len(plain) + 1  # last


def part_time_env(tmp_path, monkeypatch, fake):
    site = tmp_path / "site.toml"
    site.write_text(PART_TIME_SITE)
    components.set_runner(fake)
    monkeypatch.setattr(importlib.import_module("components.lease"), "_home", lambda _user: tmp_path / "home")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_component.slurm_build, "build", lambda v, s: [tmp_path / "slurm-smd_26.05.5_arm64.deb"])
    return site


def test_part_time_add_refuses_every_uid_mismatch_then_copies_the_key_without_showing_it(tmp_path, monkeypatch, capsys):
    local = {"alice": 1001, "bob": 1002, "carol": 1003, "slurm": 64030, "x;reboot": 1004}
    fake = PartTimeFake(local, {"alice": 2001, "carol": 1003, "x;reboot": 1004}, ["alice", "bob", "carol", "x;reboot"])
    site = part_time_env(tmp_path, monkeypatch, fake)
    root = tmp_path / "root"
    (root / "etc/munge").mkdir(parents=True)
    (root / "etc/munge/munge.key").write_bytes(b"SECRETKEYBYTES")
    (root / "var/lib/gpujobs/agent/ssh").mkdir(parents=True)
    (root / "var/lib/gpujobs/agent/ssh/id_ed25519.pub").write_text("ssh-ed25519 AAAAKEY gpujobs-copy-back\n")
    sent = []
    monkeypatch.setattr(importlib.import_module("components.part_time"), "send",
                        lambda argv, data: sent.append((argv, data)) or subprocess.CompletedProcess(argv, 0, b"", b""))
    try:
        assert run(site, root, "part-time", "add", "laptop") == 1
        err = capsys.readouterr().err
        assert "alice: uid 1001 here, 2001 on 100.1.1.1" in err and "bob: uid 1002 here, missing" in err
        assert "slurm" not in err and "carol" not in err  # slurm is created by the node install; carol matches
        assert sent == []  # nothing is copied while any id differs
        # every word sent to the node's shell is quoted, so an account name stays one word there
        assert next(c for c in fake.calls if c[:1] == ["ssh"] and c[-1] == "'x;reboot'")[-3:-1] == ["getent", "passwd"]
        fake.remote.update(alice=1001, bob=1002)  # slurm may stay missing
        fake.known_host = ""  # the admin has never ssh'd to the node: no host key to give the agent
        with pytest.raises(SystemExit, match="ssh to 100.1.1.1 once as yourself first"):
            run(site, root, "part-time", "add", "laptop")
        assert sent == []  # nothing is copied or authorized without it
        fake.known_host = HOST_KEY
        assert run(site, root, "part-time", "add", "laptop") == 0
    finally:
        components.set_runner(None)
    argv, data = sent[0]
    assert argv[-10:] == ["sudo", "-n", "install", "-m", "0400", "-o", "munge", "-g", "munge", "/dev/stdin", "/etc/munge/munge.key"][-10:]
    assert data == b"SECRETKEYBYTES"
    # B5b: SlurmUser's key (made here when absent) may only run a read-only rsync of the node's state directory
    keygen = ["runuser", "-u", "slurm", "--", "ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gpujobs-copy-back",
              "-f", "/var/lib/gpujobs/agent/ssh/id_ed25519"]
    assert keygen in fake.calls and ["install", "-d", "-m", "0700", "-o", "slurm", "-g", "slurm", "/var/lib/gpujobs/agent/ssh"] in fake.calls
    # the agent's ssh trusts the host key the admin saw, and only that one
    argv, data = sent[1]
    assert argv == ["install", "-m", "0644", "-o", "slurm", "-g", "slurm", "/dev/stdin", "/var/lib/gpujobs/agent/ssh/known_hosts"]
    assert data == b"100.1.1.1 ssh-ed25519 AAAAHOST\n"  # the entry, not ssh-keygen's "# Host ... found" comment
    argv, data = sent[2]
    assert argv[-10:] == ["sudo", "-n", "install", "-m", "0600", "-o", "slurm", "-g", "slurm", "/dev/stdin", "/var/lib/slurm/.ssh/authorized_keys"][-10:]
    assert data == b'command="/usr/bin/rrsync -ro /var/lib/gpujobs",restrict ssh-ed25519 AAAAKEY gpujobs-copy-back\n'
    out = capsys.readouterr()
    assert "SECRETKEYBYTES" not in out.out + out.err and not any("SECRETKEYBYTES" in " ".join(c) for c in fake.calls)


def apply_part_time(site, tmp_path, *extra):
    flags = ["--part-time", "laptop", "--controller", "100.2.2.2", "--slurm-uid", "64030", "--slurm-gid", "64030"]
    return run(site, tmp_path / "root", "apply", *flags, *extra)


def test_part_time_apply_refuses_without_the_trust_flag_or_the_controllers_slurm_ids_or_an_execstart(tmp_path, monkeypatch, capsys):
    fake = PartTimeFake({"slurm": 999}, {}, [])
    site = part_time_env(tmp_path, monkeypatch, fake)
    try:
        assert apply_part_time(site, tmp_path) == 1
        assert "anyone who is root here can use it to act as ANY" in capsys.readouterr().err
        assert not [c for c in fake.calls if c[:1] in (["useradd"], ["apt-get"])]
        assert apply_part_time(site, tmp_path, "--i-accept-cluster-trust") == 1  # slurm is uid 999 here, 64030 there
        assert "uid/gid 999/999" in capsys.readouterr().err
        assert not [c for c in fake.calls if c[:1] in (["useradd"], ["apt-get"])]
    finally:
        components.set_runner(None)
    # a slurmd unit the drop-in cannot extend is refused before munge and slurmd are started
    fake = PartTimeFake({}, {}, [], unit="[Service]\n")
    site = part_time_env(tmp_path, monkeypatch, fake)
    try:
        with pytest.raises(SystemExit, match="no ExecStart"):
            apply_part_time(site, tmp_path, "--i-accept-cluster-trust")
    finally:
        components.set_runner(None)
    assert not any(c[:2] == ["systemctl", "enable"] and "slurmd" in c for c in fake.calls)


@pytest.mark.parametrize("firewalld", [True, False])
def test_part_time_apply_adds_the_flag_to_the_packages_unit_and_opens_the_port_only_with_firewalld(
        tmp_path, monkeypatch, firewalld):
    fake = PartTimeFake({}, {}, [], firewalld=firewalld)
    site = part_time_env(tmp_path, monkeypatch, fake)
    try:
        assert apply_part_time(site, tmp_path, "--i-accept-cluster-trust") == 0
    finally:
        components.set_runner(None)
    calls = fake.calls
    order = [c[0] for c in calls if c[0] in ("groupadd", "useradd")] + [c[:3] for c in calls if c[:1] == ["apt-get"]]
    assert order == ["groupadd", "useradd", ["apt-get", "install", "-y"], ["apt-get", "install", "-y"]]  # slurm user first
    assert ["useradd", "-r", "-u", "64030", "-g", "slurm", "-d", "/var/lib/slurm", "-s", "/bin/sh", "slurm"] in calls
    # job state (B5b): the wrapper's site file and state dirs, rrsync from the doc copy (Fedora), a home with .ssh for `slurm`
    assert ["apt-get", "install", "-y", "munge", "rsync"] in calls
    assert (tmp_path / "root/etc/gpujobs/site.toml").read_text() == PART_TIME_SITE
    assert ["install", "-d", "-m", "1777", "/var/lib/gpujobs/jobs"] in calls and ["install", "-d", "-m", "0755", "-o", "slurm", "-g", "slurm", "/var/lib/gpujobs/remote"] in calls
    assert not any(c[:2] == ["systemctl", "enable"] and "gpujobs-agent.service" in c for c in calls)  # no agent here
    assert ["install", "-m", "0755", "/usr/share/doc/rsync/support/rrsync", "/usr/local/bin/rrsync"] in calls
    assert ["install", "-d", "-m", "0700", "-o", "slurm", "-g", "slurm", "/var/lib/slurm/.ssh"] in calls
    assert ["install", "-d", "-m", "0755", "-o", "root", "-g", "root", "/var/lib/gpujobs/owners"] in calls  # node_check's manifest
    drop_in = (tmp_path / "root/etc/systemd/system/slurmd.service.d/10-conf-server.conf").read_text()
    assert drop_in == ("[Unit]\nAfter=network-online.target tailscaled.service\nWants=network-online.target\n\n[Service]\nExecStart=\n"
                       "ExecStart=/usr/sbin/slurmd --conf-server 100.2.2.2 --systemd $SLURMD_OPTIONS\nRestart=on-failure\nRestartSec=30\n")
    assert ["install", "-d", "-m", "0755", "/var/log/slurm"] in calls
    hosts = (tmp_path / "root/etc/hosts").read_text()  # the controller's name must resolve here (a VPN laptop does not know it)
    assert hosts.startswith("100.2.2.2 ") and hosts.endswith("  # gpujobs: slurm.conf's SlurmctldHost\n")
    for hook in ("prolog.d", "epilog.d"):  # slurm 26.05 fails a job where the glob matches no script
        script = tmp_path / "root/etc/slurm" / hook / "00-none.sh"
        assert script.read_text().endswith("exit 0\n") and os.access(script, os.X_OK)
    rule = "--add-rich-rule=rule family=ipv4 source address=100.2.2.2 port port=6818 protocol=tcp accept"
    assert (["firewall-cmd", "--permanent", rule] in calls) is firewalld
    assert (["firewall-cmd", "--reload"] in calls) is firewalld
    assert calls[-1] == ["systemctl", "enable", "munge", "slurmd"]  # no key here yet (Fedora): enabled for boot, not started
    key = tmp_path / "root/etc/munge/munge.key"
    key.parent.mkdir(parents=True)
    key.write_text("k")
    assert importlib.import_module("components.part_time")._enable_services() == [["systemctl", "enable", "--now", "munge", "slurmd"]]  # the key is here: start now
    assert (tmp_path / "root/opt/gpujobs").exists()  # the tools component, so node_check.py is there


def test_shared_state_copies_nothing_back_and_installs_nothing_for_it(tmp_path, monkeypatch):
    fake = PartTimeFake({"slurm": 64030, "alice": 1}, {"slurm": 64030, "alice": 1}, ["alice"])
    site = part_time_env(tmp_path, monkeypatch, fake)
    site.write_text('shared_state = true\n' + PART_TIME_SITE)
    root = tmp_path / "root"
    (root / "etc/munge").mkdir(parents=True)
    (root / "etc/munge/munge.key").write_bytes(b"K")
    sent = []
    monkeypatch.setattr(importlib.import_module("components.part_time"), "send",
                        lambda argv, data: sent.append(argv) or subprocess.CompletedProcess(argv, 0, b"", b""))
    try:
        assert apply_part_time(site, tmp_path, "--i-accept-cluster-trust") == 0
        assert run(site, root, "part-time", "add", "laptop") == 0
    finally:
        components.set_runner(None)
    flat = [" ".join(c) for c in fake.calls]
    assert ["apt-get", "install", "-y", "munge"] in fake.calls and len(sent) == 1  # only the munge key
    assert not any("rrsync" in c or "ssh-keygen" in c or ".ssh" in c for c in flat)


def test_fast_cpus_must_be_the_hosts_fastest_cores_and_an_unset_two_kind_gpu_node_is_told(capsys, monkeypatch):
    """D46: the layout binds the GPU to `fast_cpus`, so a list that is not this host's fastest cores, a list set on
    another machine's node, or a slurmd that cannot read a layout file is refused."""
    halves = cli.cores.CoreSpeeds({cpu: 2808000 if cpu % 10 < 5 else 3900000 for cpu in range(20)})
    gpu = {"index": 0, "uuid": "GPU-0", "device": "/dev/nvidia0", "type": "gb10"}
    raw = {"name": "s", "nodes": [{"name": "n1", "cpus": 20, "real_memory_mb": 100000, "gpus": [gpu], "fast_cpus": "5-9,15-19"}]}
    site = cli.site_config.parse(raw)
    reads: list[bool | None] = [True]
    monkeypatch.setattr(cli, "_slurmd_reads_layouts", lambda: reads[0])
    assert cli.core_kind_refusals(site, halves, "n1") == []
    shown = capsys.readouterr().out
    assert "fast_cpus 5-9,15-19 are this host's fastest cores" in shown and "slurmd is built with hwloc" in shown
    reads[0] = False  # a slurm built without hwloc ignores the layout and would bind the GPU to the wrong cores
    (no_hwloc,) = cli.core_kind_refusals(site, halves, "n1")
    assert "built with hwloc" in no_hwloc
    reads[0] = True
    (elsewhere,) = cli.core_kind_refusals(site, halves, "other")  # the layout is installed for this machine's slurmd
    assert "set on node n1, but this is other" in elsewhere
    monkeypatch.setattr(cli, "_this_host", lambda: "other")
    monkeypatch.setattr(cli, "_package_refusals", lambda *_: [])
    tools = cli.components.load(site, ["tools"])  # a part-time machine: no layout file is installed there
    assert cli.platform_refusals(site, tools) == []
    assert cli.platform_refusals(site, [*tools, cli.slurm_component]) == [elsewhere]
    binned = cli.cores.CoreSpeeds({**halves.khz, 19: 4000000})  # a favoured core inside the fast kind still passes
    assert cli.core_kind_refusals(site, binned, "n1") == []
    raw["nodes"][0]["fast_cpus"] = "0-4,10-14"  # the slow kind
    (wrong,) = cli.core_kind_refusals(cli.site_config.parse(raw), halves, "n1")
    assert "are not all faster than its other cores (2808 MHz: 0-4,10-14; 3900 MHz: 5-9,15-19" in wrong
    alike = cli.cores.CoreSpeeds(dict.fromkeys(range(20), 3900000))
    assert "not all faster" in cli.core_kind_refusals(cli.site_config.parse(raw), alike, "n1")[0]
    del raw["nodes"][0]["fast_cpus"]
    unset = cli.site_config.parse(raw)
    (hint,) = cli.core_kind_warnings(unset, halves, "n1")
    assert 'fast_cpus = "5-9,15-19"' in hint
    assert cli.core_kind_warnings(unset, alike, "n1") == []
