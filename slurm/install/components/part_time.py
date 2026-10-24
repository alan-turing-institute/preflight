"""A part-time machine: someone's own computer that joins the cluster as a configless compute node (plan B5).

Not a component the installer loads by name (`components.ORDER` does not list it): `install.py part-time add NODE` runs
on the controller, and `install.py apply --part-time NODE` runs on the machine itself. Every command goes through
`components.run`, so a test swaps the one runner; the key travels through `send`, which a test swaps too.
"""

from __future__ import annotations

import grp
import os
import pwd
import re
import shlex
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import render

import components

SLURM_USER = "slurm"
SSH = ["ssh", "-o", "BatchMode=yes"]
KNOWN_HOSTS = str(Path(render.COPY_KEY).parent / "known_hosts")  # under the state dir; the agent's ssh trusts only this
OWNERS_DIR = "owners"  # under the state dir of each part-time machine: node_check.py (root) writes its manifest there
MUNGE_KEY = "/etc/munge/munge.key"
SLURMD_DROP_IN = "/etc/systemd/system/slurmd.service.d/10-conf-server.conf"
#: The controller's slurm.conf (metrics on) names `Prolog=/etc/slurm/prolog.d/*.sh` and the same for Epilog; configless
#: slurmd does not ship those scripts, and slurm 26.05 fails every job on a node where the glob matches nothing (drains
#: the node, holds the job). A script that does nothing makes it match.
NOOP_HOOK = "#!/bin/sh\n# part-time node: nothing to run here, but slurm needs one script in the directory\nexit 0\n"
HOOK_DIRS = ("/etc/slurm/prolog.d", "/etc/slurm/epilog.d")
HOOK_NAME = "00-none.sh"
SLURMD_PORT = 6818
PASSWD_UID, PASSWD_GID = 2, 3  # fields of a `getent passwd` line
NOT_FOUND = 2  # `getent` exit status for a missing entry, passed through by ssh
# sshd runs a key's forced command through the user's login shell (`-c`), and nologin refuses it: `slurm` needs a real
# shell on a part-time machine for the read-only copy-back of job state (its password stays locked, B5b)
SLURM_HOME, SLURM_SHELL = "/var/lib/slurm", "/bin/sh"
USR_BIN_RRSYNC, LOCAL_RRSYNC = "/usr/bin/rrsync", "/usr/local/bin/rrsync"
DOC_RRSYNC = "/usr/share/doc/rsync/support/rrsync"  # where Fedora ships it, mode 0644

TRUST_WARNING = """\
This makes this machine part of the cluster, and that is a trust decision.
  - The cluster's accounts are authenticated by a shared munge key; anyone who is root here can use it to act as ANY
    user of the whole cluster, including its admins, and run anything as them.
  - The controller can run any job on this machine, as any cluster user.
Only do this on a machine whose root you are willing to give that power to, and only for a cluster whose admins you
trust with this machine. Re-run with --i-accept-cluster-trust to go on."""


class PartTimeAction(StrEnum):
    ADD = "add"


def send(argv: list[str], data: bytes) -> subprocess.CompletedProcess[bytes]:
    """Run `argv` with `data` on its standard input (the munge key; never echoed, and not part of any argv)."""
    return subprocess.run(argv, input=data, capture_output=True, check=False)


def _as_admin(argv: list[str]) -> list[str]:
    """Run as the admin who called sudo, so their ssh keys and agent apply; root has none of them."""
    admin = os.environ.get("SUDO_USER")
    return ["sudo", "-u", admin, *argv] if admin and os.geteuid() == 0 else argv


def _on_node(address: str, *remote: str) -> list[str]:
    """`remote` run on the node over ssh, as the admin. ssh hands the words to the node's shell as one string, so each
    is quoted: a user name from accounting reaches here, and must stay one word."""
    return _as_admin([*SSH, address, *(shlex.quote(word) for word in remote)])


def _ids(argv: list[str]) -> tuple[int, int] | None:
    """(uid, gid) from a `getent passwd` run, None for no such user; any other failure (ssh itself) stops."""
    done = components.run(argv)
    if done.returncode == NOT_FOUND:
        return None
    if done.returncode != 0:
        raise SystemExit(f"FAILED ({done.returncode}): {' '.join(argv)}\n{done.stderr or ''}".rstrip())
    fields = (done.stdout or "").strip().split(":")
    return int(fields[PASSWD_UID]), int(fields[PASSWD_GID])


def site_users(site) -> list[str]:
    """Who the installer manages: the accounting users of the site's account, the admin group, and `slurm`."""
    done = components.run(["sacctmgr", "-n", "-P", "show", "assoc", "where", f"cluster={site.cluster_name}",
                           f"account={site.account}", "format=user"])
    try:
        admins = list(grp.getgrnam(site.admin_group).gr_mem)
    except KeyError:
        admins = []
    return sorted({*(done.stdout or "").split(), *admins, SLURM_USER})


def uid_mismatches(users: list[str], address: str) -> list[str]:
    """One line per user whose uid differs on the node, or who is missing there (only `slurm` may be: the node install
    creates it). A user the controller itself does not have is not compared."""
    out = []
    for user in users:
        here = _ids(["getent", "passwd", user])
        if here is None:
            continue
        there = _ids(_on_node(address, "getent", "passwd", user))
        if there is None and user != SLURM_USER:
            out.append(f"{user}: uid {here[0]} here, missing on {address}")
        elif there is not None and there[0] != here[0]:
            out.append(f"{user}: uid {here[0]} here, {there[0]} on {address}")
    return out


def cmd_add(site, args) -> int:
    node = next((n for n in site.part_time_nodes if n.name == args.node), None)
    if node is None:
        print(f"REFUSED: {args.node} is not a part_time node in the site file", file=sys.stderr)
        return 1
    if problems := uid_mismatches(site_users(site), node.address):
        print(f"REFUSED: user ids differ on {node.name} ({node.address}); jobs would run as the wrong person. "
              "Make them match, then run this again:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1
    host_key = "" if site.shared_state else _host_key(node)  # before anything is changed: no key, no join
    key = (components.ROOT / MUNGE_KEY.lstrip("/")).read_bytes()
    install = ["install", "-m", "0400", "-o", "munge", "-g", "munge", "/dev/stdin", MUNGE_KEY]
    done = send(_on_node(node.address, "sudo", "-n", *install), key)
    if done.returncode != 0:
        raise SystemExit(f"FAILED ({done.returncode}): copying the munge key to {node.address}\n"
                         f"{(done.stderr or b'').decode(errors='replace')}".rstrip())
    # a munge already running holds the key it started with
    # restart, not try-restart: on Fedora munge has no key until now, so neither unit has started yet
    components.run(_on_node(node.address, "sudo", "-n", "systemctl", "restart", "munge", "slurmd"))
    print(f"{node.name}: user ids match and the munge key is copied (not shown)")
    if not site.shared_state:
        _authorize_copy_back(site, node, host_key)
    return 0


def authorized_line(rrsync: str, state_dir: str, public_key: str) -> str:
    """The node's `~slurm/.ssh/authorized_keys` entry: the key may only run a read-only rsync of the state directory."""
    return f'command="{rrsync} -ro {state_dir}",restrict {public_key.strip()}'


def _host_key(node) -> str:
    """The node's host key lines from the invoking admin's own `known_hosts` (they have ssh'd there once, which is how
    they checked the key): the agent's ssh trusts exactly these and never a key on first use."""
    admin = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    known = Path(pwd.getpwnam(admin).pw_dir) / ".ssh" / "known_hosts"
    done = components.run(_as_admin(["ssh-keygen", "-F", node.address, "-f", str(known)]))
    lines = [line for line in (done.stdout or "").splitlines() if line.strip() and not line.startswith("#")]
    if done.returncode != 0 or not lines:
        raise SystemExit(f"REFUSED: no host key for {node.address} in {known}; ssh to {node.address} once as yourself "
                         "first (check its fingerprint), then run this again")
    return "\n".join(lines) + "\n"


def _authorize_copy_back(site, node, host_key: str) -> None:
    """Make SlurmUser's copy-back key here if absent, and let it read the node's state directory (and nothing else)."""
    key = Path(site.state_dir) / render.COPY_KEY
    run = components.run
    for argv in [["install", "-d", "-m", "0700", "-o", render.SLURM_USER, "-g", render.SLURM_USER, str(key.parent)],
                 *([] if (components.ROOT / str(key).lstrip("/")).exists() else
                   [["runuser", "-u", render.SLURM_USER, "--", "ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
                     "gpujobs-copy-back", "-f", str(key)]])]:
        if (done := run(argv)).returncode != 0:
            raise SystemExit(f"FAILED ({done.returncode}): {' '.join(argv)}\n{done.stderr or ''}".rstrip())
    path = Path(site.state_dir) / KNOWN_HOSTS
    done = send(["install", "-m", "0644", "-o", SLURM_USER, "-g", SLURM_USER, "/dev/stdin", str(path)], host_key.encode())
    if done.returncode != 0:
        raise SystemExit(f"FAILED ({done.returncode}): writing {path}\n{(done.stderr or b'').decode(errors='replace')}".rstrip())
    public = (components.ROOT / f"{key}.pub".lstrip("/")).read_text()
    # Ubuntu ships rrsync in /usr/bin; elsewhere `apply --part-time` put it in /usr/local/bin
    has_usr_bin = run(_on_node(node.address, "test", "-x", USR_BIN_RRSYNC)).returncode == 0
    line = authorized_line(USR_BIN_RRSYNC if has_usr_bin else LOCAL_RRSYNC, str(site.state_dir), public)
    install = ["install", "-m", "0600", "-o", SLURM_USER, "-g", SLURM_USER, "/dev/stdin", f"{SLURM_HOME}/.ssh/authorized_keys"]
    done = send(_on_node(node.address, "sudo", "-n", *install), f"{line}\n".encode())
    if done.returncode != 0:
        raise SystemExit(f"FAILED ({done.returncode}): installing the copy-back key on {node.address}\n"
                         f"{(done.stderr or b'').decode(errors='replace')}".rstrip())
    print(f"{node.name}: the agent may copy its job state, read-only (key {key}.pub)")


def _group_user_commands(uid: int, gid: int) -> list[list[str]]:
    """`slurm` with the controller's ids, made before the packages so their scripts do not pick others."""
    out = []
    if _ids(["getent", "passwd", SLURM_USER]) is None:
        if components.run(["getent", "group", SLURM_USER]).returncode != 0:
            out.append(["groupadd", "-g", str(gid), SLURM_USER])
        out.append(["useradd", "-r", "-u", str(uid), "-g", SLURM_USER, "-d", SLURM_HOME, "-s", SLURM_SHELL, SLURM_USER])
    else:
        out.append(["usermod", "-d", SLURM_HOME, "-s", SLURM_SHELL, SLURM_USER])
    return out


def refusals(site, args) -> list[str]:
    out = []
    if not any(n.name == args.part_time for n in site.part_time_nodes):
        out.append(f"{args.part_time} is not a part_time node in the site file")
    if not site.slurm_build:
        out.append("the site file has no [slurm] table: a part-time machine always runs the pinned build")
    for flag in ("slurm_uid", "slurm_gid", "controller"):
        if getattr(args, flag) is None:
            out.append(f"--{flag.replace('_', '-')} is needed (the controller's `slurm` ids from `getent passwd slurm`, "
                       "and the controller's address)")
    if out:
        return out
    existing = _ids(["getent", "passwd", SLURM_USER])
    if existing and existing != (args.slurm_uid, args.slurm_gid):
        out.append(f"a `slurm` user already exists here with uid/gid {existing[0]}/{existing[1]}, not the controller's "
                   f"{args.slurm_uid}/{args.slurm_gid}")
    return out


def slurmd_drop_in(unit: str, controller: str) -> str:
    """The unit's own ExecStart with `--conf-server` added after the program; read from the installed package."""
    found = re.search(r"^ExecStart=(\S+)(.*)$", unit, re.MULTILINE)
    if not found:
        raise SystemExit("`systemctl cat slurmd` shows no ExecStart=; cannot add --conf-server to it")
    if "--conf-server" in found[2]:
        raise SystemExit("slurmd's unit already has --conf-server; leave it, or remove it first")
    # After a reboot slurmd may start before the network or the VPN that reaches the controller is up, and then it cannot
    # fetch its settings and exits: wait for them (tailscaled.service is ignored when it does not exist), and start again
    return (f"[Unit]\nAfter=network-online.target tailscaled.service\nWants=network-online.target\n\n"
            f"[Service]\nExecStart=\nExecStart={found[1]} --conf-server {controller}{found[2]}\nRestart=on-failure\nRestartSec=30\n")


def _write_drop_in(controller: str) -> list[list[str]]:
    done = components.run(["systemctl", "cat", "slurmd"])
    if done.returncode != 0:
        raise SystemExit(f"FAILED ({done.returncode}): systemctl cat slurmd\n{done.stderr or ''}".rstrip())
    path = components.ROOT / SLURMD_DROP_IN.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(slurmd_drop_in(done.stdout or "", controller))
    return [["systemctl", "daemon-reload"]]


def _write_noop_hooks() -> list[list[str]]:
    for directory in HOOK_DIRS:
        path = components.ROOT / directory.lstrip("/") / HOOK_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(NOOP_HOOK)
        path.chmod(0o755)
    return []


def _enable_services() -> list[list[str]]:
    """Enable munge and slurmd for every boot; start them now only when the controller's key is already here (Fedora's
    munge makes none, and `part-time add` copies the key and starts both)."""
    if (components.ROOT / MUNGE_KEY.lstrip("/")).exists():
        return [["systemctl", "enable", "--now", "munge", "slurmd"]]
    print("munge has no key yet: munge and slurmd are enabled and start when `part-time add` has copied the controller's key")
    return [["systemctl", "enable", "munge", "slurmd"]]


def _after(action) -> list[list[str]]:
    """Run a step that does its own work, then add no commands."""
    action()
    return []


def _firewall(controller: str) -> list[list[str]]:
    if components.run(["systemctl", "is-active", "firewalld"]).returncode != 0:
        return []
    rule = f"rule family=ipv4 source address={controller} port port={SLURMD_PORT} protocol=tcp accept"
    return [["firewall-cmd", "--permanent", f"--add-rich-rule={rule}"], ["firewall-cmd", "--reload"]]


def _copy_back_node(site) -> list[list[str]]:
    """What the controller's read-only rsync needs here: rrsync, and a home for `slurm` with its `.ssh` (B5b)."""
    if site.shared_state:
        return []
    ssh = f"{SLURM_HOME}/.ssh"
    own = ["-o", SLURM_USER, "-g", SLURM_USER]
    return [
        *([] if (components.ROOT / USR_BIN_RRSYNC.lstrip("/")).exists() else [["install", "-m", "0755", DOC_RRSYNC, LOCAL_RRSYNC]]),
        ["install", "-d", "-m", "0755", *own, SLURM_HOME],
        ["install", "-d", "-m", "0700", *own, ssh],
        # root's alone: node_check.py lists each state file's owner here, and the controller trusts that list
        ["install", "-d", "-m", "0755", "-o", "root", "-g", "root", f"{site.state_dir}/{OWNERS_DIR}"],
    ]


def steps(site, args, manager: str, tools_apply, site_copy_apply) -> list:
    """The install, in order, for `_run_all`: a function in the list gives its argvs when reached. `tools_apply` is
    called for the tools component and `site_copy_apply` writes `/etc/gpujobs/site.toml` (each writes files itself)."""
    from . import agent as agent_component
    from . import slurm as slurm_component

    return [
        *_group_user_commands(args.slurm_uid, args.slurm_gid),
        [manager, "install", "-y", "munge", *([] if site.shared_state else ["rsync"])],
        lambda: slurm_component.install_release(*site.slurm_build),
        lambda: _after(tools_apply),
        # a job's wrapper reads the site file and writes its records here, and the controller copies them back
        lambda: _after(site_copy_apply),
        *agent_component.state_dir_commands(site),
        lambda: _copy_back_node(site),
        lambda: _write_drop_in(args.controller),
        _write_noop_hooks,
        lambda: _firewall(args.controller),
        _enable_services,
    ]

