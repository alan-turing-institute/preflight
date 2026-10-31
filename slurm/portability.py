"""Can a job written for this machine run on a part-time machine, and still be there at launch?

Searched before writing (2026-10-31): `slurm/remote.py` (rsync to another SITE; probes nothing), `dgx/assert_code_parity.py`
(a code-parity checksum against one box for a gated entrypoint; a plain command has no entrypoint, and code that is
not on shared storage is excluded here anyway), `dgx/dgx_submit.sh` and `slurm/agent.py` (ssh with `BatchMode=yes`:
the same options are used below), `slurm/node_check.py` (`exclusion()` and `has_other_node()`: used for the node
exclusion, not here); grep of `slurm/` and `dgx/` for ELF headers, `mountinfo` and `findmnt` found no reader to reuse.

One stdlib-only function, `probe(paths)`, answers for the machine it runs on: its architecture and, per path, whether it
exists, the architecture of the ELF there (or of its `#!` interpreter, one level), and which network filesystem holds it
(`where`: the export path of the mount's source, the path inside it and the file's inode, so equal `where` on both
machines means one file; the server's host is ignored on purpose, because one NAS is `nas.example:/SharedData` on one
machine and its Tailscale address `100.x.y.z:/SharedData` on another. Residual risk: two different servers exporting the
same path with the same inode would look alike). It runs
locally for this side, over ssh for the node (this file is piped to the node's interpreter, so the node never runs a
stale copy) and in the job wrapper at launch (`launch_problems`). Unknown (ssh down, output unreadable) is excluded.

The rules (plan `plan notes`, A2-A5, A8, A9):
the working folder must be shared (a checkpoint written elsewhere is left behind when the job is pushed back); the
program must exist on the node, be built for the node's architecture, and be a system file or shared; every other named
path must be shared or a system file, and exist on the node.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

#: A line of the probe's output starting with this carries the JSON answer: the laptop's shell prints warnings too.
SENTINEL = "PORTABILITY "
#: Directories every machine of the same distribution family has its own copy of.
SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64")
NETWORK_FILESYSTEMS = frozenset({"nfs", "nfs4", "cifs", "smb3"})
#: ELF `e_machine` values, by name `platform.machine()` uses.
ELF_MACHINES = {3: "i686", 40: "arm", 62: "x86_64", 183: "aarch64", 243: "riscv64"}
HEADER_BYTES = 256
ELF_FIXED_BYTES = 20  # up to and including e_machine
ELF_LITTLE_ENDIAN = 1  # EI_DATA, byte 5
SSH_CONNECT_S = 5
SSH_TOTAL_S = 30


class Role(StrEnum):
    """What a named path is to the job; decides which rule judges it."""

    WORKDIR = "workdir"
    PROGRAM = "program"
    FILE = "file"
    TOOLS = "tools"  # the folder the job script runs `job_wrapper.py` from: the node's copy must be this machine's


def _unescape(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field)


def read_mounts(text: str) -> list[tuple[str, str, str]]:
    """(mount point, filesystem type, source) for each line of `/proc/self/mountinfo`, in file order."""
    found = []
    for line in text.splitlines():
        fields = line.split()
        if "-" in fields[6:]:
            dash = fields.index("-", 6)
            found.append((_unescape(fields[4]), fields[dash + 1], _unescape(fields[dash + 2])))
    return found


def _elf(head: bytes) -> str | None:
    if len(head) < ELF_FIXED_BYTES or head[:4] != b"\x7fELF":
        return None
    number = int.from_bytes(head[18:ELF_FIXED_BYTES], "little" if head[5] == ELF_LITTLE_ENDIAN else "big")
    return ELF_MACHINES.get(number, f"machine {number}")


def _head(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(HEADER_BYTES)
    except OSError:
        return b""


def _export(source: str) -> str:
    """The export path of a mount source, without the server: `host:/SharedData` and `//host/SharedData` are both `/SharedData`."""
    if source.startswith("//"):
        return "/" + source[2:].partition("/")[2]
    return "/" + source.partition(":/")[2]


def _where(path: str, mounts: Sequence[tuple[str, str, str]]) -> str:
    """`export/path-inside-it#inode` of the network mount holding `path` (the server is ignored, see the module
    docstring), "" when the innermost mount is not a network one."""
    real = os.path.realpath(path)
    best = None
    for point, kind, source in mounts:  # later lines are mounted on top: the last of the longest wins
        if (real == point or real.startswith(point.rstrip("/") + "/")) and (best is None or len(point) >= len(best[0])):
            best = (point, kind, source)
    if best is None or best[1] not in NETWORK_FILESYSTEMS:
        return ""
    inside = os.path.relpath(real, best[0])
    return _export(best[2]).rstrip("/") + ("" if inside == "." else "/" + inside) + f"#{os.stat(real).st_ino}"


def tools_digest(folder: str) -> str:
    """sha256 over the sorted (name, bytes) of the folder's `*.py` files."""
    digest = hashlib.sha256()
    for path in sorted(Path(folder).glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def probe(paths: Iterable[str], mountinfo: str | None = None) -> dict[str, object]:
    """What this machine says about `paths`: {"machine": arch, "paths": {path: info}}. Each path is `os.stat`ed first,
    because a mount that is not yet mounted (autofs) shows as `autofs` until something touches it."""
    paths = list(paths)
    for path in paths:
        with contextlib.suppress(OSError):
            os.stat(path)
    if mountinfo is None:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
    mounts = read_mounts(mountinfo)
    found: dict[str, dict[str, object]] = {}
    for path in paths:
        if not os.path.exists(path):
            found[path] = {"exists": False}
            continue
        head = _head(path)
        info: dict[str, object] = {"exists": True, "elf": _elf(head), "where": _where(path, mounts)}
        if os.path.isdir(path):
            info["digest"] = tools_digest(path)
        if head.startswith(b"#!"):
            words = head[2:].split(b"\n", 1)[0].split()
            if words:
                interp = words[0].decode(errors="replace")
                info.update(interp=interp, interp_elf=_elf(_head(interp)))
        found[path] = info
    return {"machine": platform.machine(), "paths": found}


def parse_probe(text: str) -> dict[str, object] | None:
    """The probe's answer in `text` (the last sentinel line), None when there is none or it is not a probe's."""
    lines = [line[len(SENTINEL):] for line in text.splitlines() if line.startswith(SENTINEL)]
    try:
        answer = json.loads(lines[-1])
    except (IndexError, ValueError):
        return None
    if isinstance(answer, dict) and isinstance(answer.get("machine"), str) and isinstance(answer.get("paths"), dict):
        return answer
    return None


def ask_over_ssh(address: str, python: str, paths: Sequence[str], run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, object] | None:
    """`probe(paths)` as the node at `address` answers it, running this file with its `python`; None when it cannot be
    asked or its answer is unreadable."""
    remote = shlex.join([python, "-", json.dumps(list(paths))])
    command = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_S}", address, remote]
    try:
        done = run(command, input=Path(__file__).read_text(encoding="utf-8"), capture_output=True, text=True,
                   timeout=SSH_TOTAL_S, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_probe(done.stdout) if done.returncode == 0 else None


@dataclass(frozen=True)
class Need:
    role: Role
    path: str


def _under_system(path: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in SYSTEM_ROOTS)


def named_paths(command: Sequence[str], workdir: str | Path, repo: str | Path | None = None) -> list[Need]:
    """The paths a job depends on: its working folder, the program (command[0], searched on PATH when bare), `repo`,
    and every other token that is an absolute path or exists relative to the working folder (`--opt=PATH` counts as
    PATH). A token naming a file that does not exist yet stands for its nearest existing folder."""
    work = os.path.abspath(workdir)
    program = shutil.which(command[0]) if os.sep not in command[0] else None
    needs = [Need(Role.WORKDIR, work), Need(Role.PROGRAM, os.path.abspath(os.path.join(work, program or command[0])))]
    extra = [str(repo)] if repo else []
    for token in command[1:]:
        value = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
        if value and (os.path.isabs(value) or os.path.exists(os.path.join(work, value))):
            extra.append(value)
    for value in extra:
        path = os.path.abspath(os.path.join(work, value))
        while not os.path.exists(path) and path != os.sep:
            path = os.path.dirname(path)
        if path != os.sep:
            needs.append(Need(Role.FILE, path))
    seen: set[str] = set()
    return [n for n in needs if not (n.path in seen or seen.add(n.path))]


def _reason(need: Need, node: str, machine: str, here: Mapping[str, object], there: Mapping[str, object]) -> str | None:
    """Why `need` rules `node` out, None when it does not. `here` / `there`: the probe's info for the path, each side."""
    path = need.path
    if not here.get("exists"):
        return f"{path} does not exist on this machine"
    if need.role is Role.TOOLS:
        if there.get("exists") and there.get("digest") == here.get("digest"):
            return None
        return f"its copy of the job tools in {path} differs from this machine's: reinstall the tools there"
    shared = bool(here.get("where")) and here.get("where") == there.get("where")
    if need.role is Role.WORKDIR:
        if shared:
            return None
        return (f"working folder {path} is not on storage both machines see, so a checkpoint written there would be "
                f"left behind on {node} when the job is pushed back")
    if not (shared or _under_system(path)):
        return f"{path} lives on this machine's own disk, which {node} does not see"
    if not there.get("exists"):
        return f"{path} does not exist on {node}"
    built = there.get("elf") or there.get("interp_elf")
    if need.role is Role.PROGRAM and built != machine:
        return f"{path} is built for {built or 'no known architecture'}; {node} is {machine}"
    return None


@dataclass(frozen=True)
class Verdict:
    node: str
    reason: str | None  # None: included

    def line(self) -> str:
        return f"[submit] part_time {self.node}: " + ("included" if self.reason is None else f"excluded: {self.reason}")


@dataclass(frozen=True)
class Decision:
    verdicts: tuple[Verdict, ...]
    requirements: tuple[tuple[str, str, str], ...]  # (role, path, where it was seen) for the launch re-check

    @property
    def included(self) -> list[str]:
        return [v.node for v in self.verdicts if v.reason is None]

    @property
    def excluded(self) -> list[str]:
        return [v.node for v in self.verdicts if v.reason is not None]


def decide(
    nodes: Iterable[object], *, cpus: int, memory_mb: int | None, needs: Sequence[Need], check_paths: bool, python: str,
    ask: Callable[[str, str, Sequence[str]], dict[str, object] | None] = ask_over_ssh,
    here: Callable[[Sequence[str]], dict[str, object]] = probe,
) -> Decision:
    """Judge each part-time node (`site_config.Node`) for a job asking `cpus` cores and `memory_mb` (slurm MB; None:
    slurm's default, 0: all of a machine's memory). In order, the first failure excludes: size, then (when
    `check_paths`; a submitter's vouching skips it) reachability and the named paths."""
    paths = [n.path for n in needs]
    local = here(paths)["paths"] if check_paths else {}
    verdicts: list[Verdict] = []
    for node in nodes:
        reason = None
        if cpus > node.cores_lent:
            reason = f"asks {cpus} cores, it lends {node.cores_lent}"
        elif memory_mb is not None and (memory_mb == 0 or memory_mb > node.jobs_mb):
            asked = "all of a machine's memory" if memory_mb == 0 else f"{memory_mb * 1024**2 / 1e9:.1f} GB"
            reason = f"asks {asked}, it lends {node.jobs_mb * 1024**2 / 1e9:.1f} GB"
        elif check_paths:
            there = ask(node.address, python, paths)
            if there is None:
                reason = f"cannot ask {node.name} (ssh to {node.address} failed or its answer was unreadable)"
            else:
                reason = next((r for need in needs if (r := _reason(
                    need, node.name, str(there["machine"]), local.get(need.path, {}), there["paths"].get(need.path, {})))), None)
        verdicts.append(Verdict(node.name, reason))
    required = tuple((n.role.value, n.path, str(local.get(n.path, {}).get("where") or "")) for n in needs if n.role is not Role.TOOLS) if check_paths else ()
    return Decision(tuple(verdicts), required)


def launch_problems(requirements: Iterable[Sequence[str]], found: dict[str, object] | None = None) -> list[str]:
    """What no longer holds on THIS machine for the job's `requirements` (decide()'s): missing paths, a program for
    another architecture, a shared path that is no longer the one seen at submission. [] means the job may start."""
    requirements = [tuple(r) for r in requirements]
    found = found or probe([path for _, path, _ in requirements])
    problems = []
    for role, path, where in requirements:
        info = found["paths"].get(path, {})
        if not info.get("exists"):
            problems.append(f"{path} does not exist here")
        elif role == Role.PROGRAM and (info.get("elf") or info.get("interp_elf")) != found["machine"]:
            problems.append(f"{path} is not built for {found['machine']}")
        elif where and info.get("where") != where:
            problems.append(f"{path} is not the shared copy seen at submission ({where})")
    return problems


if __name__ == "__main__":  # piped to a node: `python3 - '["/path", ...]'`
    print(SENTINEL + json.dumps(probe(json.loads(sys.argv[1]))))
