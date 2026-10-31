"""portability.py: which part-time nodes a job may use, from what each machine says about its named paths."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import portability
import pytest
from portability import Need, Role

NAS = "nas:/share"
ELF_X86 = b"\x7fELF\x02\x01\x01" + bytes(11) + (62).to_bytes(2, "little") + bytes(40)
ELF_ARM = ELF_X86[:18] + (183).to_bytes(2, "little") + bytes(40)


def seen(exists: bool = True, elf: str | None = None, where: str = "", **more: object) -> dict:
    return {"exists": exists, "elf": elf, "where": where, **more}


def machine(arch: str, **paths: dict) -> dict:
    return {"machine": arch, "paths": paths}


def node(name: str = "fedora", cores: int = 8, mb: int = 11444) -> SimpleNamespace:
    return SimpleNamespace(name=name, address=f"{name}.lan", cores_lent=cores, jobs_mb=mb)


def judge(needs: list[Need], here: dict, there: dict | None, *, cpus: int = 4, mem: int | None = 4096, nodes=None):
    """`decide` with this machine's answer `here` and the node's `there` (None: ssh failed)."""
    return portability.decide(
        nodes or [node()], cpus=cpus, memory_mb=mem, needs=needs, check_paths=True, python="/usr/bin/python3",
        ask=lambda *_a: there, here=lambda _p: machine("aarch64", **here),
    )


WORK, PYTHON, VENV = "/mnt/shared/Container/w", "/usr/bin/python3", "/home/u/proj/.venv/bin/python"
GOOD_HERE = {WORK: seen(where=f"{NAS}/w"), PYTHON: seen(elf="aarch64")}
GOOD_THERE = machine("x86_64", **{WORK: seen(where=f"{NAS}/w"), PYTHON: seen(elf="x86_64")})
SYSTEM_JOB = [Need(Role.WORKDIR, WORK), Need(Role.PROGRAM, PYTHON)]


def reason(decision: portability.Decision) -> str | None:
    return decision.verdicts[0].reason


def test_a_system_program_with_a_shared_workdir_is_included_and_its_requirements_kept() -> None:
    decision = judge(SYSTEM_JOB, GOOD_HERE, GOOD_THERE)
    assert decision.included == ["fedora"] and reason(decision) is None
    assert decision.requirements == (("workdir", WORK, f"{NAS}/w"), ("program", PYTHON, ""))  # what the wrapper re-checks
    assert decision.verdicts[0].line() == "[submit] part_time fedora: included"


@pytest.mark.parametrize(
    ("needs", "here", "there", "fragment"),
    [
        # today's case: the project's aarch64 venv lives on this machine's own disk
        ([Need(Role.WORKDIR, WORK), Need(Role.PROGRAM, VENV)], {**GOOD_HERE, VENV: seen(elf="aarch64")}, GOOD_THERE,
         f"{VENV} lives on this machine's own disk"),
        # a program on shared storage, but built for the other architecture
        ([Need(Role.WORKDIR, WORK), Need(Role.PROGRAM, f"{WORK}/run")], {**GOOD_HERE, f"{WORK}/run": seen(elf="aarch64", where=f"{NAS}/w/run")},
         machine("x86_64", **{WORK: seen(where=f"{NAS}/w"), f"{WORK}/run": seen(elf="aarch64", where=f"{NAS}/w/run")}),
         "built for aarch64; fedora is x86_64"),
        # a working folder only this machine has
        ([Need(Role.WORKDIR, "/home/u/w"), Need(Role.PROGRAM, PYTHON)], {"/home/u/w": seen(), PYTHON: seen(elf="aarch64")},
         machine("x86_64", **{"/home/u/w": seen(), PYTHON: seen(elf="x86_64")}), "checkpoint written there would be left behind"),
        # /mnt/shared/service-c: a network share here, a plain local folder there
        (SYSTEM_JOB, GOOD_HERE, machine("x86_64", **{WORK: seen(where=""), PYTHON: seen(elf="x86_64")}), "not on storage both machines see"),
        # a file a system path names that the node does not have
        ([*SYSTEM_JOB, Need(Role.FILE, "/usr/bin/tool")], {**GOOD_HERE, "/usr/bin/tool": seen()}, GOOD_THERE, "/usr/bin/tool does not exist on fedora"),
        # a path this machine does not have either
        (SYSTEM_JOB, {WORK: seen(exists=False)}, GOOD_THERE, f"{WORK} does not exist on this machine"),
        # the node cannot be asked (ssh down, or an answer that is not a probe's)
        (SYSTEM_JOB, GOOD_HERE, None, "cannot ask fedora"),
    ],
    ids=["venv", "wrong architecture", "local workdir", "local share", "missing on node", "missing here", "ssh failure"],
)
def test_a_node_is_excluded_with_the_reason(needs: list[Need], here: dict, there: dict | None, fragment: str) -> None:
    decision = judge(needs, here, there)
    assert decision.included == [] and fragment in reason(decision)
    assert decision.verdicts[0].line().startswith("[submit] part_time fedora: excluded: ")


def test_size_is_judged_first_without_asking_the_node_and_one_of_two_nodes_can_fail() -> None:
    def never(*_a: object) -> None:
        raise AssertionError("asked a node that cannot take the job")

    for kwargs, fragment in [({"cpus": 16}, "asks 16 cores, it lends 8"), ({"mem": 20000}, "asks 21.0 GB, it lends 12.0 GB"),
                             ({"mem": 0}, "asks all of a machine's memory")]:
        decision = portability.decide([node()], cpus=kwargs.get("cpus", 4), memory_mb=kwargs.get("mem", 4096), needs=SYSTEM_JOB,
                                      check_paths=True, python="x", ask=never, here=lambda _p: machine("aarch64", **GOOD_HERE))
        assert fragment in reason(decision)
    two = judge(SYSTEM_JOB, GOOD_HERE, GOOD_THERE, nodes=[node("fedora"), node("lap", cores=2)])
    assert (two.included, two.excluded) == (["fedora"], ["lap"])  # submit.py bars the second by name
    vouched = portability.decide([node()], cpus=4, memory_mb=4096, needs=SYSTEM_JOB, check_paths=False, python="x", ask=never)
    assert vouched.included == ["fedora"] and vouched.requirements == ()  # nothing asked, nothing to re-check


def test_the_probe_reads_elf_headers_shebangs_and_network_mounts(tmp_path: Path) -> None:
    (tmp_path / "x86").write_bytes(ELF_X86)
    (tmp_path / "arm").write_bytes(ELF_ARM)
    (tmp_path / "script").write_bytes(f"#!{tmp_path}/arm\nprint(1)\n".encode())
    found = portability.probe([str(tmp_path / n) for n in ("x86", "arm", "script", "nothing")], mountinfo="")["paths"]
    assert [found[str(tmp_path / n)].get("elf") for n in ("x86", "arm", "script")] == ["x86_64", "aarch64", None]
    assert found[str(tmp_path / "script")]["interp_elf"] == "aarch64" and found[str(tmp_path / "nothing")] == {"exists": False}
    mounts = (f"1 0 8:1 / / rw - ext4 /dev/sda1 rw\n2 1 0:50 / {tmp_path} rw - autofs systemd-1 rw\n"
              f"3 2 0:51 / {tmp_path} rw - nfs4 nas:/a\\040b rw\n")  # the automount, then the real mount on top
    inode = (tmp_path / "x86").stat().st_ino
    assert portability.probe([str(tmp_path / "x86")], mountinfo=mounts)["paths"][str(tmp_path / "x86")]["where"] == f"/a b/x86#{inode}"
    assert portability.probe([str(tmp_path / "x86")], mountinfo=mounts.splitlines()[0])["paths"][str(tmp_path / "x86")]["where"] == ""


def test_named_paths_take_the_workdir_program_repo_and_path_arguments(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    needs = portability.named_paths(["python3", "train.py", "--out=/nonexistent/dir/out.json", f"--in={tmp_path}/data", "5", "/health"],
                                    tmp_path, repo=tmp_path / "data")
    paths = {n.path: n.role for n in needs}
    assert paths[str(tmp_path)] is Role.WORKDIR and paths[portability.shutil.which("python3")] is Role.PROGRAM
    assert str(tmp_path / "data") in paths and len(needs) == len(paths)  # the repo and --in= name one folder: listed once
    assert "/nonexistent/dir/out.json" not in paths and not any(p in ("/", "/health") for p in paths)  # nothing exists up to /


def test_a_probe_answer_is_read_from_its_sentinel_line_and_anything_else_is_nothing() -> None:
    answer = machine("x86_64", **{"/a": seen()})
    noisy = f"warning: motd\n{portability.SENTINEL}{json.dumps(answer)}\n"
    assert portability.parse_probe(noisy) == answer
    for garbage in ("", "x86_64\n", portability.SENTINEL + "{", portability.SENTINEL + "[]", portability.SENTINEL + '{"machine": 1}'):
        assert portability.parse_probe(garbage) is None

    def fake_run(stdout: str, code: int = 0):
        return lambda *_a, **_k: subprocess.CompletedProcess([], code, stdout, "")

    assert portability.ask_over_ssh("h", "python3", ["/a"], fake_run(noisy)) == answer
    assert portability.ask_over_ssh("h", "python3", ["/a"], fake_run("garbage")) is None
    assert portability.ask_over_ssh("h", "python3", ["/a"], fake_run(noisy, code=255)) is None
    assert portability.ask_over_ssh("h", "python3", ["/a"], lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("ssh", 1))) is None


def test_the_launch_check_names_what_changed_since_submission() -> None:
    required = (("workdir", WORK, f"{NAS}/w"), ("program", PYTHON, ""))
    ok = machine("x86_64", **{WORK: seen(where=f"{NAS}/w"), PYTHON: seen(elf="x86_64")})
    assert portability.launch_problems(required, ok) == []
    gone = machine("x86_64", **{WORK: seen(exists=False), PYTHON: seen(elf="aarch64")})
    assert portability.launch_problems(required, gone) == [f"{WORK} does not exist here", f"{PYTHON} is not built for x86_64"]
    moved = machine("x86_64", **{WORK: seen(where="nas:/other/w"), PYTHON: seen(elf="x86_64")})
    assert "not the shared copy seen at submission" in portability.launch_problems(required, moved)[0]


def test_a_node_whose_job_tools_differ_or_are_missing_is_excluded(tmp_path: Path) -> None:
    tools = "/opt/gpujobs/slurm"
    needs = [*SYSTEM_JOB, Need(Role.TOOLS, tools)]
    here = {**GOOD_HERE, tools: seen(digest="abc")}
    for there_tools, fragment in [(seen(digest="abc"), None), (seen(digest="old"), "differs from this machine's"),
                                  (seen(exists=False), "reinstall the tools there")]:
        decision = judge(needs, here, machine("x86_64", **{**GOOD_THERE["paths"], tools: there_tools}))
        assert (reason(decision) is None) if fragment is None else fragment in reason(decision)
        assert decision.requirements == judge(SYSTEM_JOB, GOOD_HERE, GOOD_THERE).requirements  # never re-checked at launch
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "b.txt").write_text("x")
    first = portability.tools_digest(str(tmp_path))
    (tmp_path / "b.txt").write_text("y")
    assert portability.tools_digest(str(tmp_path)) == first  # only *.py counts
    (tmp_path / "a.py").write_text("2")
    assert portability.tools_digest(str(tmp_path)) != first


def test_one_export_under_two_server_names_is_shared_only_for_the_same_file(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "n").mkdir()
    for folder in ("m", "n"):
        (tmp_path / folder / "x").write_text("1")

    def where(folder: str, source: str, kind: str = "nfs4") -> str:
        mounts = f"1 0 0:5 / {tmp_path / folder} rw - {kind} {source} rw\n"
        path = str(tmp_path / folder / "x")
        return portability.probe([path], mountinfo=mounts)["paths"][path]["where"]

    assert where("m", "nas.example:/e") == where("m", "100.1.2.3:/e") == where("m", "//nas/e", "cifs")  # one file, three names
    assert where("m", "h:/e") != where("n", "h2:/e")  # same export path and inside, another file (inode)
