import difflib
import importlib.util
import sys
from pathlib import Path

import site_config

ROOT = Path(__file__).resolve().parents[2]
MACHINE = ROOT / "machine" / "slurm"
GOLDEN = Path(__file__).parent / "fixtures" / "render" / "golden"
spec = importlib.util.spec_from_file_location("render", ROOT / "slurm" / "install" / "render.py")
render = importlib.util.module_from_spec(spec)
sys.modules["render"] = render  # dataclasses looks its module up here
spec.loader.exec_module(render)

# Rendered target -> today's tracked file.
TRACKED = {
    "/etc/slurm/slurm.conf": "slurm.conf",
    "/etc/slurm/cgroup.conf": "cgroup.conf",
    "/etc/slurm/gres.conf": "gres.conf",
    "/etc/slurm/slurmdbd.conf": "slurmdbd.conf",
    "/etc/mysql/mariadb.conf.d/90-slurm.cnf": "mariadb-slurm.cnf",
    **{f"/etc/systemd/system/{u}.service.d/10-restart-protect.conf": "10-restart-protect.conf" for u in render.UNITS},
}

# Every line that differs from the tracked copy, "-" tracked / "+" rendered (plan section 3.6 plus reworded comments).
EXPECTED_DIFF = {
    "slurm.conf": [
        "-# SLURM on gpu-node-1, one node. Design: design notes (revision 2, section 3.1);",
        "-# trial: build notes. Source of truth; /etc/slurm/slurm.conf is the install target.",
        "+# SLURM on gpu-node-1. Rendered from the site file by slurm/install/render.py; do not edit by hand.",
        "+# Design: design notes (revision 2, section 3.1).",
        "+# No multifactor priority: it recomputes every pending job's priority from age, and after a reboot that wiped the",
        "+# priority a hold had restored.",
        "+PriorityType=priority/basic",
        "-# (stage C, 2026-10-16). It also records each job's GPU use (`sacct`), which this box's ledger never had.",
        "+# (stage C, 2026-10-16). It also keeps the finished-job history that `sacct` reads.",
        "+# Keep each job's --comment in the accounting record.",
        "+AccountingStoreFlags=job_comment",
        "-# RealMemory: what slurm hands out once the background services have stepped aside: physical 124,610 MB minus",
        "-# the base use with the GPU idle and the box quiet (median 12,132 MB, flight recorder 2026-10-12..16) minus the",
        "-# host memory guard's 12 GiB.",
        "-NodeName=gpu-node-1 CPUs=20 Boards=1 SocketsPerBoard=1 CoresPerSocket=20 ThreadsPerCore=1 RealMemory=100000 Gres=gpu:1 State=UNKNOWN",
        "+# RealMemory is what slurm hands out: the node's real_memory_mb in the site file (its memory minus the background",
        "+# services' base use minus the host memory guard's reserve).",
        "+NodeName=gpu-node-1 CPUs=20 Boards=1 SocketsPerBoard=1 CoresPerSocket=20 ThreadsPerCore=1 RealMemory=100000 Gres=gpu:gb10:1 State=UNKNOWN",
        "-PartitionName=urgent Nodes=gpu-node-1 PriorityTier=2 PreemptMode=OFF MaxTime=INFINITE State=UP",
        "+PartitionName=urgent Nodes=gpu-node-1 PriorityTier=2 PreemptMode=OFF MaxTime=INFINITE State=UP AllowGroups=gpujobs-admin",
    ],
    "gres.conf": [
        "-# The GB10's one GPU as a schedulable token. Its memory is system RAM (unified), so no memory is declared here.",
        "-NodeName=gpu-node-1 Name=gpu File=/dev/nvidia0",
        "+# Each GPU as a schedulable token, typed to match the node's Gres= in slurm.conf. No NVML autodetect: Ubuntu's slurm",
        "+# ships no NVML plugin. On a unified-memory machine (the GB10) the GPU's memory is system RAM, so none is declared.",
        "+NodeName=gpu-node-1 Name=gpu Type=gb10 File=/dev/nvidia0",
    ],
    "slurmdbd.conf": [
        "-# kill_invalid_depend never cancelled a failed chain (stage C test 7). Source of truth; /etc/slurm/slurmdbd.conf",
        "-# is the install target, installed 0600 and owned by slurm by install_slurm.py, as slurmdbd requires.",
        "+# kill_invalid_depend never cancelled a failed chain (stage C test 7). Rendered from the site file; installed",
        "+# as /etc/slurm/slurmdbd.conf, 0600 and owned by slurm, as slurmdbd requires.",
    ],
    "mariadb-slurm.cnf": [
        "-# some sites use: on this box the RAM is also the GPU's memory, and this database holds one node's job records.",
        "-# Installed to /etc/mysql/mariadb.conf.d/90-slurm.cnf by install_slurm.py.",
        "+# some sites use: on a unified-memory machine (the GB10) the RAM is also the GPU's memory, and this database holds one node's job records.",
        "+# Installed to /etc/mysql/mariadb.conf.d/90-slurm.cnf by slurm/install/install.py.",
    ],
    "10-restart-protect.conf": [
        "-# slurmctld, slurmd and munge by install_slurm.py.",
        "+# slurmctld, slurmd and munge by slurm/install/install.py.",
    ],
}


def _changed_lines(tracked: str, rendered: str) -> list[str]:
    diff = difflib.unified_diff(tracked.splitlines(), rendered.splitlines(), lineterm="", n=0)
    return [line for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---")]


def test_golden_matches_tracked_files_except_the_planned_lines():
    site = site_config.load(MACHINE / "site.toml")
    files = render.render(site, "slurm")
    assert set(files) == set(TRACKED)
    for target, name in TRACKED.items():
        got = _changed_lines((GOLDEN / name).read_text(), files[target].content)
        assert got == EXPECTED_DIFF.get(name, []), name
    assert (files["/etc/slurm/slurmdbd.conf"].mode, files["/etc/slurm/slurmdbd.conf"].owner) == (0o600, "slurm")
    others = {(f.mode, f.owner) for t, f in files.items() if t != "/etc/slurm/slurmdbd.conf"}
    assert others == {(0o644, "root")}


def test_eight_gpu_site_and_extra_slot():
    raw = {
        "name": "big",
        "admin_group": "ops",
        "nodes": [
            {
                "name": "n1",
                "cpus": 64,
                "real_memory_mb": 500000,
                "gpus": [
                    {"index": i, "uuid": f"GPU-{i}", "device": f"/dev/nvidia{i}", "type": "h100"} for i in range(8)
                ],
            }
        ],
    }
    site = site_config.parse(raw)
    files = render.render(site, "slurm", {"slurm_conf": "Prolog=/x"})
    gres = files["/etc/slurm/gres.conf"].content.splitlines()
    assert [ln for ln in gres if not ln.startswith("#")] == [
        f"NodeName=n1 Name=gpu Type=h100 File=/dev/nvidia{i}" for i in range(8)
    ]
    conf = files["/etc/slurm/slurm.conf"].content
    assert "Gres=gpu:h100:8" in conf
    assert "AllowGroups=ops" in conf
    assert conf.endswith("State=UP AllowGroups=ops\nProlog=/x\n")
