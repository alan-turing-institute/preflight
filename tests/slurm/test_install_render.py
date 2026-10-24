import difflib
import importlib
import importlib.util
import json
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
        "-# Pre-emption by partition priority, no accounting database: an `urgent` job takes resources from `normal` jobs,",
        "+# Pre-emption by partition priority (accounting is through slurmdbd, below): an `urgent` job takes resources from `normal` jobs,",
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
        "+# cpu: every job without GPUs (submit.py routes it here), kept off the cores the site file holds for GPU jobs",
        "+# (cores_per_gpu), so a free GPU never waits for cores. ONE partition: MaxCPUsPerNode counts only its own jobs.",
        "+# Never pre-empted by slurm; urgent.py requeues a resume-class job in its way itself.",
        "+PartitionName=cpu Nodes=gpu-node-1 PriorityTier=1 PreemptMode=OFF MaxTime=INFINITE State=UP MaxCPUsPerNode=8",
        "+PartitionName=urgent Nodes=gpu-node-1 PriorityTier=2 PreemptMode=OFF MaxTime=INFINITE State=UP AllowGroups=gpujobs-admin",
        "+# service: background services' CPU stand-ins (submit.py --service); the agent suspends cpu jobs to make room.",
        "+PartitionName=service Nodes=gpu-node-1 PriorityTier=1 PreemptMode=REQUEUE MaxTime=INFINITE State=UP MaxCPUsPerNode=4 AllowGroups=gpujobs-service,gpujobs-admin",
    ],
    "gres.conf": [
        "-# The GB10's one GPU as a schedulable token. Its memory is system RAM (unified), so no memory is declared here.",
        "-NodeName=gpu-node-1 Name=gpu File=/dev/nvidia0",
        "+# Each GPU as a schedulable token, typed to match the node's Gres= in slurm.conf. No NVML autodetect: Ubuntu's slurm",
        "+# ships no NVML plugin. On a unified-memory machine (one whose GPU shares system RAM, such as the GB10) the GPU's memory is system RAM, so none is declared.",
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
        "+# some sites use: on a unified-memory machine (a GPU that shares system RAM, such as the GB10) the RAM is also the GPU's memory, and this database holds one node's job records.",
        "+# Installed to /etc/mysql/mariadb.conf.d/90-slurm.cnf by slurm/install/install.py.",
    ],
    "cgroup.conf": [
        "-# CPU-side memory is capped at its --mem (GPU memory is not charged to the cgroup on the GB10, trial B2, so the",
        "+# CPU-side memory is capped at its --mem (GPU memory is not charged to the cgroup on a unified-memory GPU such as the GB10 (trial B2), so the",
    ],
    "10-restart-protect.conf": [
        "-# slurmctld, slurmd and munge by install_slurm.py.",
        "+# slurmctld, slurmd, slurmdbd and munge by slurm/install/install.py.",
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
                "cores_per_gpu": 4,
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
    assert conf.count(" MaxCPUsPerNode=32\n") == 1  # 64 cores less 4 held for each of 8 GPUs, on the one cpu partition
    assert conf.endswith("State=UP AllowGroups=ops\nProlog=/x\n")
    assert "PartitionName=service" not in conf  # no [service]: as before
    # with one (D20): only the service partition is added; slurm's pre-emption and every other partition stay as they are
    served = render.render(site_config.parse({**raw, "service": {"cores": 4}}), "slurm")["/etc/slurm/slurm.conf"].content
    added = [line for line in served.splitlines() if line not in conf.splitlines()]
    assert len(added) == 2 and added[1].startswith("PartitionName=service ")
    assert added[1].endswith(" PriorityTier=1 PreemptMode=REQUEUE MaxTime=INFINITE State=UP MaxCPUsPerNode=4 "
                             "AllowGroups=gpujobs-service,ops")


def test_part_time_node_and_health_check():
    raw = {
        "name": "mix",
        "nodes": [
            {"name": "head", "cpus": 8, "real_memory_mb": 60000},
            {"name": "lap", "cpus": 22, "real_memory_mb": 33000, "part_time": True, "address": "100.64.0.2",
             "keep_cpus": "12-21", "jobs_memory_gb": 20, "mains_only": True, "close_below_gb": 8},
        ],
        "service": {"cores": 4},
    }
    base = render.render(site_config.parse({**raw, "nodes": raw["nodes"][:1]}), "slurm")["/etc/slurm/slurm.conf"].content
    site = site_config.parse({**raw, "health_check": {"interval_s": 45}})
    conf = render.render(site, "slurm")["/etc/slurm/slurm.conf"].content
    # the full node's lines are unchanged, and the only additions are the part-time node, its partition and the health check
    added = [ln for ln in conf.splitlines() if ln not in base.splitlines()]
    assert [ln for ln in added if not ln.startswith("#")] == [
        "HealthCheckProgram=/opt/gpujobs/slurm/node_check.py",
        "HealthCheckInterval=45",
        "HealthCheckNodeState=ANY",
        "SlurmctldParameters=enable_configless",
        "NodeName=lap NodeAddr=100.64.0.2 CPUs=22 Boards=1 SocketsPerBoard=1 CoresPerSocket=22 ThreadsPerCore=1 "
        "RealMemory=33000 CpuSpecList=12-21 MemSpecLimit=13000 State=UNKNOWN",
        "PartitionName=part_time Nodes=lap PriorityTier=1 PreemptMode=OFF MaxTime=INFINITE State=UP",
    ]
    assert "SlurmctldHost=head\n" in conf and "Nodes=lap" not in conf.replace("PartitionName=part_time Nodes=lap", "")
    # without [health_check]: no health-check lines and no per-node file; with it, one entry per node, GB as decimal MB
    tools = importlib.import_module("components.tools")
    assert tools.NODE_CHECK_JSON not in tools.files(site_config.parse(raw))
    assert "HealthCheck" not in render.render(site_config.parse(raw), "slurm")["/etc/slurm/slurm.conf"].content
    node_check = json.loads(tools.files(site)[tools.NODE_CHECK_JSON].content)
    assert node_check["lap"] == {
        "node": "lap", "close_below_mb": 8000, "push_back_below_mb": 2000, "mains_only": True,
        "meminfo_path": "/proc/meminfo", "power_supply_dir": "/sys/class/power_supply",
        "state_path": "/var/lib/gpujobs/node_check/state.json", "log_path": "/var/log/gpujobs/node_check.log",
    }
    assert (node_check["head"]["close_below_mb"], node_check["head"]["push_back_below_mb"],
            node_check["head"]["mains_only"]) == (16000, 10000, None)
