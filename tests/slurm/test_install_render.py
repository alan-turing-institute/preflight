import dataclasses
import difflib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import gpu_jobs
import site_config

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "fixtures" / "render" / "golden"
INSTALL = str(ROOT / "slurm" / "install")
if INSTALL not in sys.path:
    sys.path.append(INSTALL)  # `components` is imported by name below; another test file used to put it there by luck
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
        "-GresTypes=gpu",
        "+GresTypes=gpu,shard",
        "-# Pre-emption by partition priority, no accounting database: an `urgent` job takes resources from `normal` jobs,",
        "-# which are requeued (SIGTERM to every process of the job, then SIGKILL after KillWait) and start again when resources free.",
        "+# Pre-emption by partition priority, strictly lower tiers only (accounting is through slurmdbd, below): an `urgent` job (tier 3) takes",
        "+# resources from `normal` and `cpu_fill` jobs, and a `normal` GPU job (tier 2) from `cpu_fill` jobs; they are requeued (SIGTERM to every process of the job, then SIGKILL after KillWait) and start again when resources free.",
        # D45: slurm's Epilog on every site, which removes an ended job's docker containers
        "+# Slurm's own end-of-job hook, run as root on every node when a job's allocation is released (its end, a requeue, a",
        "+# pre-emption): removes the job's docker containers. No path, so slurm sends it to configless nodes too; this first",
        "+# Epilog= line runs before any later one (metrics adds Jobstats' epilog.d glob).",
        "+Epilog=gpujobs_containers.sh",
        "-# (stage C, 2026-10-16). It also records each job's GPU use (`sacct`), which this box's ledger never had.",
        "+# (stage C, 2026-10-16). It also keeps the finished-job history that `sacct` reads.",
        "-AccountingStorageTRES=gres/gpu",
        "+AccountingStorageTRES=gres/gpu,gres/shard",
        "+# Keep each job's --comment in the accounting record.",
        "+AccountingStoreFlags=job_comment",
        "-# RealMemory: what slurm hands out once the background services have stepped aside: physical 124,610 MB minus",
        "-# the base use with the GPU idle and the box quiet (median 12,132 MB, flight recorder 2026-10-12..16) minus the",
        "-# host memory guard's 12 GiB.",
        "-NodeName=gpu-node-1 CPUs=20 Boards=1 SocketsPerBoard=1 CoresPerSocket=20 ThreadsPerCore=1 RealMemory=100000 Gres=gpu:1 State=UNKNOWN",
        "-# normal: resumes from its checkpoint, pre-emptable. steady: cannot resume, never pre-empted. urgent:",
        "+# RealMemory is what slurm hands out: the node's real_memory_mb in the site file (its memory minus the background",
        "+# services' base use minus the host memory guard's reserve).",
        "+NodeName=gpu-node-1 CPUs=20 Boards=1 SocketsPerBoard=1 CoresPerSocket=20 ThreadsPerCore=1 RealMemory=100000 Gres=gpu:gb10:1,shard:12 State=UNKNOWN",
        "+# normal: resumes from its checkpoint, pre-emptable. steady: cannot resume, never pre-empted. Both tier 2. urgent (tier 3):",
        "-PartitionName=normal Nodes=gpu-node-1 Default=YES PriorityTier=1 PreemptMode=REQUEUE MaxTime=INFINITE State=UP",
        "-PartitionName=steady Nodes=gpu-node-1 PriorityTier=1 PreemptMode=OFF MaxTime=INFINITE State=UP",
        "-PartitionName=urgent Nodes=gpu-node-1 PriorityTier=2 PreemptMode=OFF MaxTime=INFINITE State=UP",
        "+PartitionName=normal Nodes=gpu-node-1 Default=YES PriorityTier=2 PreemptMode=REQUEUE MaxTime=INFINITE State=UP",
        "+PartitionName=steady Nodes=gpu-node-1 PriorityTier=2 PreemptMode=OFF MaxTime=INFINITE State=UP",
        "+# cpu: every `restart` job without GPUs (submit.py routes it here), kept off the cores the site file holds for GPU jobs",
        "+# (cores_per_gpu), so a free GPU never waits for cores. ONE partition: MaxCPUsPerNode counts only its own jobs.",
        "+# Never pre-empted by slurm.",
        "+PartitionName=cpu Nodes=gpu-node-1 PriorityTier=1 PreemptMode=OFF MaxTime=INFINITE State=UP MaxCPUsPerNode=8",
        "+# cpu_fill: every `resume` job without GPUs: no core cap, so it uses idle cores; tier 1 REQUEUE, so GPU (tier 2) and urgent",
        "+# (tier 3) jobs take its cores and memory back when they need them (it resumes from its checkpoint).",
        "+PartitionName=cpu_fill Nodes=gpu-node-1 PriorityTier=1 PreemptMode=REQUEUE MaxTime=INFINITE State=UP",
        "+PartitionName=urgent Nodes=gpu-node-1 PriorityTier=3 PreemptMode=OFF MaxTime=INFINITE State=UP AllowGroups=gpujobs-admin",
        "+# service: background services' CPU stand-ins (submit.py --service); the agent suspends cpu jobs to make room.",
        "+PartitionName=service Nodes=gpu-node-1 PriorityTier=2 PreemptMode=REQUEUE MaxTime=INFINITE State=UP MaxCPUsPerNode=4 AllowGroups=gpujobs-service,gpujobs-admin",
    ],
    "gres.conf": [
        "-# The GB10's one GPU as a schedulable token. Its memory is system RAM (unified), so no memory is declared here.",
        "-NodeName=gpu-node-1 Name=gpu File=/dev/nvidia0",
        "+# Each GPU as a schedulable token, typed to match the node's Gres= in slurm.conf. No NVML autodetect: Ubuntu's slurm",
        "+# ships no NVML plugin. On a unified-memory machine (one whose GPU shares system RAM, such as the GB10) the GPU's memory is system RAM, so none is declared.",
        "+NodeName=gpu-node-1 Name=gpu Type=gb10 File=/dev/nvidia0",
        "+NodeName=gpu-node-1 Name=shard Count=12 File=/dev/nvidia0",
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
    site = site_config.load(ROOT / "tests/slurm/fixtures/site.toml")
    site = dataclasses.replace(site, nodes=site.full_nodes, health_check=None)  # the golden files predate part-time machines
    files = render.render(site, "slurm")
    epilog = files.pop(render.JOB_CONTAINERS_EPILOG)  # D45: on every site, newer than the golden files
    assert (epilog.mode, epilog.owner) == (0o755, "root") and epilog.content.startswith("#!/bin/sh\n")
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
        "constrain_devices": False,
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
    assert "\nConstrainDevices=no\n" in files["/etc/slurm/cgroup.conf"].content  # the golden default is yes
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
    assert added[1].endswith(" PriorityTier=2 PreemptMode=REQUEUE MaxTime=INFINITE State=UP MaxCPUsPerNode=4 "
                             "AllowGroups=gpujobs-service,ops")


def test_sharing_card_renders_shard_lines_and_the_unshared_site_has_none():
    gpu = {"index": 0, "uuid": "GPU-0", "device": "/dev/nvidia0", "type": "gb10"}
    raw = {"name": "s", "nodes": [{"name": "n1", "cpus": 20, "real_memory_mb": 100000, "cores_per_gpu": 12, "gpus": [{**gpu, "shards": "cores"}]}]}
    shared = render.render(site_config.parse(raw), "slurm")
    conf, gres = shared["/etc/slurm/slurm.conf"].content, shared["/etc/slurm/gres.conf"].content
    assert "\nGresTypes=gpu,shard\n" in conf and "Gres=gpu:gb10:1,shard:12 " in conf and "\nAccountingStorageTRES=gres/gpu,gres/shard\n" in conf
    assert "NodeName=n1 Name=shard Count=12 File=/dev/nvidia0\n" in gres
    raw["nodes"][0]["gpus"] = [gpu]  # the same site, card not shared: no trace of shards
    plain = render.render(site_config.parse(raw), "slurm")
    assert "shard" not in plain["/etc/slurm/slurm.conf"].content + plain["/etc/slurm/gres.conf"].content


def test_unified_memory_full_node_reserve_and_gpu_node_elsewhere_refused():
    gpu = {"index": 0, "uuid": "GPU-0", "device": "/dev/nvidia0", "type": "t"}
    raw = {"name": "u", "unified_memory": True,
           "nodes": [{"name": "head", "cpus": 20, "real_memory_mb": 100000, "jobs_memory_gb": 80, "gpus": [gpu]}]}
    site = site_config.parse(raw)
    conf = render.render(site, "slurm")["/etc/slurm/slurm.conf"].content
    head = next(line for line in conf.splitlines() if line.startswith("NodeName=head "))
    assert " MemSpecLimit=23706" in head  # a full node's measured reserve: 80 decimal GB is 76294 of slurm's 2^20-byte MB
    # the memory guard watches only the agent's own node (D44): a GPU node elsewhere is refused, and only on a unified site
    agent = importlib.import_module("components.agent")
    assert agent.refusals(site, hostname="head.lan") == []
    elsewhere = agent.refusals(site, hostname="other")
    assert len(elsewhere) == 1 and elsewhere[0].startswith("node head has GPUs on a unified-memory site but is not this machine")
    assert agent.refusals(site_config.parse({**raw, "unified_memory": False}), hostname="other") == []


def test_part_time_node_and_health_check():
    raw = {
        "name": "mix",
        "nodes": [
            {"name": "head", "cpus": 8, "real_memory_mb": 60000},
            {"name": "lap", "cpus": 22, "real_memory_mb": 33000, "part_time": True, "address": "100.64.0.2",
             "keep_cpus": "12-21", "jobs_memory_gb": 20, "mains_only": True, "close_below_gb": 8, "cores_per_socket": 16},
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
        "HealthCheckNodeState=ANY",  # no timeout line: the distribution's slurm 23.11 refuses a file that has one
        "SlurmctldParameters=enable_configless",
        "Include part_time_nodes.conf",
        "PartitionName=part_time Nodes=lap PriorityTier=1 PreemptMode=OFF MaxTime=INFINITE State=UP",
    ]
    pinned = site_config.parse({**raw, "health_check": {"interval_s": 45}, "slurm": {"version": "26.05.5", "sha256": "0" * 64}})
    assert "HealthCheckTimeout=40" in render.render(pinned, "slurm")["/etc/slurm/slurm.conf"].content  # below the interval
    # the part-time node's line is in its own file, owned by SlurmUser so the agent can rewrite it; slurm.conf has no copy
    part_time = render.render(site, "slurm")[render.PART_TIME_CONF_PATH]
    assert (part_time.mode, part_time.owner) == (0o644, "slurm")
    assert [ln for ln in part_time.content.splitlines() if not ln.startswith("#")] == [
        "NodeName=lap NodeAddr=100.64.0.2 CPUs=22 Boards=1 SocketsPerBoard=1 CoresPerSocket=16 ThreadsPerCore=1 "
        "RealMemory=33000 CpuSpecList=12-21 MemSpecLimit=13927 State=UNKNOWN"]
    assert "NodeName=lap" not in conf and render.PART_TIME_CONF_PATH not in render.render(
        site_config.parse({**raw, "nodes": raw["nodes"][:1]}), "slurm")
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


def test_the_container_epilog_removes_only_the_jobs_containers_and_always_exits_0(tmp_path):
    """slurmd runs it as root with no PATH (`env -i`); exit 0 in every case, since a failing epilog drains the node."""
    calls, log = tmp_path / "calls", tmp_path / "log"
    good = "ab" * 32
    docker = tmp_path / "docker"
    docker.write_text(f'''#!/bin/sh
echo "$*" >> {calls}
case "$MODE:$1" in
hang:*) sleep 30 ;;
down:*) echo "Cannot connect to the Docker daemon" >&2; exit 1 ;;
none:ps) exit 0 ;;
rmfails:rm) echo "removal of container is already in progress" >&2; exit 1 ;;
*:ps) printf '%s\\nnot-an-id\\n{"c" * 63}\\n*\\n' {good}; echo "WARNING: * [a-f]*" >&2 ;;
esac
''')
    logger = tmp_path / "logger"
    logger.write_text(f'#!/bin/sh\nshift 3\necho "$*" >> {log}\n')  # drops `-t TAG --`
    for f in (docker, logger):
        f.chmod(0o755)
    script = tmp_path / "epilog.sh"
    script.write_text(render.job_containers_epilog())
    assert f"label={gpu_jobs.JOB_LABEL}=$job" in script.read_text()  # the one label name the Python side uses
    (tmp_path / ("f" * 64)).write_text("")  # a file whose name a glob of docker's warning would turn into an "id"

    def epilog(mode: str, job: str = "77", docker_path: Path = docker) -> list[str]:
        for f in (calls, log):
            f.unlink(missing_ok=True)
        env = {"SLURM_JOB_ID": job, "DOCKER": str(docker_path), "LOGGER": str(logger), "DOCKER_WAIT_S": "1", "MODE": mode}
        done = subprocess.run(["/usr/bin/env", "-i", *(f"{k}={v}" for k, v in env.items()), "/bin/sh", str(script)],
                              capture_output=True, text=True, timeout=20, check=False, cwd=tmp_path)
        assert done.returncode == 0, (mode, done.stderr)
        return log.read_text().splitlines() if log.exists() else []

    said = epilog("some")
    assert calls.read_text().splitlines() == ["ps -aq --no-trunc --filter label=slurm-job=77", f"rm -f {good}"]
    assert said == ["job 77: skipped 3 word(s) of docker output that are not container ids", f"job 77: removed container(s) {good}"]
    assert epilog("none") == ["job 77: no containers"]
    assert "ps --filter label=slurm-job=77' did not answer (exit 1)" in epilog("down")[0] and "rm -f" not in calls.read_text()
    assert "did not answer" in epilog("hang")[0]  # bounded: a hung daemon never holds the node COMPLETING
    assert epilog("rmfails")[-1].startswith(f"job 77: '{docker} rm -f {good}' failed (")  # names job, containers, command
    assert "not a job id" in epilog("some", job="7;rm -rf /")[0] and not calls.exists()
    missing = epilog("some", docker_path=tmp_path / "no-such-docker")
    socket_line = ("job 77: /var/run/docker.sock exists but no docker command was found: "
                   "its containers were not checked")
    assert not calls.exists() and missing in ([], [socket_line])

