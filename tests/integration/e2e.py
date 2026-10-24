#!/usr/bin/env python3
"""End-to-end test of the portable GPU jobs build in the container cluster (README, "End-to-end run").

    tests/integration/run.sh e2e            # fresh cluster, then everything below
    python3 tests/integration/e2e.py [--only STAGE,...] [--list]    # against the running cluster

Runs on the host and drives the container through `run.sh shell`. Every check prints PASS or FAIL with its
evidence as it happens; the exit status is 1 when any check failed. Stages: install, jobs, urgent, failure,
recovery, portal, proxy, ssh, lease, host, upgrade (in this order; `--only` takes a subset, later stages need `install` to have run once).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "slurm"))
from levels import Level, priority_for  # noqa: E402 - stdlib only; the priority the agent gives each waiting job

TOP_PRIORITY = 0xFFFF0000  # slurm's job_mgr.c: the most priority/basic gives a job by itself
RUN = HERE / "run.sh"
SITE = "/src/tests/integration/harness.site.toml"
INSTALL = "python3 /src/slurm/install/install.py"
STATE = "/var/lib/gpujobs"
PORTAL = "https://testnode"
APP = f"{PORTAL}/pun/sys/gpujobs"
GRAFANA_TLS = "https://testnode:3443"
USERS = ("alice", "bob", "carol")
# Test-only portal passwords of the throwaway accounts (only their bcrypt hashes go into the container).
PASSWORDS: dict[str, str] = {u: f"pw-{u}-e2e" for u in (*USERS, "dave")}
RESULTS: list[tuple[str, bool, str]] = []
CHAIN_LENGTH = 3  # jobs in the chain S1 submits
CPU_CAP = 2  # cores S3 lets the CPU-only partition use on the 4-core harness node
AGENT_LOOP_S = 5  # the agent's loop interval
REQUEUE_DELAY_S = 60  # slurm.conf AuthInfo=cred_expire: how long before a requeued job may start again
REQUEUE_WITHIN_S = 70  # the agent waits 30 s for slurmctld, then acts within one 5-s loop; slack for the restarts themselves
SCRAPED_EXPORTERS = 2  # exporters Prometheus can scrape in the container (node, cgroup; the GPU one needs NVML)


@dataclass
class Out:
    rc: int
    out: str
    err: str

    @property
    def text(self) -> str:
        return (self.out + self.err).strip()


def cl(*argv: str, user: str = "root", timeout: int = 300, stdin: str | None = None) -> Out:
    """One command in the container, as `user`."""
    cmd = [str(RUN), "shell", *(["-u", user] if user != "root" else []), *argv]
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, input=stdin, check=False,
            **({"stdin": subprocess.DEVNULL} if stdin is None else {}),
        )
    except subprocess.TimeoutExpired as exc:
        return Out(124, "", f"timed out after {timeout} s: {' '.join(argv)[:200]} {exc.stderr or ''}")
    return Out(done.returncode, done.stdout, done.stderr)


def sh(script: str, user: str = "root", timeout: int = 300) -> Out:
    return cl("bash", "-c", script, user=user, timeout=timeout)


def check(name: str, ok: bool, evidence: str = "") -> bool:
    RESULTS.append((name, ok, evidence))
    shown = " ".join(evidence.split())[:330]
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {shown}" if shown else ""), flush=True)
    return ok


NOTES: list[str] = []


def note(text: str) -> None:
    NOTES.append(text)
    print(f"NOTE  {text}", flush=True)


def section(title: str) -> None:
    print(f"\n== {title}", flush=True)


def wait_for(fn: Callable[[], object], timeout: float, every: float = 2.0) -> object:
    """The first truthy result of `fn`, or None at the timeout."""
    end = time.monotonic() + timeout
    while True:
        value = fn()
        if value:
            return value
        if time.monotonic() >= end:
            return None
        time.sleep(every)


# -- slurm helpers ---------------------------------------------------------------------------------------------


def jobinfo(job: str) -> dict[str, str]:
    done = sh(f"scontrol show job -o {job}")
    return dict(re.findall(r"(\w+)=(\S*)", done.out)) if done.rc == 0 else {}


def state(job: str) -> str:
    return jobinfo(job).get("JobState", "GONE")


def priority(job: str) -> int:
    return int(jobinfo(job).get("Priority", "-1"))


def wait_state(job: str, want: str, timeout: float = 90) -> bool:
    return bool(wait_for(lambda: state(job) == want, timeout))


def nodes_alloc() -> str:
    return sh("scontrol show node testnode -o").out


def sleep_cmd(seconds: int) -> list[str]:
    return ["python3", "-c", f"import time; time.sleep({seconds})"]


def submit(user: str, name: str, seconds: int, *flags: str, cls: str = "resume", command: list[str] | None = None) -> str:
    """Submit through the installed tool as `user`; returns the job id ('' if refused). `--mem 200M --cpus 1` keep jobs side by
    side (a GPU job without --mem takes the whole box, and the default 4 CPUs are the whole container node)."""
    argv = ["gpujobs-submit", "--project", "e2e", "--what", f"e2e {name}", "--class", cls, "--name", name, "--mem", "200M", "--cpus", "1"]
    argv += ["--resume-ok"] if cls == "resume" else []
    done = cl(*argv, *flags, "--", *(command or sleep_cmd(seconds)), user=user)
    m = re.search(r"\[submit\] job (\d+)", done.out)
    if not m:
        print(f"   submit by {user} gave: {done.text[:300]}")
    return m.group(1) if m else ""


def clear_queue() -> None:
    sh("scancel $(squeue -h -o %i) 2>/dev/null; rm -f " + STATE + "/requests/*; true")
    wait_for(lambda: not sh("squeue -h").out.strip(), 150)


def events(job: str | None = None) -> list[tuple[str, str, str]]:
    """(owner, job, action) of each event file; the owner is the actor, as the tools read it."""
    out = sh(f"stat -c '%U %n' {STATE}/events/* 2>/dev/null").out
    found = []
    for line in out.splitlines():
        owner, _, path = line.partition(" ")
        m = re.fullmatch(r".*/(\d+)-\d+-(\w+)\.json", path)
        if m and (job is None or m.group(1) == job):
            found.append((owner, m.group(1), m.group(2)))
    return found


def agent_json(name: str) -> dict:
    done = sh(f"cat {STATE}/agent/{name}")
    try:
        return json.loads(done.out)
    except ValueError:
        return {}


def placed(level: Level, job: str) -> int:
    """The priority the agent gives a waiting job, from the base id it keeps in its state."""
    return priority_for(level, job, agent_json("state.json").get("base") or 0)


def can_submit(user: str) -> tuple[bool, str]:
    """Whether slurmctld would accept a job from `user` (nothing is queued)."""
    done = cl("sbatch", "--test-only", "--wrap", "true", "--mem", "100M", user=user)
    return done.rc == 0, done.text[-140:]


def status_text(user: str) -> str:
    return cl("gpujobs-status", user=user).text


# -- stage: install --------------------------------------------------------------------------------------------

SANDBOX_OFF = (
    "PrivateUsers", "PrivateDevices", "PrivateTmp", "ProtectSystem", "ProtectHome", "ProtectHostname", "ProtectClock",
    "ProtectControlGroups", "ProtectKernelLogs", "ProtectKernelModules", "ProtectKernelTunables", "RestrictNamespaces",
    "RemoveIPC", "LockPersonality", "MemoryDenyWriteExecute", "RestrictRealtime", "RestrictSUIDSGID", "NoNewPrivileges",
)
DROPIN = (
    "[Service]\n" + "".join(f"{key}=no\n" for key in SANDBOX_OFF)
    + "ProtectProc=default\nDevicePolicy=auto\nSystemCallArchitectures=\nRestrictAddressFamilies=\nCapabilityBoundingSet=\n"
)
DROPIN_DYNAMIC = "[Service]\nDynamicUser=no\nUser=nobody\n"
GRAFANA_VHOST = """Listen 3443
<VirtualHost *:3443>
  ServerName testnode
  SSLEngine on
  SSLCertificateFile /etc/ood/ssl/portal.crt
  SSLCertificateKeyFile /etc/ood/ssl/portal.key
  ProxyPreserveHost On
  ProxyPass / http://127.0.0.1:3001/
  ProxyPassReverse / http://127.0.0.1:3001/
</VirtualHost>
"""
NOTIFY = '#!/bin/sh\necho "$GPUJOBS_JOB $GPUJOBS_STATE $GPUJOBS_USER $GPUJOBS_NAME" >> /var/tmp/harness-notify.log\n'
HASH_USERS = """
import crypt, json, os, sys, uuid
passwords = json.load(sys.stdin)
users = [{"username": u, "email": f"{u}@harness.test", "userID": str(uuid.uuid4()),
          "hash": crypt.crypt(p, crypt.mksalt(crypt.METHOD_BLOWFISH))} for u, p in passwords.items() if u != "dave"]
fd = os.open("/etc/gpujobs/portal/users.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, json.dumps(users).encode())
"""


def put(path: str, text: str, mode: str = "0644") -> None:
    target = shlex.quote(path)
    done = cl("bash", "-c", f"mkdir -p \"$(dirname {target})\" && cat > {target} && chmod {mode} {target}", stdin=text, timeout=60)
    if done.rc:
        raise SystemExit(f"could not write {path}: {done.text}")


def stage_install() -> None:
    section("install: every component, through the installer, from the harness site file")
    ready = wait_for(
        lambda: all(u in sh("sacctmgr -n -P show user format=user").out for u in USERS) and sh("sinfo -h").rc == 0, 180
    )
    check("cluster ready (slurm answers; the first-boot unit has created alice, bob and carol)", bool(ready))
    # Prerequisites an operator supplies before `apply`: portal secrets, the notify script, and a user the installer has
    # not been told about.
    sh("useradd --create-home --shell /bin/bash dave && install -d -m 0700 /etc/gpujobs/portal")
    sh("openssl rand -hex 32 | install -m 0600 /dev/stdin /etc/gpujobs/portal/crypto_passphrase")
    cl("python3", "-W", "ignore", "-c", HASH_USERS, stdin=json.dumps(PASSWORDS))
    put("/usr/local/bin/harness-notify", NOTIFY, "0755")
    # The job wrapper refuses to start a GPU job when nvidia-smi does not answer (occupancy unknown). The fake GPUs are
    # /dev/null, so no compute process ever runs on them: this stand-in answers with none.
    put("/usr/local/bin/nvidia-smi", "#!/bin/sh\nexit 0\n", "0755")
    # The container refuses the namespace sandboxing of the services' units (systemd: "Failed to set up mount
    # namespacing: Operation not supported"); a real host does not. Harness drop-ins switch it off.
    for unit in ("prometheus", "grafana-server"):
        put(f"/etc/systemd/system/{unit}.service.d/99-harness-container.conf", DROPIN)
    for unit in ("gpujobs-node-exporter", "gpujobs-gpu-exporter"):
        put(f"/etc/systemd/system/{unit}.service.d/99-harness-container.conf", DROPIN_DYNAMIC)

    before = sh("sacctmgr -n -P show assoc where account=gpujobs format=account").out.strip()
    check("before apply: slurm accounting has no record of the site's account 'gpujobs'", before == "", before)
    # A fresh machine's MariaDB has no login for slurmdbd (2026-10-21, the Turing box: "Access denied for user
    # 'slurm'@'localhost'"). The first-boot unit made one; take it away so only apply can bring slurmdbd back.
    dropped = sh("mariadb -e \"DROP USER 'slurm'@'localhost'\"")
    check("before apply: slurmdbd's MariaDB login removed, as on a fresh machine", dropped.rc == 0, dropped.text[:200])
    # Nor has it slurmctld's state folder, which slurmctld cannot make itself; with slurmctld stopped, squeue fails, so
    # this is also the first apply that stopped before starting slurmctld, which must be allowed to run again.
    fresh = sh("systemctl stop slurmctld && rm -rf /var/spool/slurmctld")
    check("before apply: slurmctld stopped and its state folder removed, as on a fresh machine", fresh.rc == 0, fresh.text[:200])

    detected = sh(f"{INSTALL} detect")
    check("install.py detect prints a draft site file for the container", detected.rc == 0 and "testnode" in detected.out
          and "[[nodes]]" in detected.out, detected.text[:200])
    rendered = sh(f"rm -rf /tmp/rendered && {INSTALL} --site {SITE} render --out /tmp/rendered && "
                  "find /tmp/rendered -type f | wc -l && grep -c Gres=gpu:fake:4 /tmp/rendered/etc/slurm/slurm.conf")
    said = re.search(r"rendered (\d+) file", rendered.out)
    on_disk = rendered.out.split()[-2:-1]
    check("install.py render writes as many files as it reports, and slurm.conf names the four fake GPUs",
          rendered.rc == 0 and bool(said) and on_disk == [said.group(1)] and rendered.out.split()[-1] == "1",
          rendered.text[-200:])
    pre = sh(f"{INSTALL} --site {SITE} check")
    check("check before apply: exit 1 and a DIFFERS line", pre.rc == 1 and "DIFFERS" in pre.out, pre.text[:200])

    started = time.monotonic()
    # The first-boot units set slurm up, not this installer, so taking it over needs the flag (e301be5).
    applied = sh(f"{INSTALL} --site {SITE} apply --replace-existing-slurm", timeout=1500)
    check(f"apply installs slurm, tools, agent, metrics and portal ({time.monotonic() - started:.0f} s)",
          applied.rc == 0 and "installed" in applied.out, applied.text[-300:])
    sh("systemctl reset-failed")
    post = sh(f"{INSTALL} --site {SITE} check")
    check("check after apply: the installed files match the rendered ones", post.rc == 0, post.text[:300])
    wait_for(lambda: "idle" in sh("sinfo -h -N -o %T").out, 120)

    found = sh("sacctmgr -n -P show assoc where account=gpujobs format=account").out.strip()
    check("apply made a slurm accounting record itself: account 'gpujobs' (the container's first-boot unit made only 'harness')",
          bool(found), found)

    refused: dict[str, str] = {}  # user -> what slurm said straight after the add
    joined = []
    for u in ("alice", "bob", "dave"):
        joined.append(cl(*INSTALL.split(), "--site", SITE, "user", "add", u))
        if not (probe := can_submit(u))[0]:
            refused[u] = probe[1]
    assoc = sh("sacctmgr -n -P show assoc where account=gpujobs format=user").out.split()
    check("install.py user add alice, bob, dave: each is a user of account gpujobs (dave was unknown to slurm before)",
          all(j.rc == 0 for j in joined) and {"alice", "bob", "dave"} <= set(assoc), f"users in gpujobs: {assoc}")
    admin = cl(*INSTALL.split(), "--site", SITE, "admin", "add", "carol")
    if not (probe := can_submit("carol"))[0]:
        refused["carol"] = probe[1]
    level = sh("sacctmgr -n -P show user carol format=adminlevel").out.strip()
    groups = sh("id -nG carol").out
    check("install.py admin add carol: slurm AdminLevel Operator and member of gpujobs-admin",
          admin.rc == 0 and level == "Operator" and "gpujobs-admin" in groups, f"{admin.text} level={level} groups={groups.strip()}")
    check("right after user add / admin add, slurmctld accepts each user's jobs (no reload needed)", not refused,
          f"refused straight after the add: {sorted(refused)}; slurm said: {next(iter(refused.values()), '')}")
    default = sh("sacctmgr -n -P show user alice format=defaultaccount").out.strip()
    check("a user accounting already knew keeps their default account (alice: harness)", default == "harness", default)

    units = ("slurmctld", "slurmd", "slurmdbd", "mariadb", "munge", "gpujobs-agent", "gpujobs-node-exporter",
             "gpujobs-cgroup-exporter", "prometheus", "grafana-server", "ondemand-dex", "apache2")
    active = {u: sh(f"systemctl is-active {u}").out.strip() for u in units}
    check("services are active: slurm, accounting, agent, exporters, Prometheus, Grafana, Dex, Apache",
          all(v == "active" for v in active.values()), str({k: v for k, v in active.items() if v != "active"} or "all active"))
    gpu = sh("systemctl is-active gpujobs-gpu-exporter; journalctl -u gpujobs-gpu-exporter --no-pager | grep -m1 'initialize nvml'")
    check("the GPU exporter is down only because the container has no NVML library (expected here)",
          gpu.out.split()[0] in ("failed", "inactive") and "ERROR_LIBRARY_NOT_FOUND" in gpu.out, gpu.text[-200:])

    boot = wait_for(lambda: agent_json("agent.json").get("boot_reconciled_at"), 150)
    check("the agent's boot step ran once after install (agent/boot_id written, boot_reconciled_at set)", bool(boot),
          f"boot_reconciled_at={boot} marker={sh(f'cat {STATE}/agent/boot_id').out.strip()}")

    # Metrics
    jobstats = sh("jobstats --help | head -3")
    check("the jobstats command runs (its Python dependencies were installed)", jobstats.rc == 0, jobstats.text[:150])
    def scraped() -> list[str]:
        found = json.loads(sh("curl -s localhost:9090/api/v1/targets").out or "{}").get("data", {}).get("activeTargets", [])
        return sorted(t["scrapeUrl"].split("/")[2] for t in found if t["health"] == "up")

    wait_for(lambda: len(scraped()) >= SCRAPED_EXPORTERS, 150, 5)
    ups = scraped()
    check("Prometheus scrapes the node and cgroup exporters", {"testnode:9101", "testnode:9306"} <= set(ups), str(ups))
    grafana = sh("curl -s localhost:3001/api/health; echo; curl -s 'localhost:3001/api/search?query=' | grep -o '\"title\":\"[^\"]*\"'")
    check("Grafana answers on 3001 with the Jobstats dashboard and ours provisioned",
          '"database": "ok"' in grafana.out and "GPU memory by job" in grafana.out and "Single Job" in grafana.out, grafana.text[:300])

    # Grafana decision (README): plain HTTP Grafana is mixed content inside the HTTPS portal; the harness puts a TLS
    # front on 3443 and the site file's metrics.grafana_url names it.
    put("/etc/apache2/conf-available/gpujobs-grafana-tls.conf", GRAFANA_VHOST)
    front = sh("a2enconf gpujobs-grafana-tls >/dev/null && apache2ctl configtest 2>&1 | tail -1 && systemctl reload apache2")
    tls = sh(f"curl -sS -D - {GRAFANA_TLS}/api/health")
    cluster = sh("grep -A1 grafana: /etc/ood/config/clusters.d/harness.yml")
    check("Grafana reached over HTTPS for the portal: TLS front on :3443, cluster file points at it, framing allowed",
          '"database": "ok"' in tls.out and f"host: \"{GRAFANA_TLS}\"" in cluster.out and "x-frame-options" not in tls.out.lower(),
          f"{front.text[-60:]} | {cluster.text}")

    tools = sh("gpujobs-submit --help | head -2; ls -l /usr/local/bin/gpujobs-pause; ls -ld /var/www/ood/apps/sys/gpujobs")
    check("the tools and the OOD sys-app symlink are installed", tools.rc == 0 and "->" in tools.out, tools.text[-200:])


# -- stage: jobs -----------------------------------------------------------------------------------------------


def stage_jobs() -> None:
    section("jobs: chain, multi-GPU, pause and resume, who may act")
    clear_queue()
    # S1: alice submits a chain of three
    j1 = submit("alice", "chain1", 14, "--gpu", cls="restart")
    j2 = submit("alice", "chain2", 3, "--gpu", "--after", j1, cls="restart")
    j3 = submit("alice", "chain3", 3, "--gpu", "--after", j2, cls="restart")
    reason = jobinfo(j2).get("Reason", "")
    check("S1: alice submits a chain of three; the second waits on the first", bool(j1 and j2 and j3) and reason == "Dependency",
          f"jobs {j1},{j2},{j3}; job {j2} reason {reason}")
    waits = status_text("bob")
    check("S1: bob's status shows the waiting reason in plain words", f"waits for job {j1}" in waits, waits.split(f" {j2} ")[-1][:200])
    prom = wait_for(lambda: sh(f"cat {STATE}/metrics/job_{j1}.prom 2>/dev/null").out, 30, 2)  # the agent looks every 5 s
    scraped = sh("curl -s localhost:9101/metrics | grep -c gpujobs_job_cpu_memory_bytes").out.strip()
    check("S1: the agent writes the running job's metrics file, and the node exporter serves it",
          bool(prom) and "gpujobs_job_cpu_memory_bytes" in str(prom) and scraped not in ("", "0"), f"{str(prom)[:160]} scraped={scraped}")
    done = wait_for(lambda: all(state(j) == "COMPLETED" for j in (j1, j2, j3)), 120, 3)
    rows = sh(f"sacct -n -X -P -j {j1},{j2},{j3} -o JobID,State,Start,End").out.strip().splitlines()
    ordered = [r.split("|") for r in rows]
    in_order = len(ordered) == CHAIN_LENGTH and ordered[0][3] <= ordered[1][2] and ordered[1][3] <= ordered[2][2]
    check("S1: the three run one after the other and complete", bool(done) and in_order, "; ".join(rows))
    record = json.loads(sh(f"cat {STATE}/jobs/{j1}.json").out or "{}")
    log = sh(f"cat {STATE}/logs/chain1-*.log").out
    check("S1: the installed wrapper ran the job with hostguard (record has a leg with exit 0, log shows no import error)",
          record.get("legs", [{}])[0].get("exit") == 0 and "Error" not in log, f"leg={record.get('last_leg')} log={log[:120]!r}")
    ended = wait_for(lambda: f"job_{j1}.prom" not in sh(f"ls {STATE}/metrics/").out, 15, 2)  # the agent looks every 5 s
    gone = sh(f"ls {STATE}/metrics/").out.strip()
    check("S1: the agent removes the metrics file once the job has ended", bool(ended), gone or "(empty)")

    # S2: bob, one 2-GPU job
    b = submit("bob", "two-gpu", 60, "--gpus", "2")
    wait_state(b, "RUNNING", 60)
    info = jobinfo(b)
    node = nodes_alloc()
    check("S2: bob's 2-GPU job runs with two GPUs allocated", info.get("JobState") == "RUNNING" and "gres/gpu=2" in info.get("AllocTRES", "")
          and "gres/gpu=2" in node, f"job AllocTRES={info.get('AllocTRES')}")

    # S4: alice cannot pause bob's job
    refused = cl("gpujobs-pause", b, user="alice")
    check("S4: alice cannot pause bob's job (slurm refuses, the job keeps running)", refused.rc != 0 and state(b) == "RUNNING",
          f"rc={refused.rc} {refused.text[-160:]}")
    killed = cl("gpujobs-kill", b, user="alice")
    check("S4: alice cannot kill bob's job either", killed.rc != 0 and state(b) == "RUNNING", f"rc={killed.rc} {killed.text[-120:]}")

    # S5: the admin pauses it; bob sees who
    paused = cl("gpujobs-pause", b, "--reason", "e2e admin pause", user="carol")
    held = wait_for(lambda: state(b) == "PENDING" and priority(b) == 0, 60)
    seen = status_text("bob")
    check("S5: carol (admin) pauses bob's job and bob sees 'paused by carol'", paused.rc == 0 and bool(held) and "paused by carol" in seen,
          f"pause rc={paused.rc}; bob's status: {seen.split(f' {b} ')[-1][:200]}")
    check("S5: the PAUSE event's actor is carol (the file's owner)", ("carol", b, "pause") in events(b), str(events(b)))
    back = cl("gpujobs-resume", b, user="bob")
    check("S5: bob can resume the job the admin paused", back.rc == 0 and bool(wait_for(lambda: priority(b) > 0, 30)), back.text[:120])
    clear_queue()

    # S6: alice cannot run urgent
    before = sh("squeue -h | wc -l").out.strip()
    urgent = cl("gpujobs-urgent", "--project", "e2e", "--what", "no", "--time", "5:00", "--name", "nope", "--", *sleep_cmd(5), user="alice")
    after = sh("squeue -h | wc -l").out.strip()
    check("S6: alice cannot run urgent.py (refused with how to be added, nothing queued)",
          urgent.rc == 1 and "only members of gpujobs-admin may submit urgent work" in urgent.text and before == after,
          urgent.text[:200])
    direct = cl("sbatch", "--partition=urgent", "--wrap", "true", user="alice")
    dj = (re.search(r"Submitted batch job (\d+)", direct.out) or [None, ""])[1]
    time.sleep(8)
    why = jobinfo(dj).get("Reason", "")
    check("S6: alice cannot sidestep the tool: a job she sends straight to the urgent partition never runs (slurm: AllowGroups)",
          bool(dj) and state(dj) == "PENDING" and "not_in_group_permitted_to_use_this_partition" in why, f"job {dj} {state(dj)}: {why}")
    clear_queue()

    # S3: CPU-only jobs never take the cores held for GPU jobs. The harness node (4 cores, 4 GPUs) cannot hold a core per
    # GPU and leave CPU jobs one, so the cap a site's cores_per_gpu renders is set on the live partition instead. The CPU
    # jobs mix both classes: slurm counts MaxCPUsPerNode per partition, so one partition must cap them together.
    capped = sh(f"scontrol update PartitionName=cpu MaxCPUsPerNode={CPU_CAP}")
    try:
        cpu = [submit("alice", f"cpu{i}", 120, cls=cls) for i, cls in enumerate(("resume", "restart", "resume"))]
        wait_for(lambda: sum(state(j) == "RUNNING" for j in cpu) == CPU_CAP, 60)
        g = submit("alice", "gpu-beside-cpu", 30, "--gpu", "--cpus", "2", cls="restart")
        started = wait_for(lambda: state(g) == "RUNNING", 90)  # a backfill pass can take up to 30 s
        parts = {jobinfo(j).get("Partition") for j in cpu}
        check("S3: CPU-only jobs of both classes share one cpu partition capped at 2 cores; the third waits, and a 2-core "
              "GPU job still starts", capped.rc == 0 and parts == {"cpu"}
              and [state(j) for j in cpu].count("RUNNING") == CPU_CAP and bool(started),
              f"cap rc={capped.rc}; partitions {parts}; cpu {[(j, state(j), jobinfo(j).get('Reason')) for j in cpu]}; "
              f"gpu {g} {state(g)}")
    finally:
        sh("scontrol update PartitionName=cpu MaxCPUsPerNode=UNLIMITED")
    clear_queue()

    keeps_place(agent_on=True)
    keeps_place(agent_on=False)
    levels_order()


def keeps_place(*, agent_on: bool) -> None:
    """S3: alice pauses and resumes her own running job; with the agent it keeps its place in the queue."""
    tag = "agent running" if agent_on else "CONTROL, agent stopped"
    clear_queue()
    if not agent_on:
        sh("systemctl stop gpujobs-agent")
    try:
        ja = submit("alice", "place-a", 300, "--gpus", "4")
        wait_state(ja, "RUNNING", 60)
        jb = submit("bob", "place-b", 300, "--gpus", "4")
        paused = cl("gpujobs-pause", ja, "--reason", "e2e", user="alice")
        paused_at = time.monotonic()
        held = wait_for(lambda: state(ja) == "PENDING" and priority(ja) == 0, 60)
        bob_runs = wait_state(jb, "RUNNING", 60)
        check(f"S3 ({tag}): alice pauses her own job; it is held and the next job runs",
              paused.rc == 0 and bool(held) and bob_runs, f"pause rc={paused.rc}; job {ja} held={bool(held)}; job {jb} running={bob_runs}")
        jc = submit("carol", "place-c", 300, "--gpus", "1")
        resumed = cl("gpujobs-resume", ja, user="alice")
        place = placed(Level.NORMAL, ja)  # its place: level, then job id (D18)
        if agent_on:
            back = wait_for(lambda: priority(ja) == place, 40)
            check(f"S3 ({tag}): after resume the agent gives the job its place again (level, then job id)",
                  resumed.rc == 0 and bool(back) and priority(ja) > priority(jc),
                  f"job {ja}: place {place}, now {priority(ja)}; later job {jc}: {priority(jc)}")
            # A requeued job may not start until slurm's requeue delay (AuthInfo=cred_expire=60 in slurm.conf) has passed, and
            # squeue says "Resources", not "BeginTime", while the card is also busy. Free the card only after that delay, or
            # the lower-priority job simply takes the card first.
            time.sleep(max(0.0, REQUEUE_DELAY_S + 5 - (time.monotonic() - paused_at)))
            cl("scancel", jb, user="bob")
            first = wait_for(lambda: state(ja) == "RUNNING", 150, 3)
            check(f"S3 ({tag}): when the card frees, the resumed job starts before the job submitted after it",
                  bool(first) and state(jc) == "PENDING", f"job {ja} {state(ja)}, job {jc} {state(jc)}")
        else:
            time.sleep(12)
            check(f"S3 ({tag}): without the agent the released job lost its place (behind the later job)",
                  resumed.rc == 0 and priority(ja) < priority(jc),
                  f"job {ja}: place {place}, now {priority(ja)}; later job {jc}: {priority(jc)}")
    finally:
        sh("systemctl start gpujobs-agent")
        clear_queue()


def levels_order() -> None:
    """D18: waiting jobs go by level, then job id; an owner may move their own waiting job; a slurmctld restart keeps
    every place; a QOS made while slurmctld runs is usable at once."""
    clear_queue()
    try:
        blocker = submit("alice", "lv-blocker", 300, "--gpus", "4")
        wait_state(blocker, "RUNNING", 60)
        low = submit("bob", "lv-low", 60, "--gpu", "--level", "low")
        normal = submit("carol", "lv-normal", 60, "--gpu")
        high = submit("alice", "lv-high", 60, "--gpu", "--level", "high")
        later = submit("bob", "lv-normal-2", 60, "--gpu")
        want = {low: Level.LOW, normal: Level.NORMAL, high: Level.HIGH, later: Level.NORMAL}
        ready = wait_for(lambda: all(priority(j) == placed(level, j) for j, level in want.items()), 30)
        seen = {j: priority(j) for j in want}
        check("L1: the agent places waiting jobs by level, then job id (high, normal, the later normal, low), all above "
              "any priority slurm gives by itself", bool(ready) and seen[high] > seen[normal] > seen[later] > seen[low] > TOP_PRIORITY,
              str(seen))
        moved = cl("scontrol", "update", f"jobid={low}", "qos=high", user="bob")
        up = wait_for(lambda: priority(low) == placed(Level.HIGH, low), 30)
        check("L2: bob moves his own waiting job to high; the agent places it among the high jobs, ahead of the later one",
              moved.rc == 0 and bool(up) and priority(low) > priority(high) > priority(normal),
              f"rc={moved.rc} {moved.text[:120]}; {low}: {priority(low)}, {high}: {priority(high)}")
        # the cold review's case: an owner's own hold and release must not put a job above the placed ones
        cl("scontrol", "hold", later, user="bob")
        cl("scontrol", "release", later, user="bob")
        released = priority(later)
        check("L5: bob holds and releases his own waiting job: slurm ranks it afresh, behind every placed job, until the "
              "agent places it again", 0 < released <= TOP_PRIORITY < min(priority(j) for j in (low, normal, high))
              and bool(wait_for(lambda: priority(later) == placed(Level.NORMAL, later), 30)), f"released at {released}")
        before = {j: priority(j) for j in want}
        sh("systemctl stop gpujobs-agent")  # so only slurm itself can keep the order across its restart
        sh("systemctl restart slurmctld")
        wait_for(lambda: sh("scontrol ping").rc == 0 and state(blocker) == "RUNNING", 90)
        after = {j: priority(j) for j in want}
        check("L3: a slurmctld restart keeps every placed priority (slurm saves them; the agent is stopped)",
              after == before, f"before {before} after {after}")
        made = sh("sacctmgr -i add qos e2elevel")
        queued = cl("sbatch", "--qos=e2elevel", "--hold", "--wrap", "true", user="alice")
        qj = (re.search(r"Submitted batch job (\d+)", queued.out) or [None, ""])[1]
        # slurm accepts an unknown QOS name without QOS enforcement (and leaves the job with none): the check is the
        # QOS the job actually has
        qos = sh(f"squeue -h -j {qj} -o %q").out.strip() if qj else ""
        check("L4: a QOS made while slurmctld runs is the next job's QOS at once (no restart)",
              made.rc == 0 and qos == "e2elevel", f"add rc={made.rc}; job {qj} qos {qos!r}; sbatch: {queued.text[:120]}")
    finally:
        sh("systemctl start gpujobs-agent")
        clear_queue()
        sh("sacctmgr -i delete qos e2elevel; true")


# -- stage: urgent ---------------------------------------------------------------------------------------------


def stage_urgent() -> None:
    section("urgent: the admin's job pre-empts a normal job; the agent restores the order")
    clear_queue()
    n1 = submit("alice", "victim", 600, "--gpus", "4")
    wait_state(n1, "RUNNING", 60)
    n2 = submit("bob", "bystander", 600, "--gpus", "1")
    wait_for(lambda: priority(n2) == placed(Level.NORMAL, n2), 30)  # the agent places the waiting job
    out = cl("gpujobs-urgent", "--project", "e2e", "--what", "e2e urgent", "--time", "5:00", "--name", "urgent1", "--",
             *sleep_cmd(25), user="carol", timeout=120)
    uj = (re.search(r"job (\d+) submitted", out.out) or [None, ""])[1]
    check("S7: carol (admin) runs urgent.py; it names the job it will pre-empt and holds the waiting job",
          out.rc == 0 and bool(uj) and n1 in out.out and f"held ['{n2}']" in out.out, out.text[-300:])
    request = sh(f"ls {STATE}/requests/ && cat {STATE}/requests/*")
    check("S7: the restore-order request is a file owned by carol", "restore_order-" in request.out and
          sh(f"stat -c %U {STATE}/requests/restore_order-*").out.strip() == "carol", request.text[:200])
    check("S7: the held job has an URGENT_HOLD event by carol", ("carol", n2, "urgent_hold") in events(n2), str(events(n2)))
    started = wait_for(lambda: state(uj) == "RUNNING", 180, 3)
    requeued = wait_for(lambda: state(n1) == "PENDING", 60, 2)  # the victim's processes end (COMPLETING) before it is requeued
    check("S7: the urgent job pre-empts the normal job (the victim is requeued, not running)",
          bool(started) and bool(requeued), f"urgent {uj} {state(uj)}; victim {n1} {state(n1)} ({jobinfo(n1).get('Reason')})")
    check("S7: the waiting job stays held while the urgent job runs", state(n2) == "PENDING" and priority(n2) == 0,
          f"{n2}: {state(n2)} priority {priority(n2)} ({jobinfo(n2).get('Reason')})")
    wait_state(uj, "COMPLETED", 120)
    ran_first = wait_for(lambda: state(n1) == "RUNNING", 240, 3)
    n2_state = state(n2)
    never_ran = sh(f"sacct -n -X -P -j {n2} -o State,Start").out.strip()
    check("S7: after the urgent job the pre-empted job runs again BEFORE the held job (restore order; the held job has not started)",
          bool(ran_first) and n2_state == "PENDING" and "Unknown" in never_ran, f"victim {n1} {state(n1)}; bystander {n2} {n2_state}, sacct {never_ran}")
    released = wait_for(lambda: priority(n2) > 0 and sh("ls " + STATE + "/requests/ | wc -l").out.strip() == "0", 90, 3)
    log = sh("journalctl -u gpujobs-agent --no-pager | grep 'released'").out
    check("S7: the agent (not the tool) releases the held job, gives it its place back and deletes the request",
          bool(released) and n2 in log and priority(n2) == placed(Level.NORMAL, n2),
          f"{n2} priority {priority(n2)}; agent log: {log.strip().splitlines()[-1:] }")
    clear_queue()


# -- stage: service --------------------------------------------------------------------------------------------------


def stage_service() -> None:
    section("service: a service's CPU stand-in starts at once by suspending a cpu job, never a GPU job (D20)")
    clear_queue()
    flags = ("--service", "--cpus", "2")
    outsider = cl("gpujobs-submit", "--project", "e2e", "--what", "e2e outsider", "--class", "restart", "--mem", "200M",
                  *flags, "--", *sleep_cmd(5), user="alice")
    check("S20: alice, outside the service group, is refused by the tool (slurm would hold her job forever)",
          outsider.rc != 0 and "only members of gpujobs-service" in outsider.text, outsider.text[-200:])
    added = sh(f"{INSTALL} --site {SITE} service add bob")
    capped = sh(f"scontrol update PartitionName=cpu MaxCPUsPerNode={CPU_CAP}")
    counter = ["python3", "-c", "import time\nfor i in range(10**6):\n open('/tmp/e2e-frozen', 'w').write(str(i)); time.sleep(1)"]
    count = lambda: int(sh("cat /tmp/e2e-frozen 2>/dev/null || echo -1").out.strip() or -1)  # noqa: E731
    try:
        c = submit("alice", "frozen", 0, "--cpus", "2", command=counter)
        g = submit("alice", "gpu-untouched", 600, "--gpu", "--cpus", "2")
        full = wait_for(lambda: state(c) == "RUNNING" and state(g) == "RUNNING", 90)
        w = submit("alice", "waits", 600, "--cpus", "2")  # the cpu cap is full: it waits, and must not take the room
        ask = cl("gpujobs-submit", "--project", "e2e", "--what", "e2e ask", "--class", "restart", "--mem", "200M", *flags,
                 "--dry-run", "--", "true", user="bob")
        line = ask.out.splitlines()[0] if ask.out else ""
        told = json.loads(line.removeprefix("[submit] service room: ")) if line.startswith("[submit] service room: ") else {}
        check("S20: bob's dry run names the cpu job a service job would suspend (D9: its user is asked first)",
              told.get("starts") == "after_suspending" and [x.get("job") for x in told.get("suspends", [])] == [c],
              ask.text[:300])
        n = submit("bob", "no-consent", 25, *flags, "--no-suspend", cls="restart")
        time.sleep(12)  # two agent steps
        check("S20: with --no-suspend the service job waits and nothing is suspended",
              state(n) == "PENDING" and state(c) == "RUNNING", f"{n} {state(n)}; cpu {c} {state(c)}")
        sh(f"scancel {n}")
        s = submit("bob", "standin", 25, *flags, "--suspend", c, cls="restart")  # the node's 4 cores are all taken
        started = wait_for(lambda: state(s) == "RUNNING", 60)
        frozen = wait_state(c, "SUSPENDED", 30)
        time.sleep(3)  # slurm sends SIGTSTP, then SIGSTOP 2 s later (suspend_grace_time): the counter may tick once
        before = count()
        time.sleep(5)
        check("S20: bob, just added (install.py service add, no wait), starts a service job at once on a full node: the "
              "agent suspends alice's cpu job (its counter stops), holds her waiting one, and her GPU job keeps running",
              added.rc == 0 and capped.rc == 0 and bool(full) and bool(started) and frozen and before == count() >= 0
              and state(g) == "RUNNING" and jobinfo(s).get("Partition") == "service" and priority(w) == 0,
              f"add rc={added.rc} {added.text[-120:]}; standin {s} {state(s)} ({jobinfo(s).get('Reason')}); cpu {c} "
              f"{state(c)} counter {before}->{count()}; gpu {g} {state(g)}; waiting {w} {state(w)} priority {priority(w)}")
        ended = wait_state(s, "COMPLETED", 90)
        resumed = wait_state(c, "RUNNING", 60)
        time.sleep(4)
        after = count()
        released = wait_for(lambda: priority(w) > 0, 30)
        check("S20: when the stand-in ends the agent resumes the cpu job, which lost nothing (its counter carries on), and "
              "releases the waiting one", ended and resumed and after > before and bool(released),
              f"standin {state(s)}; cpu {state(c)} counter {before}->{after}; waiting {w} priority {priority(w)}")
    finally:
        sh("scontrol update PartitionName=cpu MaxCPUsPerNode=UNLIMITED")
        clear_queue()


# -- stage: failure ------------------------------------------------------------------------------------------------


def stage_failure() -> None:
    section("failure: a failed job writes a FAILED event and calls the notify command")
    clear_queue()
    sh("rm -f /var/tmp/harness-notify.log")
    j = submit("alice", "fails", 0, "--gpu", cls="restart", command=["python3", "-c", "raise SystemExit(3)"])
    wait_state(j, "FAILED", 90)
    seen = wait_for(lambda: ("slurm", j, "failed") in events(j), 60, 3)
    body = sh(f"cat {STATE}/events/{j}-*-failed.json").out
    check("S8: the failed job gets a FAILED event, written by the agent (owner slurm)", bool(seen) and '"state": "FAILED"' in body,
          f"{events(j)} {' '.join(body.split())[:120]}")
    note = wait_for(lambda: f"{j} FAILED alice" in sh("cat /var/tmp/harness-notify.log 2>/dev/null").out, 30, 2)
    check("S8: notify_command ran with the job in its environment", bool(note), sh("cat /var/tmp/harness-notify.log").out.strip())
    fine = submit("alice", "succeeds", 0, "--gpu", cls="restart", command=["python3", "-c", "pass"])
    wait_state(fine, "COMPLETED", 90)
    time.sleep(2 * AGENT_LOOP_S + 2)  # two agent loops: long enough to have written an event, were it going to
    check("S8: a job that completes normally writes no FAILED event", state(fine) == "COMPLETED" and not events(fine), str(events(fine)))
    oom_guard_kills()


# The run saves on SIGTERM and exits 0, as Lightning does (job 428, 2026-10-22); a second leg finishes at once.
SAVES_ON_TERM = ("import os, signal, sys, time; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
                 "time.sleep(0 if int(os.environ.get('SLURM_RESTART_COUNT') or 0) else 300)")
# A stand-in for earlyoom: the unit name and the line it logs before each kill are earlyoom's own (kill.c, v1.7).
FAKE_GUARD = 'echo "sending SIGTERM to process $1 uid $(stat -c %u /proc/$1) \\"python3\\": badness 900, VmRSS 10 MiB"; kill -TERM "$1"\n'


def oom_guard_kills() -> None:
    section("failure: a run the host OOM guard killed never passes as finished")
    put("/var/tmp/fake-earlyoom.sh", FAKE_GUARD, "0755")
    for cls in ("resume", "restart"):
        j = submit("alice", f"oom-{cls}", 0, cls=cls, command=["python3", "-c", SAVES_ON_TERM])
        wait_state(j, "RUNNING", 90)
        time.sleep(2 * AGENT_LOOP_S)  # long enough for the agent to have seen the job's processes
        pid = sh("pgrep -u alice -f '^python3 -c import os, signal'").out.split()[:1]
        sh(f"systemd-run --unit=earlyoom --collect --wait /bin/sh /var/tmp/fake-earlyoom.sh {pid[0] if pid else 0}")
        leg = lambda: json.loads(sh(f"cat {STATE}/jobs/{j}.json").out or "{}").get("legs") or [{}]  # noqa: E731
        if cls == "resume":
            requeued = wait_for(lambda: jobinfo(j).get("Restarts") == "1", 60)
            finished = wait_state(j, "COMPLETED", REQUEUE_DELAY_S + 90)
            check("OOM: a killed resume job is requeued, the kill is in its record, and its next leg finishes",
                  bool(pid) and bool(requeued) and finished and leg()[0].get("killed_by", {}).get("pid") == int(pid[0]),
                  f"pid {pid} {jobinfo(j).get('JobState')} restarts {jobinfo(j).get('Restarts')} legs {leg()}")
        else:
            failed = wait_state(j, "FAILED", 60)
            check("OOM: a killed restart job fails with the wrapper's KILLED_FOR_MEMORY, not exit 0",
                  failed and jobinfo(j).get("ExitCode") == "5:0", f"{jobinfo(j).get('JobState')} {jobinfo(j).get('ExitCode')}")


# -- stage: recovery -----------------------------------------------------------------------------------------------


def job_pids(job: str) -> list[str]:
    """Pids slurm lists for the job (empty once its processes are gone)."""
    return [line.split()[0] for line in sh(f"scontrol listpids {job} 2>/dev/null").out.splitlines()[1:] if line.strip()]


def lose_the_node(job: str, *, agent: bool) -> float:
    """What a reboot does to a running job: everything stops, the job's processes are gone, the boot marker is old.

    Only the job's own processes are killed. `pkill -9 slurmstepd` would also kill the `slurmstepd infinity` helper that
    keeps slurmd's systemd scope alive, and slurmd started straight after then raced the scope's removal: in one run it
    came up without a scope and every later job died at launch ("testnode_slurmstepd.scope/cgroup.controllers: No such
    file").
    """
    pids = job_pids(job)
    sh("systemctl stop gpujobs-agent slurmd slurmctld")
    sh("kill -9 " + " ".join(pids) + "; true")
    sh(f"rm -f {STATE}/agent/boot_id")
    sh("systemctl start slurmctld slurmd")
    started = time.monotonic()
    if agent:
        sh("systemctl start gpujobs-agent")
    return started


def stage_recovery() -> None:
    section("recovery: slurm restarts")
    clear_queue()
    # S9 control: without the agent
    j = submit("alice", "reboot-control", 600, "--gpu")
    wait_state(j, "RUNNING", 60)
    t0 = lose_the_node(j, agent=False)
    time.sleep(100)
    s = jobinfo(j)
    check("S9 (CONTROL, agent stopped): 100 s after the restart slurm still thinks the dead job is RUNNING",
          s.get("JobState") == "RUNNING" and not job_pids(j), f"state={s.get('JobState')} pids={job_pids(j)} after {time.monotonic() - t0:.0f} s")
    sh("systemctl start gpujobs-agent")
    clear_queue()

    # S9
    j = submit("alice", "reboot", 600, "--gpu")
    wait_state(j, "RUNNING", 60)
    first = int(jobinfo(j).get("Restarts", "0"))
    t0 = lose_the_node(j, agent=True)
    requeued = wait_for(lambda: int(jobinfo(j).get("Restarts", "0")) > first, 120, 2)
    took = time.monotonic() - t0
    pending = wait_for(lambda: state(j) == "PENDING", 30, 2)
    s = jobinfo(j)
    beat = agent_json("agent.json")
    check("S9: after slurmctld+slurmd restart with the job's processes killed, the agent's boot step requeues the job at once",
          bool(requeued) and took < REQUEUE_WITHIN_S and bool(pending),
          f"{took:.0f} s after the restart: state={s.get('JobState')} reason={s.get('Reason')} restarts {first}->{s.get('Restarts')}; "
          f"agent boot_reconciled_at={beat.get('boot_reconciled_at')}")
    again = wait_for(lambda: state(j) == "RUNNING" and job_pids(j), 200, 3)
    check("S9: the requeued job starts again by itself once slurm's requeue delay is over (leg 1 has processes)", bool(again),
          f"state={state(j)} pids={job_pids(j)}")
    clear_queue()

    # S10: slurmctld only
    j = submit("alice", "ctld-only", 600, "--gpu")
    wait_state(j, "RUNNING", 60)
    pids = sorted(job_pids(j))
    before = jobinfo(j)
    sh("systemctl restart slurmctld")
    time.sleep(50)  # past the agent's 30-s settle time: it must not touch a node whose boot it has already handled
    after = jobinfo(j)
    alive = sorted(job_pids(j))
    check("S10: a slurmctld-only restart keeps the live job running (same processes, no requeue)",
          after.get("JobState") == "RUNNING" and after.get("Restarts") == before.get("Restarts") and pids == alive and bool(pids),
          f"state={after.get('JobState')} restarts={after.get('Restarts')} pids {pids} -> {alive}")
    clear_queue()


# -- stage: portal -----------------------------------------------------------------------------------------------

CURL = "curl -sS --max-time 60"


def login(user: str, password: str | None = None) -> tuple[bool, str]:
    jar = f"/tmp/cj-{user}"
    sh(f"rm -f {jar}")
    url = sh(f"{CURL} -L -c {jar} -b {jar} -o /dev/null -w '%{{url_effective}}' {PORTAL}/").out.strip()
    got = sh(
        f"{CURL} -L -c {jar} -b {jar} -o /tmp/login-{user}.html -w '%{{http_code}} %{{url_effective}}' "
        f"--data-urlencode login={user}@harness.test --data-urlencode password={shlex.quote(password or PASSWORDS[user])} {shlex.quote(url)}"
    ).out.strip()
    # the portal opens on the jobs page (root_uri); on a fresh install OOD first shows its 'Initialize App' page there
    code, _, landed = got.partition(" ")
    first_visit = code == "404" and "/nginx/init" in sh(f"cat /tmp/login-{user}.html").out
    return (code == "200" or first_visit) and landed.endswith("/pun/sys/gpujobs/"), f"{url.split('?')[0]} -> {got}"


def get(user: str, path: str, *extra: str) -> tuple[int, str]:
    out = sh(f"{CURL} -L -b /tmp/cj-{user} -c /tmp/cj-{user} -w '\\n%{{http_code}}' {' '.join(extra)} {shlex.quote(APP + path)}").out
    body, _, code = out.rpartition("\n")
    return int(code or 0), body


def post(user: str, path: str, origin: str | None, body: str = "{}") -> tuple[int, str]:
    hdr = f"-H {shlex.quote('Origin: ' + origin)}" if origin else ""
    out = sh(f"{CURL} -b /tmp/cj-{user} -w '\\n%{{http_code}}' -X POST -H 'Content-Type: application/json' {hdr} "
             f"-d {shlex.quote(body)} {shlex.quote(APP + path)}").out
    text, _, code = out.rpartition("\n")
    return int(code or 0), text


def stage_portal() -> None:
    section("portal: Open OnDemand over HTTPS with a local Dex account")
    clear_queue()
    anon = sh(f"{CURL} -o /dev/null -w '%{{http_code}} %{{redirect_url}}' {APP}/").out
    check("an anonymous request is sent to the login, not to the app", anon.startswith("302") and "/dex/auth" in anon, anon[:120])
    wrong, evidence = login("alice", "not-the-password")
    check("Dex local login: a wrong password does not log in", not wrong, evidence)
    for user in USERS:
        good, evidence = login(user)
        check(f"Dex local login over HTTPS: {user} logs in and lands on the jobs page (REMOTE_USER maps to Linux user {user})", good, evidence)
    # the first visit to a new app in a user's own nginx shows OOD's 'Initialize App' page
    first, body = get("alice", "/")
    check("first visit of the app in alice's per-user nginx asks to initialize it (OOD behaviour, README)",
          first == HTTPStatus.NOT_FOUND and "/nginx/init" in body, f"HTTP {first}")
    for user in USERS:
        init = sh(f"{CURL} -L -b /tmp/cj-{user} -c /tmp/cj-{user} -o /dev/null -w '%{{http_code}}' '{PORTAL}/nginx/init?redir=%2fpun%2fsys%2fgpujobs%2f'").out
        check(f"{user}: initializing the app works", init.strip() == "200", init)

    link = sh("readlink /var/www/ood/apps/sys/gpujobs; grep passenger_app_root /var/lib/ondemand-nginx/config/apps/sys/gpujobs.conf")
    page_code, page = get("alice", "/")
    check("symlinked OOD sys app: Passenger serves the app through /var/www/ood/apps/sys/gpujobs -> install_dir",
          page_code == HTTPStatus.OK and "Signed in as <strong>alice</strong>" in page and "/opt/gpujobs/slurm/ood_app" in link.out, link.text[:200])

    # Passenger runs our bin/python: put a marker in the installed launcher, restart alice's app, look
    sh("cp -p /opt/gpujobs/slurm/ood_app/bin/python /root/bin-python.orig && printf '%s\\n' '#!/bin/sh' "
       "'echo \"cwd=$PWD args=$*\" >> /tmp/passenger-python-marker' 'exec /usr/bin/python3 \"$@\"' > /opt/gpujobs/slurm/ood_app/bin/python")
    sh("rm -f /tmp/passenger-python-marker; pkill -u alice -f wsgi-loader; true")
    get("alice", "/")
    marker = wait_for(lambda: sh("cat /tmp/passenger-python-marker 2>/dev/null").out.strip(), 30, 2)
    sh("install -m 0755 /root/bin-python.orig /opt/gpujobs/slurm/ood_app/bin/python")
    check("Passenger starts the app with the app's own bin/python (a marker added to it was written, cwd = the app root)",
          bool(marker) and "cwd=/var/www/ood/apps/sys/gpujobs" in str(marker), str(marker)[:200])
    restored = sh(f"{INSTALL} --site {SITE} check")
    check("the installed launcher is back as installed (check is clean)", restored.rc == 0, restored.text[:200])
    get("alice", "/")

    # an alice job, a bob job
    aj = submit("alice", "portal-a", 600, "--gpus", "1")
    bj = submit("bob", "portal-b", 600, "--gpus", "1")
    wait_state(aj, "RUNNING", 60)
    wait_state(bj, "RUNNING", 60)
    code, jobs = get("alice", "/api/jobs")
    listed = {j["id"]: j for j in json.loads(jobs).get("jobs", [])} if code == HTTPStatus.OK else {}
    check("the app lists every user's jobs, with the project and description from the record",
          aj in listed and bj in listed and listed.get(bj, {}).get("user") == "bob" and listed.get(bj, {}).get("project") == "e2e"
          and listed.get(bj, {}).get("description") == "e2e portal-b", json.dumps(listed.get(bj, {}))[:250])
    for path, needle in (("/", "GPU jobs"), ("/history", "History"), ("/projects", "GPU-hours"), (f"/job/{aj}", "Dependencies"),
                         ("/static/app.js", "function")):
        code, body = get("alice", path)
        check(f"the app's page {path} loads (200)", code == HTTPStatus.OK and needle in body, f"HTTP {code}, {len(body)} bytes")
    code, own = get("alice", f"/job/{aj}")
    _, other = get("alice", f"/job/{bj}")
    check("S4: the app shows alice's buttons on her own job and hides them on bob's",
          'data-action="pause"' in own and 'data-action="pause"' not in other and "owner or an admin can pause" in other,
          f"own has pause button={'data-action=\"pause\"' in own}; bob's job page says who can act={'owner or an admin can pause' in other}")
    forced = post("alice", f"/api/job/{bj}/pause", PORTAL)
    check("S4: even if the button is bypassed, a pause of bob's job through the portal is refused by slurm and bob's job runs on",
          forced[0] == HTTPStatus.OK and '"ok": false' in forced[1] and state(bj) == "RUNNING", f"HTTP {forced[0]} {forced[1][:200]}")
    code, admin_view = get("carol", f"/job/{bj}")
    check("the app shows an admin the buttons on anyone's job ('Signed in as carol (admin)')",
          'data-action="pause"' in admin_view and "<strong>carol</strong>" in admin_view and 'class="role">admin' in admin_view, f"HTTP {code}")

    # Origin / Host reach the app unchanged
    evil = post("alice", f"/api/job/{aj}/pause", "https://evil.example")
    none = post("alice", f"/api/job/{aj}/pause", None)
    check("Origin reaches the app: a foreign Origin and a missing Origin are both refused (403) and the job is untouched",
          evil[0] == HTTPStatus.FORBIDDEN and none[0] == HTTPStatus.FORBIDDEN and state(aj) == "RUNNING", f"foreign: {evil[0]} {evil[1][:90]}; none: {none[0]}")
    dry = get("alice", f"/api/job/{aj}/pause-preview")  # while it runs: a pending job's pause loses nothing
    check("the pause preview shows what a pause loses", dry[0] == HTTPStatus.OK and "resumes from its last checkpoint" in dry[1], dry[1][:160])
    ok_pause = post("alice", f"/api/job/{aj}/pause", PORTAL)
    held = wait_for(lambda: state(aj) == "PENDING" and priority(aj) == 0, 60)
    check("Host reaches the app unchanged: a pause POST whose Origin equals the portal's host is accepted, and the job is held",
          ok_pause[0] == HTTPStatus.OK and '"ok": true' in ok_pause[1] and bool(held), f"HTTP {ok_pause[0]} {ok_pause[1][:120]}")
    check("the portal pause is recorded as alice's (the app runs as the viewing user)", ("alice", aj, "pause") in events(aj), str(events(aj)))
    ok_resume = post("alice", f"/api/job/{aj}/resume", PORTAL)
    back = wait_for(lambda: priority(aj) > 0, 40)
    check("resume through the portal releases the job", ok_resume[0] == HTTPStatus.OK and '"ok": true' in ok_resume[1] and bool(back), f"{ok_resume[1][:100]} priority={priority(aj)}")
    clear_queue()


# -- stage: proxy -------------------------------------------------------------------------------------------------

PROXY_KEY_FILE = "/etc/gpujobs/portal/proxy_key"
PROXY_SITE = "/root/proxy.site.toml"
PROBE_APP = "/var/www/ood/apps/sys/hdrprobe"
# A throwaway portal app that answers with the request headers it received (the real app has no such page).
PROBE_WSGI = """import json
def application(environ, start_response):
    headers = sorted(k for k in environ if k.startswith("HTTP_"))
    body = json.dumps({"headers": headers}).encode()
    start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
    return [body]
"""
PROBE_MANIFEST = "---\nname: Header probe\ncategory: Test\ndescription: Echoes the request header names.\n"


def stage_proxy() -> None:
    section("proxy: the portal with no sign-in page, behind a trusted proxy's key (switches the portal from Dex)")
    sh(f"openssl rand -hex 32 | install -m 0600 /dev/stdin {PROXY_KEY_FILE}")
    key = sh(f"cat {PROXY_KEY_FILE}").out.strip()
    own = sh("hostname -i").out.split()[0]
    site = Path(HERE / "harness.site.toml").read_text()
    site = site.replace('auth = "dex-local"', f'auth = "proxy"\nallow_from = ["{own}/32"]  # the proxy: this container')
    site = site.replace('users_file = "/etc/gpujobs/portal/users.json"', f'user = "alice"\nkey_file = "{PROXY_KEY_FILE}"')
    put(PROXY_SITE, site)

    dex_before = sh("systemctl is-active ondemand-dex").out.strip()
    loose = sh(f"chmod 0644 {PROXY_KEY_FILE}; {INSTALL} --site {PROXY_SITE} apply --component portal; rc=$?; chmod 0600 {PROXY_KEY_FILE}; exit $rc")
    check("a key file others can read is refused, and nothing is applied", loose.rc != 0 and "must be root-only" in loose.text, loose.text[-200:])
    applied = sh(f"{INSTALL} --site {PROXY_SITE} apply --component portal", timeout=900)
    check("apply of the portal with auth = proxy succeeds", applied.rc == 0 and "installed" in applied.out, applied.text[-300:])
    dex_after = sh("systemctl is-active ondemand-dex; systemctl is-enabled ondemand-dex")
    check("Dex was running before and is stopped and disabled by the switch",
          dex_before == "active" and dex_after.out.split() == ["inactive", "disabled"], f"before={dex_before} after={dex_after.text}")
    apache = sh("systemctl is-active apache2")
    conf = sh("stat -c '%a %U:%G' /etc/apache2/sites-available/ood-portal.conf; "
              f"grep -c {key} /etc/apache2/sites-available/ood-portal.conf; "
              "grep -c \"sha1(req('X-Frontdoor-Key'))\" /etc/apache2/sites-available/ood-portal.conf; "
              "grep -c -i -E 'openid|oidc|\\bdex\\b' /etc/apache2/sites-available/ood-portal.conf")
    mode, wrote_key, gates, login = (conf.out.split("\n") + ["", "", "", ""])[:4]
    check("Apache is up with the allow_from restriction; the portal config is 0640 root:www-data, holds the key's digest but not the key, "
          "and has no OIDC or Dex directives", apache.out.strip() == "active" and mode == "640 root:www-data"
          and wrote_key == "0" and int(gates or 0) >= 2 and login == "0", f"{apache.out.strip()} | {conf.text}")

    base = f"{CURL} -o /dev/null -w '%{{http_code}}'"
    app = f"{PORTAL}/pun/sys/gpujobs/api/jobs"
    none = sh(f"{base} {app}").out
    wrong = sh(f"{base} -H 'X-Frontdoor-Key: {'0' * 64}' {app}").out
    empty = sh(f"{base} -H 'X-Frontdoor-Key:' {app}").out
    public = sh(f"{base} {PORTAL}/public/favicon.ico").out
    init = sh(f"{base} {PORTAL}/nginx/init?redir=%2fpun%2fsys%2fgpujobs%2f").out
    check("no key, a wrong key and an empty key are each refused with 403, on the app, /public and /nginx",
          [none, wrong, empty, public, init] == ["403"] * 5, f"none={none} wrong={wrong} empty={empty} public={public} nginx={init}")

    keyed = f"-H 'X-Frontdoor-Key: {key}'"
    sh(f"{base} {keyed} '{PORTAL}/nginx/init?redir=%2fpun%2fsys%2fgpujobs%2f' -L")  # first visit in a restarted per-user server
    page = sh(f"{CURL} -L {keyed} -w '\\n%{{http_code}} %{{url_effective}}' {APP}/").out
    body, _, tail = page.rpartition("\n")
    check("with the key, the portal opens straight on the jobs page: no login page, no redirect to Dex",
          tail.startswith("200") and tail.endswith("/pun/sys/gpujobs/") and "Signed in as <strong>alice</strong>" in body, tail)
    code, _, jobs = sh(f"{CURL} {keyed} -w '\\n%{{http_code}}' {app}").out.rpartition("\n")[::-1]
    check("with the key, /api/jobs returns JSON, served as the configured user", code == "200" and isinstance(json.loads(jobs).get("jobs"), list),
          f"HTTP {code} {jobs[:80]}")
    # OOD runs each per-user nginx master as root, labelled with its user: "nginx: master process (alice) -c ..."
    puns = sorted(set(re.findall(r"nginx: master process \((\S+)\)", sh("ps -eo args").out)))
    check("the app runs as the configured user only (no per-user server for bob, carol or www-data was started by these requests)",
          "alice" in puns and not {"bob", "carol", "dave"} & set(puns), f"users with a per-user server: {puns}")

    # the header must not reach the app: a probe app that lists the HTTP_* variables it received
    put(f"{PROBE_APP}/manifest.yml", PROBE_MANIFEST)
    put(f"{PROBE_APP}/passenger_wsgi.py", PROBE_WSGI)
    put(f"{PROBE_APP}/bin/python", "#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", "0755")
    sh(f"{CURL} -L -o /dev/null {keyed} '{PORTAL}/nginx/init?redir=%2fpun%2fsys%2fhdrprobe%2f'")
    echoed = sh(f"{CURL} {keyed} -H 'X-Probe: kept' {PORTAL}/pun/sys/hdrprobe/").out
    seen = json.loads(echoed).get("headers", []) if echoed.startswith("{") else []
    check("the key header does not reach the app (a control header sent with it does)",
          "HTTP_X_PROBE" in seen and "HTTP_X_FRONTDOOR_KEY" not in seen, f"{echoed[:200]}")
    sh(f"rm -rf {PROBE_APP}; /opt/ood/nginx_stage/sbin/nginx_stage nginx_clean --force")


# -- stage: ssh ---------------------------------------------------------------------------------------------------

RELAY_KEY_FILE = "/etc/gpujobs/portal/relay_key"
SSH_SITE = "/root/ssh.site.toml"
RELAY = "/run/gpujobs-portal.sock"
LOCAL = "http://localhost:8080"  # what a person's browser asks for, through `LocalForward 8080 <RELAY>`


def through_relay(user: str, path: str, *extra: str) -> tuple[str, str]:
    """(HTTP status, body) of a request `user` sends to the relay socket, as their ssh forward would (the container
    has no sshd: on the Turing box sshd was measured opening the socket as the person, 2026-10-21)."""
    out = sh(f"{CURL} --unix-socket {RELAY} -L -w '\\n%{{http_code}}' {' '.join(extra)} {shlex.quote(LOCAL + path)}", user=user).out
    body, _, code = out.rpartition("\n")
    return code, body


def stage_ssh() -> None:
    section("ssh: the portal behind the relay, where the account that opens the connection is the sign-in (D19)")
    sh(f"openssl rand -hex 32 | install -m 0600 /dev/stdin {RELAY_KEY_FILE}")
    key = sh(f"cat {RELAY_KEY_FILE}").out.strip()
    site = Path(HERE / "harness.site.toml").read_text()
    site = site.replace('auth = "dex-local"', 'auth = "ssh"\nallow_from = ["127.0.0.0/8"]  # only the relay, on the box')
    site = site.replace('server_name = "testnode"', 'server_name = "localhost"')
    site = site.replace('users_file = "/etc/gpujobs/portal/users.json"', f'key_file = "{RELAY_KEY_FILE}"')
    put(SSH_SITE, site)
    put("/etc/systemd/system/gpujobs-portal-relay.service.d/99-harness-container.conf", DROPIN_DYNAMIC)
    applied = sh(f"{INSTALL} --site {SSH_SITE} apply --component tools --component portal", timeout=900)
    check("apply of the portal with auth = ssh succeeds", applied.rc == 0 and "installed" in applied.out, applied.text[-300:])
    units = sh("systemctl is-active apache2 gpujobs-portal-relay.socket").out.split()
    conf = sh("cat /etc/apache2/sites-available/ood-portal.conf").out
    check("Apache and the relay socket are up; the portal config has no TLS, holds the key's digest only, and takes the "
          "user only with the key", units == ["active", "active"] and "SSLEngine" not in conf and key not in conf
          and "SetEnvIfExpr" in conf and "X-Gpujobs-User" in conf, f"{units} | {conf.count('SetEnvIfExpr')}")

    direct = f"{CURL} -o /dev/null -w '%{{http_code}}' -H 'Host: localhost:8080' http://127.0.0.1/pun/sys/gpujobs/"
    plain, forged = sh(direct, user="bob").out, sh(f"{direct} -H 'X-Gpujobs-User: alice'", user="bob").out
    check("Apache straight, without the relay, is refused with 403 (not redirected), also with a claimed user", [plain, forged] == ["403"] * 2,
          f"plain={plain} forged={forged}")

    for user in ("alice", "bob"):
        through_relay(user, "/nginx/init?redir=%2fpun%2fsys%2fgpujobs%2f")  # first visit starts the per-user server
    pages = {user: through_relay(user, "/pun/sys/gpujobs/", "-H 'X-Gpujobs-User: carol'") for user in ("alice", "bob")}
    check("through the relay each person is signed in as themselves, with no login page, whatever user they claim",
          all(code == "200" and f"Signed in as <strong>{user}</strong>" in body and "carol" not in body
              for user, (code, body) in pages.items()), str({u: (c, b[:80]) for u, (c, b) in pages.items()}))
    code, jobs = through_relay("alice", "/pun/sys/gpujobs/api/jobs")
    check("through the relay /api/jobs returns JSON", code == "200" and isinstance(json.loads(jobs or "{}").get("jobs"), list),
          f"HTTP {code} {jobs[:80]}")
    root_code, _ = through_relay("root", "/pun/sys/gpujobs/")
    check("root is refused at the relay", root_code == "403", root_code)
    # a page that re-points its own domain at 127.0.0.1 still names that domain (DNS rebinding; review, 2026-10-21)
    rebound, _ = through_relay("alice", "/pun/sys/gpujobs/api/jobs", "-H 'Host: evil.example:8080'")
    check("a request naming any host but localhost is refused at the relay", rebound == "421", rebound)

    put(f"{PROBE_APP}/manifest.yml", PROBE_MANIFEST)
    put(f"{PROBE_APP}/passenger_wsgi.py", PROBE_WSGI)
    put(f"{PROBE_APP}/bin/python", "#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", "0755")
    through_relay("alice", "/nginx/init?redir=%2fpun%2fsys%2fhdrprobe%2f")
    _, echoed = through_relay("alice", "/pun/sys/hdrprobe/", "-H 'X-Probe: kept'")
    seen = json.loads(echoed).get("headers", []) if echoed.startswith("{") else []
    check("neither the key nor the user header reaches the app (a control header sent with them does)",
          "HTTP_X_PROBE" in seen and not {"HTTP_X_FRONTDOOR_KEY", "HTTP_X_GPUJOBS_USER"} & set(seen), echoed[:200])
    sh(f"rm -rf {PROBE_APP}; /opt/ood/nginx_stage/sbin/nginx_stage nginx_clean --force")


# -- stage: lease -------------------------------------------------------------------------------------------------

LEASE_SITE = '\n[lease]\nlease_dir = "/var/lib/gpujobs/gpu-lease"\nkeeper_user = "dave"\npython = "%s"\n'
LEASE_PYTHON = (
    "#!/bin/sh\n# stands in for an interpreter with localmodels: answers the installer's `-c` probe, then, as the keeper,\n"
    "# rewrites its heartbeat every 5 s where the real one does (gpu_jobs.keeper_state_path()); it exits if it cannot\n"
    '[ "$1" = -c ] && exit 0\n'
    "while :; do printf '{\"updated_at\": %s}\\n' \"$(date +%s)\" > /var/lib/gpujobs/keeper/keeper.json || exit 1; sleep 5; done\n"
)


def stage_lease() -> None:
    section("lease: the keeper's user unit, reached with systemctl --user --machine=USER@.host")
    site_text = sh(f"cat {SITE}").out
    put("/tmp/lease-bad.toml", site_text + LEASE_SITE % "/usr/bin/python3")
    put("/tmp/lease-good.toml", site_text + LEASE_SITE % "/usr/local/bin/harness-lease-python")
    put("/usr/local/bin/harness-lease-python", LEASE_PYTHON, "0755")
    bad = sh(f"{INSTALL} --site /tmp/lease-bad.toml apply --component lease")
    check("lease: apply refuses when lease.python cannot import localmodels.lease", bad.rc == 1 and "REFUSED" in bad.text, bad.text[:200])
    sh("loginctl enable-linger dave")
    good = sh(f"{INSTALL} --site /tmp/lease-good.toml apply --component lease", timeout=180)
    unit = sh("systemctl --user --machine=dave@.host is-enabled slurm-gpu-lease-keeper; systemctl --user --machine=dave@.host is-active slurm-gpu-lease-keeper")
    check("lease: systemctl --user --machine=dave@.host installs, enables and starts the keeper unit in dave's own systemd",
          good.rc == 0 and unit.out.split() == ["enabled", "active"], f"apply rc={good.rc} {good.text[-120:]} | {unit.text}")
    time.sleep(15)
    still = sh("systemctl --user --machine=dave@.host is-active slurm-gpu-lease-keeper").out.strip()
    owner = sh("stat -c %U /etc/systemd/user/slurm-gpu-lease-keeper.service; "
               "stat -c %U /home/dave/.config/systemd/user/default.target.wants/slurm-gpu-lease-keeper.service").out.split()
    check("lease: the unit file is root's (root never writes in dave's home), dave's own systemd enabled it, and the "
          "keeper survives the end of the installer's login session (linger)",
          still == "active" and owner == ["root", "dave"], f"{still} {owner}")
    sub = cl("env", "GPUJOBS_SITE=/tmp/lease-good.toml", "gpujobs-submit", "--project", "e2e", "--what", "lease", "--class", "restart",
             "--name", "lease", "--mem", "200M", "--gpu", "--", *sleep_cmd(3), user="bob")
    beat = sh("stat -c %U /var/lib/gpujobs/keeper/keeper.json").out.strip()
    check("lease: the installer made keeper/ writable by the keeper's user (a stand-in writes the file; the real keeper's "
          "heartbeat is read back by test_gpu_lease_keeper.py)",
          beat == "dave", beat)
    check("lease: a user other than the keeper's can submit while the keeper's heartbeat is fresh",
          "REFUSED" not in sub.text and sub.rc == 0, sub.text[:220])
    clear_queue()


# -- stage: host --------------------------------------------------------------------------------------------------

HOST_PROBE = """
import json, subprocess, sys
sys.dont_write_bytecode = True
sys.path.insert(0, %r)
import job_wrapper
raw = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                     capture_output=True, text=True, check=False).stdout
expected = sorted((int(l.split(",")[0]), l.split(",")[1].strip()) for l in raw.splitlines() if l.strip())
free, why = job_wrapper.card_is_free(set())
print(json.dumps({"raw": raw.strip(), "expected": expected, "found": sorted(job_wrapper.foreign_gpu_processes(set())),
                  "own": sorted(job_wrapper.foreign_gpu_processes({p for p, _ in expected})), "free": free, "why": why}))
"""


def stage_host() -> None:
    section("host: the job wrapper's foreign-process check against this machine's real nvidia-smi (read-only)")
    done = subprocess.run([sys.executable, "-I", "-B", "-c", HOST_PROBE % str(REPO / "slurm")], capture_output=True, text=True, check=False)
    try:
        r = json.loads(done.stdout)
    except ValueError:
        check("the host probe ran", False, done.stderr[-300:] or done.stdout[-300:])
        return
    found = [tuple(x) for x in r["found"]]
    expected = [tuple(x) for x in r["expected"]]
    if expected:
        check("foreign_gpu_processes parses the real nvidia-smi output (each pid with its GPU uuid)", found == expected,
              f"raw {r['raw']!r} -> {found}")
        check("a process listed as the job's own is not foreign", r["own"] == [], str(r["own"]))
        check("card_is_free says not free while a process outside the job holds the GPU", r["free"] is False and "compute process" in r["why"], r["why"])
    else:
        # Another project's process is what makes the real output non-empty; the harness starts none on the host.
        note("host: no compute process was on the GPU during this run, so the parse of a non-empty real nvidia-smi output was NOT exercised")
        check("with the GPU idle, the real nvidia-smi output parses to no foreign process and the card is free",
              found == [] and r["free"] is True, f"raw {r['raw']!r}")


# SchedMD's digests (plan notes, "Facts"): the release tarballs the harness builds.
UPGRADE_STEPS = ("25.05.9", "7be633820df896df819ac2e9f2ebd0824de509515e5fb78ba1c19da2453b39ed")
UPGRADE_TARGET = ("26.05.5", "4680cafc82b0689865c885eb703441e7ebf764896c40fd03a50f2fcb60664f68")
UPGRADE_SITE = "/tmp/upgrade.site.toml"
BACKUP_FILES = 3  # database dump, state tar, /etc/slurm tar
UPGRADE_TIMEOUT_S = 4 * 3600  # two builds and two database conversions


def stage_upgrade() -> None:
    section("upgrade: `install.py upgrade-slurm` from the image's slurm 23.11 to the site file's release (last: it replaces slurm)")
    clear_queue()
    old = submit("alice", "before-upgrade", 2)
    check("before: a job completes on the old slurm, so accounting has history to keep",
          bool(old) and wait_state(old, "COMPLETED", 60) and old in sh("sacct -X -n -P -S now-7days -o JobID").out.split(), f"job {old}")
    before = sh("sinfo -V").text
    site = sh(f"cat {SITE}").out + (f"\n[slurm]\nversion = '{UPGRADE_TARGET[0]}'\nsha256 = '{UPGRADE_TARGET[1]}'\n"
                                   f"steps = ['{':'.join(UPGRADE_STEPS)}']\n")
    put(UPGRADE_SITE, site)
    done = sh(f"{INSTALL} upgrade-slurm --site {UPGRADE_SITE}", timeout=UPGRADE_TIMEOUT_S)
    check("upgrade-slurm exits 0 after both steps", done.rc == 0, done.text[-300:])
    after = sh("sinfo -V").text
    check("slurm now reports the site file's release", UPGRADE_TARGET[0] in after and before != after, f"{before} -> {after}")
    backups = sh("ls -d /var/backups/gpujobs/slurm-*/slurm_acct_db.sql /var/backups/gpujobs/slurm-*/state_save.tar.gz "
                 "/var/backups/gpujobs/slurm-*/etc_slurm.tar.gz")
    check("the backup folder holds the database dump and both tarballs", backups.rc == 0 and len(backups.out.split()) == BACKUP_FILES, backups.text)
    kept = sh("sacct -X -n -P -S now-7days -o JobID").out.split()
    check("accounting still lists the job from before the upgrade", old in kept, " ".join(kept))
    new = submit("alice", "after-upgrade", 2)
    check("a job submitted after the upgrade runs to completion", bool(new) and wait_state(new, "COMPLETED", 90), f"job {new}")


STAGES: dict[str, Callable[[], None]] = {
    "install": stage_install,
    "jobs": stage_jobs,
    "urgent": stage_urgent,
    "service": stage_service,
    "failure": stage_failure,
    "recovery": stage_recovery,
    "portal": stage_portal,
    "proxy": stage_proxy,
    "ssh": stage_ssh,
    "lease": stage_lease,
    "host": stage_host,
    "upgrade": stage_upgrade,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=",".join(STAGES), help="comma-separated stages (default: all)")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(STAGES))
        return 0
    chosen = args.only.split(",")
    if unknown := [s for s in chosen if s not in STAGES]:
        parser.error(f"unknown stage(s) {unknown}; stages: {', '.join(STAGES)}")
    for name, stage in STAGES.items():
        if name in chosen:
            stage()
    failed = [(n, e) for n, ok, e in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed of {len(RESULTS)} checks")
    for text in NOTES:
        print(f"  NOTE: {text}")
    for name, evidence in failed:
        print(f"  FAILED: {name}\n          {' '.join(evidence.split())[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
