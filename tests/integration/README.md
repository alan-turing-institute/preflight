# Integration test cluster

A one-node slurm cluster in a container, for testing the portable GPU jobs build (installer, tools, agent, portal)
without touching this box's `/etc`, systemd, slurm daemons or queue.

    tests/integration/run.sh up      # build the image if needed, start the container, wait for the node to be idle
    tests/integration/run.sh smoke   # as alice: a trivial job shows in squeue, a gpu:2 job runs and is accounted
    tests/integration/run.sh e2e     # fresh cluster, then install every component and run all scenarios (see "End-to-end run")
    tests/integration/run.sh shell [-u USER] [CMD...]   # a shell (root by default), or one command, in the cluster
    tests/integration/run.sh down    # remove the container and network (the image stays)

Every command that talks to slurm inside the container belongs in `run.sh`. The image is `gpujobs-testcluster:dev`
(about 570 MB), the compose project is `gpujobs-test`.

## What is in it

- Ubuntu 24.04 with the distro packages this box uses: `slurm-wlm`, `slurmdbd`, `mariadb-server`, `munge` (slurm
  23.11.4, the box's line), plus `sudo` and `python3` for the installer. systemd is PID 1, so the installer's
  `systemctl` calls work. `libpam-systemd` and `dbus-user-session` are installed and `systemd-logind` runs, so
  `systemctl --user --machine=USER@.host` works (see "Findings").
- Users `alice` (2001) and `bob` (2002), plain; `carol` (2003), admin: member of the Unix group `gpujobs-admin`
  (2000). In slurm accounting all three are users of account `harness`; carol has no slurm admin level until the
  installer's `admin add carol` gives her `Operator`.
- A bootstrap single-node slurm (`bootstrap/`): node `testnode` with 4 CPUs, 4 GiB and `Gres=gpu:4`, one partition
  `normal`, accounting through `slurmdbd` + MariaDB (the `slurm` OS user logs in by `unix_socket`, as on the box).
  The first-boot units `gpujobs-db-init` and `gpujobs-accounts-init` create the database login, the cluster, the
  account and the users. The installer replaces the bootstrap files.
- The repository, read-only, at `/src`. Nothing of ours is inside the image.

## Fake GPUs: a bare count does not work on slurm 23.11.4

Observed in this image (`/var/log/slurm/slurmd.log`, `sinfo -R`), with `GresTypes=gpu`, `Gres=gpu:4` on the node and
`ConstrainDevices=no`:

| `gres.conf` line | Result |
|---|---|
| `NodeName=testnode Name=gpu Count=4` | slurmd: `warning: Ignoring file-less GPU gpu:(null) from final GRES list`; the node is set `INVAL`/`DRAIN`, reason `gres/gpu count reported lower than configured (0 < 4)` |
| `... Name=gpu Count=4 Flags=CountOnly` | the same warning |
| `... Name=gpu Count=4 File=/dev/null` | slurmd exits: `fatal: Invalid GRES record for gpu, count does not match File value` |
| `... Name=gpu File=/opt/fakegpu/gpu[0-3]` | accepted; node `idle`, `Gres=gpu:4`; a `--gres=gpu:2` job runs with `AllocTRES=...gres/gpu=2` |

So the harness uses four symlinks to `/dev/null` (`/opt/fakegpu/gpu0` to `gpu3`, made in the Dockerfile) and
`File=/opt/fakegpu/gpu[0-3]`. The site file for the harness therefore names those four device paths, not `/dev/nvidia<N>`.
`ConstrainDevices=no` is set in the bootstrap `cgroup.conf`, for the harness only.

## Container limits and what it cannot touch

`compose.yml`: label `lifecycle: ephemeral`, 8 GiB memory (no extra swap), 4 CPUs, 2048 processes, no GPU, no host
network, no published ports, no host `/etc` or munge socket, not privileged. systemd runs with a private cgroup
namespace, Docker's `--security-opt writable-cgroups=true` (a writable `/sys/fs/cgroup` that is the container's own
subtree) and tmpfs on `/run`, `/run/lock` and `/tmp`. The only host path mounted is this repository, read-only.

## Known quirks

- `slurmd` logs `error: Unable to move pid N to init root cgroup /sys/fs/cgroup` (`Device or resource busy`) at the
  end of every job step. The job still completes and is accounted; it is the step daemon's cleanup trying to move itself
  to the root cgroup of the container's namespace.
- `slurmd` also logs `can not load PMIx library` errors at start (the PMIx library is not installed; MPI is not used).
- `SlurmdParameters=config_overrides` is set in the bootstrap `slurm.conf` because the container sees the box's 20 CPUs
  and 124 GB while only 4 CPUs / 8 GiB are allowed by its cgroup limits.
- The first job after a boot logs `Could not open job state file`: there is no earlier state.
- A job's output file is written in its working directory; `run.sh` starts commands in the user's home.

## End-to-end run

    tests/integration/run.sh e2e                                      # fresh cluster (down, up), then every stage; about 30 minutes
    python3 tests/integration/e2e.py [--only STAGE,...] [--list]      # against the running cluster

`run.sh e2e` installs every component into the container through the real installer from `harness.site.toml`, runs the
scenarios below as alice, bob and carol, and exits 1 if any check failed. Each check prints `PASS` or `FAIL` with its evidence
as it happens; the summary lists the failures and any `NOTE` (something that could not be exercised). `--only` takes
`install,jobs,urgent,failure,recovery,portal,lease,host`; later stages need `install` to have run once on that cluster.

### What the run does outside the installer

Only what an operator would supply, or what the container cannot do:

- **Portal secrets** (`/etc/gpujobs/portal/crypto_passphrase`, `users.json` with bcrypt hashes of throwaway passwords), the
  `notify_command` script, and one Unix user (`dave`) the installer has not been told about.
- **Systemd drop-ins for four units** (`prometheus`, `grafana-server`, `gpujobs-node-exporter`, `gpujobs-gpu-exporter`) that
  switch off namespace sandboxing, which the unprivileged container refuses (`Failed to set up mount namespacing: Operation not
  supported`, `Failed to keep CAP_SYS_ADMIN`). A real host does not need them.
- **A TLS front for Grafana** (an Apache virtual host on 3443 proxying to Grafana on 3001); see the decision below.
- **`systemctl restart slurmctld`** right after `user add` / `admin add`, to work around an open bug (below).

### Stages and what each check proves

| Stage | Checks |
|---|---|
| `install` | `detect`, `render`, `check` before and after `apply` (clean), `apply` of slurm, tools, agent, metrics and portal; the installer itself created account `gpujobs` (the container's first-boot unit only made `harness`) and, through `user add alice bob dave` and `admin add carol`, the users' records (Operator level, group); services up; the agent's boot step ran; Prometheus scraping, Grafana with both dashboards, `jobstats` runs |
| `jobs` | S1 a chain of three (reason "waits for job", metrics file written and served, order, the installed wrapper with hostguard); S2 bob's 2-GPU job; S4 alice cannot pause or kill bob's job; S5 the admin pauses it and bob sees "paused by carol"; S6 alice cannot run `urgent.py`, nor use the urgent partition; S3 alice pauses and resumes her own job and the agent puts its priority back, with a control run without the agent |
| `urgent` | S7 the admin's urgent job pre-empts a normal job; the waiting job is held; the pre-empted job runs again before it; the agent releases it |
| `failure` | S8 a failed job gets a FAILED event (written by the agent, owner `slurm`) and `notify_command` runs |
| `recovery` | S9 slurmctld+slurmd restart with the job's processes killed: requeued by the agent's boot step, with a control run without the agent; S10 a slurmctld-only restart keeps the live job |
| `portal` | anonymous request goes to the login; Dex local login over HTTPS (wrong password refused); the app's first-visit initialization; symlinked sys app; Passenger starts our `bin/python`; every page loads; every user's jobs listed; buttons hidden or shown by owner and admin; `Origin` and `Host` reach the app; pause, preview and resume through the portal |
| `lease` | `apply --component lease` refuses a `lease.python` without localmodels; the keeper's user unit installed, enabled and started with `systemctl --user --machine=dave@.host`, surviving the installer's session |
| `host` | the job wrapper's `foreign_gpu_processes` and `card_is_free` against this machine's real `nvidia-smi` (read-only; nothing is started or changed on the host). It needs a compute process on the GPU to parse a non-empty output; when there is none the run says so in a `NOTE` and checks the empty case only |

### Decision: how Grafana is reached in the portal

Grafana serves plain HTTP, and a browser blocks that as mixed content inside the HTTPS portal. The installer has no TLS
option for Grafana, so the harness site file sets `metrics.grafana_url = "https://testnode:3443"` and `e2e.py` puts an Apache
virtual host on that port (the portal's certificate, `ProxyPass` to `127.0.0.1:3001`). The check shows that the cluster file
names that URL, that `https://testnode:3443/api/health` answers, and that the response has no `X-Frame-Options` (Grafana runs
with `allow_embedding`). Why not Grafana's own TLS: it needs a new site field and a change in the metrics component, which is a
design choice. Whether a browser then renders the panels inside the portal was **not** checked (no browser here).

### Findings (each verified in the container)

- **A `docker exec` too early at boot leaves slurmd without cgroup controllers.** A process started by `docker exec` lands in the
  container's init cgroup. If it lands there before systemd has moved itself into `init.scope` and enabled the `cpu`, `memory`
  and `cpuset` controllers for its children, the kernel refuses to enable them afterwards (`echo +memory >
  /sys/fs/cgroup/cgroup.subtree_control` gives "Device or resource busy" while the cgroup holds a process), and every later
  `slurmd` start dies with `Controller memory is not enabled! ... Unable to initialize jobacct_gather`. `run.sh up` therefore
  waits, reading the host's `/proc/PID/cgroup`, until the container's init is in `init.scope` and the memory controller is on,
  before it execs anything.
- **`systemctl --user --machine=USER@.host` works in the container** once the image has `libpam-systemd`, `dbus-user-session`
  and an unmasked `systemd-logind` (the Dockerfile now has them). It runs `systemd-run -M.host -pUser=USER -pPAMName=login
  systemd-stdio-bridge ...`; without `pam_systemd` there is no `XDG_RUNTIME_DIR`, and without `dbus-user-session` the user's
  `bus` socket never appears (both end in "Transport endpoint is not connected").
- **The installed `cgroup.conf` (`ConstrainCores/RAMSpace/Devices=yes`, the box's settings) works in the container**: jobs run and
  are accounted. slurmd logs `load_ebpf_prog: BPF load error (Operation not permitted)` for each step, so the device constraint
  is not really applied in the container.
- **A user's first visit to the app asks them to "Initialize App"** (OOD: the app is not in their per-user nginx yet;
  `/nginx/init?redir=...` fixes it and restarts that nginx). The guide does not mention it.
- **Slurm does not refuse a job sent to a partition the user may not use**: `sbatch --partition=urgent` by alice is accepted and
  stays PENDING with reason `uid_2001_not_in_group_permitted_to_use_this_partition_(urgent)._groups_allowed:_gpujobs-admin`.
- **`DefMemPerNode=16384` in the rendered `slurm.conf` is fixed.** On a node whose `real_memory_mb` is lower (the container's
  4096), a CPU job submitted without `--mem` is refused: "Memory specification can not be satisfied". GPU jobs ask for `--mem 0`
  or a measured figure. The harness always passes `--mem`.
- **A GPU job without `--mem` takes the whole node's memory, and the default 4 CPUs are the whole container node**, so the
  harness passes `--mem 200M --cpus 1` to run jobs side by side.
- **The portal's self-signed certificate is `CA:TRUE`** (Apache logs `AH01906 server certificate is a CA certificate`). It works;
  it is not what a leaf certificate should be.
- **`apply` does not check that the services it starts stay up.** Here the GPU exporter dies at once (no NVML in the container)
  and `apply` still reports success. Expected in the container; noted because a real failure would look the same.
- **A restore-order request naming jobs that have since left the queue keeps failing** (`releasing [...] failed; will retry` in
  `agent/agent.json`). Seen in passing, after the jobs of a request were cancelled by hand. Whether one gone id stops the others in
  the same `scontrol release` was not tested.
- **A requeued job cannot start for about a minute** (`AuthInfo=cred_expire=60`), and `squeue` says `Resources`, not
  `BeginTime`, while the card is also busy. In one run the check that freed the card 8 s after the pause saw the later job start
  first (the backfill scheduler gave it the free GPU, the resumed job needed all four); it now frees the card 65 s after the pause.
  So "keeps its place" is about priority, and holds from the moment the job is eligible.
- **After `scontrol reconfigure`, jobs started dying at launch in one run** (`testnode_slurmstepd.scope/cgroup.controllers: No such
  file or directory`); the cause was not established, so the harness does not use it. The same happened once after the recovery
  scenario had killed every `slurmstepd`, including the `slurmstepd infinity` helper that keeps slurmd's systemd scope alive;
  it now kills only the job's own processes.

### Open bugs left for a design decision (their checks fail on purpose)

1. **`user add` / `admin add` of a user slurm accounting already knows leaves slurmctld refusing that user's jobs.** Check "OPEN
   (design): right after user add / admin add ...". Evidence: alice, bob and carol exist in accounting (account `harness`, default
   account `harness`); right after `install.py user add alice` (it runs `sacctmgr add user alice Account=gpujobs
   DefaultAccount=gpujobs Cluster=harness`) `sbatch` says "Invalid account or account/partition combination specified",
   `scontrol show assoc_mgr users=alice flags=users` shows `DefAccount=(null)` while `sacctmgr show user` says `gpujobs`, and
   slurmctld logs "User alice doesn't have a default account". Adding a brand-new user later, `systemctl restart slurmctld` and
   `scontrol reconfigure` each heal it. A user who was new to accounting (dave, erin) was never affected. Fix options: leave an
   existing user's default account alone, or make slurmctld reload after the change.
2. **With a lease in the site, only the keeper's own user can submit.** Check "OPEN (design): with a lease in the site, a user
   other than the keeper's can submit ...". `submit.keeper_running()` runs `systemctl --user is-active slurm-gpu-lease-keeper`
   as the submitter; bob has no such user unit: "Failed to connect to bus", then "REFUSED: the lease keeper
   (slurm-gpu-lease-keeper) is not running", while `systemctl --user --machine=dave@.host is-active` says `active`. `urgent.py`
   behaves the same on a site with a lease. Fix options: read the keeper's heartbeat file instead, or ask the keeper user's systemd.

### Product fixes made by this task (separate commits)

- `c59fe85` the installer creates `logs/`, `scripts/` and `manifests/` under the state directory (step: alice submits a job);
- `58e1370` `urgent.py` asks for the lease keeper only where the site has a lease (step: the admin runs an urgent job);
- `828c599` job records, requests and agent state are written world-readable, as plan 3.3 says (steps: bob sees alice's job; the
  urgent job holds a waiting job).

### Checks shown to fail

Besides the two open bugs, a patched copy of `e2e.py` was run once with one deliberate change per check, each of which should turn
exactly its own check red: S1 expected reason "Priority"; S2 expected `gres/gpu=3`; S4 expected the pause to succeed; S5 expected
"paused by dave" and an event actor `dave`; S6 expected urgent to succeed; S3 expected the later job to have the higher priority;
S7 with the agent stopped; S8 with a job that exits 0; S9 started without the agent; S10 expected different process ids; the
"wrong password" check with the correct password; the right `Origin` replaced by a foreign one; the lease refusal check with a
working interpreter; the host parse expected to differ. The S3 and S9 control runs (agent stopped) are the same kind of proof
inside the normal run. The first `install` run, where `apply` died on a slurmd that could not start, made ten `install` checks
fail as they should.

### Not exercised

- The real lease keeper (`gpu_lease_keeper.py`): it needs `localmodels` (with pydantic). The `lease` stage installs the unit with a
  stand-in interpreter that answers the installer's import probe and idles; only the install path and `systemctl --user
  --machine=` are proven.
- Jobstats' per-job summaries and the GPU panels (no NVML in the container); the Grafana dashboards are provisioned, not viewed.
- Anything needing a browser: the portal pages were fetched with `curl`; `app.js` was not run.
- `manifests/` is created by the installer but nothing in the run writes there (needs `dgx_gate`).
