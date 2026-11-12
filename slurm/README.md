# GPU jobs through slurm

## 1. What this is

This directory turns a Linux machine with NVIDIA GPUs into a small slurm cluster with a job layer on top. Slurm
is the scheduler: it queues work, starts it when the GPU is free, and stops it again. On top of slurm the layer
adds:

- **Pause and resume.** A job that can continue from its latest checkpoint is paused by putting it back in the
  queue held, and resumed by releasing it. A job that cannot continue (a "steady" job) says so before it is paused.
- **Levels for waiting jobs.** Each job waits at a level, `low`, `normal` (the default) or `high`, open to everyone;
  within a level, the earlier job goes first, and keeps its place after a pause, a pre-emption or a slurm restart.
  A level orders the waiting line only: it never stops a running job (section 7).
- **Urgent work.** An administrator can run an urgent job now. Jobs in its way are paused, and background services
  that share the GPU through the lease (see section 5) step aside.
- **A web page** (Open OnDemand, our own page inside it) listing every GPU job with its waiting reason, log,
  efficiency, history and GPU-hours per project, with pause, resume and kill buttons.
- **Per-job metrics** (Princeton Jobstats, with Prometheus and Grafana): CPU, memory and GPU use over time.

It is for any Linux machine with NVIDIA GPUs, used by one person or by several. Everything that differs between
machines is a field in one site file (section 4); nothing is hard-coded to a machine name.

The pieces are installed by one program, `slurm/install/install.py`, in six components: `slurm`, `tools`, `agent`,
`lease`, `metrics`, `portal` (section 5).

## Quick start

**If you only submit jobs** (the machine is already set up): the tools are on your path as `gpujobs-*`. Section 7 has
every option; this is the short version.

    # run a command on the GPU. --class resume: running it again continues from its checkpoint; restart: it starts over.
    gpujobs-submit --project myproj --what 'train the model' --class resume --resume-ok --gpu --name train1 -- python train.py
    gpujobs-submit ... --dry-run -- python train.py     # print the batch script and submit nothing
    gpujobs-status                                      # the queue, who holds the GPU, and why jobs wait
    gpujobs-pause 123 ;  gpujobs-resume 123 ;  gpujobs-kill 123      # your own jobs only
    gpujobs-submit ... --level high -- ...              # wait ahead of normal jobs (levels: low, normal, high)

What else the same command can do, each set out in section 7:

| You want | Add to `gpujobs-submit` |
|---|---|
| Several copies of one job to share a card (the site must enable `shards`) | measure once with `gpujobs-sharing measure`, then submit as usual; `--max-jobs N` sets the number |
| A CPU-only job to run on a spare machine as well (a "part-time" machine, if the site lists one) | nothing for a `resume` job the machine can run; `--anywhere` if you vouch for it |
| A small CPU stand-in for a background service | `--service` |
| A job on another site's slurm | `--site NAME` (the site file's `[remote.NAME]`) |
| Urgent work (administrators only) | use `gpujobs-urgent` instead |

**If you are setting up a machine:** `python3 slurm/install/install.py detect > site.toml`, edit it against the commented
example `slurm/site.example.toml` (every field is explained in section 4; unknown keys are an error), then `render`,
`check`, `apply` (section 3). The example shows each table: one `[[nodes]]` entry per machine with its `[[nodes.gpus]]`
(add `shards` there to let jobs share a card), an optional part-time machine, `[lease]`, `[metrics]` and `[portal]`.
Add each user with `install.py ... user add USER`, and administrators with `admin add USER` (section 6).

## 2. Requirements

- **Operating system: Ubuntu 24.04.** What was tested: Ubuntu 24.04 on aarch64 (64-bit Arm), for the unit tests
  and for the container test cluster in `tests/integration/`. Anything else (other Ubuntu or Debian releases,
  x86_64 machines, real multi-node clusters) is untested. The metrics component has download entries for both
  x86_64 and aarch64, but only aarch64 was exercised.
- **What `apply` and `check` say about the machine.** They refuse, before anything is touched, when the build cannot
  work: no `apt-get`, or apt offering no `slurm-wlm` or one older than 23.11 (Ubuntu 22.04 offers 21.08). They warn
  and go ahead where it is only untested: an operating system other than Ubuntu 24.04, an architecture other than
  aarch64, a site file listing more than one node.
- **Slurm 23.11 or newer, from the distribution packages, or built from a SchedMD release.** Ubuntu 24.04 ships
  23.11.4. The installer installs `slurm-wlm`, `slurmdbd`, `mariadb-server` and `munge`; with the site file's
  `[slurm]` table it builds and installs that release instead of the first two (section 4). That Ubuntu build has no NVML plugin (NVML is NVIDIA's
  management library), so GPUs are declared by device file in `gres.conf`, and slurm itself records no per-GPU
  use; the metrics component fills that gap.
- **A machine whose slurm this installer sets up.** The `slurm` component always installs and configures slurm
  itself. On a machine that already runs a slurm someone else set up (its `/etc/slurm/slurm.conf` lacks this
  installer's "Rendered from the site file" line), `apply` refuses rather than replace their settings; taking it
  over needs `apply --replace-existing-slurm`, after backing up `/etc/slurm`. Installing only this layer on top of
  someone else's slurm is not supported.
- **An NVIDIA driver** that provides `nvidia-smi` (hardware detection uses it).
- **Internet access during the install.** Packages come from apt. The metrics and portal components also add
  Grafana's and Open OnDemand's apt repositories and download exporters and Jobstats files from GitHub. Every
  download is pinned to a version or commit and checked against a recorded sha256 sum before use.
- **sudo.** `apply` writes under `/etc`, installs packages, and starts services, so it must run as root. `check`
  needs root as well once the site has any root-only file (section 8). `render` and `detect` do not need root
  unless the portal is on (section 8).

## 3. Install walk-through

All commands run from a checkout of this repository. `--site PATH` names the site file; it defaults to
`/etc/gpujobs/site.toml`, or to the path in `$GPUJOBS_SITE`. The examples keep the draft in the current directory
and pass `--site` each time.

1. **Draft a site file** from what the machine reports (host name, CPUs, memory, GPUs, whether GPU memory is
   shared with the host):

       python3 slurm/install/install.py detect > site.toml

   `detect` only reads; it writes the file to standard output. The draft sets `real_memory_mb` to the machine's
   memory minus 12288 MB (change that with `python3 slurm/install/detect.py --reserve-mb N`), and leaves metrics
   and the portal off.

2. **Edit it.** Section 4 lists every field. `slurm/site.example.toml` is a commented example. An unknown key
   anywhere is an error, so a typo cannot silently fall back to a default.

3. **Review what would be installed**, without touching the system:

       python3 slurm/install/install.py --site site.toml render --out /tmp/gpujobs-rendered

   It writes every file the enabled components would install under that directory, with their modes. Read them.

4. **Check the machine against the site file** (exit status 1 and a `DIFFERS` line per file when the installed
   files do not match; with nothing installed yet, expect every file to differ):

       sudo python3 slurm/install/install.py --site site.toml check

5. **Install**:

       sudo python3 slurm/install/install.py --site site.toml apply

   Components run in a fixed order: `slurm`, `tools`, `agent`, `lease`, `metrics`, `portal` (only those the site
   file enables). First every component's refusals are collected and printed as `REFUSED:`; if there is any, nothing
   is changed. Then apt sources, packages, files, and the post-install steps (service restarts) run. The `slurm`
   component also gives slurm's accounting the cluster (`cluster_name`) and the site's `account`, between restarting
   `slurmdbd` and starting `slurmctld`; records that already exist are left alone. **`apply`
   refuses while a slurm job is running** (or completing or suspended), or when `squeue` fails, because restarting the
   slurm daemons would requeue or kill it; jobs that are only waiting are kept in slurmctld's saved state and do not
   count. Wait for the running jobs to end (`pause.py` sends a `resume` job back to its checkpoint), or install only other components with
   `apply --component NAME` (repeat the option for several; the choices are `slurm`, `tools`, `agent`, `lease`,
   `metrics`, `portal`). The slurm refusal belongs to the `slurm` component, so `--component portal` does not
   trigger it.

   `apply` also copies the site file to `/etc/gpujobs/site.toml`; the agent and the tools read that copy. After
   editing the site file, run `apply` again.

6. **Let each user submit.** The generated `slurm.conf` sets `AccountingStorageEnforce=associations`, so slurm
   accepts a job only from a user its accounting database knows. For each user:

       sudo python3 slurm/install/install.py --site site.toml user add USER

   This adds USER to the site's `account` on the cluster (and the cluster and account themselves, if missing); a
   user already there is left alone, and a user slurm already knows from another account keeps that default account.
   `user remove USER` takes the user out of that account.

7. **Make someone an administrator** (section 6):

       sudo python3 slurm/install/install.py --site site.toml admin add USER

   This adds the user to slurm's accounting as in step 6 if they are not there yet, adds them to the site's
   `admin_group` and sets their slurm administrator level to Operator. `admin remove USER` takes the group and the
   level away; the user can still submit.

8. **A machine that ran the earlier, per-user version** of these tools keeps its job records in an older state
   directory. Copy the job records, footprints and events into the new one (existing files are left alone):

       sudo python3 slurm/install/install.py --site site.toml migrate-state --from /home/OLDUSER/.local/state/slurm-gpu

   (`~/.local/state/slurm-gpu` is where the earlier version kept them.)

## 4. The site file

TOML. Tables: the top level, `[[nodes]]` with `[[nodes.gpus]]`, `[lease]` with `[lease.services]`, `[service]`, `[slurm]`, `[metrics]`,
`[portal]` with `[portal.auth_settings]`. "required" means there is no default. The source of truth is
`slurm/site_config.py`.

### Top level

| Field | Default | Meaning |
|---|---|---|
| `name` | required | Short name of the site. Also the default for `cluster_name`. |
| `cluster_name` | the value of `name` | Slurm's `ClusterName`; Open OnDemand names its cluster file after it and Jobstats labels its series with it. Never change it on a running cluster: `slurmctld` refuses to start when it differs from the one in its saved state. |
| `account` | `gpujobs` | The slurm account the installer creates and `user add` puts users in. |
| `state_dir` | `/var/lib/gpujobs` | Where the job layer keeps its records and events (section 8). |
| `install_dir` | `/opt/gpujobs` | Where the tools are installed (`install_dir/slurm/`). |
| `admin_group` | `gpujobs-admin` | Unix group of administrators. Also the group allowed to submit to the `urgent` partition. |
| `unified_memory` | `false` | True when the GPU shares system memory with the host (`nvidia-smi` reports total memory as `[N/A]`). `detect` sets it. On such a site the Jobstats dashboard's GPU memory panel plots the agent's per-job series, because the GPU exporter has none to report. |
| `hostguard` | `false` | The job wrapper marks every job as the preferred out-of-memory victim (`oom_score_adj` 800, through the host memory guard in `hostguard/hostguard.py`), so a memory shortage ends a job rather than a login session. The `tools` component installs the guard when this is on. |
| `dgx_gate` | `false` | Require the DGX submission gate for jobs submitted with `--entry` (other jobs are not gated). The `tools` component installs the gate (`dgx/`) and the host memory guard it checks when this is on. |
| `notify_command` | `[]` | Command (list of arguments, no shell, 30 s limit) the agent runs once per notice, with the notice in its environment. Every notice: `GPUJOBS_EVENT` (`failed`: a job ended FAILED, OUT_OF_MEMORY, TIMEOUT or NODE_FAIL; `idle`: a waste finding; `over_ask`: a running job holds at least 1 GB more memory, host plus card, than it asked for, once per run, unified-memory nodes only; `memory_stop`: the memory guard stopped a job, see "Memory on unified-memory nodes"), `GPUJOBS_JOB` (the job the notice is about), its `GPUJOBS_NAME`, `GPUJOBS_PROJECT` (the name's first `:` field) and `GPUJOBS_USER`, and `GPUJOBS_MESSAGE`, one plain sentence ready to send. A failure adds `GPUJOBS_STATE`; a memory stop adds `GPUJOBS_ENDED` (`by_wrapper`, `by_scancel` or `not_ended`). An idle finding is sent to both sides, at most once an hour per pair: `GPUJOBS_ROLE` is `waiting` (the job that waits) or `holding` (a running job holding what it lacks), and `GPUJOBS_OTHER_JOB` and `GPUJOBS_OTHER_PROJECT` name the other side. Names are whatever their submitter typed, with control characters shown as `?`: quote them wherever they go into a shell command or a mail header. Empty means no notification. `slurm/notify_mail.py` is a working example that mails each notice to the job's user through `sendmail` (`["/usr/bin/python3", "/opt/gpujobs/slurm/notify_mail.py", "--domain", "example.org"]`); any other channel is a script reading the same variables. |
| `sharing_memory_fraction` | absent | Only for sites that share a card (`shards`, below). A number above 0 and at most 1: the share of a node's jobs memory that the jobs sharing a card may fill together, counted at their measured peaks. It tightens two gates alike: `sharing.py measure` drops any level whose peak memory in use passes it (of the box's total memory), and `submit.py` caps the copies by it (of the node's jobs memory). Absent: they may fill all of it. |
| `cpus_from_history` | `recommend` | Cores for a CPU-only job given `--name` and no `--cpus` (section 7). `off`: nothing. `recommend`: `submit` prints the cores the job's 3 newest completed runs of the same `project:class:name` justify and asks as before. `enforce`: it asks for them. The figure is their highest average cores in use (core time over wall time, from `sacct`) times 1.25, rounded up, from 1 to the cores the job may have. |
| `memory_from_size` | `recommend` | Memory for a job given `--size COUNT UNIT` and no `--mem` (section 7, "Memory from a declared size"). `off`: nothing. `recommend`: `submit` prints the memory the project's earlier runs of that unit justify and asks as before. `enforce`: it asks for the larger of that figure and the name's own measured one, except a figure as large as the whole box, which is printed and never applied. |
| `require_check` | `recommend` | A submission that states neither `--check` nor `--no-check` (section 7, "Checks before queuing"). `off`: accepted silently. `recommend`: accepted with a warning. `enforce`: refused, with the two flags named. |
| `nodes` | required | One `[[nodes]]` table per compute node; at least one. |
| `lease` | absent | Optional table; leave it out when no other service shares the GPU through the lease. Without it the `lease` component is not installed. |
| `metrics` | all defaults below | Table. |
| `portal` | all defaults below | Table. |

### `[[nodes]]`

| Field | Default | Meaning |
|---|---|---|
| `name` | required | Slurm `NodeName`. The first node is also slurm's controller host (`SlurmctldHost`). |
| `cpus` | required | CPUs slurm may hand out. |
| `real_memory_mb` | required | Memory in MB slurm may hand out (the machine's memory minus what the host keeps back). |
| `gpus` | `[]` | `[[nodes.gpus]]` tables, one per GPU on this node. |
| `jobs_memory_gb` | absent | Memory jobs may be given on this node, in decimal GB, below `real_memory_mb`; slurm keeps the rest for work outside its jobs (`MemSpecLimit`). Any node may set it. On a part-time machine it is what its owner lends. On a full node, measure it: after a week or more of the agent running on a unified-memory node, `python3 slurm/status.py --reserve` on that node prints the figure, its memory less the 99th percentile of what it used outside jobs (one-minute samples, kept 14 days) less its health check's close level. Absent: no reserve, so jobs may be given all of `real_memory_mb`. |
| `cores_per_gpu` | absent | Cores held for GPU jobs, per GPU. A GPU job gets this many per GPU by default, and CPU-only jobs (partition `cpu`) are capped by `MaxCPUsPerNode` at the node's cores less those held, so a free GPU never waits for cores taken by CPU jobs (slurm.conf(5)'s own use of `MaxCPUsPerNode`; slurm 23.11 has no `RestrictedCoresPerGPU`). Set it from a measurement: the fewest cores at which your GPU jobs stop getting faster. With several nodes the cap is the tightest node's and applies to every node, CPU-only ones included. At least 1, and must leave CPU-only jobs a core. Absent: nothing is held, and a GPU job's default is its GPUs' share of the node's cores (all of them on a one-GPU node), so any CPU job running holds a GPU job back. `submit.py` refuses a CPU-only job asking for more than the cap, which would otherwise wait forever. |

### `[[nodes.gpus]]`

| Field | Default | Meaning |
|---|---|---|
| `index` | required | GPU index. |
| `uuid` | required | From `nvidia-smi --query-gpu=uuid --format=csv,noheader`. |
| `device` | required | Device file for `gres.conf`, for example `/dev/nvidia0`. |
| `type` | required | Gres type name. The node's `Gres=gpu:TYPE:COUNT` uses the type of its first GPU and the number of its GPUs. |
| `shards` | absent | Let several jobs share this card at once (slurm's `gres/shard`). `"cores"` cuts the card into one shard per core the node holds for a card (`cores_per_gpu`, which it then requires), so a job's shards and cores match one to one; a whole number from 1 to 1024 is also accepted. Absent or 0: the card is one token, one job at a time, as before. Adding shards to a running cluster needs slurm restarted or reconfigured; never reduce them while shard jobs run. See "Sharing a card" in section 7. |

### `[lease]` (optional)

| Field | Default | Meaning |
|---|---|---|
| `lease_dir` | required | Directory of lease files. |
| `keeper_user` | required | User the lease keeper runs as. |
| `python` | required | Interpreter that can import `localmodels.lease` (`GpuLeaseClient`). `apply` refuses when it cannot. |
| `services` | `{}` | `[lease.services]`: `name = "how and how fast it yields"`. Shown in the plan `urgent.py` prints. |

### `[service]` (optional)

A `service` partition for the small CPU stand-ins of background services: work a person is waiting on that must start
now while slurm holds the GPU (`submit.py --service --mem M`). When one waits for cores (and fits in memory), the agent
suspends `cpu` jobs for it (`scontrol suspend`: slurm takes back their cores, keeps their memory, and starts the service
job at once; the jobs lose no work), newest first and only those its owner agreed to (`--suspend J,J`, or none with
`--no-suspend`), after holding every waiting `cpu` job so none takes the freed cores; GPU and urgent jobs are never held or suspended.
It resumes a job only when that job's own cores are free (slurm puts it back on them unchecked), so one whose cores a
GPU job took meanwhile waits for that job; then it releases its holds. A service job that has not started 60 s after
room was made gives it back, and is helped again 5 minutes later.
`submit.py --service --dry-run` prints which jobs it would suspend (`[submit] service room: {json}`), so a service can ask
its user first. It is not lease demand, so the services keep the card. When memory is short it waits: suspending frees
no memory. Only the partition line is added to slurm.conf (`scontrol reconfigure` applies it with jobs running).
Slurm's own suspending pre-emption is not used: it needs gang scheduling, whose second pre-emption pass killed a job
slurm had just requeued for urgent work (harness, 2026-10-23; slurm 23.11.4). One node only: `apply` refuses the table
on a site of several.

| Field | Default | Meaning |
|---|---|---|
| `cores` | required | Cores the partition may use per node (`MaxCPUsPerNode`), 1 up to the cores CPU-only jobs may use. |
| `group` | `gpujobs-service` | Unix group allowed to submit (`AllowGroups=`, with the admin group). `install.py service add USER` adds one. |

### `[maintenance]` (optional)

For a site whose machine is restarted for updates by a slot of its own (a weekly window run outside these tools). The
jobs page shows the site's admins a banner of the updates that wait, with **Restart now** and **Skip this week**.
Nothing here restarts anything: the buttons only leave a request for the slot, which decides and logs its verdict.

| Field | Default | Meaning |
|---|---|---|
| `status_file` | required | Absolute path of the slot's status, JSON: `{"updated": epoch, "state": "none" \| "planned" \| "skipped" \| "running" \| "failed" \| "done", "planned_at": epoch or null, "waiting": [{"kind": "...", "summary": "one plain line"}], "detail": "one plain line"}`. A missing, malformed or more than an hour old file shows "status unavailable", never "nothing waits". |
| `request_dir` | required | Absolute path where a button leaves `{"action": "restart_now" \| "skip_week", "at": epoch}` as `<user>-<ns>-<action>.json`, owned by the admin who pressed it. Only admins who can write this folder see the buttons; the slot checks the owner itself. |

### `[slurm_release]` (optional)

Watch SchedMD's releases and prove a newer one in the test cluster (`tests/integration`) before a maintenance window
installs it; absent: no watch. Needs `[slurm]`. `slurm/slurm_release.py` has four commands:

- `check`, nightly as `test_user`: reads SchedMD's GitHub release list (stable releases with a sha256 digest only), picks
  the target and, when it has no record and no test in the queue, submits its test through `submit.py` as a `resume`
  job asking the test cluster's own cores and memory (summed from `compose.yml`, since its containers run outside the
  job's cgroup). A fetch that fails changes nothing; a test that could not run is tried again on later nights, 3 times
  at most for one release; a failed test is not retried (delete its record to ask for a retry).
- `test`, the job itself (about 50 minutes, once per release): every stage on the new release, then `install,upgrade`
  from the site's pin with `UPGRADE_TO`; the cluster is always taken down, and test images of other releases are
  removed. It writes `<records>/<version>.json` and the log beside it.
- `status --json`, anyone, no network: `waiting` is true only for the newest release whose test passed from the
  release running now and that leaves no node's slurmd three or more series older (SchedMD accepts slurmd up to
  three major releases older; one is kept as margin, and a node that does not say its version blocks it); `ready` then holds the `version` and `sha256` to pass to `apply`.
- `sudo slurm_release.py apply --site SITE --version V --sha256 S`, the window, once running jobs have ended: only with
  a record of a passing test of V from the running release with that sha256, written by `test_user` (an admin) into a
  folder nobody else can write. It pins the site file's `[slurm]` table to V (a backup beside it, the rest of the file
  unchanged), runs `install.py upgrade-slurm` and returns its exit status: 0 done, 1 refused (nothing changed, the site
  file put back), 2 failed after a package changed (do not restart; its restore commands were printed), 3 the accounting
  conversion still runs (leave it running).

| Field | Default | Meaning |
|---|---|---|
| `checkout` | required | Absolute path of a checkout of these tools with `tests/integration`; the test runs from it. |
| `records` | required | Absolute path of the test results, owned by `test_user` or root and writable by nobody else. |
| `test_user` | required | Who runs `check` and the test; must be an admin, since `apply` trusts only their records. |
| `follow` | `series` | `patch`: newer releases of the running series only. `series`: also the next series, once it has its first maintenance release (`x.y.1`); one series at a time. |
| `project` | `gpujobs` | The project the test job is submitted under. |

### `[slurm]` (optional)

Build and install a SchedMD slurm release instead of the distribution's. Absent: the distribution's packages, as
before. Both fields or neither.

| Field | Default | Meaning |
|---|---|---|
| `version` | required with `sha256` | The release, e.g. `26.05.5`. |
| `sha256` | required with `version` | The digest of its tarball (`slurm-VERSION.tar.bz2` from SchedMD's GitHub release); the build refuses a download that differs. |
| `steps` | `[]` | Releases `install.py upgrade-slurm` installs first, each as `"<version>:<sha256>"`, in strictly ascending order and each older than `version`, which is the last release. |

`apply` then runs `slurm/install/slurm_build.py` (debs on Debian-like systems, rpms on Fedora or RHEL-like ones, built
once and kept under `/var/cache/gpujobs/slurm`; a build counts as finished only when it left its completion marker, so
an interrupted one is built again). It installs only SchedMD's `slurm-smd`, `slurm-smd-client`, `slurm-smd-slurmctld`,
`slurm-smd-slurmd` and `slurm-smd-slurmdbd` (rpms: `slurm`, `slurm-perlapi`, `slurm-slurmctld`, `slurm-slurmd`,
`slurm-slurmdbd`), with `apt-get install` or `dnf install`, after removing the distribution's slurm packages
(`slurm-wlm`, `slurmctld`, `slurmd`, `slurmdbd`, `slurm-client`) when they are installed; the refusal for a queue with
jobs applies first. `munge` and `mariadb-server` still come from the distribution. The apt version check is skipped,
and a machine without `apt-get` may use `dnf`. The packages' own scripts start their daemons when installed, so every
install and removal here runs under a temporary `/usr/sbin/policy-rc.d` that forbids it (an existing one is put back
afterwards), or, on `dnf` systems, with `slurmdbd`, `slurmctld` and `slurmd` masked until it is over.

Builds are kept per system release (`/var/cache/gpujobs/slurm/VERSION/rpm-fedora45-x86_64`): a build links the libraries
of the release it was made on. After the install the packages are **locked** (`apt-mark hold`, or `dnf versionlock`,
which also holds through a release upgrade), and the lock is lifted only around the installer's own installs, so a
routine update or an operating-system upgrade cannot put the distribution's slurm in their place (SchedMD's upgrade
guide says the same; measured 2026-11-08: a Fedora 44 to 45 upgrade put Fedora's slurm of the same version, built with
hwloc, in place of ours and the node was marked invalid). On `dnf` systems the install is followed by `dnf reinstall`,
because a build made there has the same name, version and release as the distribution's package, which `dnf install`
then leaves in place. **After an operating-system upgrade** run `apply` again: it builds for the new release and
installs that build (the lock may instead make the upgrade report that the old build's libraries are missing; then lift
it with `dnf versionlock delete slurm\*` or `apt-mark unhold 'slurm-smd*'`, upgrade, and run `apply`). An `apply` that
fails between lifting and restoring the lock leaves it off until the next `apply` that succeeds.

**Upgrading.** `apply` never changes the slurm release: it refuses when slurm is installed and its version differs from
`version`. A release is installed with `sudo install.py upgrade-slurm`, which takes it from the running one to `version`
through `steps` (slurm supports upgrades within a compatibility window: 23.11 to 25.05 is inside it, 23.11 to 26.05 is
not), and only with the queue empty: the first start of a newer `slurmdbd` converts the accounting database, which can
take tens of minutes, and a downgrade is not supported once a newer daemon has started.

`upgrade-slurm` refuses unless the queue is empty and the running version (`sinfo -V`) is older than `version`. It builds
every release first, so a build or download that fails leaves slurm untouched. It then sets every partition to
`State=INACTIVE` (so nothing more can be queued), checks the queue again, stops `slurmctld` and `slurmd`, and backs up to
`/var/backups/gpujobs/slurm-DATE/` (`mariadb-dump` of `slurm_acct_db`, tars of `StateSaveLocation` and `/etc/slurm`; each
file's size is printed in MB); only then does it stop `slurmdbd`. For each of `steps` and finally `version` it installs that
release's packages, runs `slurmdbd -D` as the slurm user until it answers (up to an hour; it fails if the process exits
first, and refuses to start if the `slurmdbd` service is running), stops it, starts `slurmdbd`, `slurmctld` and `slurmd`,
and checks that `sinfo` answers and `sacct` still lists the same jobs as before, for one seven-day window fixed at the
start. At the end it sets the partitions back to the states they had and runs `apply`.

A failure before any package changed (a job that slipped in, a failed backup) starts slurm again and reopens the
partitions. A later failure stops everything and prints the commands that restore the backup and the previous packages
(under the same no-daemon guard, with the daemons stopped again before the files come back, then `systemctl restart`, then
the partitions' states). If the foreground `slurmdbd` is still converting when the wait ends, it is left running (nobody
signals a converting database server); the command prints its process id and the log it writes, `slurmdbd-conversion.log`
beside the backup, and stops there without a restore.

### `[metrics]`

| Field | Default | Meaning |
|---|---|---|
| `mode` | `off` | `local` installs Prometheus and Grafana on this machine; `external` uses the URLs below; `off` installs no metrics. |
| `prometheus_url` | `""` | Used when `mode = "external"` (required then). |
| `grafana_url` | `""` | Used when `mode = "external"` (required then). When set, also where the portal's cluster file points Grafana (see section 8). |
| `grafana_port` | `3000` | Port of the local Grafana. Refused if it equals one of the stack's own ports (9101, 9306, 9445, 9090). |
| `retention_days` | `365` | How long Prometheus keeps samples; at least 1. |
| `allow_from` | `[]` | IP networks that may reach Grafana and the three exporters (`["100.64.0.0/10"]` for a Tailscale network); the machine itself always may. Empty: any network. Every other packet of those services is dropped by systemd (`IPAddressDeny=any` plus `IPAddressAllow=`). |

### `[portal]`

| Field | Default | Meaning |
|---|---|---|
| `mode` | `off` | `install` sets up Open OnDemand here; `existing` writes only the cluster file and links our page into a portal that is already there; `off` installs nothing. |
| `auth` | `dex-local` | Login method: `oidc`, `dex-ldap`, `dex-local`, `proxy` or `ssh` (no sign-in page; below). |
| `server_name` | `""` | Host name or IP address the portal answers to. Required when `mode = "install"`. |
| `tls_cert` | `""` | Path to the certificate. Empty (with `tls_key` empty) makes the installer generate a self-signed pair. Set both or neither. |
| `tls_key` | `""` | Path to its key. |
| `session_days` | `0` | How long one sign-in lasts. `0` keeps Open OnDemand's own 8-hour browser session; a number of days keeps the session in an encrypted cookie (`client-cookie:persistent`, encrypted with the cookie passphrase) that survives browser restarts and reboots, so a person signs in once per browser for that long. |
| `allow_from` | `[]` | IP networks that may reach the portal (Apache and Dex); empty: any network. systemd drops every other packet in both directions, so include the portal's own address (Apache fetches the login metadata from `server_name`) and, with `dex-ldap`, the LDAP server's. |
| `auth_settings` | `{}` | `[portal.auth_settings]`: string keys and values for the chosen `auth`, below. Any key not listed for the mode is refused. |

Every `*_file` setting names a **root-only file** (mode 0600, no group or other access); secrets never go in the
site file itself. The default home for these files is `/etc/gpujobs/portal/`.

`[portal.auth_settings]` for **`oidc`** (login through an OpenID Connect provider):

| Key | Required | Default | Meaning |
|---|---|---|---|
| `issuer` | yes | | The provider's `https://` URL. |
| `client_id` | yes | | Client id registered with the provider. |
| `client_secret_file` | yes | | Root-only file holding the client secret. |
| `remote_user_claim` | no | `preferred_username` | Claim that becomes the Linux user name. |
| `scope` | no | `openid profile email` | Requested scopes. |
| `user_map_match` | no | none (user name taken unchanged) | Lua pattern mapping the claim to the Linux user, for example when the claim is an email. |
| `crypto_passphrase_file` | no | `/etc/gpujobs/portal/crypto_passphrase` | Root-only file with the cookie passphrase (16 or more characters, no spaces). |

For **`dex-ldap`** (Open OnDemand's bundled Dex in front of an LDAP directory):

| Key | Required | Default | Meaning |
|---|---|---|---|
| `host` | yes | | `host:port` of the LDAP server (use the LDAPS port). |
| `user_base_dn` | yes | | Base DN under which users are searched. |
| `bind_dn` | no | anonymous | DN used to search; then `bind_pw_file` is required. |
| `bind_pw_file` | with `bind_dn` | | Root-only file with the bind password. |
| `user_filter` | no | `(objectClass=posixAccount)` | Search filter. |
| `username_attr` | no | `uid` | Attribute that is the Linux user name. |
| `email_attr` | no | `mail` | Email attribute. |
| `root_ca` | no | none | Absolute path of the CA certificate that signed the LDAP server's certificate. |
| `start_tls` | no | `false` | `true` to upgrade a plain connection with StartTLS. |
| `crypto_passphrase_file` | no | `/etc/gpujobs/portal/crypto_passphrase` | As above. |

For **`dex-local`** (accounts held by Dex itself):

| Key | Required | Default | Meaning |
|---|---|---|---|
| `users_file` | yes | | Root-only JSON file: a non-empty list of `{"username", "email", "hash", "userID"}` objects. `hash` is a bcrypt hash (never a password), `userID` a UUID, and the part of `email` before `@` is the Linux user. Dex's built-in default account (`ood@localhost`) is refused. |
| `crypto_passphrase_file` | no | `/etc/gpujobs/portal/crypto_passphrase` | As above. Create it with `openssl rand -hex 32 \| sudo install -m 0600 /dev/stdin /etc/gpujobs/portal/crypto_passphrase`. |

For **`proxy`** (no sign-in: a reverse proxy you trust vouches for every request). Needs `allow_from` set to the proxy's address:

| Key | Required | Default | Meaning |
|---|---|---|---|
| `user` | yes | | The Linux user every request is served as. |
| `key_file` | yes | | Root-only file with the key (16 or more characters, no spaces), the value of the `X-Frontdoor-Key` header the proxy adds to every request. Create it with `openssl rand -hex 32 \| sudo install -m 0600 /dev/stdin /etc/gpujobs/portal/proxy_key`. A request without the right key is refused with 403 and the header is removed before the apps see it. Only the key's SHA-1 digest is written to Apache's configuration. |

The installer stops and disables `ondemand-dex` in this mode if it is running (a machine switched from a Dex login).

For **`ssh`** (no sign-in page: each person reaches the portal through their own ssh forward, and the account that opens
the connection is who they are). For a machine people already reach with ssh and nothing else, such as one behind a
firewall that admits only ssh. Needs `allow_from = ["127.0.0.0/8"]` and no `tls_cert`/`tls_key`: Apache answers only the
relay on the machine itself, over plain http, and each ssh connection is the encryption.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `key_file` | yes | | Root-only file with the key (16 or more characters, no spaces) the relay adds to every request. Create it with `openssl rand -hex 32 \| sudo install -m 0600 /dev/stdin /etc/gpujobs/portal/relay_key`. Only its SHA-1 digest is written to Apache's configuration. |

How it works: the `portal` component installs `gpujobs-portal-relay.socket`, a Unix socket at `/run/gpujobs-portal.sock`
that anyone on the machine may open, and the relay behind it (`slurm/portal_relay.py`, its own unprivileged user, the key
handed over by systemd). sshd opens the socket as the person whose session forwards to it, so the kernel names their
account for each connection. The relay takes that name, never one the request carries, refuses root, passes exactly one
request per connection to Apache with the name and the key, and Apache serves the request as that user only when the key
matches. Each person adds to their own `~/.ssh/config`, in the block for this machine:

    LocalForward 18080 /run/gpujobs-portal.sock
    LocalForward 13000 localhost:3000

and, while `ssh <machine>` is open, uses http://localhost:18080 (the jobs page) and http://localhost:13000 (Grafana). Set `server_name = "localhost"`
and `metrics.grafana_url` to the forwarded Grafana address (`"http://localhost:13000"`), so the page's links work.
Pick local ports nothing else on people's own machines uses; any port works for the portal itself.

## 5. What each component installs

**`slurm`** (always on). Packages `slurm-wlm`, `slurmdbd`, `mariadb-server`, `munge`. Files:
`/etc/slurm/slurm.conf`, `cgroup.conf`, `gres.conf`, `/etc/slurm/slurmdbd.conf` (0600, owned by `slurm`),
`/etc/mysql/mariadb.conf.d/90-slurm.cnf`, and a `10-restart-protect.conf` drop-in for the `slurmctld`, `slurmd`,
`munge` and `slurmdbd` services (restart on failure, kept away from the kernel's out-of-memory killer). Four
partitions: `normal` (default; pre-emptable, resumes from its checkpoint), `steady` (never pre-empted), `cpu`
(every job without GPUs, either class, which `submit.py` sends there; capped by `cores_per_gpu`, as one partition
because slurm counts `MaxCPUsPerNode` per partition; never pre-empted by slurm, so `urgent.py` requeues a
resume-class one in its way itself) and `urgent` (higher tier, `AllowGroups=` the admin group; pre-empts `normal`),
plus `service` when the site file has `[service]` (the agent suspends `cpu` jobs to make room for it). Priority is `priority/basic`; the agent sets each waiting job's priority from its level. The
levels `low`, `normal` and `high` are created as slurm QOS (no access lists: every level is open to everyone).
Services restarted:
`mariadb`, `munge`, `slurmdbd`, then `slurmctld`, `slurmd`. Slurm's own ports (6817, 6818, 6819) are slurm's
defaults, not set by this installer.

**`tools`** (always on). Copies the `*.py` files directly under `slurm/` and everything under `slurm/ood_app/`
to `install_dir/slurm/` (plus `hostguard/hostguard.py` to `install_dir/hostguard/` when `hostguard` or `dgx_gate` is
on, and `dgx/` to `install_dir/dgx/` when `dgx_gate` is on), and installs one small shim per tool in `/usr/local/bin`: `gpujobs-submit`,
`gpujobs-status`, `gpujobs-pause`, `gpujobs-resume`, `gpujobs-kill`, `gpujobs-urgent`, `gpujobs-wait`,
`gpujobs-footprint`, `gpujobs-audit`, `gpujobs-sharing`. Each runs the tool with `/usr/bin/python3`.

**`agent`** (always on). A system service `gpujobs-agent.service` run as slurm's `SlurmUser` (`slurm`), plus the
site file copy `/etc/gpujobs/site.toml`, the group `admin_group`, and the state directories (section 8). Every
5 seconds it does: boot re-registration, giving each waiting job its place (level, then job id), releasing held jobs after
urgent work, failure notices (slurm's accounting is asked at most every 30 s), and a heartbeat (`agent/agent.json`). Enabled and started by `apply`.
Every 10 minutes it also looks for jobs waiting beside hardware they could use: the GPU unreserved while a job that
wants it waits for memory or cores, or cores free while a CPU-only job waits only for the `cpu` partition's core cap,
both judged on the machines the job may run on. Spare memory while every core is busy is not reported, since moving a
job would only change who uses the cores. Each finding names the running jobs holding what the waiting job lacks; the
jobs page and `status.py` show them one line per cause. Each new pair (waiting job, holding job) is recorded once as
an `idle` event on both jobs: the job's page shows it in its timeline and `gpujobs-wait` prints it to whoever waits on
either job, so nobody has to be messaged. A site that also wants notices pushed sets `notify_command` (told once an
hour per pair).

**`lease`** (only when `[lease]` is present). `slurm-gpu-lease-keeper.service`, a systemd **user** unit of
`keeper_user`, written to `/etc/systemd/user/` and enabled through that user's manager (nothing is written in the
user's home). It needs that user's systemd instance running
(login session or `loginctl enable-linger`). One long-lived process holds the cross-project GPU lease for slurm
as a whole. The component also makes `state_dir/keeper/`, owned by `keeper_user`, where the keeper rewrites its
state every 5 s; `submit` and `urgent` refuse while that heartbeat is more than 60 s old.

**`metrics`** (when `metrics.mode` is not `off`). Packages: `ca-certificates`, `curl`, `python3-requests`,
`python3-blessed`; with `mode = "local"` also `prometheus` and `grafana` (from Grafana's apt repository); with
GPUs on aarch64 also `golang-go` and `gcc`, because upstream publishes no aarch64 GPU exporter and it is built
from the pinned source. Downloaded and checksummed: Jobstats `v1.1.0` files, node exporter `1.12.1`, cgroup
exporter `v0.3.4`, GPU exporter `v0.2.4`, and the Open OnDemand helper app. Services: `gpujobs-node-exporter`
(port 9101), `gpujobs-cgroup-exporter` (9306), `gpujobs-gpu-exporter` (9445, only with GPUs); with
`mode = "local"` also `prometheus` (127.0.0.1:9090) and `grafana-server` (port `grafana_port`, with anonymous
read-only access and embedding turned on when the portal is on; the first admin password is generated into
`/etc/grafana/gpujobs.env`, mode 0600). Files include `/usr/local/jobstats/` (the `jobstats` command, linked at
`/usr/local/bin/jobstats`), slurm prolog and epilog scripts under `/etc/slurm/prolog.d/` and `epilog.d/`
(GPUs only), `/usr/local/sbin/slurmctldepilog.sh`, and lines added to `slurm.conf` for them. Per-job GPU memory
is measured by the agent and written as node-exporter textfiles in `state_dir/metrics/`. With the portal on, the
helper app is installed at `/var/www/ood/apps/sys/jobstats`.

The three exporters listen on all interfaces without authentication, because Prometheus scrapes them by node name (on a site with several nodes, over the network). They expose job ids, user names, projects and per-job use, and Grafana shows them without a login when the portal is on. Set `metrics.allow_from` to the networks that should reach them (on several nodes, include the Prometheus host); with it empty, any network that reaches the machine does.

**`portal`** (when `portal.mode` is `install` or `existing`). With `install`: Open OnDemand 4.2 and its NodeSource
and OnDemand apt repositories, `python-is-python3`, plus `ondemand-dex` (Dex modes), `libapache2-mod-auth-openidc`
(`oidc`) and `openssl` (self-signed certificate); the Apache modules `ssl` and `auth_openidc`; `/etc/ood/config/ood_portal.yml`
(0600, holds secrets); the cluster file `/etc/ood/config/clusters.d/CLUSTER_NAME.yml`; services `ondemand-dex`
(Dex modes) and `apache2`. The portal's web ports are Open OnDemand's own; this
installer does not set them. A self-signed pair is created at `/etc/ood/ssl/portal.{crt,key}`
(valid 365 days; an existing pair is kept) and trusted by the host. With `existing`, only the cluster file is
written. In both modes our page is linked at `/var/www/ood/apps/sys/gpujobs`.

## 6. Users and admins

- **Owners** pause, resume and kill their own jobs.
- **Admins** act on anyone's jobs, and **only admins may submit urgent work** (`urgent.py` refuses everyone
  else and prints how to be added). An admin is a member of `admin_group` (as a listed member or as that
  primary group) and has slurm's administrator level Operator; `install.py admin add USER` sets both.
- The rules are slurm's: the tools run `scontrol` and `scancel` as the user and print slurm's own refusal. The
  `urgent` partition also limits who may submit to it by group.

**The web page.** Open OnDemand logs the person in and runs our page as that person, so the page shows every
unfinished GPU job to everybody, and shows the pause, resume and kill buttons only on jobs the viewer owns, or
on any job for an admin ("Signed in as NAME (admin)"). The buttons call the same tools as the command line, so
slurm still decides. The page has four views: Jobs (job, user, project, class, name, description, state, level, plain-language
waiting reason, elapsed, GPUs, memory), History (filter by date, user, project and state), Use per project
(GPU-hours per project and period), and a job page (record, link to Jobstats when metrics are on, actions,
command and batch script, dependency chain, efficiency, events, and a log viewer with a text filter and an errors-only filter).
Page-changing requests must carry an `Origin` header that matches `Host`.

## 7. Daily use

The commands below are the shims installed in `/usr/local/bin`; from a checkout use `python3 slurm/NAME.py`.

    # submit: the job's command follows --; --class says whether re-running continues (resume) or redoes (restart);
    # --cpus N asks for N cores (a job without GPUs: at most the cap, cores_per_gpu); no ':' in --project
    gpujobs-submit --project myproj --what 'train the model' --class resume --resume-ok --gpu --name train1 -- python train.py

    # the level: where it waits among the others (low, normal by default, high; open to everyone); it never stops a
    # running job, only urgent work does. Change a waiting job's level later with slurm itself:
    gpujobs-submit --project myproj --what 'quick check' --class restart --gpu --level high -- python check.py
    scontrol update jobid=123 qos=low

    # shrink a WAITING job's memory request without cancelling it (megabytes: a "G" suffix is refused). Only while
    # it waits, and only downwards for a normal user; gpujobs-status and the jobs page read the new value. Size it
    # from the job's own last run (sacct -j <earlier id> -o MaxRSS), not a guess.
    scontrol update jobid=123 MinMemoryNode=12288
    # cores: NumCPUs alone is not enough -- slurm keeps waiting for the per-task count (CPUs/Task, MinCPUsNode)
    # (2026-10-31, job 1551). Change all three, then check with: scontrol show job 123 | grep -E 'CPUs/Task|MinCPUsNode'
    scontrol update jobid=123 NumCPUs=4 MinCPUsNode=4 CPUsPerTask=4

    # see the queue, who holds the GPU and why jobs wait
    gpujobs-status

    # pause (preview the loss first with --dry-run); a steady job needs --lose-progress
    gpujobs-pause --dry-run 123
    gpujobs-pause 123 --reason 'need the card'

    # resume a paused job; after an unexpected reboot, return the node to service (uses sudo)
    gpujobs-resume 123
    gpujobs-resume --node

    # cancel for good
    gpujobs-kill 123 --reason 'wrong config'

    # urgent work, admins only: prints the plan, pre-empts what can be paused, then submits
    gpujobs-urgent --project myproj --what 'demo run' --time 2:00:00 --dry-run -- python demo.py

    # wait until a job and everything chained after it has finished
    gpujobs-wait 123

The urgent plan counts free cores and memory only on nodes in the `urgent` partition that can start a job (not down, drained or
failed), so a lent machine that is off adds no room. When it pre-empts a `normal` job it also prints one `KNOCK-ON` line: the requeued `normal` job restarts above `cpu_fill` and may
pre-empt `cpu_fill` jobs for their cores, including ones that start meanwhile; the line names the `cpu_fill` jobs running now that the plan does not touch.

Other options: `submit --mem 24G`, `--after JOBID` to chain, `--gpus N`, `--dry-run` to print the batch script.
`gpujobs-footprint NAME` prints the peak memory a job is sized from. `gpujobs-status` and `gpujobs-audit` take no options and run at once. `gpujobs-audit` reports GPU work running outside slurm and the lease.

Sizing from a job's own history. Give a job `--name` and the same name on every run; the job wrapper measures each run.

- Memory, without `--mem`: the highest peak (card plus host side) of the name's 3 newest clean runs (exit 0, no signal),
  plus 10 percent, rounded up to a whole slurm `G` (1.07 GB), the way Kubernetes' Vertical Pod Autoscaler sizes from history rather
  than from the newest sample. With no clean run, the largest peak of its crashed or pre-empted runs that ran 10 minutes
  or more, plus 25 percent; with no run at all, the whole box for a GPU job and slurm's default for a CPU job.
- After a run was killed for memory (slurm's `OUT_OF_MEMORY`, or the host's OOM guard) and no clean run has ended
  since, the next ask is at least 1.2 times what that run was given and at least 105 MB more (the autoscaler's rule), never
  below the history's ask nor, after a memory guard stop, below what the run used (memory plus card) plus 10 percent, and the whole box when that is more than any node gives jobs. `submit` prints
  `[submit] memory: run J of 'NAME' was killed by ... ; asking NG`.
- A hand-set `--mem` below the history's peak, or `--mem 0` for a measured job, warns; it is never refused.
- Cores, for a CPU-only job without `--cpus`: `[submit] cores: measured A avg over N completed runs; would ask K (asking D now)`,
  from `sacct` (site field `cpus_from_history`, section 4). Under `recommend` (the default) the job still asks for D;
  under `enforce` it asks for K. A hand-set `--cpus` more than twice K warns. The figure is kept in the job's record
  (`cores_from_history`) so proposals can be compared with what the jobs then used. GPU jobs keep the cores held for the card.

What a run used. At the end of every run (each leg of a requeued job) the job's log gets two lines after the exit line,
and `gpujobs-wait` repeats the last run's under each job's final state:

    [job] used: memory peak 9.4 GB of 42.9 GB asked (22%); cores 1.7 on average of 8 held (21%)
    [job] for the same work next time: --mem 10G --cpus 3

Memory asked is the job's cgroup cap (its `--mem`); the peak is the host side plus, where the card shares system memory
(`unified_memory`), the card's, so it can pass 100% there. Cores held are slurm's cpuset; the average is the cgroup's core
time over the run's wall time. The advice is this run's peak plus 10 percent and its average times 1.25 (within the cores
the job may have); after a memory kill (the job's cap killed a process, or the host's OOM guard killed it) it is the larger of
1.2 times what the run was given and its peak plus 10 percent (the rule `submit --mem auto` applies); `--mem 0` (the whole box) when that is more than any node gives jobs. GPU jobs get no
`--cpus` advice. The report is per job, never per user, and changes nothing by itself: the project decides its next
request. A figure that cannot be read shows as unknown.

### Why a job waits

`gpujobs-status`, the job page, `gpujobs-submit` and `gpujobs-urgent` explain a job's room node by node. Free room is
never summed across nodes, since a job runs on one node. Each node the job may start on (a node of one of its
partitions that is not down, draining or barred with `--exclude`) gets one line: free memory out of the memory slurm
hands to jobs there, free cores out of the node's cores, and what the job lacks. A job can lack memory, cores, fast
cores (a GPU job bound to them), cores under its partition's cap (`cpu` and `service` have one), GPUs or shards. A
card is lent either whole or as shards, never both at once. The line then names the running jobs that hold most of
the missing resource, at most three, each with its project, class and running time, and says who can move it:

    [submit] room: n1: 10.5 GB of 31.5 GB and 18 of 20 cores free; it lacks 15.3 GB memory, held by #11 myproj 21.0 GB (restart, running 2 h 5 min, only its owner or an admin can move it)

"slurm pre-empts it for this job" marks a holder whose partition requeues its jobs and ranks below the waiting job's
partition (both read live from `scontrol show partition`). Slurm then makes the room itself. A queued job whose
figures show room gets the sentence "the figures show room on n1, yet slurm has not started it". Such a job waits for
a limit these figures leave out, such as a reservation or a QOS limit. A job without `--mem` asks for slurm's default
memory, which the line names and does not count. No line estimates a start time. `submit` prints its lines as
information. `urgent` prints the same lines and decides from them whether any node lacks memory or cores.

### Memory from a declared size

A project whose steps grow with a count it knows (detectors, files, epochs) states that count with
`--size COUNT UNIT`, for example `--size 30 detectors`. COUNT is a positive number and UNIT one word, compared without
case. Runs count as comparable when they share the project, the unit, and whether they use a card. The project makes
that declaration, and nothing is inferred from job names, so a project gives each kind of step its own unit. The
figure comes from this user's clean runs of that project and unit:

- Some earlier run had a count at or above the new one. The figure is the highest peak among the runs up to the
  smallest such count (each count judged by its 3 newest runs), so a larger count never asks below a smaller one.
- Every earlier count is smaller, and the new count is at most twice the largest. The peak of the largest count is
  scaled in proportion to the new count, and never asks below any measured peak. Proportional scaling is an upper
  bound for memory that grows at most linearly with the count, and no further than twice the measured range is
  assumed.
- The new count is more than twice the largest measured one. No figure is given, and the job asks as before.

The figure gets the same 10 percent margin and rounding as `--mem auto`. The newest run of the unit killed for
memory at this count or a smaller one, with no clean run of the unit since, raises it as a memory kill raises
`--mem auto` (section 7): a run that ran out of memory at a smaller count shows that a larger count needs more. The site
field `memory_from_size` (section 4) decides what happens next:

    [submit] memory from size: would ask 26G (30 detectors, scaled from myproj 'detectors' runs: 25 -> 21.1 GB peak)

Under `recommend` the line ends there and the job asks as before. Under `enforce` the job asks for the larger of this
figure and its name's own measured figure, unless the figure is the whole box, which is printed and never applied. A
hand-set `--mem` is used as given and warns when below the measured peak. The job's record keeps the count, the unit,
the policy, the figure, and whether the figure was applied.

### Checks before queuing

A fault that a job meets in its first minute costs the whole queue wait when it surfaces only once the job starts.
Every submission therefore states how it was checked before it was queued:

    gpujobs-submit ... --check 'python train.py --dry-run' -- python train.py
    gpujobs-submit ... --no-check 'the job is itself a measurement; there is nothing smaller to try'

`--check` runs its command on this machine before anything is queued. The command runs in the job's `--workdir` with
no input, with the GPU hidden (`CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` empty), and with `GPUJOBS_CHECK=1`
set. A submission made from inside a check is refused. The command is split like a shell line but runs without a
shell, so a check that needs pipes is a script. A non-zero exit refuses the job, and nothing is queued. The check
gets at most 240 s. After that limit its whole process group gets SIGTERM, then SIGKILL 5 s later, and the job is
refused. The check is the first process the kernel's memory killer takes, and it dies with `submit`. The job's
record and page keep the command, its exit, its time and the peak memory of its largest process. A check whose
peak passes 8 GB warns, since work that size belongs in slurm.

A good check is the pipeline's own dry run or stub run on the real inputs: it loads the configuration and the data,
builds the model, and takes a step or two on the CPU, in seconds and under 4 minutes. A script that calls
`submit` with a check allows it at least 300 s. When no check is possible, `--no-check` gives the reason, which
the record and page keep. A sweep submits the same check many times: a pass of the same command from the same
folder by the same user counts for 10 minutes. Parallel submissions of one check run it once, and the rest reuse
its pass. `gpujobs-urgent` takes the same two flags. A `--site` submission runs its check here, before the code is
sent.

The site field `require_check` (section 4) decides what happens when a submission states neither flag: nothing
(`off`), a warning (`recommend`), or a refusal naming both flags (`enforce`).

### Chains

`--after JOBID` starts a job once that job has finished successfully. When the job would become step 3 or later of a chain of
steps with the same project, class, GPUs and shards, `submit` warns:

    [submit] warning: this is step 3 of a chain of same-shape jobs (#10 -> #11 -> this): each step waits in the queue again and is sized alone; ...

Each step of a chain waits in the queue again and is sized alone. One job that runs the steps in order usually holds
its memory once and fails in one place. The warning is information only. A GPU step followed by CPU steps changes
shape and never warns.

### Sharing a card

By default a card is one token: one job at a time. A site that gives a card `shards` (section 4) lets several copies of one
job run on it together, each asking for a few shards (`--gres=shard:N`) and its share of the cores held for the card. Slurm
does not limit memory by shard, and on a card that shares system memory it cannot see the card's memory, so the bound that
holds is the sum of the jobs' declared `--mem`, which slurm's accounting enforces.

Measure first, once per named job. The probe is a short finite command that does the same work every time and exits 0
(a few epochs of the real job):

    gpujobs-sharing measure --project myproj --name train1 -- python train.py --epochs 3
    gpujobs-sharing show train1

`measure` queues one job that holds the whole card. It runs the probe alone twice (the gap between the two runs is the noise),
then 2, 3, 4 copies at once, each pinned to its own cores, and records the table. The level it records is the smallest number
of copies whose throughput is within the noise of the best, or 1 (no sharing) when none beats one job alone by more than the
noise; a level where a copy failed, or where the box's memory in use passed `sharing_memory_fraction` of its total, is not a
candidate. The record is kept with the job's other records and is ignored if the card type, the node's memory or the shard
count has changed since.

After that, `gpujobs-submit ... --gpu --name train1` shares by itself and prints what it decided, for example
`level 4 (measured 4; memory allows 15; cores allow any) -> --gres=shard:3 of 12, 3 cores`. The level is the smallest of the
measured level; the node's jobs memory (times `sharing_memory_fraction`) over the job's measured peak (section 7) plus 10 percent;
and the cores held for the card over `--cpus`. The job then asks for `ceil(shards per card / level)` shards and that share
of the cores. `--max-jobs N` replaces the measured level (memory and cores still bound it, and the line says so when you
are given fewer); `--max-jobs 1` forces a whole card. A job with no measurement, no `--name`, or `--gpus` above 1 asks for the whole card as
before. When the job's measured peak is above the one at measurement time, `submit` warns: measure again.

Rules that hold at every site that shares a card. Another user's job shares your card only when both asked for shards: a plain
`--gpu` job always gets a whole card alone. `gpujobs-urgent --gpu` counts every running shard job as in the way, like a
whole-card job. A shard job starts unless a compute process that belongs to neither a slurm job nor the lease protocol is on
the card (`gpujobs-audit` lists such processes). `gpujobs-status` shows `N shards` where a whole-card job shows `1 GPU`.

### Sending a job to another site

A site file may name other sites it sends jobs to (`[remote.NAME]`: `host`, `root`; see `site.example.toml`).
`submit.py --site NAME --repo R --entry E [--search DIR] [--data DIR] <the usual options> -- command` then:
builds the import closure of `E` with `dgx/assert_code_parity.py` (an import in the repo that sits on no search root is
a refusal); copies the closure and each `--data` folder (inside `R`) to `root/<project>/` there with `rsync --checksum`,
so what is already there is skipped; reads the closure's checksums back and refuses if they differ; and runs that
site's own `gpujobs-submit` over ssh from the same relative folder. Nothing is queued unless every step passed.
A `--data` folder outside `R` (datasets kept elsewhere) needs `data_root` (absolute) in the `[remote.NAME]` table: it is
synced by the same checksum rule to `data_root/<name>` there, where `<name>` is its path relative to `--data-base DIR`
(so `DIR/gpe/x` lands at `data_root/gpe/x`; without `--data-base`, its own folder name), and a folder already there
byte for byte is only read, not sent. `--data-env NAME` runs the command as `env NAME=<data_root> command`, so a job
that reads its datasets from an environment variable finds them.
`--service`, `--anywhere`, `--suspend` and `--log` belong to the owning site and are refused. `--after` takes ids on
the other site. The job is watched with that site's own tools; `status.py --site NAME` prints its queue. This site keeps
no copy of the job, so the local page does not list it (D24).

## 8. Operations and caveats

- **`render` and `check` need root once portal secrets exist.** They read the portal's root-only files (client
  secret, bind password, users file, cookie passphrase), and `check` also reads `slurmdbd.conf` (0600, owned by
  `slurm`). Without root, `check` of an installed system stops with "run the check with sudo" (exit status 2).
  With the portal in `install` mode, both commands also print one warning that the files needing the portal's
  secrets (`/etc/ood/config/ood_portal.yml`) are left out, so the output is incomplete. Run both with `sudo` for a
  site with the portal on.
- **An `ondemand-dex` package upgrade turns logins off.** The package regenerates the login configuration
  without our users, so nobody can sign in until you run
  `sudo python3 slurm/install/install.py --site site.toml apply --component portal`.
- **A self-signed certificate turns off strict transport security.** With `tls_cert` and `tls_key` empty the
  rendered `ood_portal.yml` sets `security_strict_transport: false`; with your own certificate it is `true`.
  Browsers warn about a self-signed certificate; give the portal a real one for regular use.
- **Grafana over plain HTTP does not show inside the HTTPS portal.** When `metrics.grafana_url` is empty the
  portal's cluster file points at `http://SERVER_NAME:grafana_port`, and a browser blocks that as mixed content
  inside the HTTPS portal. Set `metrics.grafana_url` to an HTTPS address of a Grafana that serves TLS (for
  example behind a TLS proxy) for the embedded panels to work.
- **Shared folders keep predictable names.** Scripts are `scripts/NAME-STAMP.sbatch` and logs `logs/NAME-STAMP.log`.
  A batch script is always a file the submitter creates: when the name is taken (the same name twice in one second,
  or a file or link someone left), it becomes `NAME-STAMP-2.sbatch`, `-3`, and so on, and is never written into or
  through. `apply` refuses, and `check` reports, a kernel with `fs.protected_regular`, `fs.protected_symlinks` or
  `fs.protected_hardlinks` at 0: those settings stop a log, record or request left under someone else's name from
  being written through (Ubuntu turns all three on).
- **A job's docker containers** are counted with it, signalled with it and removed when it ends: see "A job's docker
  containers" below.
- **Reboot recovery.** The agent's boot step runs once per boot (it compares `agent/boot_id` with the kernel's
  boot id): 30 seconds after `slurmctld` started, it runs `scontrol update nodename=N state=undrain` for every
  node that was not drained on purpose, so slurm registers the nodes again and purges the jobs they lost
  (otherwise that takes about 32 minutes). Jobs the agent saw held keep their place: when one is released at
  a lower priority, the recorded priority is put back, once (an admin who releases a job and lowers its priority
  at once must lower it again after the agent's next look). Separately, `slurm.conf` sets `ReturnToService=1`, so a
  node that went DOWN after an unexpected reboot stays DOWN until a person runs `gpujobs-resume --node`, so a
  job that wedged the machine is not restarted into the same wedge.
- **Where state lives.** `state_dir` (default `/var/lib/gpujobs`), root-owned, mode 0755. Inside:

  | Directory | Mode | Owner | Holds |
  |---|---|---|---|
  | `jobs/` | 1777 | root | one sidecar record per job |
  | `events/` | 1777 | root | pause, resume, kill and failure events, one file each (0644) |
  | `footprints/` | 1777 | root | measured memory use per run |
  | `metrics/` | 0755 | the slurm user | node-exporter textfiles for running jobs, written by the agent alone |
  | `logs/` | 1777 | root | each job's log, when `submit` is not given `--log` |
  | `scripts/` | 1777 | root | each job's batch script |
  | `manifests/` | 1777 | root | the DGX gate's code manifest per job (the directory is always created; used only with `dgx_gate`) |
  | `requests/` | 1777 | the slurm user | restore-order requests written by admins; the agent ignores files owned by a non-admin |
  | `agent/` | 0755 | the slurm user | the agent's heartbeat and what it has learnt |
  | `keeper/` | 0755 | `keeper_user` | the lease keeper's state and heartbeat (only with `[lease]`) |

  The sticky bit (the leading 1) lets everyone write but lets only a file's owner delete it.
- **Changing the site file** means running `apply` again; the services read the copy in `/etc/gpujobs/`.

- **Without a site file** (none at `/etc/gpujobs/site.toml`, or `$GPUJOBS_SITE` naming a missing or broken one) the
  tools refuse, naming the file and what is wrong with it; there is no built-in fallback. `$SLURM_GPU_STATE_DIR`
  and `$GPU_LEASE_DIR` override the site file's two directories (the unit tests use them).

### Memory on unified-memory nodes

On a node whose GPU shares the host's memory (`unified_memory = true`), a job's card memory comes out of the same pool
as everything else, but slurm's memory limit counts only the host side: a job can pass its ask with card memory and
starve the machine. Two pieces keep the machine responsive:

- **A measured reserve.** `jobs_memory_gb` holds back the memory the machine needs outside its jobs (slurm's
  `MemSpecLimit`). Measure it with `status.py --reserve` (see the `[[nodes]]` table); a typed guess errs one way or
  the other.
- **The memory guard.** It runs in the agent and watches the agent's own node; the installer refuses a unified-memory
  site with a GPU node elsewhere, which it could not watch. Every second it reads the node's free memory. Below the
  node's close level (the health check's: `close_below_gb`, 16 GB on a full node by default, or the node's `Extra`)
  it stops the running job furthest over its ask, counting the memory its cgroup holds (page cache left out, shared
  memory counted) plus the card memory of its own processes, if that is at least 1 GB over. One job at a time: the
  next waits 10 s for the memory to come back. When no job is over its ask it does nothing, and the health check and
  earlyoom act as before. Every decision is in the agent's log (`journalctl -u gpujobs-agent | grep "memory guard"`).
- **How a stop ends a job.** The guard writes a stop request for that run in the agent's own folder. The job's
  wrapper (every job `submit.py` and `urgent.py` make) acknowledges it in the job's record and kills the command: a
  `resume` job is requeued from its checkpoint, any other job fails, exit status 5, as for an earlyoom kill. A job
  with no wrapper (plain `sbatch`), or one whose wrapper has not acknowledged within 2 s, gets `scancel --signal=KILL`;
  so does one still running 15 s after acknowledging. Each outcome is a `memory_stop` event on the job, shown on the
  job's page and by `gpujobs-wait`, and sent through `notify_command`.
- **The next run.** After a stop, `submit.py --mem auto` asks for at least what the stopped run used (memory plus
  card) plus 10 percent. No job's size is changed for it.
- **Unchanged:** sites whose GPUs have their own memory, earlyoom's settings, and the health check's push-back of
  `resume` jobs. A job's container (Docker) runs outside its cgroup and is not counted for a stop.

### A job's docker containers

A container started from a job with the label `slurm-job=$SLURM_JOB_ID` (for example `docker run --label
slurm-job=$SLURM_JOB_ID ...`) belongs to that job's current run. The Docker daemon, not the job, owns the container, so
its processes are outside the job's cgroup and slurm's own kill never reaches them; the label is how this layer finds
them. The key `slurm-job` is reserved for this use on a site that installs this layer: the end-of-job script removes,
as root, every container carrying it with the number of a job that ended. A container belongs to a run when its label
is the job's id and it was made at or after that run started, so one left by an earlier run of the same id (after a
requeue) is never counted as the new run's.

- **Counted.** The wrapper adds the memory of the run's containers (their own cgroups) to the job's memory peak, and
  their processes' card memory to its GPU memory, so `--mem auto` and the end-of-run report include them: `[job] used:
  memory peak 4.2 GB (containers 3.7 GB) of 8.6 GB asked`. The jobs page counts them too, and the memory each node
  records as used outside its jobs (`status.py --reserve`) leaves them out, since the job's ask already covers them.
  A container that lives less than one sample (5 s) can be missed.
- **Excused while the job runs.** The audit (`status.py`, `urgent.py`, a shard job's card check) does not list GPU
  work in such a container while its job runs on this node holding a GPU or shards. Otherwise it is listed with the
  reason (`container of job 7, which is not running`). A container of a CPU job using the GPU is listed: that is GPU
  work slurm did not give out. On a node with several GPUs nothing is excused, because Docker is not held to the GPU
  slurm gave the job and which card the container uses is not checked.
- **Signalled.** When slurm ends the job (its end, a pre-emption, a requeue, `scancel`), the wrapper passes slurm's
  SIGTERM to the run's running containers (`[job] passed SIGTERM to container ID`), writes the leg's records, and then
  waits while they still run (`[job] waiting for container ID to stop`, then `[job] container ID stopped`). A detached
  container (`docker run -d`) thus gets the job's KillWait (120 s) to save its work. Its main process must handle
  SIGTERM: a shell running as a container's first process ignores it unless it sets a handler (`trap 'exit 0' TERM`,
  or `docker run --init`). The wait follows any SIGTERM, so a `scancel --signal=TERM` that ends your command but not the
  job leaves the job waiting until its containers stop (stop them, or `scancel` the job). Docker is asked once before
  the command starts and then at each sample; if it does not answer, the run says so once in its log and asks it no
  more, so a hung daemon cannot slow the memory guard's stop.
- **Removed when the job ends.** Slurm's end-of-job script (`Epilog=gpujobs_containers.sh`, the first `Epilog=`
  line, run by slurmd as root on every node of the job) runs `docker rm -f` on every container labelled with the job,
  whatever its state, when the job's allocation is released, so the card is free when the next job (an urgent one
  after a pre-emption, say) starts. It has no path in `slurm.conf`, so slurm sends it to configless (part-time) nodes
  too, at slurmd start and at every `scontrol reconfigure`. It always exits 0 (a failing end-of-job script would
  drain the node), gives docker at most 10 s to list and 30 s to remove, and logs every decision: `journalctl -t
  gpujobs-job-containers`.
- **Swept, and visible.** On a site with `[health_check]`, the health check (root, every node, every `interval_s`, 30 s by default) writes the
  map `/run/gpujobs/containers.json` of the node's labelled containers, which the agent and users outside the
  `docker` group read in place of docker, and removes a container whose job is in no unfinished job and that is more
  than 2 minutes old: what the end-of-job script missed (docker did not answer, a container with a restart policy
  after a reboot, one made after the script listed). Its log line says what it removed and what it kept.
  `status.py` lists every labelled container that no current run on this node owns, so one listed for minutes is one
  neither could remove.
- **Not covered.** A suspended job's containers keep running (only `cpu` jobs are suspended, for a service's stand-in);
  a container without the label, or with another job's number, is not the job's; the memory guard of unified-memory
  nodes does not count containers for a stop (a label is set by the job's user and could be forged); a site without
  `[health_check]` has no sweep and no map, so there readers outside the `docker` group (the agent included) see no
  containers, and only the end-of-job script removes them. Membership of the `docker` group amounts to root on the
  host: give it only to people you would give root.
- **Rootless Podman** is slurm's own route for containers inside a job (slurm's Containers Guide, "Integration with
  Podman using Slurm cgroup"): with `cgroupns = "host"` and `cgroup_manager = "cgroupfs"` the container's processes
  stay in the job's cgroup, so slurm counts, limits, signals and kills them like any other process of the job, and
  none of the above is needed. It is not yet checked on this layer's sites with a GPU.

### Part-time machines

A part-time machine is somebody's own computer (a laptop, a workstation) that lends its spare capacity to the
cluster as an extra compute node. List it in the site file as a `[[nodes]]` entry with `part_time = true` and an
`address` (section 4); it never runs the controller, the database or the web page.

**Read this before joining a machine.** The cluster knows who is who through one shared secret, the munge key. A
machine that holds the key can claim to be any user of the whole cluster, the admins included, and the controller can
run jobs on the machine as any cluster user. So root on a part-time machine has the cluster's power over everyone,
and the cluster's admins have root-like reach into that machine. Join only a machine whose root you trust the cluster
with, and if you are a colleague asked to join a shared site, know that joining gives whoever is root on your
machine cluster-wide power. For that reason `apply --part-time` does nothing until you pass `--i-accept-cluster-trust`.

The site file must have a `[slurm]` table: a part-time machine always runs the pinned slurm build, never the
distribution's. The user ids of every cluster user must be the same on the machine and on the controller (jobs run
as that uid), so check this first, or the controller refuses in step 2.

1. On the part-time machine, as root, with the controller's `slurm` ids (`getent passwd slurm` there) and address:

       sudo install.py apply --part-time NODE --controller CONTROLLER_ADDRESS \
           --slurm-uid UID --slurm-gid GID --i-accept-cluster-trust

   This creates `slurm` with the controller's ids (and stops if a different `slurm` already exists), installs `munge`
   and the pinned slurm build, installs the tools (`/opt/gpujobs`, with the health check), adds `--conf-server` to the
   package's own `slurmd` service so the machine takes its settings from the controller, opens port 6818 to the
   controller's address only when firewalld is running, and enables `munge` and `slurmd`. It also writes the site file
   to `/etc/gpujobs/site.toml` and makes the state directories (a job's wrapper there reads the one and writes its
   records into the other), gives `slurm` a home (`/var/lib/slurm`, with `.ssh`) and the login shell `/bin/sh` (sshd runs
   a key's forced command through the shell, and `nologin` would refuse it), and installs `rsync` (`rrsync` to
   `/usr/local/bin` when the distribution ships it only as documentation).
2. On the controller, as root, run by an admin through `sudo`:

       sudo install.py part-time add NODE

   This compares the uid of every cluster user (accounting users, the admin group, `slurm`) here and on the machine
   over ssh, and refuses with every difference listed. When they match it copies the munge key over ssh (never
   printed) and restarts munge and slurmd there. It uses your own ssh login, so you need an ssh login and `sudo` on
   the machine. Unless the site file says `shared_state = true`, it also makes an ssh key for `slurm` on the controller
   (`STATE_DIR/agent/ssh/id_ed25519`) and lets it into the machine's `slurm` account for one thing only: a read-only
   `rrsync` of the state directory.
3. Add the node to the controller's site file if it is not there, and `sudo install.py apply` on the controller.

**A hybrid CPU (Intel Core Ultra and similar).** Slurm cannot describe performance cores of two threads beside efficiency cores of one: `slurmd -C`
on such a laptop says `CPUs=22 ... CoresPerSocket=16 ThreadsPerCore=1` (and, built with hwloc, only 12 CPUs, so the controller drains the node "Low CPUs").
Build without hwloc (the rpm build does), give the node `cores_per_socket` as `slurmd -C` prints it, and measure what `keep_cpus` names: slurm's CpuSpecList is
in its own abstract ids, in an order that is not the operating system's. Reserve each id alone (`CpuSpecList=k` on a copy of the node's conf, `slurmd -D -v`)
and read the log's "Reserved machine CPU IDs"; then run a job given all its CPUs and check `cpuset.cpus.effective` is the cores you meant to lend.

**Fast and slow cores.** On a host whose cores differ (Arm big.LITTLE, Intel performance and efficiency cores) the same job runs several times
slower on some cores: measured on an Arm host, one thread of a small torch fit took 6 to 8 s on a 3.9 GHz Cortex-X925 core and 27 to 43 s on a
2.8 GHz Cortex-A725 core. Slurm hands out the lowest free cores and cannot be told to prefer a kind by a request alone: a node feature belongs to the whole node, and a
counted resource tied to sockets did not keep a request on the right cores in slurm 26.05.5 (`design notes`,
section 10); binding the GPU to a socket of fast cores does (below). The tools also report the fact. `python3 slurm/cores.py` prints the cores of each top speed (from cpufreq's
`cpuinfo_max_freq`, which separates the kinds; `cpu_capacity` does not), `slurm/install/detect.py` puts the same lines in its draft site file, and
on such a host every run's record (`jobs/<id>.json`, in each leg) has `cores`: the cores that leg was allowed and how many ran at each top speed, e.g.
`{"cpus": "0-3,8-11", "cores_by_max_mhz": {"2808": 6, "3900": 2}}`. A host whose cores are alike records nothing.

**A GPU job holds a fast core.** A GPU job's host threads (launching kernels, feeding data) decide how busy the card is. Measured 2026-11-04 on
this host (10 slow, 10 fast cores): an LSTM job on 4 slow cores kept the GPU about 45% busy; the same job on 12 cores (two of them fast) kept it 86
to 92% busy. Two ways to make sure of it, chosen by the site file:

- **With `fast_cpus` on the GPU node (D46, the proper way).** Slurm takes a node's core layout from hwloc, and hwloc can read it from a file. The
  installer writes `/etc/slurm/core_kinds.xml`, one package per core kind (slow first when the slow kind holds CPU 0), and a slurmd drop-in
  (`slurmd.service.d/20-core-kinds.conf`) that points slurmd there (`HWLOC_XMLFILE`). Slurm then sees two real sockets, the node line says
  `SocketsPerBoard=2`, and `gres.conf` binds the GPU and its shards to the fast socket (`Cores=`, slurm's own GPU-to-socket affinity), with
  `AutoDetect=off` (NVML's detection fails under a layout file; every GPU is listed by hand anyway). A GPU or shard job whose cores fit in the
  fast socket is submitted with `--ntasks=1 --gres-flags=enforce-binding`: it gets exactly its cores, all fast, and waits for them (or pre-empts
  a `cpu_fill` job holding them) rather than run on slow ones. `--ntasks=1` matters: without it slurm 26.05.5 gave a 2-core job the whole socket.
  A GPU job asking more cores than the fast socket has is not bound strictly; slurm gives it the fast cores first. Nor is a shard job when
  the cores held for the card (which the copies sharing it hold together) exceed the fast ones: bound one by one, the last copy would wait.
  CPU jobs, placed lowest core first, fill the slow socket first. A bound GPU job can still wait for a `cpu` or `service` job that started
  on fast cores while the slow ones were full, since slurm never pre-empts those: accepted, as such `cpu` jobs ran 3.6 minutes at the
  median and 36 at the 90th percentile (14 days, this box), while a GPU job left on slow cores is slow for its whole run. `--cpus` is used
  as given, and a named GPU job is offered its cores from its own runs (`cpus_from_history`), from the wrapper's report of each run,
  which times the command alone: slurm's elapsed time also holds the wait for the card. The installer refuses `fast_cpus` that are not
  all faster than the node's other cores, that are set on another machine's node, or with a slurmd built without hwloc (which ignores
  the layout file); `install.py check` exits 1 on any refusal. The layout file carries no comment: hwloc's own reader, used where
  libhwloc-plugins is not installed, refuses one, and slurmd then stops. Turning `fast_cpus` on or off renumbers slurm's cores under running jobs, so apply it with no job running on the node
  (the installer's apply refuses otherwise). `slurmd -C` run by hand still prints the real one-socket layout: only the slurmd service reads the file.
- **Without it, on a host whose cores differ (D33).** Slurm may give a job only slow cores, but a job holding more cores than the host has slow ones
  always has a fast one, and Linux moves the busy threads onto it. So `submit.py` and `urgent.py` raise a whole-card GPU job below `slow cores + 1`
  to that number and warn; the number comes from `slurm/cores.py` on the host (11 here, 0 where cores are alike). Not applied to a shard job, a
  CPU-only job, or when the detected cores are not the GPU node's own.

**Where a part-time job's records go.** The job wrapper runs on the machine, so a job's record (`jobs/<id>.json`, with
its legs and peaks), its events (a pushed-back job's `push_back`) and its log are written to the machine's own state
directory. Every minute the controller's agent copies each machine's state directory to
`STATE_DIR/remote/<node>/` (`rsync`, over ssh, read-only) and the page and tools read those copies too; the part-time card
shows when it last worked. A copied file counts only for a job that slurm's accounting says ran on that machine, and
belongs to the user accounting names for it (copies lose their owner). If the machines mount the controller's `state_dir`
(one shared filesystem), set `shared_state = true` and nothing is copied. `node_python` (default `/usr/bin/python3`) is
the interpreter a part-time job's wrapper runs with there, never the submitter's own.

**What goes to a part-time machine (D25).** A CPU-only `resume` job (no GPUs, not `--service`, not `urgent.py`) is submitted
to `cpu_fill,part_time` when a part-time node passes the checks at submission, so the scheduler uses an idle part-time
machine instead of leaving the job waiting for the main node. Slurm cannot tell whether a job is portable, so
`submit.py` asks the node itself (one ssh, `slurm/portability.py`). In order, the first failure leaves that node out:

1. size: `--cpus` and `--mem` must fit what the node lends (cores: its `cores_per_socket` or `cpus` less the ids in
   `keep_cpus`; memory: `jobs_memory_gb`);
2. the node answers over ssh (as the submitting user, `BatchMode`, 5 s);
3. the working folder is on storage both machines see. Shared means that on both machines the path's mount is a network
   filesystem (nfs, cifs) holding the same export path and the same file (inode); the server's name is ignored, since one NAS has different addresses on different machines. So a checkpoint written there survives a push-back to the main node.
   `/mnt/shared` itself is each machine's own disk; only the NAS shares mounted under it count, and only if both machines
   mount that one. Each decision is judged by the mount, never by a declared folder prefix;
4. the program (`command[0]`) exists on the node and is built for the node's architecture, and is a system file
   (`/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`) or on shared storage. A project `.venv` or anything under a home folder
   is on one machine's own disk and is left out;
5. every other path in the command (absolute, or existing relative to the working folder; `--opt=PATH` too) and `--repo`
   is shared or a system file, and exists on the node;
6. the node's copy of the job tools (the folder `job_wrapper.py` runs from) has the same `*.py` files as this machine's.
   A node with older tools is left out until the tools are reinstalled there.

To make a project's job eligible: run it with a system tool (or one on a shared NAS folder built for both architectures)
and put its working folder on a NAS share both machines mount. Every decision is printed (`[submit] part_time fedora:
included` or `excluded: why`, with `--dry-run` too) and stored in the job's record under `part_time`; a node left out is
named in the job's `--exclude`. `--no-anywhere` keeps one job off the part-time machines. `--anywhere` is the
submitter's word that the job is fit for them: it skips checks 2 to 5 (not the size check, and it is refused if no node
can take the job). A node with `default_jobs = false` in the site file takes only such jobs, so a colleague's machine
joining a shared site is never used unasked.

When the job starts on the part-time machine its wrapper checks paths, architecture and shared sources again (the
machine may have changed during the wait). If one no longer holds it prints why, requeues the job and leaves a
`launch_refused` event in the machine's state folder; `node_check.py` (root, every 30 s) then sets the job's
`ExcNodeList` to include the node once the job waits again (a running job cannot change its own list), only for an event
written by the job's own user. Three refusals in a row fail the job (exit code 6) rather than loop.

## 9. Uninstall

There is no uninstall command. Removal is manual and not tested. Stop and disable the services you installed
(`gpujobs-agent`, the lease keeper user unit, `gpujobs-node-exporter`, `gpujobs-cgroup-exporter`,
`gpujobs-gpu-exporter`, `prometheus`, `grafana-server`, `ondemand-dex`, `apache2`, then slurm's daemons), delete the
files listed in section 5, remove the packages with `apt-get remove`, remove the `/usr/local/bin/gpujobs-*`
shims, the symlink `/var/www/ood/apps/sys/gpujobs`, the group `admin_group`, and (only if you no longer want the job
records and the slurm accounting history) `state_dir` and the `slurm_acct_db` database. Run
`python3 slurm/install/install.py --site site.toml render --out DIR` to list the files this site file installs.
