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
| `notify_command` | `[]` | Command (list of arguments, no shell, 30 s limit) the agent runs when a job ends FAILED, OUT_OF_MEMORY, TIMEOUT or NODE_FAIL, with the job in the environment: `GPUJOBS_JOB` (id), `GPUJOBS_STATE`, `GPUJOBS_USER`, `GPUJOBS_NAME`. The name is whatever its submitter typed, with control characters shown as `?`: quote it wherever it goes into a shell command or a mail header. Empty means no notification. |
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
| `cores_per_gpu` | absent | Cores held for GPU jobs, per GPU. A GPU job gets this many per GPU by default, and CPU-only jobs (partition `cpu`) are capped by `MaxCPUsPerNode` at the node's cores less those held, so a free GPU never waits for cores taken by CPU jobs (slurm.conf(5)'s own use of `MaxCPUsPerNode`; slurm 23.11 has no `RestrictedCoresPerGPU`). Set it from a measurement: the fewest cores at which your GPU jobs stop getting faster. With several nodes the cap is the tightest node's and applies to every node, CPU-only ones included. At least 1, and must leave CPU-only jobs a core. Absent: nothing is held, and a GPU job's default is its GPUs' share of the node's cores (all of them on a one-GPU node), so any CPU job running holds a GPU job back. `submit.py` refuses a CPU-only job asking for more than the cap, which would otherwise wait forever. |

### `[[nodes.gpus]]`

| Field | Default | Meaning |
|---|---|---|
| `index` | required | GPU index. |
| `uuid` | required | From `nvidia-smi --query-gpu=uuid --format=csv,noheader`. |
| `device` | required | Device file for `gres.conf`, for example `/dev/nvidia0`. |
| `type` | required | Gres type name. The node's `Gres=gpu:TYPE:COUNT` uses the type of its first GPU and the number of its GPUs. |

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
`gpujobs-footprint`, `gpujobs-audit`. Each runs the tool with `/usr/bin/python3`.

**`agent`** (always on). A system service `gpujobs-agent.service` run as slurm's `SlurmUser` (`slurm`), plus the
site file copy `/etc/gpujobs/site.toml`, the group `admin_group`, and the state directories (section 8). Every
5 seconds it does: boot re-registration, giving each waiting job its place (level, then job id), releasing held jobs after
urgent work, failure notices (slurm's accounting is asked at most every 30 s), and a heartbeat (`agent/agent.json`). Enabled and started by `apply`.

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

Other options: `submit --mem 24G` (otherwise the last measured run of `--name` plus 10 percent, else the whole
box for a GPU job), `--after JOBID` to chain, `--gpus N`, `--dry-run` to print the batch script. `gpujobs-footprint NAME`
prints a job's last measured peak memory. `gpujobs-status` and `gpujobs-audit` take no options and run at once. `gpujobs-audit` reports GPU work running outside slurm and the lease.

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
- **A job's docker containers are its own to stop.** The wrapper counts a container labelled `slurm-job=JOBID`
  toward the job's GPU memory, but the container runs outside the job's cgroup, so slurm's final SIGKILL reaches
  only the `docker` client. A job that starts a container must stop it on SIGTERM (an attached `docker run`
  passes SIGTERM on; `docker run -d` does not); a container left running holds the GPU outside slurm and the
  lease, and the job's next run waits for the card it holds. Nothing stops leftover containers for you yet.
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

**Where a part-time job's records go.** The job wrapper runs on the machine, so a job's record (`jobs/<id>.json`, with
its legs and peaks), its events (a pushed-back job's `push_back`) and its log are written to the machine's own state
directory. Every minute the controller's agent copies each machine's state directory to
`STATE_DIR/remote/<node>/` (`rsync`, over ssh, read-only) and the page and tools read those copies too; the part-time card
shows when it last worked. A copied file counts only for a job that slurm's accounting says ran on that machine, and
belongs to the user accounting names for it (copies lose their owner). If the machines mount the controller's `state_dir`
(one shared filesystem), set `shared_state = true` and nothing is copied. `node_python` (default `/usr/bin/python3`) is
the interpreter a part-time job's wrapper runs with there, never the submitter's own.

## 9. Uninstall

There is no uninstall command. Removal is manual and not tested. Stop and disable the services you installed
(`gpujobs-agent`, the lease keeper user unit, `gpujobs-node-exporter`, `gpujobs-cgroup-exporter`,
`gpujobs-gpu-exporter`, `prometheus`, `grafana-server`, `ondemand-dex`, `apache2`, then slurm's daemons), delete the
files listed in section 5, remove the packages with `apt-get remove`, remove the `/usr/local/bin/gpujobs-*`
shims, the symlink `/var/www/ood/apps/sys/gpujobs`, the group `admin_group`, and (only if you no longer want the job
records and the slurm accounting history) `state_dir` and the `slurm_acct_db` database. Run
`python3 slurm/install/install.py --site site.toml render --out DIR` to list the files this site file installs.
