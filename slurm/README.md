# GPU jobs through slurm

## 1. What this is

This directory turns a Linux machine with NVIDIA GPUs into a small slurm cluster with a job layer on top. Slurm
is the scheduler: it queues work, starts it when the GPU is free, and stops it again. On top of slurm the layer
adds:

- **Pause and resume.** A job that can continue from its latest checkpoint is paused by putting it back in the
  queue held, and resumed by releasing it. A job that cannot continue (a "steady" job) says so before it is paused.
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
- **Slurm 23.11 or newer, from the distribution packages.** Ubuntu 24.04 ships 23.11.4. The installer installs
  `slurm-wlm`, `slurmdbd`, `mariadb-server` and `munge`. That Ubuntu build has no NVML plugin (NVML is NVIDIA's
  management library), so GPUs are declared by device file in `gres.conf`, and slurm itself records no per-GPU
  use; the metrics component fills that gap.
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
   refuses while slurm has jobs**, or when `squeue` fails, because restarting the slurm daemons would requeue or
   kill them. Wait for the queue to empty, or install only other components with
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

TOML. Tables: the top level, `[[nodes]]` with `[[nodes.gpus]]`, `[lease]` with `[lease.services]`, `[metrics]`,
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
| `unified_memory` | `false` | True when the GPU shares system memory with the host (`nvidia-smi` reports total memory as `[N/A]`). `detect` sets it. Read and checked, but no tool acts on it yet. |
| `hostguard` | `false` | The job wrapper marks every job as the preferred out-of-memory victim (`oom_score_adj` 800, through the host memory guard in `hostguard/hostguard.py`), so a memory shortage ends a job rather than a login session. The `tools` component installs the guard when this is on. |
| `dgx_gate` | `false` | Require the DGX submission gate for jobs submitted with `--entry` (other jobs are not gated). The `tools` component installs the gate (`dgx/`) and the host memory guard it checks when this is on. |
| `notify_command` | `[]` | Command (list of arguments, no shell, 30 s limit) the agent runs when a job ends FAILED, OUT_OF_MEMORY, TIMEOUT or NODE_FAIL, with the job in the environment: `GPUJOBS_JOB` (id), `GPUJOBS_STATE`, `GPUJOBS_USER`, `GPUJOBS_NAME`. Empty means no notification. |
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

### `[metrics]`

| Field | Default | Meaning |
|---|---|---|
| `mode` | `off` | `local` installs Prometheus and Grafana on this machine; `external` uses the URLs below; `off` installs no metrics. |
| `prometheus_url` | `""` | Used when `mode = "external"` (required then). |
| `grafana_url` | `""` | Used when `mode = "external"` (required then). When set, also where the portal's cluster file points Grafana (see section 8). |
| `grafana_port` | `3000` | Port of the local Grafana. Refused if it equals one of the stack's own ports (9101, 9306, 9445, 9090). |
| `retention_days` | `365` | How long Prometheus keeps samples; at least 1. |

### `[portal]`

| Field | Default | Meaning |
|---|---|---|
| `mode` | `off` | `install` sets up Open OnDemand here; `existing` writes only the cluster file and links our page into a portal that is already there; `off` installs nothing. |
| `auth` | `dex-local` | Login method: `oidc`, `dex-ldap` or `dex-local`. |
| `server_name` | `""` | Host name or IP address the portal answers to. Required when `mode = "install"`. |
| `tls_cert` | `""` | Path to the certificate. Empty (with `tls_key` empty) makes the installer generate a self-signed pair. Set both or neither. |
| `tls_key` | `""` | Path to its key. |
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

## 5. What each component installs

**`slurm`** (always on). Packages `slurm-wlm`, `slurmdbd`, `mariadb-server`, `munge`. Files:
`/etc/slurm/slurm.conf`, `cgroup.conf`, `gres.conf`, `/etc/slurm/slurmdbd.conf` (0600, owned by `slurm`),
`/etc/mysql/mariadb.conf.d/90-slurm.cnf`, and a `10-restart-protect.conf` drop-in for the `slurmctld`, `slurmd`,
`munge` and `slurmdbd` services (restart on failure, kept away from the kernel's out-of-memory killer). Three
partitions: `normal` (default; pre-emptable, resumes from its checkpoint), `steady` (never pre-empted) and
`urgent` (higher tier, `AllowGroups=` the admin group). Priority is `priority/basic`. Services restarted:
`mariadb`, `munge`, `slurmdbd`, then `slurmctld`, `slurmd`. Slurm's own ports (6817, 6818, 6819) are slurm's
defaults, not set by this installer.

**`tools`** (always on). Copies the `*.py` files directly under `slurm/` and everything under `slurm/ood_app/`
to `install_dir/slurm/` (plus `hostguard/hostguard.py` to `install_dir/hostguard/` when `hostguard` or `dgx_gate` is
on, and `dgx/` to `install_dir/dgx/` when `dgx_gate` is on), and installs one small shim per tool in `/usr/local/bin`: `gpujobs-submit`,
`gpujobs-status`, `gpujobs-pause`, `gpujobs-resume`, `gpujobs-kill`, `gpujobs-urgent`, `gpujobs-wait`,
`gpujobs-footprint`, `gpujobs-audit`. Each runs the tool with `/usr/bin/python3`.

**`agent`** (always on). A system service `gpujobs-agent.service` run as slurm's `SlurmUser` (`slurm`), plus the
site file copy `/etc/gpujobs/site.toml`, the group `admin_group`, and the state directories (section 8). Every
5 seconds it does: boot re-registration, restoring the priority of released jobs, releasing held jobs after
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
from our job wrapper is written as node-exporter textfiles in `state_dir/metrics/`. With the portal on, the
helper app is installed at `/var/www/ood/apps/sys/jobstats`.

The three exporters listen on all interfaces without authentication, because Prometheus scrapes them by node name (on a site with several nodes, over the network). They expose job ids, user names, projects and per-job use, so allow ports 9101, 9306 and 9445 only from the Prometheus host (a firewall rule); with `mode = "local"` on one node, nothing outside the node needs them.

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
  else and prints how to be added; with no site file it checks nothing, see section 8). An admin is a member of `admin_group` (as a listed member or as that
  primary group) and has slurm's administrator level Operator; `install.py admin add USER` sets both.
- The rules are slurm's: the tools run `scontrol` and `scancel` as the user and print slurm's own refusal. The
  `urgent` partition also limits who may submit to it by group.

**The web page.** Open OnDemand logs the person in and runs our page as that person, so the page shows every
unfinished GPU job to everybody, and shows the pause, resume and kill buttons only on jobs the viewer owns, or
on any job for an admin ("Signed in as NAME (admin)"). The buttons call the same tools as the command line, so
slurm still decides. The page has four views: Jobs (job, user, project, class, name, description, state, plain-language
waiting reason, elapsed, GPUs, memory), History (filter by date, user, project and state), Use per project
(GPU-hours per project and period), and a job page (record, link to Jobstats when metrics are on, actions,
command and batch script, dependency chain, efficiency, events, and a log viewer with a text filter and an errors-only filter).
Page-changing requests must carry an `Origin` header that matches `Host`.

## 7. Daily use

The commands below are the shims installed in `/usr/local/bin`; from a checkout use `python3 slurm/NAME.py`.

    # submit: the job's command follows --; --class says whether re-running continues (resume) or redoes (restart)
    gpujobs-submit --project myproj --what 'train the model' --class resume --resume-ok --gpu --name train1 -- python train.py

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
  | `metrics/` | 1777 | root | node-exporter textfiles for running jobs |
  | `logs/` | 1777 | root | each job's log, when `submit` is not given `--log` |
  | `scripts/` | 1777 | root | each job's batch script |
  | `manifests/` | 1777 | root | the DGX gate's code manifest per job (the directory is always created; used only with `dgx_gate`) |
  | `requests/` | 1777 | the slurm user | restore-order requests written by admins; the agent ignores files owned by a non-admin |
  | `agent/` | 0755 | the slurm user | the agent's heartbeat and what it has learnt |
  | `keeper/` | 0755 | `keeper_user` | the lease keeper's state and heartbeat (only with `[lease]`) |

  The sticky bit (the leading 1) lets everyone write but lets only a file's owner delete it.
- **Changing the site file** means running `apply` again; the services read the copy in `/etc/gpujobs/`.

- **Without a site file** the tools still run (`site_config.load_optional()` returns nothing) and keep the settings
  of the single machine they were first written for. This is a fallback, and a wrong `$GPUJOBS_SITE` lands there
  silently. Then:
  - State and lease directories come from `$SLURM_GPU_STATE_DIR` and `$GPU_LEASE_DIR` if set, else the per-user
    `~/.local/state/slurm-gpu` and `~/.local/state/gpu-lease` (`gpu_jobs._site_dirs`).
  - `urgent` does not check the admin group, and takes the built-in service list (`service-a`, `service-b`) for
    what yields the card.
  - `submit` requires the lease keeper's heartbeat, and gates every `--entry` job with the DGX gate.
  - The job wrapper turns the host memory guard on, and metrics count as off.

## 9. Uninstall

There is no uninstall command. Removal is manual and not tested. Stop and disable the services you installed
(`gpujobs-agent`, the lease keeper user unit, `gpujobs-node-exporter`, `gpujobs-cgroup-exporter`,
`gpujobs-gpu-exporter`, `prometheus`, `grafana-server`, `ondemand-dex`, `apache2`, then slurm's daemons), delete the
files listed in section 5, remove the packages with `apt-get remove`, remove the `/usr/local/bin/gpujobs-*`
shims, the symlink `/var/www/ood/apps/sys/gpujobs`, the group `admin_group`, and (only if you no longer want the job
records and the slurm accounting history) `state_dir` and the `slurm_acct_db` database. Run
`python3 slurm/install/install.py --site site.toml render --out DIR` to list the files this site file installs.
