# Integration test cluster

A one-node slurm cluster in a container, for testing the portable GPU jobs build (installer, tools, agent, portal)
without touching this box's `/etc`, systemd, slurm daemons or queue.

    tests/integration/run.sh up      # build the image if needed, start the container, wait for the node to be idle
    tests/integration/run.sh smoke   # as alice: a trivial job shows in squeue, a gpu:2 job runs and is accounted
    tests/integration/run.sh shell [-u USER] [CMD...]   # a shell (root by default), or one command, in the cluster
    tests/integration/run.sh down    # remove the container and network (the image stays)

Every command that talks to slurm inside the container belongs in `run.sh`. The image is `gpujobs-testcluster:dev`
(about 570 MB), the compose project is `gpujobs-test`.

## What is in it

- Ubuntu 24.04 with the distro packages this box uses: `slurm-wlm`, `slurmdbd`, `mariadb-server`, `munge` (slurm
  23.11.4, the box's line), plus `sudo` and `python3` for the installer. systemd is PID 1, so the installer's
  `systemctl` calls work.
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
