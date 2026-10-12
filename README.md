# preflight

Submit-time gates and job management for the shared compute this account uses.
One repository, so a check written for one machine is reviewable, discoverable and
reusable instead of being re-invented per study. Job management lives here, not in
a project repository, for the same reason: in a project it ends up in gitignored
trees where it is never reviewed and never checked in.

## Why this is not inside a project repository

Written 2026-09-18, after a 27-cell grid waited 5 h 40 m for the Turing DGX card,
won it, and died four seconds later on a module that had been stale the whole
time — the caller had been copied to the box and the callee it imports had not.
~40 GPU-hours idle.

The root cause was durability, not carelessness. In `autocast-private`, `notes/`
carries zero tracked files and `outputs/` is gitignored, and every DGX launcher,
cell definition and study script lives under `notes/`. So hand-copying is forced
for exactly the files that break — and every DGX safety device the project had
built (an eval preflight, a GPU guard) also lived in those untracked trees, where
it could not be reviewed, found or reused.

A gate in `autocast-private` would also eventually travel into that project's
public counterpart, which is not where machine-level tooling belongs.

## The boundary: MACHINE vs PROJECT, not DGX vs cluster

A check that only needs a host, a path and a checksum lives here. A check that
imports the project's own code — Hydra config composition, a metric set, an
entrypoint's model graph — stays in that project, because it cannot run without
it. That is why the Isambard gate (`autocast-private/scripts/isambard_submit.sh`)
did **not** move: four of its five assert scripts import `autocast`/`hydra`, and
31 files reference it by path. Its generic legs can be pulled across one at a
time, never in one move.

## Layout

    dgx/assert_code_parity.py   leg 1 — the box runs the same code as this clone
    dgx/assert_gpu_free.py      leg 3 — the GPU is free, against a declared budget
    dgx/assert_hostguard.py     leg 4 — the box venv loads the current memory guard at startup
    dgx/dgx_submit.sh           the four-leg gate; runs the legs, then launches
    dgx/run_gpu_job.sh          host-wide lock + occupancy recheck, held for the foreground job
    machine/gpu/               rotating GPU/process-owner and host-memory recorder; install.sh
    hostguard/hostguard.py      unified-memory guard: caps torch at (available - reserve), marks
                                GPU jobs as the preferred OOM victim (loaded via PYTHONPATH or .pth)
                                optional fixed per-job ceiling for shared boxes:
                                AUTOCAST_HOSTGUARD_BUDGET_GIB=60 caps at min(60 GiB, available - reserve)
    hostguard/install_pth.py    puts the guard in EVERY python a venv runs (smokes, scorers too)
    machine/earlyoom/           this box's earlyoom settings (source of truth; /etc is the install
                                target): install_earlyoom.py [--check]. Fixed-size thresholds, swap ignored
    machine/watchdog/           systemd drop-in arming the board's hardware watchdog (60 s), so a
                                hung box reboots itself; install target /etc/systemd/system.conf.d/
    tests/dgx/                  26 tests, every fix mutation-checked
    jobs/queue_hold_watchdog.py resume a SIGSTOPped queue when the unit holding it ends
    jobs/pause_and_resume.py    stop a training unit at a fresh checkpoint, run another
                                job, then resume the first however that job ends
    tests/jobs/                 17 tests, mutation-checked
    tests/hostguard/            20 tests, every guard mutation-checked
    tests/machine/              4 tests, mutation-checked

## Use

    dgx/dgx_submit.sh --entry <script.py> [--search <import root>]... \
                      [--also <file>]... [--allow-diff 'path=reason']... \
                      [-- <launch command>]

The project being checked is taken from `--repo`, else `$AUTOCAST_DGX_REPO`, else
the git top level of the working directory — so running it from inside the
project's checkout needs no extra argument.

With no `--` command it runs every leg and launches nothing; so does
`AUTOCAST_DGX_GATE_CHECK_ONLY=1`.

## GPU launch lock and incident records

Install once on the GPU host, from the account that owns this checkout (requires
`uv` at `~/.local/bin/uv`):

    sudo bash machine/gpu/install.sh

This enables `preflight-gpu-usage.service` under your account and creates the
root-owned shared lock. Every five seconds the recorder writes GPU utilization,
GPU process PID/name/memory, UID/username/process start time/cgroup, and host
free/available/cache/swap/pressure to `/var/log/preflight/gpu.jsonl`. It retains
five files of up to 10 MiB each and syncs completed records to disk. Unsupported
GPU values and failed queries are explicit; command lines and environments are
not recorded. This records evidence, not a protective memory limit.

Run the wrapper ON THE GPU HOST around the actual foreground workload:

    bash dgx/run_gpu_job.sh -- uv run --no-sync python train.py

For a systemd job, put that wrapper inside its `ExecStart`, rather than wrapping
the short-lived `systemd-run` command. Local `dgx_submit.sh` launches use it
automatically; remote launch commands must include the wrapper on the remote
host. The common lock serializes participating jobs across users, and exit 75
means another launcher holds it. Launches outside this wrapper are not locked.
The occupancy gate refuses launch if `nvidia-smi` fails.

`PREFLIGHT_UV` selects the `uv` binary; `PREFLIGHT_GPU_LOCK` overrides the lock
path for tests. The existing `AUTOCAST_HOST_MEMORY_RESERVE_GIB` and
`AUTOCAST_HOSTGUARD_BUDGET_GIB` configure the PyTorch allocator guard separately.
To inspect deployment: `systemctl status preflight-gpu-usage.service` and
`tail -1 /var/log/preflight/gpu.jsonl` (timestamps are UTC).

A priority job that holds a running queue with SIGSTOP and resumes it from an EXIT
trap strands the queue if it is killed hard or never ends. Watch it from outside:

    systemd-run --user --unit=<name> jobs/queue_hold_watchdog.py \
        --holder-unit <unit> --queue-pid <pid> [--stuck-if '<cmd, exit 0 = stuck>']

To give the GPU to another job and continue a training run afterwards without losing
work (SIGTERM at a fresh checkpoint, never SIGSTOP; see the module docstring for why):

    systemd-run --user --unit=<name> jobs/pause_and_resume.py \
        --pause-unit <A> --checkpoint <A last.ckpt> --run-unit <B> --run-command '<B>' \
        --resume-unit <A resume> --resume-command '<A resume>' [--workdir <dir>]

    uv run --no-project --with pytest python -m pytest
