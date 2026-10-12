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

The root cause was durability, not carelessness. In `project-a`, `notes/`
carries zero tracked files and `outputs/` is gitignored, and every DGX launcher,
cell definition and study script lives under `notes/`. So hand-copying is forced
for exactly the files that break — and every DGX safety device the project had
built (an eval preflight, a GPU guard) also lived in those untracked trees, where
it could not be reviewed, found or reused.

A gate in `project-a` would also eventually travel into that project's
public counterpart, which is not where machine-level tooling belongs.

## The boundary: MACHINE vs PROJECT, not DGX vs cluster

A check that only needs a host, a path and a checksum lives here. A check that
imports the project's own code — Hydra config composition, a metric set, an
entrypoint's model graph — stays in that project, because it cannot run without
it. That is why the Isambard gate (`project-a/scripts/isambard_submit.sh`)
did **not** move: four of its five assert scripts import `autocast`/`hydra`, and
31 files reference it by path. Its generic legs can be pulled across one at a
time, never in one move.

## Layout

    dgx/assert_code_parity.py   leg 1 — the box runs the same code as this clone
    dgx/assert_gpu_free.py      leg 3 — query health; optional strict occupancy policy
    dgx/assert_hostguard.py     leg 4 — the box venv loads the current memory guard at startup
    dgx/dgx_submit.sh           the four-leg gate; runs the legs, then launches
    machine/gpu/               rotating GPU/process-owner and host-memory recorder; install.sh
    hostguard/hostguard.py      unified-memory guard: caps torch at (available - reserve), marks
                                GPU jobs as the preferred OOM victim (loaded via PYTHONPATH or .pth)
                                optional fixed per-job ceiling for shared boxes:
                                AUTOCAST_HOSTGUARD_BUDGET_GIB=60 caps at min(60 GiB, available - reserve)
    hostguard/install_pth.py    puts the guard in EVERY python a venv runs (smokes, scorers too)
    machine/earlyoom/           this box's earlyoom settings (source of truth; /etc is the install
                                target): install_earlyoom.py [--check]. Fixed-size thresholds, swap ignored
                                cache_first_oom.py reclaims cache before enabling earlyoom
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

## GPU incident records

Install once on the GPU host, from the account that owns this checkout (requires
`uv` at `~/.local/bin/uv`):

    sudo bash machine/gpu/install.sh

This enables `preflight-gpu-usage.service` under your account and installs a
root-owned cache-first supervisor for `earlyoom`. The installer performs one
bounded cache-reclaim test (no process killing) before changing earlyoom's service.
Every five seconds
the recorder writes GPU utilization, compute-process PID/name/memory,
UID/username/process start time/cgroup, and host
free/available/cache/swap/pressure to `/var/log/preflight/gpu.jsonl`. It retains
five files of up to 10 MiB each and syncs completed records to disk. Unsupported
GPU values and failed queries are explicit; command lines and environments are
not recorded. This records evidence, not a protective memory limit.

`dgx_submit.sh` retains its existing concurrency policy: one job of yours by
default, and refusal while another user occupies the GPU. Use `--max-jobs N`
to permit more of your own jobs and `--allow-other-users 'reason'` to explicitly
permit sharing with another user. Failed or malformed `nvidia-smi` queries also
refuse launch.

The existing `AUTOCAST_HOST_MEMORY_RESERVE_GIB` and
`AUTOCAST_HOSTGUARD_BUDGET_GIB` configure the PyTorch allocator guard separately.
`earlyoom` continuously reacts to low available system RAM across users; its
thresholds are independent of GPU utilization. A GPU allocation can still fail
while available RAM is high, so OOM killing is not a cure for every driver fault.
The supervisor tries `sync` plus `drop_caches=3` at 12 GiB available RAM, or when
a recently recorded GPU job has <=4 GiB raw free RAM and >=8 GiB cached. Repeated
non-urgent reclaims are limited to once per minute. It rechecks memory before
enabling the original earlyoom binary: SIGTERM below 8 GiB available, SIGKILL below
4 GiB, with existing process preferences unchanged. Reclaim has a 2-second limit,
shortened to 0.2 seconds in critical pressure; a stuck reclaim cannot block killing.
The old memory thresholds are read from `EARLYOOM_ARGS`, not replaced by GPU job
quotas. The original kernel OOM killer remains an independent emergency fallback.
Cache attempts/results/timeouts, memory recovery and earlyoom's PID/UID kill
records are in `journalctl -u earlyoom`. Cache-first protection is a mitigation,
not proof that NVIDIA's allocation failure or loss of connectivity is resolved.
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
