# preflight

Submit-time gates for the shared compute this account uses. One repository, so a
check written for one machine is reviewable, discoverable and reusable instead of
being re-invented per study.

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
    dgx/assert_gpu_free.py      leg 3 — the GPU is free, against a declared budget
    dgx/dgx_submit.sh           the four-leg gate; runs the legs, then launches
    tests/dgx/                  22 tests, every fix mutation-checked

## Use

    dgx/dgx_submit.sh --entry <script.py> [--search <import root>]... \
                      [--also <file>]... [--allow-diff 'path=reason']... \
                      [-- <launch command>]

The project being checked is taken from `--repo`, else `$AUTOCAST_DGX_REPO`, else
the git top level of the working directory — so running it from inside the
project's checkout needs no extra argument.

With no `--` command it runs every leg and launches nothing; so does
`AUTOCAST_DGX_GATE_CHECK_ONLY=1`.

    python3 -m pytest        # from this directory
