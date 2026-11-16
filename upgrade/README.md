# upgrade

Dependency upgrades for the uv projects on this machine.

| File | What it does |
|---|---|
| `weekly.py` | `run [--project NAME] [--dry-run] [--deadline-hours H] [--poll-seconds S]`: the weekly upgrade of every project in `projects.toml` |
| `session_notice.py` | one line at session start when the project under the working directory has an upgrade held or just applied |
| `torch_upgrade.py` | by hand: `build` a side venv with a new PyTorch stack, `check` it on the GPU (one project, one PyTorch move) |
| `upgrade_common.py` | what the two share: `run`, `freeze`, hostguard, one slurm job for a check (submit, queue, cancel), probe comparison |
| `projects.example.toml` | the registry format; copy it to `projects.toml` (local to the machine, not tracked) and list your projects |
| `install/` | the systemd user service and timer (Sunday 05:00, `Persistent=true`) |

## The weekly rule

Per project: skip when a job named `<project>:...` is queued or running, or when `uv.lock` does not describe the venv
(`uv export --frozen` against `uv pip freeze`; torch built for another CUDA is a difference). Otherwise `uv lock --upgrade`
in `~/cache/upgrades/<name>/` (the scratch lock stays), a side venv from the new lock, then the check:

* `numbers`: the probe under the old and the new venv in ONE slurm job; applied only when every value is identical.
* `tests`: the registry's fast test command in the side venv, through slurm without a GPU; applied when it passes.

Every due project is prepared first and its check job submitted at once; each project is then applied or held as its
own job leaves the queue, so a check stuck in the queue delays only its own project. The run waits for the checks until
`--deadline-hours` after it started (default 3, preparing the projects counts; the queue is read every `--poll-seconds`,
default 30). A check still queued then is cancelled (`slurm/kill.py`) and its project is DEFERRED: the log and the state
file say `deferred` with the reason and the job id, nothing is applied, and the next weekly run tries again. A run that is
stopped early (an error, SIGTERM) cancels the checks still waiting and defers their projects the same way. The unit's
`TimeoutStartSec` (4 h) must stay above the deadline by the time the last results take to apply; `tests/upgrade` checks it.

Pass: commit `uv.lock` alone (`chore(deps): Weekly dependency upgrade`, never pushed), `uv sync --frozen`, hostguard,
restart the project's active services. Anything else is HELD (numbers moved, tests failed, `uv.lock` has uncommitted
changes, the lock changed while the check ran) and announced by `session_notice.py`. At the study boundary,
`weekly.py apply-held NAME` applies the kept lock without its check (the commit body says so, the log records it); it
refuses a hold whose tests failed (fix the code, then `weekly.py run --project NAME` applies it), an uncommitted lock, a
lock that moved since the hold, and a venv the lock no longer describes. A `tests` project keeps its PyTorch stack at the
locked version, because tests do not measure numbers or speed.

State: `~/.local/state/upgrades/<name>.json` and `weekly.log`; a dry run writes `dry-run/<name>.json`. The status is
`applied`, `held`, `deferred`, `skipped`, `up_to_date` or `error`. A `skipped` or `deferred` week leaves a standing
hold or a recent apply in the file (`session_notice.py` keeps announcing it); a `deferred` one is not announced itself,
since the next run simply tries again.

## Probe contract (`check = "numbers"`)

The probe writes `{"seconds_per_step": float, "values": [float, ...]}` to `--out`. Only `values` is compared, old venv
against new, and the project is applied only when the two lists are identical (relative difference exactly 0, same
length). A probe whose JSON has no `values` list is HELD with "probe output not comparable". A probe that is not
bit-reproducible across a PyTorch move (differences near 1e-7) is held every time torch changes; choose `tests` for it.

## Registry fields

`name`, `path`, `check` (`numbers` | `tests`), `command` (numbers: the probe script and arguments, run as
`<venv>/bin/python COMMAND --out FILE`, cwd = the project; tests: a shell-free command run with the side venv first on
PATH, cwd = the project), `gpu`, `services` (user units restarted after an apply when active), `sync` (the uv flags that
rebuild the venv, e.g. `["--extra", "dev"]`), `mem` and `cpus` (slurm memory and cores for the check; cores default to the cluster's 4).

## Install (the main session enables it after review)

    cp ~/dev/turing/preflight/upgrade/install/project-upgrades.{service,timer} ~/.config/systemd/user/ \
      && systemctl --user daemon-reload && systemctl --user enable --now project-upgrades.timer

Tests: `python -m pytest tests/upgrade` from `~/dev/turing/preflight`.
