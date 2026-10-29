# upgrade

Dependency upgrades for the uv projects on this machine. Policy: `~/.claude/software-upgrade-policy.md`
("RULING 2026-10-29", "A project upgrade"). Build notes and measurements: `the upgrade build report`.

| File | What it does |
|---|---|
| `weekly.py` | `run [--project NAME] [--dry-run]`: the weekly upgrade of every project in `projects.toml` |
| `session_notice.py` | one line at session start when the project under the working directory has an upgrade held or just applied |
| `torch_upgrade.py` | by hand: `build` a side venv with a new PyTorch stack, `check` it on the GPU (one project, one PyTorch move) |
| `upgrade_common.py` | what the two share: `run`, `freeze`, hostguard, one slurm job for a check, probe comparison |
| `projects.toml` | the registry |
| `install/` | the systemd user service and timer (Sunday 05:00, `Persistent=true`) |

## The weekly rule

Per project: skip when a job named `<project>:...` is queued or running, or when `uv.lock` does not describe the venv
(`uv export --frozen` against `uv pip freeze`; torch built for another CUDA is a difference). Otherwise `uv lock --upgrade`
in `~/cache/upgrades/<name>/` (the scratch lock stays), a side venv from the new lock, then the check:

* `numbers`: the probe under the old and the new venv in ONE slurm job; applied only when every value is identical.
* `tests`: the registry's fast test command in the side venv, through slurm without a GPU; applied when it passes.

Pass: commit `uv.lock` alone (`chore(deps): Weekly dependency upgrade`, never pushed), `uv sync --frozen`, hostguard,
restart the project's active services. Anything else is HELD (numbers moved, tests failed, `uv.lock` has uncommitted
changes, the lock changed while the check ran) and announced by `session_notice.py`; apply it by hand at the study
boundary. A `tests` project keeps its PyTorch stack at the locked version, because tests do not measure numbers or speed.

State: `~/.local/state/upgrades/<name>.json` and `weekly.log`; a dry run writes `dry-run/<name>.json`.

## Registry fields

`name`, `path`, `check` (`numbers` | `tests`), `command` (numbers: the probe script and arguments, run as
`<venv>/bin/python COMMAND --out FILE`, cwd = the project; tests: a shell-free command run with the side venv first on
PATH, cwd = the project), `gpu`, `services` (user units restarted after an apply when active), `sync` (the uv flags that
rebuild the venv, e.g. `["--extra", "dev"]`), `mem` and `cpus` (slurm memory and cores for the check; cores default to the cluster's 4).

## Install (the main session enables it after review)

    cp ~/dev/turing/preflight/upgrade/install/project-upgrades.{service,timer} ~/.config/systemd/user/ \
      && systemctl --user daemon-reload && systemctl --user enable --now project-upgrades.timer

Tests: `python -m pytest tests/upgrade` from `~/dev/turing/preflight`.
