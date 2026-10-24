"""The installer's components: one module each, loaded by name in a fixed order.

A component module defines `NAME`, `enabled(site)`, `files(site)`, `packages(site)`, `refusals(site)` and
`post_install(site)`; optionally `apt_sources(site)` and `slurm_conf_lines(site)`. A `post_install` step is an argv,
or a function returning argvs, called when its turn comes (for commands that depend on what the earlier steps did).
Every external command any component runs goes through `run`, so a test swaps the one runner and nothing real is
touched.
"""

from __future__ import annotations

import importlib
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from levels import Level

if TYPE_CHECKING:
    from site_config import Site

ORDER = ["slurm", "tools", "agent", "lease", "metrics", "portal"]
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]

#: the site file as given on the command line; the agent component installs a copy of it
SITE_PATH: Path | None = None
#: the root `apply` writes under (`--root`), where the refusals read what is already installed
ROOT = Path("/")
#: `apply --replace-existing-slurm`: take over a slurm someone else set up (its settings files are replaced)
REPLACE_EXISTING_SLURM = False
#: slurmdbd is `Type=simple` on Ubuntu 24.04 (`systemctl show slurmdbd -p Type`): its restart returns before it answers
ACCOUNTING_WAIT_S = 60


def _real_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


_runner: Runner = _real_run


def set_runner(runner: Runner | None) -> None:
    """Replace the command runner (None restores the real one)."""
    global _runner
    _runner = runner or _real_run


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return _runner(argv)


#: Starts a long-lived process (`upgrade-slurm`'s foreground slurmdbd); tests swap it as they swap the runner.
popen = subprocess.Popen


def wait_for_accounting(sleep: Callable[[float], None] = time.sleep, limit_s: int = ACCOUNTING_WAIT_S, every_s: int = 1,
                        alive: Callable[[], bool] = lambda: True) -> None:
    """Until slurmdbd answers; `sacctmgr` exits 1 ("Connection refused") while it is down (measured, 23.11.4).

    `alive` tells the wait the process it waits for has not died, so a foreground slurmdbd that exits is not waited on."""
    for _ in range(max(limit_s // every_s, 1)):
        if run(["sacctmgr", "-n", "show", "cluster"]).returncode == 0:
            return
        if not alive():
            raise SystemExit("slurmdbd exited before it answered")
        sleep(every_s)
    raise SystemExit(f"slurmdbd did not answer within {limit_s} s")


def _lacks(*query: str) -> bool:
    """Whether `sacctmgr show QUERY` finds nothing (it exits 0 with no output then; measured, 23.11.4)."""
    done = run(["sacctmgr", "-n", "-P", "show", *query])
    if done.returncode != 0:
        raise SystemExit(f"FAILED ({done.returncode}): sacctmgr show {' '.join(query)}\n{done.stderr or ''}".rstrip())
    return not done.stdout.strip()


def accounting_adds(site: Site, user: str | None = None) -> list[list[str]]:
    """The `sacctmgr add`s for what slurm's accounting lacks of the site's cluster, its account, the levels (QOS, D18)
    and (given) the user.

    `AccountingStorageEnforce=associations` admits only users with an association to an account, and `add` of a
    record that exists exits 1 (measured, 23.11.4), so each is looked up first. A user accounting already knows keeps
    their default account: changing it left slurmctld refusing their jobs ("Invalid account") until it reloaded its
    accounting (container cluster, F1), and silently moving an existing user's default is not ours to do.
    """
    cluster, account = site.cluster_name, site.account
    adds = []
    if _lacks("cluster", "where", f"cluster={cluster}", "format=cluster"):
        adds.append(["sacctmgr", "-i", "add", "cluster", cluster])
    if _lacks("assoc", "where", f"cluster={cluster}", f"account={account}", "format=account"):
        adds.append(["sacctmgr", "-i", "add", "account", account, f"Cluster={cluster}",
                     f"Description={account}", f"Organization={account}"])
    # Every level is open to everyone: AccountingStorageEnforce=associations checks no QOS access lists.
    adds += [
        ["sacctmgr", "-i", "add", "qos", str(level)] for level in Level if _lacks("qos", "where", f"name={level}", "format=name")
    ]
    if user and _lacks("assoc", "where", f"cluster={cluster}", f"account={account}", f"user={user}", "format=user"):
        default = [f"DefaultAccount={account}"] if _lacks("user", "where", f"name={user}", "format=user") else []
        adds.append(["sacctmgr", "-i", "add", "user", user, f"Account={account}", *default, f"Cluster={cluster}"])
    return adds


def load(site: Site, names: list[str] | None = None) -> list[ModuleType]:
    """The enabled components among `names` (default all), in ORDER; a module not written yet is skipped."""
    out = []
    for name in ORDER:
        if names is not None and name not in names:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
            continue
        if module.enabled(site):
            out.append(module)
    return out


def slurm_conf_extra(site: Site) -> str:
    """The lines enabled components contribute to slurm.conf, joined."""
    parts = (getattr(m, "slurm_conf_lines", lambda _s: "")(site) for m in load(site))
    return "".join(p if p.endswith("\n") else p + "\n" for p in parts if p)
