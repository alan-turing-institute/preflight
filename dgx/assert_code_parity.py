#!/usr/bin/env python
"""Refuse a Turing DGX job whose code on the box differs from this clone's.

⛔ **WHY THIS EXISTS -- 2026-09-16, ~40 GPU-hours lost to a one-word mismatch.**
The 27-cell layer-norm-noise grid waited 5h40m for the card, won it at 17:54:42,
and died four seconds later::

    TypeError: run_serviced_wave() got an unexpected keyword argument 'serviced'

The CALLER (``train_grid_on_turing_dgx.py``) had been copied to the box four
minutes before the supervisor was armed. The CALLEE it imports
(``task14_stage0.py``) had not, so the box held a copy whose
``run_serviced_wave`` took no ``serviced`` parameter. Nothing trained; the card
then sat idle for forty hours.

⛔⛔ **AND NO EXISTING CHECK COULD HAVE CAUGHT IT, WHICH IS WHY THIS IS A GATE
AND NOT A HABIT.** The launcher imports stage 0 *inside a function*, on purpose
-- "stage 0 pulls torch and the scorers, which --status and --dry-run do not
need". So ``--status`` and ``--dry-run`` never import the stale module and both
pass. The mismatch is unreachable until the instant of real launch, i.e. after
the queue wait is already spent. A gate that walks the AST sees the deferred
import; a gate that runs the program does not.

⛔ **WHY GIT CANNOT DO THIS JOB HERE.** ``~/dev/turing/CLAUDE.md`` item 1 says
to prefer git-synced code, and for ``src/`` that is right. It is UNAVAILABLE for
DGX study code: ``notes/`` carries 0 tracked files (``.git/info/exclude``), and
every DGX launcher, cell definition and study script lives under ``notes/``.
Hand-copying is therefore structurally forced for exactly the files that broke,
which is why the copy must be VERIFIED rather than trusted. Item 2 already
demands an AST dependency closure and names
``autocast-rollout/scratchpad/b0prime_closure.py`` -- that file does not exist
on this machine. This is that tool, tracked so it cannot go missing again.

DECLARED EXCEPTIONS
-------------------
A file may legitimately differ -- uncommitted local work that must NOT reach the
box mid-grid is the standing case. Declare it with ``--allow-diff`` and a
reason. Every declared exception is PRINTED on each run, because an exception
nobody sees again is a permanent hole, and the point of this gate is that holes
stay visible. A bare ``--allow-diff`` with no reason is refused.

Exit 0 = every reachable file matches, or differs only as declared.
Exit 1 = refused, with each offending file named.
Exit 2 = usage error, or the closure could not be built.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

#: One ssh round trip for the whole closure. Seventy-odd sequential `ssh md5sum`
#: calls is a minute of latency, and a gate people wait for is a gate people
#: skip.
_REMOTE_TIMEOUT_S = 180

#: `md5sum` prints exactly two fields: the digest and the path.
_MD5SUM_FIELDS = 2

#: Length of a hex MD5, used to tell a digest line from a driver message.
_MD5_HEX_LEN = 32


def resolve_module(name: str, search: list[Path]) -> Path | None:
    """Map a dotted module name onto a file in one of the search roots.

    Args:
        name: Dotted module name as written in the import statement.
        search: Roots to try, in order -- the job's script directories and
            ``src``, mirroring what the entrypoint puts on ``sys.path``.

    Returns:
        The file backing ``name``, or None when it is not first-party (a stdlib
        or site-packages module, which the box's own environment owns).
    """
    relative = name.replace(".", "/")
    for root in search:
        for candidate in (root / f"{relative}.py", root / relative / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def closure(entry: Path, search: list[Path]) -> tuple[set[Path], set[str]]:
    """Every first-party file reachable from ``entry`` by any import.

    ⛔ Walks ``ast.walk`` over the WHOLE tree rather than the module's top-level
    body, so an import inside a function or an ``if`` is followed too. That is
    not a refinement: the 2026-09-16 failure was a deferred, function-body
    import, and a top-level-only walk would have reported the closure clean.

    Args:
        entry: The script the job actually runs.
        search: Roots passed to :func:`resolve_module`.

    Returns:
        The absolute paths reachable from ``entry`` (including ``entry``), and
        the dotted names that resolved to nothing on the given search roots.

    Raises:
        SyntaxError: If a reachable file does not parse. Refusing is correct --
            an unparseable file cannot be checked, and a gate that cannot
            measure must not report that it did.
    """
    seen: set[Path] = set()
    unresolved: set[str] = set()
    pending = [entry.resolve()]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                primaries = [alias.name for alias in node.names]
                probes = list(primaries)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                # `from pkg import name` may bring in a SUBMODULE or merely a
                # class; both are probed, but only `pkg` itself is a name whose
                # failure to resolve means anything -- a class never resolves to
                # a file, and counting those as unresolved imports produced two
                # false refusals the first time this check was written.
                primaries = [node.module]
                probes = [node.module]
                probes += [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in probes:
                found = resolve_module(name, search)
                if found is not None:
                    pending.append(found.resolve())
            for name in primaries:
                if resolve_module(name, search) is None:
                    unresolved.add(name)
    return seen, unresolved


def repo_module_index(local_root: Path) -> dict[str, Path]:
    """Top-level importable name -> file, for every module in the repository.

    Built once, so checking many unresolved names costs one walk.
    """
    index: dict[str, Path] = {}
    for path in local_root.rglob("*.py"):
        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue
        name = path.parent.name if path.name == "__init__.py" else path.stem
        index.setdefault(name, path)
    return index


def unreachable_first_party(unresolved: set[str], local_root: Path) -> dict[str, Path]:
    """Unresolved imports that DO exist in this repository.

    ⛔⛔ **THIS IS THE GATE'S OWN FAILURE MODE, FOUND 2026-09-18 BY TESTING IT.**
    Run without the right ``--search`` root, the closure silently shrank from 77
    files to 18 and still reported OK -- and the 59 it dropped included the very
    module whose staleness cost ~40 GPU-hours. A gate that quietly checks a
    quarter of what it was asked to is the "gate that finds nothing" failure this
    tool exists to prevent, committed by the tool itself. So an import that
    cannot be resolved but is demonstrably IN the repository is a refusal, with
    the search root it needs named.
    """
    index = repo_module_index(local_root)
    found: dict[str, Path] = {}
    for name in unresolved:
        head = name.split(".")[0]
        if head in sys.stdlib_module_names:
            continue
        if head in index:
            found[head] = index[head]
    return found


def digest(path: Path) -> str:
    """MD5 of a file, matching what ``md5sum`` prints on the box."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def remote_digests(host: str, root: str, relatives: list[str]) -> dict[str, str]:
    """MD5 every closure member on the box in ONE ssh call.

    Args:
        host: ssh destination for the DGX.
        root: The repository root ON the box.
        relatives: Repo-relative paths to checksum.

    Returns:
        Repo-relative path -> md5. A file absent on the box is simply missing
        from the mapping; the caller reports it as missing rather than equal.

    Raises:
        RuntimeError: If ssh itself fails, which is not a parity verdict.
    """
    listed = " ".join(f"'{name}'" for name in relatives)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"cd '{root}' && md5sum {listed}"],
        capture_output=True,
        text=True,
        timeout=_REMOTE_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0 and not completed.stdout.strip():
        msg = f"ssh to {host} failed: {completed.stderr.strip()[:400]}"
        raise RuntimeError(msg)
    out: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == _MD5SUM_FIELDS and len(parts[0]) == _MD5_HEX_LEN:
            out[parts[1].strip()] = parts[0]
    return out


def parse_allowances(raw: list[str]) -> dict[str, str]:
    """Turn ``path=reason`` strings into a mapping, refusing a bare path.

    Raises:
        ValueError: If a declaration carries no reason. An undeclared-reason
            exception is indistinguishable from someone silencing the gate.
    """
    out: dict[str, str] = {}
    for item in raw:
        path, separator, reason = item.partition("=")
        if not separator or not reason.strip():
            msg = f"--allow-diff needs a reason: {item!r} (use path=why)"
            raise ValueError(msg)
        out[path.strip()] = reason.strip()
    return out


def write_manifest(
    path: Path,
    *,
    entry: Path,
    host: str,
    digests: dict[str, str],
    allowances: dict[str, str],
) -> None:
    """Record the digests that were verified, so a later launch can re-check.

    ⛔ **THE DIGESTS ARE THE BOX'S, NOT THIS CLONE'S.** They differ for exactly
    the declared exceptions, and the check runs on the box -- so recording local
    digests makes a correctly-declared difference refuse every launch. A gate
    that always refuses is a gate that gets switched off, which is worse than no
    gate. Caught on the real box on 2026-09-18, before it could do that.

    ⛔⛔ **WHY A MANIFEST AND NOT A SECOND REMOTE DIFF.** The supervisor that
    actually launches runs ON the box, and the box has no copy of this clone to
    compare against -- so launch-moment parity cannot be a local-vs-remote diff.
    It has to be "do the files still match what was verified when this job was
    armed". That also closes a window the remote diff never covered: the
    2026-09-16 supervisor waited 5h40m between arming and launching, and
    anything could have been edited underneath it in that time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).isoformat(),
                "entry": str(entry),
                "host": host,
                "allowances": allowances,
                "digests": digests,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[parity] manifest written: {path} ({len(digests)} file(s))")


def check_manifest(path: Path, root: Path) -> int:
    """Verify THIS machine's files against a manifest, with no ssh at all.

    Run on the box immediately before launching. A declared difference is
    recorded in the manifest with its reason and is re-printed here, so the
    exception stays visible at the moment it actually matters.

    Returns:
        0 if every recorded digest still matches, 1 otherwise, 2 if the
        manifest cannot be read -- a check that cannot run must not pass.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[parity] REFUSED: cannot read manifest {path}: {exc}", file=sys.stderr)
        return 2

    digests: dict[str, str] = record.get("digests", {})
    if not digests:
        print(f"[parity] REFUSED: manifest {path} records no files.", file=sys.stderr)
        return 2

    moved: list[str] = []
    for name, expected in sorted(digests.items()):
        candidate = root / name
        if not candidate.is_file() or digest(candidate) != expected:
            moved.append(name)

    for name, reason in record.get("allowances", {}).items():
        print(f"[parity] DECLARED difference (from arming): {name}")
        print(f"[parity]   reason: {reason}")

    if moved:
        print(
            "[parity] REFUSED -- files changed since this job was armed.",
            file=sys.stderr,
        )
        for name in moved:
            print(f"[parity]   CHANGED or MISSING: {name}", file=sys.stderr)
        return 1

    print(f"[parity] OK -- all {len(digests)} file(s) match the arming manifest")
    return 0


def report_allowances(declared: list[str], allowances: dict[str, str]) -> None:
    """Print every declared difference, and flag any that no longer applies.

    ⛔ Printing is not decoration. An exception nobody sees again is a permanent
    hole, and a stale one should be retired rather than left standing.
    """
    for name in declared:
        print(f"[parity] DECLARED difference: {name}")
        print(f"[parity]   reason: {allowances[name]}")
    for name in allowances:
        if name not in declared:
            print(f"[parity] note: --allow-diff {name} did not differ;")
            print("[parity]   the exception is stale and should be dropped")


def covered_files(
    members: set[Path], local_root: Path, also: list[str]
) -> list[str] | None:
    """Repo-relative paths the manifest covers: the closure plus ``--also``.

    Returns None (a refusal) when an ``--also`` entry names no file -- a gate
    told to cover something it cannot find must not quietly cover less.
    """
    for name in also:
        if not (local_root / name).is_file():
            print(f"[parity] REFUSED: --also has no such file: {name}", file=sys.stderr)
            return None
    inside = {
        str(path.relative_to(local_root))
        for path in members
        if path.is_relative_to(local_root)
    }
    return sorted(inside | set(also))


def plan_comparison(
    arguments: argparse.Namespace, local_root: Path
) -> tuple[Path, list[str], dict[str, str]] | None:
    """Validate the inputs and work out what to compare.

    Every failure here is a refusal, never a smaller comparison: an entrypoint
    that is missing, an allowance with no reason, a file that will not parse and
    an ``--also`` that names nothing all mean the gate cannot measure what it
    was asked to, and a gate that cannot measure must not report that it did.
    """
    entry: Path = arguments.entry.resolve()
    if not entry.is_file():
        print(f"[parity] no such entrypoint: {entry}", file=sys.stderr)
        return None

    search = [local_root / s for s in arguments.search]
    search += [entry.parent, local_root / "src"]

    try:
        allowances = parse_allowances(arguments.allow_diff)
    except ValueError as exc:
        print(f"[parity] {exc}", file=sys.stderr)
        return None

    try:
        members, unresolved = closure(entry, search)
    except SyntaxError as exc:
        print(f"[parity] REFUSED: cannot parse {exc.filename}: {exc}", file=sys.stderr)
        return None

    stranded = unreachable_first_party(unresolved, local_root)
    if stranded:
        print(
            "[parity] REFUSED: these imports live in this repository but are not"
            " on any search root, so they would NOT have been checked:",
            file=sys.stderr,
        )
        for name, path in sorted(stranded.items()):
            root = path.parent
            print(
                f"[parity]   {name}  ->  add --search {root.relative_to(local_root)}",
                file=sys.stderr,
            )
        return None

    relatives = covered_files(members, local_root, arguments.also)
    if relatives is None:
        return None
    return entry, relatives, allowances


def compare_against_box(arguments: argparse.Namespace, local_root: Path) -> int:
    """Build the closure, checksum both sides, and report a verdict."""
    plan = plan_comparison(arguments, local_root)
    if plan is None:
        return 2
    entry, relatives, allowances = plan
    print(
        f"[parity] {len(relatives)} first-party file(s) reachable from {entry.name}",
        flush=True,
    )

    try:
        remote = remote_digests(arguments.host, arguments.remote_root, relatives)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[parity] REFUSED: could not read the box: {exc}", file=sys.stderr)
        return 2

    missing: list[str] = []
    differing: list[str] = []
    declared: list[str] = []
    for name in relatives:
        here = digest(local_root / name)
        there = remote.get(name)
        if there is None:
            missing.append(name)
        elif there != here:
            (declared if name in allowances else differing).append(name)

    report_allowances(declared, allowances)
    sys.stdout.flush()

    if missing or differing:
        print("[parity] REFUSED -- the box would run different code.", file=sys.stderr)
        for name in missing:
            print(f"[parity]   MISSING on {arguments.host}: {name}", file=sys.stderr)
        for name in differing:
            print(f"[parity]   DIFFERS  on {arguments.host}: {name}", file=sys.stderr)
        print(
            "[parity] Copy these across, or declare one with"
            " --allow-diff 'path=reason'.",
            file=sys.stderr,
        )
        return 1

    if arguments.write_manifest is not None:
        write_manifest(
            arguments.write_manifest,
            entry=entry,
            host=arguments.host,
            # What the BOX had when we verified it -- see write_manifest.
            digests={name: remote[name] for name in relatives if name in remote},
            allowances={k: v for k, v in allowances.items() if k in declared},
        )

    print(f"[parity] OK -- every reachable file matches {arguments.host}")
    return 0


def main() -> int:
    """Build the closure, checksum both sides, and report a verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "entry",
        type=Path,
        nargs="?",
        help="the script the job runs (not needed with --check-manifest)",
    )
    parser.add_argument("--host", default="aiphys", help="ssh destination")
    parser.add_argument(
        "--remote-root",
        default="/data/aiphys/fferoz/autocast-private",
        help="repository root on the box",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path.cwd(),
        # ⛔ This tool lives in its OWN repository (`~/dev/turing/preflight`), not
        # inside the project it checks, so it cannot derive the project root from
        # its own location -- an earlier version did, and after the split that
        # default pointed one directory above every project at once. The caller
        # names the root; `dgx_submit.sh` always passes it explicitly, and the
        # launch-time re-check on the box passes `--local-root "$PWD"`.
        help="root of the project being checked (default: the current directory)",
    )
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        help="extra import root (repeatable); 'src' is always included",
    )
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        help=(
            "extra repo-relative file to cover (repeatable). The closure walks"
            " PYTHON imports, so a shell supervisor, a launcher or a config that"
            " the job depends on is invisible to it unless named here"
        ),
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="on PASS, record the verified digests here for a launch-time re-check",
    )
    parser.add_argument(
        "--check-manifest",
        type=Path,
        default=None,
        help="verify THIS machine against a manifest and exit; no ssh, no remote",
    )
    parser.add_argument(
        "--allow-diff",
        action="append",
        default=[],
        help="path=reason for a file allowed to differ",
    )
    arguments = parser.parse_args()
    local_root: Path = arguments.local_root.resolve()

    if arguments.check_manifest is not None:
        return check_manifest(arguments.check_manifest, local_root)
    if arguments.entry is None:
        print(
            "[parity] an entrypoint is required without --check-manifest",
            file=sys.stderr,
        )
        return 2
    return compare_against_box(arguments, local_root)


if __name__ == "__main__":
    raise SystemExit(main())
