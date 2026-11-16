"""Run on another site, from the job's folder, by `remote.py` (piped over ssh so that site cannot hold a stale copy):
does Python there load each first-party name from the synced project folder?

`remote.py` sends the project's files and compares their checksums, which proves the bytes arrived. It does not prove
they are the ones the job imports: a package folder without its `__init__.py` is only a namespace portion, and a regular
package of the same name later on the path (a venv's editable install of another checkout) replaces it. The job then runs
other code, or none. `find_spec` answers where Python would load a name from, without running it (an entry script that
trains on import, an interpreter without the job's libraries). Argument: one JSON object, `project` (the synced project
folder as seen from the current folder), `roots` (import roots inside it) and `names` (top-level names to locate).
Exit 1 names each name loaded from outside the folder and where from.

Stdlib only; runs on whatever Python the job uses.
"""

import importlib.util
import json
import os
import sys

#: How many offenders one refusal lists: the message is read in a terminal, and the first few say what is wrong.
LISTED = 5


def outside(project: str, names: list[str]) -> list[str]:
    """One `name -> where` line for each name Python would load from outside `project` (or not find at all)."""
    lines = []
    for name in names:
        found = importlib.util.find_spec(name)
        places = [] if found is None else [found.origin, *(found.submodule_search_locations or [])]
        places = [os.path.realpath(place) for place in places if place and place != "built-in"]
        stray = [place for place in places if os.path.commonpath([project, place]) != project]
        if found is None:
            lines.append(f"{name} -> not found")
        elif stray:
            lines.append(f"{name} -> {stray[0]}")
    return lines


def main(argument: str) -> int:
    spec = json.loads(argument)
    project = os.path.realpath(spec["project"])
    sys.path[:0] = [os.path.join(project, root) for root in spec["roots"]]
    lines = outside(project, spec["names"])
    if not lines:
        return 0
    shown = "; ".join(lines[:LISTED]) + (f"; and {len(lines) - LISTED} more" if len(lines) > LISTED else "")
    print(
        f"on this machine Python loads these from outside the synced folder {project}: {shown}. A package folder "
        "without an __init__.py is replaced by a regular package of the same name found later on the path (a "
        "venv's editable install of another checkout): add the missing __init__.py, or make that install point "
        "here, then submit again",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
