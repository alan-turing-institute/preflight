"""Tests for the Turing DGX code-parity gate.

Each test names the mutation it would catch. A gate seen only green is not
evidence -- that is how the 2026-09-16 grid passed `--status` and `--dry-run`
and still died four seconds after winning the card.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import assert_code_parity as mod
import pytest
from assert_code_parity import (
    closure,
    digest,
    main,
    parse_allowances,
    resolve_module,
)


@pytest.fixture(autouse=True)
def _box(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate has no tracked default box: the tests name one, as a caller's environment would."""
    monkeypatch.setenv("AUTOCAST_DGX_HOST", "site-b")
    monkeypatch.setenv("AUTOCAST_DGX_ROOT", "/data/site/user1/project-a")


def write(path: Path, body: str) -> Path:
    """Write a module, trimming the fixture's own indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """The 2026-09-16 shape: a caller that imports its callee INSIDE a function.

    This is not a contrived arrangement -- the real launcher defers the import
    on purpose, so that `--status` and `--dry-run` need not pull torch.
    """
    root = tmp_path / "repo"
    write(root / "scripts" / "callee.py", "VALUE = 1\n")
    write(
        root / "scripts" / "entry.py",
        """\
        import argparse

        def run():
            from callee import VALUE  # deferred on purpose
            return VALUE
        """,
    )
    return root


def test_closure_follows_a_function_body_import(study: Path) -> None:
    """MUTATION: walking `tree.body` instead of `ast.walk(tree)`.

    That mutation makes the closure miss exactly the import that broke the grid,
    and the gate would report a stale box clean.
    """
    entry = study / "scripts" / "entry.py"
    found, _ = closure(entry, [study / "scripts"])
    assert study / "scripts" / "callee.py" in found


def test_stdlib_imports_are_not_first_party(study: Path) -> None:
    """MUTATION: resolving unknown modules to a path anyway.

    `argparse` belongs to the box's own environment; checksumming it would make
    the gate refuse on every interpreter difference.
    """
    assert resolve_module("argparse", [study / "scripts"]) is None


def test_a_differing_file_is_refused(
    study: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: comparing only the entrypoint, not the whole closure.

    The entrypoint was byte-identical on 2026-09-16; the callee was not.
    """
    monkeypatch.setattr(
        mod,
        "remote_digests",
        lambda *args: {
            "scripts/entry.py": digest(study / "scripts" / "entry.py"),
            "scripts/callee.py": "0" * 32,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["p", str(study / "scripts" / "entry.py"), "--local-root", str(study)],
    )
    assert main() == 1
    assert "DIFFERS" in capsys.readouterr().err


def test_a_missing_remote_file_is_refused_as_missing(
    study: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: treating an absent remote file as equal.

    `dict.get` returning None must not compare equal to a local digest.
    """
    monkeypatch.setattr(
        mod,
        "remote_digests",
        lambda *args: {"scripts/entry.py": digest(study / "scripts" / "entry.py")},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["p", str(study / "scripts" / "entry.py"), "--local-root", str(study)],
    )
    assert main() == 1
    assert "MISSING" in capsys.readouterr().err


def test_a_declared_difference_passes_and_is_printed(
    study: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: silencing a declared difference instead of printing it.

    An exception nobody sees again is a permanent hole.
    """
    monkeypatch.setattr(
        mod,
        "remote_digests",
        lambda *args: {
            "scripts/entry.py": digest(study / "scripts" / "entry.py"),
            "scripts/callee.py": "0" * 32,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "p",
            str(study / "scripts" / "entry.py"),
            "--local-root",
            str(study),
            "--allow-diff",
            "scripts/callee.py=uncommitted work that must not reach the box",
        ],
    )
    assert main() == 0
    assert "DECLARED difference" in capsys.readouterr().out


def test_an_allowance_without_a_reason_is_refused() -> None:
    """MUTATION: accepting a bare path.

    Without a reason, an exception is indistinguishable from silencing the gate.
    """
    with pytest.raises(ValueError, match="needs a reason"):
        parse_allowances(["scripts/callee.py"])


def test_a_stale_allowance_is_reported(
    study: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: dropping the no-op notice.

    An exception that no longer applies should be retired, not left standing.
    """
    monkeypatch.setattr(
        mod,
        "remote_digests",
        lambda *args: {name: digest(study / name) for name in args[2]},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "p",
            str(study / "scripts" / "entry.py"),
            "--local-root",
            str(study),
            "--allow-diff",
            "scripts/callee.py=no longer differs",
        ],
    )
    assert main() == 0
    assert "did not differ" in capsys.readouterr().out


def test_an_unparseable_file_refuses_rather_than_passing(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION: swallowing SyntaxError and checking the rest.

    A gate that cannot measure must not report that it did.
    """
    write(study / "scripts" / "callee.py", "def (:\n")
    monkeypatch.setattr(
        "sys.argv",
        ["p", str(study / "scripts" / "entry.py"), "--local-root", str(study)],
    )
    assert main() == 2


def test_manifest_round_trip_passes(
    study: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MUTATION: recording THIS clone's digests instead of the box's.

    The check runs on the box, where a declared exception legitimately differs.
    Recording local digests makes a correctly-declared difference refuse every
    launch -- and a gate that always refuses is one that gets switched off.
    """
    remote = {
        "scripts/entry.py": digest(study / "scripts" / "entry.py"),
        "scripts/callee.py": "0" * 32,  # the declared exception
    }
    monkeypatch.setattr(mod, "remote_digests", lambda *args: remote)
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "p",
            str(study / "scripts" / "entry.py"),
            "--local-root",
            str(study),
            "--write-manifest",
            str(manifest),
            "--allow-diff",
            "scripts/callee.py=declared",
        ],
    )
    assert main() == 0
    recorded = json.loads(manifest.read_text())
    assert recorded["digests"]["scripts/callee.py"] == "0" * 32


def test_manifest_check_refuses_a_file_changed_since_arming(
    study: Path, tmp_path: Path
) -> None:
    """MUTATION: comparing the manifest against itself rather than the disk.

    This is the leg that would have made the 5h40m wait safe.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "digests": {"scripts/callee.py": digest(study / "scripts/callee.py")},
                "allowances": {},
            }
        )
    )
    assert mod.check_manifest(manifest, study) == 0
    (study / "scripts" / "callee.py").write_text("VALUE = 2\n")
    assert mod.check_manifest(manifest, study) == 1


def test_an_empty_or_unreadable_manifest_refuses(tmp_path: Path, study: Path) -> None:
    """MUTATION: treating a missing or empty manifest as a pass.

    A check that cannot run must not report that it did.
    """
    assert mod.check_manifest(tmp_path / "nope.json", study) == 2
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"digests": {}}))
    assert mod.check_manifest(empty, study) == 2


def test_also_covers_a_non_python_file(study: Path) -> None:
    """MUTATION: dropping --also from the covered set.

    The closure walks python imports, so the shell supervisor that launches the
    job is invisible to it -- and that supervisor is itself hand-copied.
    """
    supervisor = write(study / "scripts" / "supervise.sh", "#!/bin/bash\necho hi\n")
    covered = mod.covered_files(
        closure(study / "scripts" / "entry.py", [study / "scripts"])[0],
        study,
        ["scripts/supervise.sh"],
    )
    assert covered is not None
    assert "scripts/supervise.sh" in covered
    assert supervisor.is_file()


def test_also_naming_a_missing_file_refuses(study: Path) -> None:
    """MUTATION: silently skipping an --also entry that does not exist.

    A gate told to cover something it cannot find must not cover less.
    """
    assert mod.covered_files(set(), study, ["scripts/not_here.sh"]) is None


def test_a_first_party_import_off_the_search_path_is_refused(
    study: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: reporting OK on a closure that silently shrank.

    ⛔ THE GATE'S OWN FAILURE MODE, found by testing it on 2026-09-18. Run
    without the right --search root, the real closure fell from 77 files to 18
    and still said OK -- and the 59 it dropped included the module whose
    staleness cost ~40 GPU-hours. Checking a quarter of what you were asked to
    and passing is worse than not checking at all.
    """
    write(study / "elsewhere" / "helper.py", "VALUE = 3\n")
    write(
        study / "scripts" / "entry.py",
        """\
        def run():
            from helper import VALUE
            return VALUE
        """,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["p", str(study / "scripts" / "entry.py"), "--local-root", str(study)],
    )
    assert main() == 2
    assert "add --search elsewhere" in capsys.readouterr().err


def test_a_class_imported_from_a_module_is_not_a_missing_module(
    study: Path,
) -> None:
    """MUTATION: counting `from mod import ClassName` as an unresolved module.

    A class never resolves to a file. Treating that as a missing search root
    produced two false refusals the first time this check was written, and a
    gate that cries wolf gets bypassed.
    """
    write(study / "scripts" / "callee.py", "class Thing:\n    pass\n")
    write(
        study / "scripts" / "entry.py",
        """\
        from callee import Thing

        def run():
            return Thing
        """,
    )
    _, unresolved = mod.closure(study / "scripts" / "entry.py", [study / "scripts"])
    assert mod.unreachable_first_party(unresolved, study) == {}


@pytest.mark.parametrize("installed", [True, False])
def test_an_installed_package_sharing_a_repo_file_name_is_not_stranded(study: Path, installed: bool) -> None:
    """MUTATION: refusing a third-party import because a repository file has its name.

    service-c has `service-c/generate/llama_cpp.py` and imports the installed
    `llama_cpp` package bare; the stranded check refused v3's submission on it
    (2026-10-17). Installed in the project's .venv, it is third-party; not
    installed, the refusal stands.
    """
    write(study / "src" / "pkg" / "llama_cpp.py", "import llama_cpp\n")
    write(study / "scripts" / "entry.py", "def run():\n    import llama_cpp\n    return llama_cpp\n")
    if installed:
        write(study / ".venv" / "lib" / "python3.13" / "site-packages" / "llama_cpp" / "__init__.py", "")
    _, unresolved = mod.closure(study / "scripts" / "entry.py", [study / "scripts"])
    stranded = mod.unreachable_first_party(unresolved, study)
    assert ("llama_cpp" in stranded) is not installed


def test_host_and_root_come_from_the_environment_then_the_local_file_never_a_tracked_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutation caught: a tracked default host or root silently returning (the box of whoever wrote it)."""
    local = tmp_path / "dgx.local.env"
    local.write_text("AUTOCAST_DGX_HOST=filehost\nAUTOCAST_DGX_ROOT=/file/root\n")
    monkeypatch.delenv("AUTOCAST_DGX_HOST", raising=False)
    monkeypatch.delenv("AUTOCAST_DGX_ROOT", raising=False)
    assert mod.local_default("AUTOCAST_DGX_HOST", local) == "filehost"
    monkeypatch.setenv("AUTOCAST_DGX_HOST", "envhost")  # the environment wins over the file
    assert mod.local_default("AUTOCAST_DGX_HOST", local) == "envhost"
    assert mod.local_default("AUTOCAST_DGX_ROOT", tmp_path / "absent.env") is None
    monkeypatch.delenv("AUTOCAST_DGX_HOST")
    monkeypatch.setattr(mod, "LOCAL_DEFAULTS", tmp_path / "absent.env")
    monkeypatch.setattr("sys.argv", ["assert_code_parity.py", "some_entry.py"])
    assert main() == 2 and "--host and --remote-root are required" in capsys.readouterr().err
