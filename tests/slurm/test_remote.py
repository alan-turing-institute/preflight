"""remote.py: what `submit.py --site` ships, checks and runs, with ssh and rsync replaced by a recording fake."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import remote
import site_config

REMOTE = site_config.Remote("box", "sync")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("X = 1\n")
    (tmp_path / "run.py").write_text("import lib\n")
    (tmp_path / "data" / "set").mkdir(parents=True)
    return tmp_path


def arguments(repo: Path, **given: object) -> argparse.Namespace:
    base = {
        "project": "p", "what": "w", "job_class": "restart", "name": "n", "wait_limit": "1h", "level": "normal",
        "max_jobs": 1, "mem": "8G", "gpus": 1, "cpus": None, "after": [], "resume_ok": False, "entry": "run.py",
        "repo": repo, "search": ["src"], "data": ["data/set"], "workdir": repo, "command": ["python", "run.py", "a b"],
        "dry_run": False, "site": "turing", "service": False, "anywhere": False, "suspend": None, "no_suspend": False,
        "log": None,
    }
    return argparse.Namespace(**{**base, **given})


class Box:
    """Stands in for subprocess.run: records calls; answers md5sum with the real digests unless told to differ."""

    def __init__(self, repo: Path, *, stale: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.repo, self.stale = repo, stale

    def __call__(self, argv: list[str], **_kw: object) -> SimpleNamespace:
        self.calls.append(argv)
        out = ""
        if argv[0] == "rsync":
            out = ">f.st...... run.py\n"
        elif argv[0] == "ssh" and "md5sum" in argv[-1]:
            names = argv[-1].split("md5sum ")[1].replace("'", "").split()
            out = "".join(
                f"{'0' * 32 if self.stale else hashlib.md5((self.repo / n).read_bytes()).hexdigest()}  {n}\n" for n in names
            )
        elif argv[0] == "ssh" and "gpujobs-submit" in argv[-1]:
            out = "[submit] job 7 (normal); log /var/lib/gpujobs/logs/n.log\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_sync_then_submit_in_the_same_place_with_the_same_arguments(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    result = remote.submit(arguments(repo), SimpleNamespace(remotes={"turing": REMOTE}))
    assert result.startswith("[submit] job 7")
    rsync = next(c for c in box.calls if c[0] == "rsync")
    assert f"{repo}/./run.py" in rsync and f"{repo}/./src/lib.py" in rsync  # the import closure, deferred module included
    assert f"{repo}/./data/set/" in rsync and "--checksum" in rsync and rsync[-1] == "box:sync/p/"
    order = [c[0] if c[0] == "rsync" else c[-1].split()[0] for c in box.calls]
    assert order.index("rsync") < order.index("cd") and "md5sum" in " ".join(c[-1] for c in box.calls)  # verified before submitting
    submitted = box.calls[-1][-1]
    assert submitted.startswith("cd sync/p/. && gpujobs-submit --project p --what w --class restart --name n")
    assert "--gpus 1" in submitted and "--mem 8G" in submitted and submitted.endswith("-- python run.py 'a b'")


def test_a_mismatch_after_the_copy_queues_nothing(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(repo, stale=True)
    monkeypatch.setattr(remote.subprocess, "run", box)
    with pytest.raises(remote.RemoteRefused, match="still differs"):
        remote.submit(arguments(repo), SimpleNamespace(remotes={"turing": REMOTE}))
    assert not any("gpujobs-submit" in c[-1] for c in box.calls if c[0] == "ssh")


def test_dry_run_copies_nothing_and_what_cannot_be_shipped_is_refused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    site = SimpleNamespace(remotes={"turing": REMOTE})
    assert "nothing copied" in remote.submit(arguments(repo, dry_run=True), site) and box.calls == []
    for bad, message in (
        ({"data": [".."]}, "expected a folder inside"),
        ({"service": True}, "belongs to the site that owns the queue"),
        ({"entry": None}, "needs --repo and --entry"),
        ({"site": "nowhere"}, "names no such remote"),
    ):
        with pytest.raises(remote.RemoteRefused, match=message):
            remote.submit(arguments(repo, **bad), site)
    (repo / "run.py").write_text("import lib\nimport ghost_in_repo\n")
    (repo / "elsewhere").mkdir()
    (repo / "elsewhere" / "ghost_in_repo.py").write_text("")
    with pytest.raises(remote.RemoteRefused, match="closure could not be built"):  # a repo import on no search root
        remote.submit(arguments(repo), site)
