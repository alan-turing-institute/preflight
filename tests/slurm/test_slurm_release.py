"""slurm_release.py: which release the watch proposes, whose records the window trusts, and what each command runs."""

from __future__ import annotations

import getpass
import json
import os
import subprocess
from pathlib import Path

import pytest
import site_config
import slurm_release as sr
from site_config import Follow, UpgradeExit

R, T, NEXT = "26.05.5", "26.05.6", "26.11.1"
SHA_R, SHA_T, SHA_N = "aa" * 32, "bb" * 32, "cc" * 32
ME = getpass.getuser()


def asset(version: str, sha: str | None) -> dict[str, object]:
    return {"name": f"slurm-{version}.tar.bz2", "digest": f"sha256:{sha}" if sha else None}


class Fake:
    """Every command the tool runs: argv recorded, answers by the command's first words."""

    def __init__(self, **answers: tuple[int, str]) -> None:
        self.answers = {"sinfo": (0, f"slurm {R}\n"), "squeue": (0, ""), "docker compose": (0, json.dumps(
            {"services": {"cluster": {"cpus": 4, "mem_limit": "8589934592"}, "part-time": {"cpus": 1.5, "mem_limit": "3221225472"}}})),
            "docker images": (0, f"dev\nslurm-25.05.9\nslurm-{R}\nslurm-{T}\nslurm-x\n"), **answers}
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append((list(argv), dict(kwargs.get("env") or {})))  # type: ignore[arg-type]
        words = " ".join(Path(a).name if i == 0 else a for i, a in enumerate(argv))
        code, out = next((v for k, v in self.answers.items() if words.startswith(k) or k in " ".join(argv)), (0, ""))
        return subprocess.CompletedProcess(argv, code, out, "")

    def ran(self, word: str) -> list[list[str]]:
        return [argv for argv, _ in self.calls if word in " ".join(argv)]


@pytest.fixture
def site(tmp_path: Path) -> Path:
    records = tmp_path / "records"
    records.mkdir(mode=0o755)
    path = tmp_path / "site.toml"
    path.write_text(f'name = "x"\n[[nodes]]\nname = "n"\ncpus = 2\nreal_memory_mb = 1000\n[slurm]\nversion = "{R}"\nsha256 = "{SHA_R}"\n'
                    f'steps = [\n  "25.05.9:{SHA_R}",\n]\n[slurm_release]\ncheckout = "{tmp_path}/co"\nrecords = "{records}"\n'
                    f'test_user = "{ME}"\n')
    return path


def record(folder: Path, to: str, *, source: str = R, sha: str = SHA_T, passed: bool = True, at: int = 1) -> None:
    sr.Record(source, to, sha, passed, at, f"{folder}/{to}.log").write(folder)


def test_the_watch_proposes_one_stable_release_at_a_time() -> None:
    raw = [{"prerelease": True, "assets": [asset("26.11.0", SHA_N)]}, {"assets": [{"name": "slurm-26.11.0-0rc1.tar.bz2", "digest": "x"}]},
           {"assets": [asset("26.11.0", SHA_N)]}, {"assets": [asset(T, SHA_T), {"name": "notes.txt"}]}, {"assets": [asset("26.05.7", None)]},
           "junk", {"assets": [asset(R, SHA_R)]}]
    found = sr.releases(raw)
    assert sorted(sr.text_of(r.version) for r in found) == [R, T, "26.11.0"]  # no candidate, no release without a digest
    assert sr.releases({"message": "rate limited"}) == [] and sr.text_of(sr.version_of(R)) == R  # SchedMD's two-digit month
    running = sr.version_of(R)
    assert sr.target(found, running, Follow.PATCH).sha256 == SHA_T
    assert sr.target(found, running, Follow.SERIES).sha256 == SHA_T  # 26.11 has only its .0: not yet
    with_point_one = [*found, sr.Release(sr.version_of(NEXT), SHA_N), sr.Release((27, 5, 1), SHA_N)]
    assert sr.text_of(sr.target(with_point_one, running, Follow.SERIES).version) == NEXT  # the next series only, never two
    assert sr.target(with_point_one, running, Follow.PATCH).sha256 == SHA_T
    assert sr.target(found, sr.version_of(T), Follow.PATCH) is None


def test_the_window_trusts_only_a_passing_record_from_the_running_release(site: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    cfg = site_config.load(site).slurm_release
    folder, running, nodes = Path(cfg.records), sr.version_of(R), {"n": R}
    watch = sr.Watch(cfg)
    sr.gpu_jobs.write_json(folder / sr.WATCH, {"target": NEXT, "series": [[25, 11], [26, 5], [26, 11]]})
    assert watch.status(running, nodes)["summary"] == f"slurm {R}: {NEXT} not tested yet"
    record(folder, NEXT, sha=SHA_N, passed=False)
    assert not watch.status(running, nodes)["waiting"] and "failed its test" in watch.status(running, nodes)["summary"]
    record(folder, T, source="26.05.4")  # passed, but from another release
    assert not watch.status(running, nodes)["waiting"]
    record(folder, T)
    status = watch.status(running, nodes)
    assert status["waiting"] and status["ready"] == {"version": T, "sha256": SHA_T}  # an older passing release still waits
    record(folder, NEXT, sha=SHA_N, at=2)
    assert watch.status(running, nodes)["ready"] == {"version": NEXT, "sha256": SHA_N}  # the newest passing record wins
    assert not watch.status(running, {"n": R, "laptop": ""})["waiting"]  # a node that does not say its slurmd blocks it
    assert "laptop outside slurm's window" in watch.status(running, {"n": R, "laptop": "24.11.1"})["summary"]
    assert sr.Record.read(folder, NEXT, "root") is None  # the owner is the identity, never the content
    os.symlink(folder / f"{NEXT}.json", folder / "26.11.2.json")
    assert sr.Record.read(folder, "26.11.2", ME) is None  # a link is never followed

    fake = Fake(**{"install.py": (UpgradeExit.REFUSED, "")})
    monkeypatch.setattr(sr.site_config, "is_admin", lambda _site, user: user == ME)
    before = site.read_text()
    assert sr.apply(site, NEXT, SHA_T, run=fake) == UpgradeExit.REFUSED and not fake.ran("install.py")  # not the tested digest
    assert sr.apply(site, "26.11", SHA_N, run=fake) == UpgradeExit.REFUSED  # not a release
    assert sr.apply(site, NEXT, SHA_N, run=fake) == UpgradeExit.REFUSED and fake.ran("install.py")  # upgrade-slurm refused
    assert site.read_text() == before and "back as it was" in capsys.readouterr().out
    assert (site.parent / f"site.toml.before-slurm-{NEXT}").read_text() == before
    fake.answers["install.py"] = (UpgradeExit.FAILED_AFTER_CHANGE, "")
    assert sr.apply(site, NEXT, SHA_N, run=fake) == UpgradeExit.FAILED_AFTER_CHANGE  # passed through: the window must not reboot
    pinned = site_config.load(site)
    assert (pinned.slurm_build, pinned.slurm_steps) == ((NEXT, SHA_N), ()) and f'version = "{NEXT}"' in capsys.readouterr().out
    site.write_text(before)
    folder.chmod(0o775)
    assert sr.apply(site, NEXT, SHA_N, run=fake) == UpgradeExit.REFUSED and "writable by nobody else" in capsys.readouterr().err
    folder.chmod(0o755)
    monkeypatch.setattr(sr.site_config, "is_admin", lambda _site, _user: False)
    assert sr.apply(site, NEXT, SHA_N, run=fake) == UpgradeExit.REFUSED and "is not an admin" in capsys.readouterr().err


def test_the_nightly_check_submits_one_test_sized_to_the_cluster(site: Path, capsys: pytest.CaptureFixture[str]) -> None:
    loaded = site_config.load(site)
    folder = Path(loaded.slurm_release.records)
    fake = Fake()

    def down() -> object:
        raise OSError("no network")

    assert sr.check(loaded, run=fake, fetch=down) == 0 and not (folder / sr.WATCH).exists()  # never read as "nothing newer"
    assert "watch unavailable" in capsys.readouterr().out
    listing = [{"assets": [asset(R, SHA_R)]}, {"assets": [asset(T, SHA_T)]}]
    assert sr.check(loaded, run=fake, fetch=lambda: listing, now=lambda: 7) == 0
    submitted = fake.ran("submit.py")
    assert len(submitted) == 1 and json.loads((folder / sr.WATCH).read_text())["target"] == T
    argv = submitted[0]
    assert argv[argv.index("--cpus") + 1] == "6" and argv[argv.index("--mem") + 1] == "11G"  # 4 + 1.5 cores, 8 + 3 GiB
    assert argv[argv.index("--class") + 1] == "resume" and argv[argv.index("--name") + 1] == sr.TEST_NAME
    assert argv[argv.index("test"):] == ["test", "--from", R, "--to", T, "--sha256", SHA_T]
    fake.answers["squeue"] = (0, f"gpujobs:resume:{sr.TEST_NAME}\n")
    assert sr.check(loaded, run=fake, fetch=lambda: listing) == 0 and len(fake.ran("submit.py")) == 1  # already queued
    fake.answers["squeue"] = (0, "")
    sr.gpu_jobs.write_json(folder / sr.ATTEMPTS, {T: sr.MAX_ATTEMPTS})
    assert sr.check(loaded, run=fake, fetch=lambda: listing) == 0 and len(fake.ran("submit.py")) == 1  # gave up on this one
    record(folder, T, passed=False)
    assert sr.check(loaded, run=fake, fetch=lambda: listing) == 0 and len(fake.ran("submit.py")) == 1  # a failure is not retried
    assert sr.check(loaded, run=Fake(**{"sinfo": (0, f"slurm {T}\n")}), fetch=lambda: listing) == 0
    assert "is the newest this site follows" in capsys.readouterr().out.splitlines()[-1]


def test_the_test_records_only_runs_that_reported_and_always_takes_the_cluster_down(site: Path) -> None:
    loaded = site_config.load(site)
    folder = Path(loaded.slurm_release.records)
    stages = "install\njobs\nupgrade\n"
    fake = Fake(**{"e2e.py": (0, stages), "run.sh e2e": (0, "3 passed, 0 failed of 3 checks\n")})
    assert sr.test(loaded, R, T, SHA_T, run=fake) == 0
    runs = [(argv, env) for argv, env in fake.calls if "e2e" in argv and "--only" in argv]
    assert [argv[-1] for argv, _ in runs] == ["install,jobs", "install,upgrade"]  # every stage but upgrade, then the upgrade
    assert runs[0][1]["GPUJOBS_TEST_IMAGE"] == sr.image(T) and runs[1][1]["GPUJOBS_TEST_IMAGE"] == sr.image(R)
    assert runs[1][1]["UPGRADE_TO"] == f"{T}:{SHA_T}" and runs[1][1]["SLURM_SHA256"] == SHA_R  # from the site's own pin
    assert fake.ran("run.sh down") and fake.ran("image rm") == [["docker", "image", "rm", "gpujobs-testcluster:slurm-25.05.9"]]
    assert sr.Record.read(folder, T, ME).passed

    failing = Fake(**{"e2e.py": (0, stages), "run.sh e2e": (1, "2 passed, 1 failed of 3 checks\n  FAILED: upgrade keeps accounting\n")})
    assert sr.test(loaded, R, T, SHA_T, run=failing) == 1
    assert sr.Record.read(folder, T, ME).failed == ("upgrade keeps accounting", "upgrade keeps accounting")

    (folder / f"{T}.json").unlink()
    broken = Fake(**{"e2e.py": (0, stages), "run.sh e2e": (1, "compose: image build failed\n")})
    assert sr.test(loaded, R, T, SHA_T, run=broken) == 1 and broken.ran("run.sh down")
    assert sr.Record.read(folder, T, ME) is None and sr.Watch(loaded.slurm_release).attempts() == {T: 1}  # tried another night
    elsewhere = Fake(**{"e2e.py": (0, stages), "run.sh e2e": (0, "3 passed, 0 failed of 3 checks\n")})
    assert sr.test(loaded, "26.05.4", T, SHA_T, run=elsewhere) == 1 and not elsewhere.calls  # only from the site's own pin
