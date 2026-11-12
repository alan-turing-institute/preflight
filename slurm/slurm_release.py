#!/usr/bin/env python3
"""SchedMD's newer slurm releases for this site, proved in the container before a maintenance window installs one (D43).

    python3 slurm/slurm_release.py check            # nightly, as [slurm_release] test_user: find a newer release, submit its test
    python3 slurm/slurm_release.py status --json    # anyone, no network: what runs, what waits, whether it passed
    python3 slurm/slurm_release.py test --from R --to T --sha256 S             # the test job itself (check submits it)
    sudo python3 slurm/slurm_release.py apply --site SITE --version T --sha256 S   # the window, once running jobs ended

Nothing here changes slurm outside `apply`, and `apply` installs only a release whose test passed from the release
running now. Exit codes of `apply` are `install.py upgrade-slurm`'s (`site_config.UpgradeExit`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
import site_config  # noqa: E402
from site_config import RELEASE_SHA256, RELEASE_VERSION, Follow, UpgradeExit  # noqa: E402

API = "https://api.github.com/repos/SchedMD/slurm/releases?per_page=100"
FETCH_TIMEOUT_S = 30
#: The one asset of a SchedMD release; a candidate's name (`26.11.0-0rc1`) does not match.
ASSET = re.compile(r"slurm-(\d+\.\d+\.\d+)\.tar\.bz2")
DIGEST = re.compile(r"sha256:([0-9a-f]{64})")
#: SchedMD: "Slurm 24.11 introduced compatibility with the previous three major releases" (upgrades.html).
SERIES_WINDOW = 3
#: A test that could not run (image build, compose) is tried again on later nights, this many times for one release.
MAX_ATTEMPTS = 3
TEST_NAME = "slurm-release-test"
#: The test cluster (tests/integration): its compose file, and one image per slurm release (`GPUJOBS_TEST_IMAGE`).
COMPOSE = ("tests", "integration", "compose.yml")
IMAGE_REPO = "gpujobs-testcluster"
IMAGE_TAG = "slurm-"
GIB = 2**30
WATCH = "watch.json"
ATTEMPTS = "attempts.json"
SUMMARY = re.compile(r"^(\d+) passed, (\d+) failed of (\d+) checks", re.MULTILINE)
FAILED = re.compile(r"^  FAILED: (.+)$", re.MULTILINE)
PREFIX = "[release]"

Version = tuple[int, int, int]
Run = Callable[..., subprocess.CompletedProcess]


def version_of(text: str) -> Version:
    major, minor, patch = (int(part) for part in text.split("."))
    return major, minor, patch


def text_of(version: Version) -> str:
    """SchedMD's spelling: the series' month has two digits (26.05.5, never 26.5.5)."""
    major, minor, patch = version
    return f"{major}.{minor:02d}.{patch}"


@dataclass(frozen=True, order=True)
class Release:
    version: Version
    sha256: str

    @property
    def series(self) -> tuple[int, int]:
        return self.version[:2]


def releases(raw: object) -> list[Release]:
    """The stable releases in GitHub's release list: candidates, and anything without a sha256 digest, left out."""
    found = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict) or entry.get("prerelease"):
            continue
        for asset in entry.get("assets") or []:
            name, digest = ASSET.fullmatch(str(asset.get("name"))), DIGEST.fullmatch(str(asset.get("digest")))
            if name and digest:
                found.append(Release(version_of(name[1]), digest[1]))
    return found


def target(found: Sequence[Release], running: Version, follow: Follow) -> Release | None:
    """The newest release of the running series, or with `series` the newest of the next series once that series has
    its first maintenance release: one series at a time, well inside slurm's window."""
    newer = [r for r in found if r.version > running]
    best = max((r for r in newer if r.series == running[:2]), default=None)
    later = sorted({r.series for r in newer if r.series > running[:2]})  # ordered from the list, not by arithmetic
    if follow is Follow.SERIES and later:
        next_series = [r for r in newer if r.series == later[0]]
        if any(r.version[2] >= 1 for r in next_series):
            best = max(next_series)
    return best


@dataclass(frozen=True)
class Record:
    """One release's test, written by the test job: from the release running then, to the one tested."""

    source: str
    to: str
    sha256: str
    passed: bool
    at: int
    log: str
    tree: str = ""  # the checkout's commit, for the log only: what is qualified is the slurm release
    failed: tuple[str, ...] = ()

    @classmethod
    def read(cls, folder: Path, to: str, owner: str) -> Record | None:
        """Trusted only from `owner`'s regular file, read through one open that follows no link (`read_json`)."""
        raw = gpu_jobs.read_json(folder / f"{to}.json", owner) if RELEASE_VERSION.fullmatch(to) else None
        try:
            record = cls(str(raw["from"]), str(raw["to"]), str(raw["sha256"]), raw["passed"] is True, int(raw["at"]),
                         str(raw["log"]), str(raw.get("tree", "")), tuple(map(str, raw.get("failed", ()))))
        except (TypeError, KeyError, ValueError):
            return None
        return record if record.to == to else None

    def write(self, folder: Path) -> None:
        payload = {k: v for k, v in dataclasses.asdict(self).items() if k != "source"} | {"from": self.source}
        gpu_jobs.write_json(folder / f"{self.to}.json", payload)


def installed(run: Run) -> Version:
    done = run(["sinfo", "-V"], capture_output=True, text=True, check=False)
    found = RELEASE_VERSION.search(done.stdout or "")
    if done.returncode != 0 or not found:
        raise SystemExit(f"{PREFIX} cannot read the running slurm from `sinfo -V`: {(done.stdout or done.stderr).strip()}")
    return version_of(found[0])


def node_versions(run: Run) -> dict[str, str]:
    """Each node's slurmd release (`scontrol show node`'s Version); "" for a node that does not say (down)."""
    done = run(["scontrol", "show", "node", "-o"], capture_output=True, text=True, check=False)
    return {m[1]: m[2] if RELEASE_VERSION.fullmatch(m[2]) else ""
            for m in re.finditer(r"NodeName=(\S+).*?\bVersion=(\S*)", done.stdout or "")}


def fetch_api(url: str = API) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "gpujobs"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310 - a fixed https URL
        return json.loads(response.read())


class Watch:
    """The site's `[slurm_release]` folder: the last check, the attempts, and the records."""

    def __init__(self, cfg: site_config.SlurmRelease) -> None:
        self.cfg, self.folder = cfg, Path(cfg.records)

    def last(self) -> dict[str, object]:
        return gpu_jobs.read_json(self.folder / WATCH, self.cfg.test_user) or {}

    def attempts(self) -> dict[str, int]:
        raw = gpu_jobs.read_json(self.folder / ATTEMPTS, self.cfg.test_user) or {}
        return {k: v for k, v in raw.items() if type(v) is int}

    def record(self, to: str) -> Record | None:
        return Record.read(self.folder, to, self.cfg.test_user)

    def passed_from(self, running: Version) -> Record | None:
        """The newest release whose test passed from `running` (with several passing records, the newest wins)."""
        found = (self.record(path.stem) for path in self.folder.glob("*.json") if RELEASE_VERSION.fullmatch(path.stem))
        return max((r for r in found if r and r.passed and r.source == text_of(running) and version_of(r.to) > running),
                   key=lambda r: version_of(r.to), default=None)

    def status(self, running: Version, nodes: dict[str, str]) -> dict[str, object]:
        """What `status --json` prints: `waiting` (with `ready`, the version and sha256 `apply` takes) only for a release
        whose test passed from the release running now and that leaves every node's slurmd inside slurm's window."""
        last = self.last()
        to, series = str(last.get("target") or ""), [tuple(s) for s in last.get("series") or [] if isinstance(s, list)]
        record = self.record(to) if to else None
        out: dict[str, object] = {"installed": text_of(running), "target": to or None, "checked_at": last.get("checked_at"),
                                  "tested": None if record is None else {"passed": record.passed, "at": record.at, "log": record.log,
                                                                         "from": record.source, "failed": list(record.failed)},
                                  "nodes": [{"node": n, "version": v or None} for n, v in sorted(nodes.items())],
                                  "waiting": False, "ready": None, "summary": ""}
        best = self.passed_from(running)
        if best is None:
            out["summary"] = f"slurm {text_of(running)}: " + (f"{to} {'failed its test' if record else 'not tested yet'}" if to
                                                                else "nothing newer to install")
            return out
        far = [n for n, v in nodes.items() if not v or _gap(series, version_of(v), version_of(best.to)) >= SERIES_WINDOW]
        if far:
            out["summary"] = f"slurm {best.to} passed, but would leave {', '.join(sorted(far))} outside slurm's window: upgrade those first"
            return out
        out |= {"waiting": True, "ready": {"version": best.to, "sha256": best.sha256},
                "summary": f"slurm {best.to} (now {text_of(running)}), container-tested {time.strftime('%Y-%m-%d', time.localtime(best.at))}"}
        return out


def _gap(series: Sequence[tuple[int, ...]], older: Version, newer: Version) -> int:
    """How many series `newer` is ahead of `older`, counted in the release list (an unknown series is far)."""
    try:
        return series.index(newer[:2]) - series.index(older[:2])
    except ValueError:
        return SERIES_WINDOW


def cluster_size(checkout: Path, run: Run) -> tuple[int, int]:
    """(cores, whole GiB) the test cluster's containers may use, summed over compose.yml's services. They run outside the
    test job's cgroup, so the job asks slurm for exactly this much; raises ValueError when compose cannot say."""
    done = run(["docker", "compose", "-f", str(checkout.joinpath(*COMPOSE)), "config", "--format", "json"],
               capture_output=True, text=True, check=False)
    try:
        services = list(json.loads(done.stdout or "null")["services"].values())
        return (math.ceil(sum(float(s["cpus"]) for s in services)),
                math.ceil(sum(int(s["mem_limit"]) for s in services) / GIB))
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"`docker compose config` gave no cpus and mem_limit per service: {(done.stderr or '').strip()[-200:]}") from exc


def check(site: site_config.Site, *, run: Run = subprocess.run, fetch: Callable[[], object] = fetch_api,
          now: Callable[[], float] = time.time) -> int:
    cfg = site.slurm_release
    if cfg is None:
        print(f"{PREFIX} the site file has no [slurm_release] table: nothing to watch")
        return 0
    watch, running = Watch(cfg), installed(run)
    try:
        found = releases(fetch())
    except (OSError, ValueError) as exc:  # a failed fetch is never "nothing newer"
        print(f"{PREFIX} watch unavailable ({exc}); the last status stands")
        return 0
    chosen = target(found, running, cfg.follow)
    gpu_jobs.write_json(watch.folder / WATCH, {
        "checked_at": int(now()), "installed": text_of(running), "target": text_of(chosen.version) if chosen else None,
        "sha256": chosen.sha256 if chosen else None, "series": [list(s) for s in sorted({r.series for r in found})]})
    if chosen is None:
        print(f"{PREFIX} slurm {text_of(running)} is the newest this site follows ({cfg.follow})")
        return 0
    to = text_of(chosen.version)
    if (record := watch.record(to)) is not None:
        print(f"{PREFIX} slurm {to} was tested from {record.source}: {'passed' if record.passed else 'FAILED'} ({record.log})")
        return 0
    queued = run(["squeue", "-h", "-u", cfg.test_user, "-o", "%j"], capture_output=True, text=True, check=False)
    if any(line.endswith(f":{TEST_NAME}") for line in (queued.stdout or "").split()):
        print(f"{PREFIX} slurm {to}: its test is already in the queue")
        return 0
    if (tries := watch.attempts().get(to, 0)) >= MAX_ATTEMPTS:
        print(f"{PREFIX} slurm {to}: the test could not run {tries} times; look at the logs in {watch.folder}")
        return 0
    checkout = Path(cfg.checkout)
    try:
        cores, gib = cluster_size(checkout, run)
    except ValueError as exc:
        print(f"{PREFIX} slurm {to}: cannot size its test ({exc}); nothing submitted")
        return 1
    argv = [sys.executable, str(checkout / "slurm" / "submit.py"), "--project", cfg.project, "--class", "resume",
            "--resume-ok", "--what", f"container test of slurm {to}", "--name", TEST_NAME, "--cpus", str(cores),
            "--mem", f"{gib}G", "--no-check", "the release test job is itself the check", "--",
            sys.executable, str(checkout / "slurm" / "slurm_release.py"), "test",
            "--from", text_of(running), "--to", to, "--sha256", chosen.sha256]
    done = run(argv, capture_output=True, text=True, check=False)
    print(f"{PREFIX} slurm {to} is newer than {text_of(running)}: test submitted ({(done.stdout or done.stderr).strip()[-200:]})")
    return 0 if done.returncode == 0 else 1


def test(site: site_config.Site, source: str, to: str, sha256: str, *, run: Run = subprocess.run) -> int:
    """Two container runs: our tools on `to`, then an upgrade from `source` (the site's pin) to `to`. A run that ends
    without its summary line is a harness error: no record, tried again another night."""
    cfg = site.slurm_release
    if cfg is None or site.slurm_build is None or site.slurm_build[0] != source:
        print(f"{PREFIX} test needs [slurm_release] and the site's [slurm] pin at {source}", file=sys.stderr)
        return 1
    watch, here = Watch(cfg), Path(cfg.checkout) / "tests" / "integration"
    log = watch.folder / f"{to}.log"
    env = {**os.environ, "GPUJOBS_TEST_PROJECT": f"slurm-release-{to.replace('.', '-')}"}
    stages = run([sys.executable, str(here / "e2e.py"), "--list"], capture_output=True, text=True, check=False).stdout.split()
    runs = [({"SLURM_VERSION": to, "SLURM_SHA256": sha256, "GPUJOBS_TEST_IMAGE": image(to)},
             ",".join(s for s in stages if s != "upgrade")),
            ({"SLURM_VERSION": source, "SLURM_SHA256": site.slurm_build[1], "GPUJOBS_TEST_IMAGE": image(source),
              "UPGRADE_TO": f"{to}:{sha256}"}, "install,upgrade")]

    def requeued(*_: object) -> None:  # a requeue ends a resume job with TERM: the finally below takes the cluster down
        raise SystemExit(f"{PREFIX} stopped (TERM): the test cluster is taken down and the test runs again from the start")

    signal.signal(signal.SIGTERM, requeued)
    outputs = []
    try:
        for extra, only in runs:
            done = run([str(here / "run.sh"), "e2e", "--only", only], env={**env, **extra}, capture_output=True, text=True, check=False)
            outputs.append(done.stdout + done.stderr)
    finally:
        run([str(here / "run.sh"), "down"], env=env, capture_output=True, text=True, check=False)
        log.write_text("\n".join(outputs), encoding="utf-8")
        prune_images(run, keep={source, to})
    summaries = [SUMMARY.search(text) for text in outputs]
    if len(outputs) != len(runs) or not all(summaries):
        tries = watch.attempts()
        gpu_jobs.write_json(watch.folder / ATTEMPTS, {**tries, to: tries.get(to, 0) + 1})
        print(f"{PREFIX} the container test of {to} could not run (no summary; see {log}); tried again another night")
        return 1
    failed = tuple(name for text in outputs for name in FAILED.findall(text))
    tree = run(["git", "-C", cfg.checkout, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    passed = not failed and all(int(m[2]) == 0 for m in summaries if m)
    Record(source, to, sha256, passed, int(time.time()), str(log), tree, failed).write(watch.folder)
    print(f"{PREFIX} slurm {to} from {source}: {'passed' if passed else f'FAILED {len(failed)} check(s)'} ({log})")
    return 0 if passed else 1


def image(version: str) -> str:
    return f"{IMAGE_REPO}:{IMAGE_TAG}{version}"


def prune_images(run: Run, keep: set[str]) -> None:
    """Remove the test cluster's images of slurm releases other than `keep` (an image in use stays: docker refuses)."""
    done = run(["docker", "images", IMAGE_REPO, "--format", "{{.Tag}}"], capture_output=True, text=True, check=False)
    old = [tag for tag in (done.stdout or "").split() if tag.startswith(IMAGE_TAG)
           and RELEASE_VERSION.fullmatch(tag.removeprefix(IMAGE_TAG)) and tag.removeprefix(IMAGE_TAG) not in keep]
    if old:
        run(["docker", "image", "rm", *(f"{IMAGE_REPO}:{tag}" for tag in old)], capture_output=True, text=True, check=False)


def _replace_keeping_owner(path: Path, text: str) -> None:
    status = path.stat()
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chown(temp, status.st_uid, status.st_gid)
    os.chmod(temp, status.st_mode & 0o7777)
    Path(temp).replace(path)


def apply(site_path: Path, version: str, sha256: str, *, run: Run = subprocess.run) -> int:
    """Install `version` with `install.py upgrade-slurm`, only from a trusted passing test of it from the release
    running now. The site file's `[slurm]` pin is set first (kept on success, put back when the upgrade refused)."""
    if not (RELEASE_VERSION.fullmatch(version) and RELEASE_SHA256.fullmatch(sha256)):
        print(f"{PREFIX} REFUSED: {version!r} / {sha256!r} is not a release and its sha256", file=sys.stderr)
        return UpgradeExit.REFUSED
    site = site_config.load(site_path)
    cfg = site.slurm_release
    problem = _untrusted(site, cfg, version, sha256, installed(run))
    if problem:
        print(f"{PREFIX} REFUSED: {problem}", file=sys.stderr)
        return UpgradeExit.REFUSED
    before = site_path.read_text(encoding="utf-8")
    backup = site_path.with_name(f"{site_path.name}.before-slurm-{version}")
    backup.write_text(before, encoding="utf-8")
    _replace_keeping_owner(site_path, site_config.with_slurm_pin(before, version, sha256))
    site_config.load(site_path)  # the edited file still loads, or nothing runs
    done = run([sys.executable, str(Path(cfg.checkout) / "slurm" / "install" / "install.py"), "--site", str(site_path),
                "upgrade-slurm"], check=False)
    if done.returncode == UpgradeExit.REFUSED:
        _replace_keeping_owner(site_path, before)
        print(f"{PREFIX} upgrade-slurm refused: {site_path} is back as it was")
    print(f"{PREFIX} the site's source file needs, in its [slurm] table: version = \"{version}\", sha256 = \"{sha256}\", no steps"
          if done.returncode != UpgradeExit.REFUSED else f"{PREFIX} nothing changed")
    return done.returncode


def _untrusted(site: site_config.Site, cfg: site_config.SlurmRelease | None, version: str, sha256: str, running: Version) -> str:
    """Why `apply` must not install `version`; "" when a passing test from the running release says it may."""
    if cfg is None:
        return "the site file has no [slurm_release] table"
    if not site_config.is_admin(site, cfg.test_user):
        return f"slurm_release.test_user {cfg.test_user} is not an admin, so their records are not trusted"
    folder = Path(cfg.records)
    try:
        status = folder.stat()
    except OSError as exc:
        return f"records folder {folder}: {exc}"
    if status.st_mode & 0o022 or gpu_jobs._user_name(status.st_uid) not in (cfg.test_user, "root"):  # noqa: SLF001
        return f"records folder {folder} must belong to {cfg.test_user} or root and be writable by nobody else"
    record = Record.read(folder, version, cfg.test_user)
    if record is None:
        return f"no record of a test of {version} written by {cfg.test_user}"
    if not record.passed or record.source != text_of(running) or record.sha256 != sha256:
        return (f"the record of {version} does not say it passed from the running {text_of(running)} with that sha256 "
                f"(passed {record.passed}, from {record.source})")
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("status").add_argument("--json", action="store_true")
    tester = sub.add_parser("test")
    for flag in ("--from", "--to", "--sha256"):
        tester.add_argument(flag, required=True, dest=flag.strip("-").replace("from", "source"))
    applier = sub.add_parser("apply")
    applier.add_argument("--site", type=Path, required=True)
    applier.add_argument("--version", required=True)
    applier.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "apply":
        return apply(args.site, args.version, args.sha256)
    site = gpu_jobs.load_site()
    if args.command == "check":
        return check(site)
    if args.command == "test":
        return test(site, args.source, args.to, args.sha256)
    if site.slurm_release is None:
        report = {"waiting": False, "summary": "no [slurm_release] table: no release watch"}
    else:
        report = Watch(site.slurm_release).status(installed(subprocess.run), node_versions(subprocess.run))
    print(json.dumps(report) if args.json else report["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
