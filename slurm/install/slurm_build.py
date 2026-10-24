#!/usr/bin/env python3
"""Build SchedMD's slurm packages (debs or rpms) from a checksummed release tarball, once per version and machine.

The installer's `[slurm]` table names a version and its sha256; `build` returns the package files, building only when
the cache lacks them. The build runs as root (it installs build dependencies) and writes only under the cache root.

    python3 slurm_build.py --version 26.05.5 --sha256 <hex> [--cache DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import re
import subprocess
import urllib.request
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

DEFAULT_CACHE = Path("/var/cache/gpujobs/slurm")
OS_RELEASE = Path("/etc/os-release")
#: SchedMD's release tag spells the version with dashes: slurm-26-05-5-1
URL = "https://github.com/SchedMD/slurm/releases/download/slurm-{dashed}-1/slurm-{version}.tar.bz2"
VERSION = re.compile(r"\d+\.\d+\.\d+")
DEB_TOOLS = ["apt-get", "install", "-y", "build-essential", "fakeroot", "devscripts", "equivs"]

Download = Callable[[str, Path], None]
Run = Callable[[list[str], Path], None]


class Family(StrEnum):
    DEB = "deb"
    RPM = "rpm"


class SlurmBuildError(Exception):
    """The build cannot go ahead; the message says why."""


def tarball_url(version: str) -> str:
    if not VERSION.fullmatch(version):
        raise SlurmBuildError(f"slurm version {version!r} is not of the form 26.05.5")
    return URL.format(dashed=version.replace(".", "-"), version=version)


def family(os_release: Path = OS_RELEASE) -> Family:
    """Debs on Debian and its derivatives, rpms on Fedora and RHEL ones, by `ID` and `ID_LIKE` in os-release."""
    text = os_release.read_text(encoding="utf-8", errors="replace") if os_release.exists() else ""
    fields = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text, re.MULTILINE))
    ids = f"{fields.get('ID', '')} {fields.get('ID_LIKE', '')}".split()
    if "debian" in ids:
        return Family.DEB
    if "fedora" in ids or "rhel" in ids:
        return Family.RPM
    raise SlurmBuildError(f"cannot build slurm packages on {fields.get('PRETTY_NAME') or 'an unknown operating system'}: "
                          "only Debian-like and Fedora/RHEL-like systems are supported")


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as src, dest.open("wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)


def _run(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _packages(workdir: Path, fam: Family) -> list[Path]:
    pattern = ("*.deb", "rpmbuild/RPMS/*/*.rpm")[fam is Family.RPM]
    return sorted(workdir.glob(pattern))


def commands(fam: Family, version: str, tarball: Path, source: Path) -> list[tuple[list[str], Path]]:
    """The build's commands with the directory each runs in (`source` is where the tarball unpacks to)."""
    workdir = source.parent
    if fam is Family.DEB:
        return [(DEB_TOOLS, workdir),
                (["tar", "xjf", str(tarball)], workdir),
                (["mk-build-deps", "-i", "debian/control"], source),
                (["debuild", "-b", "-uc", "-us"], source)]  # the debs land in the parent directory, `workdir`
    return [(["rpmbuild", "-ta", str(tarball), "--define", f"_topdir {workdir / 'rpmbuild'}"], workdir)]


def build(version: str, sha256: str, cache_root: Path = DEFAULT_CACHE, *, download: Download = _download,
          run: Run = _run, os_release: Path = OS_RELEASE) -> list[Path]:
    """The package files for `version`, built from the tarball whose sha256 is `sha256` unless already built."""
    url = tarball_url(version)
    fam = family(os_release)
    base = cache_root / version
    workdir = base / f"{fam}-{platform.machine()}"
    if found := _packages(workdir, fam):
        return found
    base.mkdir(parents=True, exist_ok=True)
    tarball = base / Path(url).name
    if not tarball.exists():
        download(url, tarball)
    if (got := _sha256(tarball)) != sha256:
        tarball.unlink()
        raise SlurmBuildError(f"{url} has sha256 {got}, not the {sha256} the site file names; nothing was built")
    workdir.mkdir(parents=True, exist_ok=True)
    for argv, cwd in commands(fam, version, tarball, workdir / f"slurm-{version}"):
        run(argv, cwd)
    if not (found := _packages(workdir, fam)):
        raise SlurmBuildError(f"the build finished but left no {fam} files under {workdir}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    try:
        packages = build(args.version, args.sha256, args.cache)
    except (SlurmBuildError, subprocess.CalledProcessError, OSError) as error:
        print(f"slurm build failed: {error}")
        return 1
    print("\n".join(map(str, packages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
