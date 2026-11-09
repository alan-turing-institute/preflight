#!/usr/bin/env python3
"""Build SchedMD's slurm packages (debs or rpms) from a checksummed release tarball, once per version and machine.

The installer's `[slurm]` table names a version and its sha256; `build` returns the package files, building only when
the cache lacks them. The build runs as root (it installs build dependencies) and writes only under the cache root.

    python3 slurm_build.py --version 26.05.5 --sha256 <hex> [--cache DIR]
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import platform
import re
import shutil
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
#: The cache is valid only once a build finished: the file holds the sha256 of the tarball it was built from.
MARKER = ".built-sha256"
#: Held while one version is checked and built: a second build of it (two installs sharing a cache) waits, then reuses it
LOCK = ".lock"
DOWNLOAD_TIMEOUT_S = 60
#: The packages `upgrade-slurm`, `apply` and the restore install (names as SchedMD packages them): the daemons, the
#: client commands and the base package. The rest of what the build leaves (-dev, -doc, -dbgsym, sview, ...) is not.
DEB_INSTALL = ["slurm-smd", "slurm-smd-client", "slurm-smd-slurmctld", "slurm-smd-slurmd", "slurm-smd-slurmdbd"]
RPM_INSTALL = ["slurm", "slurm-perlapi", "slurm-slurmctld", "slurm-slurmd", "slurm-slurmdbd"]
RPM_TOOLS = ["dnf", "install", "-y", "rpm-build", "dnf-plugins-core", "tar", "bzip2"]
#: SchedMD's spec builds cgroupv2 only when asked: without it slurmd cannot start on a cgroup v2 system (Fedora is one:
#: "cannot create cgroup context for cgroup/v2", found joining the laptop). `dnf builddep` of the spec as it stands does
#: not list its development packages. json-c gives `scontrol --json` and `squeue --json` (the health check and our tools
#: read them); yaml, curl, http-parser and lz4 match the Debian build. NOT hwloc: on the laptop's hybrid Intel CPU
#: (6 performance cores of 2 threads, 10 efficiency cores of 1) slurm with hwloc sees only 12 of its 22 CPUs and the
#: controller drains the node "Low CPUs"; without it slurm describes all 22.
RPM_WITH = ["--with", "cgroupv2", "--with", "yaml", "--with", "libcurl"]
RPM_EXTRA_DEVEL = ["dnf", "install", "-y", "dbus-devel", "kernel-headers", "libbpf-devel", "json-c-devel",
                   "libyaml-devel", "libcurl-devel", "http-parser-devel", "lz4-devel"]
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


def _os_fields(os_release: Path) -> dict[str, str]:
    text = os_release.read_text(encoding="utf-8", errors="replace") if os_release.exists() else ""
    return dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text, re.MULTILINE))


def os_key(os_release: Path = OS_RELEASE) -> str:
    """`ID` and `VERSION_ID` from os-release, e.g. `fedora45`: a build links the libraries of the release it was made on,
    and an rpm carries that release in its name (`.fc44`), so a build is reused only on the same release (a Fedora 44
    build reinstalled on Fedora 45 would be the old one)."""
    fields = _os_fields(os_release)
    return f"{fields.get('ID', 'unknown')}{fields.get('VERSION_ID', '')}"


def family(os_release: Path = OS_RELEASE) -> Family:
    """Debs on Debian and its derivatives, rpms on Fedora and RHEL ones, by `ID` and `ID_LIKE` in os-release."""
    fields = _os_fields(os_release)
    ids = f"{fields.get('ID', '')} {fields.get('ID_LIKE', '')}".split()
    if "debian" in ids:
        return Family.DEB
    if "fedora" in ids or "rhel" in ids:
        return Family.RPM
    raise SlurmBuildError(f"cannot build slurm packages on {fields.get('PRETTY_NAME') or 'an unknown operating system'}: "
                          "only Debian-like and Fedora/RHEL-like systems are supported")


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as src, dest.open("wb") as out:
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


def package_name(path: Path) -> str:
    """`slurm-smd-slurmd_26.05.5_arm64.deb` -> `slurm-smd-slurmd`; `slurm-slurmd-26.05.5-1.fc44.x86_64.rpm` -> `slurm-slurmd`."""
    return re.split(r"_|-\d", path.name, maxsplit=1)[0]


def installable(packages: list[Path]) -> list[Path]:
    """Only the packages the installer puts on a machine (`DEB_INSTALL` or `RPM_INSTALL`), in that order."""
    wanted = DEB_INSTALL if any(p.suffix == ".deb" for p in packages) else RPM_INSTALL
    by_name = {package_name(p): p for p in packages}
    return [by_name[name] for name in wanted if name in by_name]


# mk-build-deps installs with apt-get and no -y by default, which would wait at a prompt (its man page, noble).
MK_BUILD_DEPS_TOOL = "apt-get -y -o Debug::pkgProblemResolver=yes --no-install-recommends"


def commands(fam: Family, version: str, tarball: Path, source: Path) -> list[tuple[list[str], Path]]:
    """The build's commands with the directory each runs in (`source` is where the tarball unpacks to)."""
    workdir = source.parent
    if fam is Family.DEB:
        return [(DEB_TOOLS, workdir),
                (["tar", "xjf", str(tarball)], workdir),
                (["mk-build-deps", "-i", "-r", "-t", MK_BUILD_DEPS_TOOL, "debian/control"], source),
                (["debuild", "-b", "-uc", "-us"], source)]  # the debs land in the parent directory, `workdir`
    spec = f"slurm-{version}/slurm.spec"
    return [(RPM_TOOLS, workdir),
            (["tar", "xjf", str(tarball), "-C", str(workdir), spec], workdir),
            (["dnf", "builddep", "-y", str(workdir / spec)], workdir),
            (RPM_EXTRA_DEVEL, workdir),
            (["rpmbuild", "-ta", str(tarball), *RPM_WITH, "--define", f"_topdir {workdir / 'rpmbuild'}"], workdir)]


def build_dir(os_release: Path = OS_RELEASE) -> str:
    """The folder of a version's cache that holds the build for this machine: `rpm-fedora45-x86_64`."""
    return f"{family(os_release)}-{os_key(os_release)}-{platform.machine()}"


def build(version: str, sha256: str, cache_root: Path = DEFAULT_CACHE, *, download: Download = _download,
          run: Run = _run, os_release: Path = OS_RELEASE) -> list[Path]:
    """The package files for `version`, built from the tarball whose sha256 is `sha256` unless already built (a finished
    build leaves `MARKER` with that sha256; without it the directory is cleared and built again)."""
    url = tarball_url(version)
    fam = family(os_release)
    base = cache_root / version
    workdir = base / build_dir(os_release)
    marker = workdir / MARKER
    base.mkdir(parents=True, exist_ok=True)
    with (base / LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == sha256 and (found := _packages(workdir, fam)):
            return found
        if workdir.exists():  # a build that did not finish, or one of another tarball: its files are not to be installed
            shutil.rmtree(workdir)
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
        marker.write_text(f"{sha256}\n", encoding="utf-8")
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
