import hashlib
import importlib.util
import platform
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("slurm_build", ROOT / "slurm" / "install" / "slurm_build.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

BODY = b"tarball"
SHA = hashlib.sha256(BODY).hexdigest()
URL = "https://github.com/SchedMD/slurm/releases/download/slurm-26-05-5-1/slurm-26.05.5.tar.bz2"


def release(tmp_path, text):
    path = tmp_path / "os-release"
    path.write_text(text)
    return path


class Fakes:
    """Stands in for the download and the commands; a command that builds leaves the package it would."""

    def __init__(self, leaves):
        self.urls, self.ran, self.leaves = [], [], leaves

    def download(self, url, dest):
        self.urls.append(url)
        dest.write_bytes(BODY)

    def run(self, argv, cwd):
        self.ran.append((argv, cwd))
        if argv[0] in ("debuild", "rpmbuild"):
            target = cwd.parent if argv[0] == "debuild" else cwd / "rpmbuild/RPMS/x86_64"
            target.mkdir(parents=True, exist_ok=True)
            (target / self.leaves).write_bytes(b"pkg")


@pytest.mark.parametrize("text, fam, leaves, install", [
    ('ID=ubuntu\nID_LIKE=debian\n', sb.Family.DEB, "slurm-smd_26.05.5_arm64.deb", sb.DEB_TOOLS),
    ('ID=debian\n', sb.Family.DEB, "slurm-smd_26.05.5_arm64.deb", sb.DEB_TOOLS),
    ('ID=fedora\n', sb.Family.RPM, "slurm-26.05.5.rpm", None),
])
def test_build_downloads_checks_builds_and_reuses_the_cache(tmp_path, text, fam, leaves, install):
    fakes, cache = Fakes(leaves), tmp_path / "cache"
    kwargs = {"download": fakes.download, "run": fakes.run, "os_release": release(tmp_path, text)}
    packages = sb.build("26.05.5", SHA, cache, **kwargs)
    assert fakes.urls == [URL]
    assert [p.name for p in packages] == [leaves]
    assert packages[0].is_relative_to(cache / "26.05.5" / f"{fam}-{platform.machine()}")
    names = [argv[0] for argv, _ in fakes.ran]
    if fam is sb.Family.DEB:
        source = cache / "26.05.5" / f"deb-{platform.machine()}" / "slurm-26.05.5"
        assert fakes.ran[0][0] == install
        assert (["mk-build-deps", "-i", "-r", "-t", sb.MK_BUILD_DEPS_TOOL, "debian/control"], source) in fakes.ran
        assert (["debuild", "-b", "-uc", "-us"], source) in fakes.ran
        assert names.index("tar") < names.index("mk-build-deps") < names.index("debuild")
    else:
        assert names == ["rpmbuild"] and fakes.ran[0][0][1] == "-ta"
    assert sb.build("26.05.5", SHA, cache, **kwargs) == packages  # the second call builds and downloads nothing
    assert len(fakes.urls) == 1 and len(fakes.ran) == len(names)


def test_refusals(tmp_path):
    fakes = Fakes("x.deb")
    kwargs = {"download": fakes.download, "run": fakes.run, "os_release": release(tmp_path, "ID=ubuntu\nID_LIKE=debian\n")}
    with pytest.raises(sb.SlurmBuildError, match="not the 00 the site file"):
        sb.build("26.05.5", "00", tmp_path / "cache", **kwargs)
    assert fakes.ran == [] and not (tmp_path / "cache/26.05.5/slurm-26.05.5.tar.bz2").exists()  # bad file dropped
    with pytest.raises(sb.SlurmBuildError, match="Arch Linux"):
        sb.family(release(tmp_path, 'ID=arch\nPRETTY_NAME="Arch Linux"\n'))
    with pytest.raises(sb.SlurmBuildError, match="not of the form"):
        sb.tarball_url("../26.05")
