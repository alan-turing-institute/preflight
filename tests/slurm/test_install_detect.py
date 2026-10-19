import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "detect"
spec = importlib.util.spec_from_file_location("detect", ROOT / "slurm" / "install" / "detect.py")
detect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detect)


def site_from(fixture: str | None, *, reserve_mb: int = detect.DEFAULT_RESERVE_MB) -> dict:
    """Run detect against a recorded fixture dir; None = a box with no nvidia-smi."""
    base = FIXTURES / (fixture or "gb10")

    def run(argv):
        if argv[0] == "nvidia-smi":
            return (base / "nvidia_smi.csv").read_text() if fixture else None
        return (base / "sinfo_version.txt").read_text()

    text = detect.detect(
        run=run, meminfo_path=base / "meminfo", os_release_path=base / "os_release",
        hostname="testbox", cpus=20, reserve_mb=reserve_mb,
    )
    return tomllib.loads(text)


def test_gb10_is_unified_with_one_gpu():
    site = site_from("gb10")
    assert site["unified_memory"] is True
    (gpu,) = site["nodes"][0]["gpus"]
    assert gpu["device"] == "/dev/nvidia0" and gpu["type"] == "nvidia-gb10"
    assert gpu["uuid"].startswith("GPU-")
    assert site["nodes"][0]["real_memory_mb"] == 127600752 // 1024 - 12288
    assert site["metrics"]["mode"] == "off" and site["portal"]["mode"] == "off"
    assert "lease" not in site


def test_h100x8_is_discrete_with_eight_gpus():
    site = site_from("h100x8", reserve_mb=1000)
    gpus = site["nodes"][0]["gpus"]
    assert site["unified_memory"] is False
    assert [g["index"] for g in gpus] == list(range(8))
    assert gpus[0]["type"] == "nvidia-h100-80gb-hbm3"
    assert site["nodes"][0]["real_memory_mb"] == 2113474560 // 1024 - 1000


def test_no_nvidia_smi_gives_empty_gpu_list():
    site = site_from(None)
    assert "gpus" not in site["nodes"][0]
    assert site["unified_memory"] is False
