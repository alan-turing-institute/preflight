import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "detect"
spec = importlib.util.spec_from_file_location("detect", ROOT / "slurm" / "install" / "detect.py")
detect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detect)


def text_from(fixture: str | None, *, reserve_mb: int = detect.DEFAULT_RESERVE_MB, cpu_root: Path = Path("/nonexistent")) -> str:
    """Run detect against a recorded fixture dir; None = a box with no nvidia-smi. `cpu_root`: no core speeds by default."""
    base = FIXTURES / (fixture or "gb10")

    def run(argv):
        if argv[0] == "nvidia-smi":
            return (base / "nvidia_smi.csv").read_text() if fixture else None
        return (base / "sinfo_version.txt").read_text()

    return detect.detect(
        run=run, meminfo_path=base / "meminfo", os_release_path=base / "os_release",
        hostname="testbox", cpus=20, reserve_mb=reserve_mb, cpu_root=cpu_root,
    )


def site_from(fixture: str | None, *, reserve_mb: int = detect.DEFAULT_RESERVE_MB) -> dict:
    return tomllib.loads(text_from(fixture, reserve_mb=reserve_mb))


def fake_cpus(root: Path, speeds: dict[str, str]) -> Path:
    """A /sys/devices/system/cpu with one cpuN/cpufreq/cpuinfo_max_freq per entry."""
    for cpu, khz in speeds.items():
        (root / f"cpu{cpu}" / "cpufreq").mkdir(parents=True)
        (root / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq").write_text(khz)
    return root


def test_cores_that_differ_are_listed_by_top_speed_and_alike_cores_are_not(tmp_path: Path):
    mixed = fake_cpus(tmp_path / "mixed", {"0": "3900000\n", "1": "2808000\n", "2": "2808000\n", "3": "garbage"})
    (mixed / "cpu4").mkdir()  # an offline core has no cpufreq; cpu3 reports nonsense: both are left out, neither fails the draft
    text = text_from("gb10", cpu_root=mixed)
    assert "# top speed 2808 MHz: 2 cores (1-2)" in text and "# top speed 3900 MHz: 1 cores (0)" in text
    assert text.index("2808 MHz") < text.index("3900 MHz")  # slowest first, though cpu0 is the fast one
    assert "top speed" not in text_from("gb10", cpu_root=fake_cpus(tmp_path / "alike", {"0": "3900000", "1": "3900000"}))
    assert "top speed" not in text_from("gb10")  # a host that reports no speeds (no cpufreq)
    assert tomllib.loads(text)["nodes"][0]["name"] == "testbox"  # the note is a comment: the draft still parses
    assert "fast_cpus" not in text  # three cores in kinds of 1 and 2 cannot be two sockets
    halves = {str(cpu): "2808000" if cpu % 10 < 5 else "3900000" for cpu in range(20)}  # slow 0-4,10-14; fast 5-9,15-19
    two_kinds = text_from("gb10", cpu_root=fake_cpus(tmp_path / "halves", halves))
    assert 'fast_cpus = "5-9,15-19"' in two_kinds  # D46, and the draft loads as a site file
    assert detect.site_config.parse(tomllib.loads(two_kinds)).nodes[0].fast_socket_cores == "10-19"
    assert "fast_cpus" not in text_from(None, cpu_root=tmp_path / "halves")  # no GPU: nothing to bind
    uneven = {str(cpu): "2808000" if cpu < 12 else "3900000" for cpu in range(20)}  # 12 slow, 8 fast: no equal sockets
    assert "fast_cpus" not in text_from("gb10", cpu_root=fake_cpus(tmp_path / "uneven", uneven))


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
