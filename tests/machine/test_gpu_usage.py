"""Incident attribution must survive missing GPU values and disappearing PIDs."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[2] / "machine/gpu/record_gpu_usage.py"
spec = importlib.util.spec_from_file_location("record_gpu_usage", path)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def test_query_preserves_unsupported_memory(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout='7, [N/A], python\n', stderr=''))
    result = gpu.query("pid,used_gpu_memory,process_name", "compute-apps")
    assert result["error"] is None
    assert result["rows"][0]["used_gpu_memory"] == "[N/A]"


@pytest.mark.parametrize("failure", [OSError("missing"), subprocess.TimeoutExpired("nvidia-smi", 3)])
def test_query_failure_is_not_an_empty_success(monkeypatch, failure):
    def fail(*a, **kw):
        raise failure
    monkeypatch.setattr(gpu.subprocess, "run", fail)
    assert gpu.query("pid", "compute-apps")["error"]


def test_owner_records_uid_start_time_and_unit_without_command_line(tmp_path, monkeypatch):
    proc = tmp_path / "7"
    proc.mkdir()
    (proc / "stat").write_text('7 (python worker) ' + ' '.join(['S'] + ['0'] * 18 + ['456']))
    (proc / "status").write_text('Uid:\t1002\t1002\t1002\t1002\n')
    (proc / "cgroup").write_text('0::/user.slice/training.service\n')
    (proc / "cmdline").write_text('secret token')
    monkeypatch.setattr(gpu.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="colleague"))
    result = gpu.owner(7, tmp_path)
    assert result["uid"] == 1002
    assert result["username"] == "colleague"
    assert result["start_ticks"] == 456
    assert "training.service" in result["cgroup"]
    assert "secret" not in str(result)


def test_exited_process_is_explicitly_unknown(tmp_path):
    result = gpu.owner(7, tmp_path)
    assert result["uid"] is None
    assert result["owner_error"]


def test_memory_retains_free_available_and_cache_as_distinct_values(tmp_path):
    (tmp_path / "meminfo").write_text('MemFree: 1 kB\nMemAvailable: 100 kB\nCached: 90 kB\n')
    assert gpu.host_memory(tmp_path)["kib"] == {"MemFree": 1, "MemAvailable": 100, "Cached": 90}
