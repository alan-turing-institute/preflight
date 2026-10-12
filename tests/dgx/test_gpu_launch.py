"""The common lock must cover both the gate and the entire foreground job."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux GPU host wrapper")


@pytest.fixture
def launcher(tmp_path):
    source = Path(__file__).resolve().parents[2] / "dgx/run_gpu_job.sh"
    shutil.copyfile(source, tmp_path / source.name)
    (tmp_path / "assert_gpu_free.py").write_text('import os, sys\nsys.exit(int(os.environ.get("TEST_GATE_EXIT", "0")))\n')
    lock = tmp_path / "lock"
    lock.touch()
    lock.chmod(0o444)
    runtime = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    environment = dict(os.environ, PREFLIGHT_GPU_LOCK=str(lock), PREFLIGHT_UV=runtime)
    return ["bash", str(tmp_path / source.name), "--"], environment


def test_gate_refusal_never_starts_job(launcher, tmp_path):
    command, environment = launcher
    environment["TEST_GATE_EXIT"] = "1"
    marker = tmp_path / "ran"
    result = subprocess.run(command + ["touch", str(marker)], env=environment)
    assert result.returncode == 1
    assert not marker.exists()


def test_concurrent_launch_is_refused_until_job_exits(launcher, tmp_path):
    command, environment = launcher
    ready = tmp_path / "ready"
    first = subprocess.Popen(command + ["sh", "-c", 'exec 9>&-; echo ready > "$1"; sleep 5',
                                        "job", str(ready)], env=environment, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        second = subprocess.run(command + ["true"], env=environment)
        assert second.returncode == 75
        assert first.wait(timeout=10) == 0
        assert subprocess.run(command + ["true"], env=environment).returncode == 0
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGTERM)
            first.wait(timeout=5)
