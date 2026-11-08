"""components/metrics.py: what the Jobstats metrics component installs, per mode, architecture and portal setting.

No network: downloads are checked as data (pinned URL, version, sha256) and the shell steps run against local files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import site_config

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "slurm" / "install"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and `import render` look modules up here
    spec.loader.exec_module(module)
    return module


render = _load("render", INSTALL / "render.py")
metrics = _load("metrics_component", INSTALL / "components" / "metrics.py")

HEX64 = re.compile(r"[0-9a-f]{64}")
URL = re.compile(r"https?://\S+")
GPU = {"index": 0, "uuid": "GPU-1", "device": "/dev/nvidia0", "type": "gb10"}


def make_site(
    mode: str = "local", *, gpus: bool = True, portal: str = "off", state_dir: str = "/var/lib/gpujobs", **extra: object
) -> site_config.Site:
    metrics_table = {"mode": mode, "grafana_port": 3001, "retention_days": 90, **extra}
    if mode == "external":
        metrics_table = {"prometheus_url": "http://prom.example:9090", "grafana_url": "https://grafana.example", **metrics_table}
    return site_config.parse(
        {
            "name": "box",
            "state_dir": state_dir,
            "nodes": [{"name": "n1", "cpus": 4, "real_memory_mb": 1000, "gpus": [GPU] if gpus else []}],
            "metrics": metrics_table,
            "portal": {"mode": portal},
        }
    )


@pytest.fixture(params=["x86_64", "aarch64"])
def arch(request, monkeypatch) -> str:
    monkeypatch.setattr(metrics.platform, "machine", lambda: request.param)
    return request.param


def argv_text(steps: list[list[str]]) -> str:
    return "\n".join(" ".join(step) for step in steps)


def test_local_external_and_off_install_different_pieces(arch):
    local, external, off = make_site(), make_site("external"), make_site("off")
    assert [metrics.enabled(s) for s in (local, external, off)] == [True, True, False]
    assert metrics.files(off) == {} and metrics.post_install(off) == [] and metrics.slurm_conf_lines(off) == ""

    local_files, external_files = metrics.files(local), metrics.files(external)
    only_local = {
        "/etc/prometheus/gpujobs.yml",
        "/etc/systemd/system/prometheus.service.d/10-gpujobs.conf",
        "/etc/systemd/system/grafana-server.service.d/10-gpujobs.conf",
        "/etc/grafana/provisioning/datasources/gpujobs.yaml",
        "/etc/grafana/provisioning/dashboards/gpujobs.yaml",
        "/etc/grafana/gpujobs-dashboards/gpujobs-job-memory.json",
    }
    assert only_local <= set(local_files) and not only_local & set(external_files)
    assert set(external_files) <= set(local_files)
    for path in (
        "/usr/local/jobstats/config.py",
        "/etc/gpujobs/prometheus_targets.json",
        "/etc/systemd/system/gpujobs-node-exporter.service",
        "/etc/systemd/system/gpujobs-cgroup-exporter.service",
        "/etc/systemd/system/gpujobs-gpu-exporter.service",
        "/usr/local/lib/gpujobs/run-cgroup-exporter",
    ):
        assert path in external_files

    assert {"prometheus", "grafana"} <= set(metrics.packages(local))
    assert not {"prometheus", "grafana"} & set(metrics.packages(external))
    assert {"python3-requests", "python3-blessed"} <= set(metrics.packages(external))
    assert set(metrics.apt_sources(local)) == {"/etc/apt/keyrings/grafana.asc", "/etc/apt/sources.list.d/grafana.list"}
    assert metrics.apt_sources(external) == {}
    assert "signed-by=/etc/apt/keyrings/grafana.asc] https://apt.grafana.com stable main" in (
        metrics.apt_sources(local)["/etc/apt/sources.list.d/grafana.list"].content
    )
    # An external site's jobstats reads the site's own Prometheus; a local one reads the one installed here.
    assert 'PROM_SERVER = "http://prom.example:9090"' in external_files["/usr/local/jobstats/config.py"].content
    assert 'PROM_SERVER = "http://127.0.0.1:9090"' in local_files["/usr/local/jobstats/config.py"].content
    assert metrics.post_install(external) != metrics.post_install(local)
    assert "grafana-server" in argv_text(metrics.post_install(local))
    assert "grafana-server" not in argv_text(metrics.post_install(external))


def test_retention_ports_and_textfile_directory_are_substituted():
    site = make_site(state_dir="/srv/gj", retention_days=45, grafana_port=3456)
    out = metrics.files(site)
    assert "--storage.tsdb.retention.time=45d" in out["/etc/systemd/system/prometheus.service.d/10-gpujobs.conf"].content
    assert "ExecStart=\nExecStart=/usr/bin/prometheus " in out["/etc/systemd/system/prometheus.service.d/10-gpujobs.conf"].content
    assert "GF_SERVER_HTTP_PORT=3456" in out["/etc/systemd/system/grafana-server.service.d/10-gpujobs.conf"].content
    assert "PROM_RETENTION_DAYS = 45" in out["/usr/local/jobstats/config.py"].content
    node = out["/etc/systemd/system/gpujobs-node-exporter.service"].content
    assert "--collector.textfile.directory=/srv/gj/metrics" in node and "--web.listen-address=:9101" in node
    assert "url: http://127.0.0.1:9090" in out["/etc/grafana/provisioning/datasources/gpujobs.yaml"].content
    # Jobstats selects series by cluster label and node name; the targets carry both.
    (group,) = json.loads(out["/etc/gpujobs/prometheus_targets.json"].content)
    assert group["labels"] == {"cluster": site.cluster_name}
    assert group["targets"] == ["n1:9101", "n1:9306", "n1:9445"]
    slurm_conf = render.render(site, "slurm")["/etc/slurm/slurm.conf"].content
    assert f"ClusterName={site.cluster_name}\n" in slurm_conf


def test_slurm_conf_lines_match_the_jobstats_docs_and_render_into_slurm_conf():
    site = make_site()
    lines = metrics.slurm_conf_lines(site)
    assert [ln for ln in lines.splitlines() if not ln.startswith("#")] == [
        "Prolog=/etc/slurm/prolog.d/*.sh",  # docs/setup/gpu_node_scripts.md:33
        "Epilog=/etc/slurm/epilog.d/*.sh",  # docs/setup/gpu_node_scripts.md:34
        "EpilogSlurmctld=/usr/local/sbin/slurmctldepilog.sh",  # docs/setup/summaries.md:9
    ]
    assert [ln for ln in metrics.slurm_conf_lines(make_site(gpus=False)).splitlines() if not ln.startswith("#")] == [
        "EpilogSlurmctld=/usr/local/sbin/slurmctldepilog.sh"
    ]
    conf = render.render(site, "slurm", {"slurm_conf": lines})["/etc/slurm/slurm.conf"].content
    assert conf.count("Prolog=/etc/slurm/prolog.d/*.sh\n") == 1
    # the containers epilog is the first Epilog= line, so it runs before Jobstats' glob (slurm stops at a failing one)
    assert [ln for ln in conf.splitlines() if ln.startswith("Epilog=")] == ["Epilog=gpujobs_containers.sh", "Epilog=/etc/slurm/epilog.d/*.sh"]
    # docs/setup/cgroups.md:5,11-13 and overview.md:5: slurm must account by cgroup, with gpu in the TRES list.
    for needed in ("JobAcctGatherType=jobacct_gather/cgroup", "ProctrackType=proctrack/cgroup", "AccountingStorageTRES=gres/gpu"):
        assert needed in conf


def test_every_download_is_pinned_and_checked(arch):
    for d in metrics.DOWNLOADS:
        assert d.version and HEX64.fullmatch(d.sha256), d.name
        assert d.url.startswith("https://") and "/latest" not in d.url and "/main/" not in d.url, d.name
        assert d.version in d.url or re.search(r"[0-9a-f]{40}", d.url), d.name
    table = {d.url: d for d in metrics.DOWNLOADS}
    for site in (make_site(), make_site("external", portal="install"), make_site(gpus=False)):
        steps = metrics.post_install(site)
        fetches = [s for s in steps if s[:2] == ["sh", "-ec"] and "curl -fsSL" in s[2]]
        assert fetches and all("sha256sum -c" in s[2] for s in fetches)
        for step in fetches:
            assert HEX64.fullmatch(step[-1]) and table[step[-2]].sha256 == step[-1]
        # No URL appears anywhere else: nothing is fetched that the table does not pin.
        assert set(URL.findall(argv_text(steps))) <= set(table)
        # Fetches come before any step that uses what they fetched.
        last_fetch = max(steps.index(f) for f in fetches)
        assert all(steps.index(s) > last_fetch for s in steps if s[0] in {"tar", "ln", "systemctl"})


def test_fetch_keeps_a_good_download_and_refuses_a_bad_one(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "CACHE_DIR", PurePosixPath(tmp_path / "cache"))
    source = tmp_path / "payload"
    source.write_bytes(b"payload")
    good = metrics.Download("x", "1", source.as_uri(), hashlib.sha256(b"payload").hexdigest(), "x.bin")
    bad = metrics.Download("y", "1", source.as_uri(), "0" * 64, "y.bin")
    (tmp_path / "cache").mkdir()
    assert subprocess.run(metrics._fetch(good), check=False).returncode == 0
    assert (tmp_path / "cache" / "x.bin").read_bytes() == b"payload"
    source.unlink()  # a cached file that still checks out is not fetched again
    assert subprocess.run(metrics._fetch(good), check=False).returncode == 0
    source.write_bytes(b"payload")
    assert subprocess.run(metrics._fetch(bad), check=False, capture_output=True).returncode != 0
    assert not (tmp_path / "cache" / "y.bin").exists()


def test_gpu_exporter_is_a_release_binary_on_x86_and_built_on_aarch64(monkeypatch):
    site = make_site()
    monkeypatch.setattr(metrics.platform, "machine", lambda: "x86_64")
    x86 = argv_text(metrics.post_install(site))
    assert "nvml_exporter-v0.2.4-x86_64" in x86 and "go build" not in x86
    assert "golang-go" not in metrics.packages(site)
    monkeypatch.setattr(metrics.platform, "machine", lambda: "aarch64")
    arm = argv_text(metrics.post_install(site))
    assert "go build" in arm and "GOTOOLCHAIN=local" in arm and "x86_64.tar.gz" not in arm
    assert {"golang-go", "gcc"} <= set(metrics.packages(site))
    assert "linux-arm64" in arm and "cgroup_exporter-arm64" in arm
    no_gpu = argv_text(metrics.post_install(make_site(gpus=False)))
    assert "nvml_exporter" not in no_gpu
    assert not any("gpustats_helper" in path for path in metrics.files(make_site(gpus=False)))
    assert "/etc/systemd/system/gpujobs-gpu-exporter.service" not in metrics.files(make_site(gpus=False))
    assert "golang-go" not in metrics.packages(make_site(gpus=False))


def test_portal_adds_the_ood_helper_and_anonymous_grafana(arch):
    off, on = make_site(), make_site(portal="install")
    assert "ood-jobstats-helper" not in argv_text(metrics.post_install(off))
    on_text = argv_text(metrics.post_install(on))
    assert "/var/www/ood/apps/sys/jobstats" in on_text and "--strip-components=2" in on_text  # the app links there
    grafana = "/etc/systemd/system/grafana-server.service.d/10-gpujobs.conf"
    assert "GF_AUTH_ANONYMOUS_ENABLED" not in metrics.files(off)[grafana].content
    for setting in ("GF_AUTH_ANONYMOUS_ENABLED=true", "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer", "GF_SECURITY_ALLOW_EMBEDDING=true"):
        assert setting in metrics.files(on)[grafana].content
    assert "GF_SERVER_ROOT_URL" not in metrics.files(on)[grafana].content  # no proxy address: Grafana's own
    proxied = metrics.files(make_site(portal="install", grafana_url="https://g.example"))[grafana].content
    assert "Environment=GF_SERVER_ROOT_URL=https://g.example\n" in proxied
    # Without a login, who may reach Grafana and the exporters is the site's list of networks, enforced by systemd
    units = [grafana, *(f"/etc/systemd/system/gpujobs-{e}-exporter.service" for e in ("node", "cgroup", "gpu"))]
    limited = metrics.files(make_site(portal="install", allow_from=["100.64.0.0/10", "fd7a:115c:a1e0::/48"]))
    for unit in units:
        assert "IPAddressDeny=any\nIPAddressAllow=localhost 100.64.0.0/10 fd7a:115c:a1e0::/48\n" in limited[unit].content
        assert "IPAddress" not in metrics.files(on)[unit].content  # no list: any network, as before
    dash = metrics.GRAFANA_DASHBOARD  # OOD's custom.grafana block minus host
    assert dash["orgId"] == 1 and dash["dashboard"]["uid"] == "HRLkiLS7k"
    assert dash["dashboard"]["panels"] == {"cpu": 7, "memory": 9, "gpu-util": 2, "gpu-memory": 3}
    assert metrics.apt_flags(on) == ["--no-install-recommends"]


def test_refusals(arch, monkeypatch):
    assert metrics.refusals(make_site()) == []
    assert metrics.refusals(make_site("off")) == []
    external_no_urls = site_config.parse(
        {"name": "b", "nodes": [{"name": "n", "cpus": 1, "real_memory_mb": 1}], "metrics": {"mode": "external"}}
    )
    refused = metrics.refusals(external_no_urls)
    assert [("prometheus_url" in r, "grafana_url" in r) for r in refused] == [(True, False), (False, True)]
    (clash,) = metrics.refusals(make_site(grafana_port=9090))
    assert "grafana_port 9090" in clash
    monkeypatch.setattr(metrics.platform, "machine", lambda: "riscv64")
    (refusal,) = metrics.refusals(make_site())
    assert "riscv64" in refusal and metrics.post_install(make_site()) == []


def test_cgroup_exporter_wrapper_finds_slurms_scope(tmp_path):
    script = tmp_path / "run"
    script.write_text(metrics.files(make_site())["/usr/local/lib/gpujobs/run-cgroup-exporter"].content)
    fake = tmp_path / "exporter"
    fake.write_text('#!/bin/sh\necho "$@"\n')
    fake.chmod(0o755)
    root = tmp_path / "cgroup"
    (root / "system.slice").mkdir(parents=True)
    env = {"CGROUP_ROOT": str(root), "CGROUP_EXPORTER": str(fake), "PATH": "/usr/bin:/bin"}

    def run() -> str:
        return subprocess.run(["sh", str(script), "9306"], env=env, check=True, capture_output=True, text=True).stdout

    assert "--path.slurmstepd.scope=/system.slice/slurmstepd.scope" in run()  # slurmd not up yet: the exporter's default
    (root / "system.slice" / "n1_slurmstepd.scope").mkdir()  # as seen on this box: <node>_slurmstepd.scope
    out = run()
    assert "--path.slurmstepd.scope=/system.slice/n1_slurmstepd.scope" in out
    assert "--web.listen-address=:9306" in out and "--collect.fullslurm" in out
    assert f"{metrics.BIN_DIR}/cgroup_exporter" in script.read_text()


def test_grafana_steps_rewrite_the_datasource_and_create_the_admin_password_once(tmp_path):
    rewrite, password = metrics._grafana_steps(make_site())
    source = tmp_path / "dash.json"
    source.write_text('{"datasource": "${DS_DEFAULT_LOCAL PROMETHEUS}", "expr": "nvidia_gpu_memory_used_bytes{jobid=1}"}')
    target = tmp_path / "out" / "dash.json"
    subprocess.run([*rewrite[:4], str(source), "my-prom", str(target)], check=True)
    assert json.loads(target.read_text()) == {"datasource": "my-prom", "expr": "nvidia_gpu_memory_used_bytes{jobid=1}"}
    # a GPU sharing the host's memory reports none to the exporter: the memory panel reads the agent's series instead
    _, repoint, _ = metrics._grafana_steps(dataclasses.replace(make_site(), unified_memory=True))
    subprocess.run([*repoint[:-1], str(target)], check=True)
    assert json.loads(target.read_text())["expr"] == "gpujobs_job_gpu_memory_bytes{jobid=1}"
    env_file = tmp_path / "gpujobs.env"
    subprocess.run([*password[:4], str(env_file)], check=True)
    first = env_file.read_text()
    assert re.fullmatch(r"GF_SECURITY_ADMIN_PASSWORD=[A-Za-z0-9]{20,}\n", first)
    assert stat.S_IMODE(env_file.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
    subprocess.run([*password[:4], str(env_file)], check=True)
    assert env_file.read_text() == first


def test_templates_are_valid_and_jobstats_config_defines_what_jobstats_reads():
    files = metrics.files(make_site())
    own = json.loads(files["/etc/grafana/gpujobs-dashboards/gpujobs-job-memory.json"].content)
    assert own["uid"] == metrics.OWN_DASHBOARD["uid"]
    assert "gpujobs_job_gpu_memory_bytes" in own["panels"][0]["targets"][0]["expr"]
    namespace: dict[str, object] = {}
    exec(files["/usr/local/jobstats/config.py"].content, namespace)  # noqa: S102 - our own rendered text
    # Every `c.NAME` that jobstats.py / output_formatters.py v1.1.0 reads without a hasattr guard, plus PROM_SERVER.
    required = {
        "PROM_SERVER", "PROM_RETENTION_DAYS", "SAMPLING_PERIOD", "GPU_EXPORTER_JOBID", "EXTERNAL_DB_CONFIG",
        "CLUSTER_TRANS", "CLUSTER_TRANS_INV", "MAX_JOBNAME_LEN", "GPU_UTIL_RED", "CPU_UTIL_RED",
        "TIME_EFFICIENCY_RED", "TIME_EFFICIENCY_BLACK", "MIN_RUNTIME_SECONDS", "GPU_METRICS", "GPU_METRICS_EXPORTER",
        "GPU_METRICS_DOCS_URL", "NOTES",
    }  # fmt: skip
    assert required <= set(namespace) and namespace["NOTES"] == []
    key = (INSTALL / "templates" / "metrics" / "grafana.asc").read_bytes()
    assert hashlib.sha256(key).hexdigest() == "d8f5f6f4c174c3b9184cb6ebbf691a2ee69831a109425de4e821f5b43c53a2f8"


def test_slurm_scripts_are_installed_with_the_files_and_do_what_the_jobstats_docs_describe(tmp_path):
    out = metrics.files(make_site())
    names = {"prolog": "/etc/slurm/prolog.d/gpustats_helper.sh", "epilog": "/etc/slurm/epilog.d/gpustats_helper.sh"}
    ctld = "/usr/local/sbin/slurmctldepilog.sh"
    assert all(out[path].mode == 0o755 for path in (*names.values(), ctld))  # in files(): there before slurm restarts
    assert ctld in metrics.files(make_site(gpus=False)) and names["prolog"] not in metrics.files(make_site(gpus=False))
    smi = tmp_path / "smi"
    smi.write_text("#!/bin/sh\necho GPU-abc\n")
    smi.chmod(0o755)
    gpustat = tmp_path / "gpustat"
    env = {
        "PATH": "/usr/bin:/bin", "GPUSTAT_DIR": str(gpustat), "NVIDIA_SMI": str(smi), "SLURM_JOB_ID": "5",
        "SLURM_JOB_UID": "1000", "CUDA_VISIBLE_DEVICES": "0",
    }  # fmt: skip

    def run(content: str, extra: dict[str, str] | None = None) -> None:
        script = tmp_path / "script.sh"
        script.write_text(content)
        subprocess.run(["sh", str(script)], env={**env, **(extra or {})}, check=True)

    run(out[names["prolog"]].content)  # docs/setup/gpu_node_scripts.md:23-44: "<jobid> <uid>" per ordinal and UUID
    assert (gpustat / "0").read_text() == "5 1000\n" and (gpustat / "GPU-abc").read_text() == "5 1000\n"
    run(out[names["epilog"]].content)
    assert list(gpustat.iterdir()) == []
    calls = tmp_path / "calls"
    js = tmp_path / "jobstats"
    js.write_text("#!/bin/sh\necho H4sIAAAA\n")
    js.chmod(0o755)
    acct = tmp_path / "sacctmgr"
    acct.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n')
    acct.chmod(0o755)
    tools = {"JOBSTATS": str(js), "SACCTMGR": str(acct), "SETTLE_SECONDS": "0"}
    run(out[ctld].content, tools)
    assert calls.read_text() == "-i update job where jobid=5 set AdminComment=JS1:H4sIAAAA\n"
    run(out[ctld].content, {**tools, "SLURM_ARRAY_JOB_ID": "5", "SLURM_ARRAY_TASK_ID": "3"})
    assert calls.read_text().endswith("-i update job where jobid=5_3 set AdminComment=JS1:H4sIAAAA\n")
    js.write_text("#!/bin/sh\necho 'Failed to lookup job'\n")
    calls.unlink()
    run(out[ctld].content, tools)  # no summary: nothing stored, and the epilog still exits 0
    assert not calls.exists()
