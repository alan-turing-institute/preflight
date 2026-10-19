"""Property-based tests (Hypothesis) of what users and slurm control: queue parsing, the web app's request guards and
escaping, the log filters, and the site file's refusal of unknown keys.

Hypothesis is a test-only dependency (slurm/ stays stdlib-only), so this file is skipped without it:
    uv run --no-project --with pytest --with hypothesis pytest -q -p no:cacheprovider tests/slurm/test_properties.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# Hypothesis keeps a cache directory; never in the repository.
os.environ.setdefault("HYPOTHESIS_STORAGE_DIRECTORY", str(Path(tempfile.gettempdir()) / "gpujobs-hypothesis"))
pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

# No example database: it would land in the working directory and replay old failures.
settings.register_profile("repo", database=None)
settings.load_profile("repo")

import gpu_jobs  # noqa: E402 - after the Hypothesis setup above
import jobs_view  # noqa: E402 - after the Hypothesis setup above
import site_config  # noqa: E402 - after the Hypothesis setup above
from test_ood_app import Fake, call  # noqa: E402 - after the Hypothesis setup above

# A job name is any text: slurm keeps even a newline in one (measured, 23.11.4), and the owner may rename the job.
NAME = st.text(st.characters(blacklist_categories=("Cs",)), max_size=40)
USER = st.from_regex(r"[a-z_][a-z0-9_-]{0,15}", fullmatch=True)


@given(job=st.integers(1, 10**7), user=USER, priority=st.integers(0, 2**32 - 1), gpus=st.integers(0, 8), name=NAME)
def test_squeue_jobs_parse_back_to_their_fields_whatever_the_name(job: int, user: str, priority: int, gpus: int, name: str) -> None:
    def raw(job_id: int, job_name: str) -> dict[str, object]:
        return {"job_id": job_id, "partition": "normal", "job_state": ["PENDING"], "state_reason": "Priority",
                "dependency": f"afterok:{job - 1}", "user_name": user, "priority": {"set": True, "number": priority},
                "tres_per_node": f"gres/gpu:{gpus}" if gpus else "", "name": job_name}
    first, second = gpu_jobs.parse_squeue(json.dumps({"jobs": [raw(job, name), raw(job + 1, "next")]}))
    assert (first.job_id, first.user, first.priority, first.gpus, first.name) == (str(job), user, priority, gpus, name)
    assert second.job_id == str(job + 1)  # a name never swallows or forges a job


@settings(max_examples=300)
@given(job_id=st.text(st.characters(blacklist_categories=("Cs",), blacklist_characters="/?#"), min_size=1, max_size=12))
def test_only_ascii_digit_ids_reach_a_tool(job_id: str) -> None:
    fake = Fake()
    call(fake, f"/api/job/{job_id}/kill", "POST", {})
    call(fake, f"/api/job/{job_id}/pause-preview")
    for argv in fake.ran:
        assert argv[2].isascii() and argv[2].isdigit(), argv


@given(payload=st.text(min_size=1, max_size=30))
def test_anything_in_a_job_name_is_escaped_on_every_page(payload: str) -> None:
    marker = "<zq9" + payload.replace(":", "")  # no page has this tag ("<b" + "o" matched <body>)
    fake = Fake(name=f"proj:normal:{marker}")
    for path in ("/", "/job/7"):
        status, page = call(fake, path)
        assert status == "200 OK" and marker not in page


LINE = st.text(st.characters(blacklist_categories=("Cs", "Zl", "Zp", "Cc")), max_size=30)


@given(lines=st.lists(LINE, max_size=30), needle=st.text(st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=4),
       errors_only=st.booleans())
def test_log_filters_only_drop_lines(lines: list[str], needle: str, errors_only: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "job.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = jobs_view.log_tail(str(log), text=needle, errors_only=errors_only)
    kept = out.split("\n") if out else []
    assert all(line in lines and needle.lower() in line.lower() for line in kept)
    assert not errors_only or all(jobs_view.ERROR_LINE.search(line) for line in kept)
    assert len(kept) == sum(1 for line in lines if needle.lower() in line.lower() and (not errors_only or jobs_view.ERROR_LINE.search(line)))


MINIMAL = {"name": "t", "nodes": [{"name": "n", "cpus": 1, "real_memory_mb": 1}], "metrics": {}, "portal": {}}


@given(table=st.sampled_from(["top", "metrics", "portal", "node"]), key=st.from_regex(r"zz_[a-z_]{0,12}", fullmatch=True),
       value=st.one_of(st.integers(), st.text(max_size=5), st.booleans()))
def test_an_unknown_key_is_refused_in_every_table(table: str, key: str, value: object) -> None:
    raw = json.loads(json.dumps(MINIMAL))
    {"top": raw, "metrics": raw["metrics"], "portal": raw["portal"], "node": raw["nodes"][0]}[table][key] = value
    with pytest.raises(site_config.SiteError, match="unknown key"):
        site_config.parse(raw)
