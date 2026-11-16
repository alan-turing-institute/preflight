"""Fetch-back records: the state rule (design 3, 4, 6) and private, atomic storage."""

from __future__ import annotations

import dataclasses
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slurm"))

import fetch_record as fr

POLICY = fr.Policy(retry_s=(60, 300, 900), lost_after_s=3600)
NOW = 1_000_000.0
WORKDIR = "/home/alice/gpujobs-sync/proj"


def record(**given: object) -> fr.Record:
    base = fr.Record(
        site="site-b", jobs=("41", "42"), created=NOW - 100, deadline=NOW + 7 * 86400, fetch_on=fr.FetchOn.COMPLETED,
        existing=fr.Existing.UPDATE, keep_free_gb=10, repo="/home/alice/proj",
        paths=(fr.FetchPath("gpujobs-sync/proj/out", "/home/alice/proj/out", (("-", "*.ckpt"),)),),
    )
    return dataclasses.replace(base, **given)


def answer(*states: str) -> fr.Answer:
    return fr.Answer(True, {job: (state, "alice", WORKDIR) for job, state in zip(("41", "42"), states, strict=False)})


@pytest.mark.parametrize(
    ("fetch_on", "states", "expected"),
    [
        (fr.FetchOn.COMPLETED, ("COMPLETED", "RUNNING"), fr.RecordState.WAITING),
        (fr.FetchOn.COMPLETED, ("COMPLETED", "REQUEUED"), fr.RecordState.WAITING),  # not a listed end: waits, never fetches early
        (fr.FetchOn.COMPLETED, ("COMPLETED", "COMPLETED"), fr.RecordState.FETCHING),
        (fr.FetchOn.COMPLETED, ("COMPLETED", "FAILED"), fr.RecordState.SKIPPED),
        (fr.FetchOn.COMPLETED, ("COMPLETED", "CANCELLED by 1002"), fr.RecordState.SKIPPED),
        (fr.FetchOn.FINISHED, ("COMPLETED", "CANCELLED by 1002"), fr.RecordState.FETCHING),
        (fr.FetchOn.FINISHED, ("OUT_OF_MEMORY", "TIMEOUT"), fr.RecordState.FETCHING),
        (fr.FetchOn.FINISHED, ("PENDING", "TIMEOUT"), fr.RecordState.WAITING),
    ],
)
def test_a_record_fetches_only_when_every_job_ended_as_it_asks(fetch_on, states, expected) -> None:
    moved = fr.advance(record(fetch_on=fetch_on), answer(*states), NOW, POLICY)
    assert moved.state == expected
    assert (moved.owner, moved.workdir) == ("alice", WORKDIR)
    assert moved.job_states == dict(zip(("41", "42"), states, strict=True))


def test_no_answer_is_not_an_unknown_id_and_an_unknown_id_is_lost_only_after_the_grace() -> None:
    down = fr.Answer(False)
    once = fr.advance(record(), down, NOW, POLICY)
    assert (once.state, once.failures, once.next_try) == (fr.RecordState.WAITING, 1, NOW + 60)
    steps = [fr.advance(dataclasses.replace(record(), failures=n), down, NOW, POLICY).next_try - NOW for n in range(5)]
    assert steps == [60, 300, 900, 900, 900]  # the last step repeats
    assert fr.advance(once, answer("RUNNING", "RUNNING"), NOW, POLICY).failures == 0  # an answer resets the backoff

    unknown = answer("RUNNING")  # job 42 has no row: accounting does not know it (yet)
    first = fr.advance(record(), unknown, NOW, POLICY)
    assert (first.state, first.empty_since) == (fr.RecordState.WAITING, NOW)
    assert fr.advance(first, unknown, NOW + 3599, POLICY).state == fr.RecordState.WAITING
    assert fr.advance(first, unknown, NOW + 3600, POLICY).state == fr.RecordState.LOST
    known = fr.advance(first, answer("RUNNING", "PENDING"), NOW + 1800, POLICY)
    assert known.empty_since is None
    assert fr.advance(known, unknown, NOW + 3600, POLICY).state == fr.RecordState.WAITING  # the grace starts again


def test_the_deadline_expires_a_waiting_record_and_force_reopens_a_skipped_one_once() -> None:
    late = NOW + 8 * 86400
    assert fr.advance(record(), answer("RUNNING", "RUNNING"), late, POLICY).state == fr.RecordState.EXPIRED
    assert fr.advance(record(), fr.Answer(False), late, POLICY).state == fr.RecordState.EXPIRED
    assert fr.advance(record(), answer("COMPLETED", "COMPLETED"), late, POLICY).state == fr.RecordState.FETCHING

    skipped = fr.advance(record(), answer("COMPLETED", "FAILED"), NOW, POLICY)
    assert fr.advance(skipped, answer("COMPLETED", "FAILED"), NOW, POLICY) == skipped  # final: left alone
    forced = fr.advance(dataclasses.replace(skipped, force=True), answer("COMPLETED", "FAILED"), NOW, POLICY)
    assert (forced.state, forced.force) == (fr.RecordState.FETCHING, False)
    for final in fr.FINAL - {fr.RecordState.SKIPPED}:
        done = record(state=final, force=True)
        assert fr.advance(done, answer("COMPLETED", "COMPLETED"), NOW, POLICY) == done


def test_records_are_saved_privately_and_atomically_and_an_unreadable_one_is_skipped(tmp_path: Path, capsys) -> None:
    folder = fr.store_dir("site-b", {"XDG_STATE_HOME": str(tmp_path)})
    assert folder == tmp_path / "gpujobs" / "fetch" / "site-b"
    assert fr.store_dir("site-b", {"HOME": "/home/alice"}) == Path("/home/alice/.local/state/gpujobs/fetch/site-b")
    moved = fr.advance(record(then="python3 score.py"), answer("COMPLETED", "COMPLETED"), NOW, POLICY)
    path = fr.save(moved, folder)
    assert stat.S_IMODE(path.stat().st_mode) == fr.PRIVATE_FILE
    assert stat.S_IMODE(folder.stat().st_mode) == fr.PRIVATE_DIR
    (folder / "7.json").write_text('{"schema": 99}')
    (folder / "8.json").write_text("not json")
    assert fr.load_all(folder) == [moved]
    assert "skipping" in capsys.readouterr().err
    assert fr.find(folder, "42") == moved
    assert fr.find(folder, "43") is None
    assert sorted(p.name for p in folder.iterdir()) == ["41.json", "7.json", "8.json"]  # no temporary file left
