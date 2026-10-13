"""Test ordering and deadlines with fake memory/processes; never reclaim or kill."""
from types import SimpleNamespace
import signal

import cache_first_oom as mod
import pytest

GIB = mod.GIB_KIB


@pytest.fixture
def scenario(monkeypatch):
    state = {"MemAvailable": 6 * GIB, "MemFree": GIB, "Cached": 8 * GIB,
             "SReclaimable": 0}
    actions = []
    monkeypatch.setattr(mod, "memory", lambda: dict(state))
    monkeypatch.setattr(mod, "gpu_active", lambda: False)
    monkeypatch.setattr(mod, "log", lambda event, **fields: actions.append((event, fields)))
    monkeypatch.setattr(mod, "fork_reclaim", lambda: actions.append(("reclaim", {})) or 7)
    monkeypatch.setattr(mod.os, "waitpid", lambda *a: (0, 0))
    monkeypatch.setattr(mod.os, "kill", lambda pid, sig: actions.append(("signal", {"sig": sig})))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda args: actions.append(("killer", {})) or
                        SimpleNamespace(poll=lambda: None, terminate=lambda: None))
    guard = mod.Guard(["-M", f"{8 * GIB},{4 * GIB}", "-s", "100,100"])
    return guard, state, actions


def events(actions):
    return [event for event, fields in actions]


def test_reclaim_precedes_any_kill(scenario):
    guard, state, actions = scenario
    guard.step(0)
    assert "reclaim" in events(actions)
    assert "killer" not in events(actions)


def test_recovered_memory_does_not_enable_killer(scenario, monkeypatch):
    guard, state, actions = scenario
    guard.step(0)
    state["MemAvailable"] = 20 * GIB
    monkeypatch.setattr(mod.os, "waitpid", lambda *a: (7, 0))
    guard.step(0.1)
    assert "cache_reclaim_finished" in events(actions)
    assert "killer" not in events(actions)


@pytest.mark.parametrize("status", [0, 256])
def test_still_low_memory_enables_existing_killer_after_attempt(scenario, monkeypatch, status):
    guard, state, actions = scenario
    guard.step(0)
    monkeypatch.setattr(mod.os, "waitpid", lambda *a: (7, status))
    guard.step(0.1)
    assert events(actions).index("cache_reclaim_finished") < events(actions).index("killer")


def test_stuck_reclaim_cannot_block_emergency_kill(scenario):
    guard, state, actions = scenario
    state["MemAvailable"] = 3 * GIB
    guard.step(0)
    guard.step(0.3)
    assert "cache_reclaim_timeout" in events(actions)
    assert "killer" in events(actions)
    assert any(fields.get("sig") == signal.SIGKILL for event, fields in actions)


def test_emergency_shortens_an_in_progress_reclaim(scenario):
    guard, state, actions = scenario
    guard.step(0)
    state["MemAvailable"] = 3 * GIB
    guard.step(0.1)
    assert guard.deadline <= 0.31
    guard.step(0.4)
    assert "killer" in events(actions)


def test_failed_reclaim_is_not_reported_as_success(scenario, monkeypatch):
    guard, state, actions = scenario
    guard.begin_reclaim(state, 0)
    monkeypatch.setattr(mod.os, "waitpid", lambda *a: (7, 256))
    assert not guard.reclaim_pending(0.1)
    assert not guard.reclaim_ok


def test_killer_stops_once_memory_recovers(scenario, monkeypatch):
    guard, state, actions = scenario
    guard.step(0)
    monkeypatch.setattr(mod.os, "waitpid", lambda *a: (7, 0))
    guard.step(0.1)
    guard.killer.terminate = lambda: actions.append(("stop_killer", {}))
    state["MemAvailable"] = 20 * GIB
    guard.step(0.2)
    assert "stop_killer" in events(actions)


def test_gpu_low_free_high_cache_triggers_without_low_available(scenario, monkeypatch):
    guard, state, actions = scenario
    state["MemAvailable"] = 54 * GIB
    monkeypatch.setattr(mod, "gpu_active", lambda: True)
    guard.step(0)
    assert "reclaim" in events(actions)
    assert "killer" not in events(actions)


def test_reclaim_is_rate_limited_without_urgent_pressure(scenario, monkeypatch):
    guard, state, actions = scenario
    state["MemAvailable"] = 10 * GIB
    guard.step(0)
    monkeypatch.setattr(mod.os, "waitpid", lambda *a: (7, 0))
    guard.step(0.1)
    guard.step(30)
    assert events(actions).count("reclaim") == 1


def test_failed_fork_does_not_disable_killing(scenario, monkeypatch):
    guard, state, actions = scenario
    def fail():
        raise OSError("fork failed")
    monkeypatch.setattr(mod, "fork_reclaim", fail)
    guard.step(0)
    assert "cache_reclaim_failed" in events(actions)
    assert "killer" in events(actions)
