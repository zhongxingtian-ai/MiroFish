"""Closing an environment must not invent a successful simulation result."""

from types import SimpleNamespace

from flask import Flask
import pytest

from app.api import simulation as api
from app.services.simulation_runner import RunnerStatus
from app.services.simulation_manager import SimulationStatus


@pytest.mark.parametrize(("success", "authoritative", "original", "expected"), [
    (False, RunnerStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.FAILED),
    (False, RunnerStatus.COMPLETED, SimulationStatus.RUNNING, SimulationStatus.RUNNING),
    (True, RunnerStatus.FAILED, SimulationStatus.FAILED, SimulationStatus.FAILED),
    (True, RunnerStatus.RUNNING, SimulationStatus.RUNNING, SimulationStatus.RUNNING),
    (True, RunnerStatus.STOPPING, SimulationStatus.STOPPING, SimulationStatus.STOPPING),
    (True, RunnerStatus.STOPPED, SimulationStatus.RUNNING, SimulationStatus.STOPPED),
    (True, RunnerStatus.COMPLETED, SimulationStatus.COMPLETED, SimulationStatus.COMPLETED),
])
def test_close_env_projects_only_a_successful_close_and_the_authoritative_terminal_state(monkeypatch, success, authoritative, original, expected):
    state = SimpleNamespace(status=original, error="original failure" if original == SimulationStatus.FAILED else None)
    saved = []
    run = SimpleNamespace(runner_status=authoritative, error="original failure" if authoritative == RunnerStatus.FAILED else None)
    monkeypatch.setattr(api.SimulationRunner, "close_simulation_env", classmethod(lambda _cls, **kwargs: {"success": success}))
    monkeypatch.setattr(api.SimulationRunner, "get_run_state", classmethod(lambda _cls, _sim_id: run))
    monkeypatch.setattr(api, "SimulationManager", lambda: SimpleNamespace(
        get_simulation=lambda _sim_id: state, _save_simulation_state=lambda value: saved.append(value.status),
    ))
    with Flask(__name__).test_request_context("/api/simulation/close-env", method="POST", json={"simulation_id": "fixture"}):
        response = api.close_simulation_env()
    assert response.get_json()["success"] is success
    assert state.status == expected
    if not success or authoritative in {RunnerStatus.RUNNING, RunnerStatus.STOPPING}:
        assert saved == []
    if expected == SimulationStatus.FAILED:
        assert state.error == "original failure"
