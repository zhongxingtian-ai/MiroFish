"""Force restart checks only the persona files generated for enabled platforms."""

import json

from flask import Flask
import pytest

from app.api import simulation as api
from app.config import Config
from app.services.simulation_manager import SimulationManager, SimulationState, SimulationStatus
from app.services.simulation_runner import SimulationRunner, SimulationRunState, RunnerStatus
from test_simulation_interactive_completion import interactive_run


@pytest.fixture
def prepared_case(interactive_run, monkeypatch):
    case = interactive_run
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(case.sim_dir.parent))
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(case.sim_dir.parent))

    def prepare(platform, missing_required=False):
        selected = ("twitter", "reddit") if platform == "parallel" else (platform,)
        state = SimulationState(
            case.simulation_id, "project-fixture", "graph-fixture",
            enable_twitter="twitter" in selected, enable_reddit="reddit" in selected,
            status=SimulationStatus.READY, profiles_generated=True, config_generated=True,
            profiles_count=1, entities_count=1,
        )
        SimulationManager()._save_simulation_state(state)
        for item in selected:
            if not missing_required or item != selected[0]:
                filename = "twitter_profiles.csv" if item == "twitter" else "reddit_profiles.json"
                content = "user_id,name\n0,Fixture\n" if item == "twitter" else "[{}]"
                (case.sim_dir / filename).write_text(content, encoding="utf-8")
        queue = case.sim_dir / "ipc_commands"
        queue.mkdir()
        case.stale = queue / "old-close.json"
        case.stale.write_text(json.dumps({"command_id": "old-close", "command_type": "close_env", "args": {}}), encoding="utf-8")
        case.stale_before = case.stale.read_bytes()
        case.preserved = {path: path.read_bytes() for path in case.sim_dir.iterdir() if path.is_file() and path.name != "state.json"}
        return case

    return prepare


@pytest.mark.parametrize("platform", ["twitter", "reddit", "parallel"])
@pytest.mark.parametrize("missing_required", [False, True])
@pytest.mark.parametrize("start_mode", ["explicit", "default"])
def test_force_http_restart_accepts_only_enabled_personas_and_preserves_failed_attempts(
    prepared_case, monkeypatch, platform, missing_required, start_mode,
):
    case = prepared_case(platform, missing_required)
    starts = []

    def start(_cls, simulation_id, **kwargs):
        assert not case.stale.exists(), "Old commands must be cleaned before a producer can start"
        starts.append((simulation_id, kwargs["platform"]))
        return SimulationRunState(simulation_id, runner_status=RunnerStatus.RUNNING)

    monkeypatch.setattr(SimulationRunner, "start_simulation", classmethod(start))
    payload = {"simulation_id": case.simulation_id, "force": True}
    if start_mode == "explicit":
        payload["platform"] = platform
    with Flask(__name__).test_request_context("/api/simulation/start", method="POST", json=payload):
        result = api.start_simulation()
    response, status = result if isinstance(result, tuple) else (result, result.status_code)
    assert status == (400 if missing_required else 200), response.get_json()
    assert all(path.read_bytes() == content for path, content in case.preserved.items())
    if missing_required:
        assert starts == []
        assert case.stale.read_bytes() == case.stale_before
    else:
        assert starts == [(case.simulation_id, platform)]
        assert response.get_json()["data"]["force_restarted"] is True
        ready, details = api._check_simulation_prepared(case.simulation_id)
        assert ready is True
        assert details["profiles_count"] == 1


@pytest.mark.parametrize(("enabled", "requested"), [
    ("twitter", "reddit"), ("twitter", "parallel"),
    ("reddit", "twitter"), ("reddit", "parallel"),
])
def test_disabled_platform_is_rejected_before_force_cleanup_or_process_launch(prepared_case, monkeypatch, enabled, requested):
    case = prepared_case(enabled)
    before = {path: path.read_bytes() for path in case.sim_dir.rglob("*") if path.is_file()}
    monkeypatch.setattr(SimulationRunner, "start_simulation", classmethod(
        lambda *_args, **_kwargs: pytest.fail("A disabled platform must never reach process launch"),
    ))
    with Flask(__name__).test_request_context("/api/simulation/start", method="POST", json={
        "simulation_id": case.simulation_id, "platform": requested, "force": True,
    }):
        result = api.start_simulation()
    response, status = result if isinstance(result, tuple) else (result, result.status_code)
    assert status == 400, response.get_json()
    assert {path: path.read_bytes() for path in case.sim_dir.rglob("*") if path.is_file()} == before
