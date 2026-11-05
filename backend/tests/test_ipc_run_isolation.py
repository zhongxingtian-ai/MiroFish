"""Force cleanup removes stale IPC messages before actual producer consumption."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from app.services.simulation_ipc import IPCCommand, IPCResponse, CommandType, CommandStatus
from app.services.simulation_runner import SimulationRunner, SimulationRunState, RunnerStatus, SimulationStopPending
from filesystem_test_support import directory_redirect
from test_simulation_interactive_completion import interactive_run
from test_simulation_producer_protocol import producer_case


def _stale_queues(directory):
    commands, responses = directory / "ipc_commands", directory / "ipc_responses"
    commands.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    batch = commands / "old-batch.json"
    close = commands / "old-close.json"
    batch.write_text(json.dumps(IPCCommand("old-batch", CommandType.BATCH_INTERVIEW, {
        "interviews": [{"agent_id": 1, "prompt": "STALE PREVIOUS RUN"}],
    }).to_dict()), encoding="utf-8")
    close.write_text(json.dumps(IPCCommand("old-close", CommandType.CLOSE_ENV, {}).to_dict()), encoding="utf-8")
    response = responses / "old-response.json"
    response.write_text(json.dumps(IPCResponse("old-response", CommandStatus.COMPLETED, {"old": True}).to_dict()), encoding="utf-8")
    return [batch, close, response]


def test_force_cleanup_removes_stale_ipc_before_production_script_reads_it(producer_case, monkeypatch):
    directory = producer_case.directory
    simulation_id = directory.name
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(directory.parent))
    for name in ("_run_states", "_processes", "_monitor_threads", "_stdout_files", "_stderr_files", "_startup_claims", "_graph_memory_enabled", "_finalization_locks"):
        monkeypatch.setattr(SimulationRunner, name, {})
    SimulationRunner._save_run_state(SimulationRunState(simulation_id, runner_status=RunnerStatus.COMPLETED))
    stale = _stale_queues(directory)
    keep = directory / "ipc_commands" / "notes.txt"
    keep.write_text("User note, not a temporary IPC message", encoding="utf-8")
    graph = directory / "graph.sqlite3"
    graph.write_bytes(b"unrelated graph sentinel")
    preserved = {path: path.read_bytes() for path in [directory / "simulation_config.json", directory / "twitter_profiles.csv", directory / "reddit_profiles.json", keep, graph]}
    result = SimulationRunner.cleanup_simulation_logs(simulation_id)
    assert result["success"] is True
    produced = producer_case.run("twitter", ["twitter"], [])
    assert "id=old-batch" not in produced.stdout and "id=old-close" not in produced.stdout, produced.stdout
    assert produced.returncode == 0, produced.stdout + produced.stderr
    assert all(not path.exists() for path in stale)
    assert all(path.read_bytes() == content for path, content in preserved.items())
    with closing(sqlite3.connect(directory / "twitter_simulation.db")) as connection:
        prompts = [json.loads(row[0]).get("prompt") for row in connection.execute("SELECT info FROM trace WHERE action='interview'")]
    assert prompts == ["fixture question"]


def test_non_force_with_only_stale_ipc_refuses_and_preserves_queue_bytes(interactive_run):
    case = interactive_run
    paths = _stale_queues(case.sim_dir)
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="force"):
        SimulationRunner.start_simulation(case.simulation_id, platform="twitter")
    assert case.processes == []
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("phase", ["live", "starting", "failed-claim"])
def test_ipc_cleanup_refuses_live_or_unreleased_startup_ownership(interactive_run, phase):
    case = interactive_run
    if phase == "live":
        case.start("twitter", finish_platforms=False)
    else:
        status = RunnerStatus.STARTING if phase == "starting" else RunnerStatus.FAILED
        SimulationRunner._save_run_state(SimulationRunState(case.simulation_id, runner_status=status))
        from app.services.simulation_runner import _StartupClaim
        SimulationRunner._startup_claims[case.simulation_id] = _StartupClaim()
    paths = _stale_queues(case.sim_dir)
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(SimulationStopPending):
        SimulationRunner.cleanup_simulation_logs(case.simulation_id)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_ipc_redirect_is_rejected_before_any_owned_or_foreign_file_is_removed(interactive_run):
    case = interactive_run
    SimulationRunner._save_run_state(SimulationRunState(case.simulation_id, runner_status=RunnerStatus.COMPLETED))
    outside = case.sim_dir.parent.parent / "outside-queue"
    outside.mkdir()
    foreign = outside / "old-close.json"
    foreign.write_text('{"command_type":"close_env"}', encoding="utf-8")
    directory_redirect(case.sim_dir / "ipc_commands", outside)
    state_file = case.sim_dir / "run_state.json"
    before = state_file.read_bytes(), foreign.read_bytes()
    with pytest.raises(ValueError):
        SimulationRunner.cleanup_simulation_logs(case.simulation_id)
    assert (state_file.read_bytes(), foreign.read_bytes()) == before
