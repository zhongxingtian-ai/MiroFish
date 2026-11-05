"""Review reproductions for old-run isolation and cancellation before Popen returns."""

import threading

import pytest

from app.services import simulation_runner as module
from app.services.simulation_runner import SimulationRunner, SimulationRunState, RunnerStatus, SimulationStopPending
from test_simulation_interactive_completion import interactive_run


@pytest.mark.parametrize("artifact", ["run_state", "actions"])
def test_non_force_restart_rejects_old_records_without_modifying_them(interactive_run, artifact):
    case = interactive_run
    if artifact == "run_state":
        SimulationRunner._save_run_state(SimulationRunState(case.simulation_id, runner_status=RunnerStatus.COMPLETED))
        path = case.sim_dir / "run_state.json"
    else:
        case.end_platform("twitter")
        path = case.sim_dir / "twitter" / "actions.jsonl"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="force"):
        SimulationRunner.start_simulation(case.simulation_id, platform="twitter")
    assert path.read_bytes() == before
    assert case.processes == []


@pytest.mark.parametrize("phase", ["updater", "popen"])
@pytest.mark.parametrize("platform", ["twitter", "parallel"])
def test_stop_during_startup_cannot_publish_stopped_then_leave_a_running_producer(interactive_run, monkeypatch, phase, platform):
    case = interactive_run
    entered, release = threading.Event(), threading.Event()
    case.allow_drain.set()
    original_spawn = module.subprocess.Popen
    original_create = module.ZepGraphMemoryManager.create_updater

    def blocked_spawn(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_spawn(*args, **kwargs)

    def blocked_updater(_cls, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_create(*args, **kwargs)

    if phase == "popen":
        monkeypatch.setattr(module.subprocess, "Popen", blocked_spawn)
    else:
        monkeypatch.setattr(module.ZepGraphMemoryManager, "create_updater", classmethod(blocked_updater))
    outcome = {}

    def start():
        try:
            outcome["state"] = case.start(platform, finish_platforms=False)
        except Exception as error:
            outcome["error"] = error

    starter = threading.Thread(target=start, daemon=True)
    starter.start()
    try:
        assert entered.wait(timeout=1)
        with pytest.raises(SimulationStopPending):
            SimulationRunner.stop_simulation(case.simulation_id)
        assert SimulationRunner.get_run_state(case.simulation_id).runner_status == RunnerStatus.STOPPING
    finally:
        release.set()
        starter.join(timeout=2)
        assert not starter.is_alive(), "Startup cancellation must not wait while holding the publication lock"
    assert all(process.poll() is not None for process in case.processes)
    assert SimulationRunner.get_run_state(case.simulation_id).runner_status == RunnerStatus.STOPPED
    assert RunnerStatus.RUNNING not in [status for status, error in case.synced]
    assert case.simulation_id not in SimulationRunner._processes
    assert case.simulation_id not in SimulationRunner._stdout_files


@pytest.mark.parametrize("phase", ["updater", "popen"])
def test_application_cleanup_waits_for_pending_startup_cancellation_without_holding_its_lock(interactive_run, monkeypatch, phase):
    case = interactive_run
    entered, release = threading.Event(), threading.Event()
    case.allow_drain.set()
    spawn = module.subprocess.Popen
    create = module.ZepGraphMemoryManager.create_updater

    def blocked_spawn(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return spawn(*args, **kwargs)

    def blocked_create(_cls, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return create(*args, **kwargs)

    if phase == "popen":
        monkeypatch.setattr(module.subprocess, "Popen", blocked_spawn)
    else:
        monkeypatch.setattr(module.ZepGraphMemoryManager, "create_updater", classmethod(blocked_create))
    starter = threading.Thread(target=lambda: case.start("twitter", finish_platforms=False), daemon=True)
    cleaner = threading.Thread(target=SimulationRunner.cleanup_all_simulations, daemon=True)
    starter.start()
    try:
        assert entered.wait(timeout=1)
        cleaner.start()
        deadline = module.time.monotonic() + 1
        while SimulationRunner.get_run_state(case.simulation_id).runner_status != RunnerStatus.STOPPING and module.time.monotonic() < deadline:
            threading.Event().wait(0.005)
        assert SimulationRunner.get_run_state(case.simulation_id).runner_status == RunnerStatus.STOPPING
        assert cleaner.is_alive(), "Shutdown cannot finish before startup confirms cancellation"
    finally:
        release.set()
        starter.join(timeout=2)
        cleaner.join(timeout=2)
        assert not starter.is_alive()
        assert not cleaner.is_alive()
    assert all(process.poll() is not None for process in case.processes)
    assert case.simulation_id not in SimulationRunner._processes
    assert SimulationRunner._cleanup_done is True


@pytest.mark.parametrize(("failure", "platform"), [
    ("updater", "twitter"), ("updater", "parallel"),
    ("script", "twitter"), ("popen", "twitter"), ("monitor", "twitter"),
])
def test_cancelled_startup_failure_does_not_cancel_the_next_force_start(interactive_run, monkeypatch, failure, platform):
    case = interactive_run
    entered, release = threading.Event(), threading.Event()
    case.allow_drain.set()
    original_create = module.ZepGraphMemoryManager.create_updater
    original_spawn = module.subprocess.Popen
    original_exists = module.os.path.exists
    original_thread = threading.Thread
    outcome = {}

    def failed_create(_cls, *args):
        entered.set()
        assert release.wait(timeout=2)
        raise RuntimeError("synthetic updater initialization failure")

    def failed_or_delayed_spawn(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        if failure == "popen":
            raise OSError("synthetic Popen failure")
        return original_spawn(*args, **kwargs)

    def missing_script(path):
        if str(path).endswith("run_twitter_simulation.py"):
            entered.set()
            assert release.wait(timeout=2)
            return False
        return original_exists(path)

    class BrokenMonitor:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("synthetic monitor startup failure")

        def is_alive(self):
            return False

    if failure == "updater":
        monkeypatch.setattr(module.ZepGraphMemoryManager, "create_updater", classmethod(failed_create))
    elif failure == "script":
        monkeypatch.setattr(module.os.path, "exists", missing_script)
    else:
        monkeypatch.setattr(module.subprocess, "Popen", failed_or_delayed_spawn)

    def start():
        try:
            case.start(platform, finish_platforms=False)
        except Exception as error:
            outcome["error"] = error

    starter = original_thread(target=start, daemon=True)
    starter.start()
    try:
        assert entered.wait(timeout=1)
        with pytest.raises(SimulationStopPending):
            SimulationRunner.stop_simulation(case.simulation_id)
        if failure == "monitor":
            monkeypatch.setattr(module.threading, "Thread", BrokenMonitor)
    finally:
        release.set()
        starter.join(timeout=2)
        assert not starter.is_alive()
        monkeypatch.setattr(module.threading, "Thread", original_thread)
        monkeypatch.setattr(module.os.path, "exists", original_exists)
        monkeypatch.setattr(module.subprocess, "Popen", original_spawn)
        monkeypatch.setattr(module.ZepGraphMemoryManager, "create_updater", original_create)
    assert "error" in outcome
    assert SimulationRunner.get_run_state(case.simulation_id).runner_status == RunnerStatus.FAILED
    assert all(process.poll() is not None for process in case.processes)
    leaked_after_failure = case.simulation_id in SimulationRunner._manual_stop_requests

    from flask import Flask
    from types import SimpleNamespace
    from app.api import simulation as api
    from app.services.simulation_manager import SimulationStatus

    metadata = SimpleNamespace(simulation_id=case.simulation_id, status=SimulationStatus.FAILED)
    monkeypatch.setattr(api, "SimulationManager", lambda: SimpleNamespace(
        get_simulation=lambda _sim_id: metadata, _save_simulation_state=lambda _state: None,
    ))
    monkeypatch.setattr(api, "_check_simulation_prepared", lambda _sim_id: (True, {}))
    before = len(case.processes)
    with Flask(__name__).test_request_context(
        "/api/simulation/start", method="POST",
        json={"simulation_id": case.simulation_id, "platform": platform, "force": True},
    ):
        result = api.start_simulation()
    response, status = result if isinstance(result, tuple) else (result, result.status_code)
    assert status == 200, response.get_json()
    assert response.json["success"] is True
    assert response.json["data"]["runner_status"] == "running", response.get_json()
    assert len(case.processes) == before + 1
    assert leaked_after_failure is False, "A safely reclaimed failed startup must release its own cancellation marker"


def test_old_startup_finalizer_cannot_clear_a_newer_claims_cancellation(interactive_run, monkeypatch):
    case = interactive_run
    newer = {}

    def replaced_start(_cls, state, _config, _platform, _rounds, _memory, _graph, claim):
        state.runner_status = RunnerStatus.FAILED
        new_claim = type(claim)()
        new_claim.cancel_requested = True
        new_state = SimulationRunState(case.simulation_id, runner_status=RunnerStatus.STOPPING)
        with SimulationRunner._finalization_lock(case.simulation_id):
            SimulationRunner._startup_claims[case.simulation_id] = new_claim
            SimulationRunner._run_states[case.simulation_id] = new_state
            SimulationRunner._manual_stop_requests.add(case.simulation_id)
        newer["claim"], newer["state"] = new_claim, new_state
        raise RuntimeError("older startup is finishing after ownership changed")

    monkeypatch.setattr(SimulationRunner, "_start_claimed_simulation", classmethod(replaced_start))
    with pytest.raises(RuntimeError, match="ownership changed"):
        SimulationRunner.start_simulation(case.simulation_id, platform="twitter")
    assert SimulationRunner._startup_claims[case.simulation_id] is newer["claim"]
    assert newer["claim"].cancel_requested is True
    assert SimulationRunner.get_run_state(case.simulation_id) is newer["state"]
    assert case.simulation_id in SimulationRunner._manual_stop_requests
