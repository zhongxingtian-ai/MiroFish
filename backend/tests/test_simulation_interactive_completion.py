"""Bounded, model-free lifecycle tests for a live post-simulation interview process."""

import json
import threading
import time
from types import SimpleNamespace

from flask import Flask
import pytest

from app.api import report as report_api
from app.models.project import ProjectStatus
from app.services import simulation_runner as runner_module
from app.services.simulation_manager import SimulationStatus
from app.services.simulation_runner import RunnerStatus, SimulationRunner


class InterviewProcess:
    pid = 700001

    def __init__(self):
        self.exited = threading.Event()
        self.returncode = None

    def poll(self):
        return self.returncode if self.exited.is_set() else None

    def finish(self, code=0):
        self.returncode = code
        self.exited.set()


@pytest.fixture
def interactive_run(tmp_path, monkeypatch):
    runner = SimulationRunner
    for name in ("_run_states", "_processes", "_monitor_threads", "_stdout_files",
                 "_stderr_files", "_action_queues", "_graph_memory_enabled", "_finalization_locks"):
        monkeypatch.setattr(runner, name, {})
    monkeypatch.setattr(runner, "_manual_stop_requests", set())
    monkeypatch.setattr(runner, "_startup_claims", {}, raising=False)
    monkeypatch.setattr(runner, "_cleanup_done", False)
    monkeypatch.setattr(runner, "RUN_STATE_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "SCRIPTS_DIR", str(tmp_path / "scripts"))
    simulation_id = "sim-interactive-fixture"
    sim_dir = tmp_path / "runs" / simulation_id
    sim_dir.mkdir(parents=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (sim_dir / "simulation_config.json").write_text(json.dumps({
        "time_config": {"total_simulation_hours": 1, "minutes_per_round": 60},
    }), encoding="utf-8")
    for platform in ("twitter", "reddit", "parallel"):
        (scripts / f"run_{platform}_simulation.py").write_text("pass" + chr(10), encoding="utf-8")

    case = SimpleNamespace(
        simulation_id=simulation_id, sim_dir=sim_dir, processes=[], monitors=[],
        drain_entered=threading.Event(), allow_drain=threading.Event(),
        terminal=threading.Event(), drained=0, drain_error=None, actions=[],
        updaters={}, synced=[], terminated=[], producer_end_platforms=(),
    )
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(
        sleep=lambda _seconds: threading.Event().wait(0.005),
        time=time.time, monotonic=time.monotonic,
    ))

    def spawn(*_args, **_kwargs):
        process = InterviewProcess()
        case.processes.append(process)
        case.process = process
        for platform in case.producer_end_platforms:
            end_platform(platform)
        return process

    monkeypatch.setattr(runner_module.subprocess, "Popen", spawn)

    def terminate(_cls, process, sim_id, **_kwargs):
        assert sim_id == simulation_id
        case.terminated.append(process)
        process.finish()

    monkeypatch.setattr(runner, "_terminate_process", classmethod(terminate))
    monkeypatch.setattr(runner, "_sync_simulation_status", classmethod(
        lambda _cls, sim_id, status, error=None: case.synced.append((status, error))
    ))
    save = runner._save_run_state

    def save_state(_cls, state):
        save(state)
        if state.runner_status in {RunnerStatus.COMPLETED, RunnerStatus.FAILED, RunnerStatus.STOPPED}:
            case.terminal.set()

    monkeypatch.setattr(runner, "_save_run_state", classmethod(save_state))
    memory = runner_module.ZepGraphMemoryManager

    def create_updater(_cls, sim_id, graph_id):
        updater = SimpleNamespace(add_activity_from_dict=lambda data, platform: case.actions.append((data, platform)))
        case.updaters[sim_id] = updater
        return updater

    def stop_updater(_cls, sim_id):
        case.drained += 1
        case.drain_entered.set()
        assert case.allow_drain.wait(timeout=2), "Test did not release the synthetic drain"
        if case.drain_error:
            raise RuntimeError(case.drain_error)
        case.updaters.pop(sim_id, None)

    monkeypatch.setattr(memory, "create_updater", classmethod(create_updater))
    monkeypatch.setattr(memory, "stop_updater", classmethod(stop_updater))
    monkeypatch.setattr(memory, "get_updater", classmethod(lambda _cls, sim_id: case.updaters.get(sim_id)))
    monkeypatch.setattr(memory, "get_simulation_ids", classmethod(lambda _cls: list(case.updaters)))

    class IPC:
        def __init__(self, directory):
            assert directory == str(sim_dir)

        def check_env_alive(self):
            return case.process.poll() is None

        def send_interview(self, **_kwargs):
            return SimpleNamespace(status=SimpleNamespace(value="completed"), result="fixture interview answer", timestamp="fixture time")

        def send_close_env(self, **_kwargs):
            case.process.finish()
            return SimpleNamespace(status=SimpleNamespace(value="completed"), result={"closed": True}, timestamp="fixture time")

    monkeypatch.setattr(runner_module, "SimulationIPCClient", IPC)

    def end_platform(platform):
        directory = sim_dir / platform
        directory.mkdir(exist_ok=True)
        rows = [
            {"round": 1, "agent_id": 1, "agent_name": "Fixture", "action_type": "CREATE_POST", "action_args": {"content": "fixture tail"}, "success": True},
            {"event_type": "round_end", "round": 1, "simulated_hours": 1},
            {"event_type": "simulation_end", "total_rounds": 1, "total_actions": 1},
        ]
        (directory / "actions.jsonl").write_text(chr(10).join(json.dumps(row) for row in rows) + chr(10), encoding="utf-8")

    def start(platform, *, finish_platforms=True, initial_platforms=()):
        enabled = ("twitter", "reddit") if platform == "parallel" else (platform,)
        case.producer_end_platforms = (enabled if finish_platforms else ()) + tuple(initial_platforms)
        case.state = runner.start_simulation(simulation_id, platform=platform, enable_graph_memory_update=True, graph_id="graph-interactive-fixture")
        case.monitor = runner._monitor_threads.get(simulation_id)
        if case.monitor is not None:
            case.monitors.append(case.monitor)
        case.handle = runner._stdout_files.get(simulation_id)
        return case.state

    def report():
        simulation = SimpleNamespace(simulation_id=simulation_id, project_id="project-fixture", graph_id="graph-interactive-fixture", status=SimulationStatus.COMPLETED)
        project = SimpleNamespace(graph_id="graph-interactive-fixture", status=ProjectStatus.GRAPH_COMPLETED, simulation_requirement="fixture report")
        monkeypatch.setattr(report_api, "SimulationManager", lambda: SimpleNamespace(get_simulation=lambda _sim_id: simulation))
        monkeypatch.setattr(report_api.ProjectManager, "get_project", classmethod(lambda _cls, _project_id: project))
        monkeypatch.setattr(report_api.ReportManager, "get_report_by_simulation", classmethod(lambda _cls, _sim_id: SimpleNamespace(report_id="fixture-report", status=report_api.ReportStatus.COMPLETED)))
        with Flask(__name__).test_request_context("/api/report/generate", method="POST", json={"simulation_id": simulation_id}):
            result = report_api.generate_report()
        response, status = result if isinstance(result, tuple) else (result, result.status_code)
        return response.get_json(), status

    case.start, case.end_platform, case.report = start, end_platform, report
    try:
        yield case
    finally:
        case.allow_drain.set()
        for process in case.processes:
            process.finish()
        monitors = set(case.monitors) | set(runner._monitor_threads.values())
        for monitor in monitors:
            monitor.join(timeout=2)
            assert not monitor.is_alive(), "Synthetic monitor must never outlive its bounded fixture"
        for handle in runner._stdout_files.values():
            handle.close()


def _completed(case):
    assert case.drain_entered.wait(timeout=1), "All platform end events must drain before the interview process exits"
    case.allow_drain.set()
    assert case.terminal.wait(timeout=1)
    assert case.state.runner_status == RunnerStatus.COMPLETED


@pytest.mark.parametrize(("platform", "later_exit_code"), [("twitter", 0), ("reddit", 0), ("parallel", 0), ("parallel", 7)])
def test_alive_interview_process_publishes_completion_only_after_drain(interactive_run, platform, later_exit_code):
    case = interactive_run
    case.start(platform)
    assert case.drain_entered.wait(timeout=1), "Memory drain must not wait for process exit"
    assert case.state.runner_status == RunnerStatus.STOPPING
    assert case.report()[1] == 409
    assert case.process.poll() is None
    assert not case.handle.closed
    case.allow_drain.set()
    assert case.terminal.wait(timeout=1)
    assert case.state.runner_status == RunnerStatus.COMPLETED
    assert case.report()[1] == 200
    persisted = json.loads((case.sim_dir / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["runner_status"] == "completed"
    assert persisted["enabled_platforms"] == (["twitter", "reddit"] if platform == "parallel" else [platform])
    assert SimulationRunner.check_env_alive(case.simulation_id) is True
    assert SimulationRunner.interview_agent(case.simulation_id, 1, "fixture prompt")["success"] is True
    assert SimulationRunner._processes[case.simulation_id] is case.process
    assert SimulationRunner._monitor_threads[case.simulation_id].is_alive()
    assert not case.handle.closed
    assert len(case.actions) == (2 if platform == "parallel" else 1)
    assert case.drained == 1
    case.process.finish(later_exit_code)
    case.monitor.join(timeout=1)
    assert not case.monitor.is_alive()
    assert case.handle.closed
    assert case.simulation_id not in SimulationRunner._processes
    assert case.simulation_id not in SimulationRunner._action_queues
    assert case.simulation_id not in SimulationRunner._stdout_files
    assert case.state.runner_status == RunnerStatus.COMPLETED


def test_slow_second_platform_cannot_be_inferred_disabled_from_a_missing_log(interactive_run):
    case = interactive_run
    case.start("parallel", finish_platforms=False, initial_platforms=("twitter",))
    deadline = time.monotonic() + 1
    while not case.state.twitter_completed and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    assert case.state.twitter_completed
    assert not (case.sim_dir / "reddit" / "actions.jsonl").exists()
    assert SimulationRunner._check_all_platforms_completed(case.state) is False
    assert case.state.runner_status == RunnerStatus.RUNNING
    assert case.drained == 0
    assert case.report()[1] == 409
    case.end_platform("reddit")
    _completed(case)


@pytest.mark.parametrize("platform", ["twitter", "parallel"])
def test_failed_drain_blocks_reports_but_keeps_live_process_owned(interactive_run, platform):
    case = interactive_run
    case.drain_error = "synthetic ingestion incomplete"
    case.allow_drain.set()
    case.start(platform)
    assert case.terminal.wait(timeout=1), "A failed drain must publish failure while the environment is alive"
    assert case.state.runner_status == RunnerStatus.FAILED
    assert "synthetic ingestion incomplete" in case.state.error
    assert case.report()[1] == 409
    assert case.process.poll() is None
    assert SimulationRunner._processes[case.simulation_id] is case.process
    assert not case.handle.closed
    assert case.drained == 1
    case.process.finish()
    case.monitor.join(timeout=1)
    assert not case.monitor.is_alive()
    assert case.state.runner_status == RunnerStatus.FAILED
    assert case.drained == 1


@pytest.mark.parametrize("close_method", ["stop", "close_env", "shutdown"])
def test_completed_live_environment_is_reaped_before_log_cleanup_and_restart(interactive_run, close_method):
    case = interactive_run
    case.start("parallel")
    _completed(case)
    old_process, old_handle = case.process, case.handle
    if close_method == "stop":
        SimulationRunner.stop_simulation(case.simulation_id)
    elif close_method == "close_env":
        assert SimulationRunner.close_simulation_env(case.simulation_id)["success"] is True
    else:
        SimulationRunner.cleanup_all_simulations()
    case.monitor.join(timeout=1)
    assert not case.monitor.is_alive()
    assert old_process.poll() is not None
    assert old_handle.closed
    assert case.simulation_id not in SimulationRunner._processes
    assert case.simulation_id not in SimulationRunner._monitor_threads
    assert case.simulation_id not in SimulationRunner._stdout_files
    assert case.drained == 1
    assert case.state.runner_status == RunnerStatus.COMPLETED
    assert SimulationRunner.cleanup_simulation_logs(case.simulation_id)["success"] is True
    case.start("twitter", finish_platforms=False)
    assert len(case.processes) == 2
    assert case.state.runner_status == RunnerStatus.RUNNING
    assert SimulationRunner._processes[case.simulation_id] is not old_process
    assert SimulationRunner._cleanup_done is False
    SimulationRunner.cleanup_all_simulations()
    assert case.process.poll() is not None
    assert case.simulation_id not in SimulationRunner._processes


def test_direct_restart_and_log_cleanup_cannot_orphan_a_completed_live_process(interactive_run):
    case = interactive_run
    case.start("twitter")
    _completed(case)
    with pytest.raises(ValueError):
        SimulationRunner.start_simulation(case.simulation_id, platform="twitter")
    with pytest.raises((ValueError, runner_module.SimulationStopPending)):
        SimulationRunner.cleanup_simulation_logs(case.simulation_id)
    assert len(case.processes) == 1
    assert case.process.poll() is None
    assert not case.handle.closed


@pytest.mark.parametrize("platform", ["twitter", "parallel"])
def test_zero_exit_without_all_platform_end_events_is_not_completed(interactive_run, platform):
    case = interactive_run
    case.allow_drain.set()
    case.start(platform, finish_platforms=False)
    case.process.finish()
    case.monitor.join(timeout=1)
    assert not case.monitor.is_alive()
    assert case.state.runner_status == RunnerStatus.FAILED
    assert case.report()[1] == 409
    assert case.handle.closed


def test_force_restart_api_closes_the_completed_interview_process_first(interactive_run, monkeypatch):
    case = interactive_run
    case.start("twitter")
    _completed(case)
    from app.api import simulation as simulation_api

    simulation = SimpleNamespace(
        simulation_id=case.simulation_id, status=SimulationStatus.COMPLETED,
        project_id="project-fixture", graph_id="graph-interactive-fixture",
    )
    monkeypatch.setattr(simulation_api, "SimulationManager", lambda: SimpleNamespace(
        get_simulation=lambda _sim_id: simulation, _save_simulation_state=lambda _state: None,
    ))
    monkeypatch.setattr(simulation_api, "_check_simulation_prepared", lambda _sim_id: (True, {}))
    starts = []

    def start_after_cleanup(_cls, **kwargs):
        assert case.process.poll() is not None
        assert case.handle.closed
        assert case.simulation_id not in SimulationRunner._processes
        starts.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"runner_status": "running"})

    monkeypatch.setattr(SimulationRunner, "start_simulation", classmethod(start_after_cleanup))
    with Flask(__name__).test_request_context(
        "/api/simulation/start", method="POST",
        json={"simulation_id": case.simulation_id, "force": True, "platform": "twitter"},
    ):
        result = simulation_api.start_simulation()
    response, status = result if isinstance(result, tuple) else (result, result.status_code)
    assert status == 200, response.get_json()
    assert len(starts) == 1


@pytest.mark.parametrize("platform", ["twitter", "reddit"])
def test_disabled_platform_extra_log_is_not_ingested_or_used_for_completion(interactive_run, platform):
    case = interactive_run
    other = "reddit" if platform == "twitter" else "twitter"
    case.start(platform, initial_platforms=(other,))
    _completed(case)
    assert [name for _data, name in case.actions] == [platform]
    assert getattr(case.state, other + "_completed") is False


def test_monitor_error_retains_live_process_and_logs_for_explicit_stop(interactive_run, monkeypatch):
    case = interactive_run

    def fail(_cls, *_args):
        raise RuntimeError("synthetic monitor failure")

    monkeypatch.setattr(SimulationRunner, "_read_platform_logs", classmethod(fail))
    case.start("twitter", finish_platforms=False)
    assert case.terminal.wait(timeout=1)
    case.monitor.join(timeout=1)
    assert case.state.runner_status == RunnerStatus.FAILED
    assert case.process.poll() is None
    assert case.report()[1] == 409
    assert SimulationRunner._processes[case.simulation_id] is case.process
    assert not case.handle.closed
    case.allow_drain.set()
    SimulationRunner.stop_simulation(case.simulation_id)
    assert case.process.poll() is not None
    assert case.handle.closed
    assert case.simulation_id not in SimulationRunner._processes


def test_stop_api_preserves_completed_result_when_closing_its_interview_environment(interactive_run, monkeypatch):
    case = interactive_run
    case.start("twitter")
    _completed(case)
    from app.api import simulation as simulation_api

    simulation = SimpleNamespace(status=SimulationStatus.COMPLETED, error=None)
    monkeypatch.setattr(simulation_api, "SimulationManager", lambda: SimpleNamespace(
        get_simulation=lambda _sim_id: simulation, _save_simulation_state=lambda _state: None,
    ))
    with Flask(__name__).test_request_context(
        "/api/simulation/stop", method="POST", json={"simulation_id": case.simulation_id},
    ):
        result = simulation_api.stop_simulation()
    response, status = result if isinstance(result, tuple) else (result, result.status_code)
    assert status == 200, response.get_json()
    assert response.json["data"]["runner_status"] == "completed"
    assert simulation.status == SimulationStatus.COMPLETED
    assert case.process.poll() is not None
    assert case.handle.closed


def test_log_cleanup_rejects_a_startup_claim_before_process_publication(interactive_run):
    case = interactive_run
    state = runner_module.SimulationRunState(case.simulation_id, runner_status=RunnerStatus.STARTING)
    SimulationRunner._save_run_state(state)
    assert case.simulation_id not in SimulationRunner._processes
    with pytest.raises(runner_module.SimulationStopPending):
        SimulationRunner.cleanup_simulation_logs(case.simulation_id)
    assert (case.sim_dir / "run_state.json").exists()
