"""IDs and directory redirects cannot access files outside a managed storage root."""

import builtins
from contextlib import closing
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from app import create_app
from app.config import Config
from app.models.project import Project, ProjectManager, ProjectStatus
from app.services.report_agent import ReportManager
from app.services.simulation_manager import SimulationManager
from app.services.simulation_runner import SimulationRunner, SimulationRunState
from filesystem_test_support import directory_redirect


@pytest.fixture
def storage_case(tmp_path, monkeypatch):
    roots = SimpleNamespace(
        simulation=tmp_path / "storage" / "simulations",
        project=tmp_path / "storage" / "projects",
        report=tmp_path / "storage" / "reports",
        outside=tmp_path / "outside",
    )
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(roots.simulation))
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(roots.simulation))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(roots.simulation))
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "storage"))
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(roots.project))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(roots.report))
    for name in ("_run_states", "_processes", "_monitor_threads", "_stdout_files", "_stderr_files", "_finalization_locks", "_startup_claims", "_graph_memory_enabled"):
        monkeypatch.setattr(SimulationRunner, name, {})
    monkeypatch.setattr(SimulationRunner, "_manual_stop_requests", set())
    monkeypatch.setattr(Config, "LLM_API_KEY", "path-test-only")
    roots.outside.mkdir()
    (roots.outside / "state.json").write_text(json.dumps({
        "simulation_id": "outside", "project_id": "project", "graph_id": "graph", "status": "ready",
    }), encoding="utf-8")
    (roots.outside / "simulation_config.json").write_text('{"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}', encoding="utf-8")
    (roots.outside / "run_state.json").write_text('{"runner_status": "completed"}', encoding="utf-8")
    project = Project(project_id="outside", name="Foreign material", status=ProjectStatus.CREATED,
                      created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat())
    (roots.outside / "project.json").write_text(json.dumps(project.to_dict()), encoding="utf-8")
    (roots.outside / "meta.json").write_text(json.dumps({
        "report_id": "outside", "simulation_id": "outside", "graph_id": "graph",
        "simulation_requirement": "foreign report", "status": "completed", "markdown_content": "OUTSIDE MATERIAL",
    }), encoding="utf-8")
    return roots


def _directory(case, family, value):
    if family == "simulation":
        return SimulationManager()._get_simulation_dir(value)
    if family == "runner":
        return SimulationRunner._load_run_state(value)
    if family == "project":
        return ProjectManager._get_project_dir(value)
    return ReportManager._get_report_folder(value)


@pytest.mark.parametrize("family", ["simulation", "runner", "project", "report"])
@pytest.mark.parametrize("value", ["../outside", "../../outside", ".." + chr(92) + ".." + chr(92) + "outside",
    "a/b", "a" + chr(92) + "b", "", "x" * 65, "CON", "NUL", "中文", None, 123, [], {}])
def test_filesystem_ids_are_rejected_before_access(storage_case, family, value):
    before = {path.name: path.read_bytes() for path in storage_case.outside.iterdir()}
    with pytest.raises(ValueError):
        _directory(storage_case, family, value)
    assert {path.name: path.read_bytes() for path in storage_case.outside.iterdir()} == before


@pytest.mark.parametrize("family", ["simulation", "runner", "project", "report"])
def test_absolute_id_is_rejected_without_reading_or_creating(storage_case, family):
    with pytest.raises(ValueError):
        _directory(storage_case, family, str(storage_case.outside))


@pytest.mark.parametrize("value", ["a", "A_Z-09", "x" * 64])
def test_valid_id_boundaries_do_not_create_directories_when_reading(storage_case, value):
    manager = SimulationManager()
    assert manager.get_simulation(value) is None
    assert not storage_case.simulation.exists()
    assert Path(ProjectManager._get_project_dir(value)).name == value
    assert Path(ReportManager._get_report_folder(value)).name == value


@pytest.mark.parametrize("family", ["simulation", "runner", "project", "report"])
def test_foreign_directory_redirects_are_rejected_even_with_cached_state(storage_case, family):
    root = storage_case.simulation if family == "runner" else getattr(storage_case, family)
    directory_redirect(root / "alias", storage_case.outside)
    before = {path.name: path.read_bytes() for path in storage_case.outside.iterdir()}
    if family == "runner":
        SimulationRunner._run_states["alias"] = SimulationRunState("alias")
        with pytest.raises(ValueError):
            SimulationRunner.get_run_state("alias")
    elif family == "simulation":
        manager = SimulationManager()
        manager._simulations["alias"] = object()
        with pytest.raises(ValueError):
            manager.get_simulation("alias")
    else:
        with pytest.raises(ValueError):
            _directory(storage_case, family, "alias")
    assert {path.name: path.read_bytes() for path in storage_case.outside.iterdir()} == before


@pytest.mark.parametrize("family", ["project", "report"])
def test_deleting_a_redirect_cannot_delete_foreign_contents(storage_case, family):
    root = getattr(storage_case, family)
    directory_redirect(root / "alias", storage_case.outside)
    before = {path.name: path.read_bytes() for path in storage_case.outside.iterdir()}
    delete = ProjectManager.delete_project if family == "project" else ReportManager.delete_report
    with pytest.raises(ValueError):
        delete("alias")
    assert {path.name: path.read_bytes() for path in storage_case.outside.iterdir()} == before


@pytest.mark.parametrize("target", ["simulation", "project", "report", "wrong-type"])
def test_supported_http_routes_reject_bad_ids_without_reading_foreign_files(storage_case, monkeypatch, target):
    reads = []
    original_open = builtins.open

    def watch_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(storage_case.outside):
            reads.append(str(file))
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watch_open)
    client = create_app().test_client()
    escaped = ".." + chr(92) + ".." + chr(92) + "outside"
    if target == "simulation":
        response = client.get("/api/simulation/" + quote(escaped, safe="") + "/config/download")
    elif target == "project":
        response = client.get("/api/graph/project/" + quote(escaped, safe=""))
    elif target == "report":
        response = client.get("/api/report/" + quote(escaped, safe=""))
    else:
        response = client.post("/api/simulation/close-env", json={"simulation_id": 123})
    assert response.status_code in {400, 404}, response.get_json()
    assert reads == []


@pytest.mark.parametrize("endpoint", ["posts", "comments"])
def test_direct_database_routes_cannot_use_a_platform_path_to_read_foreign_sqlite(storage_case, endpoint):
    prefix = storage_case.outside / "private"
    database = storage_case.outside / "private_simulation.db"
    table = "post" if endpoint == "posts" else "comment"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(f"CREATE TABLE {table}(content TEXT, created_at TEXT)")
        connection.execute(f"INSERT INTO {table} VALUES (?,?)", ("PRIVATE OUTSIDE CONTENT", "fixture time"))
    before = database.read_bytes()
    response = create_app().test_client().get(f"/api/simulation/sim-safe/{endpoint}", query_string={"platform": str(prefix)})
    assert response.status_code in {400, 404}, response.get_json()
    assert "PRIVATE OUTSIDE CONTENT" not in response.get_data(as_text=True)
    assert database.read_bytes() == before


@pytest.mark.parametrize("name", ["ReportLogger", "ReportConsoleLogger"])
def test_report_log_writers_reject_bad_ids_before_any_output_creation(storage_case, monkeypatch, name):
    from app.services import report_agent

    logger_type = getattr(report_agent, name)
    touched = []
    monkeypatch.setattr(logger_type, "_ensure_log_file", lambda self: touched.append(self.log_file_path))
    if name == "ReportConsoleLogger":
        monkeypatch.setattr(logger_type, "_setup_file_handler", lambda self: None)
    escaped = ".." + chr(92) + ".." + chr(92) + "outside"
    with pytest.raises(ValueError):
        logger_type(escaped)
    assert touched == []


@pytest.mark.parametrize("family", ["project", "report"])
@pytest.mark.parametrize("redirect", [False, True])
def test_owned_tree_delete_preserves_normal_behavior_and_rejects_nested_redirects(storage_case, family, redirect):
    if family == "project":
        item = ProjectManager.create_project("Owned project")
        identifier = item.project_id
        directory = Path(ProjectManager._get_project_dir(identifier))
        delete = ProjectManager.delete_project
        assert ProjectManager.get_project(identifier).name == "Owned project"
    else:
        from app.services.report_agent import Report, ReportStatus
        identifier = "report-owned"
        ReportManager.save_report(Report(
            report_id=identifier, simulation_id="sim-owned", graph_id="graph-owned",
            simulation_requirement="fixture", status=ReportStatus.COMPLETED, markdown_content="Owned report",
        ))
        directory = Path(ReportManager._get_report_folder(identifier))
        delete = ReportManager.delete_report
        assert ReportManager.get_report(identifier).markdown_content == "Owned report"
    if redirect:
        directory_redirect(directory / "foreign", storage_case.outside)
        before = {path.name: path.read_bytes() for path in storage_case.outside.iterdir()}
        with pytest.raises(ValueError):
            delete(identifier)
        assert directory.exists()
        assert {path.name: path.read_bytes() for path in storage_case.outside.iterdir()} == before
    else:
        assert delete(identifier) is True
        assert not directory.exists()
