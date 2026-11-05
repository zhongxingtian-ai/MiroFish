"""Prove startup works with the cloud SDK unavailable, regardless of this venv."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


BACKEND = Path(__file__).resolve().parents[1]


def test_app_health_and_all_service_imports_work_with_cloud_sdk_blocked(tmp_path):
    code = textwrap.dedent("""
        import importlib
        import os
        import pkgutil
        import socket
        import sys
        from pathlib import Path
        import dotenv

        sys.modules['zep_cloud'] = None
        dotenv.load_dotenv = lambda *args, **kwargs: False
        def deny_network(*args, **kwargs):
            raise AssertionError('Application startup must not make network requests')
        socket.socket.connect = deny_network
        socket.create_connection = deny_network
        from app import create_app
        from app.config import Config
        Config.UPLOAD_FOLDER = str(Path(os.environ['MEMORY_DATA_DIR']).parent / 'uploads')
        Config.DEBUG = False
        application = create_app()
        imported = []
        for package_name in ('app.services', 'app.api'):
            package = importlib.import_module(package_name)
            for item in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
                importlib.import_module(item.name)
                imported.append(item.name)
        assert imported
        assert Config.validate() == []
        response = application.test_client().get('/health')
        assert response.status_code == 200
        assert response.get_json() == {'status': 'ok', 'service': 'MiroFish Backend'}
        print('LOCAL_MEMORY_STARTUP_OK')
    """)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ZEP_")}
    environment.update(
        LLM_API_KEY="local-memory-test-only", LLM_BASE_URL="http://127.0.0.1:9/v1",
        MEMORY_DATA_DIR=str(tmp_path / "memory"), PYTHONIOENCODING="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, env=environment,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOCAL_MEMORY_STARTUP_OK" in result.stdout


@pytest.mark.parametrize("filename", ["requirements.txt", "pyproject.toml", "uv.lock"])
def test_dependency_declarations_and_lock_do_not_require_cloud_sdk(filename):
    assert "zep-cloud" not in (BACKEND / filename).read_text(encoding="utf-8").casefold()


@pytest.mark.parametrize("has_llm_key", [True, False])
def test_configuration_without_legacy_variables_requires_only_llm_key(monkeypatch, has_llm_key):
    from app.config import Config

    for name in tuple(os.environ):
        if name.startswith("ZEP_"):
            monkeypatch.delenv(name)
    for name in vars(Config):
        if name.startswith("ZEP_"):
            monkeypatch.setattr(Config, name, None)
    monkeypatch.setattr(Config, "DEBUG", False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "local-memory-test-only" if has_llm_key else None)
    expected = [] if has_llm_key else ["LLM_API_KEY 未配置"]
    assert Config.validate() == expected
