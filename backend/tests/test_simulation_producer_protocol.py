"""Execute actual producer control flow with external model/environment replacements."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def producer_case(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    config = tmp_path / "simulation_config.json"
    config.write_text(json.dumps({
        "simulation_id": "producer-fixture", "llm_model": "fixture-model",
        "time_config": {"total_simulation_hours": 3, "minutes_per_round": 60,
                        "agents_per_hour_min": 1, "agents_per_hour_max": 1,
                        "off_peak_activity_multiplier": 1},
        "agent_configs": [{"agent_id": 1, "entity_name": "Fixture Agent", "active_hours": list(range(24)), "activity_level": 1}],
        "event_config": {"initial_posts": []},
    }), encoding="utf-8")
    (tmp_path / "twitter_profiles.csv").write_text("user_id,name" + chr(10) + "1,Fixture Agent" + chr(10), encoding="utf-8")
    (tmp_path / "reddit_profiles.json").write_text("[]", encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("LLM_", "OPENAI_", "ZEP_"))}
    environment.update(LLM_API_KEY="fixture-only", LLM_MODEL_NAME="fixture-model", PYTHONIOENCODING="utf-8")

    def run(script, platforms, flags):
        return subprocess.run([
            sys.executable, str(backend / "tests" / "producer_protocol_probe.py"),
            str(backend / "scripts" / f"run_{script}_simulation.py"), str(config), ",".join(platforms), *flags,
        ], cwd=backend, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=12)

    return SimpleNamespace(directory=tmp_path, run=run)


@pytest.mark.parametrize(("script", "platforms", "flags"), [
    ("twitter", ["twitter"], []), ("reddit", ["reddit"], []),
    ("parallel", ["twitter"], ["--twitter-only"]),
    ("parallel", ["reddit"], ["--reddit-only"]),
    ("parallel", ["twitter", "reddit"], []),
    ("twitter", ["twitter"], ["--no-wait"]), ("reddit", ["reddit"], ["--no-wait"]),
])
def test_real_producer_emits_actions_and_end_before_serving_interview(producer_case, script, platforms, flags):
    result = producer_case.run(script, platforms, flags)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads((producer_case.directory / "protocol-result.json").read_text(encoding="utf-8"))
    assert observed["errors"] == [], observed
    if "--no-wait" not in flags:
        assert observed["waiting"] is True
        assert observed["interview"] is True


@pytest.mark.parametrize("platform", ["twitter", "reddit"])
def test_direct_cli_rerun_rejects_existing_producer_records_without_overwriting_them(producer_case, platform):
    first = producer_case.run(platform, [platform], ["--no-wait"])
    assert first.returncode == 0, first.stdout + first.stderr
    database = producer_case.directory / (platform + "_simulation.db")
    actions = producer_case.directory / platform / "actions.jsonl"
    before = database.read_bytes(), actions.read_bytes()
    second = producer_case.run(platform, [platform], ["--no-wait"])
    assert second.returncode != 0, "Direct CLI must not append an old completion marker into a new run"
    assert "force" in second.stdout + second.stderr
    assert (database.read_bytes(), actions.read_bytes()) == before


@pytest.mark.parametrize("missing", ["twitter", "reddit"])
@pytest.mark.parametrize("entry", ["parallel", "single"])
def test_missing_selected_personas_fail_before_producing_partial_runs_or_waiting(producer_case, missing, entry):
    filename = "twitter_profiles.csv" if missing == "twitter" else "reddit_profiles.json"
    (producer_case.directory / filename).unlink()
    script = missing if entry == "single" else "parallel"
    platforms = [missing] if entry == "single" else ["twitter", "reddit"]
    result = producer_case.run(script, platforms, ["--no-wait"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert filename in result.stdout + result.stderr
    for artifact in ("simulation.log", "twitter_simulation.db", "reddit_simulation.db", "env_status.json",
                     "twitter/actions.jsonl", "reddit/actions.jsonl"):
        assert not (producer_case.directory / artifact).exists(), "Invalid selection created partial artifacts: " + artifact
