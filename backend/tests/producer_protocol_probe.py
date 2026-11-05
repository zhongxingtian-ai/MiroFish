"""Isolated production-script driver; replaces only Camel/OASIS and network I/O."""

import asyncio
from contextlib import closing
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import runpy
import socket
import sqlite3
import sys
import threading
import time
from types import ModuleType, SimpleNamespace


def deny_network(*_args, **_kwargs):
    raise AssertionError("Producer protocol probe cannot access the network")


original_connect, original_pair = socket.socket.connect, socket.socketpair
self_pipe = threading.local()


def guarded_connect(sock, address):
    # Windows implements asyncio's private self-pipe with a loopback socketpair.
    # Permit only the standard socketpair construction, never model/HTTP traffic.
    if getattr(self_pipe, "building", False):
        return original_connect(sock, address)
    return deny_network()


def local_socketpair(*args, **kwargs):
    self_pipe.building = True
    try:
        return original_pair(*args, **kwargs)
    finally:
        self_pipe.building = False


socket.socket.connect = guarded_connect
socket.socketpair = local_socketpair
socket.create_connection = deny_network
dotenv = ModuleType("dotenv")
dotenv.load_dotenv = lambda *_args, **_kwargs: False
sys.modules["dotenv"] = dotenv

names = ("CREATE_POST", "LIKE_POST", "DISLIKE_POST", "REPOST", "FOLLOW", "DO_NOTHING",
         "QUOTE_POST", "CREATE_COMMENT", "LIKE_COMMENT", "DISLIKE_COMMENT", "SEARCH_POSTS",
         "SEARCH_USER", "TREND", "REFRESH", "MUTE", "INTERVIEW")
ActionType = Enum("ActionType", {name: name.lower() for name in names})


class ManualAction:
    def __init__(self, action_type, action_args):
        self.action_type, self.action_args = action_type, action_args


class LLMAction:
    pass


class Agent:
    name = "Fixture Agent"


agent = Agent()
graph = SimpleNamespace(get_agent=lambda _agent_id: agent, get_agents=lambda: [(1, agent)])
environments = []


class Environment:
    def __init__(self, agent_graph, database_path, **_kwargs):
        self.agent_graph, self.path, self.closed = agent_graph, database_path, False
        environments.append(self)

    async def reset(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TABLE trace(user_id INTEGER, action TEXT, info TEXT, created_at TEXT)")

    async def step(self, actions):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            for action in actions.values():
                action = action[0] if isinstance(action, list) else action
                interview = isinstance(action, ManualAction) and action.action_type == ActionType.INTERVIEW
                kind = "interview" if interview else "create_post"
                info = {"response": "fixture interview answer", "prompt": action.action_args.get("prompt")} if interview else {"content": "fixture producer action"}
                connection.execute("INSERT INTO trace VALUES (?,?,?,?)", (1, kind, json.dumps(info), datetime.now().isoformat()))

    async def close(self):
        self.closed = True


async def generate_graph(**_kwargs):
    return graph


camel = ModuleType("camel")
camel.__path__ = []
models, types = ModuleType("camel.models"), ModuleType("camel.types")
models.ModelFactory = SimpleNamespace(create=lambda **_kwargs: object())
types.ModelPlatformType = SimpleNamespace(OPENAI="openai")
oasis = ModuleType("oasis")
oasis.ActionType, oasis.ManualAction, oasis.LLMAction = ActionType, ManualAction, LLMAction
oasis.DefaultPlatformType = SimpleNamespace(TWITTER="twitter", REDDIT="reddit")
oasis.make = lambda **kwargs: Environment(**kwargs)
oasis.generate_twitter_agent_graph = generate_graph
oasis.generate_reddit_agent_graph = generate_graph
sys.modules.update({"camel": camel, "camel.models": models, "camel.types": types, "oasis": oasis})


script, config = Path(sys.argv[1]), Path(sys.argv[2])
platforms = sys.argv[3].split(",")
flags = sys.argv[4:]
directory = config.parent
errors = []
observed = {"waiting": False, "interview": False}


def wait_json(path, deadline):
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.005)
    raise AssertionError("Bounded IPC wait expired: " + path.name)


def send(kind, args):
    command_id = "fixture-" + kind
    commands = directory / "ipc_commands"
    commands.mkdir(exist_ok=True)
    (commands / (command_id + ".json")).write_text(json.dumps({
        "command_id": command_id, "command_type": kind, "args": args,
    }), encoding="utf-8")
    return wait_json(directory / "ipc_responses" / (command_id + ".json"), time.monotonic() + 3)


def verify_produced_logs():
    for platform in platforms:
        path = directory / platform / "actions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        ends = [row for row in rows if row.get("event_type") == "simulation_end"]
        actions = [row for row in rows if "event_type" not in row]
        if len(ends) != 1 or ends[0]["total_rounds"] != 1:
            errors.append(platform + ": missing correct production simulation_end")
        if not any(row.get("action_args", {}).get("content") == "fixture producer action" for row in actions):
            errors.append(platform + ": actual environment action not logged")


def command_client():
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = wait_json(directory / "env_status.json", deadline)
            if status.get("status") == "alive":
                break
            time.sleep(0.005)
        else:
            raise AssertionError("Production script did not enter interview waiting mode")
        observed["waiting"] = True
        verify_produced_logs()
        response = send("interview", {"agent_id": 1, "prompt": "fixture question", "platform": platforms[0]})
        observed["interview"] = response.get("status") == "completed" and "fixture interview answer" in json.dumps(response)
        if not observed["interview"]:
            errors.append("Production interview protocol did not return the environment response")
    except Exception as error:
        errors.append(str(error))
    finally:
        try:
            send("close_env", {})
        except Exception as error:
            errors.append(str(error))


controller = None
if "--no-wait" not in flags:
    controller = threading.Thread(target=command_client, daemon=True)
    controller.start()
sys.argv = [str(script), "--config", str(config), "--max-rounds", "1", *flags]
# Match python path/to/script.py: runpy itself does not add the entry directory.
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
if controller:
    controller.join(timeout=1)
    assert not controller.is_alive()
else:
    verify_produced_logs()
assert environments and all(env.closed for env in environments)
assert len(environments) == len(platforms)
(directory / "protocol-result.json").write_text(json.dumps({"errors": errors, **observed}), encoding="utf-8")
