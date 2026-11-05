"""Shared, non-creating boundaries for ID-owned local data and child files."""

import os
from pathlib import Path, PureWindowsPath
import re


class StoragePathError(ValueError):
    """An ID or redirect would cross a local data boundary."""


IPC_DIRECTORIES = ("ipc_commands", "ipc_responses")
PLATFORM_PROFILE_FILES = {"twitter": "twitter_profiles.csv", "reddit": "reddit_profiles.json"}


def validate_storage_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value, re.ASCII) is None:
        raise StoragePathError("文件系统 ID 必须为 1–64 个 ASCII 字母、数字、下划线或横线")
    if PureWindowsPath(value).is_reserved():
        raise StoragePathError("文件系统 ID 不能使用 Windows 保留名称")
    return value


def _checked(path: Path, root: Path) -> Path:
    try:
        redirected = path.is_symlink() or getattr(path, "is_junction", lambda: False)()
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise StoragePathError("无法安全解析本地数据路径") from error
    if redirected or resolved != path or not resolved.is_relative_to(root):
        raise StoragePathError("本地数据路径不能使用符号链接、目录联接或越出所属目录")
    return path


def safe_child_path(directory, *parts) -> Path:
    """Resolve fixed/relative children without creating or following redirects."""
    base = Path(os.path.abspath(os.path.expanduser(directory)))
    _checked(base, base)
    current = base
    for part in parts:
        value = str(part)
        if not value or value.startswith(("/", chr(92))) or PureWindowsPath(value).drive:
            raise StoragePathError("本地数据子路径必须为相对路径")
        for component in value.replace(chr(92), "/").split("/"):
            if component in {"", ".", ".."} or ":" in component or PureWindowsPath(component).is_reserved():
                raise StoragePathError("本地数据子路径含无效目录或名称")
            current = _checked(current / component, base)
    return current


def storage_path(directory, value: str, *parts, suffix="") -> Path:
    validate_storage_id(value)
    if suffix and re.fullmatch(r"[.][A-Za-z0-9]+", suffix) is None:
        raise StoragePathError("本地数据文件后缀无效")
    return safe_child_path(directory, value + suffix, *parts)


def validate_tree(directory) -> Path:
    """Preflight a recursive deletion before removing any owned file."""
    path = Path(os.path.abspath(directory))
    _checked(path, path.parent)
    for parent, directories, files in os.walk(path, followlinks=False):
        relative = Path(parent).relative_to(path)
        for name in directories + files:
            safe_child_path(path, *relative.parts, name)
    return path


def ipc_message_paths(directory) -> list[Path]:
    """Preflight the flat JSON queues without reading message contents."""
    messages = []
    for name in IPC_DIRECTORIES:
        queue = safe_child_path(directory, name)
        if not queue.exists():
            continue
        if not queue.is_dir():
            raise StoragePathError("IPC 队列路径必须为目录")
        for entry in queue.iterdir():
            if entry.suffix == ".json":
                path = safe_child_path(queue, entry.name)
                if not path.is_file():
                    raise StoragePathError("IPC 临时消息必须为普通文件")
                messages.append(path)
    return messages
