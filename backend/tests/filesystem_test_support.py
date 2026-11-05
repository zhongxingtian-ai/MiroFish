"""Directory redirects used only inside pytest-owned temporary directories."""

import os
from pathlib import Path
import subprocess


_PROCESS = subprocess.Popen


def directory_redirect(link: Path, target: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # NTFS junction creation is unprivileged; unlike directory symlinks it
        # does not require Developer Mode. No shell deletion is performed.
        # Runner fixtures replace Popen for simulation producers; this is a real,
        # bounded filesystem-only helper in the pytest temporary directory.
        with _PROCESS(["cmd", "/c", "mklink", "/J", str(link), str(target)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                output, errors = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            assert process.returncode == 0, output + errors
    else:
        link.symlink_to(target, target_is_directory=True)
