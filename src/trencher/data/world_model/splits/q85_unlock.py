"""Fail-closed q85+ TEST unlock: only when TEST_ACCESSED.lock path is set and exists."""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "TRENCHER_Q85_TEST_UNLOCK"


def q85_test_unlocked() -> bool:
    """True iff env points at an existing ``TEST_ACCESSED.lock`` file."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return False
    lock = Path(raw)
    return lock.is_file() and lock.name == "TEST_ACCESSED.lock"


def set_q85_test_unlock(lock_path: Path) -> None:
    """Arm unlock after the lock file has been written."""
    lock_path = Path(lock_path)
    if not lock_path.is_file() or lock_path.name != "TEST_ACCESSED.lock":
        raise RuntimeError(f"refuse to unlock without TEST_ACCESSED.lock at {lock_path}")
    os.environ[ENV_VAR] = str(lock_path.resolve())


def clear_q85_test_unlock() -> None:
    os.environ.pop(ENV_VAR, None)
