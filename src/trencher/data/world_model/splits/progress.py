"""Progress helper for long split/loader builds."""

from __future__ import annotations

import logging
import sys
import time
from typing import TextIO

logger = logging.getLogger(__name__)


class Progress100:
    """Emit integer progress in 1..100 with a message."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.t0 = time.time()
        self.last = -1

    def __call__(self, step: int, message: str = "") -> None:
        step = max(0, min(100, int(step)))
        if step == self.last and step not in (0, 100):
            return
        self.last = step
        elapsed = time.time() - self.t0
        line = f"PROGRESS {step}/100 {message} (elapsed={elapsed:.1f}s)".rstrip()
        print(line, file=self.stream, flush=True)
        logger.info("%s", line)
