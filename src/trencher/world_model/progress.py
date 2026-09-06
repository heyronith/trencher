"""Monotonic 1–100 progress emitter for deep world-model training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class ProgressEmitter:
    """Emit ``[tag] k/100`` from real completed/planned work units."""

    def __init__(
        self,
        tag: str,
        *,
        status_path: Path | None = None,
        stream: TextIO | None = None,
        phase_weights: dict[str, float] | None = None,
    ) -> None:
        self.tag = tag
        self.status_path = Path(status_path) if status_path else None
        self.stream = stream
        self.t0 = time.time()
        self.last = -1
        self.meta: dict[str, Any] = {}
        # Default: SSL 20%, broad 55%, rich 25% of the 1–100 bar
        self.phase_weights = phase_weights or {"ssl": 0.20, "broad": 0.55, "rich": 0.25}

    def phase_progress(
        self,
        phase: str,
        completed: float,
        planned: float,
        *,
        message: str = "",
        force: bool = False,
        **extra: Any,
    ) -> int:
        """Map a single phase's local progress into the global 1–100 scale."""
        weights = self.phase_weights
        order = [p for p in ("ssl", "broad", "rich") if p in weights]
        if phase not in weights:
            return self.update(completed, planned, message=message, force=force, **extra)
        prior = 0.0
        for p in order:
            if p == phase:
                break
            prior += float(weights[p])
        planned = max(float(planned), 1e-9)
        local = min(1.0, max(0.0, float(completed) / planned))
        overall = prior + local * float(weights[phase])
        # Scale to 1..100
        step = max(1, min(100, int(overall * 100)))
        if local >= 1.0 and phase == order[-1]:
            step = 100
        return self.emit(step, message=message or f"phase={phase}", force=force, **extra)

    def update(
        self,
        completed: float,
        planned: float,
        *,
        message: str = "",
        force: bool = False,
        **extra: Any,
    ) -> int:
        planned = max(float(planned), 1e-9)
        frac = min(1.0, max(0.0, float(completed) / planned))
        if completed <= 0:
            step = 1 if self.last < 0 else self.last
        else:
            step = max(1, min(100, int(frac * 100)))
            if frac >= 1.0:
                step = 100
        return self.emit(step, message=message, force=force or step != self.last, **extra)

    def emit(
        self,
        step: int,
        *,
        message: str = "",
        force: bool = False,
        **extra: Any,
    ) -> int:
        step = max(1, min(100, int(step)))
        # Monotonic: never go backwards unless force and completing
        if step < self.last and not force:
            step = self.last
        if not force and step == self.last and step not in (1, 100):
            return step
        self.last = step
        self.meta.update(extra)
        elapsed = time.time() - self.t0
        line = f"[{self.tag}] {step}/100"
        if message:
            line += f" {message}"
        line += f" (elapsed={elapsed:.1f}s)"
        print(line, flush=True, file=self.stream)
        if self.status_path is not None:
            payload = {
                "tag": self.tag,
                "progress": step,
                "message": message,
                "elapsed_sec": elapsed,
                **self.meta,
            }
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return step

    def complete(self, message: str = "COMPLETE", **extra: Any) -> int:
        return self.emit(100, message=message, force=True, **extra)
