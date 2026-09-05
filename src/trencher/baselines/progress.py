"""Monotonic 1–100 progress emitters for baseline jobs."""

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
    ) -> None:
        self.tag = tag
        self.status_path = Path(status_path) if status_path else None
        self.stream = stream
        self.t0 = time.time()
        self.last = -1
        self.meta: dict[str, Any] = {}

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
        # Map to 1..100 while work is in progress; 0 only before start.
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


class BoostingProgressCallback:
    """Map boosting iterations → 1..100 for LGBM/XGB-style callbacks."""

    def __init__(self, emitter: ProgressEmitter, planned_iterations: int) -> None:
        self.emitter = emitter
        self.planned = max(int(planned_iterations), 1)
        self.best_metric: float | None = None
        self.best_iteration: int | None = None

    def __call__(self, env: Any) -> None:
        # LightGBM callback env
        it = int(getattr(env, "iteration", 0)) + 1
        evaluation = getattr(env, "evaluation_result_list", None) or []
        msg = f"iter={it}/{self.planned}"
        if evaluation:
            # [(data_name, eval_name, result, higher_better), ...]
            name, metric, value, higher = evaluation[-1][:4]
            msg += f" {name}:{metric}={value:.6g}"
            if self.best_metric is None:
                self.best_metric = float(value)
                self.best_iteration = it
            else:
                improved = (
                    float(value) > self.best_metric
                    if higher
                    else float(value) < self.best_metric
                )
                if improved:
                    self.best_metric = float(value)
                    self.best_iteration = it
        self.emitter.update(
            it,
            self.planned,
            message=msg,
            current_iteration=it,
            best_iteration=self.best_iteration,
            best_metric=self.best_metric,
        )
