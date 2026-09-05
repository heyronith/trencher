"""Chronological probability calibration (never on final val scoring rows)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PlattCalibrator:
    """Simple logistic (Platt) calibration fit on OOF/chronological holdout."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        # Fit P(y=1|s) = 1/(1+exp(-(a*s+b))) via Newton on tiny design
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        a, b = 1.0, 0.0
        for _ in range(50):
            z = a * s + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            w = p * (1 - p)
            w = np.clip(w, 1e-6, None)
            # gradient / hessian for [a,b]
            da = np.sum((p - y) * s)
            db = np.sum(p - y)
            haa = np.sum(w * s * s)
            hab = np.sum(w * s)
            hbb = np.sum(w)
            det = haa * hbb - hab * hab
            if abs(det) < 1e-12:
                break
            a -= (hbb * da - hab * db) / det
            b -= (-hab * da + haa * db) / det
        self.a, self.b = float(a), float(b)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        z = self.a * s + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlattCalibrator:
        return cls(a=float(d["a"]), b=float(d["b"]))


def chronological_calibration_split(
    times: np.ndarray, *, calib_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Last calib_frac of training timeline for calibration; rest for fitting."""
    t = np.asarray(times, dtype=np.float64)
    order = np.argsort(t)
    n = t.size
    n_cal = max(1, int(round(n * calib_frac)))
    cal_idx = order[-n_cal:]
    fit_idx = order[:-n_cal] if n > n_cal else order
    return fit_idx, cal_idx
