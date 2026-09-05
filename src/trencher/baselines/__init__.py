"""Causal tabular baselines for frozen world-model supervision."""

from __future__ import annotations

__all__ = [
    "PermanentTestAccessError",
    "build_baseline_contract_v1",
    "build_baseline_features_v1",
    "publish_baseline_artifacts",
    "refuse_test_split",
]


def __getattr__(name: str):
    if name == "PermanentTestAccessError":
        from trencher.baselines.split_guard import PermanentTestAccessError

        return PermanentTestAccessError
    if name == "refuse_test_split":
        from trencher.baselines.split_guard import refuse_test_split

        return refuse_test_split
    if name == "build_baseline_features_v1":
        from trencher.baselines.feature_cache import build_baseline_features_v1

        return build_baseline_features_v1
    if name in ("build_baseline_contract_v1", "publish_baseline_artifacts"):
        from trencher.baselines import audit as _audit

        return getattr(_audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
