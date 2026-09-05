"""External data package exports."""

from __future__ import annotations

__all__ = ["HistoricalDownloadResult", "download_historical"]


def __getattr__(name: str):
    if name in ("HistoricalDownloadResult", "download_historical"):
        from trencher.data.historical import HistoricalDownloadResult, download_historical

        return {
            "HistoricalDownloadResult": HistoricalDownloadResult,
            "download_historical": download_historical,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
