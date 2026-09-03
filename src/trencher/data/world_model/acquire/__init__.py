"""Acquire package exports."""

from trencher.data.world_model.acquire.bigquery import run_bigquery_acquisition
from trencher.data.world_model.acquire.gecko import run_gecko_acquisition
from trencher.data.world_model.acquire.local_corpus import run_local_acquisition
from trencher.data.world_model.acquire.rpc import run_rpc_acquisition

__all__ = [
    "run_bigquery_acquisition",
    "run_gecko_acquisition",
    "run_local_acquisition",
    "run_rpc_acquisition",
]
