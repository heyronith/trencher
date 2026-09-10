"""Stage-C wallet-prior total sequence cap wiring regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from trencher.world_model.batch_adapter import example_to_tensors
from trencher.world_model.config import load_world_model_config
from trencher.world_model.resumable_loader import FastSkipFrozenDataset
from trencher.world_model.v4_loader import FrozenV4IterableDataset, make_frozen_dataloader


def _seed_assignment_dir(split_root: Path) -> None:
    assign = split_root / "observation_assignments_broad"
    assign.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "observation_id": "oid0",
                "base_split": "train",
                "obs_in_split_interval": True,
            }
        ]
    )
    pq.write_table(table, assign / "part-00000.parquet")


def _wallet_example(n_rows: int, *, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # Distinct increasing first-channel markers so retained rows are identifiable.
    wallet = np.zeros((n_rows, 7), dtype=np.float32)
    wallet[:, 0] = np.arange(n_rows, dtype=np.float32)
    wallet[:, 1:] = rng.normal(size=(n_rows, 6)).astype(np.float32)
    return {
        "meta": {"observation_id": f"oid_{n_rows}", "base_split": "train"},
        "sequences": {"wallet_prior": wallet},
        "modality_masks": {"mask_wallet_prior": 1.0},
        "targets": {},
        "features": {},
        "causal_ok": True,
    }


def test_wallet_prior_over_cap_tensorized_to_exact_config_max_len() -> None:
    ex = _wallet_example(1000)
    out = example_to_tensors(ex, max_len={"wallet_prior": 256})
    assert int(out["wallet_x"].shape[0]) == 256
    # Existing _get_seq policy: keep the tail of the concatenated sequence.
    assert torch.allclose(
        out["wallet_x"][:, 0],
        torch.arange(1000 - 256, 1000, dtype=torch.float32),
    )


def test_wallet_prior_under_cap_unchanged() -> None:
    ex = _wallet_example(128)
    out = example_to_tensors(ex, max_len={"wallet_prior": 256})
    assert int(out["wallet_x"].shape[0]) == 128
    assert torch.allclose(
        out["wallet_x"][:, 0],
        torch.arange(128, dtype=torch.float32),
    )


def test_wallet_prior_tail_cap_is_deterministic() -> None:
    ex = _wallet_example(512, seed=7)
    a = example_to_tensors(ex, max_len={"wallet_prior": 256})
    b = example_to_tensors(ex, max_len={"wallet_prior": 256})
    assert torch.equal(a["wallet_x"], b["wallet_x"])
    assert torch.allclose(
        a["wallet_x"][:, 0],
        torch.arange(512 - 256, 512, dtype=torch.float32),
    )


def test_frozen_v4_tensorize_example_honors_seq_max_len(tmp_path: Path) -> None:
    _seed_assignment_dir(tmp_path)
    ds = FrozenV4IterableDataset(
        world_root=tmp_path,
        split_root=tmp_path,
        mode="final_train",
        population="broad",
        seq_max_len={"wallet_prior": 256},
    )
    item = ds.tensorize_example(_wallet_example(900))
    assert int(item["wallet_x"].shape[0]) == 256
    assert ds.seq_max_len == {"wallet_prior": 256}


def test_make_frozen_dataloader_wires_wallet_prior_cap(tmp_path: Path) -> None:
    _seed_assignment_dir(tmp_path)
    loader = make_frozen_dataloader(
        world_root=tmp_path,
        split_root=tmp_path,
        mode="final_train",
        population="broad",
        batch_size=1,
        num_workers=0,
        seq_max_len={"wallet_prior": 256},
    )
    ds = loader.dataset
    assert isinstance(ds, FrozenV4IterableDataset)
    assert ds.seq_max_len == {"wallet_prior": 256}
    item = ds.tensorize_example(_wallet_example(400))
    assert int(item["wallet_x"].shape[0]) == 256


def test_fast_skip_uses_base_wallet_prior_cap(tmp_path: Path) -> None:
    _seed_assignment_dir(tmp_path)
    base = FrozenV4IterableDataset(
        world_root=tmp_path,
        split_root=tmp_path,
        mode="final_train",
        population="broad",
        seq_max_len={"wallet_prior": 256},
    )
    fast = FastSkipFrozenDataset(base, skip_rows_by_worker=[0])
    # Exercise the same tensorize path FastSkip emit() uses.
    item = fast.base.tensorize_example(_wallet_example(700))
    assert int(item["wallet_x"].shape[0]) == 256


def test_launch_passes_only_wallet_prior_cap_from_frozen_config() -> None:
    """Production launch must wire cfg.wallet_prior.max_len and not invent new caps."""
    cfg = load_world_model_config(
        Path("configs/world_model/world_model_candidate_v2.yaml")
    )
    assert int(cfg.wallet_prior.max_len) == 256
    assert cfg.config_hash() == (
        "c2e7c0e51ef91df87333615f5e6cd66e0baed891fbabcb907c5f5fa55a6d8945"
    )

    import trencher.world_model.launch as launch

    source = Path(launch.__file__).read_text(encoding="utf-8")
    assert 'seq_max_len={"wallet_prior": int(cfg.wallet_prior.max_len)}' in source
    # Do not silently enforce other unused modality caps in this repair.
    assert "cfg.microstructure.max_len" not in source
    assert "cfg.ownership.max_len" not in source


def test_config_yaml_untouched_hash() -> None:
    cfg = load_world_model_config(
        Path("configs/world_model/world_model_candidate_v2.yaml")
    )
    assert cfg.config_hash() == (
        "c2e7c0e51ef91df87333615f5e6cd66e0baed891fbabcb907c5f5fa55a6d8945"
    )
