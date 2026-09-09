from __future__ import annotations

import pytest

from trencher.world_model.launch_hardened import (
    _successful_batches_by_worker,
    _worker_batch_capacities,
)


def test_worker_batch_capacities_include_partial_batches() -> None:
    assert _worker_batch_capacities([128, 129, 0], 64) == [2, 3, 0]


def test_resume_offset_distributes_round_robin_before_exhaustion() -> None:
    capacities = [45869, 45869]
    assert sum(capacities) == 91738
    assert _successful_batches_by_worker(capacities, 90000) == [45000, 45000]


def test_resume_offset_handles_worker_exhaustion() -> None:
    capacities = [2, 4]
    assert _successful_batches_by_worker(capacities, 5) == [2, 3]
    assert _successful_batches_by_worker(capacities, 6) == [2, 4]


def test_resume_offset_rejects_out_of_epoch_range() -> None:
    with pytest.raises(ValueError, match="invalid batch offset"):
        _successful_batches_by_worker([2, 3], 6)


def test_stage_b_observed_epoch_modulo() -> None:
    epoch_batches = 91738
    assert 90000 % epoch_batches == 90000
    assert 92000 % epoch_batches == 262
    assert 100000 % epoch_batches == 8262
