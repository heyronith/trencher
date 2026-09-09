"""Regression: Stage-C execution heads must not require fee-complete flags."""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.training.targets_batch import build_target_payload
from trencher.world_model.batch_adapter import _pack_valid


def _assignment() -> dict[str, Any]:
    return {
        "observation_id": "obs_exec_mask",
        "observation_time_t": 1_700_000_000.0,
        "base_split": "train",
        "eligible_graduation_tte": True,
        "eligible_explosion_tte": True,
    }


def _base_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "observation_id": "obs_exec_mask",
        "time_to_graduation_sec": None,
        "time_to_graduation_status": "modality_unavailable",
        "time_to_explosion_sec": None,
        "explosion_event_observed": False,
        "explosion_already_started_at_t": False,
        "explosion_right_censored": False,
        "exec_0p1sol_execution_target_available": 0,
        "exec_0p1sol_available": True,
        "exec_0p1sol_status": "available",
        "exec_0p1sol_gross_return": None,
        "exec_0p1sol_entry_impact_adjusted_return": None,
        "exec_0p1sol_curve_impact_adjusted_return": None,
        "exec_0p1sol_net_return": None,
        "exec_0p1sol_slippage": None,
    }
    row.update(overrides)
    return row


def _payload(target_row: dict[str, Any]) -> dict[str, Any]:
    return build_target_payload(_assignment(), target_row, None, q70=2000.0, q85=3000.0)


def _trainable(pack: dict[str, Any]) -> bool:
    value, mask = _pack_valid(pack)
    return (not pack.get("is_missing")) and mask > 0.0 and value == value


def test_fee_flag_false_but_gross_valid_is_trainable() -> None:
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_available=True,
            exec_0p1sol_status="available",
            exec_0p1sol_gross_return=0.12,
        )
    )
    gross = payload["execution_gross_0p1sol"]
    assert gross["is_missing"] is False
    assert gross["split_eligible"] is True
    assert gross["value"] == 0.12
    assert _trainable(gross)


def test_entry_impact_present_with_fee_flag_false_is_trainable() -> None:
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_gross_return=0.12,
            exec_0p1sol_entry_impact_adjusted_return=0.08,
        )
    )
    entry = payload["execution_entry_impact_0p1sol"]
    assert entry["is_missing"] is False
    assert _trainable(entry)


def test_curve_impact_present_with_fee_flag_false_is_trainable() -> None:
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_gross_return=0.12,
            exec_0p1sol_curve_impact_adjusted_return=0.05,
        )
    )
    curve = payload["execution_curve_impact_0p1sol"]
    assert curve["is_missing"] is False
    assert _trainable(curve)


def test_missing_entry_or_curve_stays_masked_even_if_gross_available() -> None:
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_gross_return=0.12,
            exec_0p1sol_entry_impact_adjusted_return=None,
            exec_0p1sol_curve_impact_adjusted_return=None,
        )
    )
    assert payload["execution_gross_0p1sol"]["is_missing"] is False
    assert payload["execution_entry_impact_0p1sol"]["is_missing"] is True
    assert payload["execution_curve_impact_0p1sol"]["is_missing"] is True
    assert not _trainable(payload["execution_entry_impact_0p1sol"])
    assert not _trainable(payload["execution_curve_impact_0p1sol"])


def test_net_return_remains_untrained_even_if_numeric_value_present() -> None:
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_gross_return=0.12,
            exec_0p1sol_net_return=0.07,
        )
    )
    net = payload["execution_net_0p1sol"]
    assert net["split_eligible"] is False
    assert net["is_missing"] is True
    assert not _trainable(net)


def test_completely_unavailable_execution_row_remains_masked() -> None:
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_available=False,
            exec_0p1sol_status="reference_price_unavailable",
            exec_0p1sol_gross_return=None,
            exec_0p1sol_entry_impact_adjusted_return=None,
            exec_0p1sol_curve_impact_adjusted_return=None,
        )
    )
    for key in (
        "execution_gross_0p1sol",
        "execution_entry_impact_0p1sol",
        "execution_curve_impact_0p1sol",
        "execution_net_0p1sol",
        "execution_slippage_0p1sol",
    ):
        assert payload[key]["is_missing"] is True
        assert not _trainable(payload[key])


def test_execution_target_available_must_not_gate_fee_independent_heads() -> None:
    """Guard against reintroducing the fee-flag dependency."""
    # Fee flag false, size available, all fee-independent values present.
    payload = _payload(
        _base_row(
            exec_0p1sol_execution_target_available=0,
            exec_0p1sol_available=True,
            exec_0p1sol_status="available",
            exec_0p1sol_gross_return=-0.03,  # 0.0-adjacent / negative still valid
            exec_0p1sol_entry_impact_adjusted_return=-0.04,
            exec_0p1sol_curve_impact_adjusted_return=-0.05,
            exec_0p1sol_net_return=-0.06,
            exec_0p1sol_slippage=0.01,
        )
    )
    assert _trainable(payload["execution_gross_0p1sol"])
    assert _trainable(payload["execution_entry_impact_0p1sol"])
    assert _trainable(payload["execution_curve_impact_0p1sol"])
    assert not _trainable(payload["execution_net_0p1sol"])
    assert not _trainable(payload["execution_slippage_0p1sol"])
