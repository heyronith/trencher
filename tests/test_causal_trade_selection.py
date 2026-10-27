"""Causal vs leaky signature selection for microstructure refetch."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trencher.data.world_model.acquire.local_corpus import (
    select_causal_trades_before,
    select_stratified_trades_for_mint,
)


def _trade(sig: str, ts: float, *, sol: float = 1.0, is_buy: bool = True) -> dict:
    return {
        "tx_signature": sig,
        "event_time": datetime.fromtimestamp(ts, tz=UTC),
        "sol_amount": sol,
        "is_buy": is_buy,
        "curve_pct_depleted": 0.1 * sol,
        "mint": "Mint111",
        "user_wallet": "Wallet111",
    }


def test_select_stratified_requires_explicit_leak_flag() -> None:
    trades = [_trade("a", 1.0), _trade("b", 2.0)]
    with pytest.raises(ValueError, match="leaky"):
        select_stratified_trades_for_mint(trades, cap=2)
    out = select_stratified_trades_for_mint(
        trades, cap=2, allow_lifetime_selection=True
    )
    assert len(out) == 2


def test_causal_selection_ignores_trades_after_T() -> None:
    """Appending arbitrary trades after T must not change the selected set."""
    t = 100.0
    before = [_trade(f"pre{i}", float(i), sol=float(i)) for i in range(1, 21)]
    after = [
        _trade("post_huge", 200.0, sol=1e9),
        _trade("post_first_buy", 150.0, sol=0.01),
        _trade("post_curve", 180.0, sol=50.0),
    ]
    k = 16
    selected_pre = select_causal_trades_before(before, t, k)
    selected_post = select_causal_trades_before(before + after, t, k)
    assert [r["tx_signature"] for r in selected_pre] == [
        r["tx_signature"] for r in selected_post
    ]
    assert all(_event_before(r, t) for r in selected_post)
    assert len(selected_post) == k
    # Last K strictly before T in time order.
    assert [r["tx_signature"] for r in selected_post] == [
        f"pre{i}" for i in range(5, 21)
    ]


def test_leaky_stratified_changes_when_post_T_extremes_added() -> None:
    """Old lifetime selector must FAIL the causal invariant (regression witness)."""
    before = [_trade(f"pre{i}", float(i), sol=float(i)) for i in range(1, 21)]
    after = [_trade("post_huge", 200.0, sol=1e9)]
    k = 8
    pre = select_stratified_trades_for_mint(
        before, cap=k, allow_lifetime_selection=True
    )
    post = select_stratified_trades_for_mint(
        before + after, cap=k, allow_lifetime_selection=True
    )
    pre_sigs = {r["tx_signature"] for r in pre}
    post_sigs = {r["tx_signature"] for r in post}
    assert pre_sigs != post_sigs
    assert "post_huge" in post_sigs


def _event_before(row: dict, t: float) -> bool:
    et = row["event_time"]
    assert isinstance(et, datetime)
    return et.timestamp() < t
