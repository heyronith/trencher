from __future__ import annotations

import torch

from trencher.world_model.config import SequenceEncoderConfig
from trencher.world_model.encoders._common import last_valid_token
from trencher.world_model.encoders.microstructure import MicrostructureEncoder
from trencher.world_model.encoders.ownership import OwnershipEncoder
from trencher.world_model.encoders.wallet_prior import WalletPriorEncoder


def _cfg(n_channels: int) -> SequenceEncoderConfig:
    return SequenceEncoderConfig(
        n_channels=n_channels,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_mult=2,
        dropout=0.0,
        max_len=16,
    )


def test_last_valid_token_accepts_fully_empty_batch() -> None:
    h = torch.empty(4, 0, 16, requires_grad=True)
    mask = torch.zeros(4, 0, dtype=torch.bool)

    pooled = last_valid_token(h, mask)

    assert pooled.shape == (4, 16)
    assert torch.equal(pooled, torch.zeros_like(pooled))
    assert pooled.requires_grad


def test_wallet_prior_empty_batch_uses_null_vector() -> None:
    enc = WalletPriorEncoder(_cfg(7)).eval()
    x = torch.empty(4, 0, 7)
    times = torch.empty(4, 0)
    mask = torch.zeros(4, 0, dtype=torch.bool)
    available = torch.zeros(4)

    out = enc(x, times, mask, available)

    expected = enc.null_vec.detach().unsqueeze(0).expand(4, -1)
    assert out.hidden.shape == (4, 0, 16)
    assert out.pooled.shape == (4, 16)
    assert torch.allclose(out.pooled, expected)
    assert torch.equal(out.available, available)


def test_microstructure_empty_batch_uses_null_vector() -> None:
    enc = MicrostructureEncoder(_cfg(11)).eval()
    x = torch.empty(4, 0, 11)
    times = torch.empty(4, 0)
    mask = torch.zeros(4, 0, dtype=torch.bool)
    available = torch.zeros(4)

    out = enc(x, times, mask, available)

    expected = enc.availability_emb.detach().unsqueeze(0).expand(4, -1)
    assert out.hidden.shape == (4, 0, 16)
    assert out.pooled.shape == (4, 16)
    assert torch.allclose(out.pooled, expected)
    assert torch.equal(out.available, available)


def test_ownership_empty_batch_uses_null_vector() -> None:
    enc = OwnershipEncoder(_cfg(9)).eval()
    x = torch.empty(4, 0, 9)
    times = torch.empty(4, 0)
    mask = torch.zeros(4, 0, dtype=torch.bool)
    available = torch.zeros(4)
    quality = torch.zeros(4)

    out = enc(x, times, mask, available, quality)

    expected = enc.null_vec.detach().unsqueeze(0).expand(4, -1)
    assert out.hidden.shape == (4, 0, 16)
    assert out.pooled.shape == (4, 16)
    assert torch.allclose(out.pooled, expected)
    assert torch.equal(out.available, available)
