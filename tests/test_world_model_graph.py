"""Graph encoder causality / availability unit tests (synthetic tensors)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trencher.world_model.config import GraphEncoderConfig  # noqa: E402
from trencher.world_model.encoders.graph import CausalTemporalNeighborhoodEncoder  # noqa: E402


def test_graph_future_edge_does_not_affect_when_masked():
    cfg = GraphEncoderConfig(n_edge_channels=7, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    enc = CausalTemporalNeighborhoodEncoder(cfg)
    enc.eval()
    b, e, c = 2, 5, 7
    edge = torch.randn(b, e, c)
    times = torch.linspace(0, 1, e).unsqueeze(0).expand(b, -1)
    mask = torch.ones(b, e, dtype=torch.bool)
    mask[:, -1] = False  # last edge "future" / padded
    with torch.no_grad():
        out0 = enc(edge, times, mask=mask)
        edge2 = edge.clone()
        edge2[:, -1] += 10.0  # mutate masked future edge
        out1 = enc(edge2, times, mask=mask)
    assert torch.allclose(out0.pooled, out1.pooled, atol=1e-5)


def test_graph_pre_t_edge_can_affect():
    cfg = GraphEncoderConfig(n_edge_channels=7, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    enc = CausalTemporalNeighborhoodEncoder(cfg)
    enc.eval()
    edge = torch.randn(1, 4, 7)
    times = torch.arange(4).float().unsqueeze(0)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        out0 = enc(edge, times, mask=mask)
        edge2 = edge.clone()
        edge2[:, 0] += 5.0
        out1 = enc(edge2, times, mask=mask)
    assert not torch.allclose(out0.pooled, out1.pooled)


def test_graph_wallet_prior_influences_messages():
    cfg = GraphEncoderConfig(n_edge_channels=7, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    enc = CausalTemporalNeighborhoodEncoder(cfg)
    enc.eval()
    edge = torch.randn(1, 3, 7)
    times = torch.arange(3).float().unsqueeze(0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    w0 = torch.zeros(1, 32)
    w1 = torch.randn(1, 32)
    with torch.no_grad():
        a = enc(edge, times, mask=mask, wallet_prior_state=w0)
        b = enc(edge, times, mask=mask, wallet_prior_state=w1)
    assert not torch.allclose(a.pooled, b.pooled)


def test_graph_unavailable_uses_null():
    cfg = GraphEncoderConfig(n_edge_channels=7, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    enc = CausalTemporalNeighborhoodEncoder(cfg)
    enc.eval()
    edge = torch.randn(2, 3, 7)
    times = torch.arange(3).float().unsqueeze(0).expand(2, -1)
    mask = torch.ones(2, 3, dtype=torch.bool)
    avail = torch.tensor([1.0, 0.0])
    with torch.no_grad():
        out = enc(edge, times, mask=mask, available=avail)
    assert torch.allclose(out.pooled[1], enc.null_vec, atol=1e-5)
