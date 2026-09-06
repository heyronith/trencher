"""Deterministic SSL input masking for event reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def apply_ssl_event_masking(
    batch: dict[str, Any],
    *,
    seed: int,
    mask_ratio: float = 0.15,
) -> dict[str, Any]:
    """Corrupt selected valid event cells; attach recon target/mask.

    - Preserves original values in ``ssl_recon_target`` (positions 0..T-2 to match
      the existing recon head over ``h[:, :-1]``).
    - Corrupts ``event_x`` only at deliberately masked valid cells (never padding).
    - Mask pattern is deterministic from ``seed`` + ``observation_id_hash``.
    """
    if "event_x" not in batch:
        return batch
    x = batch["event_x"]
    if not torch.is_tensor(x) or x.ndim != 3:
        return batch
    B, T, C = x.shape
    device = x.device
    dtype = x.dtype
    if T < 2:
        out = dict(batch)
        out["ssl_recon_target"] = torch.zeros(B, 0, C, dtype=dtype, device=device)
        out["ssl_recon_mask"] = torch.zeros(B, 0, C, dtype=torch.bool, device=device)
        return out

    event_mask = batch.get("event_mask")

    # Target matches recon head: positions 0..T-2
    target = x[:, :-1].detach().clone()
    recon_mask = torch.zeros(B, T - 1, C, dtype=torch.bool, device=device)
    x_corrupt = x.clone()

    oid = batch.get("observation_id_hash")
    if oid is None:
        oid_vals = [0] * B
    elif torch.is_tensor(oid):
        oid_vals = [int(v) for v in oid.detach().cpu().view(-1).tolist()]
        if len(oid_vals) < B:
            oid_vals = oid_vals + [0] * (B - len(oid_vals))
    else:
        oid_vals = [int(oid)] * B

    ratio = float(mask_ratio)
    for b in range(B):
        rng = np.random.default_rng(_mix_seed(seed, oid_vals[b]))
        for t in range(T - 1):
            valid = True
            if event_mask is not None:
                valid = bool(event_mask[b, t].item())
            if not valid:
                continue
            # Never mask using padding at t+1 either for next-step SSL alignment
            if event_mask is not None and not bool(event_mask[b, t + 1].item()):
                continue
            for c in range(C):
                if rng.random() < ratio:
                    recon_mask[b, t, c] = True
                    x_corrupt[b, t, c] = torch.zeros((), dtype=dtype, device=device)

    out = dict(batch)
    out["event_x"] = x_corrupt
    out["ssl_recon_target"] = target
    out["ssl_recon_mask"] = recon_mask
    return out


def _mix_seed(seed: int, oid_hash: int) -> int:
    # Stable 63-bit mix; independent of Python hash randomization.
    return int((int(seed) * 1_000_003) ^ (int(oid_hash) & 0x7FFFFFFFFFFFFFFF)) & 0x7FFFFFFF
