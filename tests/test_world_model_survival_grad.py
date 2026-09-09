"""BF16 survival-loss gradient finiteness regressions."""

from __future__ import annotations

import torch

from trencher.world_model.losses import censor_aware_survival_loss


def test_bfloat16_one_minus_eps_rounds_to_one() -> None:
    """BF16 cannot represent the historical clamp upper bound 1 - 1e-6."""
    upper = torch.tensor(1.0 - 1e-6, dtype=torch.bfloat16)
    assert upper.item() == 1.0


def _legacy_unsafe_survival_nll(
    hazard_logits: torch.Tensor,
    event: torch.Tensor,
    time_idx: torch.Tensor,
) -> torch.Tensor:
    """Pre-repair formula: log1p on all bins, then mask.

    Under BF16 this evaluates log1p(-1.0) on unused saturated bins, which is
    finite after masked_fill but yields NaN in Log1PBackward.
    """
    hazards = torch.sigmoid(hazard_logits).clamp(1e-6, 1.0 - 1e-6)
    _, h = hazards.shape
    arange = torch.arange(h, device=hazards.device).view(1, -1)
    t = time_idx.view(-1, 1).clamp(0, h - 1)
    surv_mask = arange < t
    log_surv = torch.log1p(-hazards).masked_fill(~surv_mask, 0.0).sum(dim=-1)
    gather_h = hazards.gather(1, t).squeeze(1)
    log_event = torch.log(gather_h)
    return -(log_surv + event.to(dtype=hazards.dtype) * log_event).mean()


def test_legacy_bf16_masked_saturation_log1p_backward_is_nan() -> None:
    """Document the historical finite-forward / NaN-backward failure mode."""
    logits = torch.tensor(
        [[0.0, 20.0, 20.0, 20.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    event = torch.tensor([1.0])
    time_idx = torch.tensor([0])

    loss = _legacy_unsafe_survival_nll(logits, event, time_idx)
    assert torch.isfinite(loss)

    # CPU ClampBackward can hide NaNs in the final .grad, so detect at source.
    with torch.autograd.detect_anomaly():
        try:
            loss.backward()
        except RuntimeError as exc:
            assert "nan" in str(exc).lower()
            return
    raise AssertionError(
        "expected Log1PBackward NaN under BF16 masked saturation; "
        f"got finite-looking grad={logits.grad}"
    )


def test_censor_aware_survival_bfloat16_saturation_has_finite_gradient() -> None:
    logits = torch.tensor(
        [[0.0, 20.0, 20.0, 20.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    event = torch.tensor([1.0])
    time_idx = torch.tensor([0])
    mask = torch.tensor([1.0])
    sample_mask = torch.tensor([1.0])

    loss = censor_aware_survival_loss(
        logits,
        event,
        time_idx,
        mask,
        sample_mask,
    )

    assert torch.isfinite(loss)

    with torch.autograd.detect_anomaly():
        loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_censor_aware_survival_post_event_extremes_have_finite_gradient() -> None:
    """Extreme logits after time_idx must not corrupt gradients."""
    logits = torch.tensor(
        [[0.25, 0.0, 40.0, -40.0, 40.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    event = torch.tensor([1.0])
    time_idx = torch.tensor([1])

    loss = censor_aware_survival_loss(
        logits,
        event,
        time_idx,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )
    assert torch.isfinite(loss)

    with torch.autograd.detect_anomaly():
        loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def _reference_survival_loss(
    hazard_logits: torch.Tensor,
    event: torch.Tensor,
    time_idx: torch.Tensor,
) -> torch.Tensor:
    hazards = torch.sigmoid(hazard_logits).clamp(
        1e-6,
        1.0 - 1e-6,
    )

    _, h = hazards.shape

    arange = torch.arange(
        h,
        device=hazards.device,
    ).view(1, -1)

    t = time_idx.view(-1, 1).clamp(0, h - 1)

    surv_mask = arange < t

    log_surv = (
        torch.log1p(-hazards)
        .masked_fill(~surv_mask, 0.0)
        .sum(dim=-1)
    )

    gather_h = hazards.gather(1, t).squeeze(1)

    log_event = torch.log(gather_h)

    return -(
        log_surv
        + event.to(dtype=hazards.dtype) * log_event
    ).mean()


def test_censor_aware_survival_fp32_matches_reference_formula() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 16, dtype=torch.float32) * 0.5
    event = torch.randint(0, 2, (8,), dtype=torch.float32)
    time_idx = torch.randint(0, 16, (8,), dtype=torch.long)

    repaired = censor_aware_survival_loss(logits, event, time_idx)
    reference = _reference_survival_loss(logits, event, time_idx)

    assert torch.isfinite(repaired)
    assert torch.isfinite(reference)
    assert torch.allclose(repaired, reference, rtol=1e-5, atol=1e-6)
