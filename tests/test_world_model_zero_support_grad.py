import torch

from trencher.world_model.losses import MultiTaskLoss


def test_zero_support_term_does_not_backpropagate_pathological_gradient() -> None:
    loss_fn = MultiTaskLoss()

    source = torch.tensor(0.0, requires_grad=True)
    anchor = torch.tensor(1.0, requires_grad=True)

    # Forward value is finite, but sqrt'(0) is infinite. The historical
    # graph-connected `0 * loss` path therefore produced a NaN gradient.
    pathological_loss = torch.sqrt(source)
    assert torch.isfinite(pathological_loss)

    terms: dict[str, torch.Tensor] = {}
    support: dict[str, int] = {}

    total = anchor * 0.0
    result = loss_fn._add(
        terms,
        support,
        total,
        "zero_support",
        pathological_loss,
        1.0,
        0,
    )

    assert support["zero_support"] == 0
    assert terms["zero_support"].item() == 0.0
    assert not terms["zero_support"].requires_grad

    result.backward()

    assert source.grad is None
    assert anchor.grad is not None
    assert torch.isfinite(anchor.grad)
