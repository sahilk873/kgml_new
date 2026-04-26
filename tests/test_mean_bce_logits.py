from __future__ import annotations

import torch

from kgml_new.training.eval import mean_bce_logits_link_prediction


def test_mean_bce_logits_flat_and_grouped_negatives():
    pos = torch.tensor([2.0, -1.0])
    neg_flat = torch.tensor([0.5, -0.5, 1.0, -2.0])
    loss_flat = mean_bce_logits_link_prediction(pos, neg_flat)
    assert loss_flat.ndim == 0
    assert 0.0 < float(loss_flat) < 10.0

    neg_group = torch.tensor([[0.5, -0.5], [1.0, -2.0]])
    loss_g = mean_bce_logits_link_prediction(pos, neg_group)
    assert torch.allclose(loss_flat, loss_g)
