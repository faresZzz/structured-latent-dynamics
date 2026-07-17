"""Residual MLP predictor: the generic one-step baseline, no structural prior.

``h_next = h + MLP(concat([h, u]))``. Plain feed-forward network with GELU
activations (arbitrary but declared choice over tanh; both are common, GELU
is the default elsewhere in torch's own transformer blocks so it is reused
here for consistency with ``transformer.py``). No recurrent state:
``PredictorState.m`` stays ``None`` forever, and ``local_operator`` stays
the inherited ``None`` (this family exposes no structured linear operator,
so the audit reads it by autograd on the MLP itself).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sld.predictors.base import PredictorModule, PredictorState


class MLPPredictor(PredictorModule):
    """Residual one-step MLP predictor: ``h_next = h + MLP([h, u])``.

    Args:
        d_h: latent state dimension.
        d_u: action dimension.
        hidden: sizes of the hidden layers between the ``d_h + d_u`` input
            and the ``d_h`` output.
    """

    def __init__(self, d_h: int, d_u: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.d_h = d_h
        self.d_u = d_u
        dims = (d_h + d_u, *hidden, d_h)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]:
        h_next = h + self.net(torch.cat([h, u], dim=-1))
        return h_next, state
