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
from sld.predictors.koopman_ssm import _make_mlp


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


class MLPContextBackbone(nn.Module):
    """``kappa_mlp``: the MEMORYLESS context producer, ``c_t = MLP(h_t)``.

    The ``ContextBackbone`` (``structured.py``) matching this family's own
    lack of recurrent state: ``PredictorState.m`` stays ``None`` for the whole
    trajectory, ``absorb`` returns the state it was given, and the context of
    step ``t`` is a function of ``h_t`` alone. That is the point of this arm
    of the 3x2 factorial -- it is the structured cell WITHOUT memory, so the
    two other backbones are read against it.

    ``u_t`` is invisible to this backbone at every step, not merely at its own
    step: it never enters ``m`` because there is no ``m``.
    """

    def __init__(self, d_h: int, width: int = 64, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.d_context = width
        self.net = _make_mlp(d_h, width, (width,), dtype)

    def context_step(self, h: Tensor, state: PredictorState) -> Tensor:
        return self.net(h)

    def absorb(self, h: Tensor, u: Tensor, state: PredictorState) -> PredictorState:
        # memoryless by construction: nothing to fold in, state passes through
        return state
