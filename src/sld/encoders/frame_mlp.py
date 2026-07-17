"""Per-frame MLP encoder: maps ONE observation frame to a latent z.

Markovian-observation baseline: assumes a single frame carries enough
information to determine the latent state (no partial observability). See
``window_mlp`` for the windowed variant used when that assumption fails.
"""

from __future__ import annotations

import torch
from torch import nn


class FrameMLPEncoder(nn.Module):
    """Encodes ONE observation frame ``x -> z`` via a plain MLP.

    Markovian-observation baseline: assumes a single frame is sufficient to
    determine the latent state. Hidden layers use GELU activations (smooth,
    no dead-unit risk unlike ReLU; the exact choice is not load-bearing for
    this baseline).

    Args:
        d_x: observation frame dimension.
        d_z: latent dimension.
        hidden: sizes of the hidden layers, default ``(64, 64)``.
    """

    def __init__(self, d_x: int, d_z: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        dims = (d_x, *hidden, d_z)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(..., d_x)`` -> ``z``: ``(..., d_z)``, any leading dims.

        Leading dims are flattened into a batch, the MLP is applied, and the
        result is reshaped back to the original leading dims.
        """
        if x.shape[-1] != self.d_x:
            raise ValueError(f"expected last dim {self.d_x}, got {x.shape[-1]}")
        leading = x.shape[:-1]
        z = self.net(x.reshape(-1, self.d_x))
        return z.reshape(*leading, self.d_z)
