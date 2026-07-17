"""Windowed MLP encoder: maps the ``k`` most recent observation frames to a
latent z.

Used when a single frame does not carry enough information to determine the
latent state (partial observability) -- see ``frame_mlp`` for the Markovian
baseline that assumes one frame is enough.

Window layout (read this before touching Wave 6's partial-observation code
-- a wrong window order here is a SILENT bug, the MLP still runs and
produces numbers, just conditioned on the wrong temporal order):

- Input ``x`` has shape ``(..., k, d_x)``. The ``k`` axis is ordered
  OLDEST FIRST: ``x[..., 0, :]`` is frame ``t-k+1`` (oldest) and
  ``x[..., -1, :]`` is frame ``t`` (most recent, "now").
- The last two dims are flattened TIME-MAJOR: frame 0's ``d_x`` features
  occupy the first ``d_x`` columns of the flattened vector, frame 1's
  features the next ``d_x`` columns, and so on. Equivalently,
  ``flatten(x)[..., i*d_x:(i+1)*d_x] == x[..., i, :]`` for every frame
  index ``i``.
"""

from __future__ import annotations

import torch
from torch import nn


class WindowMLPEncoder(nn.Module):
    """Encodes a window of ``k`` observation frames ``x -> z`` via a plain MLP.

    See the module docstring for the exact window layout (oldest-first,
    time-major flattening) -- getting this wrong is a silent bug.

    Args:
        d_x: single-frame observation dimension.
        d_z: latent dimension.
        k: window length (number of frames), default 8.
        hidden: sizes of the hidden layers, default ``(64, 64)``.
    """

    def __init__(
        self, d_x: int, d_z: int, k: int = 8, hidden: tuple[int, ...] = (64, 64)
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.k = k
        dims = (d_x * k, *hidden, d_z)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(..., k, d_x)``, oldest frame first -> ``z``: ``(..., d_z)``.

        See the module docstring for the exact window layout.

        Raises:
            ValueError: if ``x.shape[-2] != k`` or ``x.shape[-1] != d_x``.
        """
        if x.shape[-2] != self.k:
            raise ValueError(f"expected window length {self.k} at dim -2, got {x.shape[-2]}")
        if x.shape[-1] != self.d_x:
            raise ValueError(f"expected frame dim {self.d_x} at dim -1, got {x.shape[-1]}")
        leading = x.shape[:-2]
        flat = x.reshape(-1, self.k * self.d_x)
        z = self.net(flat)
        return z.reshape(*leading, self.d_z)
