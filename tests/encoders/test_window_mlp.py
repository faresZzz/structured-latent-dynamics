"""Tests for sld.encoders.window_mlp.WindowMLPEncoder.

Seeds are explicit local generators (torch.Generator) throughout; no global
RNG state is mutated.

The layout test is the important one: WindowMLPEncoder documents an
oldest-first, time-major flattening (frame i -> flattened columns
``[i*d_x, (i+1)*d_x)``). A wrong window order is a silent bug for partial
observation (Wave 6), so this is pinned down explicitly rather than trusted
by inspection.
"""

from __future__ import annotations

import pytest
import torch

from sld.encoders.window_mlp import WindowMLPEncoder


def test_shape_batched_input():
    enc = WindowMLPEncoder(d_x=4, d_z=3, k=6)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 7, 6, 4, generator=gen)
    z = enc(x)
    assert z.shape == (2, 7, 3)


def test_shape_unbatched_input():
    enc = WindowMLPEncoder(d_x=4, d_z=3, k=6)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(6, 4, generator=gen)
    z = enc(x)
    assert z.shape == (3,)


def test_wrong_window_length_raises():
    enc = WindowMLPEncoder(d_x=4, d_z=3, k=6)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(5, 4, generator=gen)  # k=5, encoder expects k=6
    with pytest.raises(ValueError):
        enc(x)


def test_wrong_frame_dim_raises():
    enc = WindowMLPEncoder(d_x=4, d_z=3, k=6)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(6, 5, generator=gen)  # d_x=5, encoder expects d_x=4
    with pytest.raises(ValueError):
        enc(x)


def test_layout_frame_i_maps_to_its_column_slice():
    """Zero out every frame except index i, then verify via the first
    Linear layer's weight that frame i's features land exactly in
    flattened columns [i*d_x, (i+1)*d_x) -- the documented layout.
    """
    d_x, k, d_z = 3, 5, 4
    enc = WindowMLPEncoder(d_x=d_x, d_z=d_z, k=k, hidden=(8,))
    first_linear = enc.net[0]
    rest_of_net = enc.net[1:]

    gen = torch.Generator().manual_seed(7)
    for i in range(k):
        frame = torch.randn(d_x, generator=gen)
        x = torch.zeros(k, d_x)
        x[i] = frame

        z_actual = enc(x)

        # Manual: with every other frame zero, the first-layer
        # pre-activation must equal weight[:, i*d_x:(i+1)*d_x] @ frame + bias
        # if (and only if) frame i lands in exactly those columns after
        # flattening, per the documented contract.
        w_slice = first_linear.weight[:, i * d_x : (i + 1) * d_x]
        h0_manual = frame @ w_slice.T + first_linear.bias
        z_manual = rest_of_net(h0_manual)

        max_diff = (z_actual - z_manual).abs().max().item()
        print(f"[window layout] frame_i={i} max_abs_diff = {max_diff:.3e}")
        torch.testing.assert_close(z_actual, z_manual, atol=1e-6, rtol=0)


def test_reversing_window_changes_output():
    """Time-order sensitivity: the encoder must not be order-blind."""
    enc = WindowMLPEncoder(d_x=4, d_z=5, k=6)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(6, 4, generator=gen)

    z_forward = enc(x)
    z_reversed = enc(x.flip(dims=(-2,)))

    max_diff = (z_forward - z_reversed).abs().max().item()
    print(f"[window time-order] max_abs_diff forward_vs_reversed = {max_diff:.3e}")
    assert not torch.allclose(z_forward, z_reversed)
