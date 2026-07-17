"""Tests for sld.encoders.frame_mlp.FrameMLPEncoder.

Seeds are explicit local generators (torch.Generator) throughout; no global
RNG state is mutated.
"""

from __future__ import annotations

import torch

from sld.encoders.frame_mlp import FrameMLPEncoder


def test_shape_1d_input():
    enc = FrameMLPEncoder(d_x=5, d_z=3)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(5, generator=gen)
    z = enc(x)
    assert z.shape == (3,)


def test_shape_2d_input():
    enc = FrameMLPEncoder(d_x=5, d_z=3)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(7, 5, generator=gen)
    z = enc(x)
    assert z.shape == (7, 3)


def test_shape_3d_input():
    enc = FrameMLPEncoder(d_x=5, d_z=3)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 7, 5, generator=gen)
    z = enc(x)
    assert z.shape == (2, 7, 3)


def test_deterministic_for_same_input():
    enc = FrameMLPEncoder(d_x=4, d_z=6)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(10, 4, generator=gen)

    z1 = enc(x)
    z2 = enc(x)

    torch.testing.assert_close(z1, z2, atol=0.0, rtol=0.0)


def test_output_differs_for_different_inputs():
    enc = FrameMLPEncoder(d_x=4, d_z=6)
    gen = torch.Generator().manual_seed(2)
    x1 = torch.randn(10, 4, generator=gen)
    x2 = torch.randn(10, 4, generator=gen)

    z1 = enc(x1)
    z2 = enc(x2)

    max_diff = (z1 - z2).abs().max().item()
    print(f"[frame_mlp distinct inputs] max_abs_diff = {max_diff:.3e}")
    assert not torch.allclose(z1, z2)
