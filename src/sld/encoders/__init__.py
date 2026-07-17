"""Observation encoders: one frame or a stacked window to a latent z."""

from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.encoders.window_mlp import WindowMLPEncoder

__all__ = ["FrameMLPEncoder", "WindowMLPEncoder"]
