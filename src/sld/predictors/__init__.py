"""Latent predictors: the four families and their shared building blocks.

Wave 2 ships the rotation-scaling block parametrization; the four predictor
families and the `Predictor` contract arrive in Wave 3.
"""

from sld.predictors.blocks2x2 import RotationScalingBlocks, block_diagonal

__all__ = ["RotationScalingBlocks", "block_diagonal"]
