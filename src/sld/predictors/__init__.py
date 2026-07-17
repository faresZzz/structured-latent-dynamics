"""Latent predictors: the four families and their shared building blocks.

``base`` carries the contract (``Predictor``/``PredictorState``/
``PredictorModule``); the four families are ``mlp``, ``transformer``,
``min_ssm``, ``koopman_ssm``; ``registry.get_predictor`` constructs a
parameter-matched instance by family name (fairness contract).
"""

from sld.predictors.base import Predictor, PredictorModule, PredictorState
from sld.predictors.blocks2x2 import RotationScalingBlocks, block_diagonal
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.registry import FAMILIES, get_predictor
from sld.predictors.transformer import CausalTransformerPredictor

__all__ = [
    "FAMILIES",
    "CausalTransformerPredictor",
    "KoopmanSSMPredictor",
    "MLPPredictor",
    "Predictor",
    "PredictorModule",
    "PredictorState",
    "RotationScalingBlocks",
    "SelectiveSSMPredictor",
    "block_diagonal",
    "get_predictor",
]
