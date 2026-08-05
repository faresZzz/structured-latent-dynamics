"""Latent predictors: the seven families and their shared building blocks.

``base`` carries the contract (``Predictor``/``PredictorState``/
``PredictorModule``). Four UNSTRUCTURED families -- ``mlp``, ``transformer``,
``min_ssm``, ``koopman_ssm`` -- plus the three STRUCTURED arms of the 3x2
factorial, ``mlp_k``/``transformer_k``/``ssm_k``, which share one transition
(``structured.StructuredPredictor``) and differ only by their context
producer ``kappa_b``. ``registry.get_predictor`` constructs a
parameter-matched instance by family name (fairness contract).

Seven families, six factorial cells: ``koopman_ssm`` remains constructible
and loadable for published artifacts and is never a factorial cell
(PROTOCOL.md section 13.1).
"""

from sld.predictors.base import Predictor, PredictorModule, PredictorState
from sld.predictors.blocks2x2 import RotationScalingBlocks, block_diagonal
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.registry import FAMILIES, STRUCTURED_FAMILIES, get_predictor
from sld.predictors.structured import (
    MLPKPredictor,
    SSMKPredictor,
    StructuredPredictor,
    TransformerKPredictor,
)
from sld.predictors.transformer import CausalTransformerPredictor

__all__ = [
    "FAMILIES",
    "STRUCTURED_FAMILIES",
    "CausalTransformerPredictor",
    "KoopmanSSMPredictor",
    "MLPKPredictor",
    "MLPPredictor",
    "Predictor",
    "PredictorModule",
    "PredictorState",
    "RotationScalingBlocks",
    "SSMKPredictor",
    "SelectiveSSMPredictor",
    "StructuredPredictor",
    "TransformerKPredictor",
    "block_diagonal",
    "get_predictor",
]
