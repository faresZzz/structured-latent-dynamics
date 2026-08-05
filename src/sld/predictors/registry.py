"""Predictor family registry: names -> parameter-matched constructors.

This is how experiment configs name families (``mlp``/``transformer``/
``min_ssm``/``koopman_ssm``/``mlp_k``/``transformer_k``/``ssm_k``) without
importing each predictor class directly. The fairness contract is enforced
here: every model
``get_predictor`` returns has its parameter count matched to
``target_params`` (+/-5%, ``sld.training.budget.match_budget``), so
comparing families at a fixed parameter budget is comparing like with like.
"""

from __future__ import annotations

from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.structured import MLPKPredictor, SSMKPredictor, TransformerKPredictor
from sld.predictors.transformer import CausalTransformerPredictor

# The 3 structured arms of the 3x2 factorial (PROTOCOL.md section 13.1):
# ONE shared transition, ONE shared constructor signature, ``width`` the
# single budget knob of all three. Named as a group here because every
# dispatch downstream (budget._kwargs_for, Trainer's canonicalization)
# treats them as one branch -- that is the point of the shared signature.
STRUCTURED_FAMILIES: tuple[str, ...] = ("mlp_k", "transformer_k", "ssm_k")

# family name -> constructor: lives HERE (predictors) so the dependency runs
# one way only (training.budget imports this map; predictors never imports
# training at module scope -- adversarial review round 1, circular import)
#
# SEVEN families for SIX factorial cells: ``koopman_ssm`` stays constructible
# and loadable so every published artifact keeps its constructor, but it is
# never a cell of the 3x2 factorial (PROTOCOL.md section 13.1). That
# asymmetry is deliberate and is enforced where it matters -- an arm
# resolving to it is refused by ``experiments.factorial._validate_arm``.
_FAMILY_CONSTRUCTORS: dict[str, type] = {
    "mlp": MLPPredictor,
    "transformer": CausalTransformerPredictor,
    "min_ssm": SelectiveSSMPredictor,
    "koopman_ssm": KoopmanSSMPredictor,
    "mlp_k": MLPKPredictor,
    "transformer_k": TransformerKPredictor,
    "ssm_k": SSMKPredictor,
}

FAMILIES: tuple[str, ...] = tuple(_FAMILY_CONSTRUCTORS)


def get_predictor(family: str, d_z: int, d_u: int, target_params: int = 10_000, **overrides):
    """Construct a parameter-matched predictor instance for ``family``.

    ``d_z``/``d_u`` are the latent/action dimensions; ``overrides`` are
    passed through to ``match_budget`` (and from there to the constructor)
    -- e.g. ``delta=...`` is required for ``family="koopman_ssm"`` and for
    each of ``STRUCTURED_FAMILIES`` (a physical ZOH step, not a capacity
    knob, so it has no default in any of them).

    Raises:
        ValueError: unknown ``family`` (see ``FAMILIES``), or no width
            hits ``target_params`` within +/-5% (propagated from
            ``match_budget``).
    """
    # imported lazily: training depends on predictors at module scope, never
    # the reverse (keeps `import sld.training` and `import sld.predictors`
    # both clean in a fresh process, in either order)
    from sld.training.budget import match_budget

    kwargs = match_budget(family, d_z, d_u, target_params=target_params, **overrides)
    return _FAMILY_CONSTRUCTORS[family](**kwargs)
