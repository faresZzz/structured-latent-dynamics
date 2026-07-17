"""Predictor family registry: names -> parameter-matched constructors.

This is how experiment configs name families (``mlp``/``transformer``/
``min_ssm``/``koopman_ssm``) without importing each predictor class
directly. The fairness contract is enforced here: every model
``get_predictor`` returns has its parameter count matched to
``target_params`` (+/-5%, ``sld.training.budget.match_budget``), so
comparing families at a fixed parameter budget is comparing like with like.
"""

from __future__ import annotations

from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.transformer import CausalTransformerPredictor

# family name -> constructor: lives HERE (predictors) so the dependency runs
# one way only (training.budget imports this map; predictors never imports
# training at module scope -- adversarial review round 1, circular import)
_FAMILY_CONSTRUCTORS: dict[str, type] = {
    "mlp": MLPPredictor,
    "transformer": CausalTransformerPredictor,
    "min_ssm": SelectiveSSMPredictor,
    "koopman_ssm": KoopmanSSMPredictor,
}

FAMILIES: tuple[str, ...] = tuple(_FAMILY_CONSTRUCTORS)


def get_predictor(family: str, d_z: int, d_u: int, target_params: int = 10_000, **overrides):
    """Construct a parameter-matched predictor instance for ``family``.

    ``d_z``/``d_u`` are the latent/action dimensions; ``overrides`` are
    passed through to ``match_budget`` (and from there to the constructor)
    -- e.g. ``delta=...`` is required for ``family="koopman_ssm"``.

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
