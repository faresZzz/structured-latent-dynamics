"""Tests for sld.training.budget: parameter-matched family construction and
analytic per-step / per-trajectory FLOP accounting.
"""

from __future__ import annotations

import pytest

from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.transformer import CausalTransformerPredictor
from sld.training.budget import (
    count_flops_per_step,
    count_flops_per_trajectory,
    count_params,
    match_budget,
)

D_Z, D_U = 4, 2

_FAMILY_FIXED = {
    "mlp": {},
    "transformer": {},
    "min_ssm": {},
    "koopman_ssm": {"delta": 0.05},
}
_CONSTRUCTORS = {
    "mlp": MLPPredictor,
    "transformer": CausalTransformerPredictor,
    "min_ssm": SelectiveSSMPredictor,
    "koopman_ssm": KoopmanSSMPredictor,
}
FAMILY_NAMES = tuple(_CONSTRUCTORS)


def _build(family: str, target: int = 10_000):
    kwargs = match_budget(family, D_Z, D_U, target_params=target, **_FAMILY_FIXED[family])
    return _CONSTRUCTORS[family](**kwargs)


@pytest.mark.parametrize("target", [10_000, 30_000])
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_match_budget_hits_target(family, target):
    model = _build(family, target)
    params = count_params(model)
    rel_err = abs(params - target) / target
    print(f"[budget] family={family} target={target} params={params} rel_err={rel_err:.3%}")
    assert rel_err <= 0.05


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_flops_per_step_positive(family):
    model = _build(family)
    flops = count_flops_per_step(model, t=1)
    print(f"[budget] flops_per_step family={family} t=1 -> {flops}")
    assert flops > 0


def test_transformer_flops_strictly_increasing_in_t():
    model = _build("transformer")
    flops = [count_flops_per_step(model, t=t) for t in (0, 1, 5, 50)]
    print(f"[budget] transformer flops per t -> {flops}")
    assert flops == sorted(flops)
    assert len(set(flops)) == len(flops)  # strictly increasing, not just non-decreasing


@pytest.mark.parametrize("H", [50, 100])
def test_transformer_trajectory_flops_superlinear(H):
    # a small, direct construction (not match_budget's 10k search, which
    # lands at d_model=20 where the FFN's O(d_model^2) fixed cost dwarfs
    # attention's O(d_model*t) cost until t is in the hundreds): this test
    # is about count_flops_per_trajectory's growth SHAPE, not the 10k
    # budget, so use a width where attention is a visible share of cost by
    # H=50 already
    model = CausalTransformerPredictor(d_h=D_Z, d_u=D_U, d_model=8, n_heads=2, n_layers=2)
    f_h = count_flops_per_trajectory(model, H)
    f_2h = count_flops_per_trajectory(model, 2 * H)
    ratio = f_2h / f_h
    print(f"[budget] transformer trajectory ratio H={H} -> {ratio:.3f}")
    assert ratio > 2.5


@pytest.mark.parametrize("H", [50, 100])
@pytest.mark.parametrize("family", ["mlp", "min_ssm", "koopman_ssm"])
def test_other_families_trajectory_flops_exactly_affine(family, H):
    # AFFINE in H, not proportional: koopman_ssm's trajectory bill carries a
    # constant one-time lift on top of its per-step cost (step + projection),
    # so the exact-linearity claim is equal increments, integer arithmetic
    model = _build(family)
    f_h = count_flops_per_trajectory(model, H)
    f_2h = count_flops_per_trajectory(model, 2 * H)
    f_3h = count_flops_per_trajectory(model, 3 * H)
    print(f"[budget] {family} trajectory flops H={H} -> {f_h}, 2H -> {f_2h}, 3H -> {f_3h}")
    assert f_3h - f_2h == f_2h - f_h  # constant per-step cost, exact
    assert f_2h > f_h > 0


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        match_budget("not_a_family", D_Z, D_U, target_params=10_000)


def test_koopman_without_delta_raises():
    with pytest.raises(ValueError):
        match_budget("koopman_ssm", D_Z, D_U, target_params=10_000)


def test_out_of_reach_budget_raises():
    with pytest.raises(ValueError):
        match_budget("mlp", D_Z, D_U, target_params=10)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
