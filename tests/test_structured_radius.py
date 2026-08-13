"""The third level of the structure factor: a spectral radius bounded by
construction, and the proof that it is bounded WITHOUT a side condition.

Why these tests exist in this form. The first parametrisation considered for
this arm was the symplectic-Euler adLIF map, whose determinant is exactly
``alpha*beta`` and whose radius is therefore ``sqrt(alpha*beta) < 1`` -- but
only while its eigenvalues stay complex. Pushed outside that regime the
eigenvalues go real and the radius is unbounded (measured: 368 over 2e5 random
draws). A guarantee that holds under a condition needs a test that watches the
condition; the parametrisation shipped here needs none, and
``test_radius_is_bounded_without_any_side_condition`` is what says so.

What this file asserted first, and why it was wrong (corrected 2026-08-11,
PROTOCOL section 23.1's dated precision). The strict form ``r < 1`` was
asserted on a quantity that floating point cannot hold: ``sigmoid(rho)``
rounds to exactly 1.0 once ``1 - sigmoid(rho)`` drops below half an ulp at 1,
i.e. above ``rho ~ 17.33`` in float32 (the Trainer's working precision) and
``rho ~ 37.43`` in float64. The measurement that settled it swept 400 001
values of ``rho`` in [-50, 50] at both precisions: ZERO radii above 1, at
either precision. The guarantee section 23.1 recruits this arm for -- the
radius never diverges, so ``r**H`` stays bounded for every ``H`` -- holds
exactly; the wording "strictly inside (0, 1)" holds in exact arithmetic only.

The assertions below are the corrected ones, and they are stronger in
substance than what they replace: ``r <= 1`` everywhere AND finite AND
non-negative everywhere, the saturation point LOCALISED rather than tolerated,
the saturation shown BENIGN at the horizons that are actually rolled out, and
the ``r = sigmoid(rho)`` identity kept on the domain where the working
precision can represent it.
"""

from __future__ import annotations

import math

import pytest
import torch

from sld.predictors.structured import _VALID_RADIUS, MLPKPredictor, SSMKPredictor, TransformerKPredictor

DELTA = 0.05
BUILDERS = (MLPKPredictor, TransformerKPredictor, SSMKPredictor)


def _radius(alpha: torch.Tensor, delta: float = DELTA) -> torch.Tensor:
    """Discrete spectral radius of ``exp(delta * (alpha*I + omega*J))``."""
    return torch.exp(delta * alpha)


# Measured, not assumed: the first ``rho`` at which ``sigmoid(rho)`` rounds to
# exactly 1.0, swept over 400 001 points of [-50, 50] at each precision.
_SATURATION_RHO = {torch.float32: 17.329, torch.float64: 37.430}


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_radius_never_exceeds_one_without_any_side_condition(builder, dtype) -> None:
    """The guarantee, at the precision it actually runs at: no regime, no
    clamp, no guard, and never a radius above 1.

    This is the property section 23.1 recruits the arm for. The comparison is
    ``<= 1``, not ``< 1``, because the strict form is a statement about ``R``
    and this is a statement about the machine -- see the module docstring.
    """
    model = builder(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded", dtype=dtype)
    rho = torch.linspace(-50.0, 50.0, 4001, dtype=dtype).reshape(-1, 1).expand(-1, model.r)
    r = _radius(model._alpha(rho))
    assert torch.isfinite(r).all(), "logsigmoid must not underflow to -inf in the left tail"
    assert (r >= 0.0).all(), "a spectral radius is never negative"
    assert (r <= 1.0).all(), f"radius above 1 at rho={rho[r > 1.0]}"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_the_saturation_point_is_where_it_was_measured(dtype) -> None:
    """Localise the saturation rather than tolerate it.

    Below the measured point the radius is STRICTLY below 1 -- the exact-
    arithmetic claim, verified where the precision can carry it. Above it the
    radius is exactly 1.0, and that is a floating-point fact about
    ``1 - epsilon``, not a property of the map.
    """
    rho_sat = _SATURATION_RHO[dtype]
    model = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded", dtype=dtype)

    below = torch.linspace(-50.0, rho_sat - 0.5, 2001, dtype=dtype).reshape(-1, 1).expand(-1, model.r)
    assert (_radius(model._alpha(below)) < 1.0).all(), "strict below the measured saturation"

    above = torch.linspace(rho_sat + 0.5, 50.0, 2001, dtype=dtype).reshape(-1, 1).expand(-1, model.r)
    assert (_radius(model._alpha(above)) == 1.0).all(), "exactly 1.0 above it, never beyond"

    # and the point is where it is because of the precision, not the map:
    # float64 carries the gap ~20 units of rho further than float32.
    assert _SATURATION_RHO[torch.float64] > _SATURATION_RHO[torch.float32]


@pytest.mark.parametrize("horizon", (25, 100))
def test_the_saturation_is_benign_under_rollout(horizon) -> None:
    """r = 1 gives r**H = 1: bounded, at every horizon that is rolled out.

    This is the opposite of the failure mode the arm exists to eliminate. The
    contrast with the symplectic-Euler map rejected in 23.1 -- radius measured
    up to 368 outside the oscillatory regime, so 368**25 -- is made SHARPER by
    the saturation, not softer.
    """
    model = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded")
    saturating = torch.full((3, model.r), 50.0)
    r = _radius(model._alpha(saturating))
    assert (r == 1.0).all()
    assert torch.isfinite(r**horizon).all() and (r**horizon <= 1.0).all()

    free = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="free")
    r_free = _radius(free._alpha(saturating))
    assert (r_free**horizon > 1.0).all(), "the free arm is what an unbounded radius does"


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_sigmoid_identity_holds_where_the_precision_can_carry_it(builder) -> None:
    """``r = sigmoid(rho)`` is the identity the parametrisation is built on.

    Asserted on the non-saturated domain, at the working precision's own
    tolerance: past the saturation point BOTH sides round to 1.0, so the
    identity would still hold there but would no longer be testing anything.
    """
    model = builder(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded")
    rho = torch.linspace(-50.0, _SATURATION_RHO[torch.float32] - 0.5, 4001).reshape(-1, 1)
    rho = rho.expand(-1, model.r)
    assert torch.allclose(_radius(model._alpha(rho)), torch.sigmoid(rho), atol=1e-6)


@pytest.mark.parametrize("builder", BUILDERS)
def test_free_mode_is_bit_identical_to_before(builder) -> None:
    """Non-regression: ``free`` must still be the identity on the head slice."""
    model = builder(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="free")
    theta = torch.randn(7, model.r)
    assert torch.equal(model._alpha(theta), theta)


def test_free_mode_can_be_unstable_and_bounded_mode_cannot() -> None:
    """The two levels are genuinely different treatments, not two spellings.

    This is the contrast the factorial measures: ``free`` admits a radius above
    1 (PROTOCOL section 13.2 wants unstable modes representable), ``bounded``
    does not.
    """
    free = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="free")
    bounded = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded")
    theta = torch.full((3, free.r), 20.0)  # a large positive head output
    assert (_radius(free._alpha(theta)) > 1.0).all()
    # 20.0 sits above float32's measured saturation (17.329), so the bounded
    # arm returns exactly 1.0 here. `<= 1` is the guarantee; the two levels are
    # still genuinely different treatments -- 2.72 against 1.0 at this input.
    assert (_radius(bounded._alpha(theta)) <= 1.0).all()
    assert _radius(free._alpha(theta)).min() > _radius(bounded._alpha(theta)).max()


def test_both_levels_start_at_the_same_radius_band() -> None:
    """At step 0 the two arms differ by their parametrisation and nothing else.

    PROTOCOL section 13.6 fixes the free arm's ``alpha_0 ~ U(-1.5, -0.5)``; the
    bounded arm's ``rho_0`` is initialised as that band's sigmoid pre-image, so
    both start with ``r`` in the same interval.
    """
    lo, hi = math.exp(DELTA * -1.5), math.exp(DELTA * -0.5)
    for radius in ("free", "bounded"):
        model = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=8, width=32, radius=radius)
        r0 = _radius(model._alpha(model.head.bias[: model.r].detach().unsqueeze(0)))
        assert (r0 >= lo - 1e-6).all() and (r0 <= hi + 1e-6).all(), f"{radius}: r0={r0}"


@pytest.mark.parametrize("builder", BUILDERS)
def test_unknown_radius_is_refused(builder) -> None:
    with pytest.raises(ValueError, match="radius"):
        builder(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="contractive")
    assert set(_VALID_RADIUS) == {"free", "bounded"}


def test_mutant_broken_reparametrisation_is_caught() -> None:
    """If the map stops bounding the radius, the suite must go red.

    The mutant is the plausible mistake: forgetting the division by ``Delta``,
    which leaves ``r = exp(Delta * log sigmoid(rho))`` -- still below 1 here but
    no longer EQUAL to ``sigmoid(rho)``, so the identity the protocol declares
    would be silently false.
    """
    model = MLPKPredictor(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded")
    rho = torch.tensor([[0.0, 3.0]])
    good = _radius(model._alpha(rho))
    mutant = _radius(torch.nn.functional.logsigmoid(rho))  # missing / delta
    assert torch.allclose(good, torch.sigmoid(rho), atol=1e-6)
    assert not torch.allclose(mutant, torch.sigmoid(rho), atol=1e-3)


@pytest.mark.parametrize("builder", BUILDERS)
def test_radius_does_not_change_the_head_width(builder) -> None:
    """Section 13.7's fairness argument: the head emits the SAME numbers.

    Two per block plus ``vec B``, in both levels. If ``bounded`` needed a third
    number per block, the budget contract would have to be amended; it does not.
    """
    free = builder(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="free")
    bounded = builder(d_z=4, d_u=1, delta=DELTA, r=2, width=32, radius="bounded")
    assert free.head.out_features == bounded.head.out_features
    assert sum(p.numel() for p in free.parameters()) == sum(
        p.numel() for p in bounded.parameters()
    )
