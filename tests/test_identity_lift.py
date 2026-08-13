"""The `on-stable-nolift` level: an identity lift, and what it must refuse.

PROTOCOL 25.3. The arm exists to attribute campaign A's penalty either to the
structured TRANSITION or to the `psi`/`pi` pair that accompanies it, so the
only thing that may differ between it and `on-stable` is that pair. Every test
below defends exactly that.

REGLES 7.1's two checks. The positive ones pin the identity and the frozen
backbone. The negative ones are the load-bearing half: a silent fallback to an
MLP lift would produce a plausible arm that attributes nothing, and a widened
backbone would reproduce the mu_J fault 23.2 names -- "measuring width while
calling it structure".
"""

from __future__ import annotations

import pytest
import torch

from sld.predictors.structured import (
    _VALID_LIFT,
    MLPKPredictor,
    SSMKPredictor,
    TransformerKPredictor,
)
from sld.training.budget import match_budget
from sld.training.trainer import _STRUCTURED_KEYS

BUILDERS = (MLPKPredictor, TransformerKPredictor, SSMKPredictor)
BASE = dict(d_z=4, d_u=1, delta=0.05, r=2, width=32, radius="bounded")


# --- positive ---------------------------------------------------------------


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_identity_lift_is_exactly_the_identity(builder):
    """Not approximately: `lift` and `project` must return the SAME tensor
    values, on arbitrary input. A learned Linear initialised near the identity
    would pass an `allclose` and fail this."""
    model = builder(**BASE, lift="identity")
    z = torch.randn(7, 4)
    assert torch.equal(model.lift(z), z)
    assert torch.equal(model.project(z), z)


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_identity_lift_carries_no_parameters(builder):
    """The pair is REMOVED, not shrunk. A one-unit MLP would still learn."""
    model = builder(**BASE, lift="identity")
    assert sum(p.numel() for p in model.psi.parameters()) == 0
    assert sum(p.numel() for p in model.pi.parameters()) == 0


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_mlp_lift_is_unchanged(builder):
    """Non-regression: the default path must be what it was before `lift`
    existed, so the 134 runs already measured stay comparable."""
    default = builder(**BASE)
    explicit = builder(**BASE, lift="mlp")
    assert default.lift_mode == "mlp"
    assert sum(p.numel() for p in default.parameters()) == sum(
        p.numel() for p in explicit.parameters()
    )
    assert sum(p.numel() for p in default.psi.parameters()) > 0


def test_the_nolift_arm_inherits_on_stable_s_backbone_exactly():
    """23.2's frozen backbone, by construction rather than by coincidence.

    `match_budget` sizes the nolift arm as if its lift were an MLP, so width
    and psi_hidden are IDENTICAL to `on-stable`'s and the realised parameter
    difference is exactly the pair that was removed. If the matcher were
    allowed to spend those parameters on the backbone, the arm would differ in
    two ways and could attribute nothing.
    """
    on_stable = match_budget("mlp_k", 4, 1, 10_000, delta=0.05, r=2, radius="bounded")
    nolift = match_budget("mlp_k", 4, 1, 10_000, delta=0.05, r=2, radius="bounded", lift="identity")

    assert nolift["width"] == on_stable["width"]
    assert nolift["psi_hidden"] == on_stable["psi_hidden"]
    assert nolift["lift"] == "identity" and on_stable["lift"] == "mlp"

    a, b = MLPKPredictor(**on_stable), MLPKPredictor(**nolift)
    pair = sum(p.numel() for p in a.psi.parameters()) + sum(p.numel() for p in a.pi.parameters())
    delta = sum(p.numel() for p in a.parameters()) - sum(p.numel() for p in b.parameters())
    assert delta == pair, "the only parameters the nolift arm lacks are psi and pi"


def test_lift_is_a_hashed_config_key():
    """It changes the model, so it must change the run identity (13.x)."""
    assert "lift" in _STRUCTURED_KEYS


# --- negative: the half that gives the positive half its meaning -------------


@pytest.mark.parametrize("builder", BUILDERS)
def test_identity_lift_is_refused_when_the_dimensions_do_not_match(builder):
    """`2r == d_z` is necessary AND sufficient. With `2r != d_z` the identity
    is not even type-correct, and a silent projection would be a different
    model wearing this arm's name."""
    with pytest.raises(ValueError, match="2r == d_z"):
        builder(d_z=6, d_u=1, delta=0.05, r=2, width=32, lift="identity")


@pytest.mark.parametrize("builder", BUILDERS)
def test_an_unknown_lift_is_refused_rather_than_defaulted(builder):
    """A typo must be a refusal. Falling back to "mlp" would run the whole
    campaign on the wrong arm and every gate would be green."""
    with pytest.raises(ValueError, match="lift"):
        builder(**BASE, lift="none")
    assert set(_VALID_LIFT) == {"mlp", "identity"}


def test_a_silent_fallback_to_an_mlp_lift_would_be_caught():
    """The mutant. If `lift="identity"` were ignored, the arm would still
    build, still train, and still produce a plausible number -- attributing
    nothing. Three independent signatures separate the two arms."""
    identity = MLPKPredictor(**BASE, lift="identity")
    mlp = MLPKPredictor(**BASE, lift="mlp")
    z = torch.randn(5, 4)

    assert not torch.equal(mlp.lift(z), z), "the mlp lift must NOT be the identity"
    assert torch.equal(identity.lift(z), z)
    assert sum(p.numel() for p in mlp.parameters()) > sum(p.numel() for p in identity.parameters())
    assert identity.lift_mode != mlp.lift_mode


def test_an_absent_lift_key_does_not_change_a_run_identity():
    """The 134 runs already measured carry no `lift` key. Canonicalizing a
    default into the config -- which the family branch in `Trainer` does for
    `delta_min`, `n_heads` and the rest -- would move every one of their
    `config_hash`es and orphan the whole cycle. `lift` follows `radius`: legal,
    hashed when present, never injected when absent.
    """
    from sld.training.trainer import Trainer

    cfg = dict(
        exp_name="t", seed=0, family="mlp_k", d_x=4, d_z=4, d_u=1,
        delta=0.05, r=2, radius="bounded", epochs=1, encoder_hidden=[],
    )
    before = Trainer(dict(cfg)).config_hash
    trainer = Trainer(dict(cfg))
    assert "lift" not in trainer.config, "lift must not be canonicalized into the identity"
    assert Trainer(dict(cfg)).config_hash == before

    with_lift = Trainer({**cfg, "lift": "identity"}).config_hash
    assert with_lift != before, "an identity lift IS a different model and must hash apart"
