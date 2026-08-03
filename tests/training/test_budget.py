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


def test_flops_component_locks():
    # adversarial review round 2: the affine-growth test alone also passed
    # the old undercounted formulas -- lock the individual terms.
    from torch import nn

    from sld.training.budget import count_flops_per_step

    # (1) bilinear - constant = exactly the interaction term + combine add
    kw = dict(d_z=4, n_pairs=2, d_u=3, delta=0.05, psi_hidden=(8,), pi_hidden=(8,))
    m_const = KoopmanSSMPredictor(regime="constant", **kw)
    m_bilin = KoopmanSSMPredictor(regime="bilinear", **kw)
    d_h = m_const.d_h
    diff = count_flops_per_step(m_bilin) - count_flops_per_step(m_const)
    expected = 2 * (3 * d_h * d_h) + d_h  # 2 FLOPs/MAC * d_u*d_h^2 + combine add
    print(f"[flops-lock] bilinear-constant diff = {diff}, expected = {expected}")
    assert diff == expected

    # (2) koopman trajectory = H*step + one lift + H projections, exactly
    H = 7
    psi_macs = sum(
        m.in_features * m.out_features for m in m_const.psi.modules() if isinstance(m, nn.Linear)
    )
    pi_macs = sum(
        m.in_features * m.out_features for m in m_const.pi.modules() if isinstance(m, nn.Linear)
    )
    traj = count_flops_per_trajectory(m_const, H)
    assert traj == H * count_flops_per_step(m_const) + 2 * psi_macs + H * 2 * pi_macs

    # (3) min_ssm step = 2*linear MACs (independently enumerated) + 5*d_state
    # elementwise (delta*a_cont, delta*b, A*m, B*v, sum) + d_h residual add
    m_ssm = SelectiveSSMPredictor(d_h=4, d_u=2, d_state=8)
    lin_macs = sum(
        m.in_features * m.out_features for m in m_ssm.modules() if isinstance(m, nn.Linear)
    )
    assert count_flops_per_step(m_ssm) == 2 * lin_macs + 5 * m_ssm.d_state + m_ssm.d_h


def test_count_params_against_architecture_derived_oracle():
    """Positive (REGLES 7.1 + R2): the budget tests above call
    ``count_params`` and compare it to... ``count_params``. It is
    implementation AND oracle there, so a miscount is undetectable.

    Writing ``sum(p.numel() for p in model.parameters())`` in the test would
    NOT fix that -- it is the same formula, not a second derivation. The
    independent reference here is built from the ARCHITECTURE: every
    ``nn.Linear(in, out)`` contributes ``in*out`` weights plus ``out`` biases
    (``bias=False`` contributes only the weights), and every remaining
    ``nn.Parameter`` contributes its declared shape. Walking the module tree
    by layer type and summing declared shapes is a different computation
    from summing ``p.numel()`` over a flat parameter iterator, and it fails
    if a parameter is registered but never reached, or reached twice through
    weight sharing.
    """
    from torch import nn

    from sld.predictors.registry import get_predictor
    from sld.training.budget import count_params

    for family in FAMILY_NAMES:
        # _FAMILY_FIXED carries koopman_ssm's mandatory delta (a physical
        # constant with no default, by design in match_budget)
        model = get_predictor(family, d_z=D_Z, d_u=D_U, **_FAMILY_FIXED[family])

        linear_total = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                linear_total += m.in_features * m.out_features
                if m.bias is not None:
                    linear_total += m.out_features
        # parameters that are not owned by an nn.Linear (raw nn.Parameter
        # tensors: a_log, nu, omega, B_j, positional tables, ...)
        linear_param_ids = {
            id(p) for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters()
        }
        loose_total = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad and id(p) not in linear_param_ids
        )
        oracle = linear_total + loose_total

        got = count_params(model)
        print(f"[count_params oracle] {family}: count_params={got} architecture_oracle={oracle}")
        assert got == oracle, f"{family}: {got} != {oracle}"  # exact integers, no tolerance


def test_count_params_discriminates_frozen_parameters():
    """Negative (REGLES 7.1): prove the oracle above has teeth. ``count_params``
    claims to count TRAINABLE parameters only. Freeze one tensor and the count
    must drop by exactly that tensor's numel -- a version that summed every
    parameter regardless of ``requires_grad`` would not move at all, and the
    positive test above would never notice.
    """
    from sld.predictors.registry import get_predictor
    from sld.training.budget import count_params

    model = get_predictor("mlp", d_z=D_Z, d_u=D_U)
    before = count_params(model)

    victim = next(p for p in model.parameters() if p.requires_grad)
    frozen_numel = victim.numel()
    victim.requires_grad_(False)
    after = count_params(model)

    naive_all = sum(p.numel() for p in model.parameters())  # the WRONG count
    print(
        f"[count_params negative] before={before} after={after} "
        f"frozen={frozen_numel} naive_all={naive_all}"
    )
    assert before - after == frozen_numel  # exact integers
    assert naive_all != after  # the requires_grad-blind count fails to move


def test_transformer_flops_matches_hand_count_at_two_points():
    """PORTÉE DE CE VERROU, déclarée (superviseur, 2026-08-03). Le comptage à la
    main reprend DEUX conventions posées par ``budget.py`` et ne les valide
    pas : (1) 2 FLOPs par MAC ; (2) softmax, LayerNorm et GELU hors périmètre.
    Ce test verrouille donc l'ARITHMÉTIQUE à conventions données -- il
    attraperait un terme oublié, un facteur faux, une dépendance en t erronée.
    Il n'attraperait PAS un changement de convention appliqué des deux côtés.
    Les constantes sont en revanche re-dérivables depuis l'architecture seule,
    et l'ont été indépendamment : 4 additions résiduelles (2 par bloc) + 1
    addition positionnelle (transformer.py:150), chacune large de d_model, plus
    le readout résiduel h + out_proj(y) large de d_h (transformer.py:161) ; et
    2 einsums par bloc contractant un cache de longueur (t+1).

    Positive (REGLES 7.1 + 7.2): the transformer's FLOPs counted BY HAND,
    term by term, at t=0 AND t=3 -- two points, because a single point is
    satisfied by infinitely many formulas (the negative test right below is
    anchored to match this same hand count exactly at t=0 alone, and still
    fails at t=3).

    Model: ``CausalTransformerPredictor(d_h=2, d_u=1, d_model=4, n_heads=2,
    n_layers=2)``, ``d_ff`` defaults to ``4*d_model=16``. Every ``nn.Linear``
    MAC count below is ``in_features*out_features`` read off the module
    definition directly (matching the architecture-derived-oracle style of
    ``test_count_params_against_architecture_derived_oracle`` above, not a
    call to ``budget._linear_macs``):

        in_proj:  Linear(d_h+d_u=3, d_model=4)     -> 3*4   = 12
        per block (x2 -- n_layers=2), each has:
          qkv:       Linear(4, 3*4=12)             -> 4*12  = 48
          attn_out:  Linear(4, 4)                  -> 4*4   = 16
          mlp[0]:    Linear(4, d_ff=16)            -> 4*16  = 64
          mlp[2]:    Linear(16, 4)                 -> 16*4  = 64
          block total                                       = 192
        2 blocks                                             = 384
        out_proj: Linear(d_model=4, d_h=2)         -> 4*2   =  8
        linear_macs total = 12 + 384 + 8                     = 404

    Convention declared in budget.py's own docstrings -- 2 FLOPs/MAC for
    every dense Linear "as executed"; elementwise ops counted 1 op = 1 FLOP;
    each block's 2 attention einsums (scores against the cache, weighted sum
    over the cache) are LINEAR in the (t+1)-long cache at
    ``2*d_model*(t+1)`` MACs/block = ``4*d_model*(t+1)`` FLOPs/block:

        elementwise_const (2 residual adds/block + the PE add + the final
        residual read-out add) = (2*n_layers+1)*d_model + d_h
                                = (2*2+1)*4 + 2 = 22
        c = 4*n_layers*d_model = 4*2*4                       = 32
        f0 (t=0, cache length 1)  = 2*404 + 22 + 32          = 862
        t=3 (cache length 4)      = f0 + c*3 = 862 + 96      = 958

    These two hand-derived numbers (862, 958) are the oracle -- computed
    from the model's declared architecture and budget.py's own stated
    counting convention, independently of calling ``_transformer_flops``.
    """
    model = CausalTransformerPredictor(d_h=2, d_u=1, d_model=4, n_heads=2, n_layers=2)
    from sld.training.budget import _transformer_flops

    hand_t0, hand_t3 = 862, 958
    got_t0 = _transformer_flops(model, 0)
    got_t3 = _transformer_flops(model, 3)
    print(f"[flops-hand] t=0: hand={hand_t0} _transformer_flops={got_t0}")
    print(f"[flops-hand] t=3: hand={hand_t3} _transformer_flops={got_t3}")
    assert got_t0 == hand_t0  # exact integers, no tolerance
    assert got_t3 == hand_t3


def test_transformer_flops_2t_formula_fails_hand_count():
    """Negative (REGLES 7.1): the audit's own named counterexample (REGLES
    section 7.1) is "un compteur de FLOPs satisfait par une formule en
    2^t". Anchor a 2^t-shaped false formula to match the CORRECT t=0 value
    EXACTLY -- so a test that only checked one point would be fooled -- and
    show it fails at t=3: this proves the two-point comparison above has
    teeth where a single point would not.

    It also proves this specific false formula would have sailed through
    the OLD growth-shape-only tests
    (test_transformer_flops_strictly_increasing_in_t,
    test_transformer_trajectory_flops_superlinear): it IS strictly
    increasing in t and IS superlinear over a trajectory, exactly like the
    real cost -- only the hand-derived VALUE comparison catches that the
    numbers themselves are wrong.
    """
    hand_t0, hand_t3 = 862, 958  # same hand count as the positive test above

    def false_2t_flops(t: int) -> int:
        return hand_t0 * (2**t)  # anchored at t=0, doubles every step after

    false_t0, false_t3 = false_2t_flops(0), false_2t_flops(3)
    print(
        f"[flops-hand] 2^t false formula: t=0={false_t0} (matches hand={hand_t0}) "
        f"t=3={false_t3} vs hand={hand_t3}"
    )
    assert false_t0 == hand_t0  # documents WHY a single point isn't enough
    assert false_t3 != hand_t3  # the second point is what catches it

    # ... and it would have passed the OLD shape-only tests:
    shape_ts = (0, 1, 5, 50)
    shape_vals = [false_2t_flops(t) for t in shape_ts]
    assert shape_vals == sorted(shape_vals)
    assert len(set(shape_vals)) == len(shape_vals)  # strictly increasing

    f_h = sum(false_2t_flops(t) for t in range(50))
    f_2h = sum(false_2t_flops(t) for t in range(100))
    print(f"[flops-hand] 2^t false formula trajectory ratio H=50 -> {f_2h / f_h:.3f}")
    assert f_2h / f_h > 2.5  # "superlinear" too -- and still the wrong numbers
