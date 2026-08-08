"""Tests for sld.training.budget: parameter-matched family construction and
analytic per-step / per-trajectory FLOP accounting.
"""

from __future__ import annotations

import pytest
import torch

from sld.predictors.base import PredictorState
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.registry import FAMILIES, STRUCTURED_FAMILIES
from sld.predictors.structured import MLPKPredictor, SSMKPredictor, TransformerKPredictor
from sld.predictors.transformer import CausalTransformerPredictor
from sld.training.budget import (
    count_flops_per_step,
    count_flops_per_trajectory,
    match_budget,
)

D_Z, D_U = 4, 2

_FAMILY_FIXED = {
    "mlp": {},
    "transformer": {},
    "min_ssm": {},
    "koopman_ssm": {"delta": 0.05},
    "mlp_k": {"delta": 0.05},
    "transformer_k": {"delta": 0.05},
    "ssm_k": {"delta": 0.05},
}
_CONSTRUCTORS = {
    "mlp": MLPPredictor,
    "transformer": CausalTransformerPredictor,
    "min_ssm": SelectiveSSMPredictor,
    "koopman_ssm": KoopmanSSMPredictor,
    "mlp_k": MLPKPredictor,
    "transformer_k": TransformerKPredictor,
    "ssm_k": SSMKPredictor,
}
FAMILY_NAMES = tuple(_CONSTRUCTORS)

# the two arms whose per-step cost grows with trajectory position
_ATTENTION_FAMILIES = ("transformer", "transformer_k")
_CONSTANT_COST_FAMILIES = tuple(f for f in FAMILY_NAMES if f not in _ATTENTION_FAMILIES)


def test_family_tables_cover_the_whole_registry():
    # these two dicts are LOCAL duplicates of the registry. A family added
    # there and forgotten here is covered by no budget or FLOP test at all,
    # and the suite stays green -- pin them together instead.
    assert set(FAMILY_NAMES) == set(FAMILIES)
    assert set(_FAMILY_FIXED) == set(FAMILIES)
    assert len(FAMILIES) == 7


def _build(family: str, target: int = 10_000):
    kwargs = match_budget(family, D_Z, D_U, target_params=target, **_FAMILY_FIXED[family])
    return _CONSTRUCTORS[family](**kwargs)


@pytest.mark.parametrize("target", [10_000, 30_000])
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_match_budget_hits_target(family, target):
    model = _build(family, target)
    # C6: the reference is a DIRECT sum over the model's own trainable
    # tensors, never count_params -- match_budget already uses count_params
    # internally to decide when to stop, so checking it against itself would
    # make the fairness contract its own oracle.
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rel_err = abs(params - target) / target
    print(f"[budget] family={family} target={target} params={params} rel_err={rel_err:.3%}")
    assert rel_err <= 0.05


@pytest.mark.parametrize("target", [10_000, 30_000])
@pytest.mark.parametrize("family", STRUCTURED_FAMILIES)
def test_structured_budget_covers_psi_pi_head_and_context(family, target):
    """Positive: the four parts of a structured cell sum to the whole matched
    budget -- nothing sits outside the accounting.

    Negative, in the same test because it is the same measurement: a budget
    that counted only the CONTEXT PRODUCER would land far outside +/-5%.
    psi/pi/head are not a rounding term here, and the gap is printed so the
    claim is measured, not asserted.

    Measured share of psi+pi+head across the six (family, target) pairs:
    12.9% .. 37.9% of the budget -- so dropping them as a group is always
    detectable. Dropping psi ALONE is 4.3% .. 10.0%, i.e. detectable at five
    of the six pairs and not at ssm_k/30k; that single-parameterization hole
    is why the assertion below is stated on the group, not on psi alone.
    """
    model = _build(family, target)

    def numel(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {n: numel(getattr(model, n)) for n in ("psi", "pi", "head", "backbone")}
    assert sum(parts.values()) == total  # exact: no parameter is unaccounted for

    lift_share = (parts["psi"] + parts["pi"] + parts["head"]) / target
    backbone_only_err = abs(parts["backbone"] - target) / target
    print(
        f"[budget] {family} target={target} total={total} parts={parts} "
        f"psi+pi+head = {lift_share:.1%} of budget, "
        f"backbone-only would be {backbone_only_err:.1%} off target"
    )
    assert backbone_only_err > 0.05  # omitting psi/pi/head breaks the contract
    assert lift_share > 0.05  # ... and they are why, not a rounding term


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_flops_per_step_positive(family):
    model = _build(family)
    flops = count_flops_per_step(model, t=1)
    print(f"[budget] flops_per_step family={family} t=1 -> {flops}")
    assert flops > 0


@pytest.mark.parametrize("family", _ATTENTION_FAMILIES)
def test_attention_arm_flops_strictly_increasing_in_t(family):
    model = _build(family)
    flops = [count_flops_per_step(model, t=t) for t in (0, 1, 5, 50)]
    print(f"[budget] {family} flops per t -> {flops}")
    assert flops == sorted(flops)
    assert len(set(flops)) == len(flops)  # strictly increasing, not just non-decreasing


@pytest.mark.parametrize("family", _CONSTANT_COST_FAMILIES)
def test_non_attention_arm_flops_do_not_depend_on_t(family):
    # the mirror negative of the test above: t is accepted for a uniform
    # signature and must be IGNORED by the five markovian/recurrent
    # families. A t-dependence leaking in here would silently inflate their
    # trajectory bills and break the affine-growth claim below.
    model = _build(family)
    flops = {count_flops_per_step(model, t=t) for t in (0, 1, 5, 50)}
    print(f"[budget] {family} flops across t -> {flops}")
    assert len(flops) == 1


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_analytic_flops_bracket_the_matmuls_torch_actually_dispatches(family):
    """Positive, with a reference INDEPENDENT of this repo (REGLES 7.1):
    ``torch.utils.flop_counter.FlopCounterMode`` counts the FLOPs of the
    matmul-family ops PyTorch actually dispatches while ``step`` runs, at
    the same 2-FLOPs-per-MAC convention ``budget.py`` declares.

    It ignores exactly what ``budget.py`` declares out of scope (elementwise
    adds, softmax, LayerNorm, GELU, the trig/exp block assembly), so the
    analytic count must SIT ABOVE it by that declared slack and no more:

      - a Linear or einsum missing from the analytic count (a forgotten
        head, a forgotten second W_c call, ``t`` instead of ``t + 1`` in the
        attention contraction) puts the analytic count BELOW the measured
        one -> red;
      - a term counted twice, or a block-diagonal operator counted at its
        sparse instead of its executed dense shape, pushes it far above the
        slack -> red.

    Measured at ``t = 3``, i.e. with a non-empty KV cache, so the attention
    arms' ``t``-dependence is inside the bracket rather than assumed away.

    ONE declared deviation, and it is corrected here rather than tolerated:
    ``koopman_ssm``'s bilinear interaction is written
    ``einsum("bj,jkl,bl->bk", u, B_j, h)``, which torch executes in two
    contractions -- it MATERIALIZES ``sum_j u_j B_j`` (``d_u * d_h^2`` MACs)
    and only then applies it to ``h`` (another ``d_h^2``). ``budget.py``
    counts the first term only, calling it "the dominant matmul-equivalent
    cost" in ``_koopman_flops``'s docstring. That is a pre-existing declared
    approximation of the historical class (never a factorial cell,
    PROTOCOL.md section 13.1), not a defect introduced here; the missing
    ``2 * d_h^2`` is added back below so the deviation is pinned with its
    exact size instead of being hidden by a looser tolerance.
    """
    from torch.utils.flop_counter import FlopCounterMode

    t = 3
    model = _build(family).eval()
    h = torch.randn(1, model.d_h)
    u = torch.randn(1, model.d_u)
    state = PredictorState()
    with torch.no_grad():
        for _ in range(t):
            _, state = model.step(h, u, state)
        counter = FlopCounterMode(display=False)
        with counter:
            model.step(h, u, state)
    measured = counter.get_total_flops()
    analytic = count_flops_per_step(model, t)
    if family == "koopman_ssm" and model.regime == "bilinear":
        analytic += 2 * model.d_h * model.d_h  # the einsum's materialization step
    slack = (analytic - measured) / analytic
    print(
        f"[flops oracle] {family} t={t} torch-dispatched matmul FLOPs={measured} "
        f"analytic={analytic} declared slack={slack:.2%}"
    )
    assert measured > 0
    assert analytic >= measured  # never undercount what actually runs
    assert slack < 0.06  # and never exceed the declared elementwise slack


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
@pytest.mark.parametrize("family", _ATTENTION_FAMILIES)
def test_attention_arm_trajectory_flops_strictly_superlinear(family, H):
    # STRICT superlinearity at the matched budget, stated as an exact
    # integer property instead of a ratio threshold: the second difference
    # over H, 2H, 3H is c*H^2 > 0 for a quadratic bill and exactly 0 for an
    # affine one. It cannot be satisfied by an affine formula at any width,
    # so it holds at the 10k budget where the ratio test would not bite.
    model = _build(family)
    f_h, f_2h, f_3h = (count_flops_per_trajectory(model, k * H) for k in (1, 2, 3))
    second_diff = (f_3h - f_2h) - (f_2h - f_h)
    print(
        f"[budget] {family} trajectory H={H}: {f_h}, {f_2h}, {f_3h} "
        f"second difference = {second_diff} (affine would be 0)"
    )
    assert second_diff > 0
    assert f_2h > 2 * f_h  # strictly more than doubling the horizon costs


@pytest.mark.parametrize("H", [50, 100])
@pytest.mark.parametrize("family", _CONSTANT_COST_FAMILIES)
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


@pytest.mark.parametrize("family", ("koopman_ssm", *STRUCTURED_FAMILIES))
def test_family_with_a_physical_zoh_step_refuses_to_default_it(family):
    # delta has no sane default for ANY family that carries one: the
    # structured arms get the same treatment koopman_ssm already had, so a
    # config that forgets it fails loudly instead of training a cell whose
    # discretization step was invented by the budget matcher.
    with pytest.raises(ValueError, match="delta"):
        match_budget(family, D_Z, D_U, target_params=10_000)


def test_kwargs_recipe_is_closed_over_the_registry(monkeypatch):
    # NEGATIVE, reached by mutating the registry IN MEMORY. _kwargs_for used
    # to end in a bare `else` that WAS the koopman_ssm branch: a family added
    # to the registry and forgotten there was silently built with Koopman's
    # kwargs. Registering a family with no recipe must now raise.
    import sld.predictors.registry as registry_mod
    from sld.training.budget import _kwargs_for

    monkeypatch.setitem(registry_mod._FAMILY_CONSTRUCTORS, "brand_new", MLPPredictor)
    with pytest.raises(ValueError, match="no kwargs recipe"):
        _kwargs_for("brand_new", 8, D_Z, D_U, {})


def test_out_of_reach_budget_raises():
    with pytest.raises(ValueError):
        match_budget("mlp", D_Z, D_U, target_params=10)


# -- Travail 3 (PROTOCOL.md section 19.2): r=2 is already a declarable knob --
#
# Established, not implemented: `r` is one of `_STRUCTURED_KEYS =
# frozenset({"delta", "r"})` (sld.training.trainer._FAMILY_CONFIG_KEYS,
# trainer.py) for the three structured families, so it already flows
# `Trainer.fit`'s `passthrough = {k: cfg[k] for k in
# _FAMILY_CONFIG_KEYS[cfg["family"]] if k in cfg}` -> `get_predictor(family,
# d_z, d_u, target_params=..., **passthrough)` -> `match_budget(family, d_z,
# d_u, target_params=..., **overrides)` -> here, `_kwargs_for`'s
# `r=fixed.get("r", d_z)` line (structured.py's own module docstring already
# names this exact call site). A D3 config declares `r: 2` directly in an
# arm's `overrides:` (or `base:`, if every arm in that config is a
# structured family) -- no new key, no new call site. This section proves
# it bites, at the SAME call site (`get_predictor`) `Trainer.fit` uses.


def test_r_equals_2_declared_via_get_predictor_gives_2r_equals_4_operator():
    # POSITIVE: r=2 (PROTOCOL.md section 19.2's D3 control, 2r = d = 4 for
    # d_z=4) reaches the constructed StructuredPredictor's own `r`/`d_h`,
    # and the REAL operator local_operator() returns is (2r, 2r) = (4, 4) --
    # not merely a constructor attribute, the thing the spectral/action
    # bounds of 19.2 actually read.
    from sld.predictors.registry import get_predictor
    from sld.predictors.structured import MLPKPredictor

    d_z, d_u = 4, 1
    model = get_predictor("mlp_k", d_z, d_u, target_params=2_000, delta=0.05, r=2)
    assert isinstance(model, MLPKPredictor)
    assert model.r == 2
    assert model.d_h == 4  # d_h = 2r

    h = torch.zeros(1, model.d_h, dtype=torch.get_default_dtype())
    u = torch.zeros(1, d_u, dtype=torch.get_default_dtype())
    A_bar, B_bar = model.local_operator(h, u, PredictorState())
    print(f"[budget 19.2] r=2 explicit: A_bar.shape={tuple(A_bar.shape)} (want (1, 4, 4))")
    assert A_bar.shape == (1, 4, 4)
    assert B_bar.shape == (1, 4, d_u)


def test_r_default_is_d_z_giving_2r_equals_8_operator_the_contrast():
    # NEGATIVE (contrast, same measurement): the default `r` every OTHER run
    # in this repo gets through get_predictor is d_z (the 2x lifting ratio
    # koopman_ssm gets from n_pairs=d_z, structured.py's own module
    # docstring) -- at d_z=4 that is r=4, 2r=8, NOT 4. This is the exact
    # ill-posedness PROTOCOL.md section 19.2 names ("eight learned
    # eigenvalues against four physical ones") and the reason D3 must
    # declare r=2 explicitly rather than relying on the default.
    from sld.predictors.registry import get_predictor
    from sld.predictors.structured import MLPKPredictor

    d_z, d_u = 4, 1
    model = get_predictor("mlp_k", d_z, d_u, target_params=2_000, delta=0.05)  # no r=...
    assert isinstance(model, MLPKPredictor)
    assert model.r == d_z == 4
    assert model.d_h == 8  # 2r = 8 != d = 4

    h = torch.zeros(1, model.d_h, dtype=torch.get_default_dtype())
    u = torch.zeros(1, d_u, dtype=torch.get_default_dtype())
    A_bar, _ = model.local_operator(h, u, PredictorState())
    print(f"[budget 19.2] r=default(d_z)=4: A_bar.shape={tuple(A_bar.shape)} (contrast: (1, 8, 8), NOT 4)")
    assert A_bar.shape == (1, 8, 8)


def test_r_is_a_declared_trainer_config_key_for_every_structured_family():
    # the config-key surface itself, pinned: `r` is legal (and, per
    # trainer.py's canonicalization, defaults to d_z when omitted -- see
    # test_structured_family_config_is_canonicalized_and_delta_is_required
    # in test_trainer.py) for exactly the three structured families, so a
    # config's `overrides: {r: 2}` on an `mlp_k`/`transformer_k`/`ssm_k` arm
    # is a legal, hashed Trainer config key -- not a knob this wave has to
    # invent.
    from sld.training import trainer as trainer_mod

    for family in STRUCTURED_FAMILIES:
        assert "r" in trainer_mod._FAMILY_CONFIG_KEYS[family]
    for family in ("mlp", "transformer", "min_ssm", "koopman_ssm"):
        assert "r" not in trainer_mod._FAMILY_CONFIG_KEYS[family]


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
