"""The common contract suite (predictors/base.py), parameterized over the
seven predictor families: shapes, state threading, local_operator shapes,
frozen-context coherence against autograd, and rollout determinism.
"""

from __future__ import annotations

import pytest
import torch

from sld.predictors.base import PredictorState
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.structured import MLPKPredictor, SSMKPredictor, TransformerKPredictor
from sld.predictors.transformer import CausalTransformerPredictor

D_H, D_U = 4, 2
DELTA = 0.05


def _mlp() -> MLPPredictor:
    return MLPPredictor(d_h=D_H, d_u=D_U, hidden=(16, 16))


def _transformer() -> CausalTransformerPredictor:
    return CausalTransformerPredictor(d_h=D_H, d_u=D_U, d_model=8)


def _min_ssm() -> SelectiveSSMPredictor:
    # the DEFAULT construction must satisfy the finiteness contract as-is:
    # min_ssm zero-inits its residual readout W_out precisely so an untrained
    # model is finite over any horizon (adversarial review round 1 -- the
    # earlier default init made the h-residual loop expansive and this test
    # had to downscale weights, which masked the defect instead of fixing it)
    return SelectiveSSMPredictor(d_h=D_H, d_u=D_U, d_state=8)


def _koopman(regime: str) -> KoopmanSSMPredictor:
    # d_z == D_H is an arbitrary valid choice: lift/project are never
    # exercised by these contract tests, only step/rollout in H-space
    return KoopmanSSMPredictor(
        d_z=D_H, n_pairs=2, d_u=D_U, delta=DELTA, regime=regime, psi_hidden=(8,), pi_hidden=(8,)
    )


# the three structured arms: r = D_H // 2 so the lifted state is D_H wide
# and the whole contract suite runs in the SAME H-space as the other four.
# d_z is arbitrary here (lift/project are never exercised by these tests).
def _structured(cls):
    return lambda: cls(d_z=D_H, d_u=D_U, delta=DELTA, r=D_H // 2, width=8)


FAMILY_BUILDERS = {
    "mlp": _mlp,
    "transformer": _transformer,
    "min_ssm": _min_ssm,
    "koopman_ssm": lambda: _koopman("bilinear"),
    "mlp_k": _structured(MLPKPredictor),
    "transformer_k": _structured(TransformerKPredictor),
    "ssm_k": _structured(SSMKPredictor),
}
FAMILY_NAMES = tuple(FAMILY_BUILDERS)


def test_family_builders_cover_the_whole_registry():
    # this dict is a LOCAL duplicate of the registry: a family added there
    # and forgotten here would be covered by no contract test at all, with
    # the suite still green. That is the failure mode this phase keeps
    # hitting, so the two are pinned together.
    from sld.predictors.registry import FAMILIES

    assert set(FAMILY_NAMES) == set(FAMILIES)

# local_operator families: min_ssm plus the three koopman regimes
OPERATOR_FAMILIES = ("min_ssm", "koopman_constant", "koopman_bilinear", "lpv_selective")


def _operator_builder(name: str):
    if name == "min_ssm":
        return _min_ssm

    regime = name.split("_", 1)[1]  # "koopman_constant" -> "constant"

    def build():
        model = _koopman(regime)
        if regime == "bilinear":
            # B_j initializes to all zeros (the declared affine point,
            # koopman_ssm.py); left untouched, the interaction term
            # sum_j u_j (B_bar_j h) is identically zero and this regime's
            # local_operator/coherence checks would be vacuously equal to
            # "constant" -- override with nonzero values so the interaction
            # term (and its Jacobian contribution) is actually exercised,
            # same pattern test_koopman_ssm.py uses for its own bilinear tests
            gen = torch.Generator().manual_seed(42)
            with torch.no_grad():
                model.B_j.copy_(
                    0.3 * torch.randn(model.B_j.shape, generator=gen, dtype=model.B_j.dtype)
                )
        return model

    return build


def _make(builder, seed: int):
    # nn.Module init reads torch's global RNG internally with no generator
    # hook; fork_rng() seeds it locally for this construction only and
    # restores the prior global state on exit, so tests stay independent.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return builder()


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30)).item()


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_step_and_rollout_shapes_finite(family):
    model = _make(FAMILY_BUILDERS[family], seed=0)
    B, T = 3, 50
    gen = torch.Generator().manual_seed(1)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    h1, state = model.step(h0, u_seq[0], PredictorState())
    assert h1.shape == (B, D_H)
    assert isinstance(state, PredictorState)

    out = model.rollout(h0, u_seq)
    assert out.shape == (T, B, D_H)
    assert torch.all(torch.isfinite(out))
    print(f"[contract] {family} rollout shape={tuple(out.shape)} all finite")


# ---------------------------------------------------------------------------
# state threading
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_state_threading_matches_rollout(family):
    model = _make(FAMILY_BUILDERS[family], seed=2)
    B, T = 2, 5
    gen = torch.Generator().manual_seed(3)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    h, state = h0, PredictorState()
    manual = []
    for t in range(T):
        h, state = model.step(h, u_seq[t], state)
        manual.append(h)
    manual_out = torch.stack(manual, dim=0)

    rollout_out = model.rollout(h0, u_seq)
    err = (manual_out - rollout_out).abs().max().item()
    print(f"[contract] {family} manual-vs-rollout max abs diff = {err:.3e}")
    torch.testing.assert_close(manual_out, rollout_out, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# rollout vs. a hand-recomputed AFFINE predictor (REGLES 7.1: independent
# reference, not the object against itself)
# ---------------------------------------------------------------------------
def test_affine_predictor_rollout_matches_hand_computed_dynamics():
    """``test_state_threading_matches_rollout`` above is an INTERNAL
    consistency check: it compares ``rollout`` to a loop of ``step`` calls on
    the SAME model, so a bug shared by both code paths (e.g. an off-by-one
    on the input index baked into ``step`` itself) would not be caught.

    Here the model is forced to realize an EXACT affine map
    ``h_next = A h + B u + c`` with ``A``, ``B``, ``c`` posed by hand
    (``A`` non-diagonal, ``c != 0``, ``B != 0``), then ``rollout`` is checked
    against a loop written INDEPENDENTLY in this test -- it never calls
    ``model.step`` or ``model.rollout`` for the reference side.

    Construction: ``MLPPredictor(hidden=())`` collapses ``net`` to a SINGLE
    ``nn.Linear`` with no activation -- confirmed by inspection below
    (mlp.py builds ``dims = (d_h+d_u, d_h)``, and the ``if i < len(dims)-2``
    guard that appends ``GELU`` is never true for a 1-layer stack). With
    weights set by hand, ``step``'s ``h_next = h + net(cat([h, u]))``
    becomes EXACTLY ``h_next = (I + W_h) h + W_u u + bias``, i.e.
    ``A = I + W_h``, ``B = W_u``, ``c = bias``.

    ``A`` is upper-bidiagonal, hence triangular: its eigenvalues are its
    diagonal entries (all ``0.5``), so its spectral radius is EXACTLY
    ``0.5 < 1`` (computed below, not just claimed) -- the ``H = 40``-step
    trajectory is guaranteed bounded, not "happened not to overflow".
    """
    A = torch.tensor(
        [
            [0.5, 0.1, 0.0, 0.0],
            [0.0, 0.5, 0.1, 0.0],
            [0.0, 0.0, 0.5, 0.1],
            [0.0, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float64,
    )
    B = torch.tensor(
        [[0.3, -0.2], [0.1, 0.4], [-0.3, 0.2], [0.2, -0.1]], dtype=torch.float64
    )
    c = torch.tensor([0.05, -0.05, 0.02, -0.02], dtype=torch.float64)
    spectral_radius = torch.linalg.eigvals(A).abs().max().item()
    assert spectral_radius == pytest.approx(0.5, abs=1e-12)

    model = MLPPredictor(d_h=D_H, d_u=D_U, hidden=()).double()
    assert len(model.net) == 1 and isinstance(model.net[0], torch.nn.Linear)
    linear = model.net[0]
    with torch.no_grad():
        linear.weight[:, :D_H].copy_(A - torch.eye(D_H, dtype=torch.float64))
        linear.weight[:, D_H:].copy_(B)
        linear.bias.copy_(c)

    Batch, H = 3, 40
    gen = torch.Generator().manual_seed(2024)
    h0 = torch.randn(Batch, D_H, generator=gen, dtype=torch.float64)
    u_seq = torch.randn(H, Batch, D_U, generator=gen, dtype=torch.float64)

    # independent hand loop -- NOT model.step, NOT model.rollout
    manual = []
    h = h0.clone()
    for t in range(H):
        h = h @ A.T + u_seq[t] @ B.T + c
        manual.append(h)
    manual_out = torch.stack(manual, dim=0)

    rollout_out = model.rollout(h0, u_seq)
    assert torch.isfinite(rollout_out).all()  # the H=40 trajectory stays bounded
    err = (manual_out - rollout_out).abs().max().item()
    print(f"[contract] affine rollout vs hand loop: max abs err = {err:.3e} over H={H} steps")
    assert err < 1e-12  # short float64 algebra, R3

    # Negative: shift the recomposition by one input index (u_{t+1} instead
    # of u_t) -- the actual off-by-one failure mode this test guards
    # against. Must be REJECTED by the same tolerance.
    manual_shifted = []
    h = h0.clone()
    for t in range(H - 1):
        h = h @ A.T + u_seq[t + 1] @ B.T + c
        manual_shifted.append(h)
    manual_shifted_out = torch.stack(manual_shifted, dim=0)
    err_shifted = (manual_shifted_out - rollout_out[: H - 1]).abs().max().item()
    print(f"[contract] shifted-index negative: max abs err = {err_shifted:.3e}")
    assert err_shifted > 1e-3  # order-one mismatch, nowhere near the tolerance


# ---------------------------------------------------------------------------
# local_operator: None for the generic families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["mlp", "transformer"])
def test_local_operator_none_for_generic_families(family):
    model = _make(FAMILY_BUILDERS[family], seed=4)
    gen = torch.Generator().manual_seed(5)
    h = torch.randn(3, D_H, generator=gen)
    u = torch.randn(3, D_U, generator=gen)
    assert model.local_operator(h, u, PredictorState()) is None


# ---------------------------------------------------------------------------
# local_operator: shapes for the structured families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", OPERATOR_FAMILIES)
def test_local_operator_shapes_finite(name):
    model = _make(_operator_builder(name), seed=6)
    B = 3
    gen = torch.Generator().manual_seed(7)
    h = torch.randn(B, D_H, generator=gen)
    u = torch.randn(B, D_U, generator=gen)
    A, Bm = model.local_operator(h, u, PredictorState())
    d_r = model.d_state if name == "min_ssm" else model.d_h
    assert A.shape == (B, d_r, d_r)
    assert Bm.shape == (B, d_r, D_U)
    assert torch.all(torch.isfinite(A)) and torch.all(torch.isfinite(Bm))
    print(f"[contract] {name} local_operator shapes A={tuple(A.shape)} B={tuple(Bm.shape)}")


# ---------------------------------------------------------------------------
# local_operator: frozen-context coherence gate vs torch.func.jacrev
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", OPERATOR_FAMILIES)
def test_local_operator_coherence(name):
    model = _make(_operator_builder(name), seed=8).double()
    B = 3
    gen = torch.Generator().manual_seed(9)
    h = torch.randn(B, D_H, generator=gen, dtype=torch.float64)
    u = torch.randn(B, D_U, generator=gen, dtype=torch.float64)

    if name == "min_ssm":
        m = torch.randn(B, model.d_state, generator=gen, dtype=torch.float64)
        c = model.W_c(torch.cat([h, m], dim=-1)).detach()
        A, Bm = model.local_operator(h, u, PredictorState(m=m))

        def frozen_recurrence(m_b, h_b, u_b, c_b):
            return model._frozen_recurrence(
                m_b.unsqueeze(0), h_b.unsqueeze(0), u_b.unsqueeze(0), c_b.unsqueeze(0)
            ).squeeze(0)

        worst = 0.0
        for b in range(B):
            jac_A = torch.func.jacrev(frozen_recurrence, argnums=0)(m[b], h[b], u[b], c[b])
            jac_B = torch.func.jacrev(frozen_recurrence, argnums=2)(m[b], h[b], u[b], c[b])
            worst = max(worst, _rel_err(jac_A, A[b]), _rel_err(jac_B, Bm[b]))
        print(f"[coherence] {name} worst rel err = {worst:.3e}")
        assert worst < 1e-5
        return

    A, Bm = model.local_operator(h, u, PredictorState())

    if name in ("koopman_constant", "koopman_bilinear"):

        def single_step(h_b, u_b):
            out, _ = model.step(h_b.unsqueeze(0), u_b.unsqueeze(0), PredictorState())
            return out.squeeze(0)

        worst = 0.0
        for b in range(B):
            jac_h = torch.func.jacrev(single_step, argnums=0)(h[b], u[b])
            jac_u = torch.func.jacrev(single_step, argnums=1)(h[b], u[b])
            worst = max(worst, _rel_err(jac_h, A[b]), _rel_err(jac_u, Bm[b]))
        print(f"[coherence] {name} worst rel err = {worst:.3e}")
        assert worst < 1e-5
        return

    # lpv_selective: freeze c = W_c([h, u]) and differentiate _frozen_step
    c = model.W_c(torch.cat([h, u], dim=-1)).detach()

    def frozen_step(h_b, u_b, c_b):
        return model._frozen_step(h_b.unsqueeze(0), u_b.unsqueeze(0), c_b.unsqueeze(0)).squeeze(0)

    worst = 0.0
    for b in range(B):
        jac_h = torch.func.jacrev(frozen_step, argnums=0)(h[b], u[b], c[b])
        # B = dh+/du at fixed c (REGLES 7.1 gap closed): B_bar_t is a real,
        # nontrivial quantity in this regime (koopman_ssm.py:268-271), not
        # structurally empty, so it needs the same independent-jacobian
        # coverage as A above and as the constant/bilinear branch already has
        jac_u = torch.func.jacrev(frozen_step, argnums=1)(h[b], u[b], c[b])
        worst = max(worst, _rel_err(jac_h, A[b]), _rel_err(jac_u, Bm[b]))
    print(f"[coherence] {name} worst rel err = {worst:.3e}")
    assert worst < 1e-5


# ---------------------------------------------------------------------------
# negative counterpart (REGLES 7.1) to the lpv_selective B-branch just
# added above: B_bar_t is real and nontrivial here, so a broken
# implementation that silently flips its sign (a plausible wiring bug --
# same "denied input term" failure mode the packet names) must be REJECTED
# by the same coherence comparison, at the same tolerance. Negating (rather
# than zeroing) keeps the perturbed candidate at Bm's own scale, so
# _rel_err's normalization (divides by the candidate's own norm) stays
# well-conditioned instead of blowing up against a near-zero denominator.
# ---------------------------------------------------------------------------
def test_local_operator_coherence_selective_negative_negated_b():
    model = _make(_operator_builder("lpv_selective"), seed=8).double()
    B = 3
    gen = torch.Generator().manual_seed(9)
    h = torch.randn(B, D_H, generator=gen, dtype=torch.float64)
    u = torch.randn(B, D_U, generator=gen, dtype=torch.float64)

    _, Bm = model.local_operator(h, u, PredictorState())
    c = model.W_c(torch.cat([h, u], dim=-1)).detach()

    def frozen_step(h_b, u_b, c_b):
        return model._frozen_step(h_b.unsqueeze(0), u_b.unsqueeze(0), c_b.unsqueeze(0)).squeeze(0)

    Bm_negated = -Bm  # the bug: input term present but sign-flipped/denied
    worst = 0.0
    for b in range(B):
        jac_u = torch.func.jacrev(frozen_step, argnums=1)(h[b], u[b], c[b])
        worst = max(worst, _rel_err(jac_u, Bm_negated[b]))
    print(f"[coherence-negative] lpv_selective negated-B worst rel err = {worst:.3e}")
    assert worst > 1e-5  # must be REJECTED by the exact tolerance the positive uses


# ---------------------------------------------------------------------------
# determinism: identical rollout calls with fresh state give identical output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_determinism_fresh_state_each_rollout(family):
    model = _make(FAMILY_BUILDERS[family], seed=10)
    B, T = 3, 6
    gen = torch.Generator().manual_seed(11)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    out1 = model.rollout(h0, u_seq)
    out2 = model.rollout(h0, u_seq)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
    print(f"[contract] {family} determinism ok")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
