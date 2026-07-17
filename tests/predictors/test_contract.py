"""The common contract suite (predictors/base.py), parameterized over the
four predictor families: shapes, state threading, local_operator shapes,
frozen-context coherence against autograd, and rollout determinism.
"""

from __future__ import annotations

import pytest
import torch

from sld.predictors.base import PredictorState
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.transformer import CausalTransformerPredictor

D_H, D_U = 4, 2
DELTA = 0.05


def _mlp() -> MLPPredictor:
    return MLPPredictor(d_h=D_H, d_u=D_U, hidden=(16, 16))


def _transformer() -> CausalTransformerPredictor:
    return CausalTransformerPredictor(d_h=D_H, d_u=D_U, d_model=8)


def _min_ssm() -> SelectiveSSMPredictor:
    model = SelectiveSSMPredictor(d_h=D_H, d_u=D_U, d_state=8)
    # ponytail: default nn.Linear init gives this family's h-residual loop
    # (h -> c -> b_t/v_t -> m_next -> W_out -> h_next) a spectral radius > 1
    # around h=0 at these dims -- verified empirically over 300 (seed, data)
    # combinations, ALL diverging a raw 50-step rollout regardless of input
    # scale (a finite-time blowup in the untrained dynamics, not an input-
    # magnitude effect: base.py disclaims any stability guarantee on h
    # itself). Downscaling the loop weights (same "adjust untrained init in
    # the test" pattern test_koopman_ssm.py uses for its own gates) keeps
    # the contract's "all finite" check about the STEP contract, not an
    # accident of default init variance.
    with torch.no_grad():
        for lin in (model.W_c, model.W_delta, model.W_b, model.W_in, model.W_out):
            lin.weight.mul_(0.3)
            lin.bias.mul_(0.3)
    return model


def _koopman(regime: str) -> KoopmanSSMPredictor:
    # d_z == D_H is an arbitrary valid choice: lift/project are never
    # exercised by these contract tests, only step/rollout in H-space
    return KoopmanSSMPredictor(
        d_z=D_H, n_pairs=2, d_u=D_U, delta=DELTA, regime=regime, psi_hidden=(8,), pi_hidden=(8,)
    )


FAMILY_BUILDERS = {
    "mlp": _mlp,
    "transformer": _transformer,
    "min_ssm": _min_ssm,
    "koopman_ssm": lambda: _koopman("bilinear"),
}
FAMILY_NAMES = tuple(FAMILY_BUILDERS)

# local_operator families: min_ssm plus the three koopman regimes
OPERATOR_FAMILIES = ("min_ssm", "koopman_constant", "koopman_bilinear", "koopman_selective")


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

    # koopman_selective: freeze c = W_c([h, u]) and differentiate _frozen_step
    c = model.W_c(torch.cat([h, u], dim=-1)).detach()

    def frozen_step(h_b, u_b, c_b):
        return model._frozen_step(h_b.unsqueeze(0), u_b.unsqueeze(0), c_b.unsqueeze(0)).squeeze(0)

    worst = 0.0
    for b in range(B):
        jac_h = torch.func.jacrev(frozen_step, argnums=0)(h[b], u[b], c[b])
        worst = max(worst, _rel_err(jac_h, A[b]))
    print(f"[coherence] {name} worst rel err = {worst:.3e}")
    assert worst < 1e-5


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
