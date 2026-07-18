"""Tests for the spectral display module (src/sld/audit/spectral.py).

Discipline (REGLES sections 5, 7):
 - float64 for every exactness/gold test; references built INDEPENDENTLY of
   the function under test (direct SVD loops, hand-derived closed forms,
   ``np.linalg.eigvals`` on an independently assembled operator), never
   through the module's own code path.
 - local generators only (``torch.Generator`` / ``np.random.default_rng``);
   no global ``set_seed``. Gold tests compare a model against a reference
   computed FROM the same model's parameters, so they do not depend on the
   ambient torch RNG state.
 - the EDMD gold test is the wave gate: it must be validated in MULTI-STEP
   rollout, never one-step residual alone (the gauge trap -- Wave 3
   spectral-identifiability lesson; REGLES section 7 gold-test doctrine).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sld.audit.spectral import (
    edmd_fit,
    local_spectra_field,
    operator_norm,
    operator_spectrum,
    pseudospectrum_grid,
)
from sld.predictors.base import PredictorModule, PredictorState
from sld.predictors.blocks2x2 import block_diagonal
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.systems.coupled_oscillators import CoupledOscillators

DTYPE = torch.float64


def _sort_ev(ev: np.ndarray) -> np.ndarray:
    """Same deterministic order as spectral.py's own convention (imag, then
    real) -- needed only to align two independently computed eigenvalue
    lists element-wise, not part of the logic under test."""
    order = np.lexsort((ev.real, ev.imag))
    return ev[order]


def _build_constant_koopman() -> KoopmanSSMPredictor:
    return KoopmanSSMPredictor(
        d_z=2, n_pairs=1, d_u=1, delta=0.05, regime="constant",
        psi_hidden=(), pi_hidden=(), dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# (a) GOLD: edmd_fit on the exactly-linear CoupledOscillators system
# ---------------------------------------------------------------------------
def test_gold_edmd_coupled_oscillators():
    dt = 0.05
    system = CoupledOscillators()
    rng = np.random.default_rng(0)
    n_traj, T = 8, 250

    X_list, Y_list = [], []
    for _ in range(n_traj):
        x0 = system.sample_ic(rng)
        u_seq = np.zeros((T, system.action_dim))
        traj = system.integrate(x0, u_seq, dt)  # (T+1, 4), u=0 free relaxation
        X_list.append(traj[:-1])
        Y_list.append(traj[1:])
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    # measurement-noise robustness, not a precision crutch
    X = X + rng.normal(0.0, 1e-8, size=X.shape)
    Y = Y + rng.normal(0.0, 1e-8, size=Y.shape)

    A_hat = edmd_fit(X, Y)

    # (i) eigenvalues of A_hat vs the independent analytic reference
    # exp(dt * continuous eigenvalues) -- CoupledOscillators.analytic_modes()
    # is exact (closed-form eigendecomposition of A_mat), never touching
    # edmd_fit's own lstsq path.
    learned = _sort_ev(np.linalg.eigvals(A_hat))
    expected = _sort_ev(np.exp(dt * system.analytic_modes()))
    rel = np.abs(learned - expected) / np.abs(expected)
    print(f"[gold edmd] learned  = {learned}")
    print(f"[gold edmd] expected = {expected}")
    print(f"[gold edmd] rel error = {rel}")
    assert np.all(rel < 1e-3), f"eigenvalue gate: rel={rel}"

    # (ii) MANDATORY multi-step validation -- the gauge trap (Wave 3
    # spectral-identifiability lesson, REGLES section 7): a one-step lstsq
    # fit minimizes ||X @ A.T - Y|| on the TRAINING PAIRS themselves, which
    # can look perfect without the fitted A actually reproducing the
    # dynamics multiple steps out (e.g. directions barely excited by the u=0
    # relaxation can be near-arbitrary in the fit yet invisible to a one-step
    # residual). The only real test that "A_hat IS the dynamics" is a fully
    # free multi-step rollout, from a FRESH initial condition the fit never
    # saw, checked against ground truth it was never fit to -- never
    # re-scored against the training pairs.
    x0_test = system.sample_ic(rng)  # unseen IC
    u_test = np.zeros((250, system.action_dim))
    traj_test = system.integrate(x0_test, u_test, dt)
    H = 50
    x_true = traj_test[: H + 1]
    x_hat = np.empty_like(x_true)
    x_hat[0] = x_true[0]
    for k in range(H):
        x_hat[k + 1] = A_hat @ x_hat[k]
    rms_true = np.sqrt(np.mean(x_true**2))
    errors = np.linalg.norm(x_hat - x_true, axis=1)
    max_rel_err = np.max(errors) / rms_true
    print(f"[gold edmd] multi-step max rel error over H={H} = {max_rel_err:.3e}")
    assert max_rel_err < 1e-3


def test_edmd_fit_rank_deficient_raises():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 1))
    X = x @ np.array([[1.0, 2.0, -1.0, 0.5]])  # colinear columns: rank 1, d=4
    Y = np.roll(X, -1, axis=0)
    with pytest.raises(ValueError):
        edmd_fit(X, Y)


def test_edmd_fit_ill_conditioned_raises():
    # Full numerical rank but cond(X) ~ 1e12: lstsq would silently return
    # the min-norm solution on the unresolved direction and the spectrum
    # would be an artifact of that hidden regularization -- the guard must
    # refuse loudly, not just on EXACT rank deficiency.
    rng = np.random.default_rng(3)
    base = rng.normal(size=(200, 4))
    U, _, Vt = np.linalg.svd(base, full_matrices=False)
    X = U @ np.diag([1.0, 1.0, 1.0, 1e-12]) @ Vt  # cond = 1e12
    A_true = np.diag([0.9, 0.8, 0.7, 0.6])
    Y = X @ A_true.T
    with pytest.raises(ValueError):
        edmd_fit(X, Y)
    # the same data passes when the caller consciously loosens the gate
    A_loose = edmd_fit(X, Y, cond_max=1e15)
    assert A_loose.shape == (4, 4)


def test_edmd_fit_nonfinite_raises():
    X = np.eye(4)
    Y = np.eye(4)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        edmd_fit(X_bad, Y)


# ---------------------------------------------------------------------------
# (c) pseudospectrum on a non-normal 2x2 (Jordan-like) reference
# ---------------------------------------------------------------------------
A_NONNORMAL = np.array([[0.9, 10.0], [0.0, 0.9]])


def test_pseudospectrum_grid_matches_direct_svd():
    grid = np.array([[0.9 + 0j, 1.0 + 0j], [0.5 + 0.3j, 1.2 - 0.3j]])
    direct = np.empty_like(grid, dtype=np.float64)
    for idx in np.ndindex(grid.shape):
        M = A_NONNORMAL - grid[idx] * np.eye(2)
        direct[idx] = np.linalg.svd(M, compute_uv=False)[-1]
    got = pseudospectrum_grid(A_NONNORMAL, grid)
    np.testing.assert_allclose(got, direct, atol=1e-12)


def test_pseudospectrum_at_eigenvalue_is_zero():
    # 0.9 is the (double) eigenvalue of A_NONNORMAL
    sigma_min = pseudospectrum_grid(A_NONNORMAL, np.array([0.9 + 0j]))[0]
    assert sigma_min < 1e-9


def test_pseudospectrum_nonnormality_at_distance_0p1():
    # lambda = 1.0 sits at distance 0.1 from the spectrum {0.9, 0.9}, yet
    # sigma_min collapses to ~1e-3 -- the non-normality signature.
    # Hand derivation, independent of pseudospectrum_grid/SVD: for
    # M = A - lambda*I = [[a, b], [0, a]] with a = 0.9 - 1.0 = -0.1, b = 10,
    # M^T M = [[a^2, a*b], [a*b, a^2 + b^2]] (worked out by hand), a 2x2
    # symmetric matrix whose eigenvalues solve
    #   t^2 - (2a^2 + b^2) t + a^4 = 0
    # (trace = 2a^2+b^2, det = a^2(a^2+b^2) - (ab)^2 = a^4). sigma_min is the
    # sqrt of the smaller root.
    a, b = 0.9 - 1.0, 10.0
    tr = 2 * a**2 + b**2
    det4 = 4 * a**4
    lam_min = (tr - np.sqrt(tr**2 - det4)) / 2.0
    sigma_min_hand = np.sqrt(lam_min)
    print(f"[c-iii] hand-derived sigma_min(lambda=1.0) = {sigma_min_hand:.6e}")

    sigma_min_computed = pseudospectrum_grid(A_NONNORMAL, np.array([1.0 + 0j]))[0]
    assert abs(sigma_min_computed - sigma_min_hand) < 1e-9
    # the actual claim under test: pseudospectral distance (~1e-3) is orders
    # of magnitude below the eigenvalue distance (0.1) -- non-normality.
    assert sigma_min_hand < 1e-2
    assert sigma_min_hand < 0.1 / 10.0


def test_pseudospectrum_transient_growth_reference():
    # matrix_power used RAW here is deliberate and safe: this is a fixed,
    # hand-picked 2x2 test reference with spectral radius 0.9 < 1 (no
    # overflow risk over 200 steps) used purely to characterize the display
    # function -- it is not a learned operator audited for stability, so the
    # "never raw matrix power beyond ~20 steps" rule (REGLES section 5),
    # aimed at avoiding misleading/overflowing conclusions about a MODEL's
    # stability, does not apply to this independent test oracle.
    norms = np.array(
        [operator_norm(np.linalg.matrix_power(A_NONNORMAL, k)) for k in range(1, 61)]
    )
    peak = norms.max()
    peak_k = int(np.argmax(norms)) + 1
    print(f"[c-iv] peak={peak:.3f} at k={peak_k}, ||A^1||={norms[0]:.4f}")
    assert peak > 30.0
    assert 5 <= peak_k <= 15
    assert abs(norms[0] - 10.08) < 0.01

    norm_200 = operator_norm(np.linalg.matrix_power(A_NONNORMAL, 200))
    print(f"[c-iv] ||A^200||={norm_200:.3e}, ratio to peak={norm_200 / peak:.3e}")
    assert norm_200 < 1e-3 * peak


# ---------------------------------------------------------------------------
# (d) operator_norm
# ---------------------------------------------------------------------------
def test_operator_norm_diag_exact():
    A = torch.diag(torch.tensor([3.0, 1.0], dtype=DTYPE))
    assert operator_norm(A) == pytest.approx(3.0, abs=1e-12)


def test_operator_norm_matches_numpy_2norm():
    rng = np.random.default_rng(2)
    M = rng.normal(size=(5, 5))
    assert abs(operator_norm(M) - np.linalg.norm(M, 2)) < 1e-12


def test_operator_norm_rejects_complex():
    # numpy would cast [[1j]] to float64 by DISCARDING the imaginary part
    # (norm 1.0 silently reported as 0.0) -- the audit refuses instead.
    with pytest.raises(TypeError):
        operator_norm(np.array([[1.0j]]))
    with pytest.raises(TypeError):
        operator_norm(torch.tensor([[1.0j]]))


def test_operator_norm_rejects_finite_input_with_overflowed_norm():
    with pytest.raises(ValueError):
        operator_norm(np.full((2, 2), 1e308))


# ---------------------------------------------------------------------------
# (e) pseudospectrum guards
# ---------------------------------------------------------------------------
def test_pseudospectrum_grid_rejects_large_n():
    with pytest.raises(ValueError):
        pseudospectrum_grid(np.zeros((65, 65)), np.array([1.0 + 0j]))


def test_pseudospectrum_grid_rejects_nonsquare():
    with pytest.raises(ValueError):
        pseudospectrum_grid(np.zeros((3, 4)), np.array([1.0 + 0j]))


def test_pseudospectrum_grid_rejects_overflowed_shift():
    with pytest.raises(ValueError):
        pseudospectrum_grid(np.array([[1e308]]), np.array([-1e308 + 0j]))


# ---------------------------------------------------------------------------
# (f) GOLD: operator_spectrum against an independently assembled Koopman A_bar
# ---------------------------------------------------------------------------
def test_operator_spectrum_constant_regime_gold():
    model = _build_constant_koopman()
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(4, model.d_h, generator=gen, dtype=DTYPE)
    u = torch.zeros(model.d_u, dtype=DTYPE)

    got = operator_spectrum(model, h, u)

    # independent reference: assemble A_bar directly via blocks.discretize_zoh
    # + block_diagonal (same helper the Wave 3 gold test uses), not through
    # operator_spectrum's own local_operator call.
    A_bar_blocks, _ = model.blocks.discretize_zoh(model.delta)
    expected = _sort_ev(np.linalg.eigvals(block_diagonal(A_bar_blocks).detach().numpy()))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_operator_spectrum_bilinear_regime_nonvacuous_at_zero_input():
    # B_j initializes to all-zero (the affine point) in koopman_ssm.py -- if
    # left untouched, u=0 turning off the interaction term would be
    # indistinguishable from B_j itself being zero (a vacuous test, the same
    # trap test_koopman_ssm.py's gold test documents). Randomize B_j FIRST so
    # the check demonstrates the interaction term is killed by u=0, not by a
    # zero coupling tensor.
    model = KoopmanSSMPredictor(
        d_z=2, n_pairs=1, d_u=1, delta=0.05, regime="bilinear",
        psi_hidden=(), pi_hidden=(), dtype=DTYPE,
    )
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.B_j.copy_(0.5 * torch.randn(model.B_j.shape, generator=gen, dtype=DTYPE))
    assert model.B_j.abs().max().item() > 0.05, "B_j must be genuinely nonzero"

    h = torch.randn(4, model.d_h, generator=gen, dtype=DTYPE)
    u_zero = torch.zeros(model.d_u, dtype=DTYPE)

    got = operator_spectrum(model, h, u_zero)

    A_bar_blocks, _ = model.blocks.discretize_zoh(model.delta)
    expected = _sort_ev(np.linalg.eigvals(block_diagonal(A_bar_blocks).detach().numpy()))
    np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# (g) operator_spectrum guards
# ---------------------------------------------------------------------------
class _StateDependent(PredictorModule):
    """Local test double: local_operator genuinely varies across probes."""

    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        A = torch.diag_embed(h)  # (batch, d_h, d_h), depends on h per-probe
        B = torch.zeros(h.shape[0], h.shape[1], u.shape[-1], dtype=h.dtype)
        return A, B


def test_operator_spectrum_rejects_state_dependent_operator():
    model = _StateDependent()
    gen = torch.Generator().manual_seed(3)
    h = torch.randn(4, 3, generator=gen, dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


def test_operator_spectrum_rejects_invalid_atol():
    model = _StateDependent()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(8), dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    for atol in (np.nan, np.inf):
        with pytest.raises(ValueError):
            operator_spectrum(model, h, u, atol=atol)


class _BroadcastOperator(PredictorModule):
    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        d_h = h.shape[1]
        A = torch.eye(d_h, dtype=h.dtype).unsqueeze(0)
        B = torch.zeros(1, d_h, u.shape[-1], dtype=h.dtype)
        return A, B


def test_operator_spectrum_rejects_broadcast_shape():
    model = _BroadcastOperator()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(9), dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


class _VectorOperator(PredictorModule):
    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        return torch.ones(h.shape[1], dtype=h.dtype), None


def test_operator_spectrum_rejects_vector_operator():
    model = _VectorOperator()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(12), dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, torch.zeros(1, dtype=DTYPE))


class _ScalarOperator(PredictorModule):
    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        return torch.tensor(1.0, dtype=h.dtype), None


def test_operator_spectrum_rejects_scalar_operator():
    model = _ScalarOperator()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(13), dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, torch.zeros(1, dtype=DTYPE))


class _NonfiniteOperator(PredictorModule):
    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        d_h = h.shape[1]
        A = torch.eye(d_h, dtype=h.dtype).expand(h.shape[0], -1, -1).clone()
        A[1] = torch.nan
        B = torch.zeros(h.shape[0], d_h, u.shape[-1], dtype=h.dtype)
        return A, B


def test_operator_spectrum_rejects_nonfinite_A():
    model = _NonfiniteOperator()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(10), dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


class _OverflowOperator(PredictorModule):
    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        d_h = h.shape[1]
        A = torch.full((h.shape[0], d_h, d_h), 1e308, dtype=h.dtype)
        B = torch.zeros(h.shape[0], d_h, u.shape[-1], dtype=h.dtype)
        return A, B


def test_operator_spectrum_rejects_finite_operator_with_overflowed_eigenvalues():
    model = _OverflowOperator()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(14), dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, torch.zeros(1, dtype=DTYPE))


def test_operator_spectrum_rejects_nonfinite_inputs():
    model = _build_constant_koopman()
    h = torch.randn(3, model.d_h, generator=torch.Generator().manual_seed(15), dtype=DTYPE)
    h[0, 0] = torch.nan
    with pytest.raises(ValueError):
        operator_spectrum(model, h, torch.zeros(model.d_u, dtype=DTYPE))


class _Float32Operator(PredictorModule):
    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        d_h = h.shape[1]
        A = torch.eye(d_h, dtype=torch.float32).expand(h.shape[0], -1, -1)
        B = torch.zeros(h.shape[0], d_h, u.shape[-1], dtype=torch.float32)
        return A, B


def test_float32_local_operator_is_rejected_by_spectral_entry_points():
    model = _Float32Operator()
    h = torch.randn(3, 2, generator=torch.Generator().manual_seed(11), dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)
    with pytest.raises(ValueError):
        local_spectra_field(model, h, torch.zeros(1, 3, 1, dtype=DTYPE))


def test_operator_spectrum_rejects_no_structured_operator():
    model = MLPPredictor(d_h=2, d_u=1).double()
    gen = torch.Generator().manual_seed(4)
    h = torch.randn(4, 2, generator=gen, dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


def test_operator_spectrum_rejects_identical_probes():
    # identical probe rows would make the constancy check vacuous: even the
    # blatantly state-dependent A = diag(h) double would pass it.
    model = _build_constant_koopman()
    h = torch.ones(4, model.d_h, dtype=DTYPE)  # all rows identical
    u = torch.zeros(model.d_u, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


class _MixedScale(PredictorModule):
    """Operator with a huge fixed block AND an order-one state-dependent
    small block: a GLOBAL tolerance (atol * max|A|) would compare these
    equal (threshold ~1.0 at max|A| = 1e8) and certify a state-dependent
    operator; the entrywise relative check must reject it."""

    def step(self, h, u, state):
        return h, state

    def local_operator(self, h, u, state):
        n = h.shape[0]
        A = torch.zeros(n, 2, 2, dtype=h.dtype)
        A[:, 0, 0] = 1e8
        A[:, 1, 1] = h[:, 0]  # varies across probes, order one
        B = torch.zeros(n, 2, u.shape[-1], dtype=h.dtype)
        return A, B


def test_operator_spectrum_rejects_mixed_scale_state_dependence():
    model = _MixedScale()
    h = torch.tensor([[0.5, 0.0], [0.9, 0.0], [0.7, 0.0]], dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


def test_edmd_fit_invalid_cond_max_raises():
    # cond > NaN is False for every cond: a NaN threshold would silently
    # DISABLE the identifiability guard -- it must be refused, like any
    # non-finite or sub-1 threshold.
    X = np.eye(4)
    Y = np.eye(4)
    for bad in (np.nan, np.inf, -5.0, 0.5):
        with pytest.raises(ValueError):
            edmd_fit(X, Y, cond_max=bad)


def test_operator_spectrum_rejects_complex_parameter_module():
    class _ComplexModule(PredictorModule):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0 + 1.0j]))

        def step(self, h, u, state):
            return h, state

        def local_operator(self, h, u, state):
            A = torch.eye(h.shape[1], dtype=torch.float64).expand(
                h.shape[0], h.shape[1], h.shape[1]
            )
            B = torch.zeros(h.shape[0], h.shape[1], u.shape[-1], dtype=torch.float64)
            return A, B

    model = _ComplexModule()
    gen = torch.Generator().manual_seed(7)
    h = torch.randn(3, 2, generator=gen, dtype=DTYPE)
    u = torch.zeros(1, dtype=DTYPE)
    with pytest.raises(ValueError):
        operator_spectrum(model, h, u)


# ---------------------------------------------------------------------------
# (h) local_spectra_field
# ---------------------------------------------------------------------------
def test_local_spectra_field_constant_koopman_matches_operator_spectrum():
    model = _build_constant_koopman()
    A_bar_blocks, _ = model.blocks.discretize_zoh(model.delta)
    expected = _sort_ev(np.linalg.eigvals(block_diagonal(A_bar_blocks).detach().numpy()))

    gen = torch.Generator().manual_seed(0)
    h0 = torch.randn(3, model.d_h, generator=gen, dtype=DTYPE)
    u_seq = torch.zeros(10, 3, model.d_u, dtype=DTYPE)

    times, eigvals = local_spectra_field(model, h0, u_seq, sample_every=2)

    np.testing.assert_array_equal(times, np.array([0, 2, 4, 6, 8]))
    assert eigvals.shape == (5, 3, model.d_h)
    assert eigvals.dtype == np.complex128
    # fixed-operator regime: the field is the SAME spectrum at every probed
    # (t, b) -- broadcast comparison against the independent reference.
    np.testing.assert_allclose(eigvals, np.broadcast_to(expected, eigvals.shape), atol=1e-12)


class _LinearPredictor(PredictorModule):
    """Local test double: step = h @ A0.T, local_operator inherited -> None,
    so local_spectra_field must fall back to the autograd Jacobian."""

    def __init__(self, A0: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("A0", A0)

    def step(self, h, u, state):
        return h @ self.A0.T, state


def test_local_spectra_field_autograd_fallback_gold():
    A0 = torch.tensor([[0.8, 0.1], [-0.2, 0.5]], dtype=DTYPE)
    model = _LinearPredictor(A0)
    assert model.local_operator(
        torch.zeros(1, 2, dtype=DTYPE), torch.zeros(1, 1, dtype=DTYPE), PredictorState()
    ) is None

    gen = torch.Generator().manual_seed(5)
    h0 = torch.randn(3, 2, generator=gen, dtype=DTYPE)
    u_seq = torch.zeros(6, 3, 1, dtype=DTYPE)

    times, eigvals = local_spectra_field(model, h0, u_seq)

    expected = _sort_ev(np.linalg.eigvals(A0.numpy()))
    np.testing.assert_array_equal(times, np.arange(6))
    assert eigvals.shape == (6, 3, 2)
    np.testing.assert_allclose(eigvals, np.broadcast_to(expected, eigvals.shape), atol=1e-10)


def test_local_spectra_field_mlp_smoke():
    model = MLPPredictor(d_h=2, d_u=1).double()
    gen = torch.Generator().manual_seed(6)
    h0 = torch.randn(3, 2, generator=gen, dtype=DTYPE)
    u_seq = torch.zeros(5, 3, 1, dtype=DTYPE)

    times, eigvals = local_spectra_field(model, h0, u_seq)

    np.testing.assert_array_equal(times, np.arange(5))
    assert eigvals.shape == (5, 3, 2)
    assert np.all(np.isfinite(eigvals))


def test_local_spectra_field_rejects_broadcast_operator_shape():
    model = _BroadcastOperator()
    h0 = torch.zeros(3, 2, dtype=DTYPE)
    with pytest.raises(ValueError):
        local_spectra_field(model, h0, torch.zeros(1, 3, 1, dtype=DTYPE))


def test_local_spectra_field_rejects_nonfinite_operator():
    model = _NonfiniteOperator()
    h0 = torch.zeros(3, 2, dtype=DTYPE)
    with pytest.raises(ValueError):
        local_spectra_field(model, h0, torch.zeros(1, 3, 1, dtype=DTYPE))


class _InvalidFallbackStep(PredictorModule):
    def __init__(self, value: float | None = None) -> None:
        super().__init__()
        self.value = value

    def step(self, h, u, state):
        if self.value is None:
            return h.to(torch.float32), state
        return torch.full_like(h, self.value), state


def test_local_spectra_field_rejects_nonfinite_fallback_step_output():
    with pytest.raises(ValueError):
        local_spectra_field(
            _InvalidFallbackStep(float("nan")),
            torch.zeros(3, 2, dtype=DTYPE),
            torch.zeros(1, 3, 1, dtype=DTYPE),
        )


def test_local_spectra_field_rejects_float32_fallback_step_output():
    with pytest.raises(ValueError):
        local_spectra_field(
            _InvalidFallbackStep(),
            torch.zeros(3, 2, dtype=DTYPE),
            torch.zeros(1, 3, 1, dtype=DTYPE),
        )


class _ShapeChangingStep(PredictorModule):
    def step(self, h, u, state):
        return h[:, :1], state

    def local_operator(self, h, u, state):
        d_h = h.shape[1]
        A = torch.eye(d_h, dtype=h.dtype).expand(h.shape[0], -1, -1)
        B = torch.zeros(h.shape[0], d_h, u.shape[-1], dtype=h.dtype)
        return A, B


def test_local_spectra_field_rejects_shape_changing_step():
    with pytest.raises(ValueError):
        local_spectra_field(
            _ShapeChangingStep(),
            torch.zeros(3, 2, dtype=DTYPE),
            torch.zeros(1, 3, 1, dtype=DTYPE),
        )


def test_local_spectra_field_rejects_nonfinite_inputs():
    model = _build_constant_koopman()
    h0 = torch.zeros(3, model.d_h, dtype=DTYPE)
    h0[0, 0] = torch.nan
    with pytest.raises(ValueError):
        local_spectra_field(model, h0, torch.zeros(1, 3, model.d_u, dtype=DTYPE))
