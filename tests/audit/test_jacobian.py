import math

import numpy as np
import pytest
import torch

from sld.audit.jacobian import (
    augmented_jacobian_excluding_identity,
    augmented_sigma_max_normalized,
    conditional_jacobian,
    fr_drift_curve,
    fr_lyapunov_exponents,
    jacobian_products,
    lipschitz_pi,
    total_jacobian,
)
from sld.objectives.jac_reg import Gauge, MetricIndeterminate, normalize_jacobian


def _t(arr) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float64)


# -- (a) total_jacobian: linear gold -----------------------------------------


def test_total_jacobian_linear_gold():
    rng = np.random.default_rng(0)
    M = _t(rng.standard_normal((3, 3)))

    def step_fn(q, u):
        return M @ q + u

    q = _t(rng.standard_normal(3))
    u = _t(rng.standard_normal(3))
    J = total_jacobian(step_fn, q, u)
    err = np.abs(J - M.numpy()).max()
    print(f"[total_jacobian linear] max_abs_err={err:.3e}")
    assert err < 1e-14


# -- (b) total_jacobian: scalar analytic gold ---------------------------------


def test_total_jacobian_scalar_analytic_gold():
    # F(h) = a(h) h, a(h) = 0.5 + 0.4 tanh(K(h-1)), K=100.
    # F'(h) = a(h) + h a'(h); a'(h) = 0.4 K sech^2(K(h-1)).
    # At h=1: a(1) = 0.5 + 0.4 tanh(0) = 0.5; sech^2(0) = 1 so a'(1) = 0.4*100 = 40.
    # F'(1) = 0.5 + 1*40 = 40.5 exactly.
    K = 100.0

    def step_fn(q, u):
        h = q[0]
        a = 0.5 + 0.4 * torch.tanh(K * (h - 1.0))
        return torch.stack([a * h]) + u

    q = _t([1.0])
    u = _t([0.0])
    J = total_jacobian(step_fn, q, u)
    print(f"[total_jacobian scalar] J={J[0, 0]:.9f} gold=40.5")
    assert abs(J[0, 0] - 40.5) < 1e-9


# -- (c) jacobian_products ----------------------------------------------------


def test_jacobian_products_overflow_resistance():
    # raw product of T=300 copies of 1000*I_2 would be 1000^300 -- float64
    # overflows past ~1e308 (~exponent 102 at base 1000), guaranteed inf.
    T = 300
    J = 1000.0 * np.eye(2)
    Q_final, log_diag_r = jacobian_products([J] * T)
    assert np.isfinite(log_diag_r).all()
    col_sums = log_diag_r.sum(axis=0)
    expected = T * np.log(1000.0)
    err = np.abs(col_sums - expected).max()
    print(f"[jacobian_products overflow] col_sums={col_sums} expected={expected:.6f} err={err:.3e}")
    assert err < 1e-9


def test_jacobian_products_rotation_zero_exponents():
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T = 50
    _, log_diag_r = jacobian_products([R] * T)
    max_abs = np.abs(log_diag_r).max()
    print(f"[jacobian_products rotation] max|log diag R|={max_abs:.3e}")
    assert max_abs < 1e-12


def test_jacobian_products_determinant_invariant():
    rng = np.random.default_rng(1)
    T = 15
    Js = [np.eye(3) + 0.2 * rng.standard_normal((3, 3)) for _ in range(T)]
    dets = [np.linalg.det(J) for J in Js]
    assert all(abs(d) > 1e-6 for d in dets)  # well-conditioned by construction
    expected = sum(np.log(np.abs(d)) for d in dets)

    _, log_diag_r = jacobian_products(Js)
    total = log_diag_r.sum()
    err = abs(total - expected)
    print(f"[jacobian_products det-invariant] total={total:.9f} expected={expected:.9f} err={err:.3e}")
    assert err < 1e-9


def test_jacobian_products_Q_final_orthonormal():
    rng = np.random.default_rng(2)
    Js = [np.eye(3) + 0.3 * rng.standard_normal((3, 3)) for _ in range(10)]
    Q_final, _ = jacobian_products(Js)
    gram = Q_final.T @ Q_final
    err = np.abs(gram - np.eye(3)).max()
    print(f"[jacobian_products orthonormal] max|QtQ - I|={err:.3e}")
    assert err < 1e-12


def test_jacobian_products_sign_canonicalization_oracle():
    # Independent canonical oracle: J = diag(-2, 3) has the UNIQUE
    # diag(R) >= 0 factorization Q = diag(-1, 1), R = diag(2, 3), whatever
    # sign convention the underlying LAPACK QR happens to return. This pins
    # the sign-flip algebra itself (not just determinism).
    J = np.diag([-2.0, 3.0])
    Q, log_r = jacobian_products([J])
    np.testing.assert_allclose(Q, np.diag([-1.0, 1.0]), atol=1e-15)
    np.testing.assert_allclose(log_r[0], [np.log(2.0), np.log(3.0)], atol=1e-15)
    # reconstruction: Q @ R must give back J exactly for this diagonal case
    np.testing.assert_allclose(Q @ np.diag(np.exp(log_r[0])), J, atol=1e-12)


def test_jacobian_products_rejects_complex():
    with pytest.raises(TypeError):
        jacobian_products([np.array([[1.0 + 2.0j]])])


def test_jacobian_products_rejects_nonfinite():
    # the finite-metric law: NaN/+inf Jacobians are upstream bugs and must
    # fail loudly instead of propagating into reported exponents ...
    with pytest.raises(ValueError):
        jacobian_products([np.array([[np.nan]])])
    with pytest.raises(ValueError):
        jacobian_products([np.array([[np.inf]])])


def test_jacobian_products_rejects_exactly_singular():
    with pytest.raises(ValueError):
        jacobian_products([np.diag([1.0, 0.0])])

    # Rank misalignment: the first step collapses a direction that the
    # second non-singular step cannot restore.
    with pytest.raises(ValueError):
        jacobian_products([np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([2.0, 1.0])])


def test_jacobian_products_rejects_numerically_rank_deficient_dense_step():
    with pytest.raises(ValueError):
        jacobian_products([np.array([[1.0, 2.0], [2.0, 4.0]])])

    # QR diagonal thresholds are not rank-revealing: the exact rank-one
    # matrix below has a scaled QR residual above 2 * eps.
    J = np.array([[9.0, -10.0], [-54.0, 60.0]])
    with pytest.raises(ValueError):
        jacobian_products([J])
    with pytest.raises(ValueError):
        jacobian_products([J, np.eye(2)])


def test_jacobian_products_rejects_unrepresentable_tangent_geometry():
    with pytest.raises(ValueError):
        jacobian_products([np.diag([1e200, 1e-200])])


def test_jacobian_products_preserves_small_representable_direction():
    _, logs = jacobian_products([np.diag([1.0, 1e-10])])
    np.testing.assert_allclose(logs[0], [0.0, np.log(1e-10)], atol=1e-15)
    assert np.isfinite(logs).all()


def test_jacobian_products_huge_finite_chain():
    theta = np.pi / 4.0
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T = 50
    _, log_diag_r = jacobian_products([1e155 * rotation] * T)
    expected = T * np.log(1e155)
    np.testing.assert_allclose(log_diag_r.sum(axis=0), [expected, expected], rtol=1e-12)
    assert np.isfinite(log_diag_r).all()


def test_jacobian_products_deterministic():
    rng = np.random.default_rng(3)
    Js = [np.eye(3) + 0.3 * rng.standard_normal((3, 3)) for _ in range(8)]
    Q1, L1 = jacobian_products(Js)
    Q2, L2 = jacobian_products(Js)
    assert np.array_equal(Q1, Q2)
    assert np.array_equal(L1, L2)
    # canonicalization: diag(R_t) >= 0 at every step -- exercised indirectly
    # via determinism above; also assert the sign convention directly by
    # recomputing R_t from Q_{t-1}, Q_t is awkward to expose, so instead
    # check the documented invariant: log_diag_r must be real (no nan from
    # a negative argument sneaking into a stray log of a signed value)
    assert not np.isnan(L1).any()


# -- (d)/(e) fr_lyapunov_exponents --------------------------------------------


def test_fr_lyapunov_exponents_linear_gold():
    rng = np.random.default_rng(4)
    d = 3
    Q_rot, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = np.array([0.9, 0.5, 0.2])
    M = Q_rot @ np.diag(eigs) @ Q_rot.T  # symmetric: singular values == |eigenvalues|
    M_t = _t(M)

    def step_fn(q, u):
        return M_t @ q + u

    T, discard = 1500, 200
    q0 = _t(rng.standard_normal(d))
    u_seq = torch.zeros(T, d, dtype=torch.float64)
    exps = fr_lyapunov_exponents(step_fn, q0, u_seq, discard=discard)
    expected = np.sort(np.log(eigs))[::-1]
    err = np.abs(exps - expected).max()
    print(f"[LE linear gold] exps={exps} expected={expected} err={err:.3e}")
    # for a time-invariant J, log|diag R_t| converges geometrically to the
    # true exponents; discard=200 burns off the transient from Q_0 = I
    assert err < 1e-3


def test_fr_lyapunov_exponents_complex_pair():
    theta = 0.7
    scale = 0.8
    R = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R_t = _t(R)

    def step_fn(q, u):
        return R_t @ q + u

    T, discard = 1500, 200
    rng = np.random.default_rng(5)
    q0 = _t(rng.standard_normal(2))
    u_seq = torch.zeros(T, 2, dtype=torch.float64)
    exps = fr_lyapunov_exponents(step_fn, q0, u_seq, discard=discard)
    expected = np.array([np.log(scale), np.log(scale)])
    err = np.abs(exps - expected).max()
    print(f"[LE complex pair] exps={exps} expected={expected} err={err:.3e}")
    assert err < 1e-3


def test_fr_lyapunov_exponents_nonnormal_gold():
    # NON-normal upper-triangular M: eigenvalues (0.9, 0.5) but singular
    # values (~10.07, ~0.045). This gold DISCRIMINATES the QR cocycle
    # recursion from the wrong implementation that accumulates each step's
    # singular values independently (which reports log 10.07 for the top
    # exponent here -- off by a factor > 20 in log units). The normal-matrix
    # golds above cannot see that difference (their singular values equal
    # their |eigenvalues| from step one).
    M = np.array([[0.9, 10.0], [0.0, 0.5]])
    sv = np.linalg.svd(M, compute_uv=False)
    assert sv[0] > 5.0  # non-normality is real, the discrimination binds
    M_t = _t(M)

    def step_fn(q, u):
        return M_t @ q + u

    T, discard = 1000, 200
    rng = np.random.default_rng(10)
    q0 = _t(rng.standard_normal(2))
    u_seq = torch.zeros(T, 2, dtype=torch.float64)
    exps = fr_lyapunov_exponents(step_fn, q0, u_seq, discard=discard)
    expected = np.array([np.log(0.9), np.log(0.5)])
    err = np.abs(exps - expected).max()
    print(f"[LE nonnormal gold] exps={exps} expected={expected} err={err:.3e}")
    assert err < 1e-6


def test_fr_lyapunov_exponents_noncommuting_chain_gold():
    # Time-varying, noncommuting chain with an EXACT independent oracle:
    # J_t = R((t+1)phi) D R(t phi)^T telescopes to Phi_T = R(T phi) D^T,
    # so the exponents are exactly log(diag D). The time index is threaded
    # through u_seq (also exercising the u-threading path). With Q_0 = I =
    # R(0), induction gives log|diag R_t| = log(diag D) at EVERY step --
    # no transient at all, hence the tight tolerance with discard=0.
    phi = 0.31
    diag = torch.tensor([0.9, 0.5], dtype=torch.float64)
    D = torch.diag(diag)

    def rot(a: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    def step_fn(q, u):
        t = u[0]
        J = rot((t + 1.0) * phi) @ D @ rot(t * phi).T
        return J @ q

    T = 100
    u_seq = torch.arange(T, dtype=torch.float64).unsqueeze(1)
    q0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    exps = fr_lyapunov_exponents(step_fn, q0, u_seq)
    expected = np.log(np.array([0.9, 0.5]))
    err = np.abs(exps - expected).max()
    print(f"[LE noncommuting chain] exps={exps} expected={expected} err={err:.3e}")
    assert err < 1e-12


# -- augmented_jacobian_excluding_identity: M8 (overflow-safe norms) and
# -- m11 (identity_atol validation) -------------------------------------


def test_augmented_jacobian_frobenius_overflow_from_finite_J_raises_not_silent_inf():
    # M8, adversarial review round 2 probe: step_fn(h) = 1e200*h is FINITE
    # (sigma_max=1e200 is a legitimate finite report), but naive
    # sum(J**2) ~ 1e400 overflows float64 (max ~1.8e308) -- must never
    # silently return inf.
    def step_fn(h, m, u):
        return 1e200 * h, None

    h = torch.zeros(2, dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="frobenius_sq") as excinfo:
        augmented_jacobian_excluding_identity(step_fn, h, None, u)
    print(f"[augmented_jacobian M8] 1e200*h probe -> {excinfo.value}")
    assert "1e+200" in str(excinfo.value)  # sigma_max WAS computed and is legitimate


def test_augmented_jacobian_large_but_representable_frobenius_computes_correctly():
    # non-regression companion: a large but REPRESENTABLE case must still
    # compute the correct finite value, not be swept into the overflow guard
    def step_fn(h, m, u):
        return 1e100 * h, None

    h = torch.zeros(3, dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    result = augmented_jacobian_excluding_identity(step_fn, h, None, u)
    print(
        f"[augmented_jacobian M8] 1e100*h -> sigma_max={result.sigma_max:.6g} "
        f"frobenius_sq={result.frobenius_sq:.6g}"
    )
    assert math.isfinite(result.sigma_max) and math.isfinite(result.frobenius_sq)
    assert result.sigma_max == pytest.approx(1e100, rel=1e-9)
    assert result.frobenius_sq == pytest.approx(3 * (1e100) ** 2, rel=1e-9)


def test_augmented_jacobian_frobenius_underflow_from_finite_J_raises_not_silent_zero():
    # m9, adversarial review round 3 probe: step_fn(h) = 1e-170*h is FINITE
    # (sigma_max=1e-170 is a legitimate finite report), but frob ~=
    # 1e-170*sqrt(d) is far below sqrt(float64 min normal) ~= 1.49e-154, so
    # naive frob*frob underflows SILENTLY to 0.0 -- must never report a zero
    # frobenius_sq for a genuinely nonzero (if tiny) Jacobian.
    def step_fn(h, m, u):
        return 1e-170 * h, None

    h = torch.zeros(2, dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="frobenius_sq") as excinfo:
        augmented_jacobian_excluding_identity(step_fn, h, None, u)
    print(f"[augmented_jacobian m9] 1e-170*h probe -> {excinfo.value}")
    assert "1e-170" in str(excinfo.value)  # sigma_max WAS computed and is legitimate


def test_augmented_jacobian_identity_atol_validated_finite_and_ranged():
    # m11: identity_atol was exposed with NO validation -- a value like the
    # reviewer's poison-pill 0.1 would misclassify a genuine 0.95*m
    # derivative as a verbatim identity copy and silently delete it
    def step_fn(h, m, u):
        return h, torch.cat([m, h.unsqueeze(0)], dim=0)

    h = torch.randn(3, dtype=torch.float64)
    m = torch.randn(1, 3, dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    for bad_atol in (float("nan"), float("inf"), 0.0, -1e-9, 0.1, 1.0):
        with pytest.raises(ValueError, match="identity_atol"):
            augmented_jacobian_excluding_identity(step_fn, h, m, u, identity_atol=bad_atol)


def test_augmented_jacobian_atol_ceiling_tightened_learned_map_not_deleted():
    # m13, adversarial review round 3: the OLD ceiling (1e-3) let
    # m_next = cat([0.9995 * m, new]) -- a plausible LEARNED coefficient, not
    # a verbatim copy -- be misclassified as an identity passthrough and
    # silently deleted (|0.9995 - 1.0| = 5e-4 <= 1e-3). The ceiling is now
    # 1e-4: 1e-3 itself is out of range, and at the new max permitted value
    # this map is correctly rejected as NOT a pure-append identity (loudly,
    # never silently excluded).
    def step_fn(h, m, u):
        return h, torch.cat([0.9995 * m, h.unsqueeze(0)], dim=0)

    h = torch.randn(3, dtype=torch.float64)
    m = torch.randn(2, 3, dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)

    with pytest.raises(ValueError, match="identity_atol"):
        augmented_jacobian_excluding_identity(step_fn, h, m, u, identity_atol=1e-3)

    with pytest.raises(ValueError, match="pure-append hypothesis violated"):
        augmented_jacobian_excluding_identity(step_fn, h, m, u, identity_atol=1e-4)


# -- (f) conditional_jacobian ---------------------------------------------


def test_conditional_jacobian_block_and_doctrine_docstring():
    rng = np.random.default_rng(6)
    J_q = rng.standard_normal((5, 5))
    block = conditional_jacobian(J_q, d_h=2)
    assert np.array_equal(block, J_q[:2, :2])
    assert "never concludes alone" in conditional_jacobian.__doc__


def test_conditional_jacobian_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        conditional_jacobian(np.array([[np.nan]]), d_h=1)


# -- (g) lipschitz_pi ----------------------------------------------------


def test_lipschitz_pi_linear_gold():
    rng = np.random.default_rng(7)
    W = rng.standard_normal((3, 2))  # (d_out, d_h)
    W_t = _t(W)

    def pi_fn(h):
        return h @ W_t.T

    h_samples = _t(rng.standard_normal((10, 2)))
    expected = np.linalg.norm(W, 2)

    val_max = lipschitz_pi(pi_fn, h_samples, quantile=None)
    val_q95 = lipschitz_pi(pi_fn, h_samples, quantile=0.95)
    print(f"[lipschitz_pi linear] max={val_max:.9f} q95={val_q95:.9f} expected={expected:.9f}")
    assert abs(val_max - expected) < 1e-6
    assert abs(val_q95 - expected) < 1e-6


def test_lipschitz_pi_tanh_bounded():
    rng = np.random.default_rng(8)

    def pi_fn(h):
        return 0.5 * torch.tanh(h)

    h_samples = _t(rng.standard_normal((20, 4)))
    val = lipschitz_pi(pi_fn, h_samples)
    print(f"[lipschitz_pi tanh] val={val:.6f}")
    assert val <= 0.5 + 1e-9
    assert val > 0.0


def test_lipschitz_pi_tanh_exact_value():
    """Same pi_fn / h_samples as test_lipschitz_pi_tanh_bounded above -- but
    asserting the EXACT value instead of the unfalsifiable bound ``<= 0.5``
    (REGLES 7.1: the constant 0.25 satisfies that bound too).

    pi_fn is elementwise (``h -> 0.5*tanh(h)``), so its Jacobian at any row
    is EXACTLY diagonal, entry j = ``0.5*sech^2(h_j)`` -- verified below by
    reading jacrev's own output rather than assuming it: the off-diagonal
    entries come out exactly 0.0 (no cross-dependency exists for autodiff to
    accumulate rounding into). The spectral norm (largest singular value) of
    a diagonal matrix is the max absolute diagonal entry, so the per-sample
    spectral norm is ``0.5 * max_j sech^2(h_i,j)``. ``lipschitz_pi(...,
    quantile=None)`` takes the MAX over samples
    (src/sld/audit/jacobian.py:800-801: ``return float(norms.max())`` --
    confirmed by reading the source, not assumed), so the oracle is
    ``0.5 * global max of sech^2`` over the whole (20, 4) array, computed
    here in plain numpy without ever calling lipschitz_pi.
    """
    rng = np.random.default_rng(8)

    def pi_fn(h):
        return 0.5 * torch.tanh(h)

    h_samples = _t(rng.standard_normal((20, 4)))

    # confirm the "exactly diagonal" premise instead of assuming it
    jac_fn = torch.func.jacrev(pi_fn)
    J0 = jac_fn(h_samples[0])
    off_diag = J0 - torch.diag(J0.diagonal())
    assert off_diag.abs().max().item() == 0.0

    sech2 = 1.0 / np.cosh(h_samples.numpy()) ** 2
    expected = 0.5 * sech2.max()

    val = lipschitz_pi(pi_fn, h_samples)
    err = abs(val - expected)
    print(f"[lipschitz_pi tanh exact] val={val:.12f} expected={expected:.12f} err={err:.3e}")
    # float64 chain (tanh/cosh -> jacrev -> SVD of an exactly-diagonal 4x4);
    # measured err running this test is 5.6e-17 -- tolerance is the
    # eigen/SVD float64 tier, not the short-algebra 1e-12 tier
    assert err < 1e-10

    wrong = 0.25
    gap = abs(wrong - expected)
    print(f"[lipschitz_pi tanh negative] 0.25 vs exact oracle={expected:.12f} gap={gap:.6f}")
    # 0.25 is exactly the constant REGLES 7.1 names as passing the OLD
    # bound-only test (test_lipschitz_pi_tanh_bounded above); it must fail
    # this exact-value oracle -- measured gap is 0.25
    assert gap > 1e-6


def test_lipschitz_pi_rejects_float32_output():
    def pi_fn(h):
        return (2.0 * h.to(torch.float32)).sum().unsqueeze(0)

    with pytest.raises(ValueError):
        lipschitz_pi(pi_fn, _t([[1.0, 2.0]]))


def test_lipschitz_pi_rejects_nonfinite_samples():
    with pytest.raises(ValueError):
        lipschitz_pi(lambda h: h, _t([[np.nan]]))


def test_lipschitz_pi_rejects_nonfinite_output():
    with pytest.raises(ValueError):
        lipschitz_pi(lambda h: torch.full_like(h, torch.nan), _t([[1.0]]))


def test_lipschitz_pi_rejects_infinite_spectral_norm():
    W = _t(np.full((2, 2), 1e308))
    with pytest.raises(ValueError):
        lipschitz_pi(lambda h: h @ W.T, _t([[0.0, 0.0]]))


# -- (h) fr_drift_curve ----------------------------------------------------


def test_fr_drift_curve_zero_when_equal():
    rng = np.random.default_rng(9)
    target = rng.standard_normal((5, 4, 3))
    pred = target.copy()
    curve = fr_drift_curve(pred, target)
    assert np.abs(curve).max() == 0.0


def test_fr_drift_curve_constant_target_zero_pred():
    c = np.array([1.0, -2.0, 0.5])
    T, B = 6, 4
    target = np.broadcast_to(c, (T, B, 3)).copy()
    pred = np.zeros((T, B, 3))
    curve = fr_drift_curve(pred, target)
    print(f"[fr_drift_curve constant] curve={curve}")
    assert np.allclose(curve, 1.0)


def test_fr_drift_curve_zero_target_raises():
    target = np.zeros((3, 2, 4))
    pred = np.ones((3, 2, 4))
    with pytest.raises(ValueError):
        fr_drift_curve(pred, target)


def test_fr_drift_curve_nonfinite_raises():
    # NaN/inf is a bug to chase, never an audit result (REGLES section 5)
    target = np.ones((3, 2, 4))
    pred = np.ones((3, 2, 4))
    pred[1, 0, 2] = np.nan
    with pytest.raises(ValueError):
        fr_drift_curve(pred, target)
    pred[1, 0, 2] = np.inf
    with pytest.raises(ValueError):
        fr_drift_curve(pred, target)


def test_fr_drift_curve_empty_raises():
    with pytest.raises(ValueError):
        fr_drift_curve(np.zeros((0, 2, 4)), np.zeros((0, 2, 4)))


def test_fr_drift_curve_numerically_scale_invariant():
    target = np.ones((4, 2, 3))
    pred = -np.ones((4, 2, 3))
    curve = fr_drift_curve(pred, target)
    np.testing.assert_allclose(curve, curve[0])
    for factor in (1e-300, 1e308):
        scaled_curve = fr_drift_curve(pred * factor, target * factor)
        np.testing.assert_allclose(scaled_curve, curve)


def test_fr_drift_curve_tiny_relative_error_at_huge_scale():
    target = np.full((3, 2, 2), 1e155)
    pred = target * (1.0 + 1e-15)
    curve = fr_drift_curve(pred, target)
    assert np.isfinite(curve).all()
    assert np.all(curve > 0.0)
    assert np.all(curve < 1e-12)


# -- (i) validations ----------------------------------------------------


def test_total_jacobian_rejects_float32():
    def step_fn(q, u):
        return q + u

    q = torch.zeros(3, dtype=torch.float32)
    u = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        total_jacobian(step_fn, q, u)


def test_total_jacobian_rejects_float32_returning_step_fn():
    # the OUTPUT side of the float32 guard: a checkpoint wrapper that casts
    # internally and returns float32 would otherwise be silently widened,
    # concealing float32 COMPUTE behind a float64 audit number.
    def step_fn(q, u):
        return (q.to(torch.float32) + u.to(torch.float32)).to(torch.float32)

    with pytest.raises(ValueError):
        total_jacobian(step_fn, _t([1.0]), _t([0.0]))


def test_total_jacobian_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        total_jacobian(lambda q, u: q + u, _t([np.nan]), _t([0.0]))


def test_total_jacobian_rejects_nonfinite_forward_output():
    with pytest.raises(ValueError):
        total_jacobian(lambda q, u: torch.full_like(q, torch.inf), _t([1.0]), _t([0.0]))


def test_total_jacobian_rejects_nonfinite_derivative():
    def step_fn(q, u):
        return torch.sqrt(q)

    with pytest.raises(ValueError):
        total_jacobian(step_fn, _t([0.0]), _t([0.0]))


def test_fr_lyapunov_exponents_discard_too_large_raises():
    M = _t(np.eye(2))

    def step_fn(q, u):
        return M @ q + u

    q0 = _t([1.0, 1.0])
    u_seq = torch.zeros(5, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        fr_lyapunov_exponents(step_fn, q0, u_seq, discard=5)


def test_fr_lyapunov_exponents_rejects_nonfinite_initial_state():
    def step_fn(q, u):
        return q + u

    with pytest.raises(ValueError):
        fr_lyapunov_exponents(step_fn, _t([np.nan]), torch.zeros(3, 1, dtype=torch.float64))


def test_fr_lyapunov_exponents_rejects_diverged_rollout():
    def step_fn(q, u):
        return q * 1e200

    # the divergence is now caught by total_jacobian's forward-output guard
    # (one step earlier than the rollout-level check, which remains as
    # defense in depth) -- both messages name the non-finite state
    with pytest.raises(ValueError, match="non-finite"):
        fr_lyapunov_exponents(step_fn, _t([1.0]), torch.zeros(3, 1, dtype=torch.float64))


def test_jacobian_products_empty_raises():
    with pytest.raises(ValueError):
        jacobian_products([])


# ---------------------------------------------------------------------------
# PROTOCOL.md section 10.3: augmented_sigma_max_normalized
# ---------------------------------------------------------------------------


def test_augmented_sigma_max_normalized_h_only_matches_hand_computation():
    rng = np.random.default_rng(0)
    d_h, d_u = 4, 2
    A = _t(rng.standard_normal((d_h, d_h)))
    B = _t(rng.standard_normal((d_h, d_u)))

    def step_fn(h, m, u):
        return h @ A.T + u @ B.T, None

    h = _t(rng.standard_normal((1, d_h)))
    u = _t(rng.standard_normal((1, d_u)))
    s_h = _t([1e-2, 1.0, 5.0, 1e3]).reshape(1, d_h)
    gauge = Gauge(s_h=s_h)

    audited = augmented_sigma_max_normalized(step_fn, h, None, u, gauge)

    dense = augmented_jacobian_excluding_identity(step_fn, h, None, u)
    assert dense.n_identity_rows == 0
    s_flat = s_h.reshape(-1).numpy()
    M = normalize_jacobian(dense.jacobian, s_flat, s_flat)
    expected = float(np.linalg.svd(M, compute_uv=False)[0])
    assert abs(audited - expected) <= 1e-10 * max(1.0, expected)
    # and the gauge must actually move the number relative to the raw one
    assert abs(audited - dense.sigma_max) / dense.sigma_max > 0.1


def test_augmented_sigma_max_normalized_fixed_size_m_matches_hand_computation():
    rng = np.random.default_rng(1)
    d_h, d_m, d_u = 3, 2, 1
    A = _t(rng.standard_normal((d_h, d_h)))
    C = _t(rng.standard_normal((d_h, d_m)))
    Wm = _t(rng.standard_normal((d_m, d_m)))
    Wh = _t(rng.standard_normal((d_m, d_h)))

    def step_fn(h, m, u):
        return h @ A.T + m @ C.T, m @ Wm.T + h @ Wh.T

    h = _t(rng.standard_normal((1, d_h)))
    m = _t(rng.standard_normal((1, d_m)))
    u = _t(rng.standard_normal((1, d_u)))
    s_h = _t([1e-2, 1.0, 1e2]).reshape(1, d_h)
    s_m = _t([1e3, 1e-3]).reshape(1, d_m)
    gauge = Gauge(s_h=s_h, s_m=s_m)

    audited = augmented_sigma_max_normalized(step_fn, h, m, u, gauge)
    dense = augmented_jacobian_excluding_identity(step_fn, h, m, u)
    assert dense.n_identity_rows == 0
    s_flat = torch.cat([s_h.reshape(-1), s_m.reshape(-1)]).numpy()
    M = normalize_jacobian(dense.jacobian, s_flat, s_flat)
    expected = float(np.linalg.svd(M, compute_uv=False)[0])
    assert abs(audited - expected) <= 1e-9 * max(1.0, expected)


def test_augmented_sigma_max_normalized_growing_memory_row_alignment():
    # the excluded_row_indices field must align S_out to J_excl's actual
    # (post-deletion) rows -- verified against an independently-built dense
    # normalized matrix constructed WITHOUT that field (manual bookkeeping).
    rng = np.random.default_rng(2)
    d_h, d_m, d_u, L = 3, 2, 2, 2

    A = _t(rng.standard_normal((d_h, d_h)))
    C = _t(rng.standard_normal((d_h, d_m)))
    W = _t(rng.standard_normal((d_m, d_h)))

    def step_fn(h, m, u):
        h_next = h @ A.T + m[-1] @ C.T
        m_next = torch.cat([m, (h @ W.T).unsqueeze(0)], dim=0)
        return h_next, m_next

    h = _t(rng.standard_normal((1, d_h)))
    m = _t(rng.standard_normal((L, 1, d_m)))
    u = _t(rng.standard_normal((1, d_u)))

    s_h = _t([1e-1, 1.0, 1e1]).reshape(1, d_h)
    s_positions = _t(rng.uniform(1e-3, 1e3, size=(4, 1, d_m)))  # covers up to 4 positions
    gauge = Gauge(s_h=s_h, s_m_positions=s_positions, position_axis=0)

    audited = augmented_sigma_max_normalized(step_fn, h, m, u, gauge)

    dense = augmented_jacobian_excluding_identity(step_fn, h, m, u)
    assert dense.n_identity_rows == m.numel()
    s_in = torch.cat([s_h.reshape(-1), s_positions[:L].reshape(-1)]).numpy()
    s_out_full = torch.cat([s_h.reshape(-1), s_positions[: L + 1].reshape(-1)]).numpy()
    s_out = np.delete(s_out_full, dense.excluded_row_indices)
    assert s_out.shape[0] == dense.jacobian.shape[0]
    M = normalize_jacobian(dense.jacobian, s_in, s_out)
    expected = float(np.linalg.svd(M, compute_uv=False)[0])
    assert abs(audited - expected) <= 1e-8 * max(1.0, expected)


def test_augmented_sigma_max_normalized_propagates_metric_indeterminate():
    rng = np.random.default_rng(3)
    d_h, d_u = 2, 1
    A = _t(rng.standard_normal((d_h, d_h)))

    def step_fn(h, m, u):
        return h @ A.T, None

    h = _t(rng.standard_normal((1, d_h)))
    u = _t(rng.standard_normal((1, d_u)))

    zero_scale = Gauge(s_h=_t([1.0, 0.0]).reshape(1, d_h))
    with pytest.raises(MetricIndeterminate):
        augmented_sigma_max_normalized(step_fn, h, None, u, zero_scale)

    nonfinite_scale = Gauge(s_h=_t([1.0, float("nan")]).reshape(1, d_h))
    with pytest.raises(MetricIndeterminate):
        augmented_sigma_max_normalized(step_fn, h, None, u, nonfinite_scale)


def test_augmented_sigma_max_normalized_growing_memory_position_coverage_indeterminate():
    rng = np.random.default_rng(4)
    d_h, d_m, d_u, L = 2, 2, 1, 3  # 3 old positions already in the cache

    A = _t(rng.standard_normal((d_h, d_h)))
    C = _t(rng.standard_normal((d_h, d_m)))
    W = _t(rng.standard_normal((d_m, d_h)))

    def step_fn(h, m, u):
        h_next = h @ A.T + m[-1] @ C.T
        m_next = torch.cat([m, (h @ W.T).unsqueeze(0)], dim=0)
        return h_next, m_next

    h = _t(rng.standard_normal((1, d_h)))
    m = _t(rng.standard_normal((L, 1, d_m)))
    u = _t(rng.standard_normal((1, d_u)))

    # reference rollout only covers 3 positions -- this step needs 4 (m_next)
    s_h = _t([1.0, 1.0]).reshape(1, d_h)
    s_positions = _t(rng.uniform(1e-2, 1e2, size=(3, 1, d_m)))
    gauge = Gauge(s_h=s_h, s_m_positions=s_positions, position_axis=0)

    with pytest.raises(MetricIndeterminate, match="beyond"):
        augmented_sigma_max_normalized(step_fn, h, m, u, gauge)
