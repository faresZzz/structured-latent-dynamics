"""Tests for sld.audit.invariance: the reusable reparameterization stress
test (REGLES.md section 4), plus the actual PROTOCOL.md section 10.3 gauge
invariance check on augmented_sigma_max_normalized -- the "does the declared
metric actually pass its own stress test" proof this repo requires before a
diagnostic reaches a figure or a conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sld.audit.invariance import (
    assert_diagonal_gauge_invariant,
    check_diagonal_gauge_invariance,
    check_shear_breaks_invariance,
    diagonal_gauges,
)
from sld.audit.jacobian import augmented_jacobian_excluding_identity, augmented_sigma_max_normalized
from sld.objectives.jac_reg import Gauge


def _t(arr) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float64)


# ---------------------------------------------------------------------------
# (a) the generic tool, in isolation
# ---------------------------------------------------------------------------


def test_diagonal_gauges_battery_is_finite_nonzero_and_includes_identity():
    battery = diagonal_gauges(6, seed=0)
    assert "identity" in battery
    assert np.allclose(battery["identity"], 1.0)
    for label, diag in battery.items():
        assert diag.shape == (6,)
        assert np.isfinite(diag).all(), label
        assert (diag != 0.0).all(), label
        assert np.isfinite(1.0 / diag).all(), label  # finite-invertible


def test_check_diagonal_gauge_invariance_passes_for_a_truly_invariant_measure():
    results = check_diagonal_gauge_invariance(lambda diag: 42.0, d=4)
    assert all(r.invariant for r in results)
    assert {r.label for r in results} == set(diagonal_gauges(4).keys())


def test_assert_diagonal_gauge_invariant_raises_naming_the_failing_gauges():
    # measure varies with the diagonal -> must fail, and the message must
    # name at least one failing gauge (never a silent pass)
    with pytest.raises(AssertionError, match="extreme_large"):
        assert_diagonal_gauge_invariant(lambda diag: float(diag.sum()), d=3)


def test_check_diagonal_gauge_invariance_rejects_non_finite_reference():
    with pytest.raises(ValueError, match="non-finite"):
        check_diagonal_gauge_invariance(lambda diag: float("nan"), d=3)


def test_check_shear_breaks_invariance_reports_movement():
    result = check_shear_breaks_invariance(lambda: 1.0, lambda: 1.5)
    assert result.invariant is False
    assert result.label == "shear"

    unchanged = check_shear_breaks_invariance(lambda: 1.0, lambda: 1.0 + 1e-12)
    assert unchanged.invariant is True


# ---------------------------------------------------------------------------
# (b) the real thing: augmented_sigma_max_normalized under q' = D q
# ---------------------------------------------------------------------------


def _reparam_linear_h_only(A, s_h_base, diag_h):
    """h-only (mlp-like) family reparametrized by diag(diag_h): q' = D q.

    step_fn'(h') = D . step_fn(D^-1 h') and gauge' = |D| . S -- PROTOCOL
    section 10.3's declared transform law. Returns (step_fn', gauge',
    h'=D h_ref) for a fixed reference h_ref/u.
    """
    D = torch.as_tensor(diag_h, dtype=torch.float64)

    def step_fn_prime(h, m, u):
        h_raw = h / D
        return (h_raw @ A.T) * D, None

    gauge_prime = Gauge(s_h=(s_h_base * D.abs()).reshape(1, -1))
    return step_fn_prime, gauge_prime


def test_augmented_sigma_max_normalized_invariant_under_diagonal_gauge_h_only():
    rng = np.random.default_rng(0)
    d_h, d_u = 4, 1
    A = _t(rng.standard_normal((d_h, d_h)))
    s_h_base = _t([1e-2, 1.0, 5.0, 1e2])
    h_ref = _t(rng.standard_normal((1, d_h)))
    u = _t(rng.standard_normal((1, d_u)))

    def step_fn_id(h, m, u):
        return h @ A.T, None

    def measure(diag_h: np.ndarray) -> float:
        D = torch.as_tensor(diag_h, dtype=torch.float64)
        step_fn_prime, gauge_prime = _reparam_linear_h_only(A, s_h_base, diag_h)
        h_prime = h_ref * D
        return augmented_sigma_max_normalized(step_fn_prime, h_prime, None, u, gauge_prime)

    results = assert_diagonal_gauge_invariant(measure, d_h, rtol=1e-6)
    for r in results:
        print(f"[gauge invariance, h-only] {r.label}: value={r.value:.10g} rel_diff={r.relative_diff:.3g}")

    # constructive counter-test: the RAW (un-normalized) sigma_max DOES move
    raw_reference = augmented_jacobian_excluding_identity(step_fn_id, h_ref, None, u).sigma_max
    battery = diagonal_gauges(d_h, seed=0)
    moved = False
    for label, diag in battery.items():
        if label == "identity":
            continue
        D = torch.as_tensor(diag, dtype=torch.float64)

        def step_fn_prime(h, m, u, A=A, D=D):
            return (h / D) @ A.T * D, None

        h_prime = h_ref * D
        raw = augmented_jacobian_excluding_identity(step_fn_prime, h_prime, None, u).sigma_max
        rel = abs(raw - raw_reference) / raw_reference
        print(f"[raw, unnormalized] {label}: sigma_max={raw:.6g} rel_diff_from_ref={rel:.3g}")
        if rel > 1e-3:
            moved = True
    assert moved, "the raw sigma_max must move under at least one tested diagonal -- otherwise this test proves nothing"


def test_augmented_sigma_max_normalized_invariant_under_diagonal_gauge_fixed_size_m():
    # min_ssm-like: h and m both fixed-size, gauged independently
    rng = np.random.default_rng(1)
    d_h, d_m, d_u = 3, 2, 1
    A = _t(rng.standard_normal((d_h, d_h)))
    C = _t(rng.standard_normal((d_h, d_m)))
    Wm = _t(rng.standard_normal((d_m, d_m)))
    Wh = _t(rng.standard_normal((d_m, d_h)))
    s_h_base = _t([1e-1, 1.0, 1e1])
    s_m_base = _t([1e2, 1e-2])
    h_ref = _t(rng.standard_normal((1, d_h)))
    m_ref = _t(rng.standard_normal((1, d_m)))
    u = _t(rng.standard_normal((1, d_u)))

    def measure(diag_hm: np.ndarray) -> float:
        d_h_diag = torch.as_tensor(diag_hm[:d_h], dtype=torch.float64)
        d_m_diag = torch.as_tensor(diag_hm[d_h:], dtype=torch.float64)

        def step_fn_prime(h, m, u):
            h_raw, m_raw = h / d_h_diag, m / d_m_diag
            h_next = (h_raw @ A.T + m_raw @ C.T) * d_h_diag
            m_next = (m_raw @ Wm.T + h_raw @ Wh.T) * d_m_diag
            return h_next, m_next

        gauge_prime = Gauge(
            s_h=(s_h_base * d_h_diag.abs()).reshape(1, d_h),
            s_m=(s_m_base * d_m_diag.abs()).reshape(1, d_m),
        )
        h_prime, m_prime = h_ref * d_h_diag, m_ref * d_m_diag
        return augmented_sigma_max_normalized(step_fn_prime, h_prime, m_prime, u, gauge_prime)

    results = assert_diagonal_gauge_invariant(measure, d_h + d_m, rtol=1e-5)
    for r in results:
        print(f"[gauge invariance, fixed-size m] {r.label}: value={r.value:.10g} rel_diff={r.relative_diff:.3g}")


def test_augmented_sigma_max_normalized_invariant_under_diagonal_gauge_growing_memory():
    # transformer-like: growing m, one D_m reused identically at every cache
    # position (a valid, coordinate-identity-preserving special case of the
    # declared domain)
    rng = np.random.default_rng(2)
    d_h, d_m, d_u, L = 3, 2, 1, 2
    A = _t(rng.standard_normal((d_h, d_h)))
    C = _t(rng.standard_normal((d_h, d_m)))
    W = _t(rng.standard_normal((d_m, d_h)))
    s_h_base = _t([1e-2, 1.0, 1e2])
    s_m_position_base = _t(rng.uniform(1e-1, 1e1, size=(4, 1, d_m)))  # 4 reference positions
    h_ref = _t(rng.standard_normal((1, d_h)))
    m_ref = _t(rng.standard_normal((L, 1, d_m)))
    u = _t(rng.standard_normal((1, d_u)))

    def measure(diag_hm: np.ndarray) -> float:
        d_h_diag = torch.as_tensor(diag_hm[:d_h], dtype=torch.float64)
        d_m_diag = torch.as_tensor(diag_hm[d_h:], dtype=torch.float64)  # SAME D_m at every position

        def step_fn_prime(h, m, u):
            h_raw, m_raw = h / d_h_diag, m / d_m_diag
            h_next = (h_raw @ A.T + m_raw[-1] @ C.T) * d_h_diag
            new_raw = (h_raw @ W.T).unsqueeze(0)
            m_next = torch.cat([m_raw, new_raw], dim=0) * d_m_diag
            return h_next, m_next

        gauge_prime = Gauge(
            s_h=(s_h_base * d_h_diag.abs()).reshape(1, d_h),
            s_m_positions=s_m_position_base * d_m_diag.abs().reshape(1, 1, d_m),
            position_axis=0,
        )
        h_prime, m_prime = h_ref * d_h_diag, m_ref * d_m_diag
        return augmented_sigma_max_normalized(step_fn_prime, h_prime, m_prime, u, gauge_prime)

    results = assert_diagonal_gauge_invariant(measure, d_h + d_m, rtol=1e-5)
    for r in results:
        print(f"[gauge invariance, growing m] {r.label}: value={r.value:.10g} rel_diff={r.relative_diff:.3g}")


def test_augmented_sigma_max_normalized_declared_limit_shear_breaks_invariance():
    # PROTOCOL.md section 10.3's declared limit, made explicit: the
    # normalized quantity is NOT invariant under a shear (mixing h's
    # coordinates), even though it IS invariant under any diagonal.
    rng = np.random.default_rng(3)
    d_h, d_u = 3, 1
    A = _t(rng.standard_normal((d_h, d_h)))
    s_h = _t([1.0, 1.0, 1.0]).reshape(1, d_h)  # trivial gauge: isolates the shear's own effect
    h_ref = _t(rng.standard_normal((1, d_h)))
    u = _t(rng.standard_normal((1, d_u)))
    gauge = Gauge(s_h=s_h)

    def step_fn_id(h, m, u):
        return h @ A.T, None

    def measure_reference() -> float:
        return augmented_sigma_max_normalized(step_fn_id, h_ref, None, u, gauge)

    shear = torch.eye(d_h, dtype=torch.float64)
    shear[0, 2] = 3.7  # a genuine shear: non-diagonal, invertible

    def step_fn_shear(h, m, u):
        h_raw = h @ torch.linalg.inv(shear).T
        return (h_raw @ A.T) @ shear.T, None

    def measure_shear() -> float:
        h_prime = h_ref @ shear.T
        return augmented_sigma_max_normalized(step_fn_shear, h_prime, None, u, gauge)

    result = check_shear_breaks_invariance(measure_reference, measure_shear)
    print(f"[declared limit: shear] value={result.value:.6g} reference={result.reference:.6g} "
          f"relative_diff={result.relative_diff:.3g}")
    assert result.invariant is False, (
        "the shear was expected to break invariance (PROTOCOL.md section 10.3's "
        "declared limit) -- if this now passes, either the shear is degenerate "
        "or the declared limit no longer holds and must be revisited, not silently accepted"
    )
