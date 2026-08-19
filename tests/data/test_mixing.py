import numpy as np

from sld.data.mixing import PartialObservation, SmoothMixing


def test_inverse_round_trip():
    mixing = SmoothMixing(dim=4, seed=0)
    rng = np.random.default_rng(123)
    z = rng.standard_normal((1000, 4))

    z_recovered = mixing.inverse(mixing(z))
    max_abs_error = np.max(np.abs(z_recovered - z))
    print(f"[inverse round trip] max_abs_error={max_abs_error:.3e}")
    assert max_abs_error < 1e-10


def _z_reaching(mixing, target):
    """Build z whose every argument coordinate equals ``target`` exactly.

    Forward pass is ``w = tanh(a * (z @ R_1.T) + b)``, so the argument is
    ``arg = a * (R_1 z) + b``. R_1 is orthogonal, so setting
    ``z = ((target - b) / a) @ R_1`` gives ``z @ R_1.T = (target - b)/a`` and
    therefore ``arg = target`` on every coordinate.
    """
    return ((np.full(mixing.a.shape, float(target)) - mixing.b) / mixing.a) @ mixing.R_1


# Declared interior domain of the exact-inversion identity (Wave 1,
# corrected 2026-08-03): |a_i (R_1 z)_i + b_i| <= 5 for all i.
INTERIOR_ARG = 5.0
# atanh(1 - 1e-7) = 8.4056...: past this the clamp in SmoothMixing.inverse
# engages and exact inversion is lost BY DESIGN.
BEYOND_CLAMP_ARG = 9.0


def test_inverse_round_trip_at_interior_boundary():
    """Positive (REGLES 7.1): test_inverse_round_trip above draws standard
    Gaussians, whose arguments stay small -- it never approaches the clamp,
    so it proves interior invertibility and nothing about the boundary.
    This probes the DECLARED interior boundary exactly, on the real object
    at the dimensions actually used.

    Why 5 and not 7 (supervisor correction, 2026-08-03): the round-trip
    error is the machine epsilon of the forward pass amplified by
    d/dw arctanh(w) = cosh^2(arg). At dim 1 both rotations are the 1x1
    identity, so only tanh/arctanh rounding contributes and |arg| = 7 gives
    ~9e-12. From dim 2 up, ``x @ R_2`` in the inverse carries a few ULP of
    matmul rounding, and cosh^2(7) = 3.0e5 turns that into a violation of
    the 1e-10 tolerance. Measured worst case over 40 seeds:

        |arg|      dim 1      dim 2      dim 4      dim 8
          5      2.36e-13   4.01e-12   1.20e-11   8.75e-12
          6      1.12e-12   4.92e-11   7.75e-11   8.27e-11
          7      8.96e-12   3.06e-10   6.65e-10   4.71e-10

    A first pass declared 7 from a dim-1-equivalent scalar measurement --
    a component measured, a composition declared. 5 holds across every
    dimension with a factor ~8 of margin; 6 would pass by less than 1.2x,
    which is not a margin.
    """
    worst = 0.0
    for dim in (1, 2, 4, 8):
        for seed in range(8):
            mixing = SmoothMixing(dim=dim, seed=seed)
            z = _z_reaching(mixing, INTERIOR_ARG)[None, :]
            err = float(np.max(np.abs(mixing.inverse(mixing(z)) - z)))
            worst = max(worst, err)
            assert err < 1e-10, f"dim={dim} seed={seed} err={err:.3e}"
    print(f"[mixing interior boundary |arg|={INTERIOR_ARG}] worst={worst:.3e} tol=1e-10")
    # float64 tanh/arctanh chain amplified by cosh^2(5) = 5.5e3 (R3)
    assert worst < 1e-10


def test_declared_interior_boundary_is_calibrated_not_arbitrary():
    """Negative for the DOMAIN ITSELF (REGLES 7.1). The test above would
    pass for any bound at or below 5; on its own it does not show that 5 was
    chosen rather than guessed. Here the same round trip is run one and two
    units past the declared boundary and must FAIL the same tolerance at
    dim >= 2 -- so the declared domain is falsifiable and pinned, not a
    round number with slack hidden in it.

    dim 1 is deliberately excluded: with SO(1) = {1} there is no rotation
    rounding to amplify, and |arg| = 7 still passes there. That exclusion is
    the point -- it is exactly the case whose measurement produced the wrong
    declaration in the first place.
    """
    for arg, must_fail_somewhere in ((7.0, True),):
        failures = []
        for dim in (2, 4, 8):
            for seed in range(8):
                mixing = SmoothMixing(dim=dim, seed=seed)
                z = _z_reaching(mixing, arg)[None, :]
                err = float(np.max(np.abs(mixing.inverse(mixing(z)) - z)))
                if err > 1e-10:
                    failures.append((dim, seed, err))
        print(f"[mixing past declared domain |arg|={arg}] {len(failures)}/24 exceed 1e-10")
        assert bool(failures) == must_fail_somewhere
        assert max(e for _, _, e in failures) > 1e-10


def test_inverse_round_trip_fails_beyond_clamp():
    """Negative (REGLES 7.1), two parts, on the real object.

    Part 1 -- |arg| = 9 is past the clamp threshold atanh(1 - 1e-7) =
    8.4056, so the round trip must FAIL the tolerance that held at the
    interior boundary. This is what falsifies the old, wrongly-global
    "< 1e-10 everywhere" declaration. It must stay FINITE: the precision
    loss is intentional and declared, an inf/NaN would be a real bug.

    Part 2 -- prove the failure is the clamp engaging and not some unrelated
    effect. The quantity clipped inside SmoothMixing.inverse is
    ``x @ R_2 = w @ R_2.T @ R_2 = w`` for orthogonal R_2, i.e. exactly
    tanh(arg), so the clamp engages iff |tanh(arg)| > 1 - 1e-7 iff
    |arg| > 8.4056. Asserting it engages at 9 and does NOT engage at the
    interior boundary proves the test touched the mechanism it names.
    """
    for dim in (1, 2, 4, 8):
        mixing = SmoothMixing(dim=dim, seed=dim)
        z9 = _z_reaching(mixing, BEYOND_CLAMP_ARG)[None, :]
        recovered = mixing.inverse(mixing(z9))
        assert np.all(np.isfinite(recovered))  # precision loss ok, non-finite is a bug
        err = float(np.max(np.abs(recovered - z9)))
        print(f"[mixing beyond clamp |arg|={BEYOND_CLAMP_ARG}] dim={dim} err={err:.3e}")
        assert err > 1e-10

        for target, expect_engaged in ((INTERIOR_ARG, False), (BEYOND_CLAMP_ARG, True)):
            x = mixing(_z_reaching(mixing, target)[None, :])
            pre = x @ mixing.R_2
            post = np.clip(pre, -(1.0 - 1e-7), 1.0 - 1e-7)
            engaged = bool(np.any(pre != post))
            assert engaged == expect_engaged, f"dim={dim} |arg|={target} engaged={engaged}"


def test_rotations_are_proper():
    mixing = SmoothMixing(dim=5, seed=7)
    for R in (mixing.R_1, mixing.R_2):
        assert np.max(np.abs(R @ R.T - np.eye(R.shape[0]))) < 1e-12
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_lipschitz_bound_holds():
    dim = 3
    mixing = SmoothMixing(dim=dim, seed=2)
    rng = np.random.default_rng(9)
    bound = mixing.lipschitz_bound()
    eps = 1e-6

    max_spectral_norm = 0.0
    for _ in range(100):
        z = rng.standard_normal(dim)
        J = np.empty((dim, dim))
        for j in range(dim):
            dz = np.zeros(dim)
            dz[j] = eps
            J[:, j] = (mixing(z + dz) - mixing(z - dz)) / (2.0 * eps)
        spectral_norm = np.linalg.svd(J, compute_uv=False)[0]
        max_spectral_norm = max(max_spectral_norm, spectral_norm)
        assert spectral_norm <= bound + 1e-9
    print(f"[lipschitz] bound={bound:.4f} max_observed={max_spectral_norm:.4f}")


def test_partial_observation_composability():
    dim = 2
    mixing = SmoothMixing(dim=dim, seed=3)
    partial = PartialObservation((0,))
    rng = np.random.default_rng(11)
    z = rng.standard_normal((10, 5, dim))

    x_full = mixing(z)
    x_partial = partial(x_full)
    assert x_partial.shape == z.shape[:-1] + (1,)
    assert np.array_equal(x_partial, x_full[..., :1])
