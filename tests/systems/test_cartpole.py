import numpy as np

from sld.systems.cartpole import CartPole


def test_linearize_matches_closed_form_and_unstable_eigenvalue():
    system = CartPole()
    A, B = system.linearize(np.zeros(4), np.zeros(1))
    A_analytic, B_analytic = system.linearize_upright()
    np.testing.assert_allclose(A, A_analytic, atol=1e-6)
    np.testing.assert_allclose(B, B_analytic, atol=1e-6)

    eigvals = np.linalg.eigvals(A_analytic)
    lam_expected = np.sqrt(
        (system.m_c + system.m_p) * system.g / (system.l_pole * system.m_c)
    )
    assert np.min(np.abs(eigvals - lam_expected)) < 1e-9


def test_energy_conserved_with_zero_force():
    system = CartPole()
    # large swing, away from the unstable point, keeps |E(0)| well off zero
    x0 = np.array([0.0, 0.0, 2.5, 0.0])
    dt = 1e-3
    n_steps = 10_000  # 10 s
    u_seq = np.zeros((n_steps, 1))
    traj = system.integrate(x0, u_seq, dt)

    energy = system.energy(traj)
    max_relative_drift = np.max(np.abs(energy - energy[0]) / np.abs(energy[0]))
    print(f"[test2] max_relative_drift = {max_relative_drift:.3e}")
    assert max_relative_drift < 1e-8


def test_partial_observation_returns_position_and_angle():
    system = CartPole()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, 4))
    observation = system.observe(x, "partial")
    assert observation.shape == (6, 2)
    np.testing.assert_array_equal(observation, x[:, [0, 2]])


def test_sample_ic_deterministic_bounds_and_unknown_regime():
    system = CartPole()
    for regime in ("default", "hanging", "wide"):
        rng_a = np.random.default_rng(11)
        rng_b = np.random.default_rng(11)
        x_a = system.sample_ic(rng_a, regime=regime)
        x_b = system.sample_ic(rng_b, regime=regime)
        np.testing.assert_array_equal(x_a, x_b)

    rng = np.random.default_rng(12)
    for _ in range(1000):
        x = system.sample_ic(rng, regime="hanging")
        theta = x[2]
        assert np.pi - 0.3 <= theta <= np.pi + 0.3

    try:
        system.sample_ic(rng, regime="bogus")
        assert False, "expected ValueError for unknown regime"
    except ValueError:
        pass
