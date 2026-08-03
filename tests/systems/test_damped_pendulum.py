import numpy as np

from sld.systems.damped_pendulum import DampedPendulum


def test_small_angle_matches_analytic_linear_solution():
    # At amplitude 1e-3 the nonlinear correction to sin(theta) is
    # O(theta0^3) ~ 1e-9, far below the 1e-6 tolerance below, so the
    # analytic solution of the LINEARIZED equation is a valid reference.
    zeta = 0.1
    pendulum = DampedPendulum(zeta=zeta)
    theta0, theta_dot0 = 1e-3, 0.0
    dt = 1e-3
    n_steps = 10_000  # 10 s
    u_seq = np.zeros((n_steps, 1))
    traj = pendulum.integrate(np.array([theta0, theta_dot0]), u_seq, dt)

    omega0 = pendulum.omega0
    omega_d = omega0 * np.sqrt(1.0 - zeta**2)
    t = np.arange(n_steps + 1) * dt
    theta_analytic = np.exp(-zeta * omega0 * t) * (
        theta0 * np.cos(omega_d * t)
        + ((theta_dot0 + zeta * omega0 * theta0) / omega_d) * np.sin(omega_d * t)
    )

    max_abs_error = np.max(np.abs(traj[:, 0] - theta_analytic))
    print(f"[test1] max_abs_error = {max_abs_error:.3e}")
    assert max_abs_error < 1e-6


def test_energy_conserved_at_zero_damping():
    pendulum = DampedPendulum(zeta=0.0)
    x0 = np.array([2.0, 0.0])
    dt = 1e-3
    n_steps = 10_000  # 10 s
    u_seq = np.zeros((n_steps, 1))
    traj = pendulum.integrate(x0, u_seq, dt)

    energy = pendulum.energy(traj)
    max_relative_drift = np.max(np.abs(energy - energy[0]) / energy[0])
    print(f"[test2] max_relative_drift = {max_relative_drift:.3e}")
    assert max_relative_drift < 1e-8


def test_rk4_fourth_order_convergence():
    pendulum = DampedPendulum(zeta=0.1)
    x0 = np.array([2.0, 0.5])
    duration = 5.0
    dt_coarse = 0.02
    dt_fine = 0.01
    dt_ref = dt_coarse / 64.0  # shared reference grid for both step sizes

    def final_state(dt: float) -> np.ndarray:
        n_steps = round(duration / dt)
        u_seq = np.zeros((n_steps, 1))
        return pendulum.integrate(x0, u_seq, dt)[-1]

    x_ref = final_state(dt_ref)
    x_coarse = final_state(dt_coarse)
    x_fine = final_state(dt_fine)

    error_coarse = np.linalg.norm(x_coarse - x_ref)
    error_fine = np.linalg.norm(x_fine - x_ref)
    ratio = error_coarse / error_fine
    print(f"[test3] error(dt=0.02) = {error_coarse:.3e}, error(dt=0.01) = {error_fine:.3e}, ratio = {ratio:.3f}")
    assert 14.0 <= ratio <= 18.0


def test_analytic_eigenvalues_match_finite_difference_linearization():
    pendulum = DampedPendulum(zeta=0.1)
    A, _B = pendulum.linearize(np.zeros(2), np.zeros(1))
    eig_numeric = np.sort_complex(np.linalg.eigvals(A))
    eig_analytic = np.sort_complex(pendulum.analytic_eigenvalues())
    np.testing.assert_allclose(eig_numeric, eig_analytic, atol=1e-6)


def test_sample_ic_deterministic_and_regime_bounds():
    pendulum = DampedPendulum()

    for regime in ("default", "wide"):
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)
        x_a = pendulum.sample_ic(rng_a, regime=regime)
        x_b = pendulum.sample_ic(rng_b, regime=regime)
        np.testing.assert_array_equal(x_a, x_b)

    rng = np.random.default_rng(1)
    for _ in range(1000):
        theta, theta_dot = pendulum.sample_ic(rng, regime="wide")
        assert -np.pi <= theta <= np.pi
        assert -2.0 * pendulum.omega0 <= theta_dot <= 2.0 * pendulum.omega0

    try:
        pendulum.sample_ic(rng, regime="bogus")
        assert False, "expected ValueError for unknown regime"
    except ValueError:
        pass


def test_partial_observation_keeps_angle_only():
    pendulum = DampedPendulum()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 3, 2))
    observation = pendulum.observe(x, "partial")
    assert observation.shape == (4, 3, 1)
    np.testing.assert_array_equal(observation, x[..., :1])


def test_force_matches_hand_computation_with_nonunit_parameters():
    """Positive (REGLES 7.1): every test above uses defaults m=L=1, which
    silently collapses two independent scale factors at once -- the torque
    term u[0]/(m*L^2) becomes plain u[0], and the restoring term's
    omega0^2 = g/L becomes plain g -- so a wrong m*L^2 or a wrong g/L would
    pass unnoticed. Here m != L != 1, g is away from its 9.81 default, zeta
    != 0, and theta, theta_dot, u are all nonzero, so both scale factors are
    forced to act and cannot cancel each other out (m != L rules out any
    m*L^2 confusion hiding behind m == L).

    Hand arithmetic, independent of DampedPendulum.f and DampedPendulum.omega0
    (only the equation of motion from the class docstring is used):
        m=2.0, L=3.0, g=10.0, zeta=0.2, theta=0.4, theta_dot=-0.3, u=1.5
        omega0^2 = g/L = 10.0/3.0 = 3.3333333333333335
        gravity term = -omega0^2*sin(theta) = -3.3333333333333335*sin(0.4)
                     = -3.3333333333333335*0.3894183423086505
                     = -1.2980611410288352
        damping term = -2*zeta*omega0*theta_dot, omega0 = sqrt(10/3) = 1.8257418583505538
                     = -2*0.2*1.8257418583505538*(-0.3) = 0.21908902300206645
        torque term  = u/(m*L^2) = 1.5/(2.0*9.0) = 1.5/18.0 = 0.08333333333333333
        theta_ddot   = sum of the three terms = -0.9956387846934353
    (values confirmed by running the arithmetic below, not hand-typed digits).
    """
    m, L, g, zeta = 2.0, 3.0, 10.0, 0.2
    pendulum = DampedPendulum(m=m, L=L, g=g, zeta=zeta)
    theta, theta_dot, u = 0.4, -0.3, 1.5

    omega0_sq = g / L
    omega0 = np.sqrt(omega0_sq)
    theta_ddot_hand = (
        -omega0_sq * np.sin(theta) - 2.0 * zeta * omega0 * theta_dot + u / (m * L**2)
    )
    f_hand = np.array([theta_dot, theta_ddot_hand])

    f_impl = pendulum.f(np.array([theta, theta_dot]), np.array([u]))
    max_abs_error = np.max(np.abs(f_impl - f_hand))
    print(f"[force nonunit] f_hand={f_hand}, f_impl={f_impl}, max_abs_error={max_abs_error:.3e}")
    assert max_abs_error < 1e-12  # short float64 algebra, rounding only (R3)

    # Negative: a plausible typo drops L^2 from the torque scaling
    # (u/m instead of u/(m*L^2)) -- invisible at the repo's default L=1, but
    # an 18x error in that one term at this L=3.
    theta_ddot_wrong = -omega0_sq * np.sin(theta) - 2.0 * zeta * omega0 * theta_dot + u / m
    f_wrong = np.array([theta_dot, theta_ddot_wrong])
    wrong_error = np.max(np.abs(f_wrong - f_hand))
    print(f"[force nonunit negative] wrong_error={wrong_error:.3e}")
    assert wrong_error > 1e-12  # same oracle, same tolerance: must fail
