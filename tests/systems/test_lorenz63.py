import numpy as np

from sld.systems.lorenz63 import Lorenz63


def test_largest_lyapunov_exponent_benettin():
    # Benettin-style two-trajectory renormalization estimate of the largest
    # Lyapunov exponent, checked against the Lorenz (1963) reference value.
    system = Lorenz63()
    rng = np.random.default_rng(0)
    x0 = system.sample_ic(rng, regime="on_attractor")

    delta0 = 1e-8
    e = np.array([1.0, 0.0, 0.0])
    dt = 0.01
    tau = 1.0
    n_steps = round(tau / dt)
    n_windows = 55
    u_seq = np.zeros((n_steps, 0))

    x = x0.copy()
    xp = x0 + delta0 * e
    log_growth = []
    for _ in range(n_windows):
        x_next = system.integrate(x, u_seq, dt)[-1]
        xp_next = system.integrate(xp, u_seq, dt)[-1]
        diff = xp_next - x_next
        growth = np.linalg.norm(diff) / delta0
        log_growth.append(np.log(growth))
        # renormalize the twin back to distance delta0 along the current
        # separation direction before the next window
        xp_next = x_next + delta0 * diff / np.linalg.norm(diff)
        x, xp = x_next, xp_next

    lambda_hat = np.mean(log_growth[5:]) / tau
    print(f"[test1] lambda_hat = {lambda_hat:.4f} (reference {system.LAMBDA_MAX_REF})")
    assert 0.7 <= lambda_hat <= 1.1


def test_sample_ic_on_attractor_deterministic_and_bounded():
    system = Lorenz63()
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    x_a = system.sample_ic(rng_a, regime="on_attractor")
    x_b = system.sample_ic(rng_b, regime="on_attractor")
    np.testing.assert_array_equal(x_a, x_b)
    assert np.max(np.abs(x_a)) < 60.0

    rng_c = np.random.default_rng(8)
    x_c = system.sample_ic(rng_c, regime="on_attractor")
    assert not np.array_equal(x_a, x_c)

    try:
        system.sample_ic(rng_a, regime="bogus")
        assert False, "expected ValueError for unknown regime"
    except ValueError:
        pass


def test_force_matches_hand_computation_with_nonunit_parameters():
    """Positive (REGLES 7.1): every test above uses the classic default
    parameters (sigma=10, rho=28, beta=8/3), so a hardcoded literal
    replacing self.sigma/self.rho/self.beta anywhere in f would pass
    unnoticed. Here sigma, rho, beta are all away from their defaults, at a
    nontrivial, off-attractor point (not the origin, not near the classic
    attractor's typical range).

    Hand arithmetic, independent of Lorenz63.f (only the equations of
    motion from the class docstring are used):
        sigma=12.0, rho=20.0, beta=2.5, x=(2.0, -1.0, 30.0)
        x_dot = sigma*(y-x) = 12.0*(-1.0-2.0) = 12.0*(-3.0) = -36.0
        y_dot = x*(rho-z) - y = 2.0*(20.0-30.0) - (-1.0) = -20.0+1.0 = -19.0
        z_dot = x*y - beta*z = 2.0*(-1.0) - 2.5*30.0 = -2.0-75.0 = -77.0
    (values confirmed by running the arithmetic below, not hand-typed digits).
    """
    sigma, rho, beta = 12.0, 20.0, 2.5
    system = Lorenz63(sigma=sigma, rho=rho, beta=beta)
    x, y, z = 2.0, -1.0, 30.0

    x_dot_hand = sigma * (y - x)
    y_dot_hand = x * (rho - z) - y
    z_dot_hand = x * y - beta * z
    f_hand = np.array([x_dot_hand, y_dot_hand, z_dot_hand])

    f_impl = system.f(np.array([x, y, z]), np.zeros(0))
    max_abs_error = np.max(np.abs(f_impl - f_hand))
    print(f"[lorenz force nonunit] f_hand={f_hand}, f_impl={f_impl}, max_abs_error={max_abs_error:.3e}")
    assert max_abs_error < 1e-12  # short float64 algebra, rounding only (R3)

    # Negative: a plausible typo drops the beta scaling in the z equation
    # (x*y - z instead of x*y - beta*z) -- invisible at nothing (there is
    # no default that collapses beta to 1), but beta=2.5 here versus
    # beta=1 implied by the typo is a large, easily-detected error.
    z_dot_wrong = x * y - z
    f_wrong = np.array([x_dot_hand, y_dot_hand, z_dot_wrong])
    wrong_error = np.max(np.abs(f_wrong - f_hand))
    print(f"[lorenz force nonunit negative] wrong_error={wrong_error:.3e}")
    assert wrong_error > 1e-12  # same oracle, same tolerance: must fail


def test_integrate_deterministic():
    system = Lorenz63()
    rng = np.random.default_rng(3)
    x0 = system.sample_ic(rng, regime="on_attractor")
    dt = 0.01
    u_seq = np.zeros((500, 0))
    traj_a = system.integrate(x0, u_seq, dt)
    traj_b = system.integrate(x0, u_seq, dt)
    np.testing.assert_array_equal(traj_a, traj_b)
