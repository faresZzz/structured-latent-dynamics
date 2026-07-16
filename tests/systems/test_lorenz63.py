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


def test_integrate_deterministic():
    system = Lorenz63()
    rng = np.random.default_rng(3)
    x0 = system.sample_ic(rng, regime="on_attractor")
    dt = 0.01
    u_seq = np.zeros((500, 0))
    traj_a = system.integrate(x0, u_seq, dt)
    traj_b = system.integrate(x0, u_seq, dt)
    np.testing.assert_array_equal(traj_a, traj_b)
