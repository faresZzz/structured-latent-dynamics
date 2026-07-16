import numpy as np

from sld.data.regimes import (
    ActionConditionedRegime,
    AnisotropicRegime,
    NarrowPolicyRegime,
    OURegime,
)
from sld.systems.base import DynamicalSystem
from sld.systems.damped_pendulum import DampedPendulum


class _ZeroSystem(DynamicalSystem):
    """Minimal autonomous-drift stub: 3-D action, isotropic OU has nothing to fight."""

    state_dim = 1
    action_dim = 3
    u_max = 10.0
    partial_indices = (0,)

    def f(self, x, u):
        return np.zeros(1, dtype=np.float64)

    def sample_ic(self, rng, regime="default"):
        return np.zeros(1, dtype=np.float64)


def test_ou_isotropy_converges():
    # At dt=0.01 consecutive OU samples have correlation exp(-theta_ou*dt)
    # ~ 0.990, so the integrated autocorrelation time is (1+rho)/(1-rho)
    # ~ 200 steps and the eigen-ratio estimator only sees T/200 effective
    # samples. T = 200_000 gives ~1000 effective samples, where the
    # isotropic eigen-ratio concentrates near 1 + O(sqrt(2/1000)) ~ 1.09,
    # leaving honest margin under the 1.2 threshold (at T = 20_000 the
    # median ratio is ~1.37 by pure finite-sample noise).
    system = _ZeroSystem()
    regime = OURegime(sigma_stationary=1.0)
    rng = np.random.default_rng(0)
    T, dt = 200_000, 0.01
    _, u_seq = regime.sample_episode(system, T, dt, rng)

    cov = np.cov(u_seq, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigen_ratio = eigvals.max() / eigvals.min()
    stds = np.sqrt(np.diag(cov))
    print(f"[ou isotropy] eigen_ratio={eigen_ratio:.4f} stds={stds}")
    assert eigen_ratio < 1.2
    assert np.all(np.abs(stds - 1.0) < 0.1)


def test_anisotropic_ic_ratio():
    system = DampedPendulum()
    regime = AnisotropicRegime(ic_scales=(3.0, 0.3))
    rng = np.random.default_rng(1)
    n_episodes = 2000
    x0s = np.empty((n_episodes, 2), dtype=np.float64)
    for i in range(n_episodes):
        x0, _ = regime.sample_episode(system, T=1, dt=0.01, rng=rng)
        x0s[i] = x0

    ratio = np.std(x0s[:, 0]) / np.std(x0s[:, 1])
    print(f"[anisotropic ic] ratio={ratio:.4f}")
    assert abs(ratio - 10.0) / 10.0 < 0.2


def test_action_conditioned_correlation():
    system = DampedPendulum()
    regime = ActionConditionedRegime(K_gain=np.array([[0.8, 0.4]]), sigma=0.3)
    rng = np.random.default_rng(2)
    T, dt = 500, 0.01
    x0, u_seq = regime.sample_episode(system, T, dt, rng)
    z_true = system.integrate(x0, u_seq, dt)
    x_t = z_true[:-1]

    correlations = [np.corrcoef(u_seq[:, 0], x_t[:, i])[0, 1] for i in range(x_t.shape[1])]
    max_abs_corr = max(abs(c) for c in correlations)
    print(f"[action-state corr] correlations={correlations} max_abs={max_abs_corr:.4f}")
    assert max_abs_corr > 0.3


def test_narrow_policy_pumps_toward_target_energy():
    system = DampedPendulum()
    E_target = 2.0 * system.m * system.g * system.L
    regime = NarrowPolicyRegime(E_target=E_target, k_e=1.0, sigma=0.05)
    rng = np.random.default_rng(3)
    n_episodes, T, dt = 20, 1500, 0.01

    successes = 0
    for _ in range(n_episodes):
        x0, u_seq = regime.sample_episode(system, T, dt, rng)
        z_true = system.integrate(x0, u_seq, dt)
        e0 = system.energy(z_true[0])
        eT = system.energy(z_true[-1])
        if abs(eT - E_target) < abs(e0 - E_target):
            successes += 1

    fraction = successes / n_episodes
    print(f"[narrow policy] success fraction={fraction:.2f}")
    assert fraction >= 0.7


def test_replay_contract_action_conditioned():
    system = DampedPendulum()
    regime = ActionConditionedRegime(K_gain=np.array([[0.8, 0.4]]), sigma=0.3)
    rng = np.random.default_rng(4)
    T, dt = 200, 0.01
    x0, u_seq = regime.sample_episode(system, T, dt, rng)

    # manual step-by-step replay using the recorded actions (no new randomness)
    x = x0
    for u in u_seq:
        x = system.integrate(x, u[None, :], dt)[-1]
    manual_final = x

    replay_final = system.integrate(x0, u_seq, dt)[-1]
    max_abs_diff = np.max(np.abs(replay_final - manual_final))
    print(f"[replay contract] max_abs_diff={max_abs_diff:.3e}")
    assert max_abs_diff < 1e-12


def test_determinism_all_regimes():
    system = DampedPendulum()
    regimes = [
        OURegime(sigma_stationary=1.0),
        NarrowPolicyRegime(E_target=19.62),
        AnisotropicRegime(ic_scales=(3.0, 0.3)),
        ActionConditionedRegime(K_gain=np.array([[0.8, 0.4]])),
    ]
    T, dt = 10, 0.01
    for regime in regimes:
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        x0_a, u_a = regime.sample_episode(system, T, dt, rng_a)
        x0_b, u_b = regime.sample_episode(system, T, dt, rng_b)
        assert np.array_equal(x0_a, x0_b)
        assert np.array_equal(u_a, u_b)
