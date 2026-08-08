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


class _UnclippedZeroSystem(DynamicalSystem):
    """Like ``_ZeroSystem`` but with a huge ``u_max`` so the exact-kernel
    tests below are never touched by ``_ou_action_sequence``'s +/-u_max
    clip -- the comparison must be to the RAW recursion, not a clipped one.
    ``sample_ic`` makes no ``rng`` calls (like ``_ZeroSystem``), so an
    independently-seeded RNG replays the exact same Gaussian draws
    ``_ou_action_sequence`` consumes.
    """

    state_dim = 1
    action_dim = 3
    u_max = 1e6
    partial_indices = (0,)

    def f(self, x, u):
        return np.zeros(1, dtype=np.float64)

    def sample_ic(self, rng, regime="default"):
        return np.zeros(1, dtype=np.float64)


def test_ou_exact_transition_kernel_matches_closed_form():
    """Positive (REGLES 7.1): ``test_ou_isotropy_converges`` above only
    checks loose distributional statistics over 200k steps -- an
    Euler-Maruyama discretization would pass it too, since both schemes
    share the same continuous-time stationary distribution. This test pins
    the EXACT one-step kernel instead: theta_ou*dt = 1 (not a small-dt
    regime, so exact vs. approximate schemes disagree by O(1), not O(dt)),
    u0 and the step-1 noise draw are FIXED (recovered by replaying the same
    seed through an independent RNG, since ``_UnclippedZeroSystem.sample_ic``
    makes no rng calls -- so the two streams are provably identical), and
    the comparison is against the module's own claimed convention
    (docstring of ``_ou_action_sequence``): u_1 = u_0*exp(-theta*dt) +
    sigma_stationary*sqrt(1 - exp(-2*theta*dt))*xi_1.
    """
    theta_ou, dt, sigma_stationary, seed = 1.0, 1.0, 1.0, 123  # theta_ou*dt = 1 exactly
    system = _UnclippedZeroSystem()
    regime = OURegime(theta_ou=theta_ou, sigma_stationary=sigma_stationary)

    rng = np.random.default_rng(seed)
    _, u_seq = regime.sample_episode(system, T=2, dt=dt, rng=rng)

    # independent replay of the identical seed/draw order (sample_ic above
    # makes zero rng calls, so a fresh generator's first two
    # standard_normal(3) draws are exactly xi_0 (-> u_seq[0]) and xi_1 (->
    # the diffusion term of u_seq[1]) that _ou_action_sequence consumed
    rng_oracle = np.random.default_rng(seed)
    xi0 = rng_oracle.standard_normal(system.action_dim)
    xi1 = rng_oracle.standard_normal(system.action_dim)

    decay = np.exp(-theta_ou * dt)
    diffusion = sigma_stationary * np.sqrt(1.0 - np.exp(-2.0 * theta_ou * dt))
    u0_expected = sigma_stationary * xi0
    u1_expected = u0_expected * decay + diffusion * xi1

    # float64 rounding over a handful of exp/sqrt/multiply-add ops on O(1)
    # operands: ~1e-15 unit-in-last-place accumulation, so 1e-12 leaves 3
    # orders of margin
    max_err = np.max(np.abs(u_seq - np.stack([u0_expected, u1_expected])))
    print(f"[ou exact kernel] max_err={max_err:.3e}")
    assert max_err < 1e-12


def test_ou_euler_maruyama_step_fails_exact_kernel_comparison():
    """Negative (REGLES 7.1, "this is the whole point"): given the IDENTICAL
    u0 and noise draw as the positive test above, an Euler-Maruyama
    discretization of the same OU SDE (dU = -theta*U dt + sigma_em*dW,
    sigma_em = sigma_stationary*sqrt(2*theta_ou) -- the standard SDE
    parameterization whose CONTINUOUS-time stationary variance matches
    sigma_stationary^2, i.e. genuinely "the same physical process, wrong
    discretization", not an arbitrary strawman) must FAIL the exact-kernel
    comparison at the same 1e-12 tolerance -- and must disagree with what
    the actual ``OURegime``/``_ou_action_sequence`` code produces. This is
    what ``test_ou_isotropy_converges`` could never catch: both schemes
    converge to the same stationary distribution, so only a one-step,
    fixed-noise comparison exposes the difference.
    """
    theta_ou, dt, sigma_stationary, seed = 1.0, 1.0, 1.0, 123
    system = _UnclippedZeroSystem()
    regime = OURegime(theta_ou=theta_ou, sigma_stationary=sigma_stationary)

    rng = np.random.default_rng(seed)
    _, u_seq_actual = regime.sample_episode(system, T=2, dt=dt, rng=rng)

    rng_oracle = np.random.default_rng(seed)
    xi0 = rng_oracle.standard_normal(system.action_dim)
    xi1 = rng_oracle.standard_normal(system.action_dim)
    u0 = sigma_stationary * xi0

    decay_em = 1.0 - theta_ou * dt  # Euler-Maruyama's linear-in-dt decay
    sigma_em = sigma_stationary * np.sqrt(2.0 * theta_ou)
    diffusion_em = sigma_em * np.sqrt(dt)
    u1_em = u0 * decay_em + diffusion_em * xi1

    decay_exact = np.exp(-theta_ou * dt)
    diffusion_exact = sigma_stationary * np.sqrt(1.0 - np.exp(-2.0 * theta_ou * dt))
    u1_exact = u0 * decay_exact + diffusion_exact * xi1

    max_err_em_vs_exact = np.max(np.abs(u1_em - u1_exact))
    max_err_em_vs_actual = np.max(np.abs(u1_em - u_seq_actual[1]))
    print(f"[ou EM vs exact] max_err={max_err_em_vs_actual:.3e}")
    assert max_err_em_vs_exact > 1e-12  # EM disagrees with the exact-kernel oracle
    assert max_err_em_vs_actual > 1e-12  # ...and with what the real code actually produces


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


def test_action_conditioned_policy_matches_gain_value_by_value():
    """Positive (REGLES 7.1): test_action_conditioned_correlation above only
    checks max(abs(corr(u, x))) > 0.3 -- a SIGN-FLIPPED policy (u = -K_gain
    @ x) would pass that check identically, since |corr| is sign-blind.
    Turn off the noise (sigma=0) and pin K_gain to a signed, non-degenerate
    matrix (unequal-magnitude, nonzero, mixed-sign entries: [0.6, -0.35] --
    a sign flip or a coefficient swap would both change the numbers), then
    compare u_t to K_gain @ x_t AT EVERY STEP t, exactly.

    x_t is recovered by replaying the recorded u_seq through system.integrate
    (the replay contract itself is already covered by
    test_replay_contract_action_conditioned), so this needs no access to
    ActionConditionedRegime internals beyond its public output.
    """
    system = DampedPendulum()
    K_gain = np.array([[0.6, -0.35]])
    regime = ActionConditionedRegime(K_gain=K_gain, sigma=0.0, ic_regime="default")
    rng = np.random.default_rng(100)
    T, dt = 300, 0.01
    x0, u_seq = regime.sample_episode(system, T, dt, rng)

    z_true = system.integrate(x0, u_seq, dt)
    x_t = z_true[:-1]  # state fed to the policy at each step t (see _closed_loop_episode)
    u_expected = x_t @ K_gain.T

    # the clip must never bite here -- otherwise this would be testing
    # clip(), not the policy's use of K_gain
    margin = system.u_max - np.max(np.abs(u_expected))
    max_err = np.max(np.abs(u_seq - u_expected))
    print(
        f"[action policy value-by-value] max_err={max_err:.3e} "
        f"clip_margin={margin:.4f} (u_max={system.u_max})"
    )
    assert margin > 1.0  # measured ~1.63 out of u_max=2.0: comfortably interior
    # short float64 algebra, a 2-term dot product computed two equivalent
    # ways (batched x_t @ K_gain.T vs per-step K_gain @ x): R3 1e-12
    assert max_err < 1e-12


def test_action_conditioned_wrong_sign_fails_value_check():
    """Negative (REGLES 7.1, "this is the whole point"): the check above
    (u_t == K_gain @ x_t) must have teeth against exactly the bug the old
    abs-correlation test could not see -- a sign-flipped gain. Run the SAME
    real ActionConditionedRegime/sample_episode code path with -K_gain
    (identical seed, so identical x0; the trajectories diverge after step 0
    since the applied torque differs, which is fine -- the question is
    whether the ORIGINAL K_gain still "explains" what a -K_gain-driven
    policy actually did) and assert the value-by-value check now FAILS the
    same 1e-12 tolerance.

    K_gain.T is not additionally checked here: K_gain has shape
    (action_dim, state_dim) = (1, 2) for DampedPendulum (the only real
    system usable for this check with action_dim >= 1), and a (2, 1) gain
    cannot even matmul against a (2,)-shaped state -- numpy raises a shape
    error rather than silently computing a same-shaped wrong answer. So
    there is no "transposed" computation to disagree with; the spec's own
    "si la forme de K_gain le permet" clause does not hold here.
    """
    system = DampedPendulum()
    K_gain = np.array([[0.6, -0.35]])
    regime_wrong = ActionConditionedRegime(K_gain=-K_gain, sigma=0.0, ic_regime="default")
    rng = np.random.default_rng(100)
    T, dt = 300, 0.01
    x0, u_seq_wrong = regime_wrong.sample_episode(system, T, dt, rng)

    z_true_wrong = system.integrate(x0, u_seq_wrong, dt)
    x_t_wrong = z_true_wrong[:-1]
    u_expected_original_sign = x_t_wrong @ K_gain.T

    max_err = np.max(np.abs(u_seq_wrong - u_expected_original_sign))
    print(f"[action policy wrong sign] max_err={max_err:.3e} (tol 1e-12, must exceed it)")
    assert max_err > 1e-12


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


def test_narrow_policy_on_coupled_oscillators_raises_no_energy():
    # Documented trap (PROTOCOL.md section 19, Wave D3): CoupledOscillators
    # is registered in sld.experiments.factorial/loading's system registries
    # but does NOT define energy(x) -- a damped, wall-coupled two-mass system
    # has no single natural "energy target" to shape toward, and inventing
    # one is a physics judgment out of scope for the registration wiring.
    # NarrowPolicyRegime.sample_episode requires system.energy(x) (module
    # docstring above); D3's config must use OURegime instead. This test
    # locks in that AttributeError so the trap cannot silently disappear (a
    # future energy() added elsewhere) or silently worsen (a different,
    # harder-to-diagnose failure) without this test being touched.
    from sld.systems.coupled_oscillators import CoupledOscillators

    system = CoupledOscillators()
    assert not hasattr(system, "energy")
    regime = NarrowPolicyRegime(E_target=1.0)
    rng = np.random.default_rng(0)
    try:
        regime.sample_episode(system, T=5, dt=0.02, rng=rng)
    except AttributeError as exc:
        print(f"[narrow policy / coupled oscillators] raised as documented: {exc}")
    else:
        raise AssertionError(
            "expected AttributeError: CoupledOscillators has no energy(x), "
            "but NarrowPolicyRegime.sample_episode did not raise"
        )


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
