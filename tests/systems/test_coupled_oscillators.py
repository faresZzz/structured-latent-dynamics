import numpy as np
import pytest

from sld.systems.coupled_oscillators import CoupledOscillators


def test_linearize_matches_analytic_matrices_and_modes():
    system = CoupledOscillators()
    A, B = system.linearize(np.zeros(4), np.zeros(1))
    np.testing.assert_allclose(A, system.A_mat, atol=1e-9)
    np.testing.assert_allclose(B, system.B_mat, atol=1e-9)

    eig_numeric = np.sort_complex(np.linalg.eigvals(A))
    eig_analytic = np.sort_complex(system.analytic_modes())
    np.testing.assert_allclose(eig_numeric, eig_analytic, atol=1e-10)


def test_analytic_modes_match_hand_built_state_space():
    """Positive (REGLES 7.1): the test above validates ``analytic_modes()``
    against ``system.A_mat`` -- the SAME matrix, fetched from the object
    under test, and against ``linearize()``'s finite difference of the very
    same ``f`` that ``A_mat`` parameterizes. This builds the 4x4 (A, B)
    state-space matrices INDEPENDENTLY, directly from the mass/spring/
    damper equations of motion (class docstring), with ASYMMETRIC
    parameters (m1 != m2, k1 != k2) so a left/right index-swap bug could
    not hide behind an accidental symmetry. Eigenvalues of THIS hand-built
    matrix (an entirely separate ``np.linalg.eigvals`` call on a matrix the
    TEST built, not one fetched from the system) are compared to
    ``CoupledOscillators.analytic_modes()``.
    """
    m1, m2, k1, k_c, k2, c = 1.3, 2.1, 0.9, 0.4, 1.7, 0.15  # asymmetric on purpose
    system = CoupledOscillators(m1=m1, m2=m2, k1=k1, k_c=k_c, k2=k2, c=c)

    # hand-derived from the equations of motion (class docstring):
    #   m1*q1_ddot = -k1*q1 - k_c*(q1-q2) - c*q1_dot + u
    #   m2*q2_ddot = -k2*q2 - k_c*(q2-q1) - c*q2_dot
    # state x = (q1, q2, q1_dot, q2_dot)
    A_hand = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(k1 + k_c) / m1, k_c / m1, -c / m1, 0.0],
            [k_c / m2, -(k2 + k_c) / m2, 0.0, -c / m2],
        ]
    )

    eig_hand = np.sort_complex(np.linalg.eigvals(A_hand))
    eig_system = np.sort_complex(system.analytic_modes())
    # exact algebraic construction, both sides float64: rounding only, no
    # discretization/approximation anywhere in this comparison
    np.testing.assert_allclose(eig_system, eig_hand, atol=1e-12)


def test_analytic_modes_undamped_symmetric_closed_form():
    """Positive, closed-form special case: undamped (c=0), symmetric
    (m1=m2=m, k1=k2=k) coupled oscillators decouple into a symmetric normal
    mode (q1=q2, omega=sqrt(k/m): mass 1 and 2 move together, the coupling
    spring never stretches) and an antisymmetric mode (q1=-q2,
    omega=sqrt((k+2*k_c)/m): the coupling spring stretches by 2*q1, doubling
    its restoring contribution) -- textbook normal-mode decomposition, no
    linear algebra needed to derive it. Modes are purely imaginary:
    +/-i*sqrt(k/m), +/-i*sqrt((k+2*k_c)/m). Real/imaginary parts are checked
    separately (not via ``sort_complex`` on the raw eigenvalues) because the
    two conjugate pairs' real parts are only ~1e-16 apart from each other
    here (both exactly 0 analytically), which makes ``sort_complex``'s
    real-part-first ordering unstable across float64 rounding noise.
    """
    m, k, k_c = 1.7, 2.3, 0.6
    system = CoupledOscillators(m1=m, m2=m, k1=k, k_c=k_c, k2=k, c=0.0)
    eig_system = system.analytic_modes()

    omega_sym = np.sqrt(k / m)
    omega_antisym = np.sqrt((k + 2.0 * k_c) / m)

    # undamped: purely imaginary spectrum (real part exactly 0 analytically)
    np.testing.assert_allclose(eig_system.real, 0.0, atol=1e-12)
    # match on |imaginary part| instead of sort_complex (see docstring)
    imag_sorted = np.sort(np.abs(eig_system.imag))
    expected_imag_sorted = np.sort(np.array([omega_sym, omega_sym, omega_antisym, omega_antisym]))
    np.testing.assert_allclose(imag_sorted, expected_imag_sorted, atol=1e-12)


def test_analytic_modes_sensitive_to_perturbed_physical_parameter():
    """Negative (REGLES 7.1, "this is the whole point"): perturb ONE
    physical parameter (k2) in the hand-built matrix from the first test
    above and confirm the comparison now FAILS -- proving that test is
    actually sensitive to the physics (a wrong k2 changes the modes), not
    merely checking shape/dtype/finiteness."""
    m1, m2, k1, k_c, k2, c = 1.3, 2.1, 0.9, 0.4, 1.7, 0.15
    system = CoupledOscillators(m1=m1, m2=m2, k1=k1, k_c=k_c, k2=k2, c=c)

    k2_wrong = k2 + 0.3  # perturbed -- not the system's actual k2
    A_wrong = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(k1 + k_c) / m1, k_c / m1, -c / m1, 0.0],
            [k_c / m2, -(k2_wrong + k_c) / m2, 0.0, -c / m2],
        ]
    )
    eig_wrong = np.sort_complex(np.linalg.eigvals(A_wrong))
    eig_system = np.sort_complex(system.analytic_modes())

    max_diff = np.max(np.abs(eig_system - eig_wrong))
    print(f"[perturbed k2] max_diff={max_diff:.3e}")
    assert max_diff > 1e-6  # comparable to the physical perturbation, not float rounding
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(eig_system, eig_wrong, atol=1e-12)


def test_exact_gramian_symmetric_pd_and_monotone():
    system = CoupledOscillators()
    T, dt = 10.0, 0.01
    W_T = system.exact_gramian(T, dt)
    W_2T = system.exact_gramian(2.0 * T, dt)

    # symmetrization is exact bit-for-bit, not just close
    np.testing.assert_array_equal(W_T, W_T.T)

    min_eig_T = np.min(np.linalg.eigvalsh(W_T))
    print(f"[test2] min eigval W(T) = {min_eig_T:.3e}")
    assert min_eig_T > 0.0

    min_eig_growth = np.min(np.linalg.eigvalsh(W_2T - W_T))
    print(f"[test2] min eigval W(2T)-W(T) = {min_eig_growth:.3e}")
    assert min_eig_growth > -1e-12


def test_force_matches_hand_built_physics_with_nonunit_parameters():
    """Positive (REGLES 7.1 + R2): f(x, u) reconstructed directly from the
    mass/spring/damper equations of motion in the class docstring --
        m1*q1_ddot = -k1*q1 - k_c*(q1-q2) - c*q1_dot + u
        m2*q2_ddot = -k2*q2 - k_c*(q2-q1) - c*q2_dot
    -- NEVER via system.A_mat. Using A_mat as the reference would validate
    the object against itself (the exact defect already fixed for the
    eigenvalues by test_analytic_modes_match_hand_built_state_space above,
    which this test mirrors for the force instead of the modes). m1 != m2,
    k1 != k2, k_c != 0, c != 0 are all non-unit and asymmetric, and the
    state and action are all nonzero.

    Hand arithmetic (independent of CoupledOscillators.f and .A_mat):
        m1=1.5, m2=2.5, k1=0.8, k2=1.3, k_c=0.6, c=0.25
        x = (q1, q2, q1_dot, q2_dot) = (0.3, -0.2, 0.5, -0.4), u = 0.7
        q1_ddot = (-k1*q1 - k_c*(q1-q2) - c*q1_dot + u) / m1
                = (-0.24 - 0.3 - 0.125 + 0.7) / 1.5 = 0.035 / 1.5
                = 0.02333333333333328
        q2_ddot = (-k2*q2 - k_c*(q2-q1) - c*q2_dot) / m2
                = (0.26 + 0.3 + 0.1) / 2.5 = 0.66 / 2.5 = 0.264
    (values confirmed by running the arithmetic below, not hand-typed digits).
    """
    m1, m2, k1, k2, k_c, c = 1.5, 2.5, 0.8, 1.3, 0.6, 0.25
    system = CoupledOscillators(m1=m1, m2=m2, k1=k1, k_c=k_c, k2=k2, c=c)
    q1, q2, q1_dot, q2_dot = 0.3, -0.2, 0.5, -0.4
    u = 0.7

    q1_ddot_hand = (-k1 * q1 - k_c * (q1 - q2) - c * q1_dot + u) / m1
    q2_ddot_hand = (-k2 * q2 - k_c * (q2 - q1) - c * q2_dot) / m2
    f_hand = np.array([q1_dot, q2_dot, q1_ddot_hand, q2_ddot_hand])

    x = np.array([q1, q2, q1_dot, q2_dot])
    f_impl = system.f(x, np.array([u]))
    max_abs_error = np.max(np.abs(f_impl - f_hand))
    print(f"[coupled force nonunit] f_hand={f_hand}, f_impl={f_impl}, max_abs_error={max_abs_error:.3e}")
    assert max_abs_error < 1e-12  # short float64 algebra, rounding only (R3)

    # Negative: a plausible copy-paste typo -- equation 2 written from
    # equation 1 but the divisor mass not swapped (m1 instead of m2),
    # invisible whenever m1 == m2 but a real error here since m1 != m2.
    q2_ddot_wrong = (-k2 * q2 - k_c * (q2 - q1) - c * q2_dot) / m1
    f_wrong = np.array([q1_dot, q2_dot, q1_ddot_hand, q2_ddot_wrong])
    wrong_error = np.max(np.abs(f_wrong - f_hand))
    print(f"[coupled force nonunit negative] wrong_error={wrong_error:.3e}")
    assert wrong_error > 1e-12  # same oracle, same tolerance: must fail


def test_rk4_matches_exact_linear_propagation():
    system = CoupledOscillators()
    x0 = np.array([0.5, -0.3, 0.1, 0.2])
    T = 5.0
    dt = 1e-3
    n_steps = round(T / dt)
    u_seq = np.zeros((n_steps, 1))
    x_rk4 = system.integrate(x0, u_seq, dt)[-1]

    # reference: exact matrix-exponential propagation via eigendecomposition
    lam, V = np.linalg.eig(system.A_mat)
    V_inv = np.linalg.inv(V)
    x_exact = np.real(V @ np.diag(np.exp(lam * T)) @ V_inv @ x0)

    max_abs_error = np.max(np.abs(x_rk4 - x_exact))
    print(f"[test3] max_abs_error = {max_abs_error:.3e}")
    assert max_abs_error < 1e-8
