import numpy as np

from sld.systems.coupled_oscillators import CoupledOscillators


def test_linearize_matches_analytic_matrices_and_modes():
    system = CoupledOscillators()
    A, B = system.linearize(np.zeros(4), np.zeros(1))
    np.testing.assert_allclose(A, system.A_mat, atol=1e-9)
    np.testing.assert_allclose(B, system.B_mat, atol=1e-9)

    eig_numeric = np.sort_complex(np.linalg.eigvals(A))
    eig_analytic = np.sort_complex(system.analytic_modes())
    np.testing.assert_allclose(eig_numeric, eig_analytic, atol=1e-10)


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
