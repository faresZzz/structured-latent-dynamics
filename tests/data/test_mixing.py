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
