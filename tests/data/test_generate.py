import json

import numpy as np

from sld.data.generate import build_dataset
from sld.data.mixing import SmoothMixing
from sld.data.regimes import OURegime
from sld.systems.damped_pendulum import DampedPendulum


def test_determinism_full_pipeline():
    system = DampedPendulum()
    regime = OURegime(sigma_stationary=1.0)
    kwargs = dict(system=system, regime=regime, obs=None, n_episodes=12, T=50, dt=0.01, seed=5)

    result_a = build_dataset(**kwargs)
    result_b = build_dataset(**kwargs)
    for split in ("train", "val", "test"):
        for key in ("x", "u", "z_true"):
            assert np.array_equal(result_a[split][key], result_b[split][key])


def test_splits_disjoint_and_exhaustive():
    system = DampedPendulum()
    regime = OURegime(sigma_stationary=1.0)
    n_episodes = 12
    result = build_dataset(system, regime, None, n_episodes, T=50, dt=0.01, seed=6)

    splits = result["meta"]["splits"]
    train_set, val_set, test_set = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()
    assert train_set | val_set | test_set == set(range(n_episodes))
    for name in ("train", "val", "test"):
        assert result[name]["x"].shape[0] == len(splits[name])


def test_obs_consistency():
    system = DampedPendulum()
    regime = OURegime(sigma_stationary=1.0)
    obs = SmoothMixing(2, seed=1)
    result = build_dataset(system, regime, obs, n_episodes=6, T=20, dt=0.01, seed=7)

    for name in ("train", "val", "test"):
        expected = obs(result[name]["z_true"])
        assert np.array_equal(result[name]["x"], expected)


def test_npz_and_meta_round_trip(tmp_path):
    system = DampedPendulum()
    regime = OURegime(sigma_stationary=1.0)
    obs = SmoothMixing(2, seed=1)
    out_dir = tmp_path / "dataset"

    result = build_dataset(
        system, regime, obs, n_episodes=6, T=20, dt=0.01, seed=8, out_dir=str(out_dir)
    )
    for name in ("train", "val", "test"):
        loaded = np.load(out_dir / f"{name}.npz")
        for key in ("x", "u", "z_true"):
            assert np.array_equal(loaded[key], result[name][key])

    with open(out_dir / "meta.json") as f:
        meta = json.load(f)
    assert meta["dt"] == 0.01
    assert meta["T"] == 20
    assert meta["seed"] == 8
    assert meta["obs"][0]["lipschitz_bound"] == obs.lipschitz_bound()
    all_idx = set(meta["splits"]["train"]) | set(meta["splits"]["val"]) | set(meta["splits"]["test"])
    assert all_idx == set(range(6))
