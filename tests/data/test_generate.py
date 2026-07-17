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


def test_meta_distinguishes_nonscalar_generation_params():
    # adversarial review round 3: the meta block is the dataset's identity
    # document -- distinct generation specs must produce distinct meta, for
    # tuple params (PartialObservation.indices) and array params
    # (ActionConditionedRegime.K_gain) alike.
    import numpy as np

    from sld.data.generate import build_dataset
    from sld.data.mixing import PartialObservation
    from sld.data.regimes import ActionConditionedRegime
    from sld.systems.damped_pendulum import DampedPendulum

    system = DampedPendulum()

    def _strip_paths(meta):
        return {k: v for k, v in meta.items() if k != "paths"}

    common = dict(system=system, n_episodes=4, T=10, dt=0.02, seed=3)
    regime = ActionConditionedRegime(K_gain=np.array([[0.5, 0.0]]))
    m_a = build_dataset(regime=regime, obs=PartialObservation((0,)), **common)["meta"]
    m_b = build_dataset(regime=regime, obs=PartialObservation((1,)), **common)["meta"]
    print(f"[meta] obs params a={m_a['obs'][0]['params']} b={m_b['obs'][0]['params']}")
    assert _strip_paths(m_a) != _strip_paths(m_b)

    regime_c = ActionConditionedRegime(K_gain=np.array([[0.0, 0.5]]))
    m_c = build_dataset(regime=regime_c, obs=PartialObservation((0,)), **common)["meta"]
    print(f"[meta] regime params a={m_a['regime']['params']} c={m_c['regime']['params']}")
    assert _strip_paths(m_a) != _strip_paths(m_c)


def test_obs_without_canonical_spec_rejected():
    # adversarial review round 4: every lambda serializes identically -- a
    # spec-less callable would collide distinct datasets under one identity
    import numpy as np
    import pytest

    from sld.data.generate import build_dataset
    from sld.systems.damped_pendulum import DampedPendulum
    from sld.data.regimes import OURegime

    common = dict(
        system=DampedPendulum(), regime=OURegime(), n_episodes=4, T=10, dt=0.02, seed=3
    )
    with pytest.raises(ValueError, match="canonical identity"):
        build_dataset(obs=lambda z: 2 * z, **common)

    class OpaqueObs:
        def __init__(self):
            self.lookup = {"gain": 2.0}  # dict attr: not meta-serializable

        def __call__(self, z):
            return np.asarray(z) * self.lookup["gain"]

    with pytest.raises(ValueError, match="meta-serializable"):
        build_dataset(obs=OpaqueObs(), **common)


def test_smooth_mixing_meta_roundtrip():
    # round 4: the meta must REGENERATE the mixing, not just fingerprint it
    import numpy as np

    from sld.data.generate import _spec_params
    from sld.data.mixing import SmoothMixing

    mix = SmoothMixing(dim=17, seed=5)
    params = _spec_params(mix)
    rebuilt = SmoothMixing(
        dim=params["dim"],
        seed=params["seed"],
        a_range=tuple(params["a_range"]),
        b_range=tuple(params["b_range"]),
    )
    probe = np.random.default_rng(0).normal(size=(3, 17))
    err = np.abs(mix(probe) - rebuilt(probe)).max()
    print(f"[roundtrip] SmoothMixing rebuild max abs err = {err:.3e}")
    assert err == 0.0
