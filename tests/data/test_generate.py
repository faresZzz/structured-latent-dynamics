import json

import numpy as np

from sld.data.generate import build_dataset
from sld.data.mixing import SmoothMixing
from sld.data.regimes import OURegime
from sld.systems.base import DynamicalSystem
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


class _IdentifiableEpisodeSystem(DynamicalSystem):
    """Deterministic stub for identity-tracing build_dataset's output arrays.

    f == 0 keeps every trajectory constant for its entire length; sample_ic
    returns a fresh, strictly increasing integer id per call (encoded in
    x0[0]) via an internal counter. build_dataset calls regime.sample_episode
    -> system.sample_ic exactly once per episode, in the plain loop order
    ``for ep in range(n_episodes)`` -- BEFORE the train/val/test shuffle -- so
    the trajectory recorded for the k-th generated episode is constant and
    equal to k. This lets a test read episode identity directly out of
    x/z_true instead of trusting meta["splits"].
    """

    state_dim = 1
    action_dim = 1
    u_max = 1.0
    partial_indices = (0,)

    def __init__(self):
        self._next_id = 0

    def f(self, x, u):
        return np.zeros(1, dtype=np.float64)

    def sample_ic(self, rng, regime="default"):
        x0 = np.array([self._next_id], dtype=np.float64)
        self._next_id += 1
        return x0


def test_splits_contain_the_declared_episodes():
    """Positive (REGLES 7.1): test_splits_disjoint_and_exhaustive above only
    checks that the INDEX SETS declared in meta["splits"] are disjoint and
    exhaustive, and that the arrays have the right ROW COUNTS -- it never
    checks that row k of result[split]["x"] actually IS episode
    splits[split][k]. A row permutation or an off-by-one between generating
    x/z_true and slicing by the split indices would pass it unnoticed.

    _IdentifiableEpisodeSystem makes every episode content-identifiable (its
    trajectory is constant and equal to its generation-order index), so this
    test reads identity straight out of the array and compares it to the
    declared split indices -- an independent, content-based check meta alone
    cannot provide.
    """
    system = _IdentifiableEpisodeSystem()
    regime = OURegime(sigma_stationary=0.1)
    n_episodes = 12
    result = build_dataset(system, regime, None, n_episodes, T=5, dt=0.01, seed=9)

    splits = result["meta"]["splits"]
    for name in ("train", "val", "test"):
        actual_ids = result[name]["x"][:, 0, 0].astype(int)
        expected_ids = np.array(splits[name], dtype=int)
        print(
            f"[split identity] {name}: actual={actual_ids.tolist()} "
            f"expected={expected_ids.tolist()}"
        )
        assert np.array_equal(actual_ids, expected_ids)  # exact integer ids, no tolerance


def test_splits_content_misalignment_is_detected():
    """Negative (REGLES 7.1, "this is the whole point"): prove the check
    above actually has teeth. Take the REAL build_dataset output from the
    same setup and swap two rows within the train split's real x array (a
    stand-in for a real misalignment bug -- a stray permutation, or an
    off-by-one between indexing z_true/x and the recorded split index list).
    The old test (row counts + disjoint declared index sets) would not
    notice this at all; the identity check above must.
    """
    system = _IdentifiableEpisodeSystem()
    regime = OURegime(sigma_stationary=0.1)
    n_episodes = 12
    result = build_dataset(system, regime, None, n_episodes, T=5, dt=0.01, seed=9)
    splits = result["meta"]["splits"]

    assert len(splits["train"]) >= 2  # need two distinct rows to swap
    train_x = result["train"]["x"].copy()
    train_x[[0, 1]] = train_x[[1, 0]]

    actual_ids = train_x[:, 0, 0].astype(int)
    expected_ids = np.array(splits["train"], dtype=int)
    print(
        f"[split identity corrupted] actual={actual_ids.tolist()} "
        f"expected={expected_ids.tolist()}"
    )
    assert not np.array_equal(actual_ids, expected_ids)


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


def test_spec_params_injectivity_round5():
    # int identity preserved (a float cast collides 2**53 with 2**53+1);
    # large arrays carry dtype+shape, not just bytes; spec_params() hook wins
    import numpy as np

    from sld.data.generate import _spec_params

    class S:
        pass

    a, b = S(), S()
    a.gain = [2**53]
    b.gain = [2**53 + 1]
    pa, pb = _spec_params(a), _spec_params(b)
    print(f"[spec] gains {pa['gain']} vs {pb['gain']}")
    assert pa != pb

    big = np.zeros(300, dtype=np.float64)
    c, d = S(), S()
    c.arr = big
    d.arr = big.view(np.float32).reshape(150, 4)  # same bytes, different view
    pc, pd = _spec_params(c), _spec_params(d)
    assert pc["arr"]["sha1"] == pd["arr"]["sha1"]  # same bytes indeed
    assert pc != pd  # but dtype/shape distinguish them

    class WithHook:
        def __init__(self):
            self._hidden_gain = 3.0

        def spec_params(self):
            return {"hidden_gain": self._hidden_gain}

    assert _spec_params(WithHook()) == {"hidden_gain": 3.0}


def test_rejected_obs_fails_before_any_generation():
    # round 5: spec validation must run BEFORE episodes are integrated
    import pytest

    from sld.data.generate import build_dataset
    from sld.systems.damped_pendulum import DampedPendulum

    class SpyRegime:
        def __init__(self):
            self.calls = 0

        def sample_episode(self, system, T, dt, rng):
            self.calls += 1
            raise AssertionError("generation ran before spec validation")

    spy = SpyRegime()
    with pytest.raises(ValueError, match="canonical identity"):
        build_dataset(
            system=DampedPendulum(), regime=spy, obs=lambda z: z,
            n_episodes=4, T=10, dt=0.02, seed=0,
        )
    print(f"[order] regime calls after rejection = {spy.calls}")
    assert spy.calls == 0
