"""Assemble train/val/test datasets from a system, a data regime, and an obs map."""

from __future__ import annotations

import hashlib
import json
import os
import types
from functools import partial

import numpy as np

from sld.systems.base import DynamicalSystem


def _spec_params(obj) -> dict:
    """Every generation-affecting instance attribute of ``obj``, JSON-ready.

    The meta block is the dataset's identity document (the Trainer hashes it
    into the run identity), so it must distinguish EVERY parameter that can
    change the generated arrays -- scalars, but also tuples/lists (e.g.
    ``PartialObservation.indices``) and small ndarrays (e.g.
    ``ActionConditionedRegime.K_gain``); serializing only scalars let
    distinct specs collide (adversarial review round 3). Large arrays are
    recorded by content hash instead of value.
    """
    # explicit override hook: an object whose behavior depends on non-public
    # state MUST provide its own complete canonical spec (round 5) -- the
    # vars() harvest below only sees public attributes, and that visibility
    # IS the contract for objects without the hook
    if hasattr(obj, "spec_params"):
        return dict(obj.spec_params())

    def _scalar(x):
        # types preserved exactly: int -> int (a float cast would collide
        # 2**53 with 2**53 + 1, round 5), float -> float, bool/str as-is
        if isinstance(x, (bool, str)):
            return x
        if isinstance(x, (int, np.integer)):
            return int(x)
        return float(x)

    params = {}
    for key, value in sorted(vars(obj).items()):
        if key.startswith("_"):
            continue
        if value is None or isinstance(value, (bool, str)):
            params[key] = value
        elif isinstance(value, (int, np.integer)):
            params[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            params[key] = float(value)
        elif isinstance(value, (tuple, list)) and all(
            isinstance(x, (bool, str, int, float, np.integer, np.floating)) for x in value
        ):
            params[key] = [_scalar(x) for x in value]
        elif isinstance(value, np.ndarray):
            if value.size <= 256:
                params[key] = {"dtype": str(value.dtype), "values": value.tolist()}
            else:
                # dtype and shape are part of the identity: identical bytes
                # under a different view are a different array (round 5)
                params[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha1": hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest(),
                }
        else:
            # STRICT: an attribute the meta cannot represent would make two
            # distinct specs collide silently (round 4) -- refuse instead
            raise ValueError(
                f"{type(obj).__name__}.{key} ({type(value).__name__}) is not "
                "meta-serializable: dataset spec objects may only carry "
                "None/bool/str/int/float, flat tuples/lists of scalars, or "
                "numpy arrays as public attributes (or provide spec_params())"
            )
    return params


def _as_callable_list(obs) -> list:
    if obs is None:
        return []
    if isinstance(obs, (list, tuple)):
        return list(obs)
    return [obs]


def _apply_obs(obs, z_true: np.ndarray) -> np.ndarray:
    x = z_true
    for step in _as_callable_list(obs):
        x = step(x)
    return x


def _describe_obs(obs) -> list:
    description = []
    for step in _as_callable_list(obs):
        # class name alone is NOT an identity: two PartialObservation
        # instances with different indices generate different x -- record
        # the full generation-affecting parameters (round 3). Plain
        # functions/lambdas/partials carry NO canonical spec at all (every
        # lambda serializes identically), so they are refused outright
        # (round 4): obs steps must be spec-carrying class instances.
        if isinstance(step, (types.FunctionType, types.BuiltinFunctionType, partial)):
            raise ValueError(
                "obs steps must be spec-carrying class instances (e.g. "
                "PartialObservation, SmoothMixing); plain functions/lambdas/"
                "partials have no canonical identity for the dataset meta"
            )
        entry = {
            # module-qualified: same-named custom classes must not collide
            "class": f"{type(step).__module__}.{type(step).__qualname__}",
            "params": _spec_params(step),
        }
        if hasattr(step, "lipschitz_bound"):
            entry["lipschitz_bound"] = step.lipschitz_bound()
        description.append(entry)
    return description


def build_dataset(
    system: DynamicalSystem,
    regime,
    obs,
    n_episodes: int,
    T: int,
    dt: float,
    seed: int,
    out_dir: str | None = None,
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Roll out ``n_episodes`` under ``regime``, apply ``obs``, split, optionally write to disk."""
    # the full generation spec is validated/described BEFORE any rollout: a
    # rejected obs (lambda, opaque state) must fail fast, not after episodes
    # were integrated or side effects ran (adversarial review round 5)
    system_spec = {
        "class": f"{type(system).__module__}.{type(system).__qualname__}",
        "params": _spec_params(system),
    }
    regime_spec = {
        "class": f"{type(regime).__module__}.{type(regime).__qualname__}",
        "params": _spec_params(regime),
    }
    obs_spec = _describe_obs(obs)

    rng = np.random.default_rng(seed)

    z_true = np.empty((n_episodes, T + 1, system.state_dim), dtype=np.float64)
    u = np.empty((n_episodes, T, system.action_dim), dtype=np.float64)
    for ep in range(n_episodes):
        x0, u_seq = regime.sample_episode(system, T, dt, rng)
        z_true[ep] = system.integrate(x0, u_seq, dt)
        u[ep] = u_seq

    x = _apply_obs(obs, z_true)

    # episode-level split: shuffle, cut by fraction; rounding remainder goes to train
    episode_order = rng.permutation(n_episodes)
    n_val = round(split_fractions[1] * n_episodes)
    n_test = round(split_fractions[2] * n_episodes)
    n_train = n_episodes - n_val - n_test
    split_idx = {
        "train": np.sort(episode_order[:n_train]),
        "val": np.sort(episode_order[n_train : n_train + n_val]),
        "test": np.sort(episode_order[n_train + n_val :]),
    }

    result = {
        name: {"x": x[idx], "u": u[idx], "z_true": z_true[idx]}
        for name, idx in split_idx.items()
    }
    result["meta"] = {
        "system": system_spec,
        "regime": regime_spec,
        "obs": obs_spec,
        "dt": dt,
        "T": T,
        "n_episodes": n_episodes,
        "seed": seed,
        "splits": {name: idx.tolist() for name, idx in split_idx.items()},
        "split_sizes": {name: int(len(idx)) for name, idx in split_idx.items()},
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        paths = {
            name: os.path.join(out_dir, f"{name}.npz") for name in ("train", "val", "test")
        }
        paths["meta"] = os.path.join(out_dir, "meta.json")
        for name in ("train", "val", "test"):
            np.savez(
                paths[name],
                x=result[name]["x"],
                u=result[name]["u"],
                z_true=result[name]["z_true"],
            )
        result["meta"]["paths"] = paths
        with open(paths["meta"], "w") as f:
            json.dump(result["meta"], f, indent=2)

    return result
