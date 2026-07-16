"""Assemble train/val/test datasets from a system, a data regime, and an obs map."""

from __future__ import annotations

import json
import os

import numpy as np

from sld.systems.base import DynamicalSystem


def _scalar_params(obj) -> dict:
    """Instance attributes of ``obj`` that are plain ints/floats, JSON-ready."""
    params = {}
    for key, value in vars(obj).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, np.integer)):
            params[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            params[key] = float(value)
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
        entry = {"class": type(step).__name__}
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
        "system": {"class": type(system).__name__, "params": _scalar_params(system)},
        "regime": {"class": type(regime).__name__, "params": _scalar_params(regime)},
        "obs": _describe_obs(obs),
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
