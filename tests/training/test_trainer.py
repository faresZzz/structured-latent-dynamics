"""Tests for sld.training.trainer.Trainer: config-driven JEPA training, one
code path for both a plain (mlp) and a lifted (koopman_ssm) predictor family.

Small/fast by design: tiny dims, epochs<=3, T<=30, n_episodes<=8. Trainer's
``set_seed`` mutates GLOBAL RNG state BY DESIGN (see ``training/seeding.py``)
-- tests may rely on it AFTER calling ``fit()``, but never assume a
particular global RNG state beforehand.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.systems.damped_pendulum import DampedPendulum
from sld.training.trainer import Trainer

_D_X, _D_Z, _D_U = 2, 4, 1  # pendulum state dim, tiny latent dim, torque dim


def _small_dataset(n_episodes: int = 8, T: int = 20, seed: int = 0) -> dict:
    system = DampedPendulum()
    return build_dataset(
        system=system, regime=OURegime(), obs=None, n_episodes=n_episodes, T=T, dt=0.02, seed=seed
    )


def _mlp_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "exp_name": "trainer_test_mlp",
        "seed": 0,
        "epochs": 3,
        "batch_episodes": 4,
        "lr": 5e-2,
        "h_fr": 5,
        "results_root": str(tmp_path / "results"),
    }
    config.update(overrides)
    return config


def _new_mlp(hidden: tuple[int, ...] = (16,)) -> tuple[FrameMLPEncoder, MLPPredictor]:
    return FrameMLPEncoder(_D_X, _D_Z, hidden=hidden), MLPPredictor(_D_Z, _D_U, hidden=hidden)


def test_trainer_contract_mlp_family(tmp_path):
    data = _small_dataset()
    encoder, predictor = _new_mlp()

    result = Trainer(_mlp_config(tmp_path)).fit(encoder, predictor, data)

    print(f"[trainer] top-level keys = {sorted(result.keys())}")
    assert set(result.keys()) == {"model", "history", "metrics", "paths"}

    history = result["history"]
    stable_keys = {"epoch", "train_loss", "val_loss", "jac_margin", "r_ctrl", "lip_pi", "vpt"}
    print(f"[trainer] history keys = {sorted(history.keys())}")
    assert set(history.keys()) == stable_keys
    n_epochs = len(history["epoch"])
    for key in ("jac_margin", "r_ctrl", "lip_pi", "vpt"):
        assert history[key] == [None] * n_epochs  # explicit None, never omitted

    metrics = result["metrics"]
    for key, value in metrics.items():
        if key == "config_hash":
            assert isinstance(value, str)
        else:
            assert np.isfinite(value), f"{key}={value} is not finite"

    paths = result["paths"]
    assert set(paths.keys()) == {"config", "metrics_csv", "audit", "ckpt", "figs"}
    for key, p in paths.items():
        print(f"[trainer] artifact {key} -> {p}")
        assert Path(p).exists()


def test_hash_determinism_and_lr_sensitivity(tmp_path):
    run_dir_a = Trainer(_mlp_config(tmp_path)).run_dir
    run_dir_b = Trainer(_mlp_config(tmp_path)).run_dir
    print(f"[trainer] same-config run dirs: {run_dir_a} vs {run_dir_b}")
    assert run_dir_a == run_dir_b

    run_dir_c = Trainer(_mlp_config(tmp_path, lr=1e-3)).run_dir
    print(f"[trainer] different-lr run dir: {run_dir_c}")
    assert run_dir_c != run_dir_a


def test_train_loss_decreases(tmp_path):
    data = _small_dataset()
    encoder, predictor = _new_mlp()

    config = _mlp_config(tmp_path, epochs=3, batch_episodes=6, lr=1e-2)
    result = Trainer(config).fit(encoder, predictor, data)

    losses = result["history"]["train_loss"]
    print(f"[trainer] train_loss trajectory = {losses}")
    assert losses[-1] < losses[0]


def test_ema_target_excluded_from_optimizer(tmp_path):
    data = _small_dataset()
    encoder, predictor = _new_mlp()

    result = Trainer(_mlp_config(tmp_path)).fit(encoder, predictor, data)

    optimizer_ids = {id(p) for p in list(encoder.parameters()) + list(predictor.parameters())}
    ema = result["model"]["ema_target"]
    print(f"[trainer] EMA target param count = {sum(1 for _ in ema.target.parameters())}")
    for p in ema.target.parameters():
        assert p.requires_grad is False
        assert id(p) not in optimizer_ids


def test_lifted_family_consistency_positive(tmp_path):
    data = _small_dataset(T=15)
    encoder = FrameMLPEncoder(_D_X, _D_Z, hidden=(16,))
    predictor = KoopmanSSMPredictor(d_z=_D_Z, n_pairs=2, d_u=_D_U, delta=0.05)

    config = _mlp_config(tmp_path, exp_name="trainer_test_koopman", lambda_c=0.5, h_fr=4)
    result = Trainer(config).fit(encoder, predictor, data)

    assert set(result.keys()) == {"model", "history", "metrics", "paths"}

    with open(result["paths"]["metrics_csv"]) as f:
        rows = list(csv.DictReader(f))
    consistency_values = [float(row["consistency"]) for row in rows]
    print(f"[trainer] koopman consistency column = {consistency_values}")
    assert any(v > 0.0 for v in consistency_values)
