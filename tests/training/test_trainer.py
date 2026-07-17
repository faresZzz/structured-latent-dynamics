"""Tests for sld.training.trainer.Trainer: config-driven JEPA training, one
code path for both a plain (mlp) and a lifted (koopman_ssm) predictor family.

Wave 3 post-adversarial-review semantics under test: ``fit(data)`` builds the
models FROM the config after seeding (run = config + seed, REGLES section 6),
``results_root`` is excluded from the run identity, and a nonfinite loss
aborts the run instead of writing completed-looking artifacts.

Small/fast by design: tiny dims, epochs<=3, T<=30, n_episodes<=8. Trainer's
``set_seed`` mutates GLOBAL RNG state BY DESIGN (see ``training/seeding.py``)
-- tests may rely on it AFTER calling ``fit()``, but never assume a
particular global RNG state beforehand.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
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
        "family": "mlp",
        "d_x": _D_X,
        "d_z": _D_Z,
        "d_u": _D_U,
        "encoder_hidden": [16],
        "target_params": 2_000,
        "epochs": 3,
        "batch_episodes": 4,
        "lr": 5e-2,
        "h_fr": 5,
        "results_root": str(tmp_path / "results"),
    }
    config.update(overrides)
    return config


def test_trainer_contract_mlp_family(tmp_path):
    data = _small_dataset()

    result = Trainer(_mlp_config(tmp_path)).fit(data)

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


def test_run_identity_is_config_plus_seed(tmp_path):
    # same config + seed -> same run dir AND identical initial weights: the
    # models are built inside fit, after set_seed (adversarial review round 1)
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, epochs=1)
    result_a = Trainer(cfg).fit(data)
    result_b = Trainer(dict(cfg)).fit(data)
    assert result_a["paths"]["ckpt"] == result_b["paths"]["ckpt"]

    enc_a = result_a["model"]["encoder"].state_dict()
    enc_b = result_b["model"]["encoder"].state_dict()
    pred_a = result_a["model"]["predictor"].state_dict()
    pred_b = result_b["model"]["predictor"].state_dict()
    for k in enc_a:
        torch.testing.assert_close(enc_a[k], enc_b[k], atol=0.0, rtol=0.0)
    for k in pred_a:
        torch.testing.assert_close(pred_a[k], pred_b[k], atol=0.0, rtol=0.0)
    print("[trainer] identical config+seed -> bit-identical final weights")

    # train/val artifacts too: the run is regenerable, not merely co-located
    assert result_a["history"]["train_loss"] == result_b["history"]["train_loss"]


def test_hash_determinism_sensitivity_and_results_root_exclusion(tmp_path):
    # run_dir itself is resolved at fit time (it includes the DATA hash,
    # round 2); the config side of the identity is testable pre-fit
    hash_a = Trainer(_mlp_config(tmp_path)).config_hash
    hash_b = Trainer(_mlp_config(tmp_path)).config_hash
    print(f"[trainer] same-config hashes: {hash_a} vs {hash_b}")
    assert hash_a == hash_b

    hash_c = Trainer(_mlp_config(tmp_path, lr=1e-3)).config_hash
    print(f"[trainer] different-lr hash: {hash_c}")
    assert hash_c != hash_a

    # results_root is infrastructure, not identity
    other = Trainer(_mlp_config(tmp_path, results_root=str(tmp_path / "elsewhere")))
    assert other.config_hash == hash_a

    # unresolved until data is known
    assert Trainer(_mlp_config(tmp_path)).run_dir is None


def test_train_loss_decreases(tmp_path):
    data = _small_dataset()
    config = _mlp_config(tmp_path, epochs=3, batch_episodes=6, lr=1e-2)
    result = Trainer(config).fit(data)

    losses = result["history"]["train_loss"]
    print(f"[trainer] train_loss trajectory = {losses}")
    assert losses[-1] < losses[0]


def test_ema_target_excluded_from_optimizer(tmp_path):
    data = _small_dataset()
    result = Trainer(_mlp_config(tmp_path)).fit(data)

    encoder = result["model"]["encoder"]
    predictor = result["model"]["predictor"]
    optimizer_ids = {id(p) for p in list(encoder.parameters()) + list(predictor.parameters())}
    ema = result["model"]["ema_target"]
    print(f"[trainer] EMA target param count = {sum(1 for _ in ema.target.parameters())}")
    for p in ema.target.parameters():
        assert p.requires_grad is False
        assert id(p) not in optimizer_ids


def test_lifted_family_consistency_positive(tmp_path):
    data = _small_dataset(T=15)
    config = _mlp_config(
        tmp_path,
        exp_name="trainer_test_koopman",
        family="koopman_ssm",
        n_pairs=2,
        delta=0.05,
        target_params=3_000,
        lambda_c=0.5,
        h_fr=4,
    )
    result = Trainer(config).fit(data)

    assert set(result.keys()) == {"model", "history", "metrics", "paths"}

    with open(result["paths"]["metrics_csv"]) as f:
        rows = list(csv.DictReader(f))
    consistency_values = [float(row["consistency"]) for row in rows]
    print(f"[trainer] koopman consistency column = {consistency_values}")
    assert any(v > 0.0 for v in consistency_values)


def test_zero_weight_branch_not_evaluated(tmp_path):
    # w_fr=0 must SKIP the FR rollout entirely (0 * NaN would otherwise
    # poison a TF-only run through a diverging rollout)
    data = _small_dataset(n_episodes=6, T=10)
    config = _mlp_config(tmp_path, exp_name="trainer_test_tf_only", w_fr=0.0, epochs=2)
    result = Trainer(config).fit(data)
    assert all(np.isfinite(v) for v in result["history"]["train_loss"])
    print(f"[trainer] TF-only train_loss = {result['history']['train_loss']}")


def test_missing_required_key_raises(tmp_path):
    import pytest

    config = _mlp_config(tmp_path)
    del config["family"]
    with pytest.raises(ValueError):
        Trainer(config)


def test_different_data_different_run_identity(tmp_path):
    # adversarial review round 2: the dataset generation spec is part of the
    # run identity -- same config, different data seed -> different run dir
    cfg = _mlp_config(tmp_path, epochs=1)
    result_a = Trainer(cfg).fit(_small_dataset(seed=0, n_episodes=6, T=10))
    result_b = Trainer(dict(cfg)).fit(_small_dataset(seed=1, n_episodes=6, T=10))
    dir_a = Path(result_a["paths"]["ckpt"]).parent
    dir_b = Path(result_b["paths"]["ckpt"]).parent
    print(f"[trainer] data-seed 0 -> {dir_a.name}, data-seed 1 -> {dir_b.name}")
    assert dir_a != dir_b

    # and data without its meta block is rejected outright
    import pytest

    bad = {k: v for k, v in _small_dataset(n_episodes=6, T=10).items() if k != "meta"}
    with pytest.raises(ValueError):
        Trainer(_mlp_config(tmp_path)).fit(bad)


def test_model_keys_applied_or_rejected(tmp_path):
    import pytest

    # a model-defining key not applicable to the family is REJECTED
    with pytest.raises(ValueError):
        Trainer(_mlp_config(tmp_path, delta=0.05))
    # an applicable key is APPLIED to the constructed model
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(
        tmp_path,
        exp_name="trainer_test_heads",
        family="transformer",
        n_heads=4,
        epochs=1,
        # d_model moves on an n_heads-stride grid: a 2k target is unreachable
        # within 5% at stride 4, so this sub-test budgets at 10k
        target_params=10_000,
    )
    result = Trainer(cfg).fit(data)
    assert result["model"]["predictor"].n_heads == 4
    print("[trainer] transformer n_heads=4 applied from config")


def test_weight_domain_enforced(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        Trainer(_mlp_config(tmp_path, w_fr=-1.0))  # negative != disabled
    with pytest.raises(ValueError):
        Trainer(_mlp_config(tmp_path, w_fr=float("nan")))
    with pytest.raises(ValueError):
        Trainer(_mlp_config(tmp_path, w_tf=0.0, w_fr=0.0))  # nothing to train


def test_disabled_fr_branch_never_called(tmp_path, monkeypatch):
    # adversarial review round 2: prove the skip, don't infer it -- a
    # disabled FR branch that still got called would raise here
    import sld.predictors.base as base_mod

    def _boom(self, h0, u_seq):
        raise AssertionError("FR rollout called despite w_fr=0")

    monkeypatch.setattr(base_mod.PredictorModule, "rollout", _boom)
    data = _small_dataset(n_episodes=6, T=10)
    result = Trainer(_mlp_config(tmp_path, exp_name="trainer_test_skip", w_fr=0.0)).fit(data)
    assert all(np.isfinite(v) for v in result["history"]["train_loss"])
    print("[trainer] w_fr=0: rollout never invoked")


def test_nonfinite_loss_aborts_run(tmp_path, monkeypatch):
    import pytest

    import sld.training.trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan"))
    )
    data = _small_dataset(n_episodes=6, T=10)
    with pytest.raises(RuntimeError, match="nonfinite"):
        Trainer(_mlp_config(tmp_path, exp_name="trainer_test_nan")).fit(data)
    print("[trainer] nonfinite loss aborted the run")


def test_budget_owned_and_selective_only_keys_rejected(tmp_path):
    import pytest

    # capacity knobs are owned by match_budget: naming one in a config would
    # be hashed but ignored -- rejected (adversarial review round 3)
    with pytest.raises(ValueError, match="capacity"):
        Trainer(_mlp_config(tmp_path, family="transformer", n_layers=5))
    with pytest.raises(ValueError, match="capacity"):
        Trainer(_mlp_config(tmp_path, hidden=[32, 32]))
    # selective-only clamp bounds on a fixed koopman regime: fake identity
    with pytest.raises(ValueError, match="selective"):
        Trainer(
            _mlp_config(
                tmp_path, family="koopman_ssm", delta=0.05, regime="constant", delta_min=1e-3
            )
        )


def test_failed_rerun_leaves_no_stale_success_artifacts(tmp_path, monkeypatch):
    # adversarial review round 3: success then forced-nonfinite rerun on the
    # SAME identity must not leave the old ckpt/audit beside new truncated
    # metrics (a mixed directory reads as a completed run)
    import pytest

    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_stale", epochs=1)
    result = Trainer(cfg).fit(data)
    ckpt = Path(result["paths"]["ckpt"])
    audit = Path(result["paths"]["audit"])
    assert ckpt.exists() and audit.exists()

    monkeypatch.setattr(trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="nonfinite"):
        Trainer(dict(cfg)).fit(data)
    print(f"[trainer] after failed rerun: ckpt={ckpt.exists()}, audit={audit.exists()}")
    assert not ckpt.exists() and not audit.exists()


def test_derived_policy_keys_and_bad_epochs_rejected(tmp_path):
    import pytest

    # round 4: d_h is derived, dtype is the declared precision policy --
    # hashing them while ignoring them would fake distinct identities
    with pytest.raises(ValueError, match="derived or policy"):
        Trainer(_mlp_config(tmp_path, d_h=999))
    with pytest.raises(ValueError, match="derived or policy"):
        Trainer(_mlp_config(tmp_path, family="koopman_ssm", delta=0.05, dtype="float64"))
    # loop integers validated BEFORE any artifact: an empty epoch loop would
    # write success-only artifacts then crash on the empty history
    for bad in ({"epochs": 0}, {"batch_episodes": 0}, {"h_fr": 0}, {"epochs": 2.5}):
        with pytest.raises(ValueError):
            Trainer(_mlp_config(tmp_path, **bad))
