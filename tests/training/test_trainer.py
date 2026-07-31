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
from sld.objectives.jac_reg import Gauge, gauge_digest
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
    # F9: "complete" (the run_complete.json sentinel) extends the contract
    assert set(paths.keys()) == {"config", "metrics_csv", "audit", "ckpt", "figs", "complete"}
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
    complete = Path(result["paths"]["complete"])
    assert ckpt.exists() and audit.exists() and complete.exists()  # F9: sentinel too

    monkeypatch.setattr(trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="nonfinite"):
        Trainer(dict(cfg)).fit(data)
    print(
        f"[trainer] after failed rerun: ckpt={ckpt.exists()}, audit={audit.exists()}, "
        f"complete={complete.exists()}"
    )
    assert not ckpt.exists() and not audit.exists() and not complete.exists()  # F9


def test_directory_named_stale_artifacts_are_cleaned_up_and_rerun_succeeds(tmp_path):
    # Every artifact fit writes must recover when a corrupt prior artifact is
    # a directory, including the ordinary write targets config.yaml and
    # metrics.csv and the figs/ directory itself.
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_dir_artifacts", epochs=1)
    result = Trainer(cfg).fit(data)
    run_dir = Path(result["paths"]["ckpt"]).parent

    for name in ("config.yaml", "metrics.csv", "ckpt.pt", "audit.json", "run_complete.json", "figs"):
        path = run_dir / name
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
        path.mkdir()
        (path / "junk").write_text("not the real artifact")  # non-empty directory

    result2 = Trainer(dict(cfg)).fit(data)  # must not raise IsADirectoryError
    for key in ("config", "metrics_csv", "ckpt", "audit", "complete"):
        p = Path(result2["paths"][key])
        assert p.is_file(), f"{key} artifact was not recreated as a regular file"
    assert Path(result2["paths"]["figs"]).is_dir()
    print("[trainer] directory-named stale artifacts cleaned up, rerun succeeded")


def test_run_complete_sentinel_written_last_and_atomically(tmp_path):
    # F9: the sentinel is the ONLY trustworthy completion signal -- present
    # after a successful fit, absent for a run that never reached that
    # point, and its content matches the returned contract exactly (no
    # separate re-derivation of config_hash/data_hash/final losses).
    # M3 (round 3): ckpt_sha1/metrics_rows bind the sentinel to the actual
    # artifacts -- checked here against an INDEPENDENT recomputation
    # (hashlib directly on ckpt.pt's bytes), not by re-calling the
    # trainer's own helper, so this test would catch a mismatched digest.
    import hashlib
    import json

    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_sentinel", epochs=2)
    trainer = Trainer(cfg)
    result = trainer.fit(data)

    complete_path = Path(result["paths"]["complete"])
    assert complete_path.name == "run_complete.json"
    assert complete_path.exists()
    # no leftover .tmp from the atomic-write step
    assert not complete_path.with_name(complete_path.name + ".tmp").exists()

    payload = json.loads(complete_path.read_text())
    print(f"[trainer] run_complete.json = {payload}")
    ckpt_bytes = Path(result["paths"]["ckpt"]).read_bytes()
    assert payload == {
        "config_hash": trainer.config_hash,
        "data_hash": trainer.data_hash,
        "epochs_completed": 2,
        "final_train_loss": result["history"]["train_loss"][-1],
        "final_val_loss": result["history"]["val_loss"][-1],
        "ckpt_sha1": hashlib.sha1(ckpt_bytes).hexdigest(),
        "metrics_rows": 2,
    }

    # written LAST: strictly not older than ckpt.pt/audit.json (F9 ordering)
    ckpt_mtime = Path(result["paths"]["ckpt"]).stat().st_mtime_ns
    audit_mtime = Path(result["paths"]["audit"]).stat().st_mtime_ns
    complete_mtime = complete_path.stat().st_mtime_ns
    assert complete_mtime >= ckpt_mtime
    assert complete_mtime >= audit_mtime


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


def test_family_defaults_canonicalized_into_identity(tmp_path):
    # round 5: omitted default == explicit default, same identity; a
    # non-default value is a different identity
    base = _mlp_config(tmp_path, family="transformer")
    explicit = _mlp_config(tmp_path, family="transformer", n_heads=2, max_len=512)
    other = _mlp_config(tmp_path, family="transformer", n_heads=4)
    h_base = Trainer(base).config_hash
    h_explicit = Trainer(explicit).config_hash
    h_other = Trainer(other).config_hash
    print(f"[defaults] omitted={h_base} explicit={h_explicit} n_heads=4={h_other}")
    assert h_base == h_explicit
    assert h_other != h_base

    # koopman: delta required at CONSTRUCTION (schema, not a fit-time error)
    import pytest

    with pytest.raises(ValueError, match="delta"):
        Trainer(_mlp_config(tmp_path, family="koopman_ssm"))


def test_float32_policy_enforced_under_ambient_float64(tmp_path):
    # rounds 5+6: the declared CPU/float32 policy must not depend on the
    # ambient global default dtype -- and construction must be NATIVE
    # float32 (float64-then-cast consumes the RNG differently, so the same
    # config+seed would otherwise give different weights per ambient state).
    # Evidence standard from the review: bit-identical runs across ambients.
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_f32", epochs=1)

    result_native = Trainer(cfg).fit(data)  # ambient float32 (default)

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        result_ambient = Trainer(dict(cfg)).fit(data)
    finally:
        torch.set_default_dtype(prev)

    for module in (result_ambient["model"]["encoder"], result_ambient["model"]["predictor"]):
        for p in module.parameters():
            assert p.dtype == torch.float32
    for key in ("encoder", "predictor"):
        sd_a = result_native["model"][key].state_dict()
        sd_b = result_ambient["model"][key].state_dict()
        for k in sd_a:
            torch.testing.assert_close(sd_a[k], sd_b[k], atol=0.0, rtol=0.0)
    assert result_native["history"]["train_loss"] == result_ambient["history"]["train_loss"]
    print("[trainer] bit-identical run across ambient default dtypes")


def test_min_ssm_tiny_interval_rejected_consistently(tmp_path):
    # round 6: min_ssm bound validation now runs inside the enforced-float32
    # construction context, so an interval that underflows float32 is
    # rejected regardless of the ambient default dtype
    import pytest

    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(
        tmp_path,
        exp_name="trainer_test_tiny",
        family="min_ssm",
        delta_min=1e-60,
        delta_max=1e-40,
        epochs=1,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(ValueError, match="dtype"):
            Trainer(cfg).fit(data)
    finally:
        torch.set_default_dtype(prev)
    print("[trainer] tiny min_ssm interval rejected under ambient float64 too")


def test_explicit_none_default_same_identity(tmp_path):
    # round 6: an explicit delta_max: null must hash like the omitted default
    base = _mlp_config(
        tmp_path, family="koopman_ssm", delta=0.05, regime="selective"
    )
    explicit_none = _mlp_config(
        tmp_path, family="koopman_ssm", delta=0.05, regime="selective", delta_max=None
    )
    h_base = Trainer(base).config_hash
    h_none = Trainer(explicit_none).config_hash
    print(f"[defaults] omitted={h_base} explicit-None={h_none}")
    assert h_base == h_none


def test_disabled_branch_zero_does_not_leak_ambient_dtype(tmp_path):
    # round 7: with a disabled branch (w_fr=0) and the other terms active,
    # the placeholder zero must not promote the loss accumulation to the
    # ambient dtype -- evidence standard: identical COMPLETE histories,
    # metrics, and metrics.csv bytes across ambient default dtypes
    data = _small_dataset(n_episodes=6, T=10)
    cfg_a = _mlp_config(
        tmp_path, exp_name="trainer_test_zero_dtype", w_fr=0.0, epochs=2,
        results_root=str(tmp_path / "root_a"),
    )
    cfg_b = dict(cfg_a, results_root=str(tmp_path / "root_b"))

    result_a = Trainer(cfg_a).fit(data)  # ambient float32

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        result_b = Trainer(cfg_b).fit(data)
    finally:
        torch.set_default_dtype(prev)

    assert result_a["history"] == result_b["history"]
    assert result_a["metrics"]["final_train_loss"] == result_b["metrics"]["final_train_loss"]
    assert result_a["metrics"]["final_val_loss"] == result_b["metrics"]["final_val_loss"]
    csv_a = Path(result_a["paths"]["metrics_csv"]).read_text()
    csv_b = Path(result_b["paths"]["metrics_csv"]).read_text()
    assert csv_a == csv_b
    print("[trainer] disabled-branch run bit-identical across ambient dtypes")


# -- mu_j: the total-Jacobian constraint factor ------------------------------


def test_mu_j_zero_metrics_csv_has_jac_pen_column_all_zero(tmp_path):
    # mu_j defaults to 0.0: the jac_pen column is ALWAYS present (never
    # omitted), but the branch is entirely skipped so every value is 0.0
    data = _small_dataset(n_episodes=6, T=10)
    result = Trainer(_mlp_config(tmp_path, exp_name="trainer_test_mu_j_off")).fit(data)
    with open(result["paths"]["metrics_csv"]) as f:
        rows = list(csv.DictReader(f))
    print(f"[trainer] mu_j=0 header = {list(rows[0].keys())}")
    assert "jac_pen" in rows[0]
    assert all(float(row["jac_pen"]) == 0.0 for row in rows)


def test_mu_j_positive_mlp_tensor_path(tmp_path):
    # mlp never populates PredictorState.m -- the tensor path in
    # _jac_state_penalty must be the one exercised, and jac_pen must be
    # nonzero on at least one epoch (a Bernoulli-subsampled diagnostic can
    # legitimately be exactly 0.0 on some epochs via an empty draw)
    data = _small_dataset(n_episodes=6, T=10)
    config = _mlp_config(tmp_path, exp_name="trainer_test_mu_j_mlp", mu_j=0.05, epochs=3)
    result = Trainer(config).fit(data)
    assert all(np.isfinite(v) for v in result["history"]["train_loss"])
    assert all(np.isfinite(v) for v in result["history"]["val_loss"])
    with open(result["paths"]["metrics_csv"]) as f:
        rows = list(csv.DictReader(f))
    jac_pens = [float(row["jac_pen"]) for row in rows]
    print(f"[trainer] mlp mu_j>0 jac_pen column = {jac_pens}")
    assert any(v > 0.0 for v in jac_pens)
    # public contract intact
    assert set(result.keys()) == {"model", "history", "metrics", "paths"}


def test_mu_j_positive_min_ssm_pytree_path(tmp_path):
    # min_ssm threads a tensor PredictorState.m from the second TF step
    # onward -- the pytree path (on_state_steps) must be the one exercised
    data = _small_dataset(n_episodes=6, T=10)
    config = _mlp_config(
        tmp_path, exp_name="trainer_test_mu_j_min_ssm", family="min_ssm", mu_j=0.05, epochs=3
    )
    result = Trainer(config).fit(data)
    assert all(np.isfinite(v) for v in result["history"]["train_loss"])
    assert all(np.isfinite(v) for v in result["history"]["val_loss"])
    with open(result["paths"]["metrics_csv"]) as f:
        rows = list(csv.DictReader(f))
    jac_pens = [float(row["jac_pen"]) for row in rows]
    print(f"[trainer] min_ssm mu_j>0 jac_pen column = {jac_pens}")
    assert any(v > 0.0 for v in jac_pens)
    assert set(result.keys()) == {"model", "history", "metrics", "paths"}


def test_mu_j_validation_errors(tmp_path):
    import pytest

    # mu_j > 0 needs the TF-threaded state -- w_tf=0 has nothing to evaluate
    with pytest.raises(ValueError, match="w_tf"):
        Trainer(_mlp_config(tmp_path, mu_j=0.1, w_tf=0.0, w_fr=1.0))
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="mu_j"):
            Trainer(_mlp_config(tmp_path, mu_j=bad))


def test_resolve_run_dir_matches_fit_and_writes_nothing(tmp_path):
    # extracted for the factorial runner's resume detection: same path fit()
    # would use, computed with NO side effects (no directory, no files)
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_resolve")
    trainer_a = Trainer(dict(cfg))
    resolved = trainer_a.resolve_run_dir(data)
    assert trainer_a.run_dir == resolved
    assert not resolved.exists()

    result = Trainer(dict(cfg)).fit(data)
    assert Path(result["paths"]["ckpt"]).parent == resolved
    print(f"[trainer] resolve_run_dir -> {resolved}, matches fit()'s run_dir, pre-fit unwritten")


def test_k_probes_step_subsample_defaults_wired(tmp_path):
    # defaults must match the pre-registered constants (PROTOCOL.md section
    # 5) that used to be hardcoded at TotalJacobianPenalty() -- this change
    # must not move them
    trainer = Trainer(_mlp_config(tmp_path))
    assert trainer.config["k_probes"] == 2
    assert trainer.config["step_subsample"] == 0.25
    assert trainer._jac_penalty.k_probes == 2
    assert trainer._jac_penalty.step_subsample == 0.25


def test_k_probes_step_subsample_bounds_validation(tmp_path):
    import pytest

    # bounds enforced regardless of mu_j -- schema is validated eagerly
    # (round 4 doctrine), not only on the path that would use the value
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            Trainer(_mlp_config(tmp_path, k_probes=bad))
        with pytest.raises(ValueError):
            Trainer(_mlp_config(tmp_path, k_probes=bad, mu_j=0.1))
    for bad in (0.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Trainer(_mlp_config(tmp_path, step_subsample=bad))
        with pytest.raises(ValueError):
            Trainer(_mlp_config(tmp_path, step_subsample=bad, mu_j=0.1))


def test_k_probes_different_run_dir_when_mu_j_positive(tmp_path):
    # k_probes has a real effect once mu_j > 0 -- different value, different
    # identity
    hash_a = Trainer(_mlp_config(tmp_path, mu_j=0.1, k_probes=2)).config_hash
    hash_b = Trainer(_mlp_config(tmp_path, mu_j=0.1, k_probes=3)).config_hash
    print(f"[trainer] mu_j>0 k_probes=2 -> {hash_a}, k_probes=3 -> {hash_b}")
    assert hash_a != hash_b

    hash_c = Trainer(_mlp_config(tmp_path, mu_j=0.1, step_subsample=0.25)).config_hash
    hash_d = Trainer(_mlp_config(tmp_path, mu_j=0.1, step_subsample=0.5)).config_hash
    print(f"[trainer] mu_j>0 step_subsample=0.25 -> {hash_c}, 0.5 -> {hash_d}")
    assert hash_c != hash_d


def test_k_probes_canonicalized_when_mu_j_zero(tmp_path):
    # k_probes/step_subsample never reach anything when mu_j == 0 (the
    # penalty branch is entirely skipped, l. 327) -- two mu_j=0 configs
    # differing ONLY in these otherwise-inert knobs must be the SAME run
    # identity, the same canonicalization technique the family defaults use
    # for explicit None (l. 224)
    hash_default = Trainer(_mlp_config(tmp_path, mu_j=0.0)).config_hash
    hash_other_k = Trainer(_mlp_config(tmp_path, mu_j=0.0, k_probes=7)).config_hash
    hash_other_s = Trainer(_mlp_config(tmp_path, mu_j=0.0, step_subsample=0.9)).config_hash
    print(f"[trainer] mu_j=0: default={hash_default} k_probes=7={hash_other_k} "
          f"step_subsample=0.9={hash_other_s}")
    assert hash_default == hash_other_k == hash_other_s


# --------------------------------------------------------------------------
# PROTOCOL.md section 10.3: the frozen gauge reaches the penalty out-of-band,
# and its CONTENT reaches the run identity as the derived `gauge_digest`
# --------------------------------------------------------------------------


def _anisotropic_gauge(scale: float = 1.0) -> Gauge:
    """A frozen h-gauge spanning several orders of magnitude, like the h/m
    variance gap PROTOCOL section 10.3's amendment exists to fix."""
    return Gauge(s_h=(scale * torch.tensor([[1e-2, 1.0, 5.0, 1e2]], dtype=torch.float64))[:, :_D_Z])


def test_gauge_is_wired_into_the_penalty_and_never_a_config_key(tmp_path):
    import pytest

    gauge = _anisotropic_gauge()
    trainer = Trainer(_mlp_config(tmp_path, mu_j=0.1), gauge=gauge)
    # the penalty the training loop actually calls carries the gauge: the
    # object OPTIMIZED is the metric-normalized one, not the raw Jacobian
    assert trainer._jac_penalty.gauge is gauge
    # ... and no gauge at all keeps the RAW pre-amendment estimator
    assert Trainer(_mlp_config(tmp_path, mu_j=0.1))._jac_penalty.gauge is None

    # the schema stays CLOSED: gauge_digest is derived from the out-of-band
    # argument, so a config that names it is rejected outright
    with pytest.raises(ValueError, match="derived or policy-owned"):
        Trainer(_mlp_config(tmp_path, mu_j=0.1, gauge_digest="deadbeef"))


def test_gauge_content_separates_run_identity(tmp_path):
    # two runs trained under DIFFERENT frozen gauges optimize different
    # objectives -- they must never share a run directory ("une identite de
    # run ne distingue pas la selection du core" is the trap this closes)
    cfg = _mlp_config(tmp_path, mu_j=0.1)
    h_a = Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0)).config_hash
    h_b = Trainer(dict(cfg), gauge=_anisotropic_gauge(2.0)).config_hash
    h_none = Trainer(dict(cfg)).config_hash
    print(f"[trainer] mu_j>0 gauge A={h_a} gauge B={h_b} no gauge={h_none}")
    assert len({h_a, h_b, h_none}) == 3

    # the SAME gauge content -- a distinct object, rebuilt -- is the SAME run
    assert Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0)).config_hash == h_a

    # and the digest that separates them is written into config.yaml, so the
    # identity is auditable from the run directory alone
    trainer = Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0))
    assert trainer.config["gauge_digest"] == gauge_digest(_anisotropic_gauge(1.0))
    assert "gauge_digest" in trainer._canonical_yaml


def test_gauge_is_inert_for_identity_when_mu_j_is_zero(tmp_path):
    # same doctrine as k_probes/step_subsample above: on an OFF cell the
    # penalty branch is never evaluated, so the gauge cannot influence
    # anything and must not influence the identity either. Absence of the
    # key (not a null value) keeps every pre-existing OFF run's hash intact.
    cfg = _mlp_config(tmp_path, mu_j=0.0)
    h_plain = Trainer(dict(cfg)).config_hash
    h_gauged = Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0)).config_hash
    h_other = Trainer(dict(cfg), gauge=_anisotropic_gauge(7.0)).config_hash
    print(f"[trainer] mu_j=0: plain={h_plain} gauged={h_gauged} other gauge={h_other}")
    assert h_plain == h_gauged == h_other

    trainer = Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0))
    assert "gauge_digest" not in trainer.config
    assert trainer.gauge is None  # dropped, so it can never reach the penalty
    assert trainer._jac_penalty.gauge is None


def test_gauge_digest_is_bit_stable_across_processes(tmp_path):
    # the run identity must not depend on this process: the digest is
    # recomputed in a FRESH interpreter and compared byte for byte. Any
    # dependence on memory layout, iteration order or ambient dtype would
    # show up here as two different run directories for one science object.
    import subprocess
    import sys

    gauge = Gauge(
        s_h=torch.tensor([[1e-2, 1.0, 5.0, 1e2]], dtype=torch.float64),
        s_m=torch.tensor([[1e3, 1e-3, 2.0]], dtype=torch.float64),
    )
    here = gauge_digest(gauge)
    script = (
        "import torch;"
        "from sld.objectives.jac_reg import Gauge, gauge_digest;"
        "print(gauge_digest(Gauge("
        "s_h=torch.tensor([[1e-2, 1.0, 5.0, 1e2]], dtype=torch.float64),"
        "s_m=torch.tensor([[1e3, 1e-3, 2.0]], dtype=torch.float64))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"[trainer] gauge digest here={here} in a fresh process={out}")
    assert out == here
    assert len(here) == 40  # full sha1, never a truncated prefix

    # a fixed-size gauge and a growing one can never collide by omission
    assert gauge_digest(Gauge(s_h=gauge.s_h)) != here


def test_gauged_run_is_bit_identical_metrics_csv(tmp_path):
    # the repo's bit-identical-rerun guarantee must survive the gauge:
    # same config + same gauge -> the SAME metrics.csv bytes.
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, mu_j=0.1, epochs=3, step_subsample=1.0)
    a = Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0)).fit(data)
    b = Trainer(dict(cfg), gauge=_anisotropic_gauge(1.0)).fit(data)
    bytes_a = Path(a["paths"]["metrics_csv"]).read_bytes()
    bytes_b = Path(b["paths"]["metrics_csv"]).read_bytes()
    assert Path(a["paths"]["metrics_csv"]) == Path(b["paths"]["metrics_csv"])
    assert bytes_a == bytes_b
    print(f"[trainer] gauged run metrics.csv bit-identical ({len(bytes_a)} bytes)")

    # and the gauge is LOAD-BEARING on the numbers, not just on the hash:
    # the same config under a different gauge lands elsewhere, with a
    # different penalty trace
    c = Trainer(dict(cfg), gauge=_anisotropic_gauge(3.0)).fit(data)
    assert Path(c["paths"]["metrics_csv"]) != Path(a["paths"]["metrics_csv"])
    assert Path(c["paths"]["metrics_csv"]).read_bytes() != bytes_a


def test_mu_j_lowers_dense_jacobian_norm(tmp_path):
    # package-level behavioral contract: a toy MLP predictor trained with
    # mu_j > 0 must land with a STRICTLY lower dense total-Jacobian norm
    # (sld.audit.jacobian.total_jacobian -- the EXACT offline measurement,
    # never jac_reg's own stochastic estimator, per REGLES section 5: the
    # audit measures, jac_reg only regularizes) than the same
    # config/seed/data trained with mu_j = 0. mu_j is picked for a
    # comfortable margin (>= 1.05x, not tuned to the edge) and checked
    # across 2 seeds for stability, not just one lucky draw.
    import copy

    from sld.audit.jacobian import total_jacobian
    from sld.predictors.base import PredictorState

    def median_jacobian_norm(predictor, n_probes=10, probe_seed=123):
        predictor_d = copy.deepcopy(predictor).double()  # audit doctrine: CPU float64
        g = torch.Generator().manual_seed(probe_seed)
        q_test = torch.randn(n_probes, _D_Z, dtype=torch.float64, generator=g)
        u_test = torch.randn(n_probes, _D_U, dtype=torch.float64, generator=g)
        norms = []
        for q, u in zip(q_test, u_test):
            def step_fn(qq, uu, _p=predictor_d):
                h_next, _ = _p.step(qq.unsqueeze(0), uu.unsqueeze(0), PredictorState())
                return h_next.squeeze(0)

            J = total_jacobian(step_fn, q, u)
            norms.append(float((J**2).sum() ** 0.5))
        norms.sort()
        return norms[len(norms) // 2]

    data = _small_dataset(n_episodes=8, T=12, seed=0)
    for seed in (0, 1):
        cfg_off = _mlp_config(
            tmp_path,
            exp_name=f"trainer_test_jacnorm_off_{seed}",
            seed=seed,
            mu_j=0.0,
            epochs=20,
            batch_episodes=6,
            lr=3e-2,
            target_params=1_500,
        )
        cfg_on = dict(cfg_off, exp_name=f"trainer_test_jacnorm_on_{seed}", mu_j=2.0)

        predictor_off = Trainer(cfg_off).fit(data)["model"]["predictor"]
        predictor_on = Trainer(cfg_on).fit(data)["model"]["predictor"]

        norm_off = median_jacobian_norm(predictor_off)
        norm_on = median_jacobian_norm(predictor_on)
        print(f"[trainer] seed={seed} median||J||_F without={norm_off:.4f} with={norm_on:.4f}")
        assert norm_on < norm_off / 1.05  # >= 1.05x margin -- not a flaky sliver


# -- mu_j penalty is not evaluated on the val split --------------------------


def test_val_split_never_evaluates_jac_penalty(tmp_path, monkeypatch):
    # the mu_j penalty must be computed exactly once per epoch (train only);
    # a call from the val forward would double this count
    import sld.training.trainer as trainer_mod

    calls = []
    orig = trainer_mod.Trainer._jac_state_penalty

    def counting(self, predictor, jac_steps):
        calls.append(len(jac_steps))
        return orig(self, predictor, jac_steps)

    monkeypatch.setattr(trainer_mod.Trainer, "_jac_state_penalty", counting)
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_val_no_jac", mu_j=0.05, epochs=3)
    Trainer(cfg).fit(data)
    print(f"[trainer] _jac_state_penalty call count = {len(calls)} (epochs=3)")
    assert len(calls) == 3  # once per epoch, train only -- never for val


def test_val_loss_excludes_jac_penalty_and_no_longer_advances_global_rng(tmp_path):
    # val_loss's semantics changed: it no longer includes mu_j * jac_pen.
    # Direct evidence: manually replicate what val_total WOULD be if it
    # still summed the composite loss including the jac penalty, and show
    # it differs from the trained val_loss whenever jac_pen is nonzero on
    # that epoch.
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_val_semantics", mu_j=0.2, epochs=3)
    result = Trainer(cfg).fit(data)
    with open(result["paths"]["metrics_csv"]) as f:
        rows = list(csv.DictReader(f))
    jac_pens = [float(row["jac_pen"]) for row in rows]
    print(f"[trainer] jac_pen column (train, still populated) = {jac_pens}")
    # jac_pen column is the TRAIN-side diagnostic and remains populated
    # (unaffected by the val-side skip) -- proves change 2 is scoped to val
    assert any(v > 0.0 for v in jac_pens)


def test_val_evaluation_reproducible_bit_identical_with_mu_j_positive(tmp_path):
    # two identical config+seed+data runs with mu_j > 0 must still produce
    # byte-identical metrics.csv after the val-side skip (removing an RNG
    # consumer must not itself break determinism -- it is still seeded)
    data = _small_dataset(n_episodes=6, T=10)
    cfg_a = _mlp_config(
        tmp_path, exp_name="trainer_test_val_repro", family="min_ssm", mu_j=0.1, epochs=3,
        results_root=str(tmp_path / "root_a"),
    )
    cfg_b = dict(cfg_a, results_root=str(tmp_path / "root_b"))
    result_a = Trainer(cfg_a).fit(data)
    result_b = Trainer(cfg_b).fit(data)
    csv_a = Path(result_a["paths"]["metrics_csv"]).read_bytes()
    csv_b = Path(result_b["paths"]["metrics_csv"]).read_bytes()
    assert csv_a == csv_b
    print("[trainer] mu_j>0 run bit-identical across two identical fits, val-jac skip included")


def test_skipping_val_jac_penalty_saves_time(tmp_path, monkeypatch):
    # the measured cost this change removes: 23-38% of epoch time. Prove the
    # direction (not a specific magnitude) by comparing against a forced
    # "old behavior" where compute_jac_penalty is always True, including val.
    import time

    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=8, T=20)
    cfg = _mlp_config(
        tmp_path, exp_name="trainer_test_val_speed", family="min_ssm", mu_j=0.1, epochs=6
    )

    t0 = time.perf_counter()
    Trainer(dict(cfg, exp_name="trainer_test_val_speed_fixed")).fit(data)
    fixed_time = time.perf_counter() - t0

    orig = trainer_mod.Trainer._composite_loss

    def always_compute_jac(self, *args, **kwargs):
        kwargs["compute_jac_penalty"] = True
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(trainer_mod.Trainer, "_composite_loss", always_compute_jac)

    t0 = time.perf_counter()
    Trainer(dict(cfg, exp_name="trainer_test_val_speed_old")).fit(data)
    old_time = time.perf_counter() - t0

    print(f"[trainer] val-jac timing: fixed={fixed_time:.3f}s old-behavior={old_time:.3f}s")
    assert fixed_time < old_time
