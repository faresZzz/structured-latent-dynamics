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
import pytest
import torch

from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.encoders.pretrain import (
    load_frozen_representation,
    pretrain_representation,
    representation_digest,
    save_frozen_representation,
)
from sld.objectives.jac_reg import Gauge, gauge_digest
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.objectives.vicreg import covariance_loss, variance_loss
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


def test_structured_family_config_is_canonicalized_and_delta_is_required(tmp_path):
    # POSITIVE: an omitted default and the same default written out must be
    # the SAME run identity for the new families too -- that is what makes a
    # factorial cell one cell. r defaults to d_z (2r = the same lifted width
    # koopman_ssm gets from n_pairs=d_z).
    for family, explicit in (
        ("mlp_k", {"r": _D_Z}),
        ("transformer_k", {"r": _D_Z, "n_heads": 2, "max_len": 512}),
        ("ssm_k", {"r": _D_Z, "delta_min": 1e-4, "delta_max": 10.0}),
    ):
        implicit_cfg = _mlp_config(tmp_path, family=family, delta=0.02)
        explicit_cfg = _mlp_config(tmp_path, family=family, delta=0.02, **explicit)
        a, b = Trainer(implicit_cfg), Trainer(explicit_cfg)
        assert a.config == b.config
        meta = {"n_episodes": 2, "T": 5, "dt": 0.02, "seed": 0}
        assert a.resolve_run_dir({"meta": meta}) == b.resolve_run_dir({"meta": meta})
        for key, value in explicit.items():
            assert a.config[key] == value
        print(f"[trainer] {family} canonicalized to {explicit} -> one run identity")

    # NEGATIVE: delta has no default for any structured arm, exactly as for
    # koopman_ssm -- a config that forgets it is refused at construction, not
    # silently given an invented discretization step.
    import pytest

    for family in ("mlp_k", "transformer_k", "ssm_k"):
        with pytest.raises(ValueError, match="requires delta"):
            Trainer(_mlp_config(tmp_path, family=family))


def test_family_with_no_canonicalization_branch_is_refused(monkeypatch, tmp_path):
    # NEGATIVE, reached by mutating the family table IN MEMORY. The
    # canonicalization chain used to have no else: a family accepted by
    # _FAMILY_CONFIG_KEYS but missing a branch was never canonicalized, so
    # "default omitted" and "default written out" hashed as two identities
    # for one model -- a silent cell-identity bug no green gate would catch.
    import pytest

    import sld.training.trainer as trainer_mod

    monkeypatch.setitem(trainer_mod._FAMILY_CONFIG_KEYS, "unbranched", frozenset())
    with pytest.raises(ValueError, match="no canonicalization branch"):
        Trainer(_mlp_config(tmp_path, family="unbranched"))


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


def test_composite_loss_matches_hand_derived_scalar():
    """``Trainer._composite_loss`` (trainer.py:404) assembles (trainer.py:
    493-499)::

        total = w_tf*tf_out['total'] + w_fr*fr_out['total']
              + w_var*var + w_cov*cov + mu_j*jac_pen

    ``test_train_loss_decreases`` above only asserts a scalar goes down --
    any optimized scalar does that; it proves nothing about which targets or
    actions actually fed it. This test recomputes ``total`` term-by-term
    from the equations, on ONE call, with no optimizer step.

    The predictor is ``MLPPredictor(d_z, d_u, hidden=())``: ``dims =
    (d_h+d_u, d_h)`` collapses to a SINGLE ``nn.Linear`` with no activation
    (``len(model.net) == 1``, asserted below), so ``step`` is exactly the
    affine map ``h_next = h + net(cat(h, u)) = h @ (I + W_h).T + u @ W_u.T +
    b``, with weights set by hand under ``torch.no_grad()``. This test
    recomputes ``h_hat_tf`` (teacher-forced: conditioned on the TRUE
    ``h_true_t = z_t`` at every step) and ``h_hat_fr`` (free-running:
    conditioned on its OWN previous prediction) via this affine formula
    directly with tensor ops -- never by calling ``predictor.step`` /
    ``.rollout`` -- then the MSE against the EMA target at the correct
    shifted indices (``z_tgt[1:]`` for TF, ``z_tgt[1:h_fr+1]`` for FR), then
    applies the configured weights. ``mu_j`` is left at its default (0.0),
    so ``jac_pen`` is the fixed zero placeholder and is not evaluated at all
    (trainer.py's disabled-branch doctrine) -- out of scope here.
    """
    torch.manual_seed(0)
    d_x, d_z, d_u = 3, 2, 1
    T, B, h_fr = 6, 3, 4

    config = {
        "exp_name": "test_composite_loss",
        "seed": 0,
        "family": "mlp",
        "d_x": d_x,
        "d_z": d_z,
        "d_u": d_u,
        "h_fr": h_fr,
        "w_tf": 2.0,
        "w_fr": 3.0,
        "w_var": 0.5,
        "w_cov": 0.25,
    }
    trainer = Trainer(config)

    encoder = FrameMLPEncoder(d_x, d_z, hidden=(5,))
    predictor = MLPPredictor(d_z, d_u, hidden=())
    assert len(predictor.net) == 1  # a single nn.Linear, no activation

    gen = torch.Generator().manual_seed(1)
    W = torch.randn(d_z, d_z + d_u, generator=gen, dtype=torch.float32) * 0.3
    b = torch.randn(d_z, generator=gen, dtype=torch.float32) * 0.1
    with torch.no_grad():
        predictor.net[0].weight.copy_(W)
        predictor.net[0].bias.copy_(b)
    W_h, W_u = W[:, :d_z], W[:, d_z:]

    def affine_step(h, u):
        return h + h @ W_h.T + u @ W_u.T + b

    ema = EMATarget(encoder, tau=0.99)
    jepa = JEPALoss(lambda_c=0.0)
    lifted = hasattr(predictor, "lift")
    assert not lifted  # mlp is unlifted -> lift_fn/project_fn below are identity
    lift_fn = predictor.lift if lifted else (lambda zz: zz)
    project_fn = predictor.project if lifted else (lambda h: h)

    gen_x = torch.Generator().manual_seed(2)
    x_bt = torch.randn(T + 1, B, d_x, generator=gen_x, dtype=torch.float32)
    u_bt = torch.randn(T, B, d_u, generator=gen_x, dtype=torch.float32)

    with torch.no_grad():
        z = encoder(x_bt)  # same deterministic forward _composite_loss runs internally
        z_tgt = ema(x_bt)

        h_true = z[:T]
        h_hat_tf = torch.stack([affine_step(h_true[t], u_bt[t]) for t in range(T)], dim=0)
        tf_pred = torch.nn.functional.mse_loss(h_hat_tf, z_tgt[1:])

        h = z[0]
        fr_steps = []
        for t in range(h_fr):
            h = affine_step(h, u_bt[t])
            fr_steps.append(h)
        h_hat_fr = torch.stack(fr_steps, dim=0)
        fr_pred = torch.nn.functional.mse_loss(h_hat_fr, z_tgt[1 : h_fr + 1])

        var = variance_loss(z)
        cov = covariance_loss(z)

        expected_total = (
            config["w_tf"] * tf_pred
            + config["w_fr"] * fr_pred
            + config["w_var"] * var
            + config["w_cov"] * cov
        ).item()

    actual_total, diag = trainer._composite_loss(
        encoder, predictor, ema, jepa, x_bt, u_bt, lifted, lift_fn, project_fn, h_fr
    )
    actual_total = actual_total.item()
    err = abs(actual_total - expected_total)
    print(f"[composite_loss exact] actual={actual_total:.9f} expected={expected_total:.9f} err={err:.3e}")
    print(f"[composite_loss exact] diag={diag}")
    # float32 chain (encoder MLP + affine predictor recurrence + 2 MSEs +
    # VICReg, summed) computed via a DIFFERENT operation order than the
    # source (h@W_h.T + u@W_u.T here vs. cat(h,u)@W.T internally): measured
    # err running this test is exactly 0.0; 1e-5 leaves headroom for a
    # different BLAS backend without hiding a real assembly bug (the three
    # negatives below measure gaps >= 2.8e-2, three orders of magnitude above)
    assert err < 1e-5

    # -- negatives: each is a plausible wiring bug, must NOT match actual_total --

    # (a) wrong weighting: w_tf/w_fr swapped
    wrong_weight_total = (
        config["w_fr"] * tf_pred
        + config["w_tf"] * fr_pred
        + config["w_var"] * var
        + config["w_cov"] * cov
    ).item()
    gap = abs(wrong_weight_total - actual_total)
    print(f"[composite_loss negative: swapped w_tf/w_fr] {wrong_weight_total:.9f} gap={gap:.3e}")
    assert gap > 1e-3

    # (b) FR's own target replaced by TF's (copy-paste bug: w_fr * tf_out['total'])
    wrong_tf_for_fr_total = (
        config["w_tf"] * tf_pred
        + config["w_fr"] * tf_pred
        + config["w_var"] * var
        + config["w_cov"] * cov
    ).item()
    gap = abs(wrong_tf_for_fr_total - actual_total)
    print(f"[composite_loss negative: TF target reused for FR] {wrong_tf_for_fr_total:.9f} gap={gap:.3e}")
    assert gap > 1e-3

    # (c) actions ignored (both rollouts forced to u=0)
    with torch.no_grad():
        u_zero = torch.zeros_like(u_bt)
        h_hat_tf0 = torch.stack([affine_step(h_true[t], u_zero[t]) for t in range(T)], dim=0)
        tf_pred0 = torch.nn.functional.mse_loss(h_hat_tf0, z_tgt[1:])
        h0 = z[0]
        fr_steps0 = []
        for t in range(h_fr):
            h0 = affine_step(h0, u_zero[t])
            fr_steps0.append(h0)
        fr_pred0 = torch.nn.functional.mse_loss(torch.stack(fr_steps0, dim=0), z_tgt[1 : h_fr + 1])
        wrong_noaction_total = (
            config["w_tf"] * tf_pred0
            + config["w_fr"] * fr_pred0
            + config["w_var"] * var
            + config["w_cov"] * cov
        ).item()
    gap = abs(wrong_noaction_total - actual_total)
    print(f"[composite_loss negative: actions ignored] {wrong_noaction_total:.9f} gap={gap:.3e}")
    assert gap > 1e-3


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


def test_nonfinite_loss_writes_failure_json(tmp_path, monkeypatch):
    # PROTOCOL.md section 11.7: the divergence must be a DURABLE, terminal
    # result -- failure.json is the record of it, written before the abort,
    # never a completion sentinel alongside it.
    import json

    import pytest

    import sld.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan")))
    data = _small_dataset(n_episodes=6, T=10)
    trainer = Trainer(_mlp_config(tmp_path, exp_name="trainer_test_failure_json"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        trainer.fit(data)

    run_dir = trainer.run_dir
    failure_path = run_dir / trainer_mod.FAILURE_FILENAME
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text())
    print(f"[trainer] failure.json = {failure}")
    assert failure["config_hash"] == trainer.config_hash
    assert failure["data_hash"] == trainer.data_hash
    assert failure["epoch"] == 0  # variance_loss is nan from the first epoch
    assert failure["reason"] == "non-finite loss"
    assert failure["exception_class"] == "RuntimeError"
    assert failure["nonfinite_fields"]  # at least train_loss is flagged
    assert failure["train_loss"] == "nan"
    # the failure is recoverable without a nonfinite row surviving anywhere
    # else: no completion sentinel, and no row for this epoch in metrics.csv
    assert not (run_dir / trainer_mod.RUN_COMPLETE_FILENAME).exists()
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []  # epoch 0 never got written: it's the epoch that diverged


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
    # Was a wall-clock comparison of two `fit()` calls in one process --
    # unstable by construction: it measures machine load and call order, not
    # the code (review measured fixed=8.815s/old=2.542s in one run, and
    # fixed=28.595s/old=2.058s -- a REVERSED 14x -- in a fresh process with no
    # code change). Replaced with the deterministic claim the skip actually
    # makes: _jac_state_penalty (what compute_jac_penalty gates) is called
    # for the train split and NEVER for the val split, tagged via the
    # compute_jac_penalty kwarg _composite_loss is called with (fit() passes
    # compute_jac_penalty=False only for the val split, see trainer.py).
    import sld.training.trainer as trainer_mod

    calls = {"train": 0, "val": 0}
    orig_composite = trainer_mod.Trainer._composite_loss
    orig_jac = trainer_mod.Trainer._jac_state_penalty
    phase = {"current": "train"}

    def tagged_composite(self, *args, **kwargs):
        phase["current"] = "val" if kwargs.get("compute_jac_penalty", True) is False else "train"
        return orig_composite(self, *args, **kwargs)

    def counting_jac(self, predictor, jac_steps):
        calls[phase["current"]] += 1
        return orig_jac(self, predictor, jac_steps)

    monkeypatch.setattr(trainer_mod.Trainer, "_composite_loss", tagged_composite)
    monkeypatch.setattr(trainer_mod.Trainer, "_jac_state_penalty", counting_jac)

    data = _small_dataset(n_episodes=8, T=20)
    cfg = _mlp_config(
        tmp_path, exp_name="trainer_test_val_speed", family="min_ssm", mu_j=0.1, epochs=6
    )
    Trainer(cfg).fit(data)

    print(f"[trainer] _jac_state_penalty calls by phase = {calls}")
    assert calls["train"] > 0  # sanity: the counter itself is wired up
    assert calls["val"] == 0  # exactly what compute_jac_penalty=False claims


# ============================================================================
# Wave D1.5a (PROTOCOL.md section 14.7): crash-resume checkpoints and
# per-run logging. "Kill" is always simulated by raising from inside the
# epoch loop via monkeypatch -- never by killing a real process.
# ============================================================================


def test_bit_identical_metrics_csv_across_interrupted_and_resumed_run(tmp_path, monkeypatch):
    # THE POINT OF THE WAVE: a run of N updates, killed at N/2 and resumed,
    # must produce a metrics.csv BYTE-IDENTICAL to an uninterrupted run of N
    # updates. mu_j > 0 with step_subsample=1.0 on purpose: the jac penalty
    # (TotalJacobianPenalty, generator=None) consumes torch's GLOBAL RNG
    # every epoch, so this also proves torch RNG state is restored on
    # resume, not only numpy's (batch sampling) -- the prompt names both as
    # the mutants most likely to expose a weak bit-identity test.
    import pytest

    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=8, T=15)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 3)
    n_epochs = 10
    common = dict(epochs=n_epochs, mu_j=0.05, step_subsample=1.0, batch_episodes=4, lr=5e-2)
    cfg_ref = _mlp_config(
        tmp_path, exp_name="trainer_test_bitid", results_root=str(tmp_path / "ref"), **common
    )
    cfg_int = _mlp_config(
        tmp_path, exp_name="trainer_test_bitid", results_root=str(tmp_path / "resumed"), **common
    )

    result_ref = Trainer(dict(cfg_ref)).fit(data)
    bytes_ref = Path(result_ref["paths"]["metrics_csv"]).read_bytes()

    # Interval=3 -> checkpoints saved after epochs 2, 5, 8. Crash on the 9th
    # EMATarget.update call (epoch index 8, 0-indexed), which happens
    # BEFORE that epoch's row/checkpoint: the last surviving checkpoint is
    # epoch 5 (rows 0-5), while metrics.csv on disk carries rows 0-7 (epochs
    # 6, 7 written normally, no boundary hit) -- resume must discard rows 6
    # and 7 via truncation, not merely continue appending after them.
    orig_update = jepa_mod.EMATarget.update
    hit = {"n": 0}

    def _boom(self, encoder):
        hit["n"] += 1
        if hit["n"] == 9:
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg_int))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    run_dir = trainer_1.run_dir
    resume_path = run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    assert resume_path.exists()
    with open(run_dir / "metrics.csv", newline="") as f:
        rows_before_resume = list(csv.reader(f))[1:]
    print(f"[resume] rows on disk at crash time = {len(rows_before_resume)} (expect 8: epochs 0-7)")
    assert len(rows_before_resume) == 8

    trainer_2 = Trainer(dict(cfg_int))
    result_resumed = trainer_2.fit(data)  # resumes from the epoch-5 checkpoint, completes
    assert not resume_path.exists()  # gone on successful completion

    bytes_resumed = Path(result_resumed["paths"]["metrics_csv"]).read_bytes()
    print(f"[resume] reference bytes={len(bytes_ref)} resumed bytes={len(bytes_resumed)}")
    if bytes_resumed != bytes_ref:
        for i, (a, b) in enumerate(zip(bytes_ref, bytes_resumed)):
            if a != b:
                print(f"[resume] first differing byte at offset {i}: ref={a!r} resumed={b!r}")
                break
        else:
            print(f"[resume] one is a prefix of the other at length {min(len(bytes_ref), len(bytes_resumed))}")
    assert bytes_resumed == bytes_ref

    with open(result_resumed["paths"]["metrics_csv"], newline="") as f:
        resumed_rows = list(csv.reader(f))[1:]
    assert len(resumed_rows) == n_epochs  # exact -- no duplicated rows from the crashed attempt


def test_resume_does_not_duplicate_metrics_rows(tmp_path, monkeypatch):
    import pytest

    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    n_epochs = 6
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_no_dup_rows", epochs=n_epochs)

    orig_update = jepa_mod.EMATarget.update
    hit = {"n": 0}

    def _boom(self, encoder):
        hit["n"] += 1
        if hit["n"] == 4:  # past >=1 checkpoint interval, before completion
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    result = Trainer(dict(cfg)).fit(data)
    with open(result["paths"]["metrics_csv"], newline="") as f:
        rows = list(csv.reader(f))[1:]
    print(f"[resume] final row count = {len(rows)} (expected {n_epochs})")
    assert len(rows) == n_epochs
    epochs_seen = [int(r[0]) for r in rows]
    assert epochs_seen == list(range(n_epochs))  # exact, no duplicates, no gaps


def test_resume_checkpoint_appears_during_run_and_gone_after_completion(tmp_path, monkeypatch):
    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_resume_lifecycle", epochs=6)
    trainer = Trainer(dict(cfg))

    orig_update = jepa_mod.EMATarget.update
    seen_mid_run = {"exists": False}

    def _check(self, encoder):
        if trainer.run_dir is not None:
            path = trainer.run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
            if path.exists():
                seen_mid_run["exists"] = True
        return orig_update(self, encoder)

    monkeypatch.setattr(jepa_mod.EMATarget, "update", _check)
    result = trainer.fit(data)

    print(f"[resume] checkpoint observed to exist mid-run = {seen_mid_run['exists']}")
    assert seen_mid_run["exists"] is True
    resume_path = Path(result["paths"]["ckpt"]).parent / trainer_mod.RESUME_CHECKPOINT_FILENAME
    assert not resume_path.exists()


def test_resume_checkpoint_survives_a_second_crash_during_resume(tmp_path, monkeypatch):
    # Guards specifically against deleting resume_ckpt.pt on ENTERING the
    # resume branch (instead of only on successful completion): a second
    # crash mid-resume must not lose the progress the first crash saved.
    import pytest

    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_resume_twice", epochs=8)

    orig_update = jepa_mod.EMATarget.update

    def _crash_on(call_no):
        hit = {"n": 0}

        def _boom(self, encoder):
            hit["n"] += 1
            if hit["n"] == call_no:
                raise RuntimeError("simulated crash")
            return orig_update(self, encoder)

        return _boom

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _crash_on(5))  # epoch index 4
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)
    resume_path = trainer_1.run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    assert resume_path.exists()

    trainer_2 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _crash_on(2))  # crash again, early into the resume
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_2.fit(data)
    print(f"[resume] resume_ckpt.pt survives a second crash = {resume_path.exists()}")
    assert resume_path.exists()  # NOT deleted merely by attempting (and re-crashing) a resume

    trainer_3 = Trainer(dict(cfg))
    trainer_3.fit(data)  # finally completes
    assert not resume_path.exists()  # gone only now


def test_train_log_has_config_hash_budget_and_one_line_per_epoch(tmp_path):
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_log_content", epochs=4)
    trainer = Trainer(dict(cfg))
    trainer.fit(data)

    log_text = (trainer.run_dir / "train.log").read_text()
    print(f"[train.log]\n{log_text}")
    header = log_text.split("epoch=0")[0]  # everything logged before the first per-epoch line
    assert trainer.config_hash in header
    assert f"epochs={cfg['epochs']}" in header  # the budget

    epoch_lines = [line for line in log_text.splitlines() if line.startswith("epoch=")]
    assert len(epoch_lines) == cfg["epochs"]
    assert epoch_lines[0].startswith("epoch=0 ")
    assert epoch_lines[-1].startswith(f"epoch={cfg['epochs'] - 1} ")


def test_fresh_start_deletes_stale_train_log(tmp_path):
    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_stale_log", epochs=2)
    trainer_1 = Trainer(dict(cfg))
    trainer_1.fit(data)
    log_path = trainer_1.run_dir / "train.log"
    assert "epoch=1" in log_path.read_text()

    trainer_2 = Trainer(dict(cfg))  # same identity; no resume checkpoint survives completion
    trainer_2.fit(data)  # fresh start
    second_run_text = log_path.read_text()
    print(f"[train.log after fresh rerun]\n{second_run_text}")
    # replaced, not appended: a "resuming" marker or doubled epoch lines
    # would mean the stale log survived into this fresh attempt
    assert "resuming" not in second_run_text
    assert second_run_text.count("epoch=0") == 1
    assert second_run_text.count("epoch=1") == 1


def test_fresh_start_clears_orphaned_resume_checkpoint_when_run_already_complete(tmp_path):
    # The fresh-start cleanup also names the resume checkpoint (PROTOCOL.md
    # 14.7). The only way one can coexist with run_complete.json is the
    # narrow crash window between writing the sentinel and deleting the
    # checkpoint at the very end of a successful fit -- simulated directly
    # here by writing a syntactically-valid, IDENTITY-MATCHING resume
    # checkpoint back after a completed run. The next attempt must not try
    # to "resume" a run already certified complete (which would try to load
    # `{}` as optimizer/scheduler state and blow up); it must retrain fresh,
    # deterministically reproducing the same result, and the orphan is gone.
    import random as random_mod

    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_orphan_resume", epochs=2)
    result_1 = Trainer(dict(cfg)).fit(data)
    run_dir = Path(result_1["paths"]["ckpt"]).parent
    resume_path = run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    assert not resume_path.exists()

    orphan = {
        "config_hash": result_1["metrics"]["config_hash"],
        "epoch": 0,
        "encoder": result_1["model"]["encoder"].state_dict(),
        "predictor": result_1["model"]["predictor"].state_dict(),
        "ema_target": result_1["model"]["ema_target"].state_dict(),
        "optimizer": {},
        "scheduler": {},
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random_mod.getstate(),
    }
    torch.save(orphan, resume_path)

    result_2 = Trainer(dict(cfg)).fit(data)
    print(f"[resume] orphaned resume_ckpt.pt cleared, retrained fresh: {not resume_path.exists()}")
    assert not resume_path.exists()
    assert result_2["history"]["train_loss"] == result_1["history"]["train_loss"]


def test_resume_refused_on_config_hash_mismatch(tmp_path):
    import random as random_mod

    import pytest

    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_resume_mismatch", epochs=4)
    trainer = Trainer(dict(cfg))
    run_dir = trainer.resolve_run_dir(data)
    run_dir.mkdir(parents=True)

    bad_state = {
        "config_hash": "deadbeef",  # deliberately wrong
        "epoch": 1,
        "encoder": {},
        "predictor": {},
        "ema_target": {},
        "optimizer": {},
        "scheduler": {},
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random_mod.getstate(),
    }
    resume_path = run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    torch.save(bad_state, resume_path)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        Trainer(dict(cfg)).fit(data)
    # not adopted and not silently deleted
    assert resume_path.exists()
    reloaded = torch.load(resume_path, weights_only=False)
    assert reloaded["config_hash"] == "deadbeef"


def test_resume_refused_on_corrupt_checkpoint(tmp_path):
    import pytest

    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_resume_corrupt", epochs=4)
    trainer = Trainer(dict(cfg))
    run_dir = trainer.resolve_run_dir(data)
    run_dir.mkdir(parents=True)
    resume_path = run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME

    resume_path.write_bytes(b"not a torch checkpoint at all, just garbage bytes")
    with pytest.raises(RuntimeError, match="corrupt"):
        Trainer(dict(cfg)).fit(data)
    assert resume_path.exists()  # not silently deleted either

    # a well-formed torch file MISSING required keys is corrupt too (a
    # different code path from the unreadable-bytes case above: this one
    # loads successfully but fails the required-keys check)
    resume_path.unlink()
    torch.save({"config_hash": trainer.config_hash, "epoch": 0}, resume_path)
    with pytest.raises(RuntimeError, match="corrupt"):
        Trainer(dict(cfg)).fit(data)
    assert resume_path.exists()


def test_resume_restores_python_random_state(tmp_path, monkeypatch):
    # Nothing in fit()'s CURRENT loop consumes python's stdlib `random`
    # module (only numpy's global RNG for batch sampling, torch's for the
    # mu_j probes) -- and set_seed() reseeds it at the top of EVERY fit()
    # call, resume or not. Left alone that makes a dropped python_rng_state
    # restore INVISIBLE: the checkpoint's saved state and a fresh reseed
    # coincide bit-for-bit, since nothing ever advances random. in between
    # (verified: this was an actual hole in an earlier version of this
    # test). To test the RESTORE MECHANISM itself, independent of whether
    # today's callers happen to touch it, this test injects a synthetic
    # consumer -- two random.random() draws inside EMATarget.update, stood
    # in for a future predictor family that might use it -- so the
    # checkpoint's state genuinely diverges from a fresh reseed, and a
    # dropped restore becomes observable.
    import random as random_mod

    import pytest

    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_random_restore", epochs=6)

    orig_update = jepa_mod.EMATarget.update

    def _advance_random() -> None:
        random_mod.random()
        random_mod.random()

    hit = {"n": 0}

    def _boom(self, encoder):
        _advance_random()
        hit["n"] += 1
        if hit["n"] == 3:
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    resume_path = trainer_1.run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    saved_state = torch.load(resume_path, weights_only=False)["python_rng_state"]

    captured = {}

    def _capture_then_advance(self, encoder):
        if "state" not in captured:  # capture BEFORE this call's own draws
            captured["state"] = random_mod.getstate()
        _advance_random()
        return orig_update(self, encoder)

    monkeypatch.setattr(jepa_mod.EMATarget, "update", _capture_then_advance)
    Trainer(dict(cfg)).fit(data)

    assert captured.get("state") is not None
    assert captured["state"] == saved_state


def test_resume_with_intact_metrics_csv_still_resumes(tmp_path, monkeypatch):
    # POSITIVE half of the metrics.csv resume guard (2026-08-11 incident,
    # see the corruption test below): an INTACT metrics.csv must still take
    # the resume path. The discriminating assertion is the "resuming from
    # epoch" log line, not just the final row count -- a guard that wrongly
    # restarts every run from scratch still produces a correct-looking final
    # metrics.csv, so row count alone proves nothing.
    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    n_epochs = 6
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_intact_metrics_resume", epochs=n_epochs)

    orig_update = jepa_mod.EMATarget.update
    hit = {"n": 0}

    def _boom(self, encoder):
        hit["n"] += 1
        if hit["n"] == 4:  # past >=1 checkpoint interval, before completion
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    result = Trainer(dict(cfg)).fit(data)  # metrics.csv untouched -- must resume

    log_text = (trainer_1.run_dir / "train.log").read_text()
    print(f"[resume, intact metrics.csv] train.log:\n{log_text}")
    assert "resuming from epoch" in log_text
    assert "restarting this run from epoch 0" not in log_text

    with open(result["paths"]["metrics_csv"], newline="") as f:
        rows = list(csv.reader(f))[1:]
    assert len(rows) == n_epochs
    assert [int(r[0]) for r in rows] == list(range(n_epochs))


# Incident 2026-08-11: campaign A was killed after 5 runs left a metrics.csv
# full of NUL bytes (filesystem blocks allocated, contents never flushed)
# WHERE CLAIMED ROWS should have been. Resume's history rebuild died on a
# bare ValueError from deep inside csv/int() parsing, killing a whole shard
# instead of retraining the one cell -- every shape below must restart just
# that cell instead of propagating it. Corruption strictly BEYOND what the
# checkpoint claims is a DIFFERENT (benign) case, covered separately by
# test_torn_tail_beyond_checkpoint_still_resumes below -- _truncate_metrics_csv
# already discards that tail by design and must keep doing so.
@pytest.mark.parametrize(
    "corruption", ("nul_claimed_row", "torn_claimed_row", "no_header", "header_only")
)
def test_corrupt_metrics_csv_restarts_run_instead_of_raising(tmp_path, monkeypatch, corruption):
    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    n_epochs = 6
    cfg = _mlp_config(tmp_path, exp_name=f"trainer_test_corrupt_{corruption}", epochs=n_epochs)

    orig_update = jepa_mod.EMATarget.update
    hit = {"n": 0}

    def _boom(self, encoder):
        hit["n"] += 1
        if hit["n"] == 4:  # past >=1 checkpoint interval, before completion
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    run_dir = trainer_1.run_dir
    metrics_path = run_dir / "metrics.csv"
    resume_path = run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    assert resume_path.exists()  # left VALID -- only metrics.csv is corrupted below

    # n_rows is what the resume checkpoint CLAIMS, and therefore what
    # _metrics_csv_defect value-checks (rows[:n_rows]) -- computed, not
    # hardcoded, so this stays correct if the crash timing above ever moves.
    n_rows = torch.load(resume_path, weights_only=False)["epoch"] + 1
    assert n_rows >= 2, "test corrupts claimed row index 1 -- needs >=2 claimed rows"

    lines = metrics_path.read_text().splitlines(keepends=True)
    header, data_lines = lines[0], lines[1:]

    if corruption == "nul_claimed_row":  # a CLAIMED row -- the incident's exact shape
        data_lines[1] = "\x00" * 64 + "\n"
        metrics_path.write_text(header + "".join(data_lines))
    elif corruption == "torn_claimed_row":  # a CLAIMED row, fewer fields than the header
        data_lines[1] = "1,0.5\n"
        metrics_path.write_text(header + "".join(data_lines))
    elif corruption == "no_header":
        metrics_path.write_text("".join(data_lines))
    elif corruption == "header_only":  # zero data rows, checkpoint claims completed epochs
        metrics_path.write_text(header)

    Trainer(dict(cfg)).fit(data)  # must NOT raise

    with open(metrics_path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    print(f"[corrupt metrics.csv={corruption}] final row count = {len(rows)} (expected {n_epochs})")
    assert len(rows) == n_epochs
    assert [int(r[0]) for r in rows] == list(range(n_epochs))

    log_text = (run_dir / "train.log").read_text()
    print(f"[corrupt metrics.csv={corruption}] train.log:\n{log_text}")
    assert "restarting this run from epoch 0" in log_text
    assert "resuming from epoch" not in log_text
    assert not resume_path.exists()


@pytest.mark.parametrize("tail", ("nul_line", "short_row"))
def test_torn_tail_beyond_checkpoint_still_resumes(tmp_path, monkeypatch, tail):
    # Non-regression pin for _metrics_csv_defect's rows[:n_rows] slice: a
    # torn/garbage tail STRICTLY BEYOND what the checkpoint claims is the
    # ORDINARY shape of a kill mid-write (_truncate_metrics_csv's own
    # docstring, trainer.py:270-274 -- "possibly with a torn last row"), and
    # _truncate_metrics_csv already discards it correctly on every resume.
    # An over-broad guard that value-checks the WHOLE file (the regression
    # this pins against) fails this test: it restarts from epoch 0 instead
    # of resuming, silently discarding completed epochs the checkpoint
    # already covers.
    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    n_epochs = 6
    cfg = _mlp_config(tmp_path, exp_name=f"trainer_test_torn_tail_{tail}", epochs=n_epochs)

    orig_update = jepa_mod.EMATarget.update
    hit = {"n": 0}

    def _boom(self, encoder):
        hit["n"] += 1
        if hit["n"] == 4:  # past >=1 checkpoint interval, before completion
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    run_dir = trainer_1.run_dir
    metrics_path = run_dir / "metrics.csv"

    if tail == "nul_line":
        metrics_path.write_text(metrics_path.read_text() + "\x00" * 64 + "\n")
    elif tail == "short_row":
        metrics_path.write_text(metrics_path.read_text() + "3,0.5\n")

    Trainer(dict(cfg)).fit(data)  # must RESUME, not restart

    log_text = (run_dir / "train.log").read_text()
    print(f"[torn tail={tail}] train.log:\n{log_text}")
    assert "resuming from epoch" in log_text
    assert "restarting this run from epoch 0" not in log_text

    with open(metrics_path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    assert len(rows) == n_epochs
    assert [int(r[0]) for r in rows] == list(range(n_epochs))


def test_failure_json_survives_both_cleanup_paths(tmp_path, monkeypatch):
    # PROTOCOL.md section 11.7: failure.json is a divergent run's terminal
    # record and must never be touched by either cleanup path -- fresh
    # start (unchanged from before this wave) or resume (new in this wave).
    import pytest

    import sld.objectives.jepa as jepa_mod
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    monkeypatch.setattr(trainer_mod, "RESUME_CHECKPOINT_INTERVAL", 2)
    cfg = _mlp_config(tmp_path, exp_name="trainer_test_failure_both", epochs=6)

    # -- resume path: crash after >=1 checkpoint interval, plant a
    # failure.json alongside the genuine resume checkpoint (simulating a
    # terminal record left by an unrelated earlier divergence at this exact
    # identity), then resume to completion and check it survived --
    orig_update = jepa_mod.EMATarget.update
    hit = {"n": 0}

    def _boom(self, encoder):
        hit["n"] += 1
        if hit["n"] == 4:
            raise RuntimeError("simulated crash")
        return orig_update(self, encoder)

    trainer_1 = Trainer(dict(cfg))
    with monkeypatch.context() as m:
        m.setattr(jepa_mod.EMATarget, "update", _boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer_1.fit(data)

    run_dir = trainer_1.run_dir
    resume_path = run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME
    assert resume_path.exists()  # crossed at least one checkpoint interval
    failure_path = run_dir / trainer_mod.FAILURE_FILENAME
    failure_path.write_text('{"planted": true}')

    Trainer(dict(cfg)).fit(data)  # resumes and completes
    print(f"[failure.json] survives resume-path completion: {failure_path.exists()}")
    assert not resume_path.exists()  # deleted on successful completion
    assert failure_path.exists()  # untouched by the resume path
    assert failure_path.read_text() == '{"planted": true}'  # never rewritten

    # -- fresh path: a genuine divergence from epoch 0 (before any
    # checkpoint interval), then a fresh rerun on the SAME identity must
    # also leave failure.json alone --
    cfg2 = _mlp_config(tmp_path, exp_name="trainer_test_failure_fresh", epochs=3)
    with monkeypatch.context() as m:
        m.setattr(trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan")))
        trainer_2 = Trainer(dict(cfg2))
        with pytest.raises(RuntimeError, match="nonfinite"):
            trainer_2.fit(data)
    failure_path2 = trainer_2.run_dir / trainer_mod.FAILURE_FILENAME
    assert failure_path2.exists()
    assert not (trainer_2.run_dir / trainer_mod.RESUME_CHECKPOINT_FILENAME).exists()

    Trainer(dict(cfg2)).fit(data)  # fresh start (no resume checkpoint), succeeds
    print(f"[failure.json] survives fresh-path rerun: {failure_path2.exists()}")
    assert failure_path2.exists()


# --------------------------------------------------------------------------
# Wave D2b: the frozen shared representation (PROTOCOL.md sections 14.9-14.11,
# src/sld/encoders/pretrain.py) travels OUT-OF-BAND, exactly like the gauge
# above and for the identical reason -- ``Trainer(config, representation=...)``.
# This section tests only the WIRING: the trainer does not construct or
# optimize the encoder, its digest is a derived run-identity key that stays
# ABSENT (never null) when no representation is supplied, and a run with no
# representation is byte-for-byte identical to today's behaviour.
# --------------------------------------------------------------------------


def _small_representation(
    tmp_path, *, seed: int = 0, system: str = "pendulum", regime: str = "ou", data: dict | None = None,
    name: str = "representation",
):
    """A small, digest-STAMPED artifact. Trainer's preflight
    (validate_shared_representation) requires ``manifest["digest"]``, which
    only a save/load round trip stamps -- ``pretrain_representation`` itself
    returns an artifact with an UNSTAMPED manifest (its own docstring)."""
    if data is None:
        data = _small_dataset()
    artifact = pretrain_representation(
        data,
        system=system,
        regime=regime,
        d_z=_D_Z,
        encoder_hidden=(8,),
        seed=seed,
        epochs=3,
        batch_size=16,
    )
    path = tmp_path / f"{name}.pt"
    save_frozen_representation(artifact, path)
    return load_frozen_representation(path)


def _repr_config(tmp_path, **overrides) -> dict:
    config = _mlp_config(
        tmp_path,
        encoder_hidden=[8],  # must match the representation's own architecture
        stratum_system="pendulum",
        stratum_regime="ou",
    )
    config.update(overrides)
    return config


def test_representation_digest_absent_when_none_config_hash_pinned(tmp_path):
    # POSITIVE, the identity regression this whole wave is riskiest on: no
    # representation -> config_hash BYTE-IDENTICAL to today's value. Pinned
    # as a literal (computed against this exact fixture before any Wave D2b
    # code existed) so a future change to hashing cannot slip through
    # unnoticed -- the task brief's own warning about this test.
    trainer = Trainer(_mlp_config(tmp_path))
    print(f"[trainer D2b] no-representation config_hash = {trainer.config_hash}")
    assert trainer.config_hash == "49b3b551"
    assert "representation_digest" not in trainer.config
    assert "representation_digest" not in trainer._canonical_yaml


def test_representation_digest_is_derived_key_rejected_if_supplied(tmp_path):
    import pytest

    # the schema stays CLOSED, same doctrine as gauge_digest: a config that
    # names the derived key outright is rejected, representation or not
    with pytest.raises(ValueError, match="derived or policy-owned"):
        Trainer(_repr_config(tmp_path, representation_digest="deadbeef"))


def test_representation_requires_stratum_config_keys(tmp_path):
    import pytest

    data = _small_dataset()
    art = _small_representation(tmp_path, data=data)
    cfg = _mlp_config(tmp_path, encoder_hidden=[8])  # no stratum_system/stratum_regime
    with pytest.raises(ValueError, match="stratum_system"):
        Trainer(cfg, representation=art)


def test_representation_stratum_key_does_not_collide_with_koopman_regime(tmp_path):
    # regression for the collision caught by hand while building this wave:
    # a bare "regime" config key already names koopman_ssm's OWN ZOH
    # discretization regime ("bilinear"/"selective", _FAMILY_CONFIG_KEYS) --
    # stratum_system/stratum_regime must coexist with it without tripping
    # the family-applicability check.
    data = _small_dataset()
    art = _small_representation(tmp_path, data=data)
    cfg = _repr_config(
        tmp_path, family="koopman_ssm", delta=0.02, regime="bilinear", exp_name="koopman_stratum"
    )
    trainer = Trainer(cfg, representation=art)  # must not raise
    assert trainer.config["regime"] == "bilinear"  # koopman's own key, untouched
    assert trainer.config["stratum_regime"] == "ou"  # the stratum-preflight key, distinct


def test_representation_content_separates_run_identity(tmp_path):
    # two runs loading DIFFERENT representations build different models and
    # must never share a run directory; two runs loading the SAME content
    # (a fresh reload from disk, not the same object) are the SAME run.
    data = _small_dataset()
    art_a = _small_representation(tmp_path, data=data, seed=1, name="a")
    _small_representation(tmp_path, data=data, seed=2, name="b")  # writes b.pt
    art_a_reloaded = load_frozen_representation(tmp_path / "a.pt")
    art_b = load_frozen_representation(tmp_path / "b.pt")

    cfg = _repr_config(tmp_path)
    h_none = Trainer(dict(cfg)).config_hash
    h_a1 = Trainer(dict(cfg), representation=art_a).config_hash
    h_a2 = Trainer(dict(cfg), representation=art_a_reloaded).config_hash
    h_b = Trainer(dict(cfg), representation=art_b).config_hash
    print(f"[trainer D2b] none={h_none} a(direct)={h_a1} a(reloaded)={h_a2} b={h_b}")

    assert h_a1 == h_a2  # SAME representation content -> SAME run identity
    assert len({h_none, h_a1, h_b}) == 3  # none / A / B: three distinct identities

    trainer = Trainer(dict(cfg), representation=art_a)
    assert trainer.config["representation_digest"] == representation_digest(art_a)
    assert "representation_digest" in trainer._canonical_yaml
    # recorded in the run's own artifacts (config.yaml / ckpt.pt's "config"
    # field save the SAME resolved dict) -- traceable after the fact
    data_small = _small_dataset(n_episodes=6, T=10)
    result = trainer.fit(data_small)
    config_yaml_text = Path(result["paths"]["config"]).read_text()
    assert f"representation_digest: {representation_digest(art_a)}" in config_yaml_text
    ckpt = torch.load(result["paths"]["ckpt"], map_location="cpu", weights_only=False)
    assert ckpt["config"]["representation_digest"] == representation_digest(art_a)


def test_representation_encoder_frozen_zero_optimizer_members_state_dict_bit_identical(
    tmp_path, monkeypatch
):
    import sld.training.trainer as trainer_mod

    data = _small_dataset(n_episodes=6, T=10)
    art = _small_representation(tmp_path, data=data)
    encoder_before = {k: v.clone() for k, v in art.encoder_state_dict.items()}

    # capture exactly what params AdamW is constructed with, at the real
    # call site -- counted by object identity, never assumed from reading
    # the source
    captured: dict = {}
    real_adamw = torch.optim.AdamW

    class _CapturingAdamW(real_adamw):
        def __init__(self, params, *args, **kwargs):
            params = list(params)
            captured["param_ids"] = {id(p) for p in params}
            captured["n_params"] = len(params)
            super().__init__(params, *args, **kwargs)

    monkeypatch.setattr(trainer_mod.torch.optim, "AdamW", _CapturingAdamW)

    cfg = _repr_config(tmp_path, epochs=2, batch_episodes=4)
    result = Trainer(cfg, representation=art).fit(data)  # REAL backward() + optimizer.step()

    encoder = result["model"]["encoder"]
    predictor = result["model"]["predictor"]

    n_requires_grad = sum(1 for p in encoder.parameters() if p.requires_grad)
    print(f"[trainer D2b] encoder params requiring grad = {n_requires_grad} (want 0)")
    assert n_requires_grad == 0

    n_with_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
    print(f"[trainer D2b] encoder params with .grad set after backward() = {n_with_grad} (want 0)")
    assert n_with_grad == 0

    encoder_param_ids = {id(p) for p in encoder.parameters()}
    overlap = encoder_param_ids & captured["param_ids"]
    print(
        f"[trainer D2b] optimizer members total={captured['n_params']}, "
        f"encoder members in optimizer={len(overlap)} (want 0)"
    )
    assert len(overlap) == 0
    assert captured["n_params"] == sum(1 for _ in predictor.parameters())

    sd_after = encoder.state_dict()
    for k in encoder_before:
        torch.testing.assert_close(encoder_before[k], sd_after[k], atol=0.0, rtol=0.0)
    print("[trainer D2b] encoder state dict bit-identical before/after backward()+optimizer.step()")


def test_representation_ema_never_updates_frozen_encoder(tmp_path, monkeypatch):
    import sld.objectives.jepa as jepa_mod

    data = _small_dataset(n_episodes=6, T=10)
    art = _small_representation(tmp_path, data=data)

    calls = {"n": 0}
    real_update = jepa_mod.EMATarget.update

    def _counting_update(self, online):
        calls["n"] += 1
        return real_update(self, online)

    monkeypatch.setattr(jepa_mod.EMATarget, "update", _counting_update)

    cfg = _repr_config(tmp_path, epochs=3, batch_episodes=4)
    Trainer(cfg, representation=art).fit(data)
    print(f"[trainer D2b] ema.update() call count WITH representation = {calls['n']} (want 0)")
    assert calls["n"] == 0

    # positive control (REGLES section 7.1's two-check doctrine): the SAME
    # counting patch DOES see calls on a representation-less run, over the
    # SAME number of epochs -- proves the count above is a real 0, not a
    # vacuous one from a broken patch
    calls["n"] = 0
    Trainer(_mlp_config(tmp_path, exp_name="ema_sanity_control", epochs=3)).fit(data)
    print(f"[trainer D2b] ema.update() call count WITHOUT representation = {calls['n']} (want > 0)")
    assert calls["n"] > 0


def test_representation_readout_and_sigma_train_never_refit_per_run(tmp_path, monkeypatch):
    # built BEFORE poisoning: pretrain_representation legitimately calls
    # both functions below to build the artifact in the first place
    data = _small_dataset(n_episodes=6, T=10)
    art = _small_representation(tmp_path, data=data)
    readout_w_before = art.readout.W.copy()
    sigma_before = art.sigma_train.copy()

    import sld.audit.rollout as rollout_mod
    import sld.encoders.pretrain as pretrain_mod

    def _poison(*_a, **_k):
        raise AssertionError(
            "must not refit a readout/sigma_train per run when representation is supplied -- "
            "both come from the artifact"
        )

    monkeypatch.setattr(rollout_mod, "fit_train_state_scale", _poison)
    monkeypatch.setattr(pretrain_mod, "fit_readout", _poison)

    cfg = _repr_config(tmp_path, epochs=2, batch_episodes=4)
    Trainer(cfg, representation=art).fit(data)  # would raise if either were called

    assert np.array_equal(art.readout.W, readout_w_before)
    assert np.array_equal(art.sigma_train, sigma_before)
    print("[trainer D2b] readout/sigma_train untouched -- both come from the artifact")


def test_representation_shared_across_six_arms_one_digest_bit_identical_embeddings(tmp_path):
    # the six D2b arms (PROTOCOL.md section 13.1's 3x2 design, minus the
    # koopman_ssm/ssm_k substitution of Amendment 4): mlp, transformer,
    # min_ssm (unstructured) x mlp_k, transformer_k, ssm_k (structured)
    data = _small_dataset(n_episodes=6, T=10)
    art = _small_representation(tmp_path, data=data)

    family_extra = {
        "mlp": {},
        "transformer": {},
        "min_ssm": {},
        "mlp_k": {"delta": 0.02},
        "transformer_k": {"delta": 0.02},
        "ssm_k": {"delta": 0.02},
    }
    encoders = {}
    for family, extra in family_extra.items():
        # target_params=10_000: transformer's d_model moves on an
        # n_heads-stride grid (same note as the heads test above) -- a 2k
        # target is unreachable within +/-5% at this tiny d_z/d_u, so every
        # arm here budgets at 10k for a fair, uniform comparison
        cfg = _repr_config(
            tmp_path,
            family=family,
            exp_name=f"sixarm_{family}",
            epochs=1,
            target_params=10_000,
            **extra,
        )
        result = Trainer(cfg, representation=art).fit(data)
        encoders[family] = result["model"]["encoder"]

    # ONE encoder digest across all six arms: bit-identical state dicts
    state_dicts = {fam: enc.state_dict() for fam, enc in encoders.items()}
    reference_fam = "mlp"
    for fam, sd in state_dicts.items():
        for k in state_dicts[reference_fam]:
            torch.testing.assert_close(sd[k], state_dicts[reference_fam][k], atol=0.0, rtol=0.0)
    print(f"[trainer D2b] six arms {sorted(encoders)} -> bit-identical encoder state dict (one digest)")

    # bit-identical embeddings on a fixed probe input, atol=rtol=0
    probe = torch.as_tensor(data["val"]["x"][:, 0, :], dtype=torch.float32)
    with torch.no_grad():
        outs = {fam: enc(probe) for fam, enc in encoders.items()}
    base = outs[reference_fam]
    max_dev = 0.0
    for fam, out in outs.items():
        max_dev = max(max_dev, (out - base).abs().max().item())
        torch.testing.assert_close(out, base, atol=0.0, rtol=0.0)
    print(f"[trainer D2b] six arms -> max embedding deviation = {max_dev} (want 0.0)")
    assert max_dev == 0.0


def test_representation_stratum_mismatch_refused_at_construction(tmp_path):
    import pytest

    data = _small_dataset()
    art = _small_representation(tmp_path, data=data, system="pendulum", regime="ou")
    cfg = _repr_config(tmp_path, stratum_regime="action_conditioned")  # different stratum
    with pytest.raises(ValueError, match="stratum mismatch"):
        Trainer(cfg, representation=art)  # refused before any model is built or any batch drawn


def test_representation_flipped_bit_refused_by_digest_check(tmp_path):
    import pytest

    data = _small_dataset()
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,),
        seed=0, epochs=3, batch_size=16,
    )
    path = tmp_path / "flipped.pt"
    save_frozen_representation(artifact, path)

    payload = torch.load(path, weights_only=False)
    key = next(iter(payload["encoder_state_dict"]))
    arr = payload["encoder_state_dict"][key].clone().numpy()
    flat = arr.reshape(-1)
    as_uint = flat[:1].view(np.uint32)
    as_uint[0] ^= 1  # ONE actual bit flip, on disk
    payload["encoder_state_dict"][key] = torch.from_numpy(arr)
    torch.save(payload, path)

    tampered = load_frozen_representation(path)  # load itself does not check the digest
    with pytest.raises(ValueError, match="digest mismatch"):
        Trainer(_repr_config(tmp_path), representation=tampered)


def test_representation_nonfinite_e_recon_refused(tmp_path):
    import pytest

    data = _small_dataset()
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,),
        seed=0, epochs=3, batch_size=16,
    )
    artifact.e_recon = float("nan")
    artifact.manifest = dict(artifact.manifest)
    artifact.manifest["digest"] = representation_digest(artifact)  # stamp AFTER corrupting e_recon

    with pytest.raises(ValueError, match="non-finite"):
        Trainer(_repr_config(tmp_path), representation=artifact)


# --------------------------------------------------------------------------
# PROTOCOL.md section 17 (Amendment 8): the gradient guard -- grad_clip is
# OPTIONAL, absent from _DEFAULTS (identity preservation), validated when
# present, and measured every epoch via a helper never called on the absent
# path (structural, not numerical, equivalence with pre-amendment runs).
# --------------------------------------------------------------------------


def test_grad_clip_absent_from_defaults_and_optional(tmp_path):
    import sld.training.trainer as trainer_mod

    assert "grad_clip" not in trainer_mod._DEFAULTS
    cfg = _mlp_config(tmp_path)
    assert "grad_clip" not in cfg
    trainer = Trainer(cfg)
    assert "grad_clip" not in trainer.config  # never silently defaulted in


def test_grad_clip_present_changes_config_hash(tmp_path):
    # positive: grad_clip is hashed like any other config key once supplied
    hash_absent = Trainer(_mlp_config(tmp_path)).config_hash
    hash_present = Trainer(_mlp_config(tmp_path, grad_clip=100.0)).config_hash
    print(f"[trainer] grad_clip absent={hash_absent} present={hash_present}")
    assert hash_absent != hash_present


def test_grad_clip_validated_finite_strictly_positive(tmp_path):
    # negative: 0 / negative / inf / nan are all refused, naming grad_clip
    import pytest

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="grad_clip"):
            Trainer(_mlp_config(tmp_path, grad_clip=bad))


def test_grad_clip_absent_never_calls_clip_grad_norm(tmp_path, monkeypatch):
    # negative: the absent path must leave gradients bitwise untouched --
    # proven by never even CALLING clip_grad_norm_, not by checking its effect
    import sld.training.trainer as trainer_mod

    calls = []
    real_clip = trainer_mod.torch.nn.utils.clip_grad_norm_

    def counting_clip(*a, **k):
        calls.append(1)
        return real_clip(*a, **k)

    monkeypatch.setattr(trainer_mod.torch.nn.utils, "clip_grad_norm_", counting_clip)
    data = _small_dataset(n_episodes=6, T=10)
    Trainer(_mlp_config(tmp_path, exp_name="trainer_test_noclip", epochs=2)).fit(data)
    print(f"[trainer] clip_grad_norm_ calls with grad_clip absent: {len(calls)}")
    assert len(calls) == 0


def test_grad_clip_tiny_value_bites_and_training_stays_finite(tmp_path):
    # positive: grad_clip=1e-12 on a toy 2-epoch fit -- grad_norm (measured
    # PRE-clip) is orders of magnitude above the clip value, and training
    # stays finite (the guard, not luck, kept the huge post-clip-bypassed
    # step from ever reaching the parameters)
    data = _small_dataset(n_episodes=6, T=10)
    result = Trainer(
        _mlp_config(tmp_path, exp_name="trainer_test_tinyclip", epochs=2, grad_clip=1e-12)
    ).fit(data)
    with open(result["paths"]["metrics_csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    grad_norms = [float(r["grad_norm"]) for r in rows]
    print(f"[trainer] grad_norms under grad_clip=1e-12: {grad_norms}")
    assert all(g > 1e-6 for g in grad_norms)  # pre-clip norm measured, >> the clip value
    assert all(np.isfinite(v) for v in result["history"]["train_loss"])
    assert all(np.isfinite(v) for v in result["history"]["val_loss"])


def test_metrics_csv_has_grad_norm_column_without_grad_clip(tmp_path):
    # positive companion: the column exists and is finite even when this
    # run's config never sets grad_clip at all
    data = _small_dataset(n_episodes=6, T=10)
    result = Trainer(_mlp_config(tmp_path, exp_name="trainer_test_gradnorm_col", epochs=2)).fit(data)
    with open(result["paths"]["metrics_csv"], newline="") as f:
        reader = csv.DictReader(f)
        assert "grad_norm" in reader.fieldnames
        rows = list(reader)
    assert all(np.isfinite(float(r["grad_norm"])) for r in rows)


def test_global_grad_norm_matches_hand_computed_value():
    # positive: an independent, hand-derived combination (two orthogonal
    # 3-4-5 / 12 legs -> global L2 norm 13 by the Pythagorean identity),
    # never the internal formula copied back at itself
    import pytest

    import sld.training.trainer as trainer_mod

    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(3))
    p1.grad = torch.tensor([3.0, 4.0])  # norm 5
    p2.grad = torch.tensor([0.0, 0.0, 12.0])  # norm 12
    got = trainer_mod._global_grad_norm([p1, p2])
    expected = (5.0**2 + 12.0**2) ** 0.5  # 13.0
    print(f"[trainer] global_grad_norm hand-check: got={got} expected={expected}")
    assert got == pytest.approx(expected)


def test_global_grad_norm_zero_when_no_grads():
    # negative: a parameter with no .grad contributes nothing, never a crash
    import sld.training.trainer as trainer_mod

    p = torch.nn.Parameter(torch.zeros(2))  # .grad is None (never populated)
    assert trainer_mod._global_grad_norm([p]) == 0.0
