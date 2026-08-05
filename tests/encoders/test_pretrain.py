"""Tests for sld.encoders.pretrain: the frozen shared representation
(PROTOCOL.md sections 12.1, 12.3, 14.10 -- 14.10 supersedes 14.9's loss).
Wave D2a/D2a-v2.

Small/fast by design: tiny dims, epochs<=3 for structural tests, T<=20,
n_episodes<=8 -- mirrors tests/training/test_trainer.py's fixture discipline.
The collapse/alignment/teacher-drift tests below need more epochs than the
structural tests to show a real effect and use a stated, higher epoch count
locally -- still small enough to run in a few seconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.encoders.pretrain import (
    FrozenRepresentationArtifact,
    apply_readout,
    fit_readout,
    load_frozen_representation,
    pretrain_representation,
    representation_digest,
    save_frozen_representation,
    validate_shared_representation,
)
from sld.encoders.pretrain import _build_two_views, _tau1_condition_holds, _train_step, _VIEW_OFFSET
from sld.audit.rollout import fit_train_state_scale, normalized_state_error, persistence_rollout
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.systems.damped_pendulum import DampedPendulum
from sld.training.seeding import set_seed

_D_X = 2  # DampedPendulum.state_dim; obs=None below so x == z_true
_D_Z = 4


def _rebuild_encoder(artifact: FrozenRepresentationArtifact) -> FrameMLPEncoder:
    encoder = FrameMLPEncoder(
        artifact.manifest["d_x"], artifact.manifest["d_z"], tuple(artifact.manifest["encoder_hidden"])
    )
    encoder.load_state_dict(artifact.encoder_state_dict)
    encoder.eval()
    return encoder


def _embedding_std(artifact: FrozenRepresentationArtifact, data: dict) -> torch.Tensor:
    encoder = _rebuild_encoder(artifact)
    x_train = torch.as_tensor(data["train"]["x"], dtype=torch.float32).reshape(-1, _D_X)
    with torch.no_grad():
        z = encoder(x_train)
    return z.std(dim=0)


def _train_small_fixture(
    *,
    epochs: int = 20,
    w_var: float = 1.0,
    w_cov: float = 1.0,
    w_recon: float = 1.0,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 0,
    data: dict | None = None,
):
    """Builds the same modules pretrain_representation builds, then drives
    them through the REAL call site (``_train_step``, the exact function
    ``pretrain_representation`` calls -- not a hand-duplicated loss
    expression) so tests inspect intermediate state -- per-epoch alignment
    loss, the teacher network, gradients -- that the public function's
    return value (the frozen artifact) does not expose, WITHOUT losing
    coverage of the actual production wiring (REGLES section 7: verify the
    call site).

    Returns: (student_encoder, ema, epoch_align_means, initial_encoder_state_dict).
    """
    if data is None:
        data = _small_dataset(seed=0)
    train = data["train"]
    set_seed(seed)
    encoder = FrameMLPEncoder(_D_X, _D_Z, (8,))
    readout_head = torch.nn.Linear(_D_Z, _D_X)
    initial_state = {k: v.clone() for k, v in encoder.state_dict().items()}
    ema = EMATarget(encoder, tau=0.99)
    jepa = JEPALoss(lambda_c=0.0)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(readout_head.parameters()), lr=lr
    )

    x_view0, x_view1, x_true_view0 = _build_two_views(train, _D_X, _D_X)
    n = x_view0.shape[0]

    epoch_aligns = []
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue
            metrics = _train_step(
                encoder,
                readout_head,
                ema,
                jepa,
                optimizer,
                x_view0[idx],
                x_view1[idx],
                x_true_view0[idx],
                w_align=1.0,
                w_var=w_var,
                w_cov=w_cov,
                w_recon=w_recon,
            )
            batch_losses.append(metrics["align"])
        epoch_aligns.append(sum(batch_losses) / len(batch_losses))
    return encoder, ema, epoch_aligns, initial_state


def _small_dataset(n_episodes: int = 8, T: int = 20, seed: int = 0) -> dict:
    system = DampedPendulum()
    return build_dataset(
        system=system, regime=OURegime(), obs=None, n_episodes=n_episodes, T=T, dt=0.02, seed=seed
    )


def _make_small_artifact(
    *, seed: int = 0, system: str = "pendulum", regime: str = "ou", data: dict | None = None
) -> FrozenRepresentationArtifact:
    if data is None:
        data = _small_dataset()
    return pretrain_representation(
        data,
        system=system,
        regime=regime,
        d_z=_D_Z,
        encoder_hidden=(8,),
        seed=seed,
        epochs=3,
        batch_size=16,
    )


# -- pretrain_representation: measured values -------------------------------


def test_e_recon_finite_and_gate_verdict_recorded_either_way():
    """PROTOCOL.md section 12.3's pre-registered condition (3*e_recon <
    e_1_pers) is computed and recorded whichever way it falls -- never
    raised, never adjusted (section 14.9)."""
    artifact = _make_small_artifact(seed=0)
    assert math.isfinite(artifact.e_recon)
    assert math.isfinite(artifact.manifest["e_1_pers"])
    holds = 3.0 * artifact.e_recon < artifact.manifest["e_1_pers"]
    assert artifact.manifest["gated"] == holds
    print(
        f"[pretrain D2a] e_recon={artifact.e_recon:.6e} e_1_pers={artifact.manifest['e_1_pers']:.6e} "
        f"3*e_recon={3 * artifact.e_recon:.6e} gated={artifact.manifest['gated']} "
        f"readout r2={artifact.readout.r2:.6f} rank={artifact.readout.rank} "
        f"cond={artifact.readout.cond:.6e} indeterminate={artifact.readout.indeterminate}"
    )


def test_tau1_condition_is_strict():
    """The 3*e_recon < e_1_pers comparison is STRICT -- exact equality does
    NOT hold (mutation campaign target: a non-strict '<=' must redden this)."""
    assert _tau1_condition_holds(1.0, 3.0 + 1e-9) is True
    assert _tau1_condition_holds(1.0, 3.0) is False
    assert _tau1_condition_holds(1.0, 3.0 - 1e-9) is False


def test_pretrain_deterministic_for_same_seed():
    """Same dataset + same pretraining seed -> bit-identical artifact
    (config + seed = run identity, Trainer.fit's own discipline)."""
    data = _small_dataset(seed=0)
    kwargs = dict(
        system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=7, epochs=3, batch_size=16
    )
    a1 = pretrain_representation(data, **kwargs)
    a2 = pretrain_representation(data, **kwargs)
    d1, d2 = representation_digest(a1), representation_digest(a2)
    print(f"[pretrain determinism] digest1={d1} digest2={d2}")
    assert d1 == d2


def test_readout_fit_matches_independent_train_refit():
    """R2 (independent reference): recompute the TRAIN-split embeddings
    from the artifact's own frozen encoder, independently, and refit the
    readout with the same public `fit_readout` -- must match the artifact's
    stored readout exactly. This is the test the mutation campaign's
    'fit the readout on the held-out split instead of train' targets: had
    pretrain_representation fit on val instead, this train-based refit
    would disagree."""
    data = _small_dataset(seed=0)
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=0, epochs=3, batch_size=16
    )

    encoder = FrameMLPEncoder(artifact.manifest["d_x"], artifact.manifest["d_z"], tuple(artifact.manifest["encoder_hidden"]))
    encoder.load_state_dict(artifact.encoder_state_dict)
    encoder.eval()

    x_train = torch.as_tensor(data["train"]["x"], dtype=torch.float32).reshape(-1, _D_X)
    x_true_train = np.asarray(data["train"]["z_true"], dtype=np.float64).reshape(-1, _D_X)
    with torch.no_grad():
        z_train_np = encoder(x_train).double().numpy()

    independent = fit_readout(z_train_np, x_true_train)

    np.testing.assert_allclose(independent.W, artifact.readout.W, atol=1e-10, rtol=1e-8)
    assert independent.rank == artifact.readout.rank
    assert independent.indeterminate == artifact.readout.indeterminate
    assert independent.r2 == pytest.approx(artifact.readout.r2, abs=1e-10)


def test_e_recon_matches_independent_val_computation():
    """R2 (independent reference): recompute e_recon on the VAL split from
    scratch using the artifact's own readout, via the same public
    normalized_state_error -- must match the stored e_recon. Mutation
    campaign target: 'measure e_recon on the fitting split instead of
    held-out' -- had e_recon been measured on train, this val-based
    recomputation would disagree."""
    data = _small_dataset(seed=0)
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=0, epochs=3, batch_size=16
    )

    encoder = FrameMLPEncoder(artifact.manifest["d_x"], artifact.manifest["d_z"], tuple(artifact.manifest["encoder_hidden"]))
    encoder.load_state_dict(artifact.encoder_state_dict)
    encoder.eval()

    x_val = torch.as_tensor(data["val"]["x"], dtype=torch.float32).reshape(-1, _D_X)
    x_true_val = np.asarray(data["val"]["z_true"], dtype=np.float64).reshape(-1, _D_X)
    with torch.no_grad():
        z_val_np = encoder(x_val).double().numpy()
    pred_val = apply_readout(z_val_np, artifact.readout.W)
    e_recon_independent = float(
        normalized_state_error(pred_val[:, None, :], x_true_val[:, None, :], artifact.sigma_train)[0]
    )
    assert e_recon_independent == pytest.approx(artifact.e_recon, abs=1e-10)


def test_e_1_pers_matches_independent_persistence_computation():
    """R2: e_1_pers recomputed independently via persistence_rollout +
    normalized_state_error on the val split's own (x0, x1) pairs."""
    data = _small_dataset(seed=0)
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=0, epochs=3, batch_size=16
    )
    sigma_train = fit_train_state_scale(data["train"]["z_true"])
    x0 = np.asarray(data["val"]["z_true"][:, 0, :], dtype=np.float64)
    x1 = np.asarray(data["val"]["z_true"][:, 1, :], dtype=np.float64)
    pers_pred = persistence_rollout(x0, horizon=1)
    e_1_pers_independent = float(normalized_state_error(pers_pred, x1[:, None, :], sigma_train)[0])
    assert e_1_pers_independent == pytest.approx(artifact.manifest["e_1_pers"], abs=1e-10)


# -- fit_readout / apply_readout: the recovered readout ----------------------


def test_fit_readout_gauge_invariance_extreme_diagonal():
    """The property the recovery exists to preserve: fitted values and the
    determinability verdict are unchanged under ANY diagonal gauge, from
    1e-20 to 1e20. A raw (non-equilibrated) lstsq loses rank at these
    scales despite the target being exactly affine -- mutation campaign
    target: solving on the raw design instead of the equilibrated one MUST
    redden this test."""
    rng = np.random.default_rng(17)
    z = rng.normal(size=(128, 2))
    affine = np.array([[1.5, -0.75], [0.25, 2.0], [0.2, -0.4]])
    x_true = np.concatenate([z, np.ones((len(z), 1))], axis=1) @ affine

    baseline = fit_readout(z, x_true)
    assert baseline.rank == 3
    assert not baseline.indeterminate
    assert baseline.r2 == pytest.approx(1.0, abs=1e-12)

    gauges = {
        "1e-20": np.diag([1e-20, 1.0]),
        "1e20": np.diag([1e20, 1.0]),
        "2**20_uniform": np.diag([2.0**20, 2.0**20]),
        "diag(1e9,1)": np.diag([1e9, 1.0]),
    }
    for name, gauge in gauges.items():
        z_scaled = z @ gauge.T
        fit = fit_readout(z_scaled, x_true)
        assert fit.rank == baseline.rank, name
        assert fit.indeterminate == baseline.indeterminate, name
        assert fit.r2 == pytest.approx(baseline.r2, abs=1e-10), name
        decoded = apply_readout(z_scaled, fit.W)
        max_err = np.max(np.abs(decoded - x_true))
        print(f"[gauge invariance] {name}: rank={fit.rank} r2={fit.r2:.12f} max_decode_err={max_err:.3e}")
        assert max_err < 1e-8, name


def test_fit_readout_rank_deficient_is_indeterminate():
    """Positive/negative pair (REGLES 7.1) for fit_readout's determinability
    verdict: a collapsed latent coordinate (z2 = z1, a legal outcome) makes
    the design column-rank-deficient -- classified indeterminate; a
    full-rank, well-conditioned control is not."""
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(64, 1))
    z_collapsed = np.concatenate([z0, z0], axis=1)
    x_true = rng.normal(size=(64, 2))
    fit = fit_readout(z_collapsed, x_true)
    assert fit.indeterminate
    assert fit.rank < 3

    z_ok = rng.normal(size=(64, 2))
    fit_ok = fit_readout(z_ok, z_ok @ rng.normal(size=(2, 2)) + 0.01 * rng.normal(size=(64, 2)))
    assert not fit_ok.indeterminate


# -- digest: pure function of content ----------------------------------------


def test_digest_flips_on_one_bit_of_encoder():
    artifact = _make_small_artifact(seed=0)
    d1 = representation_digest(artifact)

    key = next(iter(artifact.encoder_state_dict))
    arr = artifact.encoder_state_dict[key].clone().numpy()
    flat = arr.reshape(-1)
    as_uint = flat[:1].view(np.uint32)
    as_uint[0] ^= 1  # exactly one bit
    artifact.encoder_state_dict[key] = torch.from_numpy(arr)

    d2 = representation_digest(artifact)
    print(f"[digest] before={d1} after_one_bit_flip={d2}")
    assert d1 != d2


def test_digest_identical_for_identical_content():
    data = _small_dataset(seed=0)
    kwargs = dict(
        system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=3, epochs=3, batch_size=16
    )
    a1 = pretrain_representation(data, **kwargs)
    a2 = pretrain_representation(data, **kwargs)
    assert representation_digest(a1) == representation_digest(a2)


# -- save / load: atomicity, round-trip, six loads ---------------------------


def test_six_loads_bit_identical_digest_and_embeddings(tmp_path):
    artifact = _make_small_artifact(seed=0)
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)

    fixed_x = torch.randn(5, _D_X, generator=torch.Generator().manual_seed(123))
    digests = set()
    embeddings = []
    for _ in range(6):
        loaded = load_frozen_representation(path)
        digests.add(representation_digest(loaded))
        encoder = FrameMLPEncoder(
            loaded.manifest["d_x"], loaded.manifest["d_z"], tuple(loaded.manifest["encoder_hidden"])
        )
        encoder.load_state_dict(loaded.encoder_state_dict)
        encoder.eval()
        with torch.no_grad():
            embeddings.append(encoder(fixed_x))

    max_dev = 0.0
    for e in embeddings[1:]:
        torch.testing.assert_close(e, embeddings[0], atol=0.0, rtol=0.0)
        max_dev = max(max_dev, (e - embeddings[0]).abs().max().item())
    print(f"[six loads] digest_count={len(digests)} max_embedding_deviation={max_dev:.3e}")
    assert len(digests) == 1


def test_round_trip_preserves_dtype_device_split(tmp_path):
    artifact = _make_small_artifact(seed=0)
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)
    loaded = load_frozen_representation(path)

    assert loaded.manifest["dtype"] == artifact.manifest["dtype"] == "torch.float32"
    assert loaded.manifest["device"] == artifact.manifest["device"] == "cpu"
    assert loaded.manifest["fit_split"] == "train"
    assert loaded.manifest["eval_split"] == "val"
    for key, tensor in artifact.encoder_state_dict.items():
        torch.testing.assert_close(loaded.encoder_state_dict[key], tensor, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(loaded.readout.W, artifact.readout.W)
    assert loaded.e_recon == artifact.e_recon


def test_save_is_atomic_on_replace_failure(tmp_path, monkeypatch):
    """Mutation campaign target: dropping the tmp+os.replace atomicity means
    save() would write torch.save(payload, path) directly -- os.replace is
    then never called, no exception is raised, and this test's own
    pytest.raises block fails, catching the mutant."""
    path = tmp_path / "artifact.pt"
    artifact1 = _make_small_artifact(seed=0)
    save_frozen_representation(artifact1, path)
    digest1 = representation_digest(artifact1)

    artifact2 = _make_small_artifact(seed=1)

    import sld.encoders.pretrain as pretrain_mod

    def _boom(*_a, **_kw):
        raise OSError("simulated crash between tmp write and replace")

    monkeypatch.setattr(pretrain_mod.os, "replace", _boom)

    with pytest.raises(OSError, match="simulated crash"):
        save_frozen_representation(artifact2, path)

    reloaded = load_frozen_representation(path)
    assert representation_digest(reloaded) == digest1


# -- validate_shared_representation: the preflight ---------------------------


def test_validate_accepts_matching_stratum(tmp_path):
    artifact = _make_small_artifact(seed=0, system="pendulum", regime="ou")
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)
    loaded = load_frozen_representation(path)
    validate_shared_representation(loaded, system="pendulum", regime="ou")  # must not raise


def test_validate_refuses_flipped_bit_via_digest_mismatch(tmp_path):
    artifact = _make_small_artifact(seed=0)
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)

    payload = torch.load(path, weights_only=False)
    key = next(iter(payload["encoder_state_dict"]))
    arr = payload["encoder_state_dict"][key].clone().numpy()
    flat = arr.reshape(-1)
    as_uint = flat[:1].view(np.uint32)
    as_uint[0] ^= 1
    payload["encoder_state_dict"][key] = torch.from_numpy(arr)
    torch.save(payload, path)

    loaded = load_frozen_representation(path)  # load itself does not check the digest
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_shared_representation(loaded, system=artifact.manifest["system"], regime=artifact.manifest["regime"])


def test_validate_refuses_different_stratum(tmp_path):
    artifact = _make_small_artifact(seed=0, system="pendulum", regime="ou")
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)
    loaded = load_frozen_representation(path)
    with pytest.raises(ValueError, match="stratum mismatch"):
        validate_shared_representation(loaded, system="pendulum", regime="action_conditioned")


def test_validate_refuses_nonfinite_e_recon():
    artifact = _make_small_artifact(seed=0)
    artifact.e_recon = float("nan")
    artifact.manifest = dict(artifact.manifest)
    artifact.manifest["digest"] = representation_digest(artifact)  # stamp AFTER corrupting e_recon
    with pytest.raises(ValueError, match="non-finite"):
        validate_shared_representation(
            artifact, system=artifact.manifest["system"], regime=artifact.manifest["regime"]
        )


# -- load_frozen_representation: structural and self-consistency refusals ----


def test_load_refuses_truncated_file(tmp_path):
    artifact = _make_small_artifact(seed=0)
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(RuntimeError, match="corrupt or unreadable"):
        load_frozen_representation(path)


def test_load_refuses_missing_top_level_keys(tmp_path):
    path = tmp_path / "artifact.pt"
    torch.save({"encoder_state_dict": {}, "readout": {}}, path)
    with pytest.raises(RuntimeError, match="missing top-level keys"):
        load_frozen_representation(path)


def test_load_refuses_dtype_mismatch(tmp_path):
    artifact = _make_small_artifact(seed=0)
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)
    payload = torch.load(path, weights_only=False)
    payload["manifest"]["dtype"] = "torch.float64"  # actual tensors are float32
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        load_frozen_representation(path)


def test_load_refuses_device_mismatch(tmp_path):
    artifact = _make_small_artifact(seed=0)
    path = tmp_path / "artifact.pt"
    save_frozen_representation(artifact, path)
    payload = torch.load(path, weights_only=False)
    payload["manifest"]["device"] = "cuda:0"  # actual tensors are on cpu
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="device mismatch"):
        load_frozen_representation(path)


# -- section 14.10: two-view self-distillation, no predictor -----------------


def test_default_objective_does_not_collapse():
    """The collapse test that matters (PROTOCOL.md 14.10): under the
    SHIPPED default objective (align + var + cov + recon, all weight 1.0),
    the student embedding stays well-spread -- per-coordinate variance above
    a floor -- and two genuinely different physical states embed apart."""
    data = _small_dataset(seed=0)
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=0, epochs=15, batch_size=16
    )
    std = _embedding_std(artifact, data)
    print(f"[collapse] default objective per-coord std={std.tolist()} min={std.min().item():.5f}")
    assert std.min().item() > 0.15

    encoder = _rebuild_encoder(artifact)
    x_true = data["train"]["z_true"].reshape(-1, _D_X)
    x = data["train"]["x"].reshape(-1, _D_X)
    dmat = np.linalg.norm(x_true[:, None, :] - x_true[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
    with torch.no_grad():
        za = encoder(torch.as_tensor(x[i : i + 1], dtype=torch.float32))
        zb = encoder(torch.as_tensor(x[j : j + 1], dtype=torch.float32))
    dist = (za - zb).norm().item()
    print(f"[collapse] distinct-states embedding dist={dist:.5f} (true-state dist={dmat[i, j]:.3f})")
    assert dist > 0.5


def test_variance_term_prevents_collapse_isolated_ablation():
    """PROTOCOL.md 14.10's own citation: predictor-free BYOL with nothing
    else collapses to a constant -- VICReg's variance term is what this
    project substitutes for BYOL's predictor head. Isolated here with
    w_recon=0 so the reconstruction term's OWN (separate, genuine --
    see test_default_objective_does_not_collapse above) anti-collapse
    effect cannot mask the variance term's specific contribution. A higher
    lr and more epochs than the structural tests are used because collapse
    is a training-DYNAMICS property that needs enough optimization steps to
    show up -- still ~5s total, not the full-suite budget.

    This is also the mutation campaign's item 1 ('remove the VICReg
    variance term and confirm the collapse test reddens'), exercised
    through the public w_var=0.0 knob -- functionally identical to deleting
    the term from the loss, no source edit needed -- and kept as a
    PERMANENT regression test rather than a transient mutation, since it is
    exactly the property section 14.10 asks to guard."""
    data = _small_dataset(seed=0)
    kwargs = dict(
        system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=0,
        epochs=400, batch_size=16, lr=1e-2, w_recon=0.0,
    )
    a_on = pretrain_representation(data, w_var=1.0, **kwargs)
    a_off = pretrain_representation(data, w_var=0.0, **kwargs)
    std_on = _embedding_std(a_on, data)
    std_off = _embedding_std(a_off, data)
    print(
        f"[collapse ablation] w_var=1 min_std={std_on.min().item():.5f} "
        f"w_var=0 min_std={std_off.min().item():.5f}"
    )
    assert std_on.min().item() > 0.3  # baseline: healthy
    assert std_off.min().item() < 0.05  # mutant: collapsed


def test_alignment_term_decreases_when_isolated():
    """The alignment term is doing something -- gradient descent on it
    actually reduces it -- verified with align as the ONLY active term
    (w_var=w_cov=w_recon=0).

    NOT tested under the default, all-terms-on objective: measured
    separately (epochs=60, default weights, printed for the record), the
    alignment loss RISES monotonically from ~0.0014 to ~0.39 over training,
    not falls. This is the module docstring's declared tension made
    concrete: the reconstruction term needs the embedding to stay sensitive
    to the true-state DIFFERENCE between frame t and t+_VIEW_OFFSET (the
    pendulum genuinely moves between adjacent frames), while the alignment
    term wants that difference erased -- at equal weights (all 1.0) here,
    reconstruction's pull dominates and actively works against invariance,
    not merely fails to help it. Isolating the alignment term (this test)
    is what actually verifies the MECHANISM (student learns to track
    sg[teacher]) rather than the mechanism-plus-a-term-that-fights-it."""
    _, _, epoch_aligns_isolated, _ = _train_small_fixture(
        epochs=30, w_var=0.0, w_cov=0.0, w_recon=0.0
    )
    print(
        f"[alignment isolated] epoch0={epoch_aligns_isolated[0]:.6f} "
        f"epoch_last={epoch_aligns_isolated[-1]:.6f}"
    )
    assert epoch_aligns_isolated[-1] < epoch_aligns_isolated[0]

    _, _, epoch_aligns_default, _ = _train_small_fixture(epochs=60)
    print(
        f"[alignment default weights, for the record] epoch0={epoch_aligns_default[0]:.6f} "
        f"epoch_last={epoch_aligns_default[-1]:.6f} (rises -- see docstring)"
    )


def test_teacher_is_ema_not_identical_to_student_and_not_frozen_at_init():
    """The teacher must be a LIVE EMA: after training, it differs from the
    student (not a shared/tied network) AND differs from its own INITIAL
    state (not a plain copy that stopped updating)."""
    encoder, ema, _, initial_state = _train_small_fixture(epochs=20)
    max_diff_from_student = max(
        (p_s - p_t).abs().max().item()
        for p_s, p_t in zip(encoder.parameters(), ema.target.parameters())
    )
    max_diff_from_init = max(
        (initial_state[k] - p_t).abs().max().item() for k, p_t in ema.target.state_dict().items()
    )
    print(
        f"[teacher] max|teacher-student|={max_diff_from_student:.6e} "
        f"max|teacher-init|={max_diff_from_init:.6e}"
    )
    assert max_diff_from_student > 0.0  # not identical to the student
    assert max_diff_from_init > 0.0  # not frozen at construction (EMA actually moved)


def test_stop_gradient_on_teacher_is_real():
    """After a backward pass through _train_step -- the REAL call site
    pretrain_representation uses, not a hand-built loss expression -- the
    teacher's parameters carry NO gradient: sg[teacher] is real, not
    aspirational."""
    data = _small_dataset(seed=0)
    train = data["train"]
    x_view0, x_view1, x_true_view0 = _build_two_views(train, _D_X, _D_X)

    encoder = FrameMLPEncoder(_D_X, _D_Z, (8,))
    readout_head = torch.nn.Linear(_D_Z, _D_X)
    ema = EMATarget(encoder, tau=0.99)
    jepa = JEPALoss(lambda_c=0.0)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(readout_head.parameters()), lr=1e-3
    )

    _train_step(
        encoder, readout_head, ema, jepa, optimizer,
        x_view0[:8], x_view1[:8], x_true_view0[:8],
        w_align=1.0, w_var=1.0, w_cov=1.0, w_recon=1.0,
    )

    for p in ema.target.parameters():
        assert p.grad is None
    # AND the student side genuinely trained (grad WAS real there) --
    # otherwise "teacher grad is None" would be trivially true because
    # nothing trained at all
    assert any(p.grad is not None for p in encoder.parameters())


def test_pretrain_module_has_no_predictor_import():
    """Static, module-level guard against the mutation campaign's item 4
    ('reintroduce a predictor head'): pretrain.py must not import from
    sld.predictors -- a predictor head between the two branches would need
    a predictor class, and this is the cheapest place to catch that. Does
    NOT catch an inline, unnamed extra nn.Linear used as a de facto
    predictor head without importing sld.predictors -- see
    test_pretrain_constructs_exactly_the_expected_learnable_modules below
    for that case, and the report for the residual gap neither test closes.
    """
    import inspect

    import sld.encoders.pretrain as pretrain_mod

    source = inspect.getsource(pretrain_mod)
    assert "sld.predictors" not in source


def test_pretrain_constructs_exactly_the_expected_learnable_modules():
    """Structural guard against the mutation campaign's item 4: counts how
    many nn.Linear layers pretrain_representation constructs during one
    call. FrameMLPEncoder(d_x, d_z, hidden=(8,)) builds 2 Linear layers
    (d_x->8, 8->d_z); the training-time readout_head is 1 more Linear
    (d_z->d_state) -- 3 total. A predictor head inserted between the two
    branches needs at least one more learnable module; this test would
    catch that specific shape of the mutation. It would NOT catch a
    'predictor' built from zero new parameters (e.g. reusing an existing
    Linear or a parameter-free transform) -- see the report for this gap,
    stated plainly rather than silently.
    """
    import torch.nn as torch_nn

    counts = {"n": 0}
    real_init = torch_nn.Linear.__init__

    def _counting_init(self, *a, **kw):
        counts["n"] += 1
        return real_init(self, *a, **kw)

    data = _small_dataset(seed=0)
    orig_init = torch_nn.Linear.__init__
    torch_nn.Linear.__init__ = _counting_init
    try:
        pretrain_representation(
            data, system="pendulum", regime="ou", d_z=_D_Z, encoder_hidden=(8,), seed=0, epochs=1, batch_size=16
        )
    finally:
        torch_nn.Linear.__init__ = orig_init

    print(f"[predictor-free] nn.Linear() constructions during one call = {counts['n']}")
    assert counts["n"] == 3


def test_build_two_views_uses_a_genuine_nonzero_offset():
    """R2/R7 positive control for the mutation campaign's 'same frame for
    both views (offset 0)' item: hand-built frames with a KNOWN, exactly
    identifiable value per (episode, t) -- x[ep, t, 0] = ep*1000 + t -- so
    view1[i] - view0[i] must equal EXACTLY _VIEW_OFFSET, not merely
    'different'. Offset 0 (view1 == view0 verbatim) fails this outright, as
    does any wrong offset value. This is invisible to an end-to-end
    training test (var/cov/recon still operate and dominate the loss
    regardless of the offset) but is exactly what this function must get
    right."""
    n_ep, steps_per_ep = 3, 6
    x = np.zeros((n_ep, steps_per_ep, 1), dtype=np.float64)
    for ep in range(n_ep):
        for t in range(steps_per_ep):
            x[ep, t, 0] = ep * 1000 + t
    train = {"x": x, "z_true": x.copy()}

    x_view0, x_view1, x_true_view0 = _build_two_views(train, d_x=1, d_state=1)

    assert not torch.equal(x_view0, x_view1)
    torch.testing.assert_close(x_view1 - x_view0, torch.full_like(x_view0, float(_VIEW_OFFSET)))
    torch.testing.assert_close(x_view0, x_true_view0)
