"""Tests for ``sld.experiments.loading`` and ``sld.viz`` (fig1/fig2 v0).

Builds a MINIATURE run directory + dataset entirely under ``tmp_path`` --
no ``Trainer``, no real training loop. v0 viz only needs a well-formed run
artifact (config.yaml/metrics.csv/ckpt.pt/index.csv, REGLES section 6
contract) to load and plot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml

from sld.audit.jacobian import fr_drift_curve
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.experiments.loading import (
    build_dataset_from_expconfig,
    check_run_matches_dataset,
    data_hash_of,
    load_run,
    read_index,
)
from sld.predictors.registry import get_predictor
from sld.viz.fig1_rollout import _aggregate_curves, make_fig1
from sld.viz.fig2_spectrum import _plot_family_panel, make_fig2

D_X, D_Z, D_U = 2, 4, 1
TARGET_PARAMS = 1000


def _make_run_dir(
    tmp_path: Path,
    family: str = "mlp",
    constraint: str = "on",
    seed: int = 0,
    data_hash: str = "00000000",
) -> tuple[Path, dict]:
    # run dir name mirrors Trainer.resolve_run_dir's real shape (a trailing
    # 8-hex-char data_hash suffix) so check_run_matches_dataset (F6) can be
    # exercised the same way it is against a real grid
    run_dir = tmp_path / "results" / "pendulum_core" / f"{family}-{constraint}-{seed}-{data_hash}"
    (run_dir / "figs").mkdir(parents=True)

    ambient = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        encoder = FrameMLPEncoder(D_X, D_Z, (8, 8))
        predictor = get_predictor(family, D_Z, D_U, target_params=TARGET_PARAMS)
    finally:
        torch.set_default_dtype(ambient)
    encoder.eval()
    predictor.eval()

    config = {
        "exp_name": "pendulum_core",
        "seed": seed,
        "family": family,
        "d_x": D_X,
        "d_z": D_Z,
        "d_u": D_U,
        "encoder_hidden": [8, 8],
        "target_params": TARGET_PARAMS,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "pred", "consistency", "var", "cov"])
        writer.writerow([0, 0.5, 0.4, 1e-3, 0.3, 0.0, 0.1, 0.05])

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "ema_target": encoder.state_dict(),  # placeholder, unused by load_run
            "config": config,
        },
        run_dir / "ckpt.pt",
    )
    (run_dir / "audit.json").write_text(json.dumps({}))
    return run_dir, {"encoder": encoder, "predictor": predictor, "config": config}


def _write_index(results_root: Path, exp_name: str, rows: list[list]) -> None:
    index_dir = results_root / exp_name
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "regime", "family", "structure", "seed", "status", "final_val_loss"])
        for row in rows:
            writer.writerow(row)


def _write_exp_yaml(path: Path, dt: float = 0.05, T: int = 10, n_episodes: int = 20, seed: int = 0) -> None:
    # n_episodes=20 with build_dataset's default split (0.8/0.1/0.1) gives a
    # non-empty val split (2 episodes); build_dataset_from_expconfig has no
    # split_fractions knob in the schema (matches pendulum_core.yaml), so the
    # fixture must clear the default rounding-to-zero at very small n
    config = {
        "datasets": {
            "ou": {
                "system": {"name": "DampedPendulum", "params": {}},
                "regime": {"name": "OURegime", "params": {}},
                "obs": None,
                "n_episodes": n_episodes,
                "T": T,
                "dt": dt,
                "seed": seed,
            }
        }
    }
    path.write_text(yaml.safe_dump(config, sort_keys=True))


def _ou_data_hash(yaml_path: Path) -> str:
    """The data_hash the CURRENT ``yaml_path``'s "ou" dataset resolves to
    (Trainer.resolve_run_dir's recipe, reused via data_hash_of) -- fixtures
    use this so their run dirs' names match what check_run_matches_dataset
    (F6) expects, exactly like a real ``sld-factorial`` grid."""
    return data_hash_of(build_dataset_from_expconfig(yaml_path, "ou")["meta"])


# -- load_run --------------------------------------------------------------


def test_load_run_roundtrip(tmp_path: Path) -> None:
    run_dir, built = _make_run_dir(tmp_path)
    loaded = load_run(run_dir)

    assert loaded["config"]["family"] == "mlp"
    for k, v in built["encoder"].state_dict().items():
        assert torch.equal(v, loaded["encoder"].state_dict()[k])
    for k, v in built["predictor"].state_dict().items():
        assert torch.equal(v, loaded["predictor"].state_dict()[k])
    assert not loaded["encoder"].training
    assert not loaded["predictor"].training


def test_load_run_missing_ckpt_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(run_dir)


# -- read_index --------------------------------------------------------------


def test_read_index_missing_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_index(tmp_path / "results", "pendulum_core")


def test_read_index_reads_rows(tmp_path: Path) -> None:
    # check positif (phase B4): an index.csv written TODAY carries the
    # canonical "structure" column (``_write_index`` above matches
    # ``factorial._write_index``'s real header) and reads back under that name.
    results_root = tmp_path / "results"
    run_dir, _ = _make_run_dir(tmp_path)
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])
    rows = read_index(results_root, "pendulum_core")
    assert len(rows) == 1
    assert rows[0]["seed"] == 0
    assert rows[0]["final_val_loss"] == 0.4
    assert rows[0]["status"] == "complete"
    assert rows[0]["structure"] == "on"


def test_read_index_accepts_legacy_constraint_column(tmp_path: Path) -> None:
    """Legacy check (phase B4): frozen pre-B4 traces (results/, results_amended/,
    reports/historical/mu_j/traces/) carry "constraint" instead of "structure"
    and are never rewritten -- read_index must still parse them, with the
    value landing at the canonical "structure" key."""
    results_root = tmp_path / "results"
    index_dir = results_root / "pendulum_core"
    index_dir.mkdir(parents=True)
    with open(index_dir / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "regime", "family", "constraint", "seed", "status", "final_val_loss"])
        writer.writerow(["/nonexistent", "ou", "mlp", "on", 0, "complete", 0.4])
    rows = read_index(results_root, "pendulum_core")
    assert len(rows) == 1
    assert rows[0]["structure"] == "on"


def test_read_index_rejects_index_missing_both_structure_and_constraint(tmp_path: Path) -> None:
    """Check négatif (phase B4): an index.csv carrying NEITHER "structure"
    nor the legacy "constraint" alias must be rejected with a clear error,
    never silently read with an empty/missing field."""
    results_root = tmp_path / "results"
    index_dir = results_root / "pendulum_core"
    index_dir.mkdir(parents=True)
    with open(index_dir / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "regime", "family", "seed", "status", "final_val_loss"])
        writer.writerow(["/nonexistent", "ou", "mlp", 0, "complete", 0.4])
    with pytest.raises(ValueError, match="structure"):
        read_index(results_root, "pendulum_core")


# -- build_dataset_from_expconfig --------------------------------------------


def test_build_dataset_from_expconfig(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    data = build_dataset_from_expconfig(yaml_path, "ou")
    assert data["train"]["x"].shape[-1] == D_X
    assert "meta" in data


def test_build_dataset_from_expconfig_unknown_regime_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    with pytest.raises(KeyError):
        build_dataset_from_expconfig(yaml_path, "does_not_exist")


# -- check_run_matches_dataset (F6) ------------------------------------------


def test_check_run_matches_dataset_ok(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)
    run_dir, _ = _make_run_dir(tmp_path, data_hash=dh)
    dataset = build_dataset_from_expconfig(yaml_path, "ou")
    check_run_matches_dataset(run_dir, dataset["meta"])  # must not raise


def test_check_run_matches_dataset_mismatch_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    run_dir, _ = _make_run_dir(tmp_path, data_hash="deadbeef")
    dataset = build_dataset_from_expconfig(yaml_path, "ou")
    with pytest.raises(ValueError, match="data_hash"):
        check_run_matches_dataset(run_dir, dataset["meta"])


def test_check_run_matches_dataset_bad_name_shape_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "not_a_hash_shaped_name"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="does not carry"):
        check_run_matches_dataset(run_dir, {"seed": 0})


# -- fig1: F1 gauge invariance ------------------------------------------------


def test_fr_drift_target_is_true_state_not_encoder_reconstruction() -> None:
    """Root-cause regression for the F1 metric-target bug in the now-
    retired per-run OLS readout (``fig1_rollout``'s old ``_fit_run_readout``
    -- see that module's docstring, phase D): it used to compare the
    rollout prediction against a READOUT-DECODED RECONSTRUCTION of the
    ENCODER'S OWN EMBEDDING instead of the dataset's actual ground-truth
    physical state (``z_true``), so a predictor that merely reproduces the
    encoder's own drift away from reality read as zero error ("perfect").

    SELF-CONTAINED (no dependency on any symbol removed with the retired
    readout): redoes a minimal inline OLS fit -- exactly what
    ``_fit_readout`` used to do, reduced to its essential lines -- against a
    deliberately NONLINEAR (quadratic) encoder distortion of the true
    state, ``z_enc = z_true + 0.2 * z_true**2``, a known discrepancy an
    affine readout cannot cancel (so the correct-target curve must be
    nonzero). A "cheat" predictor copies the encoder's own future
    embeddings exactly; decoded against the OLD buggy target (a decode of
    that same copied array) the curve collapses to a false zero.
    """
    T = 30
    t = np.arange(T + 1, dtype=np.float64)
    z0 = 0.5 * np.sin(0.3 * t)
    z1 = 0.5 * np.cos(0.2 * t) + 0.1 * t / T  # not collinear with z0 or the intercept
    z_true = np.stack([z0, z1], axis=-1)  # (T+1, d_state)

    # the encoder's reconstruction differs from the true state by this
    # KNOWN, deliberately nonlinear amount -- no affine readout can cancel it
    z_enc = z_true + 0.2 * z_true**2  # (T+1, d_state)

    # minimal inline OLS readout -- what the retired _fit_readout did,
    # reduced to its 3 essential lines (no bias/rank/cond bookkeeping)
    design = np.concatenate([z_enc, np.ones((T + 1, 1))], axis=1)
    W, *_ = np.linalg.lstsq(design, z_true, rcond=None)

    def decode(z: np.ndarray) -> np.ndarray:
        return np.concatenate([z, np.ones((len(z), 1))], axis=1) @ W

    H = 10
    # a "cheat" predictor whose rollout is EXACTLY the encoder's own future
    # embeddings, undecoded -- the exact scenario the bug hid
    pred_z = z_enc[1 : H + 1]
    pred_state = decode(pred_z)[:, None, :]  # (H, B=1, d_state)

    target_true = z_true[1 : H + 1][:, None, :]  # correct target: dataset ground truth
    target_old_buggy = decode(pred_z)[:, None, :]  # OLD buggy target: decode of the same array

    err_new = fr_drift_curve(pred_state, target_true)
    err_old = fr_drift_curve(pred_state, target_old_buggy)

    # negative: the old buggy target is EXACTLY pred_state (the identical
    # array decoded by the identical W), so its curve collapses to a FALSE
    # ZERO at the project's declared 1e-12 tolerance -- this is what the old
    # code would have called "perfect", even though the encoder demonstrably
    # does not track reality
    assert np.allclose(err_old, 0.0, atol=1e-12)

    # positive: the fixed target must not report perfection -- it follows
    # z_true, and the nonlinear encoder distortion is not exactly cancelled
    # by the affine readout, so the error is both nonzero and non-negligible
    assert np.all(err_new > 1e-3)


def test_make_fig1_raises_pending_phase_d() -> None:
    """Fig. 1's per-run OLS readout is retired (a free parameter per arm,
    not an at-arms-length comparison method) pending phase D's shared,
    frozen readout; a wrong figure is worse than no figure, so
    ``make_fig1`` now refuses to render instead of plotting on the
    superseded method."""
    with pytest.raises(NotImplementedError, match="phase D"):
        make_fig1()


# -- fig1 / fig2 --------------------------------------------------------------


def test_make_fig2_produces_png(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=0, data_hash=dh)
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])

    out_path = tmp_path / "assets" / "fig2_spectrum.png"
    make_fig2(
        results_root=str(results_root),
        exp_name="pendulum_core",
        exp_yaml=str(yaml_path),
        out_path=str(out_path),
        require_full_cells=False,  # only 'mlp' present -- explicit opt-out (F12)
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_make_fig2_missing_family_raises_in_strict_mode(tmp_path: Path) -> None:
    """F12: a family with no cell run at all aborts the render in strict
    (default) mode, naming the family, instead of shipping a blank panel."""
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=0, data_hash=dh)
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])

    with pytest.raises(ValueError, match=r"family 'koopman_ssm' has no regime='ou'/seed=0 run"):
        make_fig2(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig2.png"),
        )


def test_make_fig2_dataset_mismatch_raises(tmp_path: Path) -> None:
    """F6: same hard-failure guarantee as fig1, for fig2's per-family cell."""
    results_root = tmp_path / "results"
    run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=0, data_hash="deadbeef")
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)

    with pytest.raises(ValueError, match="data_hash"):
        make_fig2(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig2.png"),
            require_full_cells=False,
        )


def test_make_fig2_panel_titles_show_provenance(tmp_path: Path, monkeypatch) -> None:
    """F12: every rendered panel names its family, constraint, and seed."""
    captured: dict = {}

    def _capture(fig, path):
        captured["fig"] = fig
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    monkeypatch.setattr("sld.viz.fig2_spectrum.save_fig", _capture)

    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=0, data_hash=dh)
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])

    make_fig2(
        results_root=str(results_root),
        exp_name="pendulum_core",
        exp_yaml=str(yaml_path),
        out_path=str(tmp_path / "assets" / "fig2.png"),
        require_full_cells=False,
    )

    fig = captured["fig"]
    titles = [ax.get_title() for ax in fig.axes]
    mlp_title = next(t for t in titles if t.startswith("mlp"))
    assert "constraint=on" in mlp_title
    assert "seed=0" in mlp_title


# -- round-2 locks: M5 strict fallback (fig2), M6/m12 aggregation -----------


def test_fig2_strict_mode_refuses_off_substitution_for_missing_on() -> None:
    """M5 (round 2): require_full_cells=True must REFUSE the on->off
    fallback -- an OFF-only family used to survive strict mode by silent
    substitution, contradicting the completeness guarantee."""
    fig, ax = plt.subplots()
    rows = [
        {"run_dir": "/nonexistent", "regime": "ou", "family": "mlp",
         "structure": "off", "seed": 0, "status": "complete", "final_val_loss": 0.4},
    ]
    with pytest.raises(ValueError, match="refusing to silently substitute"):
        _plot_family_panel(ax, "mlp", rows, exp_yaml="unused.yaml", require_full_cells=True)
    plt.close(fig)


def test_aggregate_curves_numeric_lock() -> None:
    """M6/m12 (round 2): the seed aggregation is the MEAN +/- 1 SE -- for
    seed curves [1, 1, 1, 1, 100] the center is 20.8 (the reviewer's
    reproduction: a median would say 1), the half-band is
    std(ddof=1)/sqrt(5). This is the numeric lock the old shape-only test
    lacked (it did not catch the mean/median contradiction)."""
    curves = np.array([[1.0], [1.0], [1.0], [1.0], [100.0]]) * np.ones((5, 3))
    mean_curve, se = _aggregate_curves(curves)
    assert np.allclose(mean_curve, 20.8)
    expected_se = np.std([1.0, 1.0, 1.0, 1.0, 100.0], ddof=1) / np.sqrt(5)
    assert np.allclose(se, expected_se)
    # n=1: SE is exactly 0 (no spread observed), never NaN
    m1, se1 = _aggregate_curves(np.ones((1, 3)))
    assert np.allclose(m1, 1.0) and np.allclose(se1, 0.0)


def test_expected_grid_derives_from_yaml_arms_and_seeds(tmp_path: Path) -> None:
    """m7 (round 3): the strict-mode expected grid comes from the yaml's OWN
    arms/seeds declarations when present -- smoke.yaml's 2 arms x 1 seed,
    never an invented style-domain cross product. Fallback (no arms):
    style domains with the pre-registered 5 seeds."""
    from sld.viz.fig1_rollout import _expected_grid

    yaml_path = tmp_path / "exp.yaml"
    config = {
        "datasets": {"ou": {"system": {"name": "DampedPendulum", "params": {}},
                             "regime": {"name": "OURegime", "params": {}},
                             "obs": None, "n_episodes": 8, "T": 5, "dt": 0.05, "seed": 0}},
        "arms": [
            {"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}},
            {"family": "koopman_ssm", "constraint": "on", "overrides": {"mu_j": 0.01}},
        ],
        "seeds": [0],
    }
    yaml_path.write_text(yaml.safe_dump(config))
    cells, seeds = _expected_grid(str(yaml_path))
    assert cells == [("ou", "mlp", "off"), ("ou", "koopman_ssm", "on")]
    assert seeds == frozenset({0})

    # fallback: no arms declared -> style domains x pre-registered seeds
    _write_exp_yaml(tmp_path / "noarms.yaml")
    cells_fb, seeds_fb = _expected_grid(str(tmp_path / "noarms.yaml"))
    assert len(cells_fb) == 8  # 1 regime x 4 families x 2 constraints
    assert seeds_fb == frozenset(range(5))
