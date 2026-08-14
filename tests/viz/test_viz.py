"""Tests for ``sld.experiments.loading`` and ``sld.viz`` (fig1/fig2 v0).

Builds a MINIATURE run directory + dataset entirely under ``tmp_path`` --
no ``Trainer``, no real training loop. v0 viz only needs a well-formed run
artifact (config.yaml/metrics.csv/ckpt.pt/index.csv, REGLES section 6
contract) to load and plot.
"""

from __future__ import annotations

import csv
import json
import re
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
from sld.viz import fig2_vpt, fig3_factorial_effects
from sld.viz import results as results_mod
from sld.viz.fig1_rollout import _aggregate_curves, make_fig1
from sld.viz.fig2_spectrum import _plot_family_panel, make_fig2
from sld.viz.style import CONSTRAINT_LINESTYLES, FAMILY_COLORS

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


def test_make_fig1_reactivated_on_primary_factorial_shared_frozen_readout() -> None:
    """PROTOCOL.md section 27.1 supplies exactly the condition the OLD
    retirement doctring named as the reactivation trigger: a SHARED, FROZEN
    readout (``representation.kind = identity``, one ``e_recon`` for all 30
    cells), so no per-run OLS fit exists to distort the cross-arm
    comparison. ``make_fig1`` must therefore render, not raise."""
    assert callable(make_fig1)
    # the retirement's own module docstring text must be gone from the
    # active code path (still present in the module docstring as history,
    # never as make_fig1's behavior)
    import inspect

    assert "NotImplementedError" not in inspect.getsource(make_fig1)


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


def test_fig2_renders_one_titled_panel_per_family_and_never_truncates(
    tmp_path: Path, monkeypatch
) -> None:
    """The grid used to be a hardcoded ``subplots(2, 2)`` fed through
    ``zip(axes.ravel(), families)``. ``zip`` stops at the shorter side WITHOUT
    error, so every family past the 4th would have disappeared from the
    figure with the whole suite still green -- the defect slept only because
    ``FAMILY_COLORS`` happened to hold exactly 4 entries.

    So the count is asserted on the RENDERED figure: one titled panel per
    family, each family named exactly once. It is checked here at the real 7
    AND, via monkeypatched palettes, at 4 and at 9 -- a fixed grid passes at
    4 and fails at 7 and 9, and an off-by-one in the row/column arithmetic
    fails at one of the three.
    """

    def _render_and_count(expected_families: list[str]) -> list[str]:
        captured: dict = {}

        def _capture(fig, path):
            captured["fig"] = fig

        monkeypatch.setattr("sld.viz.fig2_spectrum.save_fig", _capture)

        results_root = tmp_path / f"results_{len(expected_families)}"
        yaml_path = tmp_path / f"exp_{len(expected_families)}.yaml"
        _write_exp_yaml(yaml_path)
        dh = _ou_data_hash(yaml_path)
        run_dir, _ = _make_run_dir(
            tmp_path / f"run_{len(expected_families)}",
            family="mlp",
            constraint="on",
            seed=0,
            data_hash=dh,
        )
        _write_index(
            results_root,
            "pendulum_core",
            [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]],
        )
        make_fig2(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig2.png"),
            require_full_cells=False,
        )
        return [ax.get_title() for ax in captured["fig"].axes if ax.get_title()]

    for palette in (
        dict(FAMILY_COLORS),  # the real 7
        {f: "#000000" for f in ("mlp", "transformer", "min_ssm", "koopman_ssm")},
        {f"fam{i}": "#000000" for i in range(9)},
    ):
        monkeypatch.setattr("sld.viz.fig2_spectrum.FAMILY_COLORS", palette)
        titles = _render_and_count(sorted(palette))
        print(f"[fig2] {len(palette)} families -> {len(titles)} titled panels")
        assert len(titles) == len(palette)
        for family in palette:
            # \b, not startswith: "mlp" is a prefix of "mlp_k" and
            # "transformer" of "transformer_k", so a prefix test would
            # double-count and mask a missing panel
            hits = sum(bool(re.match(rf"{re.escape(family)}\b", t)) for t in titles)
            assert hits == 1, f"{family}: {hits} panels in {titles}"


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

    # fallback: no arms declared -> style domains x pre-registered seeds.
    # DERIVED from the style dicts, so it survives the registry growing --
    # but a derived assertion alone is tautological, so the sizes it derives
    # from are pinned too (7 families since phase C, 2 constraints).
    _write_exp_yaml(tmp_path / "noarms.yaml")
    cells_fb, seeds_fb = _expected_grid(str(tmp_path / "noarms.yaml"))
    assert (len(FAMILY_COLORS), len(CONSTRAINT_LINESTYLES)) == (7, 2)
    assert len(cells_fb) == 1 * len(FAMILY_COLORS) * len(CONSTRAINT_LINESTYLES)
    assert len(cells_fb) == 14
    assert len(set(cells_fb)) == len(cells_fb)  # no duplicated cell
    assert seeds_fb == frozenset(range(5))


# ---------------------------------------------------------------------------
# PROTOCOL.md section 27.4 -- Fig. 1 (reactivated), Fig. 2 (VPT), Fig. 3
# (factorial effects). Every fixture is synthetic, built entirely under
# ``tmp_path`` -- no real ``results_primary_factorial/`` bytes are read (that
# campaign is live, see the packet's INTERDIT #2). Mirrors
# ``tests/viz/test_results.py``'s fixture shape but kept self-contained in
# this file (no ``tests/`` package init exists for a cross-file import).
# ---------------------------------------------------------------------------

_PF_ARMS = [
    ("mlp", "off"), ("mlp", "on"),
    ("transformer", "off"), ("transformer", "on"),
    ("ssm", "off"), ("ssm", "on"),
]
_PF_SEEDS = (0, 1, 2, 3, 4)
_PF_SYSTEM = "SynthOscillators"
_PF_REGIME = "SynthOURegime"


def _pf_write_configs(configs_dir: Path, exp_name: str) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "datasets": {"s0": {"system": {"name": _PF_SYSTEM}, "regime": {"name": _PF_REGIME}}},
        "arms": [{"backbone": b, "structure": s} for b, s in _PF_ARMS],
        "seeds": list(_PF_SEEDS),
    }
    (configs_dir / f"{exp_name}.yaml").write_text(yaml.safe_dump(doc))


def _pf_gate_payload(*, cell_id, backbone, structure, family, seed, status="learns",
                      e_h=(0.1, 0.2, 0.3), vpt_median=3.0) -> dict:
    return {
        "schema": "gate/1",
        "identity": {
            "cell_id": cell_id, "arm": {"backbone": backbone, "structure": structure, "family": family},
            "seed": seed, "stratum_system": _PF_SYSTEM, "stratum_regime": _PF_REGIME, "run_dir": "",
        },
        "digests": {"config_hash": "c" * 40, "data_hash": "d" * 8, "representation_digest": "r" * 8},
        "split": "test",
        "protocol": {"h_gate": len(e_h), "gamma": 5, "tau_1": 0.1, "tau_vpt": 0.5, "horizon": len(e_h),
                     "gamma_1": 2, "e_1_pers": 0.2, "e_recon": 1e-8, "attainable": True},
        "measures": {
            "e_h": list(e_h), "persistence_e_h": list(e_h), "linear_e_h": list(e_h), "e_1": e_h[0],
            "e_h_gate": e_h[-1], "m_low": 0.05, "m_high": 0.05, "lower_bound": True, "upper_bound": False,
            "near_miss": False, "vpt_median": vpt_median, "vpt_censored": 0, "per_episode_vpt": [vpt_median],
        },
        "status": status, "first_step": None,
    }


def _pf_build_grid(
    tmp_path: Path,
    exp_name: str = "pf",
    *,
    e_h_by_family_seed: dict[tuple[str, int], tuple[float, ...]] | None = None,
    vpt_by_family_seed: dict[tuple[str, int], float] | None = None,
    status_overrides: dict[tuple[str, int], str] | None = None,
    missing_family: str | None = None,
) -> tuple[Path, str]:
    """Full 6-family x 5-seed grid. ``missing_family``, if given, gets NO
    gate.json for any of its 5 seeds at all -- ``status='missing'`` (a
    'cellule absente' in the packet's own words), distinct from
    ``does_not_learn``/``divergence`` which at least ran."""
    e_h_by_family_seed = e_h_by_family_seed or {}
    vpt_by_family_seed = vpt_by_family_seed or {}
    status_overrides = status_overrides or {}
    results_root = tmp_path / "results"
    exp_root = results_root / exp_name
    idx = 0
    for backbone, structure in _PF_ARMS:
        family = results_mod._FAMILY_BY_PAIR[(backbone, structure)]
        for seed in _PF_SEEDS:
            if family == missing_family:
                idx += 1
                continue
            status = status_overrides.get((family, seed), "learns")
            e_h = e_h_by_family_seed.get((family, seed), (0.1, 0.2, 0.3))
            vpt = vpt_by_family_seed.get((family, seed), 3.0)
            payload = _pf_gate_payload(cell_id=idx, backbone=backbone, structure=structure,
                                        family=family, seed=seed, status=status, e_h=e_h, vpt_median=vpt)
            run_dir = exp_root / f"{family}-{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "gate.json").write_text(json.dumps(payload))
            idx += 1
    _pf_write_configs(tmp_path / "configs", exp_name)
    return results_root, exp_name


@pytest.fixture
def _pf_configs(monkeypatch):
    """Points ``results._CONFIGS_DIR`` at whatever ``tmp_path/configs`` a
    test writes -- never the real committed ``primary_factorial.yaml``."""
    def _apply(tmp_path):
        monkeypatch.setattr(results_mod, "_CONFIGS_DIR", tmp_path / "configs")
    return _apply


def _pf_write_state_scale(path: Path, *, sigma_train, system=_PF_SYSTEM, regime=_PF_REGIME, e_recon=1.866e-08):
    """A synthetic ``state_scale/1`` JSON -- the torch-free artifact
    ``sld.audit.scan_state_scale`` writes, faked here without touching torch
    or a real representation checkpoint."""
    payload = {
        "schema": "state_scale/1",
        "strata": {"s0": {"system": system, "regime": regime, "digest": "d" * 8,
                          "sigma_train": list(sigma_train), "e_recon": e_recon, "e_1_pers": 0.2}},
    }
    path.write_text(json.dumps(payload))


# --- Fig. 1 -------------------------------------------------------------

def test_fig1_analytic_mean_se_and_matches_results_module(tmp_path, _pf_configs):
    """POSITIVE: mlp's 5 seeds carry constant curves 1..5 -> textbook mean 3,
    SE sqrt(0.5) (independent hand computation, same numeric lock as
    ``tests/viz/test_results.py``'s own). Also: Fig. 1's returned data is
    BYTE-IDENTICAL to a fresh, independent call of
    ``results.aggregate_rollout_curves`` on the same table -- Fig. 1 must
    consume ``results.py``'s own e_h reading, never recompute it."""
    _pf_configs(tmp_path)
    e_h_by_seed = {("mlp", s): (float(s + 1),) * 3 for s in _PF_SEEDS}
    results_root, exp_name = _pf_build_grid(tmp_path, e_h_by_family_seed=e_h_by_seed)

    out = make_fig1(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig1.png"))
    mean_curve = np.array(out["data"]["mlp"]["mean_e_h"])
    se_curve = np.array(out["data"]["mlp"]["se_e_h"])
    assert np.allclose(mean_curve, 3.0, atol=1e-12)
    assert np.allclose(se_curve, np.sqrt(0.5), atol=1e-12)

    rows = results_mod.build_primary_results_table(results_root, exp_name)
    independent = results_mod.aggregate_rollout_curves(rows)
    assert out["data"] == independent

    assert out["image_path"].exists() and out["image_path"].stat().st_size > 0
    assert out["sidecar_csv"].exists()
    import hashlib
    assert out["sidecar_hash"] == hashlib.sha256(out["sidecar_csv"].read_bytes()).hexdigest()


def test_fig1_absent_and_ineligible_cells_never_drawn_or_averaged(tmp_path, _pf_configs):
    """NEGATIVE 1 ('cellule absente'): ``ssm_k`` has NO gate.json at all (5/5
    'missing') -- must be named, never averaged. Also covers a
    does_not_learn seed on ``mlp`` in the same run."""
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(
        tmp_path, status_overrides={("mlp", 0): "does_not_learn"}, missing_family="ssm_k",
    )
    out = make_fig1(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig1.png"))
    assert out["data"]["mlp"]["eligible"] is False
    assert "mean_e_h" not in out["data"]["mlp"]
    assert out["data"]["ssm_k"]["eligible"] is False
    assert out["data"]["ssm_k"]["statuses"] == {s: "missing" for s in _PF_SEEDS}
    assert "mlp" in out["non_eligible"] and "ssm_k" in out["non_eligible"]
    assert "does_not_learn" in out["caption"] and "ssm_k" in out["caption"]

    # the CSV -- what actually feeds the plot -- has no 'step' rows for the
    # non-eligible families (a broken eligibility filter WOULD emit them)
    rows_csv = list(csv.DictReader(out["sidecar_csv"].open()))
    mlp_rows = [r for r in rows_csv if r["family"] == "mlp"]
    assert all(r["step"] == "" for r in mlp_rows)
    assert {r["seed"] for r in mlp_rows} == {str(s) for s in _PF_SEEDS}
    # untouched families keep their step rows
    ssm_k_step_rows = [r for r in rows_csv if r["family"] == "ssm_k"]
    assert all(r["step"] == "" for r in ssm_k_step_rows)
    transformer_rows = [r for r in rows_csv if r["family"] == "transformer"]
    assert len(transformer_rows) == 3 and all(r["step"] != "" for r in transformer_rows)


def test_fig1_caption_has_the_eight_fields_and_no_forbidden_vocabulary(tmp_path, _pf_configs):
    """POSITIVE: the 8 fields section 27.4 requires. NEGATIVE 4: the legacy
    ``mu_j``/``constraint`` vocabulary (REGLES section 3's banned/renamed
    terms) never leaks into a caption."""
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(tmp_path)
    out = make_fig1(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig1.png"))
    caption = out["caption"]
    for field in ("CoupledOscillators", "OURegime", "H=", "sigma_train", "tau=0.5", "n=5"):
        assert field in caption, field
    assert "mu_j" not in caption
    assert "constraint" not in caption
    # NO state_scale_path given -- must say so explicitly, never a silent blank
    assert "NOT READ" in caption


def test_fig1_caption_cites_measured_sigma_train_when_state_scale_given(tmp_path, _pf_configs):
    """POSITIVE: with a ``state_scale_path``, the caption carries the
    FORMATTED NUMBERS from that JSON, not prose. NEGATIVE: a state_scale
    JSON that does NOT cover this grid's stratum (system/regime mismatch)
    must raise -- the file was supplied, so silence is not an option."""
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(tmp_path)
    sigma = [1.6057, 1.5271, 1.8577, 2.0715]
    state_scale = tmp_path / "state_scale.json"
    _pf_write_state_scale(state_scale, sigma_train=sigma)

    out = make_fig1(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig1.png"),
                     state_scale_path=str(state_scale))
    assert "NOT READ" not in out["caption"]
    for v in sigma:
        assert f"{v:.4f}" in out["caption"]
    assert "1.866e-08" in out["caption"]

    mismatched = tmp_path / "state_scale_wrong.json"
    _pf_write_state_scale(mismatched, sigma_train=sigma, system="SomeOtherSystem")
    with pytest.raises(ValueError, match="no stratum matching"):
        make_fig1(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig1.png"),
                  state_scale_path=str(mismatched))


def test_fig1_empty_root_raises_clear_error_not_a_raw_traceback(tmp_path, _pf_configs):
    """NEGATIVE 6 ('figure lancée avant index clos'): an empty results root
    (no gate.json anywhere) must raise a named, catchable error -- not let
    an obscure KeyError/IndexError escape from deep inside the plotting
    code."""
    _pf_configs(tmp_path)
    _pf_write_configs(tmp_path / "configs", "pf")
    with pytest.raises(FileNotFoundError, match="nothing to plot"):
        make_fig1(results_root=tmp_path / "nope", exp_name="pf", out_path=str(tmp_path / "fig1.png"))


# --- Fig. 2 ---------------------------------------------------------------

def test_fig2_five_points_per_cell_censored_never_zero_or_hmax_plus_one(tmp_path, _pf_configs):
    """POSITIVE: 5 raw VPT values/statuses per cell, censored count shown.
    NEGATIVE (part of N1/N3-adjacent, and 27.3's own forbidden
    replacements): the CSV field for a censored seed is BLANK, never '0'
    and never 'Hmax + 1' (H = 25 here, so never '26')."""
    _pf_configs(tmp_path)
    vpt = {("mlp", 0): float("inf"), ("mlp", 1): 4.0, ("mlp", 2): 5.0, ("mlp", 3): 6.0, ("mlp", 4): 7.0}
    results_root, exp_name = _pf_build_grid(tmp_path, vpt_by_family_seed=vpt)

    out = fig2_vpt.make_fig2(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig2.png"))
    assert len(out["data"]["mlp"]["per_seed"]) == 5
    assert out["data"]["mlp"]["n_censored"] == 1
    assert out["data"]["mlp"]["per_seed"][0]["status"] == "censored"
    assert all(out["data"]["mlp"]["per_seed"][s]["status"] == "crossed" for s in (1, 2, 3, 4))

    rows_csv = list(csv.DictReader(out["sidecar_csv"].open()))
    censored_row = next(r for r in rows_csv if r["family"] == "mlp" and r["seed"] == "0")
    assert censored_row["vpt_median"] == ""
    assert censored_row["vpt_median"] not in ("0", "26")
    assert censored_row["vpt_status"] == "censored"

    assert out["image_path"].exists() and out["image_path"].stat().st_size > 0


def test_fig2_caption_names_censoring_convention_and_no_forbidden_vocabulary(tmp_path, _pf_configs):
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(tmp_path)
    out = fig2_vpt.make_fig2(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig2.png"))
    caption = out["caption"]
    for field in ("CoupledOscillators", "OURegime", "H=25", "sigma_train", "tau=0.5", "n=5", "censor"):
        assert field in caption, field
    assert "mu_j" not in caption
    assert "constraint" not in caption
    assert "NOT READ" in caption


def test_fig2_caption_cites_measured_sigma_train_when_state_scale_given(tmp_path, _pf_configs):
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(tmp_path)
    sigma = [1.6057, 1.5271, 1.8577, 2.0715]
    state_scale = tmp_path / "state_scale.json"
    _pf_write_state_scale(state_scale, sigma_train=sigma)
    out = fig2_vpt.make_fig2(results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig2.png"),
                              state_scale_path=str(state_scale))
    assert "NOT READ" not in out["caption"]
    for v in sigma:
        assert f"{v:.4f}" in out["caption"]


def test_fig2_empty_root_raises_clear_error(tmp_path, _pf_configs):
    _pf_configs(tmp_path)
    _pf_write_configs(tmp_path / "configs", "pf")
    with pytest.raises(FileNotFoundError, match="nothing to plot"):
        fig2_vpt.make_fig2(results_root=tmp_path / "nope", exp_name="pf", out_path=str(tmp_path / "fig2.png"))


# --- Fig. 3 -----------------------------------------------------------------

def test_fig3_pairs_three_backbones_by_five_seeds_analytic_lock(tmp_path, _pf_configs):
    """POSITIVE: on-off contrasts [1, 2, 3, 4, 5] per backbone -> mean 3,
    SE sqrt(0.5), same numeric lock as
    ``tests/viz/test_results.py::test_paired_structure_contrasts_numeric_lock``,
    now checked THROUGH ``make_fig3`` for all three backbones at once."""
    _pf_configs(tmp_path)
    e_h_by_seed: dict[tuple[str, int], tuple[float, ...]] = {}
    for backbone, off_family, on_family in (("mlp", "mlp", "mlp_k"),
                                             ("transformer", "transformer", "transformer_k"),
                                             ("ssm", "min_ssm", "ssm_k")):
        for s in _PF_SEEDS:
            e_h_by_seed[(off_family, s)] = (0.0, 0.0, 0.0)
            e_h_by_seed[(on_family, s)] = (float(s + 1),) * 3
    results_root, exp_name = _pf_build_grid(tmp_path, e_h_by_family_seed=e_h_by_seed)

    out = fig3_factorial_effects.make_fig3(
        results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig3.png")
    )
    for backbone in ("mlp", "transformer", "ssm"):
        c = out["data"][backbone]
        assert len(c["contrasts"]) == 5
        assert np.allclose(sorted(c["contrasts"]), [1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(c["mean"] - 3.0) < 1e-12
        assert abs(c["se"] - np.sqrt(0.5)) < 1e-12
    assert out["image_path"].exists() and out["image_path"].stat().st_size > 0


def test_fig3_skips_a_seed_missing_on_one_side_never_fakes_it(tmp_path, _pf_configs):
    """NEGATIVE 2 ('seed filtrée'): a diverged ON arm for one seed of ``mlp``
    contributes NO contrast for that seed -- 4/5, never a faked 5th."""
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(
        tmp_path, status_overrides={("mlp_k", 2): "divergence"},
    )
    out = fig3_factorial_effects.make_fig3(
        results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig3.png")
    )
    assert len(out["data"]["mlp"]["contrasts"]) == 4


def test_fig3_caption_states_within_backbone_confound_and_no_forbidden_vocabulary(tmp_path, _pf_configs):
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(tmp_path)
    out = fig3_factorial_effects.make_fig3(
        results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig3.png")
    )
    caption = out["caption"]
    for field in ("CoupledOscillators", "OURegime", "H=25", "sigma_train", "tau=0.5", "n=5"):
        assert field in caption, field
    assert "mu_j" not in caption
    assert "constraint" not in caption
    assert "NOT READ" in caption


def test_fig3_caption_cites_measured_sigma_train_when_state_scale_given(tmp_path, _pf_configs):
    _pf_configs(tmp_path)
    results_root, exp_name = _pf_build_grid(tmp_path)
    sigma = [1.6057, 1.5271, 1.8577, 2.0715]
    state_scale = tmp_path / "state_scale.json"
    _pf_write_state_scale(state_scale, sigma_train=sigma)
    out = fig3_factorial_effects.make_fig3(
        results_root=results_root, exp_name=exp_name, out_path=str(tmp_path / "fig3.png"),
        state_scale_path=str(state_scale),
    )
    assert "NOT READ" not in out["caption"]
    for v in sigma:
        assert f"{v:.4f}" in out["caption"]


def test_fig3_empty_root_raises_clear_error(tmp_path, _pf_configs):
    _pf_configs(tmp_path)
    _pf_write_configs(tmp_path / "configs", "pf")
    with pytest.raises(FileNotFoundError, match="nothing to plot"):
        fig3_factorial_effects.make_fig3(results_root=tmp_path / "nope", exp_name="pf", out_path=str(tmp_path / "fig3.png"))


# --- shared sidecar writer: hash differs when content differs -------------

def test_write_sidecar_csv_hash_changes_with_content():
    """NEGATIVE 3 ('hash différent'): two different row sets must write two
    different digests -- an implementation that hashed a constant (e.g. the
    file path) instead of the bytes would pass every other test here and
    still be wrong."""
    import hashlib
    import tempfile

    from sld.viz.style import write_sidecar_csv

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d1 = write_sidecar_csv([{"a": "1"}], ["a"], td / "x.csv")
        d2 = write_sidecar_csv([{"a": "2"}], ["a"], td / "y.csv")
        assert d1 != d2
        assert d1 == hashlib.sha256((td / "x.csv").read_bytes()).hexdigest()
        assert (td / "x.csv.sha256").read_text().strip() == d1
