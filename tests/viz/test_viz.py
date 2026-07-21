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
from sld.viz.fig1_rollout import (
    ReadoutFit,
    _aggregate_curves,
    _apply_readout,
    _fit_readout,
    _run_drift_curve,
    make_fig1,
)
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
        writer.writerow(["run_dir", "regime", "family", "constraint", "seed", "status", "final_val_loss"])
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


class _ScaledEncoder:
    """Test double (F1 invariance test): wraps ``base`` so its output is
    the reparameterized latent ``z' = S @ z`` (row-vector convention:
    ``z_row @ S.T``)."""

    def __init__(self, base: torch.nn.Module, S: torch.Tensor) -> None:
        self.base = base
        self.S = S

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) @ self.S.T


class _ScaledPredictor:
    """Test double (F1 invariance test): ``base`` conjugated by ``S`` so it
    operates in the SAME reparameterized coordinates as ``_ScaledEncoder``
    -- ``h' = S @ h``. This is the genuine "same dynamics, different latent
    basis" gauge transform: scaling the encoder alone (without also
    conjugating the predictor's own recurrence) would not describe the same
    underlying dynamics, just a different, inconsistent system.
    """

    def __init__(self, base, S: torch.Tensor) -> None:
        self.base = base
        self.S = S
        self.Sinv = torch.linalg.inv(S)

    def rollout(self, h0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        h0_orig = h0 @ self.Sinv.T
        h_orig = self.base.rollout(h0_orig, u_seq)
        return h_orig @ self.S.T


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
    results_root = tmp_path / "results"
    run_dir, _ = _make_run_dir(tmp_path)
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])
    rows = read_index(results_root, "pendulum_core")
    assert len(rows) == 1
    assert rows[0]["seed"] == 0
    assert rows[0]["final_val_loss"] == 0.4
    assert rows[0]["status"] == "complete"


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


def test_readout_drift_invariant_to_linear_reparam(tmp_path: Path) -> None:
    """F1 blocker close-out. The reviewer's exact reproduction: an
    invertible ``S=diag(100,1)`` relabeling of the latent axes used to flip
    both the drift curve's height and its shape when drift was read in the
    run's raw encoder coordinates. The fix reads drift out through a
    per-run OLS readout into TRUE-STATE coordinates
    (``fig1_rollout._run_drift_curve``); this builds a baseline run and an
    S-reparameterized run representing the IDENTICAL dynamics in a
    different latent basis (encoder AND predictor both conjugated by S --
    see ``_ScaledEncoder``/``_ScaledPredictor`` above) and checks the two
    readout-based curves coincide to tight tolerance.
    """
    d_x, d_z, d_u = 2, 2, 1
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path, n_episodes=20)
    dataset = build_dataset_from_expconfig(yaml_path, "ou")

    ambient = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        encoder = FrameMLPEncoder(d_x, d_z, (8, 8))
        predictor = get_predictor("mlp", d_z, d_u, target_params=200)
    finally:
        torch.set_default_dtype(ambient)
    encoder.eval()
    predictor.eval()

    curve_base, fit_base = _run_drift_curve(
        {"encoder": encoder, "predictor": predictor}, dataset, horizon=5
    )

    # bounded-conditioning adversarial gauges (round 3): rescaling, SHEAR,
    # and their composition -- the curve AND the determinability verdict
    # must be identical for every operationally-equivalent gauge
    gauges = {
        "diag(100,1)": torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float32)),
        "shear[[1,10],[0,1]]": torch.tensor([[1.0, 10.0], [0.0, 1.0]], dtype=torch.float32),
        "shear@diag": torch.tensor([[1.0, 10.0], [0.0, 1.0]], dtype=torch.float32)
        @ torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float32)),
    }
    assert np.isfinite(fit_base.r2) and not fit_base.indeterminate
    for name, S in gauges.items():
        run_scaled = {
            "encoder": _ScaledEncoder(encoder, S),
            "predictor": _ScaledPredictor(predictor, S),
        }
        curve_scaled, fit_scaled = _run_drift_curve(run_scaled, dataset, horizon=5)
        assert fit_scaled.indeterminate == fit_base.indeterminate, name
        assert np.isfinite(fit_scaled.r2)
        assert np.allclose(curve_base, curve_scaled, atol=1e-5, rtol=1e-4), (
            f"gauge invariance broken by {name}: base={curve_base}, scaled={curve_scaled}"
        )


# -- fig1 / fig2 --------------------------------------------------------------


def test_make_fig1_produces_png(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in range(2):
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    out_path = tmp_path / "assets" / "fig1_rollout.png"
    make_fig1(
        results_root=str(results_root),
        exp_name="pendulum_core",
        exp_yaml=str(yaml_path),
        out_path=str(out_path),
        horizon=5,
        require_full_cells=False,  # only 2/5 seeds -- explicit opt-out (F5)
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_make_fig1_missing_index_exits_cleanly(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        make_fig1(results_root=str(tmp_path / "results"), out_path=str(tmp_path / "fig1.png"))
    assert exc_info.value.code == 1
    assert "does not exist" in capsys.readouterr().err


def _restrict_grid_to_mlp_on(monkeypatch) -> None:
    """Shrink the expected grid (M4: derived from FAMILY_COLORS x
    CONSTRAINT_LINESTYLES x yaml regimes) to the single cell the fixtures
    populate, so strict-mode tests can target ONE cell's seed-completeness
    without fabricating checkpoints for all four families."""
    monkeypatch.setattr("sld.viz.fig1_rollout.FAMILY_COLORS", {"mlp": "#000000"})
    monkeypatch.setattr("sld.viz.fig1_rollout.CONSTRAINT_LINESTYLES", {"on": "-"})


def test_make_fig1_incomplete_cell_raises_in_strict_mode(tmp_path: Path, monkeypatch) -> None:
    """F5: a cell missing any of the 5 pre-registered seeds is refused in
    strict (default) mode, naming the cell and the missing seeds."""
    _restrict_grid_to_mlp_on(monkeypatch)
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in range(4):  # seed 4 missing
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    with pytest.raises(ValueError, match=r"missing seeds \[4\]"):
        make_fig1(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"),
            horizon=5,
        )


def test_make_fig1_wholly_missing_cell_raises_in_strict_mode(tmp_path: Path) -> None:
    """M4 (round 2): a cell with ZERO runs -- absent from the index
    entirely, so it never enters the surviving-cells dict -- must fail
    strict mode exactly like a partially-filled one. With the full expected
    grid (1 regime x 4 families x 2 constraints here), populating only
    mlp/on leaves e.g. koopman_ssm/off wholly missing."""
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in range(5):
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    with pytest.raises(ValueError, match=r"missing seeds \[0, 1, 2, 3, 4\]"):
        make_fig1(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"),
            horizon=5,
        )


def test_make_fig1_draws_se_bands_for_a_full_cell(tmp_path: Path, monkeypatch) -> None:
    """F5: a complete (5-seed) cell in strict mode plots a +/-1 SE band,
    not just a bare point estimate."""
    _restrict_grid_to_mlp_on(monkeypatch)
    captured: dict = {}

    def _capture(fig, path):
        captured["fig"] = fig
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    monkeypatch.setattr("sld.viz.fig1_rollout.save_fig", _capture)

    def _spy_aggregate(curves):
        out = _aggregate_curves(curves)
        captured["aggregate"] = out  # single cell here, so one call
        return out

    monkeypatch.setattr("sld.viz.fig1_rollout._aggregate_curves", _spy_aggregate)

    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in range(5):
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    make_fig1(
        results_root=str(results_root),
        exp_name="pendulum_core",
        exp_yaml=str(yaml_path),
        out_path=str(tmp_path / "assets" / "fig1.png"),
        horizon=5,
    )

    fig = captured["fig"]
    ax = fig.axes[0]
    assert len(ax.collections) >= 1  # fill_between drew an SE band
    assert "n=5" in " ".join(h.get_label() for h in ax.get_legend_handles_labels()[0])

    # m12 (round 2): the band must BE mean +/- 1 SE of the actual curves --
    # a zero-width or unrelated collection must fail. The captured
    # _aggregate_curves output is the single plotting source (its docstring
    # locks that), so the PolyCollection vertices must match it.
    mean_curve, se = captured["aggregate"]
    verts = ax.collections[0].get_paths()[0].vertices
    # m12 (round 3): POINTWISE lock, not just global extrema -- rebuild the
    # x -> (lo, hi) mapping from the fill_between polygon (fill_between may
    # add closing vertices; group y-values by x and take min/max per x)
    by_x: dict = {}
    for x, y in verts:
        by_x.setdefault(round(float(x), 9), []).append(float(y))
    xs = sorted(by_x)
    assert len(xs) == len(mean_curve)
    for i, x in enumerate(xs):
        lo, hi = min(by_x[x]), max(by_x[x])
        assert np.isclose(hi, mean_curve[i] + se[i], rtol=1e-6), f"upper band at x={x}"
        assert np.isclose(lo, mean_curve[i] - se[i], rtol=1e-6), f"lower band at x={x}"
    assert se.max() > 0  # five distinct seeds: a zero-width band is a bug


def test_make_fig1_dataset_mismatch_raises(tmp_path: Path) -> None:
    """F6: a run dir whose data_hash suffix does not match what the
    CURRENT experiment YAML rebuilds is a hard failure, not silently
    evaluated."""
    results_root = tmp_path / "results"
    run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=0, data_hash="deadbeef")
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)

    with pytest.raises(ValueError, match="data_hash"):
        make_fig1(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"),
            horizon=5,
        )


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


# -- round-2 locks: B1 domain, M5 strict fallback, M6/m12 aggregation ---------


def test_fit_readout_classifies_rank_deficient_design_indeterminate() -> None:
    """B1 (rounds 2-3): collapsed latents (z_2 = z_1, a LEGAL outcome --
    latent collapse is an object of study) make the affine design
    [z_enc | 1] column-rank-deficient; the Moore-Penrose solution is then
    not equivariant outside the span, so the fit MUST be classified
    indeterminate, never used silently."""
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(64, 1))
    z_enc = np.concatenate([z0, z0], axis=1)  # duplicated coordinate: rank 2 < 3
    z_gt = rng.normal(size=(64, 2))
    fit = _fit_readout(z_enc, z_gt)
    assert fit.indeterminate
    assert fit.rank < 3

    # full-rank, well-conditioned control: determinate
    z_ok = rng.normal(size=(64, 2))
    fit_ok = _fit_readout(z_ok, z_ok @ rng.normal(size=(2, 2)) + 0.01 * rng.normal(size=(64, 2)))
    assert not fit_ok.indeterminate


def test_fit_readout_stable_under_diagonal_gauge_round4() -> None:
    """B1-v3 (round 4 supervisor decision -- this test REPLACES and
    INVERTS the old ``test_fit_readout_adversarial_gauge_flips_
    determinability_as_declared``): column-equilibrating the design
    before measuring cond (each column of ``[z_enc | 1]`` divided by its
    own RMS) makes the determinability verdict STABLE under the ENTIRE
    diagonal-rescaling class -- strictly stronger than round 3's
    declaration, which conceded that ``diag(1e9,1)`` flipped the verdict
    and called that an unavoidable cost of "genuinely destroying
    information at float32". Round 4's reviewer reproduction: multiplying
    EVERY latent coordinate by ``2**20`` is exactly representable and
    round-trips bit-exact through float32 (``cond(S)=1`` -- no unit is
    singled out), yet the OLD raw-design cond jumped from ~1 to ~1e6
    purely because the intercept column stays fixed at 1 while the z
    columns are rescaled -- a unit change, not destroyed information.
    This test locks that the verdict no longer flips for EITHER round-3's
    ``diag(1e9,1)`` probe or round-4's uniform ``2**20`` probe."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(128, 2))
    z_gt = rng.normal(size=(128, 2))
    fit = _fit_readout(z, z_gt)
    assert not fit.indeterminate  # well-conditioned baseline

    S_diag = np.diag([1e9, 1.0])
    fit_diag = _fit_readout(z @ S_diag.T, z_gt)
    assert fit_diag.indeterminate == fit.indeterminate  # round 3 probe: now STABLE
    assert fit_diag.rank == fit.rank

    # round 4's exact reproduction: a bit-exact float32 round-trip power-
    # of-two rescale of EVERY coordinate (not just one)
    scale = np.float32(2.0**20)
    z_pow2 = (z.astype(np.float32) * scale).astype(np.float64)
    fit_pow2 = _fit_readout(z_pow2, z_gt)
    assert fit_pow2.indeterminate == fit.indeterminate  # round 4 probe: STABLE
    assert fit_pow2.rank == fit.rank

    # The raw ``lstsq`` rank would flip at these harmless but finite unit
    # changes. The verdict rank must come from the equilibrated design too.
    for scale in (1e-30, 1e20):
        fit_extreme = _fit_readout(z @ np.diag([scale, 1.0]).T, z_gt)
        assert fit_extreme.indeterminate == fit.indeterminate
        assert fit_extreme.rank == fit.rank


def test_equilibrated_readout_solve_locks_extreme_gauge_outputs_and_drift() -> None:
    """B1-v4: the solve, not merely its eligibility verdict, is invariant
    under extreme per-coordinate unit gauges. A raw ``lstsq`` loses rank at
    1e20/1e-20 here despite the target being exactly affine."""
    rng = np.random.default_rng(17)
    z = rng.normal(size=(128, 2))
    affine = np.array([[1.5, -0.75], [0.25, 2.0], [0.2, -0.4]])
    z_gt = np.concatenate([z, np.ones((len(z), 1))], axis=1) @ affine
    rollout_target = rng.normal(size=(5, 16, 2))
    rollout_pred = rollout_target + np.array([0.15, -0.05])

    baseline = _fit_readout(z, z_gt)
    baseline_decoded = _apply_readout(rollout_target, baseline.W)
    baseline_curve = fr_drift_curve(_apply_readout(rollout_pred, baseline.W), baseline_decoded)
    assert baseline.rank == 3
    assert baseline.r2 == pytest.approx(1.0, abs=1e-14)

    gauges = {
        "1": np.eye(2),
        "2**20": np.diag([2.0**20, 2.0**20]),
        "1e20": np.diag([1e20, 1.0]),
        "1e-20": np.diag([1e-20, 1.0]),
        "diag(1e9,1)": np.diag([1e9, 1.0]),
    }
    for name, gauge in gauges.items():
        z_scaled = z @ gauge.T
        fit = _fit_readout(z_scaled, z_gt)
        decoded_target = _apply_readout(rollout_target @ gauge.T, fit.W)
        decoded_pred = _apply_readout(rollout_pred @ gauge.T, fit.W)
        assert fit.rank == baseline.rank, name
        assert fit.r2 == pytest.approx(baseline.r2, abs=1e-14), name
        assert np.max(np.abs(_apply_readout(z_scaled, fit.W) - z_gt)) < 1e-12, name
        assert np.allclose(decoded_target, baseline_decoded, atol=1e-12, rtol=1e-12), name
        assert np.allclose(fr_drift_curve(decoded_pred, decoded_target), baseline_curve, atol=1e-12, rtol=1e-12), name


def test_fit_readout_catches_quasi_collinearity_after_equilibration() -> None:
    """B1-v3 (round 4): the equilibrated criterion is not vacuous -- it
    still catches genuine near-collinearity between latent coordinates.
    ``z2 = z1 + tiny_noise`` is FULL column rank (unlike the exact-
    duplicate case in ``test_fit_readout_classifies_rank_deficient_
    design_indeterminate`` above, so the equilibrated design reports rank 3,
    not degenerate), but the two z-columns are nearly identical once
    equilibrated by their own RMS, so the equilibrated cond still blows up
    and the verdict is indeterminate anyway."""
    rng = np.random.default_rng(2)
    z1 = rng.normal(size=(256, 1))
    z2 = z1 + 1e-9 * rng.normal(size=(256, 1))  # near-duplicate, not exact
    z_enc = np.concatenate([z1, z2], axis=1)
    z_gt = rng.normal(size=(256, 2))
    fit = _fit_readout(z_enc, z_gt)
    assert fit.rank == 3  # full column rank -- not the exact-duplicate case
    assert fit.indeterminate  # near-collinearity still caught after equilibration


def test_make_fig1_indeterminate_readout_raises_in_strict_mode(tmp_path: Path, monkeypatch) -> None:
    """B1 (rounds 2-3): in strict mode an indeterminate readout FAILS the
    render naming the run; M5 (round 3): the per-run CSV verdict is
    written BEFORE the strict failure, so it survives the exception."""
    _restrict_grid_to_mlp_on(monkeypatch)
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in range(5):
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    real_fit = _fit_readout

    def _collapsed_fit(z_enc, z_gt):
        fit = real_fit(z_enc, z_gt)
        return ReadoutFit(
            W=fit.W, r2=fit.r2, rank=fit.rank, cond=fit.cond,
            sensitivity=fit.sensitivity, indeterminate=True,
        )

    monkeypatch.setattr("sld.viz.fig1_rollout._fit_readout", _collapsed_fit)

    with pytest.raises(ValueError, match="indeterminate"):
        make_fig1(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"),
            horizon=5,
        )

    # M5 (round 3): the per-run verdict is durable -- the CSV was written
    # BEFORE the strict failure, and it carries the indeterminate status
    r2_csv = tmp_path / "fig1_readout_r2.csv"
    assert r2_csv.is_file()
    content = r2_csv.read_text()
    assert "indeterminate" in content


def test_make_fig1_drift_failure_still_records_r2_in_csv(tmp_path: Path, monkeypatch) -> None:
    """M8: a run whose readout fit succeeds (full-rank, well-conditioned,
    a real R^2) but whose drift-curve computation itself fails downstream
    (e.g. its decoded target is ~0 and trips ``fr_drift_curve``'s own
    zero-target guard) must not vanish before any record of that ALREADY
    -KNOWN readout verdict is written. Before this fix, the old monolithic
    ``_run_drift_curve`` computed the fit AND the curve in one call, and
    make_fig1's try/except around that call swallowed ANY exception --
    including one raised only by ``fr_drift_curve``, well after the fit
    had already succeeded -- as a generic silent skip with no r2_records
    entry at all."""
    _restrict_grid_to_mlp_on(monkeypatch)
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in range(5):
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    real_drift_curve = fr_drift_curve
    calls = {"n": 0}

    def _flaky_drift(pred, target):
        calls["n"] += 1
        if calls["n"] == 1:
            # simulates fr_drift_curve's OWN "target identically zero"
            # guard tripping on a decoded target ~0, downstream of a
            # perfectly fine (full-rank, well-conditioned) readout fit
            raise ValueError("target is identically zero; normalization is undefined")
        return real_drift_curve(pred, target)

    monkeypatch.setattr("sld.viz.fig1_rollout.fr_drift_curve", _flaky_drift)

    # seed 0's drift failure removes it from its cell -> strict mode now
    # ALSO reports that cell as missing seed 0 (a separate, expected
    # consequence -- the point of this test is the CSV record, not this)
    with pytest.raises(ValueError, match=r"missing seeds \[0\]"):
        make_fig1(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"),
            horizon=5,
        )

    r2_csv = tmp_path / "fig1_readout_r2.csv"
    assert r2_csv.is_file()
    with open(r2_csv, newline="") as f:
        rows_out = list(csv.DictReader(f))
    drift_failed = [r for r in rows_out if r["drift_status"].startswith("drift-failed")]
    assert len(drift_failed) == 1, "the drift-failed run must leave exactly one CSV record"
    rec = drift_failed[0]
    assert rec["seed"] == "0"
    assert rec["readout_status"] == "ok"
    assert rec["drift_status"] == "drift-failed: zero-or-negligible-decoded-target-signal"
    # M9: the readout verdict is FULLY present, not just a status label
    assert float(rec["r2"]) == float(rec["r2"])  # parses as a real, finite float
    assert int(rec["rank"]) > 0


def test_make_fig1_extra_seed_raises_in_strict_mode(tmp_path: Path, monkeypatch) -> None:
    """M10: strict mode used to check ONLY for missing seeds -- any
    complete row, regardless of whether its seed was one of the pre-
    registered ones, was folded into the cell's mean. A stray extra seed
    (e.g. seed 7, when only 0-4 are pre-registered) must be a NAMED
    strict-mode violation too, not silently averaged in."""
    _restrict_grid_to_mlp_on(monkeypatch)
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    rows = []
    for seed in [*range(5), 7]:  # 0-4 required, plus an unexpected extra
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    with pytest.raises(ValueError, match=r"unexpected seed\(s\) \[7\]"):
        make_fig1(
            results_root=str(results_root),
            exp_name="pendulum_core",
            exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"),
            horizon=5,
        )


def test_make_fig1_r2_csv_carries_full_determinability_evidence(tmp_path: Path) -> None:
    """M9: PROTOCOL promises R^2, rank, and determinability evidence; the
    old CSV writer exported only ``r2``/``readout_status`` -- rank, the
    design's column count, equilibrated cond, sensitivity, and the applied
    threshold were all missing, so an "indeterminate" verdict was not
    auditable from the published artifact alone. This locks the full
    column set and sane values for a genuine "ok" run."""
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)

    run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=0, data_hash=dh)
    _write_index(results_root, "pendulum_core", [[str(run_dir), "ou", "mlp", "on", 0, "complete", 0.4]])

    make_fig1(
        results_root=str(results_root),
        exp_name="pendulum_core",
        exp_yaml=str(yaml_path),
        out_path=str(tmp_path / "assets" / "fig1.png"),
        horizon=5,
        require_full_cells=False,
    )

    r2_csv = tmp_path / "assets" / "fig1_readout_r2.csv"
    with open(r2_csv, newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        rows_out = list(reader)

    required_columns = {
        "run_dir", "regime", "family", "constraint", "seed",
        "r2", "rank", "n_design_cols", "cond_equilibrated",
        "sensitivity", "sensitivity_threshold", "readout_status", "drift_status",
    }
    assert required_columns <= columns
    assert len(rows_out) == 1
    rec = rows_out[0]
    assert int(rec["rank"]) > 0
    assert int(rec["n_design_cols"]) == D_Z + 1  # [z_enc | 1]
    assert float(rec["cond_equilibrated"]) >= 1.0
    assert float(rec["sensitivity"]) >= 0.0
    assert float(rec["sensitivity_threshold"]) == pytest.approx(1e-2)
    assert rec["readout_status"] == "ok"
    assert rec["drift_status"] == "ok"


def test_make_fig1_keeps_indeterminacy_when_drift_also_fails(tmp_path: Path, monkeypatch) -> None:
    """M8/B1-v4: downstream drift failure is a second diagnosis, never a
    replacement for the strict indeterminate-readout violation. Undefined
    conditioning is exported as empty categorical absence, not ``inf``."""
    _restrict_grid_to_mlp_on(monkeypatch)
    results_root = tmp_path / "results"
    yaml_path = tmp_path / "pendulum_core.yaml"
    _write_exp_yaml(yaml_path)
    dh = _ou_data_hash(yaml_path)
    rows = []
    for seed in range(5):
        run_dir, _ = _make_run_dir(tmp_path, family="mlp", constraint="on", seed=seed, data_hash=dh)
        rows.append([str(run_dir), "ou", "mlp", "on", seed, "complete", 0.4])
    _write_index(results_root, "pendulum_core", rows)

    real_fit = _fit_readout

    def _rank_deficient_fit(z_enc, z_gt):
        fit = real_fit(z_enc, z_gt)
        return ReadoutFit(
            W=fit.W, r2=fit.r2, rank=1, cond=float("inf"),
            sensitivity=float("inf"), indeterminate=True,
        )

    monkeypatch.setattr("sld.viz.fig1_rollout._fit_readout", _rank_deficient_fit)
    monkeypatch.setattr(
        "sld.viz.fig1_rollout.fr_drift_curve",
        lambda pred, target: (_ for _ in ()).throw(ValueError("target is identically zero; normalization is undefined")),
    )

    with pytest.raises(ValueError, match="indeterminate readout"):
        make_fig1(
            results_root=str(results_root), exp_name="pendulum_core", exp_yaml=str(yaml_path),
            out_path=str(tmp_path / "fig1.png"), horizon=5,
        )

    with open(tmp_path / "fig1_readout_r2.csv", newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 5
    assert all(record["readout_status"] == "indeterminate" for record in records)
    assert all(record["drift_status"].startswith("drift-failed") for record in records)
    assert all(record["cond_equilibrated"] == "" for record in records)
    assert all(record["sensitivity"] == "" for record in records)
    finite_numeric_columns = ("r2", "rank", "n_design_cols", "sensitivity_threshold")
    assert all(
        np.isfinite(float(record[column]))
        for record in records
        for column in finite_numeric_columns
    )


def test_fig2_strict_mode_refuses_off_substitution_for_missing_on() -> None:
    """M5 (round 2): require_full_cells=True must REFUSE the on->off
    fallback -- an OFF-only family used to survive strict mode by silent
    substitution, contradicting the completeness guarantee."""
    fig, ax = plt.subplots()
    rows = [
        {"run_dir": "/nonexistent", "regime": "ou", "family": "mlp",
         "constraint": "off", "seed": 0, "status": "complete", "final_val_loss": 0.4},
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
