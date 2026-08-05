"""Tests for the factorial runner: expansion, closed schema, resume/index.

The mini end-to-end test below exercises ``Trainer.resolve_run_dir`` -- a
method a parallel agent was adding to ``sld.training.trainer.Trainer`` at
the time this file was written (see the task brief). If it has not landed
yet, that one test fails with ``AttributeError: 'Trainer' object has no
attribute 'resolve_run_dir'``; every other test here is independent of it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from sld.experiments.factorial import (
    _cell_status,
    _final_val_loss,
    _validate_completion_sentinel,
    expand,
    main,
)
from sld.training.trainer import Trainer

_CONFIGS = Path(__file__).resolve().parents[2] / "src" / "sld" / "experiments" / "configs"


def _minimal_dataset_entry(**overrides) -> dict:
    entry = {
        "system": {"name": "DampedPendulum", "params": {}},
        "regime": {"name": "OURegime", "params": {"theta_ou": 1.0}},
        "obs": None,
        "n_episodes": 2,
        "T": 5,
        "dt": 0.02,
        "seed": 0,
    }
    entry.update(overrides)
    return entry


_SIX_CELLS = {
    ("mlp", "off"): "mlp",
    ("mlp", "on"): "mlp_k",
    ("transformer", "off"): "transformer",
    ("transformer", "on"): "transformer_k",
    ("ssm", "off"): "min_ssm",
    ("ssm", "on"): "ssm_k",
}


def test_expand_produces_exactly_the_six_factorial_cells():
    # phase C: all 3 backbones x 2 structures now resolve. The mapping is
    # PINNED here, value by value -- ("ssm", "on") in particular moved from
    # koopman_ssm to ssm_k at Amendment 4 (PROTOCOL.md section 13.2), and
    # that remap must never drift back or wander silently.
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [
            {"backbone": b, "structure": s, "overrides": ({"delta": 0.02} if s == "on" else {})}
            for b, s in _SIX_CELLS
        ],
        "seeds": [0],
    }
    cells = expand(config, dry_run=True)
    assert len(cells) == 6
    pairs = [(c["backbone"], c["structure"]) for c in cells]
    assert set(pairs) == set(_SIX_CELLS)
    assert len(set(pairs)) == 6  # no duplicates
    for cell in cells:
        assert cell["family"] == _SIX_CELLS[(cell["backbone"], cell["structure"])]
    # 6 distinct cells, 6 distinct families: no two cells of the factorial
    # are the same model wearing two labels
    assert len({c["family"] for c in cells}) == 6
    print(f"[factorial] 6 cells, 0 duplicates: {sorted(_SIX_CELLS.items())}")


def test_factorial_pair_table_is_exactly_the_declared_cross_product():
    # the table is the full 3x2 and nothing more: a missing pair would make
    # an arm unrunnable, an extra one would smuggle a 7th cell into a grid
    # PROTOCOL.md section 13.3 sizes at 6 arms x 2 datasets x 5 seeds
    from sld.experiments.factorial import _BACKBONES, _FAMILY_BY_PAIR, _STRUCTURES

    assert set(_FAMILY_BY_PAIR) == {(b, s) for b in _BACKBONES for s in _STRUCTURES}
    assert _FAMILY_BY_PAIR == _SIX_CELLS


@pytest.mark.parametrize("config_name", ("calibration.yaml", "pendulum_core.yaml"))
def test_historical_mu_j_configs_are_no_longer_executable(config_name):
    # phase B1: these 2 configs are the frozen trace of the retired mu_J
    # campaign (reports/historical/mu_j/) -- kept on the OLD family/
    # constraint schema ON PURPOSE, which is exactly what makes the new
    # backbone/structure validator refuse them before anything is built.
    config = yaml.safe_load((_CONFIGS / config_name).read_text())
    with pytest.raises(ValueError):
        expand(config, dry_run=False)


def test_unknown_top_level_key_raises():
    config = {
        "exp_name": "x",
        "datasets": {},
        "arms": [],
        "seeds": [],
        "bogus": 1,
    }
    with pytest.raises(ValueError, match="bogus"):
        expand(config)


def test_unknown_registry_name_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry(system={"name": "NotASystem", "params": {}})},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="unknown system"):
        expand(config)


def test_null_override_raises_in_run_mode_but_not_dry_run():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "ssm", "structure": "on", "overrides": {"mu_j": None}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="calibration"):
        expand(config, dry_run=False)

    cells = expand(config, dry_run=True)
    assert len(cells) == 1
    assert cells[0]["mu_j"] is None


# -- M11: empty factorial axes must fail loudly, not silently expand to 0 --


def test_empty_datasets_axis_raises():
    config = {
        "exp_name": "x",
        "datasets": {},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="datasets"):
        expand(config)


def test_empty_arms_axis_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="arms"):
        expand(config)


def test_empty_seeds_axis_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [],
    }
    with pytest.raises(ValueError, match="seeds"):
        expand(config)


def test_end_to_end_resume_and_index(tmp_path):
    """Run a tiny grid twice: 2nd invocation must SKIP (resume) and index.csv
    must reflect a completed run. Designed to finish in well under 30s
    (4 episodes, T=10, 2 epochs, target_params=200, mlp only, 1 seed)."""
    results_root = tmp_path / "results"
    config = {
        "exp_name": "mini",
        "datasets": {
            # n_episodes=10 so the default 0.8/0.1/0.1 split leaves >=1
            # episode in val (VICReg's variance_loss needs N>=2 samples;
            # an empty val split would crash there, not a factorial bug)
            "ou": _minimal_dataset_entry(n_episodes=10, T=10),
        },
        "base": {
            "d_x": 2,
            "d_z": 2,
            "d_u": 1,
            "epochs": 2,
            "batch_episodes": 2,
            "target_params": 200,
            "results_root": str(results_root),
        },
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    config_path = tmp_path / "mini.yaml"
    config_path.write_text(yaml.safe_dump(config))

    rc1 = main(["--config", str(config_path)])
    assert rc1 == 0

    index_path = results_root / "mini" / "index.csv"
    assert index_path.exists()
    rows = list(csv.DictReader(index_path.open()))
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    assert rows[0]["final_val_loss"] != ""

    run_dir = Path(rows[0]["run_dir"])
    ckpt_mtime = (run_dir / "ckpt.pt").stat().st_mtime

    rc2 = main(["--config", str(config_path)])
    assert rc2 == 0
    # unchanged mtime -> the 2nd invocation resumed (SKIP), it did not retrain
    assert (run_dir / "ckpt.pt").stat().st_mtime == ckpt_mtime

    rows2 = list(csv.DictReader(index_path.open()))
    assert len(rows2) == 1
    assert rows2[0]["status"] == "complete"


def _mini_run_config(results_root: Path, **arm_overrides) -> dict:
    """Config shared by the F9/F17 end-to-end tests below: same shape as
    ``test_end_to_end_resume_and_index``'s, mlp-only, tiny and fast."""
    return {
        "exp_name": "mini",
        "datasets": {"ou": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": 2,
            "d_u": 1,
            "epochs": 2,
            "batch_episodes": 2,
            "target_params": 200,
            "results_root": str(results_root),
        },
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0, **arm_overrides}}],
        "seeds": [0],
    }


# -- F7: closed schema over base AND every arm's overrides -------------------


def test_unknown_key_in_base_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"lrr": 0.1},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="lrr"):
        expand(config)


def test_unknown_key_in_overrides_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0, "mu_jj": 1.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="mu_jj"):
        expand(config)


def test_runner_owned_keys_rejected_in_base_and_overrides():
    # base owns nothing the runner injects -- exp_name/seed/family are the
    # runner's, never the user's, at either level (F7)
    base_leak = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"seed": 5},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="injected by the runner"):
        expand(base_leak)

    overrides_leak = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0, "family": "mlp"}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="injected by the runner"):
        expand(overrides_leak)


def test_budget_owned_and_derived_keys_rejected_in_base_and_overrides():
    # these ARE known Trainer names -- they get the Trainer's own message,
    # not a generic "unknown key" one (F7)
    budget_in_base = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"hidden": [32, 32]},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="capacity"):
        expand(budget_in_base)

    derived_in_overrides = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0, "d_h": 4}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="derived or policy"):
        expand(derived_in_overrides)


# -- F8: backbone/structure schema + run-identity collision preflight -------


def test_invalid_structure_value_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "maybe", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="structure"):
        expand(config)


def test_old_constraint_field_rejected():
    # phase B1: the retired family/constraint schema must be refused
    # outright, never silently accepted or misread as backbone/structure.
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="constraint"):
        expand(config)


def test_koopman_ssm_is_registered_but_never_a_factorial_family():
    # the asymmetry PROTOCOL.md section 13.1 declares, asserted on the two
    # objects that carry it: 7 registered families, 6 factorial cells, and
    # koopman_ssm is the one in the first set and not the second.
    from sld.experiments.factorial import _FAMILY_BY_PAIR
    from sld.predictors.registry import FAMILIES

    assert "koopman_ssm" in FAMILIES
    assert len(FAMILIES) == 7
    assert set(_FAMILY_BY_PAIR.values()) == set(FAMILIES) - {"koopman_ssm"}
    assert len(set(_FAMILY_BY_PAIR.values())) == 6


def test_factorial_arm_resolving_to_koopman_ssm_is_refused_before_any_run(monkeypatch):
    # NEGATIVE, and deliberately reached by mutating the mapping IN MEMORY:
    # the guard's whole job is to survive someone remapping ("ssm", "on")
    # back to the historical contraction-constrained class -- the exact
    # regression Amendment 4 makes possible. Asserting the guard only
    # through the current (correct) table would test nothing.
    import sld.experiments.factorial as factorial_mod

    monkeypatch.setitem(factorial_mod._FAMILY_BY_PAIR, ("ssm", "on"), "koopman_ssm")
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "ssm", "structure": "on", "overrides": {"delta": 0.02}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="NEVER a cell of the 3x2 factorial"):
        expand(config, dry_run=True)


def test_duplicate_structured_cell_under_alias_is_refused_before_any_run(tmp_path):
    # (ssm, on) declared TWICE, the second arm writing out `r` at exactly
    # the value the Trainer canonicalizes it to (d_z) -- an alias of the
    # first cell, not a second treatment. Both hash to the same run
    # identity, so the preflight must refuse the grid before any fit().
    # This also proves the new families' defaults ARE canonicalized: drop
    # the `r` resolution in Trainer and the two arms hash apart, the
    # collision disappears, and the grid silently trains one cell twice.
    results_root = tmp_path / "results"
    d_z = 2
    config = {
        "exp_name": "alias",
        "datasets": {"ou": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": d_z,
            "d_u": 1,
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 2_000,
            "results_root": str(results_root),
        },
        "arms": [
            {"backbone": "ssm", "structure": "on", "overrides": {"delta": 0.02}},
            {"backbone": "ssm", "structure": "on", "overrides": {"delta": 0.02, "r": d_z}},
        ],
        "seeds": [0],
    }
    config_path = tmp_path / "alias.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="collision"):
        main(["--config", str(config_path)])
    assert not results_root.exists()  # nothing trained: preflight is before fit()


def test_all_six_cells_round_trip_from_config_and_seed(tmp_path):
    """The wave's integration proof, end to end and in one place: the six
    cells are expanded, trained, and READ BACK.

    Covered here and nowhere else together: (1) the runner builds all six
    from config + seed; (2) the six resolve to six DISTINCT run
    directories -- doublons 0, checked on the artifacts rather than on the
    expansion; (3) ``load_run`` reconstructs each one from its checkpoint
    alone and gets back an instance of the class the family names.

    target_params=1100 with d_z=2 is the smallest budget at which all six
    families land inside +/-5% (scanned, not guessed -- at 200 neither
    `transformer` nor `min_ssm` fits, which is a pre-existing property of a
    quadratic width knob at tiny targets, not of the structured arms).
    """
    from sld.experiments.loading import load_run
    from sld.predictors.registry import _FAMILY_CONSTRUCTORS

    results_root = tmp_path / "results"
    config = {
        "exp_name": "sixcells",
        "datasets": {"ou": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": 2,
            "d_u": 1,
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 1_100,
            "encoder_hidden": [8, 8],
            "results_root": str(results_root),
        },
        "arms": [
            {
                "backbone": b,
                "structure": s,
                "overrides": {"mu_j": 0.0, **({"delta": 0.02} if s == "on" else {})},
            }
            for b, s in _SIX_CELLS
        ],
        "seeds": [0],
    }
    config_path = tmp_path / "sixcells.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path)]) == 0

    rows = list(csv.DictReader((results_root / "sixcells" / "index.csv").open()))
    assert len(rows) == 6
    assert {r["status"] for r in rows} == {"complete"}
    assert {r["family"] for r in rows} == set(_SIX_CELLS.values())
    assert len({r["run_dir"] for r in rows}) == 6  # doublons 0, on the artifacts

    for row in rows:
        loaded = load_run(row["run_dir"])
        family = loaded["config"]["family"]
        assert family == row["family"]
        assert isinstance(loaded["predictor"], _FAMILY_CONSTRUCTORS[family])
        print(f"[factorial] round-trip {family} -> {type(loaded['predictor']).__name__}")


def test_load_run_refuses_a_checkpoint_built_by_another_family(tmp_path):
    # NEGATIVE for the round-trip above: load_run rebuilds the architecture
    # from the config and only then loads the weights, so a checkpoint whose
    # config says one family while its tensors came from another must fail
    # loudly. It is the exact accident a hand-edited config or a copied
    # ckpt.pt produces, and a silent partial load would evaluate the wrong
    # model under the right label.
    import torch

    from sld.experiments.loading import load_run
    from sld.predictors.registry import get_predictor

    cfg = {
        "exp_name": "mismatch",
        "seed": 0,
        "family": "mlp_k",
        "d_x": 2,
        "d_z": 2,
        "d_u": 1,
        "delta": 0.02,
        "r": 2,
        "encoder_hidden": [8, 8],
        "target_params": 1_100,
    }
    encoder_sd = {}
    from sld.encoders.frame_mlp import FrameMLPEncoder

    encoder_sd = FrameMLPEncoder(2, 2, (8, 8)).state_dict()
    impostor = get_predictor("ssm_k", 2, 1, target_params=1_100, delta=0.02, r=2)

    run_dir = tmp_path / "mismatch_run"
    run_dir.mkdir()
    torch.save(
        {"encoder": encoder_sd, "predictor": impostor.state_dict(), "config": cfg},
        run_dir / "ckpt.pt",
    )
    with pytest.raises(RuntimeError, match="state_dict"):
        load_run(run_dir)


def test_run_identity_collision_raises(tmp_path):
    # both arms resolve to the SAME (backbone, structure) pair -- (mlp,
    # off) twice -- but the 2nd arm's explicit lr equals the Trainer's own
    # default, so BOTH resolve to the byte-identical hashed config: a real
    # collision, caught by the run-identity preflight (never silently
    # trained twice into the same run_dir).
    results_root = tmp_path / "results"
    config = {
        "exp_name": "collide",
        "datasets": {"ou": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": 2,
            "d_u": 1,
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 200,
            "results_root": str(results_root),
        },
        "arms": [
            {"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}},
            {"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0, "lr": 1e-3}},
        ],
        "seeds": [0],
    }
    config_path = tmp_path / "collide.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="collision"):
        main(["--config", str(config_path)])
    # nothing was trained -- the preflight runs before any fit() call
    assert not results_root.exists()


def test_on_cell_run_mode_refuses_mu_j_positive_but_dry_run_lists_it():
    # mu_j is no longer an experimental axis (structure is architectural),
    # but Trainer still accepts a mu_j knob, and sld-factorial never
    # threads a gauge into it -- so any cell resolving mu_j > 0 trained
    # through here would silently fall back to Trainer's undeclared raw
    # Jacobian estimator instead of the PROTOCOL.md section 10.3 declared,
    # gauge-normalized one. The run path must refuse before touching a
    # dataset or a Trainer; --dry-run must still list the cell, same
    # convention as the null-override case above.
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "ssm", "structure": "on", "overrides": {"mu_j": 0.05}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="gauge"):
        expand(config, dry_run=False)

    cells = expand(config, dry_run=True)
    assert len(cells) == 1
    assert cells[0]["mu_j"] == 0.05


# -- F9: run_complete.json sentinel, never ckpt.pt, drives resume/status -----


def test_resume_skips_only_when_sentinel_present(tmp_path):
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_sentinel.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path)]) == 0

    index_path = results_root / "mini" / "index.csv"
    rows = list(csv.DictReader(index_path.open()))
    run_dir = Path(rows[0]["run_dir"])
    ckpt_path = run_dir / "ckpt.pt"
    sentinel_path = run_dir / "run_complete.json"
    assert ckpt_path.exists() and sentinel_path.exists()

    # remove ONLY the sentinel, keep ckpt.pt: a ckpt.pt without it must
    # never be trusted as "complete" -- the cell is RE-EXECUTED (F9)
    ckpt_mtime_before = ckpt_path.stat().st_mtime_ns
    sentinel_path.unlink()

    rc2 = main(["--config", str(config_path)])
    assert rc2 == 0
    print(f"[factorial] ckpt.pt rewritten after sentinel-only deletion: "
          f"{ckpt_path.stat().st_mtime_ns != ckpt_mtime_before}")
    assert ckpt_path.stat().st_mtime_ns != ckpt_mtime_before
    assert sentinel_path.exists()

    rows2 = list(csv.DictReader(index_path.open()))
    assert rows2[0]["status"] == "complete"


# -- PROTOCOL.md section 11.7: divergence is durable, indexed, never retried -


def test_diverged_cell_is_indexed_as_divergence_not_missing(tmp_path, monkeypatch):
    # a non-finite loss must be a DURABLE, distinguishable result -- not
    # indistinguishable from a cell that was simply never run.
    import torch

    import sld.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan")))

    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "diverge.yaml"
    config_path.write_text(yaml.safe_dump(config))

    rc = main(["--config", str(config_path)])
    assert rc == 1  # the grid did not fully succeed

    index_path = results_root / "mini" / "index.csv"
    rows = list(csv.DictReader(index_path.open()))
    assert len(rows) == 1
    print(f"[factorial] diverged cell status = {rows[0]['status']!r}")
    assert rows[0]["status"] == "divergence"

    run_dir = Path(rows[0]["run_dir"])
    assert (run_dir / "failure.json").exists()
    assert not (run_dir / "run_complete.json").exists()
    status, _ = _cell_status(run_dir)
    assert status == "divergence"


def test_diverged_cell_not_retrained_on_resume(tmp_path, monkeypatch):
    # a cell carrying failure.json is a TERMINAL result (11.7): the next
    # sld-factorial invocation must report and move on, never silently
    # retrain it -- variance_loss stays mocked nan for BOTH invocations, so
    # a retrain would raise again and rewrite failure.json (new mtime).
    import torch

    import sld.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "variance_loss", lambda z: torch.tensor(float("nan")))

    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "diverge_resume.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path)]) == 1
    index_path = results_root / "mini" / "index.csv"
    run_dir = Path(list(csv.DictReader(index_path.open()))[0]["run_dir"])
    failure_path = run_dir / "failure.json"
    assert failure_path.exists()
    mtime_before = failure_path.stat().st_mtime_ns

    rc2 = main(["--config", str(config_path)])
    assert rc2 == 1  # still reported as not-fully-succeeded
    print(f"[factorial] failure.json rewritten by resume: "
          f"{failure_path.stat().st_mtime_ns != mtime_before}")
    assert failure_path.stat().st_mtime_ns == mtime_before  # untouched -- not retrained

    rows2 = list(csv.DictReader(index_path.open()))
    assert rows2[0]["status"] == "divergence"


# -- M9: resume and index share ONE sentinel validator ------------------------


def test_corrupted_sentinel_is_reexecuted_and_never_indexed_complete(tmp_path):
    # M9 (adversarial review round 2): json.loads happily accepts the bare
    # NaN literal Python's json module writes, and an empty {} is valid
    # JSON too -- neither is a trustworthy "complete". A sentinel surviving
    # from a DIFFERENT run identity (wrong hash) must be rejected the same
    # way. In every case: the validator returns None, the index reports a
    # status distinct from "complete", and the NEXT invocation re-executes
    # (never silently resumed) and ends up complete again.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "sentinel_corrupt.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path)]) == 0
    index_path = results_root / "mini" / "index.csv"
    run_dir = Path(list(csv.DictReader(index_path.open()))[0]["run_dir"])
    sentinel_path = run_dir / "run_complete.json"
    good_sentinel = json.loads(sentinel_path.read_text())
    assert _validate_completion_sentinel(run_dir) == good_sentinel  # sanity: happy path validates

    cases = {
        "nan_final_val_loss": {**good_sentinel, "final_val_loss": float("nan")},
        "empty_sentinel": {},
        "wrong_config_hash": {**good_sentinel, "config_hash": "deadbeef"},
    }
    for label, bad_sentinel in cases.items():
        sentinel_path.write_text(json.dumps(bad_sentinel))

        assert _validate_completion_sentinel(run_dir) is None, label
        status, _ = _cell_status(run_dir)
        print(f"[factorial] M9 corrupted sentinel {label!r} -> index status={status!r}")
        assert status != "complete", label

        ckpt_mtime_before = (run_dir / "ckpt.pt").stat().st_mtime_ns
        rc = main(["--config", str(config_path)])
        assert rc == 0
        # RE-EXECUTED, never silently resumed on a corrupted sentinel
        assert (run_dir / "ckpt.pt").stat().st_mtime_ns != ckpt_mtime_before, label

        rows = list(csv.DictReader(index_path.open()))
        assert rows[0]["status"] == "complete", label
        assert _validate_completion_sentinel(run_dir) is not None, label


# -- M3: the sentinel is BOUND to the artifacts it certifies ------------------
# (adversarial review round 3): recomputed digests/hashes, not string
# comparisons or mere existence checks -- see the reviewer's 3 repros below.


def _two_completed_mini_runs(tmp_path) -> tuple[Path, Path]:
    """Two independently-completed mini runs of the SAME shape (same
    family/d_z/d_u/target_params) but DIFFERENT identities (different
    seeds) -- material for the cross-run substitution repros below."""
    results_root = tmp_path / "results"
    config_a = _mini_run_config(results_root)
    config_a["seeds"] = [0]
    config_b = _mini_run_config(results_root)
    config_b["seeds"] = [1]
    path_a, path_b = tmp_path / "run_a.yaml", tmp_path / "run_b.yaml"
    path_a.write_text(yaml.safe_dump(config_a))
    path_b.write_text(yaml.safe_dump(config_b))
    assert main(["--config", str(path_a)]) == 0
    assert main(["--config", str(path_b)]) == 0

    run_dirs = sorted(p for p in (results_root / "mini").iterdir() if p.is_dir())
    assert len(run_dirs) == 2
    return run_dirs[0], run_dirs[1]


def test_swapped_config_yaml_between_runs_invalidates_sentinel(tmp_path):
    # M3 repro (a): the reviewer presented a GENUINE sentinel from run A next
    # to run B's config.yaml (claimed hashes still match the folder name --
    # only the config.yaml BYTES are wrong). Recomputing config_hash from
    # config.yaml's own bytes, not just comparing sentinel strings, catches it.
    run_dir_a, run_dir_b = _two_completed_mini_runs(tmp_path)
    assert _validate_completion_sentinel(run_dir_a) is not None  # sanity

    config_a_text = (run_dir_a / "config.yaml").read_text()
    config_b_text = (run_dir_b / "config.yaml").read_text()
    assert config_a_text != config_b_text  # different seeds -> different config

    (run_dir_a / "config.yaml").write_text(config_b_text)
    assert _validate_completion_sentinel(run_dir_a) is None
    status, _ = _cell_status(run_dir_a)
    print(f"[factorial] M3 swapped config.yaml -> status={status!r}")
    assert status != "complete"


def test_directory_or_zero_byte_ckpt_invalidates_sentinel(tmp_path):
    # M3 repro (b): a zero-byte file OR a directory named ckpt.pt must never
    # satisfy the artifact check -- only ``exists()`` was checked before.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_artifact.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    assert _validate_completion_sentinel(run_dir) is not None  # sanity

    ckpt_path = run_dir / "ckpt.pt"
    ckpt_bytes = ckpt_path.read_bytes()

    ckpt_path.write_bytes(b"")  # zero-byte file
    assert _validate_completion_sentinel(run_dir) is None
    ckpt_path.write_bytes(ckpt_bytes)  # restore -- sanity that it validates again
    assert _validate_completion_sentinel(run_dir) is not None

    ckpt_path.unlink()
    ckpt_path.mkdir()  # directory named ckpt.pt
    assert _validate_completion_sentinel(run_dir) is None
    print("[factorial] M3 directory/zero-byte ckpt.pt both rejected")


def test_substituted_checkpoint_same_shape_invalidates_sentinel(tmp_path):
    # M3 repro (c): a checkpoint from a DIFFERENT run of the SAME shape
    # (same family/d_z/d_u/target_params, different seed -- so it loads
    # fine) swapped into this run_dir must still be rejected: the digest
    # binds the sentinel to ITS OWN ckpt.pt bytes, never merely "a checkpoint
    # of the right shape exists".
    run_dir_a, run_dir_b = _two_completed_mini_runs(tmp_path)
    ckpt_a_bytes = (run_dir_a / "ckpt.pt").read_bytes()
    ckpt_b_bytes = (run_dir_b / "ckpt.pt").read_bytes()
    assert ckpt_a_bytes != ckpt_b_bytes  # different seeds -> different weights

    (run_dir_a / "ckpt.pt").write_bytes(ckpt_b_bytes)
    assert _validate_completion_sentinel(run_dir_a) is None
    print("[factorial] M3 substituted same-shape checkpoint rejected by digest")


def test_metrics_csv_row_count_and_last_epoch_mismatch_invalidates_sentinel(tmp_path):
    # M3: the sentinel's claimed metrics_rows/epochs_completed must match
    # what metrics.csv actually recorded, read back from disk -- not merely
    # parse. A truncated OR a padded metrics.csv are both rejected.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_metrics.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    assert _validate_completion_sentinel(run_dir) is not None  # sanity

    metrics_path = run_dir / "metrics.csv"
    original = metrics_path.read_text()
    lines = original.splitlines(keepends=True)

    metrics_path.write_text("".join(lines[:-1]))  # one row short
    assert _validate_completion_sentinel(run_dir) is None

    metrics_path.write_text(original + lines[-1])  # one row too many (duplicated)
    assert _validate_completion_sentinel(run_dir) is None
    print("[factorial] M3 metrics.csv row-count/last-epoch mismatch rejected")


def _rewrite_last_metrics_row(metrics_path: Path, **column_overrides: str) -> None:
    """Rewrite ``metrics_path``'s LAST data row, replacing named columns.

    Preserves row count and every other row/column untouched -- the M3
    round-4 repros corrupt metrics.csv alone, nothing else.
    """
    with open(metrics_path, newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    header, data_rows = all_rows[0], all_rows[1:]
    last = dict(zip(header, data_rows[-1]))
    last.update(column_overrides)
    data_rows[-1] = [last[col] for col in header]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


def test_nan_in_last_metrics_row_invalidates_sentinel(tmp_path):
    # M3 round 4: corruption confined ENTIRELY to metrics.csv (sentinel JSON
    # untouched, row count and epoch sequence preserved) used to validate
    # clean -- the old check only ever looked at the sentinel's OWN claimed
    # losses for finiteness, never the CSV content actually on disk.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_nan_last_row.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    assert _validate_completion_sentinel(run_dir) is not None  # sanity

    metrics_path = run_dir / "metrics.csv"
    n_rows_before = len(list(csv.DictReader(metrics_path.open())))
    _rewrite_last_metrics_row(metrics_path, train_loss="nan", val_loss="nan")
    n_rows_after = len(list(csv.DictReader(metrics_path.open())))
    assert n_rows_after == n_rows_before  # row count and epoch sequence preserved

    assert _validate_completion_sentinel(run_dir) is None
    status, _ = _cell_status(run_dir)
    print(f"[factorial] M3 nan in last metrics.csv row -> status={status!r}")
    assert status != "complete"

    # The same validator gates resume: a CSV-only nonfinite corruption must
    # be retrained, then regenerated as a complete, finite artifact set.
    ckpt_mtime_before = (run_dir / "ckpt.pt").stat().st_mtime_ns
    rc = main(["--config", str(config_path)])
    print(f"[factorial] M3 nan metrics.csv row -> rerun rc={rc}")
    assert rc == 0
    assert (run_dir / "ckpt.pt").stat().st_mtime_ns != ckpt_mtime_before
    rows = list(csv.DictReader((results_root / "mini" / "index.csv").open()))
    assert rows[0]["status"] == "complete"
    assert _validate_completion_sentinel(run_dir) is not None

    # A trailing cell is not exempt merely because the header remains exact:
    # DictReader represents it under a None key and the entire CSV must be
    # validated, including rejecting hidden nonfinite data.
    metrics_path.write_text(metrics_path.read_text().rstrip("\n") + ",nan\n")
    assert _validate_completion_sentinel(run_dir) is None


def test_finite_but_mismatched_last_row_loss_invalidates_sentinel(tmp_path):
    # M3 round 4: "toute substitution finie passe aussi" -- a FINITE
    # substitution of the last row's val_loss, different from what the
    # sentinel claims, must also be rejected: the sentinel is a claim ABOUT
    # metrics.csv, not an independent source of truth to trust on its own.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_mismatched_last_row.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    assert _validate_completion_sentinel(run_dir) is not None  # sanity

    metrics_path = run_dir / "metrics.csv"
    last_row_before = list(csv.DictReader(metrics_path.open()))[-1]
    substituted = str(float(last_row_before["val_loss"]) + 1.0)  # finite, different
    _rewrite_last_metrics_row(metrics_path, val_loss=substituted)

    assert _validate_completion_sentinel(run_dir) is None
    status, _ = _cell_status(run_dir)
    print(f"[factorial] M3 finite but mismatched last-row val_loss -> status={status!r}")
    assert status != "complete"


# -- M4: decode errors invalidate the sentinel, never crash the caller -------


def test_invalid_utf8_sentinel_reexecutes_without_crash(tmp_path):
    # M4 (adversarial review round 3): an invalid UTF-8 byte in the sentinel
    # itself must never raise UnicodeDecodeError through resume/index
    # generation -- it is treated exactly like a malformed JSON sentinel
    # (invalid -> re-run -> classified, never a crash).
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_badutf8_sentinel.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])

    (run_dir / "run_complete.json").write_bytes(b"\xff\xfe{")
    assert _validate_completion_sentinel(run_dir) is None  # no crash
    status, _ = _cell_status(run_dir)  # no crash
    print(f"[factorial] M4 invalid-UTF-8 sentinel -> status={status!r}")
    assert status != "complete"

    ckpt_mtime_before = (run_dir / "ckpt.pt").stat().st_mtime_ns
    rc = main(["--config", str(config_path)])  # no crash, re-executes
    assert rc == 0
    assert (run_dir / "ckpt.pt").stat().st_mtime_ns != ckpt_mtime_before
    rows = list(csv.DictReader((results_root / "mini" / "index.csv").open()))
    assert rows[0]["status"] == "complete"


def test_invalid_utf8_config_yaml_reexecutes_without_crash(tmp_path):
    # M4: same for config.yaml -- read by the validator to recompute the
    # config hash and check epochs_completed.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_badutf8_cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])

    (run_dir / "config.yaml").write_bytes(b"\xff\xfe{")
    assert _validate_completion_sentinel(run_dir) is None  # no crash
    status, _ = _cell_status(run_dir)  # no crash
    print(f"[factorial] M4 invalid-UTF-8 config.yaml -> status={status!r}")
    assert status != "complete"

    ckpt_mtime_before = (run_dir / "ckpt.pt").stat().st_mtime_ns
    rc = main(["--config", str(config_path)])  # no crash, re-executes
    assert rc == 0
    assert (run_dir / "ckpt.pt").stat().st_mtime_ns != ckpt_mtime_before


def test_oversized_json_integer_in_sentinel_reexecutes_without_crash(tmp_path):
    # M4 round 4: json.loads parses an unquoted 400-digit integer as a
    # regular Python int without complaint -- converting THAT int to float
    # (the finite check) raises OverflowError, not ValueError/TypeError.
    # Before this fix that exception was uncaught and crashed BOTH resume
    # (main()'s sentinel check) and index generation (_cell_status), instead
    # of being classified as "invalid, re-run".
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_giant_int.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    good_sentinel = json.loads((run_dir / "run_complete.json").read_text())

    giant_int = int("1" * 400)
    bad_sentinel = {**good_sentinel, "final_val_loss": giant_int}
    (run_dir / "run_complete.json").write_text(json.dumps(bad_sentinel))

    assert _validate_completion_sentinel(run_dir) is None  # no crash (OverflowError caught)
    status, _ = _cell_status(run_dir)  # no crash
    print(f"[factorial] M4 oversized JSON integer -> status={status!r}")
    assert status != "complete"

    ckpt_mtime_before = (run_dir / "ckpt.pt").stat().st_mtime_ns
    rc = main(["--config", str(config_path)])  # no crash, re-executes
    assert rc == 0
    assert (run_dir / "ckpt.pt").stat().st_mtime_ns != ckpt_mtime_before
    rows = list(csv.DictReader((results_root / "mini" / "index.csv").open()))
    assert rows[0]["status"] == "complete"


# -- M5: an invalid DIRECTORY artifact must be RECOVERABLE, not just rejected


def test_directory_named_ckpt_recovers_via_full_rerun(tmp_path):
    # M5 round 4: the M3 test above (test_directory_or_zero_byte_ckpt_...)
    # already proves a directory named ckpt.pt is REJECTED as an artifact --
    # but Trainer.fit()'s stale-artifact cleanup used to call
    # Path.unlink() unconditionally, which raises IsADirectoryError instead
    # of clearing the way for a retrain. This proves RECOVERY end to end:
    # the runner decides to retrain this cell and it SUCCEEDS.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_dir_ckpt.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    assert _validate_completion_sentinel(run_dir) is not None  # sanity

    ckpt_path = run_dir / "ckpt.pt"
    ckpt_path.unlink()
    ckpt_path.mkdir()
    (ckpt_path / "junk").write_text("not a checkpoint")  # non-empty directory
    assert _validate_completion_sentinel(run_dir) is None  # M3: rejected as unusable

    rc = main(["--config", str(config_path)])
    print(f"[factorial] M5 directory-named ckpt.pt -> rerun rc={rc}")
    assert rc == 0  # retrained WITH SUCCESS, not merely "rejected"
    assert ckpt_path.is_file()
    assert _validate_completion_sentinel(run_dir) is not None

    rows = list(csv.DictReader((results_root / "mini" / "index.csv").open()))
    assert rows[0]["status"] == "complete"


@pytest.mark.parametrize("artifact_name", ("config.yaml", "metrics.csv"))
def test_directory_named_required_file_recovers_via_full_rerun(tmp_path, artifact_name):
    # A bad required-file artifact makes the validator request a retrain.
    # The fresh fit must clear the directory and finish, rather than failing
    # while writing config.yaml or opening metrics.csv.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / f"mini_dir_{artifact_name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])

    artifact_path = run_dir / artifact_name
    artifact_path.unlink()
    artifact_path.mkdir()
    (artifact_path / "junk").write_text("not the required file")
    assert _validate_completion_sentinel(run_dir) is None

    rc = main(["--config", str(config_path)])
    print(f"[factorial] M5 directory-named {artifact_name} -> rerun rc={rc}")
    assert rc == 0
    assert artifact_path.is_file()
    assert _validate_completion_sentinel(run_dir) is not None


@pytest.mark.parametrize("corruption", ("missing", "regular_file"))
def test_missing_or_regular_file_figs_recovers_via_full_rerun(tmp_path, corruption):
    # REGLES section 6 promises figs/ as a directory. It therefore participates
    # in the completion contract: either corruption must invalidate resume,
    # rerun successfully, and restore the directory.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / f"mini_figs_{corruption}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])

    figs_path = run_dir / "figs"
    ckpt_mtime_before = (run_dir / "ckpt.pt").stat().st_mtime_ns
    figs_path.rmdir()
    if corruption == "regular_file":
        figs_path.write_text("not a figures directory")
    assert _validate_completion_sentinel(run_dir) is None

    rc = main(["--config", str(config_path)])
    print(f"[factorial] M6 {corruption} figs/ -> rerun rc={rc}")
    assert rc == 0
    assert (run_dir / "ckpt.pt").stat().st_mtime_ns != ckpt_mtime_before
    assert figs_path.is_dir()
    assert _validate_completion_sentinel(run_dir) is not None


# -- F17: index resilience to interrupted/corrupted runs ---------------------


def test_final_val_loss_tolerates_missing_truncated_and_columnless_metrics(tmp_path):
    assert _final_val_loss(tmp_path / "never_created") == ""  # F17b: no metrics.csv at all

    empty_file = tmp_path / "empty"
    empty_file.mkdir()
    (empty_file / "metrics.csv").write_text("")
    assert _final_val_loss(empty_file) == ""  # zero-byte file

    header_only = tmp_path / "header_only"
    header_only.mkdir()
    (header_only / "metrics.csv").write_text("epoch,train_loss,val_loss\n")
    assert _final_val_loss(header_only) == ""  # header but zero data rows

    no_val_loss_col = tmp_path / "no_col"
    no_val_loss_col.mkdir()
    (no_val_loss_col / "metrics.csv").write_text("epoch,train_loss\n0,0.5\n")
    assert _final_val_loss(no_val_loss_col) == ""  # column missing entirely

    torn_row = tmp_path / "torn_row"
    torn_row.mkdir()
    # header declares 3 columns, the data row supplies only 2 -- simulates a
    # write interrupted mid-row
    (torn_row / "metrics.csv").write_text("epoch,train_loss,val_loss\n0,0.5")
    assert _final_val_loss(torn_row) == ""

    healthy = tmp_path / "healthy"
    healthy.mkdir()
    (healthy / "metrics.csv").write_text("epoch,train_loss,val_loss\n0,0.5,0.3\n1,0.4,0.2\n")
    assert _final_val_loss(healthy) == "0.2"  # sanity: the happy path still works


def test_nonfinite_metrics_value_classified_explicitly(tmp_path):
    nonfinite_dir = tmp_path / "nonfinite_run"
    nonfinite_dir.mkdir()
    (nonfinite_dir / "metrics.csv").write_text("epoch,train_loss,val_loss\n0,0.5,nan\n")
    status, val_loss = _cell_status(nonfinite_dir)  # no sentinel -- interrupted/corrupted
    print(f"[factorial] nonfinite metrics -> status={status!r} val_loss={val_loss!r}")
    assert status == "nonfinite"
    assert val_loss == "nan"  # shown, not silently accepted as a normal number

    finite_incomplete = tmp_path / "finite_incomplete"
    finite_incomplete.mkdir()
    (finite_incomplete / "metrics.csv").write_text("epoch,train_loss,val_loss\n0,0.5,0.3\n")
    status2, val2 = _cell_status(finite_incomplete)
    assert status2 == "missing"  # no sentinel -- not complete, but not corrupt either
    assert val2 == "0.3"

    never_touched = tmp_path / "never_touched"
    status3, val3 = _cell_status(never_touched)
    assert status3 == "missing"
    assert val3 == ""


def test_index_covers_full_grid_despite_uncaught_exception_mid_grid(tmp_path, monkeypatch):
    # F17a/F17d: a bug unrelated to the finite-metrics law (NOT a
    # RuntimeError -- those are already caught per-cell) must not make
    # later cells vanish from the index, and the ORIGINAL exception must
    # still be what propagates out of main() (not masked by a working
    # index write)
    results_root = tmp_path / "results"
    config = {
        "exp_name": "resilience",
        "datasets": {"ou": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": 2,
            "d_u": 1,
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 200,
            "results_root": str(results_root),
        },
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0, 1, 2],
    }
    config_path = tmp_path / "resilience.yaml"
    config_path.write_text(yaml.safe_dump(config))

    original_fit = Trainer.fit
    calls = {"n": 0}

    def _flaky_fit(self, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyError("simulated bug -- not a RuntimeError")
        return original_fit(self, data)

    monkeypatch.setattr(Trainer, "fit", _flaky_fit)

    with pytest.raises(KeyError, match="simulated bug"):
        main(["--config", str(config_path)])

    index_path = results_root / "resilience" / "index.csv"
    rows = list(csv.DictReader(index_path.open()))
    print(f"[factorial] rows after mid-grid crash = {[(r['seed'], r['status']) for r in rows]}")
    assert len(rows) == 3  # ALL 3 cells present, not just the 1 that completed
    statuses = [r["status"] for r in rows]
    assert statuses.count("complete") == 1  # seed=0, finished before the crash
    assert statuses.count("missing") == 2  # seed=1 (crashed) and seed=2 (never attempted)


def test_original_exception_not_masked_by_index_write_failure(tmp_path, monkeypatch):
    # F17d: an exception writing the index inside the failure handler must
    # NEVER replace the original exception from the grid loop
    import sld.experiments.factorial as factorial_mod

    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mask.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def _boom_fit(self, data):
        raise ValueError("boom-from-fit")

    def _boom_index(*args, **kwargs):
        raise OSError("boom-from-index-write")

    monkeypatch.setattr(Trainer, "fit", _boom_fit)
    monkeypatch.setattr(factorial_mod, "_write_index", _boom_index)

    with pytest.raises(ValueError, match="boom-from-fit"):
        main(["--config", str(config_path)])
