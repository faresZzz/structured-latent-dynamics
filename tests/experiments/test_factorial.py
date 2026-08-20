"""Tests for the factorial runner: expansion, closed schema, resume/index.

The mini end-to-end test below exercises ``Trainer.resolve_run_dir`` -- a
method a parallel agent was adding to ``sld.training.trainer.Trainer`` at
the time this file was written (see the task brief). If it has not landed
yet, that one test fails with ``AttributeError: 'Trainer' object has no
attribute 'resolve_run_dir'``; every other test here is independent of it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from sld.encoders.pretrain import (
    load_frozen_representation,
    pretrain_representation,
    representation_digest,
    save_frozen_representation,
)
from sld.experiments.factorial import (
    _build_dataset_for,
    _cell_status,
    _final_val_loss,
    _load_or_build_representation,
    _OPTIONAL_UNDEFAULTED_TRAINER_KEYS,
    _parse_shard,
    _preflight_resolve_and_check_collisions,
    _resolve_stratum_representations,
    _shard_cells,
    _stratum_representation_path,
    _trainer_allowed_keys,
    _validate_completion_sentinel,
    expand,
    main,
)
from sld.training import trainer as trainer_mod
from sld.training.trainer import Trainer

_CONFIGS = Path(__file__).resolve().parents[2] / "src" / "sld" / "experiments" / "configs"

# The deliberately-frozen historical trace (phase B1): kept on the OLD
# family/constraint schema ON PURPOSE, so it must be REFUSED by the current
# validator, never expand cleanly. Named explicitly -- this is the one
# hardcoded list that is allowed, because it is a documented, deliberate
# choice, not "the configs someone has gotten around to fixing". Shared by
# both tests below so the frozen set can never silently drift between what
# "must fail" and what "must pass" each believes it is.
_FROZEN_HISTORICAL_CONFIGS = frozenset({"calibration.yaml", "pendulum_core.yaml"})


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


@pytest.mark.parametrize("config_name", sorted(_FROZEN_HISTORICAL_CONFIGS))
def test_historical_mu_j_configs_are_no_longer_executable(config_name):
    # phase B1: these 2 configs are the frozen trace of the retired mu_J
    # campaign (reports/historical/mu_j/) -- kept on the OLD family/
    # constraint schema ON PURPOSE, which is exactly what makes the new
    # backbone/structure validator refuse them before anything is built.
    config = yaml.safe_load((_CONFIGS / config_name).read_text())
    with pytest.raises(ValueError):
        expand(config, dry_run=False)


@pytest.mark.parametrize(
    "config_name",
    sorted(p.name for p in _CONFIGS.glob("*.yaml") if p.name not in _FROZEN_HISTORICAL_CONFIGS),
)
def test_every_non_frozen_config_expands_cleanly(config_name):
    # the complement of the test above, and the actual guard this wave adds:
    # every config under configs/ that is NOT in the deliberately-frozen
    # historical set must expand() cleanly under the CURRENT schema, in run
    # mode (dry_run=False -- the stricter of the two: it also refuses null
    # overrides and a resolved mu_j > 0). smoke.yaml went stale after
    # Amendment 4 remapped (ssm, on) from koopman_ssm to ssm_k --
    # koopman_ssm's constrained/n_pairs keys are not ssm_k's -- and nothing
    # in the suite noticed until someone ran it by hand. The parametrize
    # list is DISCOVERED from the directory (glob), never hardcoded: a
    # hardcoded "known-good" list would have passed right through that
    # staleness, and would miss the next config anyone adds the same way.
    config = yaml.safe_load((_CONFIGS / config_name).read_text())
    cells = expand(config, dry_run=False)
    print(f"[factorial] {config_name}: expands to {len(cells)} cells")
    assert len(cells) > 0


def test_smoke_config_runs_end_to_end(tmp_path):
    # smoke.yaml's ENTIRE purpose is to be the cheap end-to-end check
    # (Makefile `smoke` target, README.md `make smoke`) -- "expands
    # cleanly" (test_every_non_frozen_config_expands_cleanly, above) is the
    # wrong bar for this ONE config specifically: a dataset/regime param
    # invalid only at BUILD time, or a Trainer crash, sails straight
    # through expand()'s schema check and is only ever caught by actually
    # training. Kept to this one config -- forcing every non-frozen config
    # through main() would be a slow test for no gain; the other configs
    # are not the smoke path's job.
    results_root = tmp_path / "results"
    config = yaml.safe_load((_CONFIGS / "smoke.yaml").read_text())
    config["base"]["results_root"] = str(results_root)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))

    rc = main(["--config", str(config_path)])
    assert rc == 0

    index_path = results_root / "smoke" / "index.csv"
    rows = list(csv.DictReader(index_path.open()))
    print(f"[factorial] smoke.yaml e2e: {[(r['family'], r['structure'], r['status']) for r in rows]}")
    assert len(rows) == 2  # the 2 arms the Makefile/README `make smoke` target expects
    for row in rows:
        assert row["status"] == "complete"
        run_dir = Path(row["run_dir"])
        # the public artifact contract (REGLES section 6), same set
        # factorial.py's own _REQUIRED_ARTIFACTS/_REQUIRED_DIRECTORIES check
        for artifact in ("ckpt.pt", "metrics.csv", "config.yaml", "audit.json", "run_complete.json"):
            assert (run_dir / artifact).exists(), f"{run_dir}/{artifact} missing"
        assert (run_dir / "figs").is_dir()


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


def test_coupled_oscillators_is_a_registered_system():
    # Wave D3 (PROTOCOL.md section 19): CoupledOscillators wired into the
    # factorial's system registry, same convention as DampedPendulum.
    # POSITIVE: a config declaring it expands cleanly and builds a real
    # dataset at its own state_dim (4), not silently coerced to some other
    # shape. NEGATIVE, same test (an unrelated bogus name must still be
    # refused, with the SAME message the registry has always used --
    # widening the accepted set must not weaken the refusal path).
    config = {
        "exp_name": "x",
        "datasets": {
            "ou": _minimal_dataset_entry(
                system={"name": "CoupledOscillators", "params": {}},
                regime={"name": "OURegime", "params": {"theta_ou": 1.0}},
            )
        },
        "base": {"d_x": 4, "d_z": 4, "d_u": 1},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    assert len(cells) == 1
    data = _build_dataset_for(config["datasets"]["ou"])
    assert data["train"]["x"].shape[-1] == 4  # CoupledOscillators.state_dim
    assert data["train"]["u"].shape[-1] == 1  # CoupledOscillators.action_dim
    print("[factorial D3] CoupledOscillators expands + builds a 4-D/1-D dataset")

    with pytest.raises(ValueError, match="unknown system"):
        expand(
            {
                **config,
                "datasets": {"ou": _minimal_dataset_entry(system={"name": "NotASystem", "params": {}})},
            }
        )


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


# -- PROTOCOL.md 17.1 (Amendment 8): closed metrics.csv header generations --
# grad_norm is a NEW column; a run's identity never included the metrics.csv
# SCHEMA, so a run genuinely written before this amendment is complete with
# its own era's header -- trainer_mod.METRICS_HEADER_GENERATIONS is the
# closed tuple of headers this repo has ever written, and the validator
# accepts an EXACT match against any one of them, never a prefix/subset.


def _drop_metrics_column(metrics_path: Path, column: str) -> None:
    """Rewrite ``metrics_path`` with ``column`` removed from the header AND
    every row -- simulates a run genuinely written under an EARLIER
    METRICS_HEADER generation (identical values, one fewer column), never a
    corruption of the columns that remain."""
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        header = [c for c in reader.fieldnames if c != column]
        rows = list(reader)
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in header])


def test_legacy_header_generation_validates_as_complete(tmp_path):
    # positive (hérité): a run genuinely written before Amendment 8 (no
    # grad_norm column) validates as complete -- simulated by stripping
    # grad_norm from an otherwise-genuine complete run's metrics.csv, since
    # Trainer itself always writes the CURRENT generation.
    import sld.training.trainer as trainer_mod

    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_legacy_header.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    assert _validate_completion_sentinel(run_dir) is not None  # sanity: current-gen validates

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, newline="") as f:
        header_before = csv.DictReader(f).fieldnames
    assert "grad_norm" in header_before  # sanity: this run really is current-gen

    _drop_metrics_column(metrics_path, "grad_norm")
    with open(metrics_path, newline="") as f:
        header_after = csv.DictReader(f).fieldnames
    assert header_after == list(trainer_mod.METRICS_HEADER_LEGACY)

    result = _validate_completion_sentinel(run_dir)
    print(f"[factorial] legacy header (no grad_norm) validates: {result is not None}")
    assert result is not None


def test_current_header_generation_validates_as_complete(tmp_path):
    # positive (courant): the header Trainer actually writes today, with
    # grad_norm, validates.
    import sld.training.trainer as trainer_mod

    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_current_header.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])

    with open(run_dir / "metrics.csv", newline="") as f:
        header = csv.DictReader(f).fieldnames
    assert header == list(trainer_mod.METRICS_HEADER)
    assert "grad_norm" in header

    result = _validate_completion_sentinel(run_dir)
    print(f"[factorial] current header (grad_norm) validates: {result is not None}")
    assert result is not None


def test_foreign_header_neither_generation_invalidates_sentinel(tmp_path):
    # negative: a header that is NEITHER generation -- legacy minus its own
    # "cov" column -- is rejected. No prefix/subset match: dropping a SECOND
    # column off the legacy generation does not degrade gracefully into "an
    # even older legacy", it is simply foreign/corrupt.
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_foreign_header.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    metrics_path = run_dir / "metrics.csv"

    _drop_metrics_column(metrics_path, "grad_norm")  # -> legacy generation, validates
    assert _validate_completion_sentinel(run_dir) is not None  # sanity

    _drop_metrics_column(metrics_path, "cov")  # legacy minus ANOTHER column: neither generation
    result = _validate_completion_sentinel(run_dir)
    print(f"[factorial] foreign header (legacy minus cov) invalidates: {result is None}")
    assert result is None


def test_nonfinite_grad_norm_invalidates_sentinel(tmp_path):
    # negative: the finite-metrics law applies to the NEW column too -- a
    # non-finite grad_norm in a current-generation run invalidates, exactly
    # like a non-finite value in any other column (M3 round 4's own rule,
    # now extended by construction since the per-row check reads the header
    # actually found on disk).
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "mini_nonfinite_gradnorm.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(config_path)]) == 0
    run_dir = Path(list(csv.DictReader((results_root / "mini" / "index.csv").open()))[0]["run_dir"])
    metrics_path = run_dir / "metrics.csv"
    n_rows_before = len(list(csv.DictReader(metrics_path.open())))

    _rewrite_last_metrics_row(metrics_path, grad_norm="nan")
    n_rows_after = len(list(csv.DictReader(metrics_path.open())))
    assert n_rows_after == n_rows_before  # row count/epoch sequence preserved

    result = _validate_completion_sentinel(run_dir)
    print(f"[factorial] nonfinite grad_norm invalidates: {result is None}")
    assert result is None


# real, previously-published campaign artifact -- the regression this whole
# fix exists for (coordinator-demonstrated: this run validated before
# METRICS_HEADER gained grad_norm and stopped validating after, until this
# fix). Resolved relative to the repo root, not CWD, so it works regardless
# of where pytest is invoked from.
_REAL_MLP_K_UPDATES_4000_RUN_DIR = (
    Path(__file__).resolve().parents[2]
    / "results_probe_d" / "runs" / "mlp_k__hfr25_updates_4000" / "42cf513a-a2aaac76"
)


@pytest.mark.skipif(
    not _REAL_MLP_K_UPDATES_4000_RUN_DIR.exists(),
    reason="results_probe_d/ campaign artifacts not present on this machine",
)
def test_real_published_mlp_k_run_still_validates_after_header_generations_fix():
    # REAL regression, read-only: PROTOCOL.md 17.1 added grad_norm to
    # METRICS_HEADER, and this REAL, previously-complete round-2 run (written
    # before that amendment) went from validating to
    # _validate_completion_sentinel returning None -- meaning --round3c
    # would have retrained all 37 cells. No write happens in this test.
    result = _validate_completion_sentinel(_REAL_MLP_K_UPDATES_4000_RUN_DIR)
    print(f"[factorial] real mlp_k__hfr25_updates_4000 run validates: {result is not None}")
    assert result is not None


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


# -- D1.5b: sharded concurrent execution (PROTOCOL.md section 14.7) ---------


def _linear_config(n_seeds: int) -> dict:
    """1 dataset x 1 arm x ``n_seeds`` seeds -- cell index == seed, so
    partition/shard assertions below can compare index sets directly."""
    return {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": list(range(n_seeds)),
    }


def test_parse_shard_accepts_valid_i_of_k():
    assert _parse_shard("2/3") == (2, 3)
    assert _parse_shard("1/1") == (1, 1)


def test_shard_cells_none_is_identity():
    cells = expand(_linear_config(3), dry_run=True)
    assert _shard_cells(cells, None) is cells


def test_shard_partition_covers_and_is_disjoint():
    # the property the whole design rests on: for k in {2, 3}, the union of
    # shards 1..k is exactly the full grid, and every pair of shards is
    # disjoint -- by construction (stride partition), never by convention.
    # 7 cells (prime) stresses uneven strides.
    cells = expand(_linear_config(7), dry_run=True)
    all_indices = {c["index"] for c in cells}
    assert all_indices == set(range(7))

    for k in (2, 3):
        shard_index_sets = [
            {c["index"] for c in _shard_cells(cells, (i, k))} for i in range(1, k + 1)
        ]
        union = set().union(*shard_index_sets)
        print(
            f"[factorial] k={k}: shard sizes={[len(s) for s in shard_index_sets]}, "
            f"union size={len(union)} (full grid = {len(all_indices)})"
        )
        assert union == all_indices  # every cell is covered by exactly one shard's union
        for a in range(len(shard_index_sets)):
            for b in range(a + 1, len(shard_index_sets)):
                inter = shard_index_sets[a] & shard_index_sets[b]
                print(f"[factorial] k={k}: shard {a + 1} n shard {b + 1} intersection={len(inter)}")
                assert inter == set()  # pairwise disjoint


def test_shard_flag_absent_dry_run_output_unchanged(tmp_path, capsys):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_linear_config(3)))

    assert main(["--config", str(config_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 4  # 3 cell lines + the "total" line
    assert "total: 3 runs" in out
    for seed in (0, 1, 2):
        assert f"seed={seed}" in out


def test_dry_run_shard_narrows_the_listing(tmp_path, capsys):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_linear_config(3)))

    assert main(["--config", str(config_path), "--dry-run", "--shard", "2/3"]) == 0
    out = capsys.readouterr().out
    print(f"[factorial] dry-run --shard 2/3 output:\n{out}")
    assert out.count("\n") == 2  # 1 cell (index 1, seed=1) + total line
    assert "seed=1" in out
    assert "seed=0" not in out
    assert "seed=2" not in out
    assert "total: 1 runs" in out


def test_sharded_run_writes_full_grid_index_not_just_its_shard(tmp_path):
    # the regression this wave exists to prevent: a shard writing only its
    # own subset would silently drop the OTHER shards' completed runs from
    # the index -- artifacts that exist on disk but that the index denies.
    results_root = tmp_path / "results"
    config = {
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
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0, 1, 2],
    }
    config_path = tmp_path / "shard_index.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # shard 1/3 runs ONLY the seed=0 cell -- seeds 1 and 2 are never touched
    # by THIS process at all
    rc = main(["--config", str(config_path), "--shard", "1/3"])
    assert rc == 0

    index_path = results_root / "mini" / "index.csv"
    rows = list(csv.DictReader(index_path.open()))
    print(f"[factorial] shard 1/3 index rows: {[(r['seed'], r['status']) for r in rows]}")
    assert len(rows) == 3  # the FULL declared grid, not just this shard's 1 cell
    by_seed = {r["seed"]: r for r in rows}
    assert by_seed["0"]["status"] == "complete"
    # seeds 1 and 2 belong to OTHER shards -- never run by this process, but
    # present with their proper non-complete status, never absent
    assert by_seed["1"]["status"] == "missing"
    assert by_seed["2"]["status"] == "missing"


def test_collision_across_shards_caught_by_global_preflight(tmp_path):
    # both arms resolve to the byte-identical hashed config (same pattern as
    # test_run_identity_collision_raises), landing the 2 colliding cells in
    # DIFFERENT shards under k=2 -- a preflight run on the SHARD's own
    # subset alone would never see the other cell and would miss the
    # collision entirely. The preflight must stay global (14.7).
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
    config_path = tmp_path / "collide_shard.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="collision"):
        main(["--config", str(config_path), "--shard", "1/2"])
    assert not results_root.exists()  # nothing trained -- preflight ran before any fit()


def test_atomic_index_write_leaves_no_tmp_residue(tmp_path):
    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "tmp_residue.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path)]) == 0
    index_dir = results_root / "mini"
    leftover = list(index_dir.glob("*.tmp"))
    print(f"[factorial] leftover .tmp files after index write: {leftover}")
    assert leftover == []
    assert (index_dir / "index.csv").exists()


def test_index_write_uses_atomic_replace(tmp_path, monkeypatch):
    # pins the MECHANISM (a race between concurrent shards is not
    # deterministically reproducible in a single-process test): the write
    # must land in a temp file, fully populated, THEN be moved onto
    # index.csv's final name with a single os.replace call -- same
    # discipline as run_complete.json/failure.json in trainer.py. A mutant
    # reverting to a bare open(final_path, "w") calls os.replace zero times
    # for index.csv and reddens this.
    import sld.experiments.factorial as factorial_mod

    results_root = tmp_path / "results"
    config = _mini_run_config(results_root)
    config_path = tmp_path / "atomic.yaml"
    config_path.write_text(yaml.safe_dump(config))

    calls = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        assert Path(src).stat().st_size > 0  # never rename an empty/partial stub
        return real_replace(src, dst)

    monkeypatch.setattr(factorial_mod.os, "replace", _spy_replace)

    assert main(["--config", str(config_path)]) == 0
    index_calls = [(s, d) for s, d in calls if d.name == "index.csv"]
    print(f"[factorial] os.replace calls landing on index.csv: {index_calls}")
    assert len(index_calls) >= 1
    src, dst = index_calls[-1]
    assert src.name == "index.csv.tmp"
    assert src.parent == dst.parent  # same directory -- same filesystem, atomic on POSIX


@pytest.mark.parametrize(
    "bad_value,fragment",
    [
        ("0/3", "1 <= i <= k"),
        ("4/3", "1 <= i <= k"),
        ("abc", "must be i/k"),
        ("3", "must be i/k"),
    ],
)
def test_shard_flag_rejects_malformed_values(tmp_path, capsys, bad_value, fragment):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_linear_config(1)))

    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(config_path), "--dry-run", "--shard", bad_value])
    assert excinfo.value.code == 2  # argparse's own clean CLI rejection, never a traceback
    err = capsys.readouterr().err
    print(f"[factorial] --shard {bad_value!r} rejected: {err.strip().splitlines()[-1]!r}")
    assert fragment in err


# ==========================================================================
# Wave D2c: the frozen shared representation, WIRED into the runner.
# src/sld/encoders/pretrain.py builds/saves/loads the artifact and
# Trainer(config, representation=...) already accepts it (Wave D2b) -- this
# wave is ONLY the wiring: one build-or-load per (system, regime) stratum,
# `stratum_system`/`stratum_regime` injected into each cell of that
# stratum, and the loaded artifact threaded into every cell's OWN
# Trainer(..., representation=...) construction (design decisions 1-5,
# task brief). R7's whole point: an implementation that expands, preflights,
# trains and writes a green index while never passing `representation=` to
# Trainer would pass every test that only checks the code PATH exists --
# the tests below check the call site itself.
# ==========================================================================


def _rep_cfg(**overrides) -> dict:
    cfg = {
        "epochs": 2,
        "lr": 1e-2,
        "seed": 0,
        "w_align": 1.0,
        "w_var": 1.0,
        "w_cov": 1.0,
        "w_recon": 1.0,
        "view_offset": 1,
        "tau_ema": 0.99,
        "d_z": 4,
        "encoder_hidden": [8],
    }
    cfg.update(overrides)
    return cfg


def _repr_run_config(results_root: Path, **overrides) -> dict:
    config = {
        "exp_name": "repr_mini",
        "datasets": {"ou": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": 4,  # must match _rep_cfg()'s d_z: Trainer loads the
            "d_u": 1,  # frozen encoder's state_dict into its OWN, config-
            "epochs": 1,  # built FrameMLPEncoder(d_x, d_z, encoder_hidden)
            "batch_episodes": 2,
            "target_params": 500,
            "encoder_hidden": [8],  # must match _rep_cfg()'s encoder_hidden
            "results_root": str(results_root),
        },
        "representation": _rep_cfg(),
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    config.update(overrides)
    return config


def _prepare_cells(config: dict, results_root: Path):
    """Mirrors ``main()``'s OWN pre-training setup (expand -> build datasets
    -> resolve representations -> preflight) by calling the exact SAME
    private functions ``main()`` calls, in the same order -- a test built on
    this exercises the real call site (REGLES 7.1), not a parallel
    reimplementation that could drift from what ``main()`` actually does."""
    cells = expand(config, dry_run=False)
    exp_name = config["exp_name"]
    datasets = {name: _build_dataset_for(cfg) for name, cfg in config["datasets"].items()}
    representations = (
        _resolve_stratum_representations(config, results_root, exp_name, datasets)
        if config.get("representation") is not None
        else None
    )
    _preflight_resolve_and_check_collisions(cells, datasets, representations)
    return cells, datasets, representations


# -- schema: `representation:` is closed, and its one value-level check -----


# -- Wave D3 / Travail 4 (PROTOCOL.md section 17.1): grad_clip is declarable
# without ever moving the config_hash of a run that predates it -----------


def test_grad_clip_declarable_in_base_and_reaches_the_trainer(tmp_path):
    # POSITIVE: `base: {grad_clip: 100}` -- exactly what
    # configs/positive_control.yaml (section 18.1) declares -- must no
    # longer be rejected as an "unknown Trainer config key", and the value
    # must actually reach the Trainer's OWN resolved config (R7: verify the
    # call site, not the schema alone).
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"d_x": 2, "d_z": 4, "d_u": 1, "grad_clip": 100.0, "results_root": str(tmp_path / "results")},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    assert cells[0]["config"]["grad_clip"] == 100.0
    trainer = Trainer(cells[0]["config"])
    assert trainer.config["grad_clip"] == 100.0
    print(f"[factorial 17.1] grad_clip reaches Trainer.config: {trainer.config['grad_clip']}")


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_grad_clip_bad_value_refused_by_trainer(bad, tmp_path):
    # NEGATIVE: expand() accepts the KEY at the schema level; Trainer itself
    # refuses a non-finite or non-positive VALUE (trainer.py's own grad_clip
    # guard, section 17.1) -- the two checks live at different layers on
    # purpose, and this proves the second one is still reachable through
    # the factorial now that the first one lets the key through.
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"d_x": 2, "d_z": 4, "d_u": 1, "grad_clip": bad, "results_root": str(tmp_path / "results")},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    with pytest.raises(ValueError, match="grad_clip"):
        Trainer(cells[0]["config"])


def test_grad_clip_absent_config_hash_pinned_and_defaults_mutation_reddens(tmp_path, monkeypatch):
    """NEGATIVE + mutation guard for the exact property PROTOCOL.md section
    17.1 paid for: a config that never declares grad_clip must hash EXACTLY
    as every config has since before grad_clip existed. Pinned as a literal
    (same convention as tests/experiments/test_probe.py's "42cf513a" and
    tests/training/test_trainer.py's "49b3b551") -- not re-derived here, so
    a change to _trainer_allowed_keys/_DEFAULTS that moves this hash is
    caught even if no OTHER test happens to exercise this exact config.

    Mutation campaign target named directly by the task brief: "adding
    grad_clip to _DEFAULTS must redden the config_hash stability test". This
    test performs that exact mutation (monkeypatch, scoped to this test
    only) and asserts the hash MOVES -- i.e. this test (and every other
    pinned-hash test in the suite) would fail for real if that mutation ever
    landed outside a test.
    """
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"d_x": 2, "d_z": 4, "d_u": 1, "results_root": str(tmp_path / "results")},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    trainer = Trainer(dict(cells[0]["config"]))
    assert "grad_clip" not in trainer.config
    print(f"[factorial 17.1] no-grad_clip config_hash = {trainer.config_hash}")
    assert trainer.config_hash == "041714e0"

    assert "grad_clip" not in trainer_mod._DEFAULTS  # the property being protected
    monkeypatch.setitem(trainer_mod._DEFAULTS, "grad_clip", 100.0)
    mutant_hash = Trainer(dict(cells[0]["config"])).config_hash
    print(f"[factorial 17.1 mutant] grad_clip added to _DEFAULTS -> config_hash = {mutant_hash}")
    assert mutant_hash != "041714e0"


def test_grad_clip_is_optional_and_undefaulted_not_in_defaults():
    # a direct, structural check alongside the behavioral ones above: the
    # allow-list addition is NOT a _DEFAULTS addition (that distinction is
    # the entire point -- see _OPTIONAL_UNDEFAULTED_TRAINER_KEYS's comment).
    assert "grad_clip" in _OPTIONAL_UNDEFAULTED_TRAINER_KEYS
    assert "grad_clip" not in trainer_mod._DEFAULTS
    assert "grad_clip" in _trainer_allowed_keys({"mlp"})


# -- Wave D3 / Travail 2 (PROTOCOL.md section 19.1): representation.kind ---


def _identity_rep_cfg(**overrides) -> dict:
    cfg = {"kind": "identity", "d_z": 4}
    cfg.update(overrides)
    return cfg


def test_identity_representation_block_accepts_only_kind_and_d_z():
    # POSITIVE: kind="identity" + d_z is a complete, valid block.
    config = {
        "exp_name": "x",
        "datasets": {
            "ou": _minimal_dataset_entry(
                system={"name": "CoupledOscillators", "params": {}},
                regime={"name": "OURegime", "params": {"theta_ou": 1.0}},
            )
        },
        "base": {"d_x": 4, "d_z": 4, "d_u": 1},
        "representation": _identity_rep_cfg(),
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    assert len(cells) == 1
    print("[factorial D3] representation {kind: identity, d_z: 4} validates cleanly")


@pytest.mark.parametrize(
    "pretrained_key",
    sorted(
        {
            "epochs", "lr", "seed", "w_align", "w_var", "w_cov",
            "w_recon", "view_offset", "tau_ema", "encoder_hidden",
        }
    ),
)
def test_identity_representation_block_rejects_pretrained_only_keys(pretrained_key):
    # NEGATIVE: every one of the 10 pretrained-only keys is refused BY NAME
    # in identity mode -- none of them silently validates as "known but
    # inert" (exactly the defect this repo's closed-schema discipline
    # exists to catch).
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "representation": _identity_rep_cfg(**{pretrained_key: 1}),
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match=pretrained_key):
        expand(config, dry_run=False)


def test_identity_representation_block_missing_d_z_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "representation": {"kind": "identity"},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="d_z"):
        expand(config, dry_run=False)


def test_representation_unknown_kind_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "representation": {"kind": "bogus", "d_z": 4},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="kind"):
        expand(config, dry_run=False)


def test_representation_kind_explicit_pretrained_validates_like_kind_absent():
    # an explicit kind="pretrained" must validate EXACTLY like today's
    # kind-absent configs (byte-for-byte the same 11-key schema) -- the
    # default is a spelling of the same thing, not a different path.
    config_absent = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "representation": _rep_cfg(),
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    config_explicit = {**config_absent, "representation": {**_rep_cfg(), "kind": "pretrained"}}
    cells_absent = expand(config_absent, dry_run=False)
    cells_explicit = expand(config_explicit, dry_run=False)
    assert cells_absent[0]["config"] == cells_explicit[0]["config"]


def test_identity_representation_builds_a_real_artifact_end_to_end(tmp_path):
    # POSITIVE, end to end: the SAME private call sequence main() uses
    # (_prepare_cells) with kind="identity" produces an ordinary,
    # digest-validated FrozenRepresentationArtifact -- MEASURED e_recon
    # (float32 noise floor, not exactly 0 and never hand-written -- see
    # build_identity_representation's docstring), e_1_pers and gated
    # computed by the same code the pretrained path uses.
    results_root = tmp_path / "results"
    config = {
        "exp_name": "repr_identity",
        "datasets": {
            "ou": _minimal_dataset_entry(
                system={"name": "CoupledOscillators", "params": {}},
                regime={"name": "OURegime", "params": {"theta_ou": 1.0}},
                n_episodes=10,
                T=10,
            )
        },
        "base": {
            "d_x": 4,
            "d_z": 4,
            "d_u": 1,
            "encoder_hidden": [],  # must match the identity encoder's own architecture
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 500,
            "results_root": str(results_root),
        },
        "representation": _identity_rep_cfg(),
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells, datasets, representations = _prepare_cells(config, results_root)
    artifact = representations["ou"]
    assert artifact.manifest["kind"] == "identity"
    assert artifact.manifest["d_z"] == artifact.manifest["d_x"] == 4
    assert artifact.e_recon == artifact.e_recon  # not NaN
    assert artifact.e_recon < 1e-5  # far below D3's 1e-3 bound (float32 noise floor)
    assert artifact.manifest["e_1_pers"] == artifact.manifest["e_1_pers"]  # not NaN
    assert artifact.manifest["gated"] == (3.0 * artifact.e_recon < artifact.manifest["e_1_pers"])
    print(
        f"[factorial D3] identity artifact e_recon={artifact.e_recon:.3e} "
        f"e_1_pers={artifact.manifest['e_1_pers']:.3e} gated={artifact.manifest['gated']}"
    )

    # the SAME Trainer construction main() would do, unmodified: the trainer
    # neither knows nor cares this representation is identity (item 3's
    # "n'ouvre pas un chemin parallèle" requirement) -- same class, same
    # validate_shared_representation preflight, same representation_digest
    # derivation into the run identity.
    trainer = cells[0]["trainer"]
    assert trainer.representation is artifact
    assert trainer.config["representation_digest"] == artifact.manifest["digest"]


def test_representation_kind_absent_dispatches_to_pretrain_representation_unchanged(tmp_path, monkeypatch):
    """BACKWARD COMPATIBILITY (item 1, the property the task brief asked to
    be proven rather than assumed): a `representation:` block with no
    "kind" key -- every config written before this key existed -- must
    still call pretrain_representation with EXACTLY the same 8 keyword
    arguments as before this wave, so its build is byte-for-byte what it
    always was and no existing config_hash/run_dir moves. Verified at the
    REAL call site (R7, same discipline as
    test_load_or_build_representation_save_path_is_atomic_replace's os.replace
    spy just above this section), not re-derived from reading the source.
    """
    import sld.experiments.factorial as factorial_mod

    calls = []
    real_pretrain = factorial_mod.pretrain_representation

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real_pretrain(*args, **kwargs)

    monkeypatch.setattr(factorial_mod, "pretrain_representation", _spy)

    ds_cfg = _minimal_dataset_entry(n_episodes=10, T=10)
    data = _build_dataset_for(ds_cfg)
    rep_cfg = _rep_cfg()  # NO "kind" key -- exactly what every pre-D3 config carries
    assert "kind" not in rep_cfg
    path = tmp_path / "reps" / "legacy.pt"

    _load_or_build_representation(
        path, system="DampedPendulum", regime="OURegime", data=data, rep_cfg=rep_cfg
    )

    assert len(calls) == 1
    assert calls[0] == {
        "system": "DampedPendulum",
        "regime": "OURegime",
        "d_z": rep_cfg["d_z"],
        "encoder_hidden": tuple(rep_cfg["encoder_hidden"]),
        "seed": rep_cfg["seed"],
        "epochs": rep_cfg["epochs"],
        "lr": rep_cfg["lr"],
        "tau_ema": rep_cfg["tau_ema"],
        "view_offset": rep_cfg["view_offset"],
        "w_align": rep_cfg["w_align"],
        "w_var": rep_cfg["w_var"],
        "w_cov": rep_cfg["w_cov"],
        "w_recon": rep_cfg["w_recon"],
    }
    print(
        "[factorial D3 backward-compat] kind-absent dispatch calls "
        "pretrain_representation with the pre-D3 kwargs, unchanged"
    )


_REAL_RESULTS_AMENDED_PENDULUM_CORE_DIR = (
    Path(__file__).resolve().parents[2] / "results_amended" / "pendulum_core"
)


@pytest.mark.skipif(
    not _REAL_RESULTS_AMENDED_PENDULUM_CORE_DIR.exists(),
    reason="results_amended/pendulum_core artifacts not present on this machine",
)
def test_results_amended_pendulum_core_forty_run_dirs_still_self_consistent():
    """Investigation note (task brief asked for a test reconstructing
    results_amended/pendulum_core's 40 run-dir identities): those 40 dirs
    were written by configs/pendulum_core.yaml, the FROZEN pre-B1
    family/constraint config -- which
    test_historical_mu_j_configs_are_no_longer_executable (above) already
    asserts expand() REFUSES outright, and has refused since phase B1, well
    before this wave's `representation:` schema (_validate_representation_block,
    _load_or_build_representation) existed at all. No config path this wave
    touches can reach these 40 identities, since pendulum_core.yaml declares
    no `representation:` block for that code to ever see.

    What IS checked here, read-only, straight off disk: every one of the 40
    run dirs' own config.yaml still hashes (sha1, first 8 hex chars --
    Trainer.__init__'s own recipe, trainer.py, UNTOUCHED by this wave) to
    the config_hash half of its own directory name. Trainer.py's hashing was
    not modified by this wave, so this property holds exactly as it did
    before -- this test pins it as a literal, measured fact instead of an
    assumption.
    """
    results_dir = _REAL_RESULTS_AMENDED_PENDULUM_CORE_DIR
    run_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    assert len(run_dirs) == 40, f"expected 40 run dirs, found {len(run_dirs)}: {run_dirs}"
    for run_dir in run_dirs:
        config_hash, sep, data_hash = run_dir.name.partition("-")
        assert sep and len(config_hash) == 8 and len(data_hash) == 8, run_dir.name
        config_text = (run_dir / "config.yaml").read_text()
        recomputed = hashlib.sha1(config_text.encode("utf-8")).hexdigest()[:8]
        assert recomputed == config_hash, run_dir
    print(
        f"[backward-compat] {len(run_dirs)}/40 results_amended/pendulum_core "
        "run dirs self-consistent (read-only, unreachable by this wave's "
        "representation.kind schema -- see docstring)"
    )


def test_unknown_key_in_representation_block_raises_before_any_dataset_built():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "representation": {**_rep_cfg(), "w_alignn": 1.0},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="w_alignn"):
        expand(config, dry_run=False)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1", True])
def test_representation_view_offset_below_one_is_refused(bad):
    # view_offset is a REAL parameter of pretrain_representation now, so any
    # integer >= 1 is legal. Offset 0 is refused ON PURPOSE and is NOT the
    # pure-autoencoder ablation: the teacher is an EMA of the student, so
    # pairing a frame with ITSELF still yields a non-zero alignment term --
    # that is BYOL on identical views, the CONFIGURATION THAT COLLAPSES.
    # The ablation is w_align = 0 (PROTOCOL.md section 14.11).
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "representation": _rep_cfg(view_offset=bad),
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="view_offset"):
        expand(config, dry_run=False)


def test_representation_view_offset_two_is_accepted_and_changes_the_artifact(tmp_path):
    # the axis PROTOCOL.md section 14.11 puts on the probe must actually be
    # executable, and must actually change what is produced -- a knob that
    # is accepted but inert is the defect this project keeps paying for.
    results_root = tmp_path / "results"
    digests = {}
    for offset in (1, 2):
        config = _repr_run_config(
            results_root, exp_name=f"offset{offset}", datasets={"ou": _minimal_dataset_entry(n_episodes=10, T=10)}
        )
        config["representation"]["view_offset"] = offset
        _cells, _datasets, reps = _prepare_cells(config, results_root)
        digests[offset] = reps["ou"].manifest["digest"]
        assert reps["ou"].manifest["view_offset"] == offset
    print(f"[factorial D2c] view_offset digests: {digests}")
    assert digests[1] != digests[2]


# -- no `representation:` block: byte-for-byte unchanged ---------------------


def test_no_representation_block_leaves_cell_config_untouched():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    assert len(cells) == 1
    assert "stratum_system" not in cells[0]["config"]
    assert "stratum_regime" not in cells[0]["config"]


def test_no_representation_block_pins_the_same_config_hash_as_before_this_wave(tmp_path):
    # POSITIVE, the identity regression this wave is riskiest on (same
    # doctrine as test_trainer.py's D2b pin): a config with no
    # `representation:` block must resolve to the IDENTICAL config_hash a
    # pre-Wave-D2c factorial.py would produce. Pinned as a literal so a
    # future change to the wiring cannot silently move every existing run's
    # identity.
    config = {
        "exp_name": "hash_pin",
        "datasets": {"d1": _minimal_dataset_entry(n_episodes=10, T=10)},
        "base": {
            "d_x": 2,
            "d_z": 2,
            "d_u": 1,
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 200,
            "results_root": str(tmp_path / "results"),
        },
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    cells = expand(config, dry_run=False)
    trainer = Trainer(cells[0]["config"])
    print(f"[factorial D2c] no-representation config_hash = {trainer.config_hash}")
    assert trainer.config_hash == "8c438faa"


# -- the wiring test, the point of this suite (R7) --------------------------


def test_representation_wired_into_trainer_at_the_call_site(tmp_path):
    """Not "the code path exists" -- the Trainer object main()'s own
    preflight constructs actually CARRIES the loaded artifact, by identity,
    and that artifact's digest is the one recorded in the run's own
    config.yaml/ckpt.pt. The unwired defect this wave exists to prevent
    (``Trainer(cell["config"])`` with no ``representation=`` kwarg) would
    still expand/preflight/train/write a green index -- every assertion
    below is exactly what that defect would fail.
    """
    results_root = tmp_path / "results"
    config = _repr_run_config(results_root)
    cells, datasets, representations = _prepare_cells(config, results_root)

    assert len(cells) == 1
    cell = cells[0]
    trainer = cell["trainer"]
    expected_artifact = representations["ou"]

    # the call-site assertion itself: IDENTITY, not merely "a" representation
    assert trainer.representation is expected_artifact
    assert trainer.config["representation_digest"] == representation_digest(expected_artifact)

    result = trainer.fit(datasets["ou"])
    config_yaml_text = Path(result["paths"]["config"]).read_text()
    assert f"representation_digest: {representation_digest(expected_artifact)}" in config_yaml_text
    ckpt = torch.load(result["paths"]["ckpt"], map_location="cpu", weights_only=False)
    assert ckpt["config"]["representation_digest"] == representation_digest(expected_artifact)


# -- the digest table: one stratum, one digest; two strata, two digests -----


def test_representation_digest_shared_within_stratum_differs_across_strata(tmp_path):
    results_root = tmp_path / "results"
    config = _repr_run_config(
        results_root,
        exp_name="digest_table",
        datasets={
            "ou_a": _minimal_dataset_entry(n_episodes=10, T=10),
            # SAME stratum NAMES as ou_a (DampedPendulum/OURegime) but
            # DIFFERENT params: these are genuinely different data, so they
            # must NOT share an encoder. Keying by names alone would have
            # given ou_b an encoder pretrained on ou_a's data -- half the
            # cells training against a representation they never saw.
            "ou_b": _minimal_dataset_entry(
                n_episodes=10, T=10, regime={"name": "OURegime", "params": {"theta_ou": 5.0}}
            ),
            "aniso": _minimal_dataset_entry(
                n_episodes=10,
                T=10,
                regime={"name": "AnisotropicRegime", "params": {"ic_scales": [1.0, 1.0]}},
            ),
        },
        arms=[{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
    )
    cells, datasets, representations = _prepare_cells(config, results_root)
    assert len(cells) == 3
    digests = {cell["dataset"]: cell["trainer"].config["representation_digest"] for cell in cells}
    print(f"[factorial D2c] digest table: {digests}")
    # same names, different data => DIFFERENT representations (the footgun)
    assert digests["ou_a"] != digests["ou_b"]
    assert representations["ou_a"] is not representations["ou_b"]
    # a different stratum name => different too
    assert digests["aniso"] != digests["ou_a"]


# -- idempotence: 2nd invocation loads, never rebuilds -----------------------


def test_second_invocation_loads_existing_artifact_zero_pretrain_calls(tmp_path, monkeypatch):
    import sld.experiments.factorial as factorial_mod

    results_root = tmp_path / "results"
    config = _repr_run_config(results_root)
    _cells1, _datasets1, representations1 = _prepare_cells(config, results_root)
    assert representations1["ou"].manifest.get("digest")  # first invocation built + saved

    calls = []
    real_pretrain = factorial_mod.pretrain_representation

    def _counting_pretrain(*args, **kwargs):
        calls.append((args, kwargs))
        return real_pretrain(*args, **kwargs)

    monkeypatch.setattr(factorial_mod, "pretrain_representation", _counting_pretrain)

    _cells2, _datasets2, representations2 = _prepare_cells(config, results_root)
    print(f"[factorial D2c] pretrain_representation calls on 2nd invocation: {len(calls)}")
    assert len(calls) == 0
    assert representations2["ou"].manifest["digest"] == representations1["ou"].manifest["digest"]


# -- shard-race safety: simulated interleaving, not a docstring claim -------


def test_shard_race_concurrent_builds_are_atomic_and_idempotent(tmp_path, monkeypatch):
    """Simulates the --shard interleaving design decision 3 declares safe:
    two 'shards' both find no artifact on disk and BOTH pretrain
    independently -- the actual race window is BEFORE either has saved.
    Pretraining is deterministic given the same data/seed, so their
    in-memory artifacts hash identically; ``save_frozen_representation``
    writes temp-file + ``os.replace``, so whichever save physically lands
    first leaves a COMPLETE file and the other's save (also atomic) simply
    replaces it with byte-identical content -- no reader of ``path``, at any
    instant, ever observes a torn file.
    """
    import sld.encoders.pretrain as pretrain_mod

    ds_cfg = _minimal_dataset_entry(n_episodes=10, T=10)
    data = _build_dataset_for(ds_cfg)
    rep_cfg = _rep_cfg()
    path = tmp_path / "reps" / "DampedPendulum__OURegime.pt"
    path.parent.mkdir(parents=True)

    def _build():
        return pretrain_representation(
            data,
            system="DampedPendulum",
            regime="OURegime",
            d_z=rep_cfg["d_z"],
            encoder_hidden=tuple(rep_cfg["encoder_hidden"]),
            seed=rep_cfg["seed"],
            epochs=rep_cfg["epochs"],
            lr=rep_cfg["lr"],
            tau_ema=rep_cfg["tau_ema"],
            w_align=rep_cfg["w_align"],
            w_var=rep_cfg["w_var"],
            w_cov=rep_cfg["w_cov"],
            w_recon=rep_cfg["w_recon"],
        )

    # both shards build BEFORE either saves -- the real race window
    artifact_shard_a = _build()
    artifact_shard_b = _build()

    replace_calls = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        assert Path(src).stat().st_size > 0  # never rename a partial write
        replace_calls.append(Path(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(pretrain_mod.os, "replace", _spy_replace)

    # shard B "wins": its atomic save lands first
    digest_b = save_frozen_representation(artifact_shard_b, path)
    # shard A saves SECOND, onto the SAME path -- an atomic overwrite, never
    # a moment where a reader could see a torn file
    digest_a = save_frozen_representation(artifact_shard_a, path)

    print(f"[factorial D2c] shard race: digest_a={digest_a} digest_b={digest_b} replace_calls={replace_calls}")
    assert digest_a == digest_b  # deterministic pretraining -> identical bytes
    assert len(replace_calls) == 2  # both writes went through the atomic path
    assert all(p == path for p in replace_calls)

    # a THIRD shard, arriving after the dust settles, gets a valid artifact
    # for the stratum regardless of which write physically landed last
    loaded = _load_or_build_representation(
        path, system="DampedPendulum", regime="OURegime", data=data, rep_cfg=rep_cfg
    )
    assert representation_digest(loaded) == digest_a


def test_load_or_build_representation_save_path_is_atomic_replace(tmp_path, monkeypatch):
    # pins the MECHANISM at the exact call site build-then-save happens
    # inside _load_or_build_representation: a kill mid-write must never
    # leave a partial artifact at `path` -- same discipline as
    # run_complete.json/index.csv (trainer.py / factorial.py). A mutant that
    # bypasses save_frozen_representation's temp-file + os.replace and
    # writes the final payload straight to `path` (e.g. a bare
    # torch.save(payload, path)) calls os.replace ZERO times here and
    # reddens this.
    import sld.encoders.pretrain as pretrain_mod

    ds_cfg = _minimal_dataset_entry(n_episodes=10, T=10)
    data = _build_dataset_for(ds_cfg)
    rep_cfg = _rep_cfg()
    path = tmp_path / "reps" / "atomic.pt"

    calls = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        assert Path(src).stat().st_size > 0  # never rename an empty/partial stub
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(pretrain_mod.os, "replace", _spy_replace)

    _load_or_build_representation(
        path, system="DampedPendulum", regime="OURegime", data=data, rep_cfg=rep_cfg
    )
    print(f"[factorial D2c] os.replace calls during build-and-save: {calls}")
    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == path
    assert src.name == path.name + ".tmp"
    assert src.parent == dst.parent  # same directory -- same filesystem, atomic on POSIX


# -- discoverability: path, digest, stratum all traceable after a run -------


def test_representation_path_digest_and_stratum_are_discoverable_after_a_run(tmp_path):
    results_root = tmp_path / "results"
    config = _repr_run_config(results_root)
    config_path = tmp_path / "repr.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path)]) == 0

    # discovered from disk, not reconstructed: the path carries the stratum
    # names AND the data fingerprint, and exactly one artifact exists here
    rep_dir = results_root / config["exp_name"] / "representations"
    found = sorted(rep_dir.glob("DampedPendulum__OURegime__*.pt"))
    assert len(found) == 1, sorted(pp.name for pp in rep_dir.iterdir())
    expected_path = found[0]
    # and the name is exactly the scheme _stratum_representation_path builds
    data_hash = expected_path.stem.rsplit("__", 1)[1]
    assert expected_path == _stratum_representation_path(
        results_root, config["exp_name"], "DampedPendulum", "OURegime", data_hash
    )
    artifact = load_frozen_representation(expected_path)  # readable, valid
    digest = artifact.manifest["digest"]
    assert artifact.manifest["system"] == "DampedPendulum"
    assert artifact.manifest["regime"] == "OURegime"

    index_path = results_root / config["exp_name"] / "index.csv"
    rows = list(csv.DictReader(index_path.open()))
    assert len(rows) == 1
    run_dir = Path(rows[0]["run_dir"])
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert run_config["representation_digest"] == digest
    assert run_config["stratum_system"] == "DampedPendulum"
    assert run_config["stratum_regime"] == "OURegime"


# -- negatives: exception type + distinctive message fragment ---------------


def test_stale_artifact_with_wrong_stratum_on_disk_is_refused(tmp_path):
    ds_cfg = _minimal_dataset_entry(n_episodes=10, T=10)
    data = _build_dataset_for(ds_cfg)
    rep_cfg = _rep_cfg()
    path = tmp_path / "wrong.pt"
    # an artifact genuinely built (and saved) for a DIFFERENT stratum
    _load_or_build_representation(
        path, system="OtherSystem", regime="OtherRegime", data=data, rep_cfg=rep_cfg
    )
    with pytest.raises(ValueError, match="stratum mismatch"):
        _load_or_build_representation(
            path, system="DampedPendulum", regime="OURegime", data=data, rep_cfg=rep_cfg
        )


def test_corrupt_artifact_on_disk_is_refused_never_silently_rebuilt(tmp_path, monkeypatch):
    ds_cfg = _minimal_dataset_entry(n_episodes=10, T=10)
    data = _build_dataset_for(ds_cfg)
    rep_cfg = _rep_cfg()
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch checkpoint")

    import sld.experiments.factorial as factorial_mod

    calls = []
    real_pretrain = factorial_mod.pretrain_representation
    monkeypatch.setattr(
        factorial_mod,
        "pretrain_representation",
        lambda *a, **k: (calls.append(1), real_pretrain(*a, **k))[1],
    )

    with pytest.raises(RuntimeError, match="corrupt or unreadable"):
        _load_or_build_representation(
            path, system="DampedPendulum", regime="OURegime", data=data, rep_cfg=rep_cfg
        )
    print(f"[factorial D2c] pretrain_representation calls after a corrupt file: {calls}")
    assert calls == []  # refused, never silently rebuilt over


def test_cell_stratum_mismatch_against_loaded_artifact_refused_before_first_batch(tmp_path):
    # simulates the exact defect this wave exists to prevent from ever
    # reaching Trainer.fit's first batch: a representation for the WRONG
    # stratum wired in as if it were "ou"'s. The preflight (construction of
    # every cell's Trainer, BEFORE the run loop) must refuse it.
    results_root = tmp_path / "results"
    config = _repr_run_config(results_root)
    cells = expand(config, dry_run=False)
    datasets = {name: _build_dataset_for(cfg) for name, cfg in config["datasets"].items()}
    wrong_artifact = _load_or_build_representation(
        results_root / "wrong.pt",
        system="NotDampedPendulum",
        regime="NotOURegime",
        data=datasets["ou"],
        rep_cfg=config["representation"],
    )
    with pytest.raises(ValueError, match="stratum mismatch"):
        _preflight_resolve_and_check_collisions(cells, datasets, {"ou": wrong_artifact})
