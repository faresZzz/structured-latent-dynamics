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


def test_pendulum_core_dry_run_expands_to_80():
    config = yaml.safe_load((_CONFIGS / "pendulum_core.yaml").read_text())
    cells = expand(config, dry_run=True)
    assert len(cells) == 80  # 2 datasets x 8 arms (4 families x off/on) x 5 seeds
    # deterministic order: datasets in YAML order, then arms, then seeds
    # (seed is the innermost loop -- 5 consecutive cells per (dataset, arm))
    assert [c["dataset"] for c in cells[:40]] == ["ou"] * 40
    assert [c["dataset"] for c in cells[40:]] == ["action_conditioned"] * 40
    assert [c["seed"] for c in cells[:5]] == [0, 1, 2, 3, 4]
    first_cell = (cells[0]["family"], cells[0]["constraint"])
    assert all((c["family"], c["constraint"]) == first_cell for c in cells[:5])
    assert (cells[5]["family"], cells[5]["constraint"]) != first_cell


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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="unknown system"):
        expand(config)


def test_null_override_raises_in_run_mode_but_not_dry_run():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "on", "overrides": {"mu_j": None}}],
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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0, **arm_overrides}}],
        "seeds": [0],
    }


# -- F7: closed schema over base AND every arm's overrides -------------------


def test_unknown_key_in_base_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "base": {"lrr": 0.1},
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="lrr"):
        expand(config)


def test_unknown_key_in_overrides_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0, "mu_jj": 1.0}}],
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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="injected by the runner"):
        expand(base_leak)

    overrides_leak = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0, "family": "mlp"}}],
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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="capacity"):
        expand(budget_in_base)

    derived_in_overrides = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0, "d_h": 4}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="derived or policy"):
        expand(derived_in_overrides)


# -- F8: constraint label semantics + run-identity collision preflight -------


def test_invalid_constraint_value_raises():
    config = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "maybe", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="constraint"):
        expand(config)


def test_constraint_mu_j_semantic_mismatch_raises():
    # 'off' with a positive mu_j -- the exact scenario the reviewer used to
    # make two arms collide (F8's repro): off/on both effectively mu_j=0
    off_with_positive_mu_j = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.5}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="'off'"):
        expand(off_with_positive_mu_j)

    on_with_zero_mu_j = {
        "exp_name": "x",
        "datasets": {"d1": _minimal_dataset_entry()},
        "arms": [{"family": "mlp", "constraint": "on", "overrides": {"mu_j": 0.0}}],
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="'on'"):
        expand(on_with_zero_mu_j)


def test_run_identity_collision_raises(tmp_path):
    # both arms are legitimately 'on' (mu_j=0.05 > 0, F8b is satisfied) --
    # but the 2nd arm's explicit lr equals the Trainer's own default, so
    # BOTH resolve to the byte-identical hashed config: a real collision the
    # F8b semantic check does NOT catch, caught by the preflight instead
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
            {"family": "mlp", "constraint": "on", "overrides": {"mu_j": 0.05}},
            {"family": "mlp", "constraint": "on", "overrides": {"mu_j": 0.05, "lr": 1e-3}},
        ],
        "seeds": [0],
    }
    config_path = tmp_path / "collide.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="collision"):
        main(["--config", str(config_path)])
    # nothing was trained -- the preflight runs before any fit() call
    assert not results_root.exists()


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
        "arms": [{"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0}}],
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
