"""PROTOCOL 28.2: the frozen basis shipped in `basis_root.yaml`, audited.

The config carries `T` as a LITERAL rather than recomputing it, because
`numpy.linalg.eig` does not guarantee eigenvector scaling across LAPACK builds
and the latent scale -- unlike the block form -- is NOT invariant under
per-block rescaling. A frozen literal is reproducible; a recomputed one is not.

The price of freezing a literal is that nothing checks it. This file is that
check. Every property 28.2 claims of `T` is RECOMPUTED here from the config's
own bytes, so the section cannot drift from the matrix it describes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from sld.encoders.pretrain import (
    build_identity_representation,
    load_frozen_representation,
    save_frozen_representation,
)
from sld.experiments.factorial import _build_dataset_for, expand
from sld.predictors.structured import MLPKPredictor
from sld.systems.coupled_oscillators import CoupledOscillators
from sld.systems.representability import representability_distance
from sld.training import trainer as trainer_mod
from sld.training.budget import count_params, match_budget
from sld.training.trainer import Trainer

CONFIG = Path(__file__).resolve().parents[1] / "src/sld/experiments/configs/basis_root.yaml"
PRIMARY_FACTORIAL_CONFIG = (
    Path(__file__).resolve().parents[1] / "src/sld/experiments/configs/primary_factorial.yaml"
)
RESULTS_PRIMARY_FACTORIAL = Path(__file__).resolve().parents[1] / "results_primary_factorial"

# 28.2's declared sha256 over T's float64 bytes.
DECLARED_SHA256 = "e026f402194929b09fdb6c7c88fe801e5d945131daa537947cd13ac7b8736be5"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def T(cfg) -> np.ndarray:
    oracle = [a for a in cfg["arms"] if a["overrides"].get("lift") == "frozen"]
    assert len(oracle) == 1, "exactly one oracle arm"
    return np.array(oracle[0]["overrides"]["psi_matrix"], dtype=float)


# -- the grid ---------------------------------------------------------------


def test_the_grid_is_the_one_28_2_declares(cfg):
    """1 stratum x 3 arms x 5 seeds = 15 (28.2)."""
    assert cfg["exp_name"] == "basis_root"
    assert cfg["base"]["results_root"] == "results_basis"
    assert len(cfg["datasets"]) * len(cfg["arms"]) * len(cfg["seeds"]) == 15


def test_the_stratum_is_27_1_s_verbatim(cfg):
    """Commensurability with 27.8.3 is the reason this root can be compared to
    it at all; a stratum that drifted would silently break that."""
    ds = cfg["datasets"]["c010_n32"]
    assert ds["system"]["name"] == "CoupledOscillators"
    assert ds["regime"] == {"name": "OURegime", "params": {"theta_ou": 1.0}}
    assert (ds["n_episodes"], ds["T"], ds["dt"], ds["seed"]) == (32, 120, 0.05, 0)
    assert cfg["base"]["h_fr"] == 25
    assert cfg["base"]["epochs"] == 8000
    assert cfg["base"]["target_params"] == 10000
    assert cfg["representation"]["kind"] == "identity"


def test_the_third_arm_differs_from_the_second_by_the_lift_alone(cfg):
    """28.2: same rate, dataset, seed, budget, width, psi_hidden. If any other
    key differed, the contrast would attribute nothing."""
    _, learned, oracle = cfg["arms"]
    assert learned["backbone"] == oracle["backbone"] == "mlp"
    assert learned["structure"] == oracle["structure"] == "on"
    a, b = dict(learned["overrides"]), dict(oracle["overrides"])
    assert b.pop("lift") == "frozen"
    b.pop("psi_matrix")
    assert a == b, f"arms differ beyond the lift: {set(a.items()) ^ set(b.items())}"


def test_every_arm_shares_one_rate(cfg):
    """23.5(2) forbids a second sweep: the oracle arm inherits, never selects."""
    assert {a["overrides"]["lr"] for a in cfg["arms"]} == {0.001}


# -- the frozen matrix ------------------------------------------------------


def test_T_matches_its_declared_hash(T):
    assert hashlib.sha256(T.tobytes()).hexdigest() == DECLARED_SHA256


def test_T_block_diagonalises_the_true_generator(T):
    """The one property that makes this arm an ORACLE rather than a guess."""
    A = CoupledOscillators(c=0.10).A_mat
    M = T @ A @ np.linalg.inv(T)
    assert representability_distance(M)[0] < 1e-12
    assert np.abs(M[:2, 2:]).max() < 1e-12
    assert np.abs(M[2:, :2]).max() < 1e-12
    # 26.0's two pairs, read off the block diagonal
    assert M[0, 0] == pytest.approx(-0.05, abs=1e-9)
    assert M[2, 2] == pytest.approx(-0.05, abs=1e-9)
    assert abs(M[0, 1]) == pytest.approx(1.5187, abs=5e-4)
    assert abs(M[2, 3]) == pytest.approx(1.0902, abs=5e-4)


def test_T_carries_the_conditioning_28_2_publishes(T):
    s = np.linalg.svd(T, compute_uv=False)
    np.testing.assert_allclose(np.sort(s), [1.1966, 1.3494, 1.4894, 1.8208], atol=5e-4)
    assert float(np.linalg.cond(T)) == pytest.approx(1.5216, abs=5e-4)


def test_T_does_not_hand_the_oracle_arm_a_scale_advantage(T):
    """28.2's declared normalisation: the latent scale stays within 1.5x of the
    identity's. A basis that also rescaled the latent would confound the
    contrast with conditioning."""
    z = np.random.default_rng(0).standard_normal((20_000, 4))
    ratio = np.linalg.norm(z @ T.T, axis=1).mean() / np.linalg.norm(z, axis=1).mean()
    assert ratio == pytest.approx(1.476, abs=5e-3)


def test_the_block_form_is_gauge_invariant_but_the_scale_is_not(T):
    """Why the literal is frozen rather than recomputed: rescaling per block
    preserves everything 26.0 measured and moves the latent scale anyway."""
    A = CoupledOscillators(c=0.10).A_mat
    T2 = np.diag([3.0, 3.0, 0.2, 0.2]) @ T
    assert representability_distance(T2 @ A @ np.linalg.inv(T2))[0] < 1e-12
    z = np.random.default_rng(0).standard_normal((20_000, 4))
    r1 = np.linalg.norm(z @ T.T, axis=1).mean()
    r2 = np.linalg.norm(z @ T2.T, axis=1).mean()
    assert r2 / r1 > 1.5, "if this ever became 1.0 the gauge argument would be vacuous"


def test_a_broken_basis_is_detected_by_the_block_diagonalisation_check(T):
    """REGLES 7.1's negative half, paired with the positive one at l.90 and
    with `test_the_block_form_is_gauge_invariant_but_the_scale_is_not` just
    above: that test's benign per-block rescale leaves
    `representability_distance` at < 1e-12 -- a check that fired on ANY
    perturbation would prove nothing about detection, only that the metric is
    sensitive to noise. This mutant instead MIXES the two 2x2 subspaces the
    true T keeps apart (`T2[0, 2] += 0.1` couples block {0,1} into block
    {2,3}) and must be caught: measured distance 0.15462926506... , five
    orders of magnitude above the true T's 1.31e-15 and the benign rescale's
    4.16e-15. Together the two tests show the metric neither over-fires (the
    benign case) nor under-fires (this one) -- REGLES 7.1's actual criterion,
    "would a false implementation fail this test", needs both.
    """
    A = CoupledOscillators(c=0.10).A_mat
    T2 = T.copy()
    T2[0, 2] += 0.1
    distance = representability_distance(T2 @ A @ np.linalg.inv(T2))[0]
    assert distance > 1e-6, "mixing the two subspaces must not block-diagonalise"


# -- psi_matrix and the run identity -----------------------------------------


def test_psi_matrix_is_a_hashed_config_key_and_its_value_moves_the_hash(T):
    """28.2's oracle arm, and any future second oracle arm built from a
    DIFFERENT frozen basis, must never collide on run_dir: `psi_matrix` has
    to move `config_hash` both by PRESENCE (frozen vs the mlp-lift default)
    and by VALUE (two different bases, both frozen). Verified empirically
    that `Trainer.__init__` never constructs a predictor or encoder -- only
    `fit` does -- so this is exercised at construction time alone, with no
    dataset built and no training run.
    """
    base_cfg = dict(
        exp_name="t", seed=0, family="mlp_k", d_x=4, d_z=4, d_u=1,
        delta=0.05, r=2, radius="bounded", epochs=1, encoder_hidden=[],
    )
    plain = Trainer(dict(base_cfg))
    frozen = Trainer({**base_cfg, "lift": "frozen", "psi_matrix": T.tolist()})
    assert frozen.config_hash != plain.config_hash

    T_other = T.copy()
    T_other[0, 0] += 0.1
    frozen_other_basis = Trainer({**base_cfg, "lift": "frozen", "psi_matrix": T_other.tolist()})
    assert frozen_other_basis.config_hash != frozen.config_hash
    assert frozen_other_basis.config_hash != plain.config_hash


@pytest.mark.skipif(
    not RESULTS_PRIMARY_FACTORIAL.exists(),
    reason="results_primary_factorial/ absent (fresh clone) -- the pinned "
    "hashes below still document the identity even when this is skipped",
)
def test_the_primary_factorial_s_mlp_cells_pin_real_run_directories(tmp_path):
    """Independent reference for config_hash (REGLES 7.1's corollary: never
    validate an object against its own stored output alone) -- the two
    hashes below are LITERALS, not re-derived from whatever this checkout's
    disk happens to hold, so a drift in `_DEFAULTS`/`_trainer_allowed_keys`/
    representation handling is caught even where results_primary_factorial/
    has since been pruned. Same convention as
    `test_grad_clip_absent_config_hash_pinned_and_defaults_mutation_reddens`
    (tests/experiments/test_factorial.py:1814): pinned as a literal.

    The stored config.yaml carries a `representation_digest` key (checked
    against disk before writing this test), which means production called
    `Trainer(config, representation=...)`. Reproduced here exactly:
    `build_identity_representation` has no random draw (its own docstring:
    "no random init draw happens here"), so building it fresh from the same
    deterministically-generated dataset reproduces production's artifact
    bit-for-bit. Saved and reloaded through `tmp_path`, never through
    `results_primary_factorial/` (this task's only writable file is
    tests/test_basis_root.py) -- which is exactly what stamps the manifest's
    "digest" the way `validate_shared_representation` requires, without
    writing anywhere production's own artifact cache lives.
    """
    primary_cfg = yaml.safe_load(PRIMARY_FACTORIAL_CONFIG.read_text())
    cells = expand(primary_cfg, dry_run=False)
    off_cell = next(c for c in cells if c["backbone"] == "mlp" and c["structure"] == "off")
    on_cell = next(c for c in cells if c["backbone"] == "mlp" and c["structure"] == "on")

    ds_cfg = primary_cfg["datasets"]["c010_n32"]
    data = _build_dataset_for(ds_cfg)
    artifact = build_identity_representation(
        data,
        system=ds_cfg["system"]["name"],
        regime=ds_cfg["regime"]["name"],
        d_z=primary_cfg["representation"]["d_z"],
    )
    scratch = tmp_path / "repr.pt"
    save_frozen_representation(artifact, scratch)
    artifact = load_frozen_representation(scratch)

    expected_hash = {"off": "7b73e3d6", "on": "d12ac614"}
    for label, cell in (("off", off_cell), ("on", on_cell)):
        trainer = Trainer(dict(cell["config"]), representation=artifact)
        trainer.resolve_run_dir(data)
        assert trainer.data_hash == "3cb3e669"
        assert trainer.config_hash == expected_hash[label]
        run_dir = (
            RESULTS_PRIMARY_FACTORIAL / "primary_factorial"
            / f"{trainer.config_hash}-{trainer.data_hash}"
        )
        assert run_dir.exists(), f"{run_dir} missing -- the pinned identity matches no real run"


def test_an_undefaulted_psi_matrix_would_move_every_run_s_hash_if_canonicalized(monkeypatch):
    """Mutation-campaign target named by the task brief: canonicalizing
    `psi_matrix` into `_DEFAULTS` (the way `delta_min`/`n_heads` already are
    for their families) would silently move `config_hash` for the 134
    historical runs and everything since, none of which ever declared it.
    Same convention as
    `test_grad_clip_absent_config_hash_pinned_and_defaults_mutation_reddens`
    (tests/experiments/test_factorial.py:1843-1848): monkeypatch the exact
    mutation, scoped to this test, and assert the hash actually moves.
    """
    base_cfg = dict(
        exp_name="t", seed=0, family="mlp_k", d_x=4, d_z=4, d_u=1,
        delta=0.05, r=2, radius="bounded", epochs=1, encoder_hidden=[],
    )
    assert "psi_matrix" not in trainer_mod._DEFAULTS
    before = Trainer(dict(base_cfg)).config_hash
    monkeypatch.setitem(trainer_mod._DEFAULTS, "psi_matrix", None)
    after = Trainer(dict(base_cfg)).config_hash
    assert after != before


# -- match_budget with lift="frozen" -----------------------------------------


def test_the_frozen_lift_oracle_arm_inherits_on_learned_s_budget_exactly(T):
    """28.2/23.2: budget matching is SUSPENDED for lift="frozen" (psi/pi
    carry zero TRAINABLE parameters), so the oracle arm must inherit
    on-learned's width/psi_hidden EXACTLY rather than have the matcher spend
    the freed parameters on the backbone -- the same contract
    `test_the_nolift_arm_inherits_on_stable_s_backbone_exactly`
    (tests/test_identity_lift.py:68) proves for lift="identity".
    """
    on_learned = match_budget("mlp_k", 4, 1, 10_000, delta=0.05, r=2, radius="bounded")
    oracle = match_budget(
        "mlp_k", 4, 1, 10_000, delta=0.05, r=2, radius="bounded",
        lift="frozen", psi_matrix=T.tolist(),
    )
    assert oracle["width"] == on_learned["width"]
    assert oracle["psi_hidden"] == on_learned["psi_hidden"]

    learned_model = MLPKPredictor(**on_learned)
    oracle_model = MLPKPredictor(**oracle)

    total_learned = sum(p.numel() for p in learned_model.parameters())
    trainable_learned = count_params(learned_model)
    assert total_learned == trainable_learned == 9997

    total_oracle = sum(p.numel() for p in oracle_model.parameters())
    trainable_oracle = count_params(oracle_model)
    assert total_oracle == 8455
    assert trainable_oracle == 8423

    assert total_oracle - trainable_oracle == 32 == 2 * 4**2, (
        "the frozen, non-trainable remainder is exactly T and T^-1 at d_z=4"
    )

    pair = sum(p.numel() for p in learned_model.psi.parameters()) + sum(
        p.numel() for p in learned_model.pi.parameters()
    )
    assert trainable_learned - trainable_oracle == pair, (
        "the only parameters the oracle arm lacks are the learned psi/pi pair"
    )


def test_the_budget_tolerance_exempts_the_frozen_arm_by_name_not_by_widening(T):
    """budget.py's own words (the comment directly above its +/-5% check):
    the contract's exemption for a zero-trainable-param lift is applied
    "exempt BY NAME rather than by a widened tolerance". The oracle call
    measures 15.77% off target_params (8423 trainable against 10 000) and
    must NOT raise; an unrelated, genuinely unreachable target for the same
    family (no lift override -- an ordinary matched call) must still raise.
    """
    oracle = match_budget(
        "mlp_k", 4, 1, 10_000, delta=0.05, r=2, radius="bounded",
        lift="frozen", psi_matrix=T.tolist(),
    )
    achieved = count_params(MLPKPredictor(**oracle))
    rel_err = abs(achieved - 10_000) / 10_000
    assert rel_err > 0.05, "the exemption must be doing real work, not accidentally within tolerance"

    with pytest.raises(ValueError, match="match_budget"):
        match_budget("mlp_k", 4, 1, 1, delta=0.05, r=2, radius="bounded")
