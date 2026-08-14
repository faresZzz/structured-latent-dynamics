"""``scan_state_scale``: the torch-ful scan that persists ``sigma_train`` as a
torch-free JSON, so ``sld/viz/`` can cite it without importing torch.

REGLES section 7.1. Positive: a synthetic ``.pt`` artifact with a KNOWN
``sigma_train`` -> the payload carries the right schema, the right number of
strata, and the exact values, copied verbatim (never rescaled/rounded).
Negative: no ``.pt`` at all; a degenerate ``sigma_train`` (zero, negative,
NaN, inf -- any of the four makes it unusable as a divisor and must never
reach a caption); and ``load_state_scale`` refusing a JSON of the wrong
schema. Every fixture is synthetic, written under ``tmp_path``.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from sld.audit.scan_state_scale import SCHEMA, load_state_scale, scan_state_scale


def _write_artifact(path, *, sigma_train, system="CoupledOscillators", regime="OURegime",
                     digest="d" * 8, e_recon=1e-8, e_1_pers=0.2):
    blob = {
        "sigma_train": sigma_train,
        "e_recon": e_recon,
        "manifest": {"system": system, "regime": regime, "digest": digest, "e_1_pers": e_1_pers},
    }
    torch.save(blob, path)


def test_scan_state_scale_reads_exact_values_from_a_synthetic_artifact(tmp_path):
    """POSITIVE: two strata, two DISTINCT known sigma_train vectors -- the
    payload must carry both, unchanged."""
    reps = tmp_path / "representations"
    reps.mkdir()
    sigma_a = [1.0, 2.0, 3.0, 4.0]
    sigma_b = [0.5, 1.5, 2.5, 3.5]
    _write_artifact(reps / "ou.pt", sigma_train=sigma_a, system="CoupledOscillators", regime="OURegime")
    _write_artifact(reps / "ac.pt", sigma_train=sigma_b, system="CoupledOscillators", regime="ActionRegime")

    payload = scan_state_scale(reps)
    assert payload["schema"] == SCHEMA == "state_scale/1"
    assert set(payload["strata"]) == {"ou", "ac"}
    assert payload["strata"]["ou"]["sigma_train"] == sigma_a
    assert payload["strata"]["ac"]["sigma_train"] == sigma_b
    assert payload["strata"]["ou"]["system"] == "CoupledOscillators"
    assert payload["strata"]["ou"]["regime"] == "OURegime"
    assert payload["strata"]["ou"]["e_recon"] == pytest.approx(1e-8)


def test_scan_state_scale_empty_dir_raises_file_not_found(tmp_path):
    """NEGATIVE: no *.pt under the directory -- must name the directory and
    refuse rather than return an empty (silently blank) payload."""
    reps = tmp_path / "representations"
    reps.mkdir()
    with pytest.raises(FileNotFoundError, match=str(reps)):
        scan_state_scale(reps)


@pytest.mark.parametrize("bad_sigma", [
    [1.0, 0.0, 3.0, 4.0],           # zero -- a caption/normalisation divides by this
    [1.0, -2.0, 3.0, 4.0],          # negative
    [1.0, float("nan"), 3.0, 4.0],  # NaN
    [1.0, float("inf"), 3.0, 4.0],  # inf
])
def test_scan_state_scale_degenerate_sigma_raises_value_error(tmp_path, bad_sigma):
    """NEGATIVE: a zero, negative, NaN or inf entry in sigma_train must raise
    -- an implementation that only checked ``np.isfinite`` would let the
    negative and zero cases silently through, which is exactly the case
    this parametrization is built to catch."""
    reps = tmp_path / "representations"
    reps.mkdir()
    _write_artifact(reps / "ou.pt", sigma_train=bad_sigma)
    with pytest.raises(ValueError, match="sigma_train"):
        scan_state_scale(reps)


def test_load_state_scale_wrong_schema_raises(tmp_path):
    """NEGATIVE: a JSON that does not carry schema 'state_scale/1' must be
    refused, not read as if it were one (a wrong-schema file could carry a
    same-shaped-but-differently-meant array)."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "gate/1", "strata": {}}))
    with pytest.raises(ValueError, match="schema"):
        load_state_scale(path)


def test_load_state_scale_roundtrips_a_scan_state_scale_payload(tmp_path):
    """POSITIVE, independent of scan_state_scale's own return: write the
    payload to disk exactly as main() does, and load_state_scale must read
    it back unchanged (the torch-free half of the contract)."""
    reps = tmp_path / "representations"
    reps.mkdir()
    _write_artifact(reps / "ou.pt", sigma_train=[1.0, 2.0, 3.0, 4.0])
    payload = scan_state_scale(reps)
    out_path = tmp_path / "state_scale.json"
    out_path.write_text(json.dumps(payload))
    reloaded = load_state_scale(out_path)
    assert reloaded == payload
    assert math.isfinite(reloaded["strata"]["ou"]["e_recon"])
