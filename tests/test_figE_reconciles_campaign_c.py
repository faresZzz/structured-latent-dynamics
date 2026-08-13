"""Figure E must reproduce the table PROTOCOL section 24.0 publishes.

PROTOCOL 23.5(3), binding: *"no figure ships whose recomputation disagrees with
the table it cites -- the discrepancy becomes the result"*. So the figure's own
reader is run against section 24.0's numbers, cell by cell, from the artifacts
on disk. If the campaign is re-run and a number moves, this goes red before the
figure can quietly disagree with the amendment that cites it.

Skipped when the root is absent, so the suite still runs on a machine that has
not executed campaign C (the same skip-if-absent discipline the phase-D
sentinel tests already use).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sld.viz.figE_ssm_delta import load_cells

ROOT = Path("results_ssm_delta")
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="campagne C non executee sur cette machine")


@pytest.fixture(scope="module")
def cells():
    return load_cells(ROOT)


def _cell(cells, family, delta_max):
    return [c for c in cells if c["family"] == family and c["delta_max"] == delta_max]


def test_the_grid_is_the_declared_one(cells):
    """23.6: two families x two delta_max levels x five seeds."""
    assert len(cells) == 20
    for family in ("min_ssm", "ssm_k"):
        for level in ("10.0", "0.2"):
            group = _cell(cells, family, level)
            assert len(group) == 5, f"{family} @ {level}: {len(group)} seeds"
            assert sorted(c["seed"] for c in group) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("family", ("min_ssm", "ssm_k"))
def test_every_cell_at_delta_max_10_diverged(cells, family):
    """24.0's table: 0 / 5 completed, both families."""
    group = _cell(cells, family, "10.0")
    assert all(c["diverged"] for c in group), f"{family} @ 10.0 was expected 5/5 diverged"


@pytest.mark.parametrize("family", ("min_ssm", "ssm_k"))
def test_every_cell_at_the_bounded_level_completed(cells, family):
    """24.0's table: 5 / 5 completed, both families."""
    group = _cell(cells, family, "0.2")
    assert not any(c["diverged"] for c in group), f"{family} @ 0.20 was expected 5/5 complete"


def test_min_ssm_at_10_wrote_no_row_at_all(cells):
    """The absence the left panel annotates rather than drops: non-finite at
    epoch 0, before `metrics.csv` received a single row."""
    group = _cell(cells, "min_ssm", "10.0")
    assert all(c["loss"] == [] for c in group)
    assert all(c["grad0"] is None for c in group)
    assert all(c["failed_at"] == 0 for c in group)


def test_ssm_k_at_10_died_at_the_epochs_24_0_reports(cells):
    """24.0: epochs 28, 36, 37, 56, 56 -- the timing is part of the result."""
    group = _cell(cells, "ssm_k", "10.0")
    assert sorted(c["failed_at"] for c in group) == [28, 36, 37, 56, 56]


@pytest.mark.parametrize(
    ("family", "lo", "hi"),
    [("min_ssm", 22.39, 27.86), ("ssm_k", 1.80, 2.36)],
)
def test_first_step_gradient_norms_are_the_range_24_1_cites(cells, family, lo, hi):
    """24.1's load-bearing number. `min_ssm` at the bounded level sits ON the
    `mlp` reference of 28.2 -- which is why 24.1 can say the 18-decade anomaly
    belonged to the configuration. That reference is NOT asserted here: it was
    measured on another root (23.0), and this test only pins what this root
    produced.

    The bounds are rounded OUTWARD from the measured extremes
    (22.396526336669922 and 1.8062461614608765). This test caught 24.0's first
    table citing `22.4` and `1.81`, both of which EXCLUDE the value they claim
    to contain -- 23.5(3) doing its job before the figure shipped."""
    values = [c["grad0"] for c in _cell(cells, family, "0.2")]
    assert all(v is not None for v in values)
    assert lo <= min(values) and max(values) <= hi, f"{family} @ 0.20: {sorted(values)}"


def test_the_reader_keeps_diverged_cells_rather_than_dropping_them(cells):
    """The negative check. A reader that silently skipped runs without a
    completion sentinel would return 10 tidy cells and the figure would show
    a family that trains everywhere -- the exact misreading 15.3 exists to
    prevent. Ten of these twenty cells have no `run_complete.json`."""
    assert sum(c["diverged"] for c in cells) == 10
    assert sum(not c["diverged"] for c in cells) == 10
