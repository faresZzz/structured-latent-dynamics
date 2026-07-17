"""Fresh-process import gates.

Adversarial review round 1: a circular import between ``sld.predictors`` and
``sld.training`` was invisible to the full test suite because pytest imports
test modules alphabetically (predictors before training). Each statement below
runs in a FRESH interpreter so import order is exactly what it says.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "stmt",
    [
        "import sld.training",
        "import sld.predictors",
        "import sld.training; import sld.predictors",
        "import sld.predictors; import sld.training",
        "from sld.training.seeding import set_seed",
        "from sld.predictors.registry import get_predictor",
        "from sld.training.trainer import Trainer",
    ],
)
def test_fresh_process_import(stmt):
    proc = subprocess.run(
        [sys.executable, "-c", stmt], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"`{stmt}` failed:\n{proc.stderr}"
