"""Run-level RNG seeding for the Trainer.

Seeds python ``random``, numpy's global RNG, and torch (CPU + CUDA if
available). This MUTATES global RNG state -- it is the run-level
reproducibility knob for ``Trainer`` (call once at the start of a run).
Unit tests in this repo keep using local generators
(``torch.Generator().manual_seed(...)``, ``numpy.random.default_rng(...)``)
instead, precisely so they don't depend on -- or clobber -- this global
state.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python ``random``, numpy's global RNG, and torch (CPU + CUDA).

    MUTATES global RNG state. This is the run-level reproducibility knob
    for ``Trainer``; it is not meant to be called inside unit tests, which
    should use local generators instead (see module docstring).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
