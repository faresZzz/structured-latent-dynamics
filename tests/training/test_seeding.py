"""Tests for sld.training.seeding.set_seed.

set_seed mutates GLOBAL RNG state by contract (see its docstring), so this
test necessarily touches global random/numpy/torch state too -- unlike
every other test module in this repo, which uses local generators. At the
end, global state is best-effort restored by reseeding from OS entropy so
a fixed seed doesn't leak into later tests running in the same process.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from sld.training.seeding import set_seed


def test_same_seed_gives_identical_draws():
    set_seed(42)
    t1 = torch.randn(5)
    n1 = np.random.rand(5)
    r1 = random.random()

    set_seed(42)
    t2 = torch.randn(5)
    n2 = np.random.rand(5)
    r2 = random.random()

    torch.testing.assert_close(t1, t2, atol=0.0, rtol=0.0)
    assert np.array_equal(n1, n2)
    assert r1 == r2


def test_different_seeds_give_different_draws():
    set_seed(1)
    t1 = torch.randn(5)
    n1 = np.random.rand(5)
    r1 = random.random()

    set_seed(2)
    t2 = torch.randn(5)
    n2 = np.random.rand(5)
    r2 = random.random()

    print(f"[seeding] torch_max_abs_diff = {(t1 - t2).abs().max().item():.3e}")
    assert not torch.allclose(t1, t2)
    assert not np.array_equal(n1, n2)
    assert r1 != r2

    # best-effort restore: reseed every RNG from OS entropy so this test
    # doesn't leave the process pinned to a fixed seed for later tests
    random.seed()
    np.random.seed()
    torch.seed()
