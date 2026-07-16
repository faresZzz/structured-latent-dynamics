import torch

from sld.predictors.blocks2x2 import RotationScalingBlocks, block_diagonal

DTYPE = torch.float64
N_PAIRS = 100
DELTAS = (0.01, 0.1, 1.0)


def _make(constrained: bool, seed: int, n_pairs: int = N_PAIRS) -> RotationScalingBlocks:
    torch.manual_seed(seed)
    return RotationScalingBlocks(n_pairs, constrained=constrained, dtype=DTYPE)


def test_zoh_closed_form_matches_matrix_exp():
    # 100 draws x 2 modes (constrained / free) x 3 deltas, compared against
    # torch.matrix_exp on the same batch of 2x2 generators.
    worst = 0.0
    for constrained, seed in ((True, 0), (False, 1)):
        blocks = _make(constrained, seed)
        A = blocks.continuous_blocks()
        for delta in DELTAS:
            expected = torch.matrix_exp(delta * A)
            A_bar, _ = blocks.discretize_zoh(delta)
            err = (A_bar - expected).abs().max().item()
            worst = max(worst, err)
            assert err < 1e-6, f"constrained={constrained} delta={delta} err={err:.3e}"
    print(f"[test_zoh_closed_form] max abs error over all draws/deltas = {worst:.3e}")


def test_b_bar_matches_direct_solve():
    worst = 0.0
    eye = torch.eye(2, dtype=DTYPE)
    for constrained, seed in ((True, 2), (False, 3)):
        blocks = _make(constrained, seed)
        A = blocks.continuous_blocks()
        torch.manual_seed(seed + 100)
        B_cont = torch.randn(2 * N_PAIRS, 3, dtype=DTYPE)
        B = B_cont.reshape(N_PAIRS, 2, 3)
        for delta in DELTAS:
            A_bar, B_bar = blocks.discretize_zoh(delta, B_cont=B_cont)
            # expected value built from torch.matrix_exp, NOT from the
            # implementation's A_bar — keeps the reference fully independent
            # (de-circularized after adversarial review)
            A_bar_ref = torch.matrix_exp(delta * A)
            expected = torch.linalg.solve(A, (A_bar_ref - eye) @ B)
            err = (B_bar.reshape(N_PAIRS, 2, 3) - expected).abs().max().item()
            worst = max(worst, err)
            assert err < 1e-10, f"constrained={constrained} delta={delta} err={err:.3e}"
    print(f"[test_b_bar_matches_direct_solve] max abs error over all draws/deltas = {worst:.3e}")


def test_on_cell_stable_for_all_delta():
    blocks = _make(constrained=True, seed=4)
    for delta in (1e-3, 1e-2, 0.1, 1.0, 10.0):
        moduli = blocks.spectral_moduli(delta)
        assert torch.all(moduli < 1.0)
        A_bar, _ = blocks.discretize_zoh(delta)
        eigvals = torch.linalg.eigvals(A_bar)
        assert torch.all(eigvals.abs() <= 1.0 + 1e-12)


def test_off_cell_can_be_unstable_with_finite_gradients():
    blocks = _make(constrained=False, seed=5)
    with torch.no_grad():
        blocks.nu[0] = 0.3  # nu IS alpha in free mode
    moduli = blocks.spectral_moduli(delta=1.0)
    assert moduli[0] > 1.0

    torch.manual_seed(6)
    B_cont = torch.randn(2 * N_PAIRS, 3, dtype=DTYPE)
    A_bar, B_bar = blocks.discretize_zoh(0.1, B_cont=B_cont)
    scalar = A_bar.sum() + B_bar.sum()
    scalar.backward()
    for name, p in blocks.named_parameters():
        assert p.grad is not None, name
        assert torch.all(torch.isfinite(p.grad)), name

    # repeat the grad-finiteness check in constrained mode
    blocks_on = _make(constrained=True, seed=7)
    torch.manual_seed(8)
    B_cont_on = torch.randn(2 * N_PAIRS, 3, dtype=DTYPE)
    A_bar_on, B_bar_on = blocks_on.discretize_zoh(0.1, B_cont=B_cont_on)
    (A_bar_on.sum() + B_bar_on.sum()).backward()
    for name, p in blocks_on.named_parameters():
        assert p.grad is not None, name
        assert torch.all(torch.isfinite(p.grad)), name


def test_batched_delta():
    blocks = _make(constrained=True, seed=9)
    torch.manual_seed(10)
    delta = torch.rand(4, N_PAIRS, dtype=DTYPE) * 0.5 + 1e-3  # distinct positive values

    A_bar_batched, _ = blocks.discretize_zoh(delta)
    assert A_bar_batched.shape == (4, N_PAIRS, 2, 2)

    for j in range(4):
        A_bar_row, _ = blocks.discretize_zoh(delta[j])
        err = (A_bar_batched[j] - A_bar_row).abs().max().item()
        assert err < 1e-12, f"row {j} err={err:.3e}"


def test_discrete_mode():
    torch.manual_seed(11)
    on = RotationScalingBlocks(5, constrained=True, mode="discrete", dtype=DTYPE)
    with torch.no_grad():
        on.nu_bar[0] = 5.0
        on.nu_bar[1] = -5.0
    assert torch.all(on.spectral_moduli() <= 1.0)

    torch.manual_seed(12)
    off = RotationScalingBlocks(5, constrained=False, mode="discrete", dtype=DTYPE)
    with torch.no_grad():
        off.nu_bar[0] = 1.0
    assert torch.any(off.spectral_moduli() > 1.0)

    try:
        on.discretize_zoh(0.1)
        raise AssertionError("expected ValueError in discrete mode")
    except ValueError:
        pass


def _phi_van_loan(A: torch.Tensor, delta: float) -> torch.Tensor:
    # independent reference for int_0^delta exp(tA) dt via the Van Loan
    # augmented matrix: expm(delta*[[A, I],[0, 0]]) has that integral as its
    # top-right block — no shared code path with the implementation
    n = A.shape[-1]
    top = torch.cat([A, torch.eye(n, dtype=A.dtype)], dim=-1)
    bottom = torch.zeros(n, 2 * n, dtype=A.dtype)
    M = torch.cat([top, bottom], dim=-2)
    return torch.matrix_exp(delta * M)[:n, n:]


def test_b_bar_is_entire_through_zero():
    # adversarial-review lock: Phi = int_0^delta exp(tA) dt is an entire
    # function of A — B_bar exists AT (alpha, omega) = (0, 0) (= delta*B)
    # and varies continuously through it; an earlier revision wrongly
    # raised there. Checked against the Van Loan reference across
    # magnitudes straddling the series/closed-form branch switch.
    blocks = _make(constrained=False, seed=13, n_pairs=6)
    magnitudes = (0.0, 1e-9, 1e-6, 1e-4, 1e-2, 0.5)
    with torch.no_grad():
        for i, mag in enumerate(magnitudes):
            blocks.nu[i] = mag       # nu IS alpha in free mode
            blocks.omega[i] = mag / 2.0
    torch.manual_seed(14)
    B_cont = torch.randn(12, 3, dtype=DTYPE)
    delta = 0.1
    _, B_bar = blocks.discretize_zoh(delta, B_cont=B_cont)
    B_bar = B_bar.reshape(6, 2, 3)
    A_blocks = blocks.continuous_blocks().detach()
    worst = 0.0
    for i in range(6):
        expected = _phi_van_loan(A_blocks[i], delta) @ B_cont.reshape(6, 2, 3)[i]
        err = (B_bar[i] - expected).abs().max().item()
        worst = max(worst, err)
        assert err < 1e-12, f"magnitude={magnitudes[i]} err={err:.3e}"
    # the exact-zero block is delta*B on the nose
    torch.testing.assert_close(
        B_bar[0], delta * B_cont.reshape(6, 2, 3)[0], rtol=0.0, atol=1e-14
    )
    print(f"[test_b_bar_entire] max abs error across branch switch = {worst:.3e}")


def test_nonpositive_delta_rejected():
    # adversarial-review lock: the ON-cell stability claim holds for
    # delta > 0 only, so nonpositive deltas must be rejected loudly
    import pytest

    blocks = _make(constrained=True, seed=16, n_pairs=4)
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError):
            blocks.discretize_zoh(bad)
        with pytest.raises(ValueError):
            blocks.spectral_moduli(bad)
    batched = torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=DTYPE)
    with pytest.raises(ValueError):
        blocks.discretize_zoh(batched)


def test_block_diagonal_assembly():
    blocks = _make(constrained=True, seed=15, n_pairs=3)
    A = blocks.continuous_blocks()
    full = block_diagonal(A)
    assert full.shape == (6, 6)
    for i in range(3):
        torch.testing.assert_close(full[2 * i : 2 * i + 2, 2 * i : 2 * i + 2], A[i])
    off_block = full[0:2, 2:4]
    assert torch.all(off_block == 0.0)


def test_nonfinite_delta_rejected():
    # NaN fails every comparison so `<= 0` alone would admit it (and +inf)
    # straight into NaN matrices — finiteness must be enforced explicitly
    import pytest

    blocks = _make(constrained=True, seed=17, n_pairs=4)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            blocks.discretize_zoh(bad)
        with pytest.raises(ValueError):
            blocks.spectral_moduli(bad)
