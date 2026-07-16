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


def test_zero_alpha_omega_guard_raises():
    blocks = _make(constrained=False, seed=13)
    with torch.no_grad():
        blocks.nu[0] = 0.0
        blocks.omega[0] = 0.0
    torch.manual_seed(14)
    B_cont = torch.randn(2 * N_PAIRS, 3, dtype=DTYPE)
    try:
        blocks.discretize_zoh(0.1, B_cont=B_cont)
        raise AssertionError("expected ValueError from the singular-inverse guard")
    except ValueError:
        pass


def test_block_diagonal_assembly():
    blocks = _make(constrained=True, seed=15, n_pairs=3)
    A = blocks.continuous_blocks()
    full = block_diagonal(A)
    assert full.shape == (6, 6)
    for i in range(3):
        torch.testing.assert_close(full[2 * i : 2 * i + 2, 2 * i : 2 * i + 2], A[i])
    off_block = full[0:2, 2:4]
    assert torch.all(off_block == 0.0)
