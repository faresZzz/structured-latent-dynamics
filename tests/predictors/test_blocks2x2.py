import math

import pytest
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


def test_zoh_hand_built_generator_pins_sign_and_layout():
    """Positive (REGLES 7.1 + R2): test_zoh_closed_form_matches_matrix_exp
    above builds A = blocks.continuous_blocks() and compares
    discretize_zoh(delta) to torch.matrix_exp(delta * A) -- an independent
    EXPONENTIAL (torch.matrix_exp), but applied to a generator A that is
    the implementation's OWN continuous_blocks() output. A sign convention
    flip applied consistently to both continuous_blocks() and
    discretize_zoh() (e.g. J -> -J everywhere) would still pass that test,
    since both sides would agree on the same wrong convention -- neither
    side is tied to the convention actually DECLARED in the module
    docstring.

    Here the block's parameter tensors are set BY HAND, without going
    through continuous_blocks() at all: alpha=0, omega=pi/2, delta=1.
    alpha=0 needs mode='free' (constrained=False): with constrained=True,
    alpha = -exp(nu) < 0 always (RotationScalingBlocks._alpha), so alpha=0
    is unreachable there; in free mode alpha = nu directly, so nu=0 gives
    alpha=0 exactly.

    Declared convention (module docstring): A = alpha*I + omega*J with
    J = [[0,-1],[1,0]], so A_bar = exp(delta*A) = exp(delta*alpha)*R(delta*omega).
    At alpha=0, delta=1, omega=pi/2:
        R(pi/2) = [[cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]]
                = [[0, -1], [1, 0]]
    (value independently confirmed by the supervisor). This single 2x2
    number pins down BOTH the sign of the rotation (rotating the other way
    gives [[0,1],[-1,0]]) AND the row/column layout (a transposed layout
    swaps which off-diagonal entry is +1 and which is -1) in one check.
    """
    blocks = RotationScalingBlocks(1, constrained=False, mode="continuous", dtype=DTYPE)
    with torch.no_grad():
        blocks.nu[0] = 0.0  # free mode: alpha = nu -> alpha = 0 exactly
        blocks.omega[0] = math.pi / 2.0
    delta = 1.0

    A_bar, _ = blocks.discretize_zoh(delta)
    oracle = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=DTYPE)
    err = (A_bar[0] - oracle).abs().max().item()
    print(f"[zoh sign+layout positive] A_bar={A_bar[0].tolist()} err={err:.3e}")
    assert err < 1e-12  # short float64 algebra: cos/sin of one fixed angle


def test_zoh_oracle_discriminates_opposite_rotation_sense():
    """Negative (REGLES 7.1, "the whole point"): the SAME oracle as the test
    above, [[0,-1],[1,0]] at (alpha=0, omega=pi/2, delta=1), compared to the
    matrix a J -> -J sign convention would produce at the identical
    parameters (equivalent to evaluating the declared closed form
    exp(delta*alpha)*R(delta*omega) with omega negated, since
    alpha*I + omega*(-J) = alpha*I + (-omega)*J). Demonstrates the oracle
    actually discriminates rotation sense rather than passing regardless of
    it -- the same tolerance that admits the correct answer above rejects
    this one.

    Hardened by the supervisor: the wrong matrix is not hand-typed here, it is
    produced by running the REAL discretize_zoh under the mirrored parameter.
    Since alpha*I + omega*(-J) = alpha*I + (-omega)*J, driving the actual
    implementation at omega = -pi/2 yields exactly the matrix a J -> -J
    convention would produce at omega = +pi/2. So this executes the production
    code path and asserts the oracle rejects its output -- rather than asserting
    that a matrix the test itself typed differs from another matrix the test
    itself typed, which would prove nothing about the implementation.
    """
    oracle = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=DTYPE)

    mirrored = RotationScalingBlocks(1, constrained=False, mode="continuous", dtype=DTYPE)
    with torch.no_grad():
        mirrored.nu[0] = 0.0
        mirrored.omega[0] = -(math.pi / 2.0)  # J -> -J at the declared omega
    opposite, _ = mirrored.discretize_zoh(1.0)

    err = (opposite[0] - oracle).abs().max().item()
    print(f"[zoh sign+layout negative] opposite={opposite[0].tolist()} err={err:.3e}")
    assert err > 1e-12  # same oracle, same tolerance: must fail
    # and it must fail by the full swing of the off-diagonal, not by rounding:
    # +1 against -1 in both off-diagonal entries
    assert abs(err - 2.0) < 1e-12


def test_b_bar_hand_built_oracle_pins_the_zoh_input_integral():
    """Positive (REGLES 7.1 + R2), added by the supervisor after A2 review.

    test_b_bar_matches_direct_solve below builds its reference from
    A = blocks.continuous_blocks(), so it carries the SAME generator
    circularity the two tests above just closed for A_bar: a sign convention
    flipped consistently in continuous_blocks() and in discretize_zoh()'s
    (I, J) coefficient algebra would leave both sides of that test equal.
    Closing it for A_bar does not close it for B_bar: the B_bar path
    evaluates its own coefficients (a_cf, b_cf / a_ser, b_ser) and could
    carry an independent layout error.

    Hand oracle, derived without touching the implementation. With
    Phi = int_0^delta exp(tA) dt = A^{-1}(A_bar - I) and, at
    (alpha, omega, delta) = (0, pi/2, 1), A = (pi/2) J, A_bar = [[0,-1],[1,0]]:

        A^{-1} = (2/pi) J^{-1} = -(2/pi) J
        A_bar - I = [[-1,-1],[1,-1]]
        J (A_bar - I) = [[-1, 1],[-1,-1]]
        Phi = -(2/pi) [[-1, 1],[-1,-1]] = (2/pi) [[1,-1],[1,1]]

    so with B_cont = I the expected B_bar is (2/pi)[[1,-1],[1,1]], i.e.
    0.6366197723675814 * [[1,-1],[1,1]]. Verified independently by the
    supervisor against numpy's solve on the same hand-built A.

    Note this point sits in the CLOSED-FORM branch: delta^2 (alpha^2 +
    omega^2) = (pi/2)^2 = 2.47 > 1e-6, so it exercises a_cf/b_cf. The series
    branch keeps its own gradient/continuity coverage elsewhere; a hand
    oracle for it belongs to A4's branch-gradient item, not here.
    """
    blocks = RotationScalingBlocks(1, constrained=False, mode="continuous", dtype=DTYPE)
    with torch.no_grad():
        blocks.nu[0] = 0.0
        blocks.omega[0] = math.pi / 2.0

    B_cont = torch.eye(2, dtype=DTYPE)
    _, B_bar = blocks.discretize_zoh(1.0, B_cont=B_cont)

    oracle = (2.0 / math.pi) * torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=DTYPE)
    err = (B_bar.reshape(2, 2) - oracle).abs().max().item()
    print(f"[b_bar hand oracle] B_bar={B_bar.reshape(2, 2).tolist()} err={err:.3e}")
    assert err < 1e-12  # short float64 algebra on one fixed angle

    # Negative: the mirrored convention (J -> -J, i.e. omega negated) must be
    # REJECTED by the same oracle at the same tolerance. Phi's off-diagonal is
    # odd in omega, so the mirrored run flips it -- run through the real code
    # path, not hand-typed.
    mirrored = RotationScalingBlocks(1, constrained=False, mode="continuous", dtype=DTYPE)
    with torch.no_grad():
        mirrored.nu[0] = 0.0
        mirrored.omega[0] = -(math.pi / 2.0)
    _, B_bar_wrong = mirrored.discretize_zoh(1.0, B_cont=B_cont)
    err_wrong = (B_bar_wrong.reshape(2, 2) - oracle).abs().max().item()
    print(f"[b_bar hand oracle negative] err={err_wrong:.3e}")
    assert err_wrong > 1e-12


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


# ---------------------------------------------------------------------------
# gradients across the series/closed-form branch switch (A4 gap: the tests
# above check VALUES only -- test_b_bar_is_entire_through_zero across
# magnitudes, and the grad-finiteness check in
# test_off_cell_can_be_unstable_with_finite_gradients at one random point --
# neither differentiates AT or straddling the branch switch itself)
# ---------------------------------------------------------------------------
def _blocks_at(alpha_val: float, omega_val: float) -> RotationScalingBlocks:
    # constrained=False: alpha = nu directly (RotationScalingBlocks._alpha),
    # an IDENTITY map, so nu.grad IS d(loss)/d(alpha) exactly -- no extra
    # chain-rule factor to carry by hand. Same construction idiom as
    # test_zoh_hand_built_generator_pins_sign_and_layout above.
    b = RotationScalingBlocks(1, constrained=False, mode="continuous", dtype=torch.float64)
    with torch.no_grad():
        b.nu[0] = alpha_val
        b.omega[0] = omega_val
    return b


@pytest.mark.parametrize(
    "alpha0,omega0",
    [
        (0.0, 0.0),  # exactly the removable-singularity point
        (5e-4, 5e-4),  # series branch, near the switch (denom=5e-7 < 1e-6 at delta=1)
        (1e-3, 1e-3),  # closed-form branch, near the switch (denom=2e-6 > 1e-6 at delta=1)
    ],
)
def test_discretize_zoh_gradients_finite_and_match_finite_differences(alpha0, omega0):
    """Backprop through ``discretize_zoh`` at ``(alpha, omega) = (0, 0)`` and
    on either side of the ``small`` branch switch (``delta**2 * (alpha**2 +
    omega**2) <= 1e-6``, blocks2x2.py:201; ``delta=1`` here puts the switch
    exactly at ``alpha**2 + omega**2 == 1e-6``).

    Fact this locks in (module docstring): ``Phi = int_0^delta exp(tA) dt``
    is an ENTIRE function of ``A``, equal to ``delta*I`` at ``A=0`` -- the
    ``A^{-1}(A_bar - I)`` closed form has a REMOVABLE singularity there, so
    gradients must exist and stay finite/continuous through it. That is
    what this test demonstrates by MEASUREMENT, not by assertion of the
    docstring's claim alone.

    Reference: central finite differences on the scalar loss
    ``A_bar.sum() + B_bar.sum()``, step ``h=1e-5`` -- picked empirically as
    the float64 sweet spot (truncation ~h^2, rounding ~eps/h, both ~1e-10 to
    1e-11 here); the worst-case relative error measured against autograd
    over these three points at this h was ~3.8e-9. Tolerance 1e-6 relative
    leaves ~1000x margin over that measurement. This is a genuinely
    independent reference -- repeated forward evaluations only, autograd is
    never called on the reference side.
    """
    delta = 1.0
    B_cont = torch.tensor([[0.7], [-0.3]], dtype=torch.float64)

    blocks = _blocks_at(alpha0, omega0)
    A_bar, B_bar = blocks.discretize_zoh(delta, B_cont)
    loss = A_bar.sum() + B_bar.sum()
    loss.backward()
    grad_alpha, grad_omega = blocks.nu.grad.item(), blocks.omega.grad.item()
    assert math.isfinite(grad_alpha) and math.isfinite(grad_omega)

    def loss_at(a: float, o: float) -> float:
        b = _blocks_at(a, o)
        with torch.no_grad():
            A_bar_, B_bar_ = b.discretize_zoh(delta, B_cont)
            return (A_bar_.sum() + B_bar_.sum()).item()

    h = 1e-5
    fd_alpha = (loss_at(alpha0 + h, omega0) - loss_at(alpha0 - h, omega0)) / (2 * h)
    fd_omega = (loss_at(alpha0, omega0 + h) - loss_at(alpha0, omega0 - h)) / (2 * h)

    rel_alpha = abs(grad_alpha - fd_alpha) / (abs(grad_alpha) + 1e-30)
    rel_omega = abs(grad_omega - fd_omega) / (abs(grad_omega) + 1e-30)
    print(
        f"[zoh grad @ ({alpha0},{omega0})] autograd=({grad_alpha:.6e},{grad_omega:.6e}) "
        f"fd=({fd_alpha:.6e},{fd_omega:.6e}) rel=({rel_alpha:.3e},{rel_omega:.3e})"
    )
    assert rel_alpha < 1e-6
    assert rel_omega < 1e-6


def test_where_without_denom_safe_guard_produces_nan_gradient():
    """Negative: demonstrates the ``denom_safe`` guard (blocks2x2.py:205) is
    not decorative. Reproduces, LOCALLY (src untouched), the exact
    ``torch.where`` pattern at ``denom = 0``: the untaken branch's LOCAL
    derivative (``1/denom``) is infinite, and IEEE ``0 * inf = nan`` poisons
    the gradient of the branch that WAS selected -- even though the FORWARD
    value is perfectly finite (the selected branch's own value is what
    comes out). This is exactly the trap
    ``denom_safe = torch.where(small, ones, denom)`` exists to avoid before
    ever dividing.
    """
    alpha = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    denom = alpha**2  # mirrors denom = alpha**2 + omega**2 at alpha=omega=0
    small = denom <= 1e-6  # mirrors the src guard condition; True here
    unguarded = alpha / denom  # the closed-form branch WITHOUT denom_safe
    safe_branch = torch.zeros_like(denom)  # stands in for the series branch
    out = torch.where(small, safe_branch, unguarded)

    assert torch.isfinite(out)  # forward: fine, the safe branch is selected
    out.backward()
    print(f"[where-NaN-trap] out={out.item()}, alpha.grad={alpha.grad.item()}")
    assert torch.isnan(alpha.grad)  # but the gradient is poisoned regardless


def test_b_bar_series_branch_third_order_term_bounds_ill_conditioned_B():
    """Positive (REGLES 7.1 + R2), C1 review fix.

    The series branch of ``rotation_scaling_zoh`` truncates
    ``Phi = delta*(I + delta*A/2 + delta^2*A^2/6 + delta^3*A^3/24)``. Before
    this term was added, the guarantee in the docstring ("relative
    truncation error < 1e-10") was true of the intermediate ``Phi`` but NOT
    of the value the function actually returns, ``B_bar = Phi @ B_cont``:
    when ``B_cont`` has components of very different magnitude, each
    component of ``B_bar`` is its own cancellation and the relative error
    inflates independently of ``Phi``'s own error. Reference here is
    ``_phi_van_loan`` (module-level torch.matrix_exp on the Van Loan
    augmented matrix) -- an independent computation from the series/closed
    form pair in the source, matrix_exp is never used in production code.

    Point: alpha = omega = 1e-3/sqrt(2), delta = 1 -- exactly at the
    small-branch switch (delta^2*(alpha^2+omega^2) == 1e-6, the switch
    condition is "<="), so this is the worst point *inside* the series
    branch. B_cont = [1, 1e-3] is deliberately mismatched across components
    (three orders of magnitude) to stress exactly the failure mode above.
    """
    alpha_val = 1e-3 / (2**0.5)
    omega_val = 1e-3 / (2**0.5)
    delta = 1.0

    blocks = RotationScalingBlocks(1, constrained=False, mode="continuous", dtype=DTYPE)
    with torch.no_grad():
        blocks.nu[0] = alpha_val
        blocks.omega[0] = omega_val

    B_cont = torch.tensor([[1.0], [1e-3]], dtype=DTYPE)
    A_bar, B_bar = blocks.discretize_zoh(delta, B_cont=B_cont)

    A = blocks.continuous_blocks()[0].detach()
    ref_Phi = _phi_van_loan(A, delta)
    ref_B = ref_Phi @ B_cont

    rel_err = ((B_bar - ref_B).abs() / ref_B.abs()).max().item()
    print(f"[b_bar series 3rd-order] rel_err={rel_err:.3e}")
    # measured ~8.2e-15 with the delta^3*A^3/24 term; generous 1e-12 margin
    assert rel_err < 1e-12


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
