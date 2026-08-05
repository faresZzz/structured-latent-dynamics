"""Rotation-scaling blocks: a block-diagonal real linear operator assembled
from ``n_pairs`` independent 2x2 blocks, each block holding one conjugate
eigenvalue pair.

Two parametrization modes:
  - "continuous" (primary): learnable continuous-time generators
    A_i = alpha_i*I + omega_i*J, plus a closed-form zero-order-hold (ZOH)
    discretization exploiting J^2 = -I and the fact that alpha_i*I and
    omega_i*J commute.
  - "discrete" (minimal, declared): learnable discrete-time operators
    directly (a modulus and a rotation angle per block) — there is no
    continuous-time generator to discretize.

Vocabulary discipline: these are rotation-scaling blocks for structured
linear state transitions, nothing more. The operator-theoretic framing for
a fixed linear operator on observables belongs to a later module with
strict conditions attached, and is intentionally not used here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_VALID_MODES = ("continuous", "discrete")


def rotation_scaling_zoh(
    alpha: torch.Tensor,
    omega: torch.Tensor,
    delta: torch.Tensor | float,
    B_cont: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Zero-order-hold discretization of GIVEN rotation-scaling generators.

    The parameter-free primitive behind :meth:`RotationScalingBlocks.discretize_zoh`
    and behind the context-dependent structured cell, whose ``(alpha, omega, B)``
    are produced by a head rather than held as parameters. Everything is batched:
    ``alpha``/``omega`` are ``(..., n_pairs)`` with an arbitrary leading batch,
    ``B_cont`` is ``(..., 2*n_pairs, m)`` (block ``i`` occupying rows ``2i:2i+2``),
    and the two batch shapes only have to broadcast against each other.

    Each block is ``A_i = alpha_i*I + omega_i*J`` with ``J = [[0,-1],[1,0]]``,
    ``J^2 = -I``; ``I`` and ``J`` commute, so

        A_bar_i = exp(delta*A_i) = exp(delta*alpha_i) * R(delta*omega_i)

    in closed form, with ``R`` the 2x2 rotation. The input integral

        B_bar_i = Phi_i B_i,   Phi_i = int_0^delta exp(t A_i) dt

    is an ENTIRE function of ``A_i``: the familiar ``A^{-1}(A_bar - I)`` has a
    merely REMOVABLE singularity at ``(alpha, omega) = (0, 0)``, where
    ``Phi = delta*I``. Evaluation is branch-selected per block: closed form where
    ``delta^2*(alpha^2 + omega^2) > 1e-6``, else the series
    ``delta*(I + delta*A/2 + delta^2*A^2/6 + delta^3*A^3/24)``, which avoids
    the ``exp(delta*A) - I`` cancellation at small ``delta*||A||``. The bound
    that matters is on the RETURNED ``B_bar = Phi @ B_cont``, not on the
    intermediate ``Phi`` alone: at the worst point of this branch (the
    switch boundary), the relative error of ``Phi`` itself is ``~3e-11``, but
    the componentwise relative error of ``B_bar`` scales with how
    ill-conditioned ``B_cont`` is (its per-component magnitude spread) since
    each component is a separate cancellation; measured up to ``~2e-8`` for a
    deliberately mismatched ``B_cont`` before this series term was added.
    With this term, the same worst-case configuration measures ``~1e-14``.

    Args:
        alpha: ``(..., n_pairs)`` real parts of the continuous generators. FREE:
            no stability reparametrization is applied here.
        omega: ``(..., n_pairs)`` angular frequencies, broadcastable with ``alpha``.
        delta: ZOH step; scalar, ``(n_pairs,)``, or broadcastable onto ``alpha``.
        B_cont: optional ``(..., 2*n_pairs, m)`` continuous input matrix.

    Returns:
        ``(A_bar, B_bar)`` with ``A_bar`` of shape ``(..., n_pairs, 2, 2)`` and
        ``B_bar`` of shape ``(..., 2*n_pairs, m)`` (``None`` if ``B_cont`` is).

    Raises:
        ValueError: if any element of ``delta`` is not finite and strictly
            positive. NaN fails every comparison, so ``<= 0`` alone would let
            NaN (and ``+inf``) through into NaN matrices — finiteness is checked
            explicitly.
    """
    delta_t = torch.as_tensor(delta, dtype=alpha.dtype, device=alpha.device)
    if not torch.all(torch.isfinite(delta_t)) or torch.any(delta_t <= 0):
        raise ValueError("delta must be finite and strictly positive")

    # closed-form matrix exponential of a rotation-scaling block
    scale = torch.exp(delta_t * alpha)
    cos = torch.cos(delta_t * omega)
    sin = torch.sin(delta_t * omega)
    row0 = torch.stack([scale * cos, -scale * sin], dim=-1)
    row1 = torch.stack([scale * sin, scale * cos], dim=-1)
    A_bar = torch.stack([row0, row1], dim=-2)

    if B_cont is None:
        return A_bar, None

    # Phi_i = int_0^delta exp(t A_i) dt in the (I, J) coefficient basis:
    # every matrix in play is a*I + b*J, and (pI + qJ)(rI + sJ) =
    # (pr - qs)I + (ps + qr)J, so the whole computation stays per-block
    denom = alpha**2 + omega**2
    small = (delta_t**2) * denom <= 1e-6

    # closed-form branch: Phi = A^{-1}(A_bar - I), guarded against the
    # untaken-branch NaN-gradient pitfall of torch.where by a safe denom
    denom_safe = torch.where(small, torch.ones_like(denom), denom)
    r = scale * cos - 1.0
    s = scale * sin
    a_cf = (alpha * r + omega * s) / denom_safe
    b_cf = (alpha * s - omega * r) / denom_safe

    # series branch (entire-function evaluation through A = 0):
    # Phi = delta*(I + delta*A/2 + delta^2*A^2/6 + delta^3*A^3/24), with
    # A^2 = (alpha^2 - omega^2)*I + 2*alpha*omega*J and, treating (alpha, omega)
    # as (Re, Im) of a complex number since (I, J) obeys the same algebra as
    # (1, i): A^3 = (alpha + i*omega)^3 = (alpha^3 - 3*alpha*omega^2)
    # + i*(3*alpha^2*omega - omega^3), so
    # A^3 = (alpha^3 - 3*alpha*omega^2)*I + (3*alpha^2*omega - omega^3)*J
    a_ser = delta_t * (
        1.0
        + delta_t * alpha / 2.0
        + delta_t**2 * (alpha**2 - omega**2) / 6.0
        + delta_t**3 * (alpha**3 - 3.0 * alpha * omega**2) / 24.0
    )
    b_ser = delta_t * (
        delta_t * omega / 2.0
        + delta_t**2 * (2.0 * alpha * omega) / 6.0
        + delta_t**3 * (3.0 * alpha**2 * omega - omega**3) / 24.0
    )

    a_phi = torch.where(small, a_ser, a_cf)
    b_phi = torch.where(small, b_ser, b_cf)
    phi_row0 = torch.stack([a_phi, -b_phi], dim=-1)
    phi_row1 = torch.stack([b_phi, a_phi], dim=-1)
    Phi = torch.stack([phi_row0, phi_row1], dim=-2)

    n_pairs = Phi.shape[-3]
    B = B_cont.reshape(*B_cont.shape[:-2], n_pairs, 2, B_cont.shape[-1])
    B_bar = Phi @ B  # batch dims of Phi and B broadcast here
    return A_bar, B_bar.reshape(*B_bar.shape[:-3], 2 * n_pairs, B_bar.shape[-1])


class RotationScalingBlocks(nn.Module):
    """``n_pairs`` independent 2x2 rotation-scaling blocks.

    Each block i is a real 2x2 matrix representing one conjugate eigenvalue
    pair: alpha_i +/- j*omega_i in continuous time, or r_i*exp(+/-j*omega_bar_i)
    in discrete time. The two eigenvalues of a block always share the same
    modulus and are complex conjugates of each other, so one real 2x2 block
    is the minimal real (non-complex) representation of that pair.

    Args:
        n_pairs: number of independent 2x2 blocks (state dim = 2 * n_pairs).
        constrained: if True, the stability axis (alpha in continuous mode,
            modulus r in discrete mode) is constrained so the block cannot
            be unstable/expanding ("ON cell"). If False, that axis is a free
            parameter that can express instability/expansion ("OFF cell") —
            this is deliberate: the constraint ablation needs a true off.
        mode: "continuous" (default) or "discrete", see module docstring.
        dtype: parameter dtype at construction. Computations resolve dtype
            and device from the parameters at use time, so later
            ``.to()``/``.double()`` moves are safe.
    """

    def __init__(
        self,
        n_pairs: int,
        constrained: bool,
        mode: str = "continuous",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"unknown mode: {mode!r}, expected one of {_VALID_MODES}")
        self.n_pairs = n_pairs
        self.constrained = constrained
        self.mode = mode
        self.dtype = dtype

        if mode == "continuous":
            # nu drives alpha (the stability axis), omega is the angular
            # frequency; init is contractive-ish and oscillatory by
            # declaration, not by tuning
            nu = torch.empty(n_pairs, dtype=dtype).uniform_(-1.5, -0.5)
            omega = torch.empty(n_pairs, dtype=dtype).uniform_(0.1, 2.0)
            self.nu = nn.Parameter(nu)
            self.omega = nn.Parameter(omega)
        else:
            # discrete mode parametrizes the operator directly: nu_bar drives
            # the modulus r, omega_bar is the rotation angle. Same init
            # ranges as continuous mode, reused here as a judgment call —
            # nothing in the spec pins a different range for discrete mode.
            nu_bar = torch.empty(n_pairs, dtype=dtype).uniform_(-1.5, -0.5)
            omega_bar = torch.empty(n_pairs, dtype=dtype).uniform_(0.1, 2.0)
            self.nu_bar = nn.Parameter(nu_bar)
            self.omega_bar = nn.Parameter(omega_bar)

    def _require_mode(self, expected: str, method: str) -> None:
        if self.mode != expected:
            raise ValueError(f"{method}() requires mode={expected!r}, got mode={self.mode!r}")

    def _alpha(self) -> torch.Tensor:
        # constrained: alpha = -exp(nu) < 0 always (continuous-time stable).
        # free: alpha = nu directly, an unconstrained real (can be >= 0).
        if self.constrained:
            return -torch.exp(self.nu)
        return self.nu

    def _modulus_r(self) -> torch.Tensor:
        # constrained: r = exp(-softplus(nu_bar)) = 1/(1 + exp(nu_bar)),
        # exact range the OPEN interval (0, 1) — r = 1 is a limit as
        # nu_bar -> -inf, never attained (discrete-time strictly stable).
        # free: r = exp(nu_bar) > 0, can exceed 1.
        if self.constrained:
            return torch.exp(-F.softplus(self.nu_bar))
        return torch.exp(self.nu_bar)

    def continuous_blocks(self) -> torch.Tensor:
        """Continuous-time generators A_i = alpha_i*I + omega_i*J.

        A_i = [[alpha_i, -omega_i], [omega_i, alpha_i]], shape (n_pairs, 2, 2).
        """
        self._require_mode("continuous", "continuous_blocks")
        alpha = self._alpha()
        omega = self.omega
        row0 = torch.stack([alpha, -omega], dim=-1)
        row1 = torch.stack([omega, alpha], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def discrete_blocks(self) -> torch.Tensor:
        """Discrete-time operators A_bar_i = r_i * R(omega_bar_i) directly.

        R(theta) = [[cos theta, -sin theta], [sin theta, cos theta]].
        Shape (n_pairs, 2, 2).
        """
        self._require_mode("discrete", "discrete_blocks")
        r = self._modulus_r()
        omega_bar = self.omega_bar
        cos, sin = torch.cos(omega_bar), torch.sin(omega_bar)
        row0 = torch.stack([r * cos, -r * sin], dim=-1)
        row1 = torch.stack([r * sin, r * cos], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def discretize_zoh(
        self, delta: torch.Tensor | float, B_cont: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero-order-hold discretization (A_bar exact; B_bar exact or ~1e-14).

        Since A_i = alpha_i*I + omega_i*J with J = [[0,-1],[1,0]], J^2 = -I,
        and I, J commute, exp(delta*A_i) has the closed form:
            A_bar_i = exp(delta*alpha_i) * R(delta*omega_i)
        with R the 2x2 rotation matrix. ``delta`` broadcasts against the
        (n_pairs,) block parameters: scalar, (n_pairs,), or (batch, n_pairs)
        all work, giving A_bar shape (..., n_pairs, 2, 2) (the batched-delta
        path is exercised here for reuse by a later input-dependent variant).

        If ``B_cont`` is given (shape (2*n_pairs, m), block i occupying rows
        2i:2i+2), also returns the matching B_bar via the ZOH input integral
            B_bar_i = Phi_i B_i,   Phi_i = int_0^delta exp(t A_i) dt,
        which is an ENTIRE function of A_i: Phi -> delta*I as A -> 0. The
        familiar closed form A^{-1}(A_bar - I) has a merely REMOVABLE
        singularity at (alpha, omega) = (0, 0) — B_bar is defined there
        (= delta*B), a point an earlier revision wrongly rejected (caught
        in adversarial review). Evaluation is branch-selected per block:
        the closed form where delta^2*(alpha^2 + omega^2) > 1e-6, else the
        series delta*(I + delta*A/2 + delta^2*A^2/6 + delta^3*A^3/24), which
        also avoids the exp(delta*A) - I cancellation at small delta*||A||.
        The truncation-error bound that matters is on the RETURNED B_bar,
        not on the intermediate Phi: Phi's own relative error at the branch
        switch is ~3e-11, but B_bar = Phi @ B_cont's componentwise relative
        error additionally depends on how ill-conditioned B_cont is (spread
        across component magnitudes) — measured up to ~2e-8 for a badly
        scaled B_cont before this series term was added, ~1e-14 after. B_bar is
        returned with the same shape as B_cont, (2*n_pairs, m). Delegates the
        whole computation to ``rotation_scaling_zoh``, this method only
        supplies the block parameters — including its batched-delta B_bar
        support, so scalar, (n_pairs,), or (batch, n_pairs) delta all work
        with B_cont too.

        Raises:
            ValueError: in mode="discrete" (nothing to discretize — the
                block already parametrizes the discrete operator directly).
            ValueError: if any element of delta is not strictly positive:
                the ON-cell guarantee |eig(A_bar)| < 1 requires delta > 0
                (exp(delta*alpha) > 1 for delta < 0 even with alpha < 0),
                and a nonpositive ZOH step is meaningless in this repo.
        """
        self._require_mode(
            "continuous",
            "discretize_zoh (mode='discrete' parametrizes A_bar directly)",
        )
        alpha = self._alpha()
        omega = self.omega
        # dtype/device resolved from the parameters at use time, so the
        # module survives .to()/.double() moves
        delta_t = torch.as_tensor(delta, dtype=omega.dtype, device=omega.device)
        # NaN fails every comparison, so `<= 0` alone would let NaN (and
        # +inf) through to produce NaN matrices — require finite AND positive
        # (re-checked inside rotation_scaling_zoh; kept here for a fast reject)
        if not torch.all(torch.isfinite(delta_t)) or torch.any(delta_t <= 0):
            raise ValueError("delta must be finite and strictly positive")
        return rotation_scaling_zoh(alpha, omega, delta_t, B_cont)

    def spectral_moduli(self, delta: torch.Tensor | float | None = None) -> torch.Tensor:
        """Modulus of the shared eigenvalue pair, per block.

        mode="continuous": |eig(A_bar_i)| = exp(delta * alpha_i) — both
        eigenvalues of a rotation-scaling block share this one modulus, so
        ``delta`` is required here. mode="discrete": returns r_i directly;
        ``delta`` is not used (the block already parametrizes A_bar).
        """
        if self.mode == "discrete":
            return self._modulus_r()
        if delta is None:
            raise ValueError("mode='continuous' requires delta")
        alpha = self._alpha()
        delta_t = torch.as_tensor(delta, dtype=alpha.dtype, device=alpha.device)
        if not torch.all(torch.isfinite(delta_t)) or torch.any(delta_t <= 0):
            raise ValueError("delta must be finite and strictly positive")
        return torch.exp(delta_t * alpha)


def block_diagonal(blocks: torch.Tensor) -> torch.Tensor:
    """Assemble a block-diagonal operator from (..., n_pairs, 2, 2) blocks.

    Returns shape (..., 2*n_pairs, 2*n_pairs) with block i occupying rows
    and columns 2i:2i+2.
    """
    n_pairs = blocks.shape[-3]
    dim = 2 * n_pairs
    out = torch.zeros(
        *blocks.shape[:-3], dim, dim, dtype=blocks.dtype, device=blocks.device
    )
    for i in range(n_pairs):
        out[..., 2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = blocks[..., i, :, :]
    return out
