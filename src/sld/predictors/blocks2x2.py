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
        dtype: parameter dtype, fixed at construction (not re-synced if the
            module is later moved with ``.to()``/``.double()``).
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
        # constrained: r = exp(-softplus(nu_bar)) in (0, 1] for any real
        # nu_bar (discrete-time stable by construction).
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
        """Exact zero-order-hold discretization of the continuous generators.

        Since A_i = alpha_i*I + omega_i*J with J = [[0,-1],[1,0]], J^2 = -I,
        and I, J commute, exp(delta*A_i) has the closed form:
            A_bar_i = exp(delta*alpha_i) * R(delta*omega_i)
        with R the 2x2 rotation matrix. ``delta`` broadcasts against the
        (n_pairs,) block parameters: scalar, (n_pairs,), or (batch, n_pairs)
        all work, giving A_bar shape (..., n_pairs, 2, 2) (the batched-delta
        path is exercised here for reuse by a later input-dependent variant).

        If ``B_cont`` is given (shape (2*n_pairs, m), block i occupying rows
        2i:2i+2), also returns the matching B_bar via the standard ZOH
        integral in closed form:
            B_bar_i = A_i^{-1} (A_bar_i - I) B_i
        using the closed-form inverse A_i^{-1} = 1/(alpha_i^2+omega_i^2) *
        [[alpha_i, omega_i], [-omega_i, alpha_i]]. B_bar is returned with the
        same shape as B_cont, (2*n_pairs, m). This path only supports scalar
        or (n_pairs,) delta (a batched-delta B_bar is not needed yet).

        Raises:
            ValueError: in mode="discrete" (nothing to discretize — the
                block already parametrizes the discrete operator directly).
            ValueError: if B_cont is given and min(alpha^2 + omega^2) <=
                1e-12 for any block — A_i is singular there (this can only
                happen with the free/OFF alpha parametrization at
                (alpha, omega) = (0, 0); the constrained/ON parametrization
                alpha = -exp(nu) < 0 makes this trivially unreachable).
        """
        self._require_mode(
            "continuous",
            "discretize_zoh (mode='discrete' parametrizes A_bar directly)",
        )
        alpha = self._alpha()
        omega = self.omega
        delta_t = torch.as_tensor(delta, dtype=self.dtype, device=alpha.device)

        # closed-form matrix exponential of a rotation-scaling block
        scale = torch.exp(delta_t * alpha)
        cos = torch.cos(delta_t * omega)
        sin = torch.sin(delta_t * omega)
        row0 = torch.stack([scale * cos, -scale * sin], dim=-1)
        row1 = torch.stack([scale * sin, scale * cos], dim=-1)
        A_bar = torch.stack([row0, row1], dim=-2)

        if B_cont is None:
            return A_bar, None

        if delta_t.dim() > 1:
            raise ValueError(
                "B_bar path only supports scalar or (n_pairs,) delta; "
                "batched-delta B_bar is not needed yet"
            )

        denom = alpha**2 + omega**2
        if torch.min(denom).item() <= 1e-12:
            raise ValueError(
                "A_i is singular at (alpha, omega) = (0, 0) for at least one "
                "block; B_bar is undefined there"
            )

        # closed-form inverse of A_i = alpha*I + omega*J
        inv_diag = alpha / denom
        inv_off = omega / denom
        inv_row0 = torch.stack([inv_diag, inv_off], dim=-1)
        inv_row1 = torch.stack([-inv_off, inv_diag], dim=-1)
        A_inv = torch.stack([inv_row0, inv_row1], dim=-2)

        eye = torch.eye(2, dtype=self.dtype, device=alpha.device)
        B = B_cont.reshape(self.n_pairs, 2, -1)
        B_bar = A_inv @ (A_bar - eye) @ B
        return A_bar, B_bar.reshape(2 * self.n_pairs, -1)

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
        delta_t = torch.as_tensor(delta, dtype=self.dtype, device=alpha.device)
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
