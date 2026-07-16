"""JEPA prediction loss and its EMA target encoder.

Latent convention (see repo charter): an observation x is encoded to an
embedding z by an online encoder; a target encoder (an EMA copy of the
online one) produces the regression targets; predictors operate on a latent
state h (h = psi(z) for lifted families with a learned lifting psi, h = z
for plain families). Wave 2 ships the loss against tensors and optional
callables only -- concrete encoder/predictor classes arrive in Wave 3.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn


class EMATarget(nn.Module):
    """Exponential-moving-average target encoder (JEPA/BYOL-style stop-grad target).

    Wraps a deepcopy of ``online`` and tracks it as xi <- tau * xi + (1 - tau)
    * theta, theta being the online parameters. Two independent guards
    against the classic silent JEPA bug (gradient leaking into the target):
    every target parameter has ``requires_grad_(False)`` set here (belt), and
    ``JEPALoss`` additionally detaches everything it reads from the target
    (suspenders) -- neither guard alone is trusted.
    """

    def __init__(self, online: nn.Module, tau: float = 0.996) -> None:
        super().__init__()
        # the update is a convex combination only for tau in [0, 1];
        # anything else silently extrapolates (adversarial-review lock)
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.tau = tau
        self.target = copy.deepcopy(online)
        # deterministic target: eval mode freezes any train-time stochastic
        # or stats-mutating layers (dropout, BatchNorm batch-stats). A no-op
        # for this repo's BN-free MLP encoders, declared for safety.
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> EMATarget:
        # nn.Module.train() recurses into children, which would silently
        # re-arm the target's train mode (e.g. via an enclosing model's
        # .train()) — the target's eval mode must be persistent
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        """EMA-update parameters; copy buffers verbatim.

        Static-topology contract: ``online`` must keep the same parameter/
        buffer names as at construction — no tying, adding, renaming or
        removing afterwards (lookup is by canonical name; a rename raises
        KeyError, a removal would silently leave stale target state).

        Buffer policy (declared, not claimed optimal): buffers are copied
        verbatim, i.e. the target SHARES the online network's running
        statistics, and the target always runs in eval mode. For this
        repo's encoders (BN-free MLPs) the branch is inert. If a future
        encoder carries BatchNorm, revisit explicitly: copied online
        moments under EMA-lagged weights are a design choice, not a
        mathematical necessity.
        """
        target_params = dict(self.target.named_parameters())
        for name, p_online in online.named_parameters():
            p_target = target_params[name]
            p_target.mul_(self.tau).add_(p_online, alpha=1.0 - self.tau)

        target_buffers = dict(self.target.named_buffers())
        for name, b_online in online.named_buffers():
            target_buffers[name].copy_(b_online)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # params are already frozen and the caller detaches; no_grad here
        # only saves graph memory, it is not load-bearing for correctness.
        with torch.no_grad():
            return self.target(x)


class JEPALoss(nn.Module):
    """JEPA prediction loss, horizon-agnostic.

    Scoping decision: the teacher-forcing / free-running rollout horizon
    schedule is the Trainer's job (Wave 3). This loss consumes a single,
    already-rolled-out batch of (h_hat, z_target) pairs at matching time
    indices and has no notion of a schedule or a horizon itself.

    Design decision (Wave 2, lifted families): with a lifting ``psi`` but no
    read-only projection ``pi`` back to z-space yet (pi arrives in Wave 3),
    the only loss we can honestly write is "h_hat ~ sg[psi(z)]" -- the
    predictor must match the *stopped* lift of the target embedding. That
    single quantity is computed once as ``pred``; ``consistency`` is left at
    zero and ``total == pred``. ``lambda_c`` is stored but inert at Wave 2:
    once pi exists, ``pred`` moves to projection space and the lifted term
    becomes the lambda_c-weighted consistency regularizer. This is a
    deliberate scoping choice, not an oversight -- Wave 3 attaches to
    ``lambda_c`` and to the ``consistency`` key.
    """

    def __init__(self, lambda_c: float = 0.0) -> None:
        super().__init__()
        self.lambda_c = lambda_c

    def forward(
        self,
        h_hat: torch.Tensor,
        z_target: torch.Tensor,
        lift: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # stop-grad on the target regardless of what the caller already did
        z_sg = z_target.detach()

        if lift is None:
            target_tensor = z_sg
        else:
            # both the target embedding and its lift are stopped: this term
            # shapes only the predictor path (h_hat), never psi.
            target_tensor = lift(z_sg).detach()
        # exact shape match enforced: F.mse_loss would silently broadcast
        # mismatched time/batch dims into a wrong comparison
        if h_hat.shape != target_tensor.shape:
            raise ValueError(
                f"h_hat {tuple(h_hat.shape)} and its regression target "
                f"{tuple(target_tensor.shape)} must match exactly"
            )
        pred = nn.functional.mse_loss(h_hat, target_tensor)

        consistency = torch.zeros((), dtype=pred.dtype, device=pred.device)
        total = pred
        return {"total": total, "pred": pred, "consistency": consistency}
