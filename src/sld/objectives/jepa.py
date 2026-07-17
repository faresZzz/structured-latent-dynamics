"""JEPA prediction loss and its EMA target encoder.

Latent convention (see repo charter): an observation x is encoded to an
embedding z by an online encoder; a target encoder (an EMA copy of the
online one) produces the regression targets; predictors operate on a latent
state h (h = psi(z) for lifted families with a learned lifting psi, h = z
for plain families). The loss operates against tensors and optional
callables only -- it has no knowledge of concrete encoder/predictor classes,
which the Trainer wires in.
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
    schedule is the Trainer's job. This loss consumes a single,
    already-rolled-out batch of tensors at matching time indices and has no
    notion of a schedule or a horizon itself.

    Wave 3 semantics: ``pred`` lives in PROJECTION space -- it compares the
    predictor's read-out ``pred_z = pi(h_hat)`` (identity for plain, z-space
    families with no lifting) against the stopped target embedding
    ``sg[target_z]``. ``consistency`` is the ``lambda_c``-weighted LIFTED-space
    regularizer ``h_hat ~ sg[psi(z_target)]``: the predictor's raw H-space
    state must match the lift of the (stopped) target embedding.
    ``consistency`` is computed only when BOTH ``h_hat`` and ``lift`` are
    given; passing exactly one of them raises ``ValueError`` -- the classic
    silent half-wired JEPA bug. ``total = pred + lambda_c * consistency``.

    Stop-grad discipline: ``target_z`` is ALWAYS detached before either term
    is formed, regardless of what the caller already did to it, and the
    lifted regression target ``lift(sg[target_z])`` is detached as a WHOLE
    (the plan's ``h_hat ~ sg[psi(z_target)]``): ``psi`` must not be able to
    move the target toward ``h_hat`` through this term. ``psi`` still
    trains through the prediction path, where ``lift(z)`` feeds the
    predictor's input upstream of ``pred_z``.
    """

    def __init__(self, lambda_c: float = 0.0) -> None:
        super().__init__()
        self.lambda_c = lambda_c

    def forward(
        self,
        pred_z: torch.Tensor,
        target_z: torch.Tensor,
        h_hat: torch.Tensor | None = None,
        lift: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # stop-grad on the target regardless of what the caller already did
        z_sg = target_z.detach()

        # exact shape match enforced: F.mse_loss would silently broadcast
        # mismatched time/batch dims into a wrong comparison
        if pred_z.shape != z_sg.shape:
            raise ValueError(
                f"pred_z {tuple(pred_z.shape)} and target_z {tuple(z_sg.shape)} "
                "must match exactly"
            )
        pred = nn.functional.mse_loss(pred_z, z_sg)

        # silent half-wiring lock: h_hat/lift must be given together or not at all
        if (h_hat is None) != (lift is None):
            raise ValueError("h_hat and lift must both be given, or both omitted")

        if h_hat is None:
            consistency = torch.zeros((), dtype=pred.dtype, device=pred.device)
        else:
            # the WHOLE lifted target is stopped (sg[psi(sg[z])], OBJECTIFS
            # 4.5): psi must not be able to move the regression target toward
            # h_hat through this term. psi still trains through the pred path
            # (h_true = lift(z) feeds the predictor upstream).
            lifted_target = lift(z_sg).detach()
            if h_hat.shape != lifted_target.shape:
                raise ValueError(
                    f"h_hat {tuple(h_hat.shape)} and its lifted regression target "
                    f"{tuple(lifted_target.shape)} must match exactly"
                )
            consistency = nn.functional.mse_loss(h_hat, lifted_target)

        total = pred + self.lambda_c * consistency
        return {"total": total, "pred": pred, "consistency": consistency}
