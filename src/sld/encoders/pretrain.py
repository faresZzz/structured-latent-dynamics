"""The frozen shared representation: one encoder + one linear state readout
per ``(system, regime)`` stratum, shared by all six arms of the 3x2 design
(PROTOCOL.md sections 12.1, 12.3, 14.10 -- 14.10 supersedes 14.9's LOSS while
keeping 14.9's scope unchanged). Wave D2a/D2a-v2.

``pretrain_representation`` implements section 14.10's committed contract: a
two-view self-distillation objective, DINO/BYOL-shaped, with NO predictor
anywhere. Two views are two frames of the SAME episode at a short temporal
offset (``_VIEW_OFFSET``, a judgment -- see below); a student encoder is
trained against a stop-gradient EMA teacher (``sld.objectives.jepa.EMATarget``,
reused rather than rebuilt), with the alignment term taken DIRECTLY between
the two encoders' raw outputs (``sld.objectives.jepa.JEPALoss`` with
``lambda_c=0`` and no ``h_hat``/``lift`` -- this already IS "student vs
sg[teacher], no head, no predictor", so it is reused rather than rewritten).
VICReg variance + covariance on the student embedding are RETAINED, not
optional: section 14.10 is explicit that predictor-free BYOL with nothing
else collapses to a constant (BYOL's own ablation), and this project already
carries the standard non-contrastive anti-collapse mechanism, which stands in
for DINO's centering/sharpening. The linear readout's reconstruction term is
also RETAINED from 14.9, because ``e_recon``/``tau_1`` are only meaningful if
the latent stays linearly decodable -- an invariant-but-undecodable latent
would publish every stratum ungated under section 12.3.

**The declared tension (14.10 asks this be stated, not hidden).** The
alignment term pulls the encoder toward being INVARIANT across the temporal
offset -- toward smoothing the dynamics out of the representation. The
readout reconstruction term pulls the opposite way, toward STATE FIDELITY
(the embedding must stay linearly decodable to the true state, which changes
between the two views). ``view_offset = 1`` (the default, ``_VIEW_OFFSET``) keeps this
tension mild: two adjacent frames of a physical system are close in state
space, so "invariant" and "decodable" are not yet in sharp conflict. A future
reader who raises the offset trades a stronger invariance signal for a
sharper fight between the two terms -- past some offset the two terms pull
against each other hard enough that the readout's R^2 (already reported,
never hidden) is expected to degrade.

The frozen artifact also carries ``e_recon``, section 12.3's reconstruction
floor: ``readout(encoder(o_t))`` against the true state, on HELD-OUT data,
with zero predictors trained. ``tau_1 = 3 * e_recon`` is the positive-control
gate's upper bound; the pre-registered sanity condition ``3 * e_recon <
e_1^pers`` is checked here, once, and recorded either way (never adjusted,
never a raise -- a stratum where it fails is published as ungated).

The FROZEN encoder is the STUDENT (online) network, never the teacher: the
teacher is a training-time-only EMA scaffold that supplies regression
targets, exactly like the gradient-trained ``readout_head`` below is
training-time-only and discarded after training. This is a judgment, not
derived from 14.10 (which does not say which of the two becomes the
artifact) -- consistent with the readout already being fit against the
student's embeddings, and with 14.9's unchanged artifact contract, which
never mentions a teacher.

The readout (``ReadoutFit``, ``fit_readout``, ``apply_readout``) is
RECOVERED, not rewritten, from ``sld.viz.fig1_rollout`` as it stood at commit
``92eb903`` -- before the phase-B cleanup (``af4e762``) deleted it. It carries
six rounds of adversarial review and properties a fresh reimplementation
would risk losing: the OLS solve, rank, conditioning and R^2 are computed on
the COLUMN-EQUILIBRATED affine design (each column of ``[z | 1]`` divided by
its own RMS, so a harmless per-coordinate unit change of the latent cannot
masquerade as new degeneracy), ``W`` is transformed back so applying it to
raw coordinates is unchanged, and the resulting ``indeterminate`` verdict is
stable under the entire diagonal-rescaling class -- declared gauge-dependent
only under general shears, which is the one residual limit no continuous
criterion can remove (exact rank is the only GL(d)-invariant notion of
degeneracy).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from sld.audit.rollout import fit_train_state_scale, normalized_state_error, persistence_rollout
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.objectives.vicreg import covariance_loss, variance_loss
from sld.training.seeding import set_seed

# The temporal offset defining the two self-distillation views (section
# 14.10): view0 = frame t, view1 = frame t + _VIEW_OFFSET, same episode.
# DECLARED, not derived (14.10's own wording) -- 1 (adjacent frames) is the
# mildest possible offset, chosen specifically to keep the invariance-vs-
# reconstruction tension (module docstring, above) as small as it can be
# while still being a genuine second view.
_VIEW_OFFSET: int = 1
assert _VIEW_OFFSET >= 1

# latents are stored float32; storage noise is relative PER COORDINATE
# (~1.19e-7 of each coordinate's own magnitude) -- ported unchanged from
# fig1_rollout.py@92eb903, same constants, same reasoning.
_LATENT_EPS: float = float(np.finfo(np.float32).eps)
_INDETERMINATE_SENSITIVITY: float = 1e-2
assert np.isfinite(_LATENT_EPS) and np.isfinite(_INDETERMINATE_SENSITIVITY)


class ReadoutFit(NamedTuple):
    """What ``fit_readout`` measured about one linear+bias readout fit
    (ported from ``fig1_rollout.py@92eb903``'s ``ReadoutFit``, public name).

    Attributes:
        W: fitted readout, ``(d_z + 1, d_state)`` (last row is the bias
            term, fit via a ones column).
        r2: coefficient of determination on the true state -- reported,
            never hidden (REGLES section 5). Gauge-invariant (fitted values
            are a projection onto the design's column span).
        rank: numerical rank of the COLUMN-EQUILIBRATED ``[z | 1]`` design.
            This is the rank the determinability verdict uses; the raw
            least-squares rank is deliberately not used because its
            floating-point cutoff changes under a harmless unit rescale.
        cond: condition number (largest / smallest singular value) of the
            COLUMN-EQUILIBRATED design -- ``inf`` when that design is
            rank-deficient or any column has zero RMS. This internal
            undefined value is exported as an empty CSV field by callers,
            never as a non-finite public metric. The OLS solve is performed
            on this same equilibrated design; ``W`` is transformed back so
            applying it to raw coordinates is unchanged.
        sensitivity: ``cond * _LATENT_EPS`` -- the relative indeterminacy of
            the decoded output induced by float32 latent storage noise,
            measured on the equilibrated design so a per-coordinate unit
            change cannot masquerade as sensitivity.
        indeterminate: ``True`` when the equilibrated design is
            column-rank-deficient or ``sensitivity > _INDETERMINATE_SENSITIVITY``
            -- the decoded output is then not numerically determined by the
            data at the latents' storage precision. STABLE under the entire
            diagonal-rescaling class (any per-coordinate unit change);
            DECLARED gauge-dependent only under general shears.
    """

    W: np.ndarray
    r2: float
    rank: int
    cond: float
    sensitivity: float
    indeterminate: bool


def fit_readout(z: np.ndarray, x_true: np.ndarray) -> ReadoutFit:
    """OLS-fit a linear+bias readout ``z -> x_true`` (ported from
    ``fig1_rollout.py@92eb903``'s ``_fit_readout``, public name; ``z_enc``/
    ``z_gt`` renamed ``z``/``x_true`` -- this module has no encoder-embedding-
    vs-ground-truth distinction to keep separate, so the old names carried no
    extra information here).

    ``z``/``x_true`` are ``(N, ·)`` float64 -- flattened over time and batch.

    The OLS solve, rank, conditioning and R^2 are all computed on the
    COLUMN-EQUILIBRATED affine design (each column of ``[z | 1]`` divided by
    its own RMS). The resulting coefficients are transformed back by the
    inverse column scale, so applying ``W`` to the raw design is exactly
    unchanged. A zero-RMS column is rank-deficient by construction and
    short-circuits straight to "indeterminate" rather than dividing by zero.
    This is what makes the verdict stable under the entire diagonal-rescaling
    class (e.g. ``diag(1e9,1)`` or a uniform ``2**20``): multiplying every
    latent coordinate by ``2**20`` is exactly representable and round-trips
    bit-exact through float32, so it destroys no information, yet a RAW
    ``lstsq`` on the un-equilibrated design would see its cond inflate from
    ~1 to ~1e6 purely because the intercept column stays fixed at 1 while the
    z columns are rescaled -- a unit mismatch, not degeneracy. Genuine
    near-collinearity between latent coordinates (e.g. a near-collapse
    ``z2 ~ z1``) is still caught: equilibrated columns that are nearly
    identical still blow up the equilibrated cond.

    Returns:
        ``ReadoutFit`` -- ``rank``, ``cond`` and ``sensitivity`` all come
        from the same equilibrated design, never recomputed elsewhere, so
        the indeterminacy classification cannot drift under a diagonal unit
        rescale.

    Raises:
        ValueError: ``x_true`` has zero variance across the split (R^2
            undefined), or the fit produces a non-finite R^2.
    """
    design = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)

    # RMS-normalize each column so a per-coordinate unit change (the
    # intercept column staying fixed at 1 while z columns are rescaled)
    # cannot masquerade as degeneracy. Compute each RMS after max-scaling
    # its column: direct ``mean(design**2)`` would overflow for a valid
    # large unit rescale and underflow for a valid small one.
    col_max = np.abs(design).max(axis=0)
    scaled = design / np.where(col_max == 0.0, 1.0, col_max)
    col_rms = col_max * np.sqrt(np.mean(scaled**2, axis=0))
    # A zero column is retained as zero (rather than divided by zero), so
    # lstsq can report its rank deficiency on the same solved design.
    column_scale = np.where(col_rms == 0.0, 1.0, col_rms)
    design_bal = design / column_scale
    W_eq, _, rank, singular_values_bal = np.linalg.lstsq(design_bal, x_true, rcond=None)
    W = W_eq / column_scale[:, None]
    fitted = design_bal @ W_eq
    ss_res = float(np.sum((x_true - fitted) ** 2))
    ss_tot = float(np.sum((x_true - x_true.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot <= 0.0:
        raise ValueError("x_true has zero variance across the split; readout R^2 is undefined")
    r2 = 1.0 - ss_res / ss_tot
    if not np.isfinite(r2):
        raise ValueError(f"readout R^2 is non-finite ({r2}) -- bug, not a result")
    rank = int(rank)
    if np.any(col_rms == 0.0) or rank < design.shape[1] or singular_values_bal[-1] == 0.0:
        cond = float("inf")
        sensitivity = float("inf")
    else:
        cond = float(singular_values_bal[0] / singular_values_bal[-1])
        sensitivity = cond * _LATENT_EPS
    indeterminate = rank < design.shape[1] or sensitivity > _INDETERMINATE_SENSITIVITY
    return ReadoutFit(
        W=W, r2=r2, rank=int(rank), cond=cond, sensitivity=sensitivity, indeterminate=indeterminate
    )


def apply_readout(z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply the fitted readout ``W`` (``(d_z+1, d_state)``, bias last row)
    to ``z`` (``(..., d_z)``), returning ``(..., d_state)``. Ported from
    ``fig1_rollout.py@92eb903``'s ``_apply_readout``, public name, unchanged."""
    ones = np.ones((*z.shape[:-1], 1), dtype=z.dtype)
    return np.concatenate([z, ones], axis=-1) @ W


def _tau1_condition_holds(e_recon: float, e_1_pers: float) -> bool:
    """PROTOCOL.md section 12.3's pre-registered sanity condition, checked
    once at D2 before any predictor is trained: ``3 * e_recon < e_1_pers``
    (STRICT). Returns the verdict only -- never raises, never adjusts either
    number. A stratum where this does NOT hold is recorded as non-gated
    (section 14.9): "the numbers are not adjusted" is section 12.3's own
    wording.
    """
    return 3.0 * e_recon < e_1_pers


def _freeze_and_measure(
    encoder: nn.Module, train: dict, val: dict, d_x: int, d_state: int
) -> tuple["ReadoutFit", np.ndarray, float, float, bool]:
    """Everything the manifest needs from an ALREADY-FROZEN ``encoder``: the
    canonical OLS readout (fit on the FULL train split), ``e_recon`` and
    ``e_1_pers`` (both on held-out val), and the section 12.3 gate verdict.

    Factored out of ``pretrain_representation`` (Wave D3) so
    ``build_identity_representation`` -- whose encoder is hand-set to the
    identity rather than trained -- runs the EXACT SAME measurement code.
    "e_recon = 0 for the identity encoder" must be a MEASURED outcome, never
    a constant written by hand (R2, REGLES section 7.1); sharing this
    function is what makes that true by construction rather than by
    discipline. Callers own freezing (``eval()`` + ``requires_grad_(False)``)
    and any seeding -- this function only measures.
    """
    # fit against the ORIGINAL float64 z_true (never a float32-rounded
    # copy): the OLS solve is meant to see full RK4-oracle precision. The
    # ENCODER's forward pass still runs at float32 (the input is cast
    # below) -- the same CPU/float32 policy every encoder in this repo is
    # evaluated under, trained or hand-built alike.
    x_train_full = torch.as_tensor(train["x"], dtype=torch.float32).reshape(-1, d_x)
    with torch.no_grad():
        z_train_np = encoder(x_train_full).double().numpy()
    x_true_train_np = np.asarray(train["z_true"], dtype=np.float64).reshape(-1, d_state)
    readout = fit_readout(z_train_np, x_true_train_np)

    sigma_train = fit_train_state_scale(train["z_true"])

    # -- e_recon: the readout's own reconstruction error, on HELD-OUT (val)
    # data -- PROTOCOL.md section 12.3, "readout(encoder(o_t)) against x_t",
    # measured with the ONE existing normalized_state_error (H=1: every held-
    # out frame is scored independently, there is no rollout here) --
    x_val = torch.as_tensor(val["x"], dtype=torch.float32).reshape(-1, d_x)
    x_true_val_np = np.asarray(val["z_true"], dtype=np.float64).reshape(-1, d_state)
    with torch.no_grad():
        z_val_np = encoder(x_val).double().numpy()
    pred_val = apply_readout(z_val_np, readout.W)
    e_recon = float(
        normalized_state_error(pred_val[:, None, :], x_true_val_np[:, None, :], sigma_train)[0]
    )

    # -- e_1_pers: the persistence baseline's one-step error, on the SAME
    # held-out split (section 12.3's baselines share "same coordinates, same
    # episodes, same sigma^train") -- reuses persistence_rollout, no second
    # implementation --
    x0 = np.asarray(val["z_true"][:, 0, :], dtype=np.float64)
    x1 = np.asarray(val["z_true"][:, 1, :], dtype=np.float64)
    pers_pred = persistence_rollout(x0, horizon=1)
    e_1_pers = float(normalized_state_error(pers_pred, x1[:, None, :], sigma_train)[0])

    gated = _tau1_condition_holds(e_recon, e_1_pers)
    return readout, sigma_train, e_recon, e_1_pers, gated


@dataclass
class FrozenRepresentationArtifact:
    """One frozen ``(encoder, readout)`` pair for one ``(system, regime)``
    stratum -- the object all six D2b arms of that stratum share.

    Attributes:
        encoder_state_dict: ``FrameMLPEncoder.state_dict()`` of the frozen
            encoder.
        readout: the linear state readout fit on the FIT split (``fit_readout``
            above), with its full determinability diagnostics.
        sigma_train: per-coordinate train-split scale
            (``sld.audit.rollout.fit_train_state_scale``), frozen before any
            evaluation -- PROTOCOL.md section 12.1's ``sigma_i^train``.
        e_recon: the readout's own reconstruction error in true-state
            coordinates (``sld.audit.rollout.normalized_state_error``),
            measured on HELD-OUT data -- PROTOCOL.md section 12.3's
            reconstruction floor.
        manifest: identity and provenance -- ``system``, ``regime``,
            ``fit_split``, ``eval_split``, encoder architecture
            (``encoder_class``, ``d_x``, ``d_z``, ``encoder_hidden``),
            ``pretraining_seed``, ``dtype``, ``device``, the measured
            ``e_1_pers`` and the section 12.3 gate verdict (``gated``), and
            -- once saved -- the content ``digest``.
    """

    encoder_state_dict: dict[str, torch.Tensor]
    readout: ReadoutFit
    sigma_train: np.ndarray
    e_recon: float
    manifest: dict


def _build_two_views(
    train: dict, d_x: int, d_state: int, view_offset: int = _VIEW_OFFSET
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two self-distillation views (PROTOCOL.md section 14.10): view0 =
    frame ``t``, view1 = frame ``t + view_offset``, SAME episode. Row ``i``
    of ``x_view0``/``x_view1``/``x_true_view0`` all come from the same
    ``(episode, t)`` pair (identical slicing + reshape order on all three,
    so the pairing survives the flatten into a batch).

    Factored out for direct, positive-control testing (REGLES section
    7.1): an off-by-offset bug (e.g. views collapsed to the same frame) is
    invisible to an end-to-end training test -- var/cov/recon still operate
    normally and dominate the loss -- but is exactly this function's job to
    get right, so it is tested directly against hand-built, exactly-known
    frame values.
    """
    # view_offset >= 1 is enforced by the caller AND here: at offset 0 the
    # slice `[:, :-0, :]` is EMPTY (a python slice trap), and the objective
    # would silently become BYOL on two IDENTICAL views -- which is the
    # collapsing configuration, not the pure-autoencoder ablation someone
    # reaching for offset 0 would expect. That ablation is `w_align = 0`
    # (PROTOCOL.md section 14.11); the two are NOT interchangeable, because
    # the teacher is an EMA of the student, so an identical-view alignment
    # term is still non-zero and still pulls.
    if view_offset < 1:
        raise ValueError(
            f"view_offset must be >= 1, got {view_offset}; to remove the temporal "
            "term entirely set w_align = 0 (the pure-autoencoder ablation of "
            "PROTOCOL.md section 14.11) -- offset 0 would pair each frame with "
            "ITSELF against an EMA teacher, which is the collapsing configuration"
        )
    x_view0 = torch.as_tensor(train["x"][:, :-view_offset, :], dtype=torch.float32).reshape(-1, d_x)
    x_view1 = torch.as_tensor(train["x"][:, view_offset:, :], dtype=torch.float32).reshape(-1, d_x)
    x_true_view0 = torch.as_tensor(
        train["z_true"][:, :-view_offset, :], dtype=torch.float32
    ).reshape(-1, d_state)
    return x_view0, x_view1, x_true_view0


def _train_step(
    encoder: nn.Module,
    readout_head: nn.Module,
    ema: EMATarget,
    jepa: JEPALoss,
    optimizer: torch.optim.Optimizer,
    xb0: torch.Tensor,
    xb1: torch.Tensor,
    xb0_true: torch.Tensor,
    *,
    w_align: float,
    w_var: float,
    w_cov: float,
    w_recon: float,
) -> dict[str, float]:
    """One two-view self-distillation step (PROTOCOL.md section 14.10):
    student forward on view0, stop-gradient EMA teacher forward on view1,
    the four loss terms, backward, optimizer step, EMA refresh.

    Factored OUT of ``pretrain_representation``'s loop for one reason: so a
    test can call the EXACT call site production uses (``jepa(z_student,
    z_teacher)``, never a hand-duplicated loss expression) rather than a
    parallel reimplementation that could drift from it or fail to catch a
    call-site regression (REGLES section 7's "verify the call site, not the
    docstring"). ``pretrain_representation`` itself only needs the trained
    modules' side effects (``encoder``/``readout_head``/``ema`` are mutated
    in place); the returned dict is for callers -- tests -- that want the
    per-step loss values.

    Returns:
        ``{"align", "var", "cov", "recon"}`` -- the four (weight-free) loss
        VALUES for this step, as plain floats, taken AFTER ``optimizer.step()``
        (i.e. the loss the step was computed from, not a post-step
        re-evaluation).
    """
    z_student = encoder(xb0)
    z_teacher = ema(xb1)  # EMATarget.forward: no_grad, target(xb1)
    align = jepa(z_student, z_teacher)["total"]  # detaches z_teacher again (belt+suspenders)
    zero = torch.zeros((), dtype=torch.float32)
    var = variance_loss(z_student) if w_var > 0 else zero
    cov = covariance_loss(z_student) if w_cov > 0 else zero
    recon_pred = readout_head(z_student)
    recon = torch.mean((recon_pred - xb0_true) ** 2)
    loss = w_align * align + w_var * var + w_cov * cov + w_recon * recon
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update(encoder)  # EMA target refreshed after EVERY optimizer step (Trainer's own discipline)
    return {"align": align.item(), "var": var.item(), "cov": cov.item(), "recon": recon.item()}


def pretrain_representation(
    data: dict,
    *,
    system: str,
    regime: str,
    d_z: int,
    encoder_hidden: tuple[int, ...] = (64, 64),
    seed: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    tau_ema: float = 0.99,
    view_offset: int = _VIEW_OFFSET,
    w_align: float = 1.0,
    w_var: float = 1.0,
    w_cov: float = 1.0,
    w_recon: float = 1.0,
) -> FrozenRepresentationArtifact:
    """Pretrain and freeze one shared representation (PROTOCOL.md section
    14.10 -- supersedes 14.9's loss, keeps 14.9's scope). NO predictor
    anywhere in this function. The objective has four terms:

    1. **Alignment**: a student encoder sees frame ``t`` (view0), a
       stop-gradient EMA teacher sees frame ``t + view_offset`` of the SAME
       episode (view1); ``sld.objectives.jepa.JEPALoss`` (``lambda_c=0``, no
       ``h_hat``/``lift``) compares the two RAW embeddings directly -- no
       head, no projector, no predictor between them.
    2. **VICReg variance** on the student embedding -- the anti-collapse term
       14.10 requires (predictor-free BYOL with nothing else collapses).
    3. **VICReg covariance** on the student embedding.
    4. **Linear readout reconstruction**: a jointly-trained ``nn.Linear``
       reconstructs the true state from the student embedding. Training-time
       only, discarded after training -- the artifact's OWN readout
       (``readout`` below, and ``e_recon``) is the CANONICAL one: an
       independent OLS fit (``fit_readout``, the recovered code) on the
       FROZEN student encoder's embeddings, computed after training stops.

    Args:
        data: a ``sld.data.generate.build_dataset`` dict with ``"train"`` and
            ``"val"`` splits (each ``{"x", "u", "z_true"}``). ``"val"`` is
            held out from both the encoder's training loop and the readout's
            OLS fit -- ``e_recon`` and ``e_1_pers`` are both measured there.
        system, regime: stratum identity, recorded in the manifest verbatim
            (this function does not interpret them).
        d_z: latent dimension.
        encoder_hidden: ``FrameMLPEncoder`` hidden layer sizes.
        seed: pretraining seed (``sld.training.seeding.set_seed`` is called
            first, exactly as ``Trainer.fit`` does, so model construction is
            seeded -- run identity = config + seed).
        epochs, batch_size, lr, weight_decay: training-loop knobs.
        tau_ema: the teacher's EMA decay (``EMATarget``'s own parameter,
            same default as ``Trainer``'s ``tau_ema``).
        w_align, w_var, w_cov, w_recon: objective weights (section 14.10's
            four terms).

    Returns:
        A ``FrozenRepresentationArtifact`` with an UNSTAMPED manifest (no
        ``"digest"`` key yet -- ``save_frozen_representation`` stamps one).

    Raises:
        ValueError: ``data`` is missing a required split, the train split
            has too few physical steps per episode for a ``view_offset``
            pair, fewer than 2 view-pairs overall (VICReg's variance needs
            N >= 2), or the val split has fewer than 2 physical steps
            (``e_1_pers`` needs a step-1 target).
    """
    for split_name in ("train", "val"):
        if split_name not in data:
            raise ValueError(f"data must include a {split_name!r} split")

    train, val = data["train"], data["val"]
    d_x = train["x"].shape[-1]
    d_state = train["z_true"].shape[-1]
    steps_per_episode = train["z_true"].shape[1]
    if view_offset < 1:
        raise ValueError(
            f"view_offset must be >= 1, got {view_offset}; the pure-autoencoder "
            "ablation is w_align = 0 (PROTOCOL.md section 14.11), NOT offset 0"
        )
    if steps_per_episode - view_offset < 1:
        raise ValueError(
            f"train split has {steps_per_episode} physical steps per episode, too few "
            f"for a temporal offset of {view_offset} (need at least {view_offset + 1})"
        )
    n_pairs = train["z_true"].shape[0] * (steps_per_episode - view_offset)
    if n_pairs < 2:
        raise ValueError(
            f"train split has {n_pairs} view-pairs at offset {view_offset}; VICReg "
            "variance needs N >= 2"
        )
    if val["z_true"].shape[1] < 2:
        raise ValueError(
            "val split needs at least 2 physical steps (T >= 1) to measure e_1_pers"
        )

    set_seed(seed)  # seed BEFORE model construction -- run identity = config + seed

    # declared CPU/float32 policy, enforced at construction (Trainer.fit's
    # own discipline: init draws consume the RNG differently per dtype)
    ambient = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        encoder = FrameMLPEncoder(d_x, d_z, tuple(encoder_hidden))  # the STUDENT
        readout_head = nn.Linear(d_z, d_state)  # training-time only, discarded after
    finally:
        torch.set_default_dtype(ambient)

    ema = EMATarget(encoder, tau=tau_ema)  # the TEACHER: reused, not rebuilt
    jepa = JEPALoss(lambda_c=0.0)  # "student vs sg[teacher], no head" -- reused, not rewritten

    # two views, same episode, offset view_offset: row i of view0 and view1
    # both come from (episode, t) pair i (identical slicing + reshape order
    # on both, so the pairing survives the flatten)
    x_view0, x_view1, x_true_view0 = _build_two_views(train, d_x, d_state, view_offset)
    n = x_view0.shape[0]

    params = list(encoder.parameters()) + list(readout_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue  # VICReg variance needs N >= 2; drop a ragged last batch of 1
            _train_step(
                encoder,
                readout_head,
                ema,
                jepa,
                optimizer,
                x_view0[idx],
                x_view1[idx],
                x_true_view0[idx],
                w_align=w_align,
                w_var=w_var,
                w_cov=w_cov,
                w_recon=w_recon,
            )

    # -- freeze: the CANONICAL readout is an independent OLS fit (recovered
    # code) on the frozen STUDENT encoder's TRAIN-split embeddings -- ALL
    # frames, not just the view0 half used for training (the readout fit is
    # a separate, post-hoc step, not part of the two-view objective).
    # e_recon/e_1_pers/the gate verdict share ONE measurement function with
    # build_identity_representation's hand-set encoder (_freeze_and_measure,
    # R2) -- so "measured, not postulated" holds for both callers alike.
    encoder.eval()
    readout, sigma_train, e_recon, e_1_pers, gated = _freeze_and_measure(
        encoder, train, val, d_x, d_state
    )

    manifest = {
        "system": system,
        "regime": regime,
        "fit_split": "train",
        "eval_split": "val",
        "encoder_class": "FrameMLPEncoder",
        "d_x": int(d_x),
        "d_z": int(d_z),
        "encoder_hidden": [int(h) for h in encoder_hidden],
        "pretraining_seed": int(seed),
        "dtype": str(next(encoder.parameters()).dtype),
        "device": str(next(encoder.parameters()).device),
        "e_1_pers": e_1_pers,
        "tau1_threshold_factor": 3.0,
        "gated": gated,
        "objective": "14.10",  # two-view self-distillation; supersedes 14.9's loss
        "view_offset": view_offset,
        "tau_ema": float(tau_ema),
    }

    return FrozenRepresentationArtifact(
        encoder_state_dict={k: v.detach().clone() for k, v in encoder.state_dict().items()},
        readout=readout,
        sigma_train=sigma_train,
        e_recon=e_recon,
        manifest=manifest,
    )


def build_identity_representation(
    data: dict, *, system: str, regime: str, d_z: int
) -> FrozenRepresentationArtifact:
    """D3's control-stratum representation (PROTOCOL.md section 19.1): the
    EXACT identity, not a trained encoder. D3 is a capability control on the
    predictor/optimizer, not an arm comparison -- routing it through the
    shared trained representation introduces the readout's own
    reconstruction error as a THIRD error source (on top of learning and
    float64 arithmetic), which measured 10x above D3's ``1e-3`` bound and
    made the state bound unattainable by construction. Removing that
    confound means ``d_z = d_x``, encoder = identity, readout ``W = I``, so
    ``e_recon`` measures (at the float32 noise floor -- see below) rather
    than confounds.

    ``d_z`` MUST equal ``d_x`` (the observation dimension found in ``data``):
    the identity has no free width to pad or truncate with, so a mismatch is
    refused explicitly here rather than silently zero-padding or truncating.

    The frozen encoder is a REAL, ordinary ``FrameMLPEncoder`` (``hidden=()``
    -- a single ``nn.Linear``, no activation), never a new module class: its
    weight is hand-set to the identity matrix and its bias to zero, then
    frozen (``eval()``, every parameter ``requires_grad=False``) -- the SAME
    class ``Trainer.fit`` always constructs and loads a representation's
    ``state_dict`` into, so no downstream code path (trainer, gate, digests,
    run identity) has to know this artifact is an identity rather than a
    trained one.

    ``e_recon``/``e_1_pers``/the readout are all MEASURED by
    ``_freeze_and_measure`` -- the EXACT function ``pretrain_representation``
    runs on its own trained encoder -- never postulated: the manifest's
    "e_recon = 0" claim is a result of running the same code on a
    hand-frozen encoder, not a constant written by hand. Because the
    encoder's forward pass runs under this repo's declared CPU/float32
    policy (the same policy ``Trainer.fit`` always applies before loading
    ANY representation's weights, regardless of what dtype built them), the
    measured ``z`` is bit-identical to the FLOAT32-ROUNDED observation, so
    ``e_recon`` lands at the float32 relative-precision floor (~1e-7), not
    at float64's (~1e-16) -- still about four orders of magnitude under
    D3's ``1e-3`` bound, and, as everywhere else in this module, a measured
    number rather than an assumed one.

    Raises:
        ValueError: ``data`` is missing a required split, or ``d_z != d_x``.
    """
    for split_name in ("train", "val"):
        if split_name not in data:
            raise ValueError(f"data must include a {split_name!r} split")
    train, val = data["train"], data["val"]
    d_x = train["x"].shape[-1]
    if d_z != d_x:
        raise ValueError(
            f"identity representation requires d_z == d_x, got d_z={d_z}, d_x={d_x} -- "
            "the identity encoder has no free width to pad or truncate with "
            "(PROTOCOL.md section 19.1)"
        )
    d_state = train["z_true"].shape[-1]

    # declared CPU/float32 policy, enforced at construction -- same
    # discipline as pretrain_representation/Trainer.fit, even though no
    # random init draw happens here (d_z == d_x makes the weight shape
    # deterministic, but the module's PARAMETER dtype must still match what
    # every downstream construction of FrameMLPEncoder uses)
    ambient = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        encoder = FrameMLPEncoder(d_x, d_z, hidden=())  # single nn.Linear, no activation
    finally:
        torch.set_default_dtype(ambient)
    with torch.no_grad():
        weight = encoder.net[0].weight  # (d_z, d_x): out_features x in_features
        weight.copy_(torch.eye(d_z, d_x, dtype=weight.dtype))
        encoder.net[0].bias.zero_()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    readout, sigma_train, e_recon, e_1_pers, gated = _freeze_and_measure(
        encoder, train, val, d_x, d_state
    )

    manifest = {
        "system": system,
        "regime": regime,
        "fit_split": "train",
        "eval_split": "val",
        "encoder_class": "FrameMLPEncoder",
        "d_x": int(d_x),
        "d_z": int(d_z),
        "encoder_hidden": [],
        # inert: the identity encoder has no random init draw to seed
        "pretraining_seed": 0,
        "dtype": str(next(encoder.parameters()).dtype),
        "device": str(next(encoder.parameters()).device),
        "e_1_pers": e_1_pers,
        "tau1_threshold_factor": 3.0,
        "gated": gated,
        "objective": "19.1",  # exact identity; D3's control stratum only
        "kind": "identity",
    }

    return FrozenRepresentationArtifact(
        encoder_state_dict={k: v.detach().clone() for k, v in encoder.state_dict().items()},
        readout=readout,
        sigma_train=sigma_train,
        e_recon=e_recon,
        manifest=manifest,
    )


def representation_digest(artifact: FrozenRepresentationArtifact) -> str:
    """SHA1 over ``artifact``'s content -- a PURE function of content (REGLES
    section 7.1's positive/negative pair: identical content gives identical
    digests; flipping one bit of the encoder changes it). The manifest's own
    ``"digest"`` key (present only after a save/load round-trip) is excluded
    from the hash, since it records THIS function's own output -- including
    it would make the digest a function of itself.
    """
    h = hashlib.sha1()
    for key in sorted(artifact.encoder_state_dict):
        tensor = artifact.encoder_state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(np.array(tensor.shape, dtype=np.int64).tobytes())
        h.update(tensor.numpy().tobytes())
    h.update(np.ascontiguousarray(artifact.readout.W, dtype=np.float64).tobytes())
    h.update(
        repr(
            (
                artifact.readout.r2,
                artifact.readout.rank,
                artifact.readout.cond,
                artifact.readout.sensitivity,
                artifact.readout.indeterminate,
            )
        ).encode("utf-8")
    )
    h.update(np.ascontiguousarray(artifact.sigma_train, dtype=np.float64).tobytes())
    h.update(repr(float(artifact.e_recon)).encode("utf-8"))
    manifest_for_hash = {k: v for k, v in artifact.manifest.items() if k != "digest"}
    h.update(json.dumps(manifest_for_hash, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


_REQUIRED_ARTIFACT_KEYS = frozenset({"encoder_state_dict", "readout", "sigma_train", "e_recon", "manifest"})
_REQUIRED_READOUT_KEYS = frozenset({"W", "r2", "rank", "cond", "sensitivity", "indeterminate"})
_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "system",
        "regime",
        "fit_split",
        "eval_split",
        "encoder_class",
        "d_x",
        "d_z",
        "encoder_hidden",
        "pretraining_seed",
        "dtype",
        "device",
        "digest",
    }
)


def save_frozen_representation(artifact: FrozenRepresentationArtifact, path: str | Path) -> str:
    """Atomic save (temp file + ``os.replace`` in the same directory --
    identical discipline to ``sld.training.trainer``'s ``run_complete.json``
    and resume-checkpoint writes: a kill mid-write never leaves a partially
    written artifact at ``path``). Stamps the artifact's OWN content digest
    (``representation_digest``) into a COPY of its manifest before writing --
    ``artifact.manifest`` itself is left unstamped -- so a later load can
    detect tampering (``validate_shared_representation``).

    Returns:
        The stamped digest.
    """
    digest = representation_digest(artifact)
    manifest = dict(artifact.manifest)
    manifest["digest"] = digest
    payload = {
        "encoder_state_dict": artifact.encoder_state_dict,
        "readout": {
            "W": artifact.readout.W,
            "r2": artifact.readout.r2,
            "rank": artifact.readout.rank,
            "cond": artifact.readout.cond,
            "sensitivity": artifact.readout.sensitivity,
            "indeterminate": artifact.readout.indeterminate,
        },
        "sigma_train": artifact.sigma_train,
        "e_recon": artifact.e_recon,
        "manifest": manifest,
    }
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return digest


def load_frozen_representation(path: str | Path) -> FrozenRepresentationArtifact:
    """Load a ``FrozenRepresentationArtifact`` saved by
    ``save_frozen_representation``, refusing loudly rather than coercing.

    Three distinct refusals, each ``RuntimeError`` with a distinctive
    message fragment, none of them ever adopting or silently discarding the
    offending file:

    - unreadable/corrupt file ("corrupt or unreadable");
    - missing required keys, at the top level, in ``readout``, or in
      ``manifest`` ("corrupt: missing" / "missing required");
    - a stored encoder tensor whose dtype or device disagrees with the
      manifest's own declared ``"dtype"``/``"device"`` ("dtype mismatch" /
      "device mismatch") -- this is a self-consistency check (the manifest
      records what was true when it was saved; a mismatch means the file was
      hand-edited or corrupted since), never a cross-device/cross-dtype
      coercion. This function always loads onto CPU
      (``torch.load(..., map_location="cpu")``); it does not accept a target
      device or dtype to convert to.

    This function does NOT verify the content digest against
    ``manifest["digest"]`` -- that is ``validate_shared_representation``'s
    job (the preflight a caller runs before trusting an artifact for a
    specific stratum), kept separate because it is the more expensive,
    semantic check.
    """
    path = Path(path)
    try:
        # weights_only=False is required, not a default left unexamined
        # (same trust-boundary note as trainer.py's _load_resume_checkpoint):
        # the payload carries a nested dict (readout diagnostics, manifest),
        # not a bare tensor. Trust boundary: this file is written ONLY by
        # save_frozen_representation above -- never sourced from a peer, a
        # download, or any other untrusted origin.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"frozen representation artifact at {path} is corrupt or unreadable ({exc!r}) -- "
            "refusing to adopt or silently discard it"
        ) from exc

    if not isinstance(payload, dict) or not _REQUIRED_ARTIFACT_KEYS <= set(payload):
        raise RuntimeError(
            f"frozen representation artifact at {path} is corrupt: missing top-level keys -- "
            "refusing to adopt or silently discard it"
        )
    manifest = payload["manifest"]
    if not isinstance(manifest, dict) or not _REQUIRED_MANIFEST_KEYS <= set(manifest):
        raise RuntimeError(
            f"frozen representation artifact at {path} is corrupt: manifest missing required "
            "keys -- refusing to adopt or silently discard it"
        )
    readout_payload = payload["readout"]
    if not isinstance(readout_payload, dict) or not _REQUIRED_READOUT_KEYS <= set(readout_payload):
        raise RuntimeError(
            f"frozen representation artifact at {path} is corrupt: readout missing required "
            "keys -- refusing to adopt or silently discard it"
        )

    encoder_state_dict = payload["encoder_state_dict"]
    declared_dtype = manifest["dtype"]
    declared_device = manifest["device"]
    for key, tensor in encoder_state_dict.items():
        if str(tensor.dtype) != declared_dtype:
            raise RuntimeError(
                f"frozen representation artifact at {path}: encoder tensor {key!r} dtype "
                f"{tensor.dtype} does not match manifest-declared dtype {declared_dtype!r} -- "
                "dtype mismatch refused, never coerced"
            )
        if str(tensor.device) != declared_device:
            raise RuntimeError(
                f"frozen representation artifact at {path}: encoder tensor {key!r} device "
                f"{tensor.device} does not match manifest-declared device {declared_device!r} -- "
                "device mismatch refused, never coerced"
            )

    readout = ReadoutFit(
        W=np.asarray(readout_payload["W"]),
        r2=float(readout_payload["r2"]),
        rank=int(readout_payload["rank"]),
        cond=float(readout_payload["cond"]),
        sensitivity=float(readout_payload["sensitivity"]),
        indeterminate=bool(readout_payload["indeterminate"]),
    )
    return FrozenRepresentationArtifact(
        encoder_state_dict=encoder_state_dict,
        readout=readout,
        sigma_train=np.asarray(payload["sigma_train"]),
        e_recon=float(payload["e_recon"]),
        manifest=dict(manifest),
    )


def validate_shared_representation(
    artifact: FrozenRepresentationArtifact, *, system: str, regime: str
) -> None:
    """The Wave D2b preflight: a D2b arm calls this BEFORE trusting an
    artifact for its ``(system, regime)`` cell. Checked in this order, each
    a distinct ``ValueError``:

    1. **Digest.** ``representation_digest(artifact)`` is recomputed and
       compared against ``artifact.manifest["digest"]`` -- a mismatch means
       the artifact's content (encoder weights, readout, ``e_recon``, or any
       manifest field) changed since it was saved.
    2. **Stratum.** ``artifact.manifest["system"]``/``["regime"]`` must equal
       the caller's ``system``/``regime`` -- an artifact from a different
       stratum is refused, never silently reused.
    3. **``e_recon`` usability.** Must be present and finite -- an artifact
       whose reconstruction floor cannot be trusted must not silently enter
       ``tau_1 = 3 * e_recon``.

    Never coerces, never substitutes a value, never raises anything other
    than ``ValueError``.
    """
    recomputed = representation_digest(artifact)
    recorded = artifact.manifest.get("digest")
    if recorded is None or recomputed != recorded:
        raise ValueError(
            f"digest mismatch: artifact content hashes to {recomputed!r}, manifest records "
            f"{recorded!r} -- the artifact was modified after it was saved, refusing to use it"
        )
    if artifact.manifest.get("system") != system or artifact.manifest.get("regime") != regime:
        raise ValueError(
            f"stratum mismatch: artifact is for (system={artifact.manifest.get('system')!r}, "
            f"regime={artifact.manifest.get('regime')!r}), expected (system={system!r}, "
            f"regime={regime!r}) -- refusing an artifact from a different stratum"
        )
    if not math.isfinite(artifact.e_recon):
        raise ValueError(
            f"e_recon is missing or non-finite ({artifact.e_recon!r}) -- refusing an artifact "
            "whose reconstruction floor cannot be trusted"
        )
