"""Config-driven JEPA training loop, one code path for all predictor families.

A run is ``config + seed -> results/<exp_name>/<hash>/`` (REGLES section 6):
deterministic, regenerable, and returning the public
``{"model", "history", "metrics", "paths"}`` contract shared with
``sld-factorial`` so ``viz/`` can read any family's results without knowing
which one produced them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.objectives.vicreg import covariance_loss, variance_loss
from sld.predictors.base import PredictorState
from sld.predictors.registry import get_predictor
from sld.training.seeding import set_seed

# resolved for any key the caller omits; unknown extra keys the caller DOES
# pass are kept as-is (they still flow into the config hash and config.yaml)
_DEFAULTS = {
    "epochs": 50,
    "batch_episodes": 8,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "tau_ema": 0.99,
    "lambda_c": 0.0,
    "w_tf": 1.0,
    "w_fr": 1.0,
    "h_fr": 5,
    "w_var": 1.0,
    "w_cov": 1.0,
    "encoder_hidden": [64, 64],
    "target_params": 10_000,
    "results_root": "results",
}

# model-building keys every run must declare (a run is config + seed, so the
# models MUST be reconstructible from the config alone -- REGLES section 6)
_REQUIRED_KEYS = ("exp_name", "seed", "family", "d_x", "d_z", "d_u")

# model-defining config keys accepted PER FAMILY: a key present in the config
# but not applicable to the declared family is REJECTED at construction --
# silently ignoring it would let distinct run identities build identical
# models (adversarial review round 2)
_FAMILY_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "mlp": frozenset(),
    "transformer": frozenset({"n_heads", "max_len"}),
    "min_ssm": frozenset({"delta_min", "delta_max"}),
    "koopman_ssm": frozenset(
        {"delta", "regime", "constrained", "n_pairs", "delta_min", "delta_max"}
    ),
}
_MODEL_KEYS = frozenset().union(*_FAMILY_CONFIG_KEYS.values())

# capacity knobs are OWNED by match_budget (the fairness contract sizes them
# from target_params): a config naming one directly would be hashed into the
# run identity yet silently ignored by the forwarding above -- reject them
# outright (adversarial review round 3)
_BUDGET_OWNED_KEYS = frozenset(
    {"d_model", "d_ff", "n_layers", "hidden", "d_state", "psi_hidden", "pi_hidden"}
)

# derived or policy-owned constructor names, rejected for every family
# (round 4 -- closes the schema over ALL public predictor-constructor names):
# d_h is derived (d_z for plain families, 2*n_pairs for koopman_ssm); dtype
# is the Trainer's precision policy (CPU float32, declared in the class
# docstring), not a per-run knob. Classification of every constructor name:
#   forwarded: d_z, d_u (required keys) + _FAMILY_CONFIG_KEYS per family
#   budget-owned (rejected): hidden, d_model, d_ff, n_layers, d_state,
#     psi_hidden, pi_hidden
#   derived/policy (rejected): d_h, dtype
_DERIVED_OR_POLICY_KEYS = frozenset({"d_h", "dtype"})

# objective weights: must be finite and >= 0 (a negative weight silently
# behaving as "disabled" was adversarial review round 2's finding), and at
# least one PREDICTION objective must be active or there is nothing to train
_WEIGHT_KEYS = ("w_tf", "w_fr", "w_var", "w_cov")

_HISTORY_DIAGNOSTIC_KEYS = ("jac_margin", "r_ctrl", "lip_pi", "vpt")


def _time_major(split: dict, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Episodes ``idx`` of a ``build_dataset`` split -> time-major float32 tensors.

    ``split["x"]`` is (n_ep, T+1, d_x), ``split["u"]`` is (n_ep, T, d_u), both
    float64 numpy; cast to float32 and transpose to (T[+1], B, d) here, once,
    at the single point where numpy meets torch.
    """
    x = torch.as_tensor(split["x"][idx], dtype=torch.float32).transpose(0, 1).contiguous()
    u = torch.as_tensor(split["u"][idx], dtype=torch.float32).transpose(0, 1).contiguous()
    return x, u


class Trainer:
    """Config-driven JEPA trainer: wires an encoder + one predictor family
    (of the four in ``sld.predictors``) through ONE code path -- teacher-forced
    (TF) and free-running (FR) prediction, plus VICReg anti-collapse.

    Family dispatch is a single line: ``lifted = hasattr(predictor, "lift")``.
    Lifted families (``koopman_ssm``) carry a separate H-space reached by
    ``predictor.lift``/``predictor.project``; plain families operate directly
    in z-space, so lift/project are the identity. ``JEPALoss`` (see
    ``objectives/jepa.py``) then supplies both the projection-space ``pred``
    term (always) and, only when a family is lifted, the lifted-space
    ``consistency`` term.

    TF/FR mix is entirely config-declared (``w_tf``, ``w_fr``, ``h_fr``) and
    every value is printed into ``config.yaml`` -- the factorial study's
    comparability requirement is that no family gets a silently different
    schedule. TF threads ``predictor.step`` over the TRUE prefix (state
    accumulates across ground-truth tokens; the model never feeds its own
    predictions back in -- that is what teacher forcing means, including for
    a transformer's KV cache). FR uses ``predictor.rollout`` from ``h0`` for
    ``h_fr <= T`` steps.

    v0 batching (declared, not a stand-in for something more elaborate
    later in this wave): one sampled batch of ``batch_episodes`` train
    episodes per epoch, no dataloader machinery. Val loss is evaluated on
    the WHOLE val split every epoch under ``torch.no_grad()``.

    BatchNorm-free assumption: encoders in this repo are plain MLPs; the EMA
    target's buffer-copy policy is declared in ``EMATarget`` (jepa.py) and is
    a no-op here. Runs on CPU, float32, by default -- data arrives as
    float64 numpy and is cast to float32 once, at ``_time_major``.
    """

    def __init__(self, config: dict) -> None:
        missing = [k for k in _REQUIRED_KEYS if k not in config]
        if missing:
            raise ValueError(f"config must include {missing}")
        self.config = {**_DEFAULTS, **config}

        family = self.config["family"]
        if family not in _FAMILY_CONFIG_KEYS:
            raise ValueError(
                f"unknown family: {family!r}, expected one of {tuple(_FAMILY_CONFIG_KEYS)}"
            )
        # every model-defining key present must be applicable to the family:
        # applied or rejected, never silently ignored
        inapplicable = sorted((_MODEL_KEYS - _FAMILY_CONFIG_KEYS[family]) & set(config))
        if inapplicable:
            raise ValueError(
                f"config keys {inapplicable} are not applicable to family {family!r}"
            )
        budget_owned = sorted(_BUDGET_OWNED_KEYS & set(config))
        if budget_owned:
            raise ValueError(
                f"config keys {budget_owned} are capacity knobs owned by match_budget -- "
                "set target_params instead (params +/-5% is the fairness contract)"
            )
        derived = sorted(_DERIVED_OR_POLICY_KEYS & set(config))
        if derived:
            raise ValueError(
                f"config keys {derived} are derived or policy-owned (d_h follows from "
                "d_z/n_pairs; precision is the Trainer's declared CPU/float32 policy)"
            )
        # loop integers must be usable BEFORE any artifact is created: an
        # empty epoch loop would still write success-only artifacts and then
        # crash on the empty history (round 4)
        for key in ("epochs", "batch_episodes", "h_fr"):
            v = self.config[key]
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise ValueError(f"{key} must be an integer >= 1, got {v!r}")
        # selective-only knobs on a fixed regime would hash into the identity
        # while constructing identical dynamics (round 3)
        if family == "koopman_ssm" and self.config.get("regime", "bilinear") != "selective":
            selective_only = sorted({"delta_min", "delta_max"} & set(config))
            if selective_only:
                raise ValueError(
                    f"config keys {selective_only} apply only to regime='selective'"
                )
        for key in _WEIGHT_KEYS:
            w = self.config[key]
            if not (isinstance(w, (int, float)) and math.isfinite(w) and w >= 0):
                raise ValueError(f"{key} must be finite and >= 0, got {w!r}")
        if self.config["w_tf"] <= 0 and self.config["w_fr"] <= 0:
            raise ValueError("at least one prediction objective (w_tf or w_fr) must be > 0")

        # canonical config string: sort_keys=True makes it a pure function of
        # the resolved dict's content, not of insertion order; this EXACT
        # string is both what gets hashed and what is written to config.yaml.
        # results_root is EXCLUDED from the hashed identity: it is
        # infrastructure (where artifacts land), not science (what the run
        # is) -- adversarial review round 1.
        hashed = {k: v for k, v in self.config.items() if k != "results_root"}
        self._canonical_yaml = yaml.safe_dump(hashed, sort_keys=True)
        self.config_hash = hashlib.sha1(self._canonical_yaml.encode("utf-8")).hexdigest()[:8]
        # the FULL run identity also covers the dataset's generation spec
        # (adversarial review round 2): run_dir is resolved in fit(), once
        # the data -- and hence its fingerprint -- is known
        self.run_dir: Path | None = None

    def _composite_loss(
        self,
        encoder: nn.Module,
        predictor,
        ema: EMATarget,
        jepa: JEPALoss,
        x_bt: torch.Tensor,
        u_bt: torch.Tensor,
        lifted: bool,
        lift_fn,
        project_fn,
        h_fr: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One composite loss evaluation: TF + FR + VICReg, on one episode batch."""
        cfg = self.config
        T = u_bt.shape[0]
        z = encoder(x_bt)  # (T+1, B, d_z) online
        z_tgt = ema(x_bt)  # (T+1, B, d_z) EMA target, no grad by construction

        # zero-weighted branches are NOT evaluated at all: a diverging FR
        # rollout must not poison a TF-only run through 0 * NaN (adversarial
        # review round 1), and skipping them also skips their compute
        zero = torch.zeros(())

        if cfg["w_tf"] > 0:
            # -- teacher-forced term: step over the TRUE prefix h_true_t = lift(z_t) --
            h_true = lift_fn(z[:T])  # (T, B, d_h)
            state = PredictorState()
            h_hats = []
            for t in range(T):
                h_next, state = predictor.step(h_true[t], u_bt[t], state)
                h_hats.append(h_next)
            h_hat_tf = torch.stack(h_hats, dim=0)  # (T, B, d_h)
            pred_z_tf = project_fn(h_hat_tf)  # (T, B, d_z)
            tf_out = jepa(
                pred_z_tf,
                z_tgt[1:],
                h_hat=h_hat_tf if lifted else None,
                lift=lift_fn if lifted else None,
            )
        else:
            tf_out = {"total": zero, "pred": zero, "consistency": zero}

        if cfg["w_fr"] > 0:
            # -- free-running term: rollout h_fr <= T steps from h0 = lift(z_0) --
            h0 = lift_fn(z[0])  # (B, d_h)
            h_hat_fr = predictor.rollout(h0, u_bt[:h_fr])  # (h_fr, B, d_h)
            pred_z_fr = project_fn(h_hat_fr)  # (h_fr, B, d_z)
            fr_out = jepa(
                pred_z_fr,
                z_tgt[1 : h_fr + 1],
                h_hat=h_hat_fr if lifted else None,
                lift=lift_fn if lifted else None,
            )
        else:
            fr_out = {"total": zero, "pred": zero, "consistency": zero}

        if cfg["w_var"] > 0 or cfg["w_cov"] > 0:
            # -- VICReg anti-collapse on the lifted online embeddings (every arm) --
            zl = lift_fn(z)  # (T+1, B, d_h); the losses flatten leading dims themselves
            var = variance_loss(zl) if cfg["w_var"] > 0 else zero
            cov = covariance_loss(zl) if cfg["w_cov"] > 0 else zero
        else:
            var = cov = zero

        total = (
            cfg["w_tf"] * tf_out["total"]
            + cfg["w_fr"] * fr_out["total"]
            + cfg["w_var"] * var
            + cfg["w_cov"] * cov
        )
        # diagnostic breakdown for metrics.csv: TF+FR pred/consistency summed
        # unweighted, var/cov are the raw VICReg values (weights live in config.yaml)
        diag = {
            "pred": tf_out["pred"].item() + fr_out["pred"].item(),
            "consistency": tf_out["consistency"].item() + fr_out["consistency"].item(),
            "var": var.item(),
            "cov": cov.item(),
        }
        return total, diag

    def fit(self, data: dict) -> dict:
        """Build the models FROM the config, then train on ``data``.

        A run is ``config + seed + dataset spec`` (REGLES section 6): ``fit``
        seeds the global RNG FIRST and only then constructs the encoder
        (``FrameMLPEncoder(d_x, d_z, encoder_hidden)``) and the
        parameter-matched predictor (``get_predictor(family, d_z, d_u,
        target_params, ...)``), so identical config + seed reconstructs
        identical initial weights -- models built by the caller with
        unseeded RNG would break run-identity (adversarial review round 1).
        The run directory is ``<results_root>/<exp_name>/<config_hash>-
        <data_hash>`` where ``data_hash`` fingerprints ``data["meta"]`` (the
        ``build_dataset`` generation spec: system/regime and their params,
        seed, dt, T, episode splits, observation maps) -- scientifically
        different datasets can never share a run identity (adversarial
        review round 2). ``data`` without a ``meta`` block is rejected.
        Declared scope: the fingerprint covers the generation SPEC; arrays
        mutated out-of-band without touching ``meta`` are outside it (the
        replay contract in ``sld.data`` is what ties spec to arrays).
        Encoder scope at v0: FRAME encoder only; the windowed-observation
        data path (sliding windows feeding ``WindowMLPEncoder``) arrives
        with Wave 6 and is NOT wired here.

        Returns the REGLES section 6 public contract:
        ``{"model", "history", "metrics", "paths"}``. Writes artifacts to
        ``self.run_dir``: ``config.yaml``, ``metrics.csv``, ``ckpt.pt``,
        ``audit.json`` (placeholder ``{}``, filled by the Wave 4 audit
        offline), ``figs/`` (empty). Any nonfinite train/val loss ABORTS
        the run with ``RuntimeError`` (finite-metrics law: NaN is a bug,
        never a result).
        """
        cfg = self.config
        # dataset spec enters the run identity BEFORE anything is written
        if "meta" not in data:
            raise ValueError(
                "data must carry its build_dataset 'meta' block: the dataset "
                "generation spec is part of the run identity"
            )
        meta = {k: v for k, v in data["meta"].items() if k != "paths"}  # paths = infra
        data_hash = hashlib.sha1(
            json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:8]
        self.run_dir = (
            Path(cfg["results_root"]) / cfg["exp_name"] / f"{self.config_hash}-{data_hash}"
        )

        set_seed(cfg["seed"])  # seed BEFORE model construction (run identity)

        encoder = FrameMLPEncoder(cfg["d_x"], cfg["d_z"], tuple(cfg["encoder_hidden"]))
        passthrough = {k: cfg[k] for k in _FAMILY_CONFIG_KEYS[cfg["family"]] if k in cfg}
        predictor = get_predictor(
            cfg["family"],
            cfg["d_z"],
            cfg["d_u"],
            target_params=cfg["target_params"],
            **passthrough,
        )

        lifted = hasattr(predictor, "lift")
        lift_fn = predictor.lift if lifted else (lambda z: z)
        project_fn = predictor.project if lifted else (lambda h: h)

        ema = EMATarget(encoder, tau=cfg["tau_ema"])
        jepa = JEPALoss(lambda_c=cfg["lambda_c"])

        params = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

        train_split, val_split = data["train"], data["val"]
        n_train = train_split["x"].shape[0]
        T = train_split["u"].shape[1]
        h_fr = min(cfg["h_fr"], T)
        val_idx = np.arange(val_split["x"].shape[0])

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figs").mkdir(exist_ok=True)
        # success-only artifacts from a PREVIOUS run of this identity are
        # removed up front: an aborted rerun must not leave a stale ckpt/audit
        # beside fresh truncated metrics, masquerading as a complete run
        # (adversarial review round 3)
        for stale in ("ckpt.pt", "audit.json"):
            (self.run_dir / stale).unlink(missing_ok=True)
        config_path = self.run_dir / "config.yaml"
        config_path.write_text(self._canonical_yaml)

        history = {k: [] for k in ("epoch", "train_loss", "val_loss", *_HISTORY_DIAGNOSTIC_KEYS)}
        metrics_path = self.run_dir / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "lr", "pred", "consistency", "var", "cov"]
            )

            for epoch in range(cfg["epochs"]):
                # one sampled train batch per epoch (v0 scale, declared above)
                batch_size = min(cfg["batch_episodes"], n_train)
                idx = np.random.choice(n_train, size=batch_size, replace=False)
                x_bt, u_bt = _time_major(train_split, idx)

                optimizer.zero_grad()
                train_total, diag = self._composite_loss(
                    encoder, predictor, ema, jepa, x_bt, u_bt, lifted, lift_fn, project_fn, h_fr
                )
                train_total.backward()
                lr_used = optimizer.param_groups[0]["lr"]  # rate actually used for this step
                optimizer.step()
                ema.update(encoder)  # EMA target refreshed after EVERY optimizer step
                scheduler.step()

                with torch.no_grad():
                    x_v, u_v = _time_major(val_split, val_idx)
                    val_total, _ = self._composite_loss(
                        encoder, predictor, ema, jepa, x_v, u_v, lifted, lift_fn, project_fn, h_fr
                    )

                train_loss, val_loss = float(train_total.item()), float(val_total.item())
                # finite-metrics law (REGLES section 6): a nonfinite loss is a
                # bug, never a result -- abort loudly instead of writing a
                # completed-looking run (adversarial review round 1)
                if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                    raise RuntimeError(
                        f"nonfinite loss at epoch {epoch}: train={train_loss}, "
                        f"val={val_loss} -- run aborted, artifacts incomplete"
                    )
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                for key in _HISTORY_DIAGNOSTIC_KEYS:
                    history[key].append(None)  # arrives with the Wave 4 audit, never omitted

                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        val_loss,
                        lr_used,
                        diag["pred"],
                        diag["consistency"],
                        diag["var"],
                        diag["cov"],
                    ]
                )

        ckpt_path = self.run_dir / "ckpt.pt"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "ema_target": ema.state_dict(),
                "config": cfg,
            },
            ckpt_path,
        )

        audit_path = self.run_dir / "audit.json"
        # placeholder: filled by the Wave 4 offline audit, never computed here
        audit_path.write_text(json.dumps({}))

        paths = {
            "config": str(config_path.resolve()),
            "metrics_csv": str(metrics_path.resolve()),
            "audit": str(audit_path.resolve()),
            "ckpt": str(ckpt_path.resolve()),
            "figs": str((self.run_dir / "figs").resolve()),
        }
        metrics = {
            "final_train_loss": history["train_loss"][-1],
            "best_train_loss": min(history["train_loss"]),
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
            "seed": int(cfg["seed"]),
            "config_hash": self.config_hash,
        }
        return {
            "model": {"encoder": encoder, "predictor": predictor, "ema_target": ema},
            "history": history,
            "metrics": metrics,
            "paths": paths,
        }
