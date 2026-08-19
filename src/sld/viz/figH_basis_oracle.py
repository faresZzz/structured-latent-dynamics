"""Fig H -- the basis-oracle root: is the penalty the structured class
itself, or the joint discovery of basis and operator.

PROTOCOL.md section 28 removes exactly one thing from the structured `mlp_k`
arm -- the requirement to discover the true basis by gradient descent -- and
changes nothing else. One backbone (`mlp`), three arms, five seeds each,
fifteen `gate.json` under `results_basis/`:

- `off`         -- unstructured, PROTOCOL.md 27's `mlp` column;
- `on-learned`  -- structured, psi/pi discovered jointly with the operator,
  exactly 27's `mlp_k`;
- `on-oracle`   -- structured, psi/pi FROZEN at the true basis (`T`, section
  28.2), zero trainable lift parameters, built from the true generator no
  other arm in this repository is allowed to see.

`gate.json`'s own identity (`backbone`, `structure`, `family`) cannot tell
`on-learned` from `on-oracle` apart -- both are `(mlp, on, mlp_k)`. The two
differ only in the run's `config.yaml`: `on-oracle` is the one written with
`lift: frozen` (section 28.2's table); `on-learned` carries no `lift` key at
all. This module reads that field alongside `gate.json` for exactly that
reason (see `_arm_label`).

**`on-oracle` is an ORACLE arm, not a competitor to `off`** (section 28.2):
it measures a ceiling, what the structured class achieves when the basis is
given rather than discovered. Every reading of P3 below carries that word.

**The verdict this figure exists to draw (section 28.7).** P1 replicates
27.8.3's `mlp` column (control, `supported`; if it had not replicated, P2 and
P3 would not be read at all, 28.4). P2 (`on-oracle` < `on-learned`) is
`supported`, 0+/5-: handing over the true basis removes most of the
learned-basis penalty and restores the positive-control gate (`on-oracle`
5/5 `learns` against `on-learned` 0/5). P3 (`on-oracle` < `off`) is
`indeterminate`, 3+/2-: it does not reverse the sign against the
unstructured arm. Per 28.4, the report is **"the coupling is a contributing
mechanism; it is not sufficient to reverse the sign"** -- never "structure
works when you fix the basis".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sld.viz.ingest import load_gate
from sld.viz.style import (
    FAMILY_COLORS,
    INK_FAINT,
    INK_MUTED,
    INK_PRIMARY,
    apply_style,
    reference_line,
    save_fig,
    write_sidecar_csv,
)

DEFAULT_ROOT = "results_basis"
DEFAULT_OUT = "assets/figH_basis_oracle.png"

ARMS: tuple[str, str, str] = ("off", "on-learned", "on-oracle")

# Okabe & Ito (2008) swatches, the same ones ``FAMILY_COLORS`` uses elsewhere
# -- reused here for a DIFFERENT axis (arm, not family), because this root is
# ``mlp``-only (28.2): ``off`` literally IS family ``mlp`` and ``on-learned``
# literally IS family ``mlp_k``, so their usual colors still mean what they
# always mean. ``on-oracle`` is ALSO family ``mlp_k`` (same backbone, same
# structure -- only ``config.yaml``'s ``lift`` differs, see ``_arm_label``)
# and therefore needs a third, otherwise-unclaimed hue; ``koopman_ssm``'s
# slot is free because that family never appears in this root.
ARM_COLORS: dict[str, str] = {
    "off": FAMILY_COLORS["mlp"],
    "on-learned": FAMILY_COLORS["mlp_k"],
    "on-oracle": FAMILY_COLORS["koopman_ssm"],
}

# PROTOCOL.md section 28.3 -- the three predictions, written before the
# numbers existed. ``sign`` is the predicted sign of ``minuend - subtrahend``.
PREDICTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("P1", "on-learned", "off", ">0"),
    ("P2", "on-oracle", "on-learned", "<0"),
    ("P3", "on-oracle", "off", "<0"),
)

_CSV_FIELDS = ["arm", "seed", "e_h_gate", "verdict"]


def _lift_of(root: Path, run_dir: str) -> str | None:
    """The ``lift`` field of one run's ``config.yaml`` -- ``None`` when the
    key is absent (``off`` and ``on-learned`` cells never write one; only
    ``on-oracle`` does, PROTOCOL.md section 28.2's table)."""
    cfg_path = root / run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path}: run has no config.yaml (needed to tell on-learned from on-oracle apart)"
        )
    return yaml.safe_load(cfg_path.read_text()).get("lift")


def _arm_label(root: Path, structure: str, run_dir: str) -> str:
    """``off`` / ``on-learned`` / ``on-oracle``. ``structure`` alone (as
    ``gate.json``'s identity carries it) cannot separate the last two: both
    are ``(backbone=mlp, structure=on, family=mlp_k)``. They differ only in
    ``config.yaml``'s ``lift`` field -- ``"frozen"`` for ``on-oracle`` (psi =
    the true basis, no gradient), absent for ``on-learned`` (psi discovered
    jointly with the operator, exactly 27's ``mlp_k``)."""
    if structure == "off":
        return "off"
    return "on-oracle" if _lift_of(root, run_dir) == "frozen" else "on-learned"


def _load_arms(root: str | Path) -> pd.DataFrame:
    """The 15 ``gate.json`` rows under ``root``, with the ``arm`` column
    ``load_gate`` alone cannot produce (see ``_arm_label``), sorted into a
    deterministic ``(arm, seed)`` order for a reproducible plot and CSV."""
    root = Path(root)
    scalars, _ = load_gate(root)
    scalars = scalars.copy()
    scalars["arm"] = [
        _arm_label(root, structure, run_dir)
        for structure, run_dir in zip(scalars["structure"], scalars["run_dir"], strict=True)
    ]
    order = {arm: i for i, arm in enumerate(ARMS)}
    return scalars.sort_values(
        by=["arm", "seed"], key=lambda col: col.map(order) if col.name == "arm" else col
    ).reset_index(drop=True)


def _contrast(df: pd.DataFrame, minuend: str, subtrahend: str) -> tuple[np.ndarray, float, float]:
    """Paired, within-seed ``minuend - subtrahend`` (PROTOCOL.md 28.3: same
    seed, same `data_hash`, same budget -- never file order or group mean).
    Returns ``(per-seed diffs sorted by seed, mean, SE = std(ddof=1)/sqrt(n))``.
    """
    piv = df.pivot(index="seed", columns="arm", values="e_h_gate").sort_index()
    diffs = (piv[minuend] - piv[subtrahend]).to_numpy()
    return diffs, float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def _verdict(mean: float, se: float, predicted_sign: str) -> str:
    """PROTOCOL.md section 28.3: ``supported`` when the paired contrast has
    the predicted strict sign and its +-1 SE interval excludes zero;
    ``contradicted`` on the opposite strict sign under the same condition;
    ``indeterminate`` otherwise. No p-value, ever (28.3)."""
    lo, hi = mean - se, mean + se
    if predicted_sign == ">0":
        return "supported" if lo > 0 else "contradicted" if hi < 0 else "indeterminate"
    return "supported" if hi < 0 else "contradicted" if lo > 0 else "indeterminate"


def _plot_by_arm(ax, df: pd.DataFrame) -> None:
    """Panel 1: the 5 seed points per arm, mean +-1 SE, and thin lines
    pairing the same seed across the three arms (28.3's pairing rule, made
    visible rather than only stated)."""
    xpos = {arm: i for i, arm in enumerate(ARMS)}
    piv = df.pivot(index="seed", columns="arm", values="e_h_gate").sort_index()[list(ARMS)]
    for _, row in piv.iterrows():
        ax.plot([xpos[a] for a in ARMS], row.to_numpy(), color=INK_FAINT, linewidth=0.8, zorder=1)

    xticklabels = []
    for arm in ARMS:
        sub = df[df["arm"] == arm]
        jitter = np.linspace(-0.09, 0.09, len(sub))
        ax.scatter(
            xpos[arm] + jitter, sub["e_h_gate"], s=40, color=ARM_COLORS[arm],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        mean = float(sub["e_h_gate"].mean())
        se = float(sub["e_h_gate"].std(ddof=1) / np.sqrt(len(sub)))
        ax.plot([xpos[arm], xpos[arm]], [mean - se, mean + se], color=ARM_COLORS[arm],
                linewidth=2.0, solid_capstyle="round", zorder=4)
        ax.plot([xpos[arm] - 0.16, xpos[arm] + 0.16], [mean, mean], color=ARM_COLORS[arm],
                linewidth=2.6, solid_capstyle="round", zorder=5)
        n_learns = int((sub["status"] == "learns").sum())
        xticklabels.append(f"{arm}\n{n_learns}/5 learns")

    # headroom for the annotation below -- the default 5% autoscale margin
    # is not enough clearance above the highest point (on-learned's spread
    # reaches close to the axes top on its own)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    ax.annotate(
        "oracle: psi = true basis, frozen -- a ceiling, not a competitor (28.2)",
        xy=(xpos["on-oracle"], 0.98), xycoords=("data", "axes fraction"),
        ha="center", va="top", fontsize=6.6, color=INK_MUTED, style="italic", zorder=6,
    )
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(xticklabels, color=INK_PRIMARY)
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_ylabel(r"$e_H$ at $H=25$ (true-state coordinates, lower is better)", color=INK_MUTED)
    ax.set_title("Per-seed error by arm -- thin lines pair the same seed", color=INK_PRIMARY)


def _plot_contrasts(ax, df: pd.DataFrame) -> dict[str, dict]:
    """Panel 2: the three paired contrasts of 28.3, drawn as mean +-1 SE
    over their own 5 per-seed differences. Returns the per-prediction
    statistics so the caller can cite the SAME numbers in the caption
    instead of retyping them."""
    reference_line(ax, 0.0, "zero -- no effect", style="--")
    results: dict[str, dict] = {}
    xticklabels = []
    bounds: list[float] = []
    for i, (name, a, b, sign) in enumerate(PREDICTIONS):
        diffs, mean, se = _contrast(df, a, b)
        verdict = _verdict(mean, se, sign)
        n_pos, n_neg = int((diffs > 0).sum()), int((diffs < 0).sum())
        results[name] = {
            "a": a, "b": b, "mean": mean, "se": se,
            "verdict": verdict, "n_pos": n_pos, "n_neg": n_neg,
        }
        bounds += [float(diffs.min()), float(diffs.max()), mean - se, mean + se]

        jitter = np.linspace(-0.09, 0.09, len(diffs))
        ax.scatter(np.full(len(diffs), i) + jitter, diffs, s=34, color=INK_PRIMARY,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot([i, i], [mean - se, mean + se], color=INK_PRIMARY, linewidth=2.0,
                solid_capstyle="round", zorder=4)
        ax.plot([i - 0.16, i + 0.16], [mean, mean], color=INK_PRIMARY, linewidth=2.6,
                solid_capstyle="round", zorder=5)
        anchor = max(float(diffs.max()), mean + se)
        ax.annotate(
            f"{mean:+.4f} $\\pm$ {se:.4f}\n{n_pos}+/{n_neg}-, {verdict}",
            xy=(i, anchor), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=6.8, color=INK_PRIMARY,
        )
        xticklabels.append(f"{name}\n{a} - {b}")

    ax.set_xticks(range(len(PREDICTIONS)))
    ax.set_xticklabels(xticklabels, fontsize=8, color=INK_PRIMARY)
    ax.set_xlim(-0.5, len(PREDICTIONS) - 0.5)
    lo, hi = min(bounds), max(bounds)
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 3.2 * pad)
    ax.set_ylabel(r"paired $\Delta e_H$ (mean $\pm$ 1 SE, same seed)", color=INK_MUTED)
    ax.set_title("The three predictions (section 28.3)", color=INK_PRIMARY)
    return results


def make_figH(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT, *, dpi: int | None = None) -> Path:
    """Render Fig H from the 15 ``gate.json`` artifacts under ``root`` --
    PROTOCOL.md section 28's basis-oracle root. Also writes the sidecar CSV
    (``arm, seed, e_h_gate, verdict``) and its ``.sha256`` beside the PNG
    (section 27.4's doctrine, same helper fig1-3 use)."""
    import matplotlib.pyplot as plt

    root = Path(root)
    df = _load_arms(root)
    apply_style()

    fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    _plot_by_arm(ax_e, df)
    contrasts = _plot_contrasts(ax_c, df)
    p1, p2, p3 = contrasts["P1"], contrasts["P2"], contrasts["P3"]

    # SHORT, like figA/figB's suptitles -- the detailed P1/P2/P3 numbers are
    # already on panel 2 and in the caption below. A suptitle long enough to
    # need wrapping used to blow up ``bbox_inches="tight"``'s canvas (one
    # unwrapped line far wider than the axes); ``wrap=True`` is kept too, as
    # a second guard against the same failure re-appearing.
    fig.suptitle(
        "Basis root (PROTOCOL.md 28), CoupledOscillators, mlp only -- on-oracle is an ORACLE arm "
        "(psi = the true basis, frozen): a ceiling, not a competitor to off. The coupling is a "
        f"contributing mechanism; it is not sufficient to reverse the sign (section 28.7).  n = {len(df)} "
        "runs, 5 seeds x 3 arms.",
        fontsize=9.5, color=INK_PRIMARY, y=1.05, wrap=True,
    )
    fig.text(
        0.5, -0.12,
        "Primary statistic: e_H at H=25 (28.3), mean over 5 seeds, SE = std(ddof=1)/sqrt(5); every "
        "contrast paired within seed (same seed, same data_hash, same budget -- never file order or group "
        f"mean). P1 (on-learned - off, {p1['mean']:+.4f} $\\pm$ {p1['se']:.4f}, {p1['n_pos']}+/"
        f"{p1['n_neg']}-, {p1['verdict']}) replicates 27.8.3's mlp column to the printed precision -- "
        "determinism, not an independent draw, read as the control 28.3 declares it: had it failed to "
        "replicate, P2 and P3 would not be read at all (28.4). Giving the structured arm the true basis "
        f"(P2, on-oracle - on-learned, {p2['mean']:+.4f} $\\pm$ {p2['se']:.4f}, {p2['n_pos']}+/"
        f"{p2['n_neg']}-, {p2['verdict']}) removes most of the learned-basis penalty on almost every seed "
        "and restores the positive-control gate, but does not put the structured class ahead of the "
        f"unstructured one (P3, on-oracle - off, {p3['mean']:+.4f} $\\pm$ {p3['se']:.4f}, {p3['n_pos']}+/"
        f"{p3['n_neg']}-, {p3['verdict']}). on-oracle inherits on-learned's width and psi_hidden exactly "
        "and trades the learned psi/pi pair (1574 trainable parameters, 28.2) for a frozen linear map "
        "built from the true generator that no other arm in this repository has seen. Scope: mlp backbone "
        "only, CoupledOscillators/OURegime(c=0.10) only -- does not touch DampedPendulum, transformer, "
        "ssm, or any threshold (28.5); 27.8.1, 27.8.3 and 27.9 stand exactly as written whatever this "
        "root returns (28.4).",
        ha="center", fontsize=7.2, color=INK_MUTED, wrap=True,
    )

    out = Path(out)
    save_fig(fig, out, dpi=dpi)
    plt.close(fig)

    rows = df[["arm", "seed", "e_h_gate", "status"]].rename(columns={"status": "verdict"}).to_dict("records")
    write_sidecar_csv(rows, _CSV_FIELDS, out.with_suffix(".csv"))
    return out


def main() -> None:
    path = make_figH()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
