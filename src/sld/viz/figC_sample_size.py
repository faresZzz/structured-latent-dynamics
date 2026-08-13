"""Figure C — the structure axis against sample size, and figure G — the
decomposition (PROTOCOL sections 23.5, 25.3 and 26.0).

Both panels of C show `m_low` per cell: the five seeds as individual points, the
median as a bar, AND an explicit dispersion indicator. The dispersion is the
point, not decoration -- `on-free` at `n = 8` spans 1.18 to 8.57, and a median
alone would report that cell as "1.78" and hide a factor of seven. Range and IQR
are BOTH drawn: an IQR alone would clip exactly the tail that carries the
finding.

Reading rule, from 23.5 and unchanged by anything measured since: the primary
estimand is the SIGN of the interaction and its effect size, never a pass/fail.
These roots have no gate (23.4).

Torch-free (REGLES section 2).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sld.viz.style import INK_FAINT, INK_MUTED, INK_PRIMARY, STRUCTURE_COLORS, apply_style, save_fig

DEFAULT_ROOT = "results_osc_n"
DEFAULT_OUT = "assets/figC_sample_size.png"
DECOMP_ROOT = "results_osc_decomp"
DECOMP_OUT = "assets/figG_decomposition.png"

_ARM_ORDER = ("off", "on-free", "on-stable", "on-stable-nolift")
_NOLIFT_COLOR = "#CC79A7"  # Okabe & Ito; NOT a structure level, a fourth arm


def _arm_of(cfg: dict) -> str:
    if cfg["family"] == "mlp":
        return "off"
    if cfg.get("lift") == "identity":
        return "on-stable-nolift"
    return "on-stable" if cfg.get("radius") == "bounded" else "on-free"


def load_cells(root: str | Path) -> dict[tuple[str, str], list[tuple[int, float | None]]]:
    """``{(dataset, arm): [(seed, m_low or None), ...]}`` -- a diverged cell is
    KEPT with ``None`` so a figure cannot silently drop it (15.3)."""
    root = Path(root)
    exp = next(p for p in root.iterdir() if p.is_dir())
    idx = {r["run_dir"].split("/")[-1]: r["regime"] for r in csv.DictReader((exp / "index.csv").open())}
    cells: dict[tuple[str, str], list[tuple[int, float | None]]] = {}
    for d in sorted(exp.iterdir()):
        if not d.is_dir() or not (d / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        value = None
        if (d / "gate.json").exists() and not (d / "failure.json").exists():
            value = json.loads((d / "gate.json").read_text())["measures"]["m_low"]
        cells.setdefault((idx.get(d.name, "?"), _arm_of(cfg)), []).append((cfg["seed"], value))
    return cells


def _draw_cell(ax, x: float, values: list[float], color: str) -> None:
    """Five seeds, their median, and BOTH dispersion measures."""
    v = np.sort(np.asarray(values))
    lo, hi = v[0], v[-1]
    q1, q3 = np.percentile(v, [25, 75])
    ax.vlines(x, lo, hi, color=color, linewidth=1.0, alpha=0.55, zorder=2)          # range
    ax.vlines(x, q1, q3, color=color, linewidth=4.5, alpha=0.28, zorder=2)          # IQR
    jitter = np.linspace(-0.055, 0.055, len(v))
    ax.scatter(x + jitter, v, s=26, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    ax.hlines(np.median(v), x - 0.075, x + 0.075, color=INK_PRIMARY, linewidth=1.8, zorder=5)


def _panel(ax, cells, datasets, arms, xlabels, colors, title, xlabel):
    for slot, ds in enumerate(datasets):
        for k, arm in enumerate(arms):
            group = cells.get((ds, arm))
            if not group:
                continue
            vals = [v for _, v in sorted(group) if v is not None]
            n_div = sum(1 for _, v in group if v is None)
            x = slot + (k - (len(arms) - 1) / 2) * 0.24
            if vals:
                _draw_cell(ax, x, vals, colors[arm])
            if n_div:
                ax.annotate(f"{n_div} div", xy=(x, max(vals) if vals else 1.0), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=6.5, color=INK_MUTED)
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color=INK_MUTED, zorder=1)
    ax.annotate("m_low = 1: above this line, neither trivial baseline is beaten",
                xy=(0.012, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 4), ha="left",
                textcoords="offset points", fontsize=6.8, color=INK_MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$m_{low}$   (lower is better)")
    ax.set_title(title, loc="left")
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 2.2, hi * 1.5)   # place pour l'annotation et les etiquettes "div"


_CONFOUNDS = (
    "CONFOUNDS, both declared before the runs.  (1) The structure axis moves a COMPOSITE: the "
    "transition, psi and pi together (13.7) -- no cell isolates the transition alone, which is "
    "what results_osc_decomp/ was declared to address.  (2) The 23.3 sweep's argmin was lr=1e-3 "
    "for all three arms, i.e. at the EDGE of the declared list {1e-4, 3e-4, 1e-3} with the "
    "criterion still decreasing there; 16.2(4) forbids an off-list rate, so all three arms may "
    "be under-tuned in the SAME direction and the comparison inherits that.  "
    "No gate on this root (23.4): the median displays, it decides nothing, and every seed is drawn."
)


def make_figC(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT) -> Path:
    apply_style()
    cells = load_cells(root)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    arms = ("off", "on-free", "on-stable")
    _panel(ax, cells, ["n8", "n32", "n256"], arms, ["8", "32", "256"], STRUCTURE_COLORS,
           "Structure against sample size, prior exact", "n_episodes (log spacing)")
    ax.legend(handles=[plt.Line2D([], [], color=STRUCTURE_COLORS[a], marker="o", linestyle="none",
                                  markersize=5, label=a) for a in arms] +
                      [plt.Line2D([], [], color=INK_PRIMARY, linewidth=1.7, label="median"),
                       plt.Line2D([], [], color=INK_MUTED, linewidth=4.5, alpha=0.28, label="IQR"),
                       plt.Line2D([], [], color=INK_MUTED, linewidth=1.0, alpha=0.55, label="range")],
              fontsize=7.5, loc="lower left", framealpha=0.95)
    fig.text(0.0, -0.07, "PROTOCOL 23.5. Root: results_osc_n/, 45 cells, CoupledOscillators c=0.10, "
             "8000 updates, h_fr=25.    Scope (25.6): measured UNDER THE DECLARED REGIME (h_fr = 25); the sign of the structure effect is unchanged at h_fr = 5 (25.8)." + _CONFOUNDS, fontsize=6.6, color=INK_MUTED, va="top", wrap=True)
    out = Path(out)
    save_fig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def make_figG(root: str | Path = DECOMP_ROOT, out: str | Path = DECOMP_OUT) -> Path:
    """The decomposition, with the number that explains it."""
    from sld.systems.coupled_oscillators import CoupledOscillators
    from sld.systems.representability import real_modal_distance, representability_distance

    apply_style()
    cells = load_cells(root)
    colors = {**STRUCTURE_COLORS, "on-stable-nolift": _NOLIFT_COLOR}
    arms = ("off", "on-stable", "on-stable-nolift")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _panel(ax, cells, ["n8", "n256"], arms, ["8", "256"], colors,
           "Removing the lift pair makes it worse, not better", "n_episodes")

    A = CoupledOscillators(c=0.10).A_mat
    identity_basis, _ = representability_distance(A)
    best_basis = real_modal_distance(A)
    ax.annotate(
        "Why nolift is worse, and it is not the optimizer:\n"
        f"A is {identity_basis:.3f} from the rotation-scaling class in the IDENTITY basis\n"
        f"({100 * identity_basis / np.linalg.norm(A):.0f}% of $\\|A\\|_F$), and {best_basis:.0e} in the best basis.\n"
        "psi/pi ARE that change of basis (26.0).",
        xy=(0.015, 0.03), xycoords="axes fraction", ha="left", va="bottom", fontsize=7.2,
        color=INK_PRIMARY,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": INK_FAINT, "linewidth": 0.8},
    )
    ax.legend(handles=[plt.Line2D([], [], color=colors[a], marker="o", linestyle="none",
                                  markersize=5, label=a) for a in arms],
              fontsize=7.5, loc="upper right", framealpha=0.95)
    fig.text(0.0, -0.07,
             "PROTOCOL 25.3 / 26.0. Root: results_osc_decomp/, 30 cells, h_fr=25, 0 divergences.  "
             "25.3's reading rule placed nolift BETWEEN off and on-stable; it is above both, so none of "
             "its three branches applies and the result is published outside the partition.  CONFOUNDS, "
             "both declared before the runs: nolift carries 15.7% fewer parameters by construction (23.2 "
             "freezes the backbone rather than matching the budget, so the arms differ only by the pair), "
             "and it inherits on-stable's lr because 23.5(2) forbids a second sweep.  No gate (23.4).",
             fontsize=6.6, color=INK_MUTED, va="top", wrap=True)
    out = Path(out)
    save_fig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=("C", "G", "both"), default="both")
    args = ap.parse_args()
    if args.which in ("C", "both"):
        make_figC()
    if args.which in ("G", "both"):
        make_figG()


if __name__ == "__main__":
    main()
