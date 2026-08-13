"""Shared matplotlib conventions for ``sld`` figures -- the single source of
truth for family colors, constraint linestyles, and save/style settings, so
``fig1_rollout.py``/``fig2_spectrum.py``/future figures never disagree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures regenerate headless from results/, never interactively

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# colorblind-safe (Okabe & Ito 2008 palette), one fixed color per predictor
# family. SEVEN families for six factorial cells: koopman_ssm keeps its color
# because published artifacts still plot under it, even though it is never a
# cell of the 3x2 factorial (PROTOCOL.md section 13.1). The Okabe & Ito
# palette has exactly 8 entries; 7 are used here, yellow (#F0E442) is left
# out as the one with too little contrast on white.
FAMILY_COLORS: dict[str, str] = {
    "mlp": "#0072B2",
    "transformer": "#D55E00",
    "min_ssm": "#009E73",
    "koopman_ssm": "#CC79A7",
    "mlp_k": "#56B4E9",
    "transformer_k": "#E69F00",
    "ssm_k": "#000000",
}

# constraint axis (off = unconstrained cell, on = structurally constrained cell)
CONSTRAINT_LINESTYLES: dict[str, str] = {"off": "--", "on": "-"}


def apply_style() -> None:
    """Sober rcParams shared by every ``sld`` figure: sizes, light grid, dpi."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: Figure, path: str | Path, *, dpi: int | None = None) -> None:
    """Write ``fig`` to ``path``, creating parent directories, tight bbox.

    ``dpi`` overrides ``savefig.dpi`` for this write only. Shipped figures take
    the rcParam; the override exists for cheap low-resolution proofs during
    review, which must never be what lands in ``assets/``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(path, **kwargs)


# --- structure axis (three levels, PROTOCOL.md section 23) -------------------
# The factor the oscillator campaigns vary. Okabe & Ito entries again, chosen
# so the three are separable under protanopia and deuteranopia: worst adjacent
# pair (#009E73 vs #E69F00) is dE 11.4 protan / 24.2 normal vision, measured
# rather than eyeballed. `on-free` (#E69F00) sits below 3:1 contrast on white,
# so every figure using it carries a direct label -- contrast relief is not
# dismissable.
STRUCTURE_COLORS: dict[str, str] = {
    "off": "#0072B2",
    "on-free": "#E69F00",
    "on-stable": "#009E73",
}

# Ink for anything that is not a data mark. Reference lines, thresholds and
# annotations never wear a series color: identity belongs to the marks.
INK_PRIMARY = "#1a1a19"
INK_MUTED = "#6b6b68"
INK_FAINT = "#b3b3ae"


def reference_line(ax, y: float, label: str, *, style: str = ":", color: str = INK_MUTED) -> None:
    """Draw a horizontal reference at ``y``, labelled INSIDE the axes.

    Reference lines are thresholds and analytic anchors (a planted gold, a
    least-squares residual), never measurements -- so they are drawn in ink,
    dashed or dotted, and always labelled. An unlabelled reference line is a
    line a reader has to guess at.

    The label sits inside the axes, left-aligned just above the line, and not
    in the right margin: in a multi-panel figure a right-margin label lands on
    the next panel's y-axis and only the last panel stays readable.
    """
    ax.axhline(y, linestyle=style, linewidth=1.0, color=color, zorder=1)
    ax.annotate(
        label,
        xy=(0.02, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 2),
        textcoords="offset points",
        va="bottom",
        ha="left",
        fontsize=7,
        color=color,
        zorder=5,
    )
