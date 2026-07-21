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

# colorblind-safe (Okabe & Ito 2008 palette), one fixed color per predictor family
FAMILY_COLORS: dict[str, str] = {
    "mlp": "#0072B2",
    "transformer": "#D55E00",
    "min_ssm": "#009E73",
    "koopman_ssm": "#CC79A7",
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


def save_fig(fig: Figure, path: str | Path) -> None:
    """Write ``fig`` to ``path``, creating parent directories, tight bbox."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
