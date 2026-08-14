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


def sigma_train_caption_fragment(
    state_scale_path: str | Path | None, stratum_system: str, stratum_regime: str
) -> str:
    """The ``sigma_train`` clause a caption cites, read from
    ``sld.audit.scan_state_scale``'s ``state_scale/1`` JSON.

    The schema is re-checked here with plain ``json`` rather than by
    importing ``sld.audit.scan_state_scale.load_state_scale``: that module's
    OWN top-level imports are torch-free, but it lives inside the ``sld.audit``
    PACKAGE, whose ``__init__.py`` imports ``jacobian``/``rollout`` and pulls
    in ``torch`` as a side effect of the package import alone -- exactly what
    ``sld/viz`` must never do (REGLES section 2). Same doctrine as
    ``figF_action_conditioning.py``, which reads ``action_jacobian.json``
    with ``json.loads`` directly rather than importing
    ``sld.audit.scan_action_jacobian``.

    Three outcomes, never a silent blank: ``state_scale_path`` is ``None``
    -> the caption says explicitly that ``sigma_train`` was not read (never
    a fabricated number, never an empty field); the path is given and a
    stratum matching ``(stratum_system, stratum_regime)`` exists -> the
    formatted values and the cross-checkable ``e_recon``; the path is given
    but no matching stratum exists -> raise, because the file WAS supplied
    and did not have what the caption needs -- that is a real defect, not a
    "not read yet" state.
    """
    import json

    if state_scale_path is None:
        return "sigma_train: NOT READ (no state_scale.json path given -- pass state_scale_path=... to cite it)"
    payload = json.loads(Path(state_scale_path).read_text())
    if payload.get("schema") != "state_scale/1":
        raise ValueError(f"{state_scale_path}: expected schema 'state_scale/1', got {payload.get('schema')!r}")
    for block in payload["strata"].values():
        if block["system"] == stratum_system and block["regime"] == stratum_regime:
            formatted = ", ".join(f"{v:.4f}" for v in block["sigma_train"])
            return f"sigma_train=[{formatted}] (state_scale/1, e_recon={block['e_recon']:.3e})"
    raise ValueError(
        f"{state_scale_path}: no stratum matching system={stratum_system!r} regime={stratum_regime!r}"
    )


def write_sidecar_csv(rows: list[dict], fieldnames: list[str], out_csv: str | Path) -> str:
    """Write ``rows`` to ``out_csv`` and a ``.sha256`` digest beside it.

    PROTOCOL.md section 27.4: "each figure writes a sidecar CSV and a hash".
    One shared writer so fig1/fig2/fig3 agree on the mechanics rather than
    each reinventing csv+hashlib -- returns the hex digest, which the caller
    also gets back from the file it writes, so a test can check either.
    """
    import csv
    import hashlib

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    out_csv.with_suffix(out_csv.suffix + ".sha256").write_text(digest + "\n")
    return digest


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
