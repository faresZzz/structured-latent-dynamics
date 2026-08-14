"""Fig. 2 -- VPT, the five raw per-seed values per cell, PROTOCOL.md section
27.4/27.7.

Every one of the five seeds is drawn, per cell, always -- never dropped,
never averaged. A ``crossed`` value (the run's error crossed ``tau=0.5``
within the scored horizon) is a filled marker at its true height. A
``censored`` value (``vpt_median`` is ``None`` or non-finite -- 27.3: no run
at all, or the error never crossed ``tau``) is drawn as an OPEN marker placed
at a labelled ceiling line, never at zero and never at ``Hmax + 1`` (27.3's
own two forbidden replacements): the ceiling is a plot placement, not a
value, and the reference line under it says so.

Section 27.7 is why this figure exists but carries no order verdict: on the
162 already-scored runs across the six prior roots, 87.6% of episodes are
censored -- VPT is censored BY CONSTRUCTION on this system, so a VPT-primary
reading would return 'indeterminate' on very nearly every one of the nine
relations. ``e_h_gate`` (Fig. 1/3) is the ordering statistic; VPT is
published here for the pre-registered record, with its censoring shown in
full, not because it decides anything.

Torch-free (REGLES section 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sld.viz.style import (
    FAMILY_COLORS,
    INK_MUTED,
    INK_PRIMARY,
    apply_style,
    reference_line,
    save_fig,
    sigma_train_caption_fragment,
    write_sidecar_csv,
)

_FAMILY_ORDER: tuple[str, ...] = ("mlp", "mlp_k", "transformer", "transformer_k", "min_ssm", "ssm_k")

_CSV_FIELDS = ["family", "backbone", "structure", "seed", "run_status", "vpt_median", "vpt_status"]


def _sidecar_rows(rows: list[dict], vpt: dict) -> list[dict]:
    out = []
    by_key = {(r["family"], r["seed"]): r for r in rows}
    for family in _FAMILY_ORDER:
        for seed, entry in sorted(vpt[family]["per_seed"].items()):
            r = by_key[(family, seed)]
            # censored (27.3) is never written as a value -- not the literal
            # None/inf repr, not 0, not Hmax + 1 -- blank, exactly like a
            # cell that never ran at all.
            value = entry["vpt_median"]
            out.append({
                "family": family, "backbone": r["backbone"], "structure": r["structure"], "seed": seed,
                "run_status": r["status"], "vpt_median": "" if entry["status"] == "censored" else value,
                "vpt_status": entry["status"],
            })
    return out


def make_fig2(
    results_root: str = "results_primary_factorial",
    exp_name: str = "primary_factorial",
    out_path: str = "assets/fig2_vpt.png",
    sidecar_path: str | None = None,
    state_scale_path: str | None = None,
) -> dict:
    """PROTOCOL.md section 27.4, Fig. 2: five raw per-seed VPT values per
    cell with their statuses, censored values visibly marked and counted --
    never zero, never ``Hmax + 1``, never omitted (27.3).

    Returns a dict with the plotted data (``vpt``, ``sld.viz.results``'s own
    ``summarize_vpt`` output), the image path, and the sidecar CSV path +
    hash.
    """
    from sld.viz import results as results_mod

    rows = results_mod.build_primary_results_table(results_root, exp_name)
    if all(r["status"] == "missing" for r in rows):
        raise FileNotFoundError(
            f"fig2: no gate.json under {Path(results_root) / exp_name} -- nothing to plot yet"
        )
    vpt = results_mod.summarize_vpt(rows)
    by_key = {(r["family"], r["seed"]): r for r in rows}

    apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    finite = [
        entry["vpt_median"] for family in _FAMILY_ORDER
        for entry in vpt[family]["per_seed"].values()
        if entry["status"] == "crossed"
    ]
    ceiling = max(finite) * 1.15 if finite else 30.0

    for i, family in enumerate(_FAMILY_ORDER):
        per_seed = vpt[family]["per_seed"]
        n = len(per_seed)
        jitter = np.linspace(-0.14, 0.14, n) if n > 1 else np.zeros(1)
        for j, seed in enumerate(sorted(per_seed)):
            entry = per_seed[seed]
            run_status = by_key[(family, seed)]["status"]
            x = i + jitter[j]
            color = FAMILY_COLORS[family]
            if entry["status"] == "crossed":
                ax.scatter(x, entry["vpt_median"], s=32, color=color, edgecolor="white",
                           linewidth=0.6, zorder=4, marker="o")
            else:
                ax.scatter(x, ceiling, s=42, facecolor="none", edgecolor=color, linewidth=1.4,
                           zorder=4, marker="^")
            if run_status != "ok":
                ax.annotate(run_status, xy=(x, entry["vpt_median"] if entry["status"] == "crossed" else ceiling),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.6, color=INK_MUTED)
        n_censored = vpt[family]["n_censored"]
        ax.annotate(f"{n_censored}/{n} censored", xy=(i, -0.06), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6.8, color=INK_MUTED)

    reference_line(ax, ceiling, "censored -- placement only, NOT a value (never 0, never Hmax+1)", style=":")
    ax.set_xticks(range(len(_FAMILY_ORDER)))
    ax.set_xticklabels(_FAMILY_ORDER)
    ax.set_xlim(-0.5, len(_FAMILY_ORDER) - 0.5)
    ax.set_ylim(0, ceiling * 1.12)
    ax.set_xlabel("family (backbone x structure)")
    ax.set_ylabel(r"VPT   ($\tau=0.5$, higher is better)   -- open marker = censored")
    ax.set_title("VPT, five raw per-seed values -- published, not the ordering statistic (27.7)", loc="left")
    ax.legend(handles=[
        plt.Line2D([], [], color=INK_PRIMARY, marker="o", linestyle="none", markersize=5.5, label="crossed"),
        plt.Line2D([], [], markerfacecolor="none", markeredgecolor=INK_PRIMARY, marker="^", linestyle="none",
                   markersize=6, label="censored"),
    ], fontsize=7.5, loc="upper right", framealpha=0.95)

    sigma_clause = sigma_train_caption_fragment(
        state_scale_path, rows[0]["stratum_system"], rows[0]["stratum_regime"]
    )
    caption = (
        "PROTOCOL 27.1/27.4/27.7. System: CoupledOscillators. Regime: OURegime(theta_ou=1.0). H=25. "
        f"{sigma_clause} (sld.audit.scan_state_scale, identical across all six cells (27.1)). "
        "tau=0.5. Censoring convention: vpt_median is None or "
        "non-finite -> 'censored' (27.3); censored points are plotted at a labelled ceiling as PLACEMENT "
        "ONLY, never as a value, never 0, never Hmax+1. n=5 seeds per cell. Non-eligible cells: none hidden "
        "here -- unlike Fig. 1, every seed of every cell is drawn (27.4). Why VPT is published without "
        "carrying the order verdicts (27.7): across the 162 already-scored runs of the six prior roots, "
        "87.6% of evaluation episodes are censored -- VPT is censored BY CONSTRUCTION on this system, so a "
        "VPT-primary reading would return 'indeterminate' on very nearly every one of the nine relations; "
        "e_h_gate (Fig. 1/3) is the ordering statistic instead."
    )
    fig.text(0.0, -0.13, caption, fontsize=6.6, color=INK_MUTED, va="top", wrap=True)

    out_path = Path(out_path)
    save_fig(fig, out_path)
    plt.close(fig)

    sidecar_path = Path(sidecar_path) if sidecar_path else out_path.with_suffix(".csv")
    digest = write_sidecar_csv(_sidecar_rows(rows, vpt), _CSV_FIELDS, sidecar_path)

    print(f"wrote {out_path}")
    return {
        "image_path": out_path,
        "sidecar_csv": sidecar_path,
        "sidecar_hash": digest,
        "data": vpt,
        "caption": caption,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig. 2 -- PROTOCOL.md section 27.4")
    ap.add_argument("--results-root", default="results_primary_factorial")
    ap.add_argument("--exp-name", default="primary_factorial")
    ap.add_argument("--out", default="assets/fig2_vpt.png")
    ap.add_argument("--state-scale", default=None, help="path to a state_scale/1 JSON (sld.audit.scan_state_scale)")
    args = ap.parse_args()
    try:
        make_fig2(
            results_root=args.results_root, exp_name=args.exp_name, out_path=args.out,
            state_scale_path=args.state_scale,
        )
    except FileNotFoundError as exc:
        print(f"fig2: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
