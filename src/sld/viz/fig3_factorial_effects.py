"""Fig. 3 -- the three paired ON-OFF contrasts, PROTOCOL.md section 27.4.

``mlp_k - mlp``, ``transformer_k - transformer``, ``ssm_k - min_ssm``: each
pair shares its backbone, seed, dataset hash and representation (27.3's
pairing rule, ``results.paired_structure_contrasts``), so this is the ONE
comparison on this grid free of the cross-backbone rate confound named in
27.2 -- off and on of the SAME backbone train at the SAME rate. The statistic
is ``e_h_gate`` (27.7), lower is better: a NEGATIVE contrast means the
structured arm is better. The zero line is therefore the reading key, drawn
with ``style.reference_line`` and labelled so the sign is not left for the
reader to work out.

Torch-free (REGLES section 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sld.viz.style import (
    INK_MUTED,
    INK_PRIMARY,
    STRUCTURE_COLORS,
    apply_style,
    reference_line,
    save_fig,
    sigma_train_caption_fragment,
    write_sidecar_csv,
)

# section 27.1's three backbones, in the order 27.4 lists the contrasts.
_BACKBONES: tuple[str, ...] = ("mlp", "transformer", "ssm")
_CONTRAST_LABEL = {"mlp": "mlp_k - mlp", "transformer": "transformer_k - transformer", "ssm": "ssm_k - min_ssm"}
_COLOR = {"mlp": STRUCTURE_COLORS["off"], "transformer": "#D55E00", "ssm": STRUCTURE_COLORS["on-stable"]}

_CSV_FIELDS = ["backbone", "contrast", "seed_index", "diff_e_h_gate", "mean", "se"]


def _sidecar_rows(contrasts: dict) -> list[dict]:
    rows = []
    for backbone in _BACKBONES:
        c = contrasts[backbone]
        for i, diff in enumerate(c["contrasts"]):
            rows.append({
                "backbone": backbone, "contrast": _CONTRAST_LABEL[backbone], "seed_index": i,
                "diff_e_h_gate": diff, "mean": c["mean"], "se": c["se"],
            })
        if not c["contrasts"]:
            rows.append({"backbone": backbone, "contrast": _CONTRAST_LABEL[backbone], "seed_index": "",
                         "diff_e_h_gate": "", "mean": "", "se": ""})
    return rows


def make_fig3(
    results_root: str = "results_primary_factorial",
    exp_name: str = "primary_factorial",
    out_path: str = "assets/fig3_factorial_effects.png",
    sidecar_path: str | None = None,
    state_scale_path: str | None = None,
) -> dict:
    """PROTOCOL.md section 27.4, Fig. 3: the three paired ON-OFF contrasts of
    ``e_h_gate``, five seeds each, via ``results.paired_structure_contrasts``.

    Returns a dict with the plotted data (``contrasts``), the image path,
    and the sidecar CSV path + hash.
    """
    from sld.viz import results as results_mod

    rows = results_mod.build_primary_results_table(results_root, exp_name)
    if all(r["status"] == "missing" for r in rows):
        raise FileNotFoundError(
            f"fig3: no gate.json under {Path(results_root) / exp_name} -- nothing to plot yet"
        )
    contrasts = results_mod.paired_structure_contrasts(rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for i, backbone in enumerate(_BACKBONES):
        c = contrasts[backbone]
        diffs = c["contrasts"]
        color = _COLOR[backbone]
        if diffs:
            jitter = np.linspace(-0.09, 0.09, len(diffs)) if len(diffs) > 1 else np.zeros(1)
            ax.scatter(i + jitter, diffs, s=26, color=color, alpha=0.55, edgecolor="white",
                       linewidth=0.5, zorder=3)
        if c["mean"] is not None:
            ax.errorbar([i], [c["mean"]], yerr=[c["se"]], fmt="D", color=INK_PRIMARY, markersize=6,
                       capsize=4, linewidth=1.6, zorder=5)
        n = len(diffs)
        ax.annotate(f"n={n}", xy=(i, 0), xycoords=("data", "axes fraction"), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=7, color=INK_MUTED)

    reference_line(ax, 0.0, "0: no effect  (negative = structure helps, lower e_h_gate is better)", style="--")
    ax.set_xticks(range(len(_BACKBONES)))
    ax.set_xticklabels([_CONTRAST_LABEL[b] for b in _BACKBONES])
    ax.set_xlim(-0.5, len(_BACKBONES) - 0.5)
    ax.set_xlabel("paired ON - OFF contrast, by backbone")
    ax.set_ylabel(r"$e_h\_gate$(on) $-$ $e_h\_gate$(off)   (negative = structure helps)")
    ax.set_title("Structure effect, paired by seed within backbone -- the one confound-free contrast (27.2)", loc="left")
    ax.legend(handles=[
        plt.Line2D([], [], color=INK_MUTED, marker="o", linestyle="none", markersize=5, alpha=0.55, label="per-seed diff"),
        plt.Line2D([], [], color=INK_PRIMARY, marker="D", linestyle="none", markersize=6, label="mean ± 1 SE"),
    ], fontsize=7.5, loc="best", framealpha=0.95)

    sigma_clause = sigma_train_caption_fragment(
        state_scale_path, rows[0]["stratum_system"], rows[0]["stratum_regime"]
    )
    caption = (
        "PROTOCOL 27.1/27.2/27.4/27.7. System: CoupledOscillators. Regime: OURegime(theta_ou=1.0). H=25. "
        f"{sigma_clause} (sld.audit.scan_state_scale, identical across all six cells (27.1)). "
        "tau=0.5 (not this figure's statistic; e_h_gate is, 27.7). "
        "Censoring convention: n/a to e_h_gate directly -- a seed is included only when BOTH its on and off "
        "arm have a readable e_h_gate (status 'ok' or 'does_not_learn'); a seed missing on either side is "
        "skipped from that backbone's contrast, never faked as zero (27.3). n=5 seeds per backbone when "
        "complete, fewer shown as n= below each column. Non-eligible cells: none named here -- this figure "
        "pairs by SEED, not by whole-cell eligibility (27.3's cell eligibility gates Fig. 1's curves and the "
        "nine order verdicts, not this per-seed pairing). CONFOUND, declared before the runs (27.2): the "
        "three backbones do NOT share a learning rate (mlp/transformer 1e-3, ssm 3e-4), so this WITHIN-"
        "backbone contrast is the one clean comparison on this grid -- off and on of the same backbone share "
        "rate, dataset, seed and budget; comparing the THREE CONTRASTS to each other still inherits the "
        "cross-backbone rate confound and is not what this figure claims."
    )
    fig.text(0.0, -0.14, caption, fontsize=6.6, color=INK_MUTED, va="top", wrap=True)

    out_path = Path(out_path)
    save_fig(fig, out_path)
    plt.close(fig)

    sidecar_path = Path(sidecar_path) if sidecar_path else out_path.with_suffix(".csv")
    digest = write_sidecar_csv(_sidecar_rows(contrasts), _CSV_FIELDS, sidecar_path)

    print(f"wrote {out_path}")
    return {
        "image_path": out_path,
        "sidecar_csv": sidecar_path,
        "sidecar_hash": digest,
        "data": contrasts,
        "caption": caption,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig. 3 -- PROTOCOL.md section 27.4")
    ap.add_argument("--results-root", default="results_primary_factorial")
    ap.add_argument("--exp-name", default="primary_factorial")
    ap.add_argument("--out", default="assets/fig3_factorial_effects.png")
    ap.add_argument("--state-scale", default=None, help="path to a state_scale/1 JSON (sld.audit.scan_state_scale)")
    args = ap.parse_args()
    try:
        make_fig3(
            results_root=args.results_root, exp_name=args.exp_name, out_path=args.out,
            state_scale_path=args.state_scale,
        )
    except FileNotFoundError as exc:
        print(f"fig3: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
