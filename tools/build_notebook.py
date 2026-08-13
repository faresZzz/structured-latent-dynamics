"""Build a .ipynb from a percent-format source (stdlib only, no jupytext).

Source format, one cell per block:

    # %% [markdown]
    # Une cellule markdown. Chaque ligne commence par "# " ou est vide.

    # %%
    import numpy as np          # une cellule de code, telle quelle

Why this exists rather than jupytext: the notebooks are authored by appending
chunks to a plain-text file, which survives a dropped connection mid-write in a
way a JSON document does not. The conversion is then a pure function of the
source, so a partial write is a visible syntax problem and never a silently
corrupt notebook.

Outputs carry no execution counts and no results: the workspace standard wants
notebooks committed clean (`nb-clean`), and a notebook whose outputs were
written by a generator would be a figure that never ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    kind: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            lines = [
                ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                for ln in body.split("\n")
            ]
            src = "\n".join(lines)
            cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": body.splitlines(keepends=True),
                }
            )

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    flush()
    return cells


def build(src: Path, out: Path) -> int:
    cells = parse_cells(src.read_text())
    if not cells:
        print(f"{src}: aucune cellule trouvee", file=sys.stderr)
        return 1
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    print(f"{out}: {len(cells)} cellules ({n_md} markdown, {len(cells) - n_md} code)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("output")
    a = ap.parse_args()
    raise SystemExit(build(Path(a.source), Path(a.output)))


if __name__ == "__main__":
    main()
