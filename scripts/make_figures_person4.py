"""Two mechanism diagrams for the retrieval-models part of the talk.

fig6  one query, one document, three models - what each one actually computes
fig7  BM25's two fixes drawn as curves: saturation, and length normalisation

These explain HOW the models score, not how well they did. The result figures
(fig1-fig5) belong to the evaluation speaker; nothing here overlaps them.

Every number is computed with the same functions retrieve.py uses, so the slide
cannot drift away from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem import PorterStemmer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import retrieve as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROC = ROOT / "data" / "processed"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PALE = "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK,
})


def strip(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_color(GRID)
    ax.tick_params(length=0)


def gather():
    """Real numbers for one query / one document, via retrieve.py's own code."""
    X, vocab, term_id = R.load_index()
    stats = R.build_stats(X)
    queries = pd.read_parquet(PROC / "queries.parquet")
    corpus = pd.read_parquet(PROC / "corpus.parquet")
    stemmer = PorterStemmer()

    q = queries.iloc[0]
    cols = R.analyse(q.text, vocab, term_id, stemmer)

    run = pd.read_parquet(PROC / "run_bm25.parquet")
    top = run[run["query_id"] == q.query_id].sort_values("rank").iloc[0]
    doc_pos = int(np.flatnonzero(corpus["doc_id"].values == top.doc_id)[0])

    rows = []
    for col in dict.fromkeys(cols):
        docs, tf = R.postings(X, col)
        hit = np.flatnonzero(docs == doc_pos)
        tf_here = int(tf[hit[0]]) if len(hit) else 0
        rows.append(dict(term=vocab["terms"][col], col=col, tf=tf_here,
                         df=int(stats["df"][col]),
                         idf=float(stats["idf"][col]),
                         idf_bm25=float(stats["idf_bm25"][col])))
    return dict(query=q.text, doc_id=top.doc_id,
                title=str(corpus.iloc[doc_pos]["title"]),
                doc_len=float(stats["doc_len"][doc_pos]),
                avg_len=float(stats["avg_len"]),
                norm=float(stats["doc_norm"][doc_pos]),
                n_docs=int(stats["n_docs"]), rows=rows)


def fig_walkthrough(d) -> None:
    k1, b = R.BM25_K1, R.BM25_B
    fig, ax = plt.subplots(figsize=(11.6, 7.1))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 7.1); ax.axis("off")

    ax.text(0.3, 6.78, "One query, one document, three models",
            fontsize=13.5, fontweight="bold", color=INK)
    ax.text(0.3, 6.50, "every number below is computed by the same code that "
                       "produced our results", fontsize=9.5, color=MUTED)

    ax.text(0.3, 6.08, "QUERY", fontsize=9, color=BLUE)
    ax.text(1.6, 6.08, f'"{d["query"]}"', fontsize=11, color=INK)
    terms = ", ".join(r["term"] for r in d["rows"])
    ax.text(1.6, 5.80, f"after cleaning  ->  {terms}", fontsize=9.5,
            color=INK2, family="monospace")

    ax.text(0.3, 5.40, "DOCUMENT", fontsize=9, color=BLUE)
    ax.text(1.6, 5.40, f'"{d["title"][:58]}"', fontsize=11, color=INK)
    ax.text(1.6, 5.12, f'{d["doc_len"]:.0f} terms long '
                       f'(collection average {d["avg_len"]:.0f})',
            fontsize=9.5, color=INK2)

    # what the index gives us
    y = 4.45
    ax.text(0.3, y, "WHAT THE INDEX TELLS US", fontsize=9, color=BLUE)
    y -= 0.32
    ax.text(0.5, y, f"{'term':<10}{'in this doc':>13}{'in how many docs':>19}",
            fontsize=9.5, color=MUTED, family="monospace")
    for r in d["rows"]:
        y -= 0.28
        ax.text(0.5, y, f"{r['term']:<10}{r['tf']:>10} times"
                        f"{r['df']:>14,} of {d['n_docs']:,}",
                fontsize=9.5, color=INK, family="monospace")

    # three models
    boolean = sum(1 for r in d["rows"] if r["tf"] > 0)
    vsm = sum((1 + np.log(r["tf"])) * r["idf"] * (1 + np.log(1)) * r["idf"]
              for r in d["rows"] if r["tf"] > 0) / d["norm"]
    bm25 = sum(r["idf_bm25"] * (r["tf"] * (k1 + 1)) /
               (r["tf"] + k1 * (1 - b + b * d["doc_len"] / d["avg_len"]))
               for r in d["rows"] if r["tf"] > 0)

    panels = [
        (BLUE, "BOOLEAN", "are the query words in it?",
         f"{boolean} of {len(d['rows'])} present", f"score = {boolean:.0f}"),
        (ORANGE, "VECTOR SPACE", "angle between two lists of numbers",
         "(1+log tf) x idf, then divide\nby the document's own length",
         f"score = {vsm:.3f}"),
        (AQUA, "BM25", "same, but saturated + length-fixed",
         f"tf saturates (k1={k1}); length\ncompared to average (b={b})",
         f"score = {bm25:.3f}"),
    ]
    for i, (color, name, idea, how, score) in enumerate(panels):
        x = 0.3 + i * 3.78
        ax.add_patch(plt.Rectangle((x, 0.25), 3.5, 1.88, facecolor=SURFACE,
                                   edgecolor=color, linewidth=1.8, zorder=2))
        ax.text(x + 0.16, 1.82, name, fontsize=11, fontweight="bold",
                color=color)
        ax.text(x + 0.16, 1.50, idea, fontsize=8, color=INK2, wrap=True)
        ax.text(x + 0.16, 1.00, how, fontsize=8, color=MUTED,
                family="monospace")
        ax.text(x + 0.16, 0.45, score, fontsize=11.5, fontweight="bold",
                color=INK)

    ax.text(0.3, 2.45, "The three scores are on different scales, so they are "
                       "never compared to each other -", fontsize=9.5, color=INK2)
    ax.text(0.3, 2.21, "each one only ranks documents within its own model.",
            fontsize=9.5, color=INK2)

    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig6_scoring_walkthrough.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_curves(d) -> None:
    k1, b = R.BM25_K1, R.BM25_B
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # --- panel A: saturation -------------------------------------------
    ax = axes[0]
    tf = np.arange(1, 21)
    ax.plot(tf, tf / tf[0], color=MUTED, lw=1.6, ls=(0, (4, 3)),
            label="raw count (no damping)")
    ax.plot(tf, (1 + np.log(tf)) / (1 + np.log(1)), color=ORANGE, lw=2.2,
            marker="s", ms=4, label="TF-IDF  (1 + log tf)")
    bm = (tf * (k1 + 1)) / (tf + k1)
    ax.plot(tf, bm / bm[0], color=AQUA, lw=2.2, marker="o", ms=4,
            label=f"BM25  (k1 = {k1})")
    ax.set_ylim(0, 5.2)
    ax.annotate("raw count keeps\nclimbing (off the chart)", xy=(5.3, 5.05),
                xytext=(6.3, 4.45), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.annotate("BM25 flattens: the 10th\nmention adds almost nothing",
                xy=(13, bm[12] / bm[0]), xytext=(8.0, 3.05), fontsize=9,
                color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("times the word appears in the document")
    ax.set_ylabel("how much it adds to the score\n(relative to appearing once)")
    ax.set_title("Fix 1 — saturation", fontsize=11.5, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    strip(ax); ax.legend(frameon=False, labelcolor=INK2, fontsize=8.5)

    # --- panel B: length -------------------------------------------------
    ax = axes[1]
    ratio = np.linspace(0.05, 4, 200)
    for bb, color, style, lw in [(0.0, MUTED, (0, (4, 3)), 1.8),
                                 (b, AQUA, "solid", 2.4),
                                 (1.0, BLUE, (0, (1, 2)), 1.8)]:
        mult = 1.0 / (1 - bb + bb * ratio)
        ax.plot(ratio, mult, color=color, lw=lw, ls=style,
                label=f"b = {bb}" + ("   <- ours" if bb == b else ""))
    ax.axvline(1.0, color="#c3c2b7", lw=1)
    ax.text(1.06, 3.55, "average-length document", fontsize=8.5, color=MUTED)
    ax.annotate("b = 0: length ignored completely.\nThis is what the vector space\n"
                "model effectively does.",
                xy=(2.6, 1.0), xytext=(1.55, 2.35), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("b = 1: fully corrected -\nshort documents boosted hard",
                xy=(0.34, 1 / 0.34), xytext=(1.35, 3.05), fontsize=9,
                color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("document length  ÷  average length")
    ax.set_ylabel("score multiplier for a term")
    ax.set_ylim(0, 4)
    ax.set_title("Fix 2 — length normalisation", fontsize=11.5, loc="left",
                 pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    strip(ax); ax.legend(frameon=False, labelcolor=INK2, fontsize=8.5)

    fig.suptitle("BM25's two fixes to TF-IDF", fontsize=13, x=0.006,
                 ha="left", y=1.04, color=INK, fontweight="bold")
    fig.text(0.006, 0.985, "these are the formulas themselves, drawn - not "
                           "measurements of our results", fontsize=9.5,
             color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_bm25_two_fixes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = gather()
    fig_walkthrough(d)
    fig_curves(d)
    for name in ["fig6_scoring_walkthrough.png", "fig7_bm25_two_fixes.png"]:
        print(f"wrote figures/{name}")


if __name__ == "__main__":
    main()
