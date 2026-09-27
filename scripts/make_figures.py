"""Figures for the slides. Every number is read from the run files.

fig1  the headline: nDCG@10, MAP, MRR for the three models
fig2  precision / recall / F1 at k = 5, 10, 20
fig3  what is actually in a top 10: relevant / judged wrong / unjudged
fig4  document length of what each model returns - why VSM loses
fig5  per-query nDCG@10, all 49 queries, so the spread is visible
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
RED = "#e34948"
YELLOW = "#eda100"

MODELS = ["boolean", "vsm", "bm25"]
NICE = {"boolean": "Boolean", "vsm": "VSM (TF-IDF)", "bm25": "BM25"}
COLOR = {"boolean": BLUE, "vsm": ORANGE, "bm25": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK,
})


def strip(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_color(GRID)
    ax.tick_params(length=0)


def load():
    path = PROC / "per_query.parquet"
    if not path.exists():
        raise SystemExit("run scripts/evaluate.py first")
    return pd.read_parquet(path)


def fig_headline(pq: pd.DataFrame) -> None:
    measures = ["nDCG@10", "MAP", "MRR"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(measures))
    width = 0.26
    for i, name in enumerate(MODELS):
        sub = pq[pq["model"] == name]
        vals = [sub[m].mean() for m in measures]
        ax.bar(x + (i - 1) * width, vals, width, color=COLOR[name],
               label=NICE[name], zorder=3)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(x, measures)
    ax.set_ylabel("score (higher is better)")
    ax.set_ylim(0, max(pq[m].mean() for m in measures) * 1.55)
    ax.set_title("BM25 wins on every ranking measure",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "49 queries, Touche-2020; relevance threshold level >= 1",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_headline.png", dpi=200)
    plt.close(fig)


def fig_prf(pq: pd.DataFrame) -> None:
    cutoffs = [5, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, measure in zip(axes, ["P", "R", "F1"]):
        x = np.arange(len(cutoffs))
        width = 0.26
        for i, name in enumerate(MODELS):
            sub = pq[pq["model"] == name]
            vals = [sub[f"{measure}@{k}"].mean() for k in cutoffs]
            ax.bar(x + (i - 1) * width, vals, width, color=COLOR[name],
                   label=NICE[name], zorder=3)
        ax.set_xticks(x, [f"k={k}" for k in cutoffs])
        ax.set_title({"P": "Precision@k", "R": "Recall@k",
                      "F1": "F1@k"}[measure], fontsize=11, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        strip(ax)
    axes[0].set_ylabel("score")
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.suptitle("Set-based measures: the same ordering at every cutoff",
                 fontsize=12, x=0.008, ha="left", y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_precision_recall_f1.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_top10_composition() -> None:
    qrels = pd.read_parquet(PROC / "qrels.parquet")
    levels = {qid: dict(zip(g["doc_id"], g["level"]))
              for qid, g in qrels.groupby("query_id")}
    rows = []
    for name in MODELS:
        run = pd.read_parquet(PROC / f"run_{name}.parquet")
        counts = {"relevant": 0, "judged wrong": 0, "unjudged": 0}
        for qid, g in run.groupby("query_id"):
            lv = levels.get(qid, {})
            for doc in g.sort_values("rank")["doc_id"].head(10):
                if doc not in lv:
                    counts["unjudged"] += 1
                elif lv[doc] > 0:
                    counts["relevant"] += 1
                else:
                    counts["judged wrong"] += 1
        rows.append(counts)

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))
    for label, color in [("relevant", AQUA), ("judged wrong", RED),
                         ("unjudged", "#c3c2b7")]:
        vals = np.array([r[label] for r in rows], dtype=float)
        vals = vals / np.array([sum(r.values()) for r in rows])
        ax.barh(y, vals, 0.55, left=left, color=color, label=label, zorder=3,
                edgecolor=SURFACE, linewidth=2)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0.05:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                        fontsize=9,
                        color="#ffffff" if label != "unjudged" else INK2)
        left += vals
    ax.set_yticks(y, [NICE[m] for m in MODELS])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the 490 documents in the top 10 of 49 queries")
    ax.set_title("What is actually in a top 10",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.02, "'judged wrong' = a human looked at it and said no; "
                     "'unjudged' = nobody checked",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    strip(ax, keep=("bottom",))
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, fontsize=9,
              loc="lower center", bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_top10_composition.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_length() -> None:
    corpus = pd.read_parquet(PROC / "corpus.parquet").set_index("doc_id")
    corpus_len = corpus["text"].str.split().str.len()
    data, labels, colors = [], [], []
    for name in MODELS:
        run = pd.read_parquet(PROC / f"run_{name}.parquet")
        top = []
        for qid, g in run.groupby("query_id"):
            top.extend(g.sort_values("rank")["doc_id"].head(10))
        data.append(np.log10(corpus_len.loc[top].clip(lower=1).values))
        labels.append(NICE[name])
        colors.append(COLOR[name])
    data.append(np.log10(corpus_len.clip(lower=1).values))
    labels.append("the collection")
    colors.append("#c3c2b7")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("none")
    for i, values in enumerate(data, start=1):
        ax.scatter([i], [np.median(values)], color=SURFACE, s=28, zorder=4,
                   edgecolor=INK2, linewidth=1.2)
        ax.text(i + 0.16, np.median(values), f"{10 ** np.median(values):,.0f}",
                fontsize=9, color=INK2, va="center")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.tick_params(axis="x", labelcolor=INK2)
    ax.set_yticks([0, 1, 2, 3, 4], ["1", "10", "100", "1,000", "10,000"])
    ax.set_ylabel("document length in words (log scale)")
    ax.set_title("Why the vector space model loses:\nit returns very short "
                 "documents", fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "each shape is the length distribution of the documents "
                      "that model puts in a top 10",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_length_bias.png", dpi=200)
    plt.close(fig)


def fig_per_query(pq: pd.DataFrame) -> None:
    order = (pq[pq["model"] == "bm25"].sort_values("nDCG@10")["query_id"]
             .tolist())
    pos = {q: i for i, q in enumerate(order)}
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for name in MODELS:
        sub = pq[pq["model"] == name]
        xs = [pos[q] for q in sub["query_id"]]
        ax.scatter(xs, sub["nDCG@10"], s=26, color=COLOR[name],
                   label=NICE[name], alpha=0.85, zorder=3,
                   edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("the 49 queries, ordered by BM25's score")
    ax.set_ylabel("nDCG@10")
    ax.set_xticks([])
    ax.set_title("Averages hide the spread: BM25 scores 0.00 on some queries\n"
                 "and above 0.90 on others",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "one dot per query per model",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_per_query.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    pq = load()
    fig_headline(pq)
    fig_prf(pq)
    fig_top10_composition()
    fig_length()
    fig_per_query(pq)
    for path in sorted(FIG.glob("*.png")):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
