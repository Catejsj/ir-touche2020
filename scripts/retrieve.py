"""The three retrieval models. Rubric criterion 4.

    boolean   set membership. A document either contains the query terms or it
              does not; there is no notion of "how well". Ranked here only by
              coordination level (how many distinct query terms matched), which
              is the minimum needed to compare it against ranked models at all.

    vsm       Vector Space Model. Documents and the query become TF-IDF vectors
              and are compared by cosine similarity. Length is normalised away
              by the cosine, so a long document has no built-in advantage.

    bm25      Probabilistic model. Term frequency saturates - the tenth mention
              of "tenure" adds far less than the second - and document length is
              corrected against the collection average rather than normalised
              away entirely.

All three read the SAME index built by preprocess_index.py and analyse the
query with the SAME pipeline used on the documents. If the query were tokenized
or stemmed differently from the index, nothing would match and the cause would
be invisible.

    python scripts/retrieve.py --model bm25
    python scripts/retrieve.py --model all

Output: data/processed/run_{model}.parquet, reports/03_retrieval.txt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "03_retrieval.txt"

TOP_K = 100          # how deep a run we store; evaluation cuts at 10 and 20

# BM25 parameters. k1 controls how fast term frequency saturates; b controls
# how hard long documents are penalised. These are the values Robertson and
# Zaragoza report as robust defaults, and they are swept in [5].
BM25_K1 = 1.2
BM25_B = 0.75


def load_index():
    if not (MODELS / "index.npz").exists():
        raise SystemExit("run scripts/preprocess_index.py first")
    X = sp.load_npz(MODELS / "index.npz").tocsc()
    vocab = json.loads((MODELS / "vocab.json").read_text())
    term_id = {t: i for i, t in enumerate(vocab["terms"])}
    return X, vocab, term_id


def analyse(text: str, vocab: dict, term_id: dict, stemmer) -> list[int]:
    """Query -> column ids, using the analyser the documents went through."""
    stop = set(vocab["stopwords"])
    tokens = re.findall(vocab["token_pattern"].replace("(?u)", ""), text.lower())
    tokens = [t for t in tokens if t not in stop]
    if vocab["stemmed"]:
        tokens = [stemmer.stem(t) for t in tokens]
    return [term_id[t] for t in tokens if t in term_id]


def postings(X, col: int):
    """The postings list for one term: (document ids, term frequencies)."""
    start, end = X.indptr[col], X.indptr[col + 1]
    return X.indices[start:end], X.data[start:end]


def score_boolean(X, cols, n_docs, **_):
    """Coordination level: how many distinct query terms the document has."""
    hits = np.zeros(n_docs, dtype=np.int32)
    for col in set(cols):
        docs, _ = postings(X, col)
        hits[docs] += 1
    return hits.astype(np.float64)


def score_vsm(X, cols, n_docs, idf, doc_norm, **_):
    """TF-IDF with cosine similarity.

    Document weight: (1 + log tf) * idf, the standard sub-linear tf.
    Query weight: the same, so a repeated query term counts once more.
    Cosine divides by the document's own vector length, which is what stops a
    long document scoring highly just by containing more words.
    """
    scores = np.zeros(n_docs, dtype=np.float64)
    counts = pd.Series(cols).value_counts()
    for col, q_tf in counts.items():
        docs, tf = postings(X, col)
        w_d = (1.0 + np.log(tf)) * idf[col]
        w_q = (1.0 + np.log(q_tf)) * idf[col]
        scores[docs] += w_d * w_q
    nonzero = scores != 0
    scores[nonzero] /= doc_norm[nonzero]
    return scores


def score_bm25(X, cols, n_docs, idf_bm25, doc_len, avg_len, k1, b, **_):
    scores = np.zeros(n_docs, dtype=np.float64)
    for col in set(cols):
        docs, tf = postings(X, col)
        tf = tf.astype(np.float64)
        denom = tf + k1 * (1.0 - b + b * doc_len[docs] / avg_len)
        scores[docs] += idf_bm25[col] * (tf * (k1 + 1.0)) / denom
    return scores


def build_stats(X):
    n_docs = X.shape[0]
    df = np.diff(X.indptr)                       # documents per term
    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    avg_len = doc_len.mean()

    # Classic idf for the vector space model.
    idf = np.log(n_docs / np.maximum(df, 1)) + 1.0

    # BM25's idf comes from the probabilistic derivation and is a different
    # formula, not a cosmetic variant - it can go negative for a term in more
    # than half the collection. The +0.5 smoothing keeps it finite.
    idf_bm25 = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    # Cosine needs each document's TF-IDF vector length. Computed once.
    Xc = X.tocsr()
    w = Xc.copy().astype(np.float64)
    w.data = (1.0 + np.log(w.data)) * idf[w.indices]
    doc_norm = np.sqrt(np.asarray(w.multiply(w).sum(axis=1)).ravel())
    doc_norm[doc_norm == 0] = 1.0
    return dict(n_docs=n_docs, df=df, doc_len=doc_len, avg_len=avg_len,
                idf=idf, idf_bm25=idf_bm25, doc_norm=doc_norm)


def run_model(name: str, X, stats, queries, doc_ids, cols_per_query,
              k1=BM25_K1, b=BM25_B, top_k=TOP_K) -> tuple[pd.DataFrame, float]:
    fn = {"boolean": score_boolean, "vsm": score_vsm, "bm25": score_bm25}[name]
    rows = []
    t0 = time.time()
    for qid, cols in cols_per_query.items():
        if not cols:
            continue
        scores = fn(X, cols, stats["n_docs"], idf=stats["idf"],
                    doc_norm=stats["doc_norm"], idf_bm25=stats["idf_bm25"],
                    doc_len=stats["doc_len"], avg_len=stats["avg_len"],
                    k1=k1, b=b)
        hit = np.flatnonzero(scores > 0)
        if len(hit) > top_k:
            part = hit[np.argpartition(-scores[hit], top_k)[:top_k]]
        else:
            part = hit
        order = part[np.argsort(-scores[part], kind="stable")]
        for rank, idx in enumerate(order, start=1):
            rows.append((qid, doc_ids[idx], rank, float(scores[idx])))
    elapsed = time.time() - t0
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "rank",
                                       "score"]), elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["all", "boolean", "vsm", "bm25"])
    ap.add_argument("--k1", type=float, default=BM25_K1)
    ap.add_argument("--b", type=float, default=BM25_B)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep BM25 k1/b and report it")
    args = ap.parse_args()

    X, vocab, term_id = load_index()
    queries = pd.read_parquet(PROC / "queries.parquet")
    doc_ids = pd.read_parquet(PROC / "doc_meta.parquet")["doc_id"].values
    stemmer = PorterStemmer()
    stats = build_stats(X)

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    cols_per_query = {q.query_id: analyse(q.text, vocab, term_id, stemmer)
                      for q in queries.itertuples()}

    say("[1] QUERY PROCESSING")
    say("    A query goes through the analyser the documents went through:")
    say("    lowercase, the same token pattern, the same stopword list, the same")
    say("    Porter stemmer. Any mismatch here and terms silently fail to meet.")
    say()
    say(f"    {'query':<46} {'terms':>6}  indexed terms")
    for q in queries.itertuples():
        cols = cols_per_query[q.query_id]
        terms = ", ".join(vocab["terms"][c] for c in cols)
        say(f"    {q.text[:44]:<46} {len(cols):>6}  {terms[:44]}")
        if q.Index >= 5:
            break
    say()
    lengths = [len(c) for c in cols_per_query.values()]
    say(f"    query terms after analysis: mean {np.mean(lengths):.1f}, "
        f"min {min(lengths)}, max {max(lengths)}")
    empty = [q for q, c in cols_per_query.items() if not c]
    say(f"    queries left with no indexed term: {len(empty)}")
    say()
    say("    Note what stopword removal did: 'Should teachers get tenure?' is six")
    say("    words and becomes two terms. That is correct - 'should', 'get' and")
    say("    the question mark cannot discriminate between documents - but it")
    say("    means the ranking rests on very little, and it is why the evaluation")
    say("    section looks at per-query variance rather than only averages.")
    say()

    say("[2] THE THREE MODELS")
    say()
    say("    BOOLEAN")
    say("      A document matches if it contains query terms. Pure set")
    say("      membership: no weighting, no notion of degree.")
    say("      We use OR, not AND. With a 2-3 term query, AND over a 60,000")
    say("      document collection returns a handful of documents or none at")
    say("      all, and a system that returns nothing cannot be evaluated.")
    say("      To compare it against ranked models we order by COORDINATION")
    say("      LEVEL - the number of distinct query terms present. This is")
    say("      already a concession: real Boolean retrieval returns an unordered")
    say("      set, and every ranking metric assumes an order. We state it")
    say("      rather than pretend Boolean is a ranking model.")
    say()
    say("    VECTOR SPACE MODEL")
    say("      Document weight  (1 + log tf) x idf")
    say("      Query weight     (1 + log qtf) x idf")
    say("      Similarity       cosine = dot product / document vector length")
    say("      The log dampens term frequency; idf rewards rare terms; the")
    say("      cosine removes length. A 4,000-word argument and a 40-word one")
    say("      are judged on direction, not size.")
    say()
    say("    BM25")
    say(f"      k1 = {args.k1}  how fast term frequency saturates")
    say(f"      b  = {args.b}  how hard long documents are penalised")
    say("      idf = log(1 + (N - df + 0.5) / (df + 0.5))")
    say("      Unlike the VSM, BM25 does not normalise length away completely -")
    say("      it compares each document against the collection average. Given")
    say("      this corpus (median 127 words, mean 300, max 16,162) that")
    say("      difference is the main reason to expect the two to diverge.")
    say()

    chosen = ["boolean", "vsm", "bm25"] if args.model == "all" else [args.model]
    say("[3] RUNS")
    say(f"    {'model':<10} {'queries':>8} {'rows':>9} {'docs/query':>11} "
        f"{'seconds':>9}")
    for name in chosen:
        run, elapsed = run_model(name, X, stats, queries, doc_ids,
                                 cols_per_query, args.k1, args.b, args.top_k)
        run.to_parquet(PROC / f"run_{name}.parquet", index=False)
        per_q = run.groupby("query_id").size()
        say(f"    {name:<10} {run['query_id'].nunique():>8} {len(run):>9,} "
            f"{per_q.mean():>11.1f} {elapsed:>9.2f}")
    say()
    say(f"    Runs are stored {args.top_k} deep. Evaluation cuts at 10 and 20;")
    say("    storing deeper costs nothing and lets recall be measured honestly.")
    say()

    say("[4] WHAT THE MODELS DISAGREE ABOUT  (one query, top 5 each)")
    example = queries.iloc[0]
    say(f"    query: {example.text}")
    say()
    corpus = pd.read_parquet(PROC / "corpus.parquet").set_index("doc_id")
    for name in chosen:
        run = pd.read_parquet(PROC / f"run_{name}.parquet")
        top = run[run["query_id"] == example.query_id].head(5)
        say(f"    --- {name}")
        for r in top.itertuples():
            title = str(corpus.loc[r.doc_id, "title"])[:58]
            say(f"      {r.rank}. {r.score:>8.3f}  {title}")
        say()

    if args.sweep and "bm25" in chosen:
        say("[5] BM25 PARAMETER SWEEP")
        say("    Reported for transparency. We do NOT tune on this data and then")
        say("    report the best: Touche has 49 queries and no validation split,")
        say("    so choosing k1 and b by test score would be fitting the test set.")
        say("    The defaults are used; the sweep only shows how much they matter.")
        say()
        qrels = pd.read_parquet(PROC / "qrels.parquet")
        relevant = {qid: set(g.loc[g.level > 0, "doc_id"])
                    for qid, g in qrels.groupby("query_id")}
        say(f"    {'k1':>5} {'b':>5} {'P@10':>8} {'recall@100':>11}")
        for k1 in [0.9, 1.2, 1.6, 2.0]:
            for b in [0.3, 0.5, 0.75, 0.9]:
                run, _ = run_model("bm25", X, stats, queries, doc_ids,
                                   cols_per_query, k1, b, args.top_k)
                p10, rec = [], []
                for qid, g in run.groupby("query_id"):
                    rel = relevant.get(qid, set())
                    if not rel:
                        continue
                    top10 = set(g.nsmallest(10, "rank")["doc_id"])
                    p10.append(len(top10 & rel) / 10)
                    rec.append(len(set(g["doc_id"]) & rel) / len(rel))
                say(f"    {k1:>5} {b:>5} {np.mean(p10):>8.3f} "
                    f"{np.mean(rec):>11.3f}")
        say()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
