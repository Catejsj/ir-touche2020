"""Load the Touche-2020 collection and describe it. Rubric criterion 2.

Three files come from the BEIR distribution:

    data/raw/corpus.parquet    382,545 arguments from args.me
    data/raw/queries.parquet   49 debate questions
    data/raw/qrels_test.tsv    relevance judgments, graded 0 / 1 / 2

A qrel row is one human judgment: "for query 1, document X was rated 2".
Every document a judge looked at is here, including the ones rated 0 - which is
unusual and is the reason this collection was chosen. Most IR datasets only
record what was relevant, so a wrong answer is just absent. Here a wrong answer
is an explicit human "no", which is what makes the error analysis in
scripts/evaluate.py possible.

Output: data/processed/{corpus,queries,qrels}.parquet, reports/01_corpus.txt
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "01_corpus.txt"

# A handful of args.me documents are empty or a single punctuation mark. They
# cannot be retrieved meaningfully and they distort the length statistics.
MIN_DOC_WORDS = 3

# The full args.me collection is 369,390 documents and 110 million words, which
# does not fit in this machine's memory once an index is built on top of it. We
# therefore retrieve over a POOLED SUBSET: every document a human judged, plus a
# random sample of the rest as distractors. Every relevance judgment is kept, so
# the evaluation is unaffected in kind - only the difficulty changes, and that
# is stated in the report and the limitations.
DEFAULT_MAX_DOCS = 60_000
SEED = 42


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                    help="size of the pooled subset; 0 keeps the full corpus")
    args = ap.parse_args()

    missing = [p.name for p in
               [RAW / "corpus.parquet", RAW / "queries.parquet",
                RAW / "qrels_test.tsv"] if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing} in data/raw - see README")

    corpus = pd.read_parquet(RAW / "corpus.parquet")
    queries = pd.read_parquet(RAW / "queries.parquet")
    qrels = pd.read_csv(RAW / "qrels_test.tsv", sep="\t",
                        dtype={"query-id": str, "corpus-id": str})

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    n_raw = len(corpus)

    say("[1] SOURCE")
    say("    collection   Webis-Touche-2020, Task 1 (argument retrieval)")
    say("    shared task  Touche @ CLEF 2020 (Bondarenko et al.)")
    say("    distributed  as part of BEIR (Thakur et al., NeurIPS 2021)")
    say("    obtained     huggingface.co/datasets/BeIR/webis-touche2020")
    say("    documents    args.me corpus - arguments crawled from five debate")
    say("                 portals (debatewise, idebate, debatepedia, debate.org,")
    say("                 canadian parliament)")
    say("    licence      CC BY-SA 4.0")
    say()
    say("    The retrieval task: given a controversial question a person might")
    say("    actually type - 'Should teachers get tenure?' - return arguments that")
    say("    help them take a side. Relevance is argument quality and stance")
    say("    support, not topical aboutness, which is why the judges used a scale")
    say("    rather than a yes/no.")
    say()

    # --- cleaning --------------------------------------------------------
    corpus["title"] = corpus["title"].map(clean)
    corpus["text"] = corpus["text"].map(clean)
    queries["text"] = queries["text"].map(clean)

    word_count = corpus["text"].str.split().str.len()
    too_short = int((word_count < MIN_DOC_WORDS).sum())
    corpus = corpus[word_count >= MIN_DOC_WORDS].reset_index(drop=True)

    before = len(corpus)
    corpus = corpus.drop_duplicates(subset=["_id"]).reset_index(drop=True)
    dup_ids = before - len(corpus)

    say("[2] CLEANING")
    say(f"    documents as distributed   {n_raw:,}")
    say(f"    dropped, under {MIN_DOC_WORDS} words     -{too_short:,}")
    say(f"    dropped, duplicate id      -{dup_ids:,}")
    say(f"    kept                       {len(corpus):,} "
        f"({len(corpus) / n_raw:.1%})")
    say()
    say("    Unicode is normalised (NFKC) and whitespace collapsed, per field.")
    say("    Nothing is lowercased or stemmed here - that belongs to the indexing")
    say("    stage, and the raw text has to survive so results can be shown to a")
    say("    reader in the form the debater wrote them.")
    say()

    # --- pooled subset ----------------------------------------------------
    n_full = len(corpus)
    judged_ids = set(qrels["corpus-id"])
    if args.max_docs and n_full > args.max_docs:
        is_judged = corpus["_id"].isin(judged_ids)
        pool = corpus[is_judged]
        rest = corpus[~is_judged]
        n_extra = max(0, args.max_docs - len(pool))
        sample = rest.sample(n=min(n_extra, len(rest)), random_state=SEED)
        corpus = pd.concat([pool, sample]).sort_values("_id").reset_index(drop=True)

        say("[2b] POOLED SUBSET")
        say(f"    The full collection is {n_full:,} documents and about 110")
        say("    million words. Building an index over it exhausted the memory on")
        say("    the machine available to us - the process was killed by the")
        say("    kernel. Rather than hide that, we retrieve over a pooled subset,")
        say("    which is standard practice when a full collection is out of reach:")
        say()
        say(f"      every JUDGED document kept      {len(pool):,}")
        say(f"      random unjudged distractors     {len(sample):,} (seed {SEED})")
        say(f"      collection searched             {len(corpus):,} "
            f"({len(corpus) / n_full:.1%} of the full corpus)")
        say()
        n_judgments = int(qrels["corpus-id"].isin(set(pool["_id"])).sum())
        say(f"    ({len(pool):,} distinct documents carrying {n_judgments:,} "
            "judgments - a few documents")
        say("    were judged for more than one query.)")
        say()
        say("    Why this keeps the evaluation valid: a metric can only be computed")
        say("    over documents a human judged, and every one of them is still")
        say("    here. No query loses a relevant document, and no measure changes")
        say("    definition.")
        say()
        say("    What it does change, and we say so on the slide: the task is")
        say("    easier. A system now sifts 60,000 documents instead of 369,390,")
        say("    so there are roughly six times fewer chances to rank an irrelevant")
        say("    document above a relevant one. Our scores are therefore an")
        say("    upper bound on what the same systems would achieve on the full")
        say("    collection, and they are not comparable to published Touche")
        say("    numbers. Re-running at full scale is the first item in future")
        say("    work - the code takes --max-docs 0 and needs only more memory.")
        say()

    # --- description -----------------------------------------------------
    kept_ids = set(corpus["_id"])
    judged = qrels[qrels["corpus-id"].isin(kept_ids)]
    lost = len(qrels) - len(judged)

    say("[3] SIZE")
    wc = corpus["text"].str.split().str.len()
    say(f"    documents searched   {len(corpus):,}"
        + (f"   (pooled subset of {n_full:,})" if len(corpus) < n_full else ""))
    say(f"    queries (test)       {queries['_id'].nunique():,}")
    say(f"    relevance judgments  {len(judged):,}"
        + (f"  ({lost} dropped with the short documents)" if lost else ""))
    say()
    say("    document length in words")
    for label, value in [("min", wc.min()), ("25th pct", wc.quantile(.25)),
                         ("median", wc.median()), ("mean", wc.mean()),
                         ("75th pct", wc.quantile(.75)),
                         ("95th pct", wc.quantile(.95)), ("max", wc.max())]:
        say(f"      {label:<10} {value:>9,.0f}")
    say()
    say(f"    total words          {wc.sum():,}")
    say("    The distribution is heavily right-skewed - the mean is well above")
    say("    the median because a few arguments run to thousands of words. This")
    say("    is the reason BM25's length normalisation matters here and is")
    say("    revisited in the retrieval report.")
    say()

    ql = queries["text"].str.split().str.len()
    say("    query length in words")
    say(f"      mean {ql.mean():.1f}, median {ql.median():.0f}, "
        f"min {ql.min()}, max {ql.max()}")
    say()
    say("    example queries")
    for text in queries["text"].head(6):
        say(f"      {text}")
    say()

    # --- the judgments ---------------------------------------------------
    say("[4] RELEVANCE JUDGMENTS  (this is the part that decides the evaluation)")
    counts = judged["score"].value_counts().sort_index()
    total = int(counts.sum())
    meaning = {0: "judged NOT relevant", 1: "relevant",
               2: "highly relevant"}
    say(f"    {'level':>6} {'judgments':>11} {'share':>8}  meaning")
    for level, count in counts.items():
        say(f"    {level:>6} {count:>11,} {count / total:>7.1%}  "
            f"{meaning.get(int(level), '')}")
    say()
    per_query = judged.groupby("query-id").size()
    rel_per_query = judged[judged["score"] > 0].groupby("query-id").size()
    say(f"    documents judged per query   mean {per_query.mean():.1f}, "
        f"min {per_query.min()}, max {per_query.max()}")
    say(f"    RELEVANT (level >= 1) per query  mean {rel_per_query.mean():.1f}, "
        f"min {rel_per_query.min()}, max {rel_per_query.max()}")
    say()
    say("    Two consequences, both of which shape the evaluation:")
    say()
    say(f"    1. Graded relevance. {counts.get(2, 0):,} documents are rated 2 and")
    say(f"       {counts.get(1, 0):,} are rated 1. A system that puts level-2")
    say("       arguments on top is better than one that puts level-1 arguments")
    say("       on top, and nDCG is the measure that can see that difference.")
    say("       Precision/recall/F1 cannot - they need a yes/no, so we state the")
    say("       threshold we use (level >= 1 counts as relevant) rather than")
    say("       leaving it implicit.")
    say()
    say(f"    2. {counts.get(0, 0):,} documents were judged and found NOT relevant.")
    say("       Most collections omit these, so a wrong answer is indistinguishable")
    say("       from an unjudged one. Here, when a system ranks one of these in the")
    say("       top 10, a human has explicitly said it does not answer the query -")
    say("       which is a much stronger error analysis than 'not in our list'.")
    say()
    say("    Unjudged documents: the other 382,000-odd documents were never shown")
    say("    to a judge. Standard TREC practice treats them as not relevant, and")
    say("    that is what we do, but it is an assumption and it is in the")
    say("    limitations.")
    say()

    # --- legal -----------------------------------------------------------
    say("[5] LEGAL AND ETHICAL NOTES")
    say("    - Licence: CC BY-SA 4.0. Reuse, modification and redistribution are")
    say("      permitted with attribution and share-alike. The Touche task and the")
    say("      BEIR paper are both cited in the references.")
    say("    - Provenance: args.me crawled five public debate portals. We use the")
    say("      published research corpus, not our own crawl, so no site's terms of")
    say("      service are engaged by this project.")
    say("    - Personal data: arguments were posted publicly under usernames.")
    say("      The BEIR distribution carries no author field, no username and no")
    say("      timestamp - only an id, a title and the argument text. There is")
    say("      therefore nothing to anonymise, and we add nothing back.")
    say("    - We do not attempt to identify authors, link arguments to people, or")
    say("      aggregate a person's positions across documents.")
    say("    - Content: the corpus is debate material on abortion, gun control,")
    say("      religion, immigration and similar. Some arguments are offensive.")
    say("      They are retrieved and scored, never endorsed; any example shown in")
    say("      the report is chosen to illustrate a retrieval behaviour, and")
    say("      slides carry a content note.")
    say("    - Our system ranks arguments by how well they match a query. It does")
    say("      not judge whether an argument is true, and it should not be")
    say("      presented as doing so.")

    PROC.mkdir(parents=True, exist_ok=True)
    corpus.rename(columns={"_id": "doc_id"})[["doc_id", "title", "text"]] \
        .to_parquet(PROC / "corpus.parquet", index=False)
    queries.rename(columns={"_id": "query_id"})[["query_id", "text"]] \
        .to_parquet(PROC / "queries.parquet", index=False)
    judged.rename(columns={"query-id": "query_id", "corpus-id": "doc_id",
                           "score": "level"}).to_parquet(
        PROC / "qrels.parquet", index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {(PROC / 'corpus.parquet').relative_to(ROOT)} and 2 more")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
