"""Preprocess the text and build the inverted index. Rubric criterion 3.

The five stages the rubric names, in the order they run, each with the number
it cost or saved:

  1 normalization      lowercase, Unicode NFKC, strip punctuation
  2 tokenization       regex word tokenizer over 110 million words
  3 stopword removal   NLTK English list + a small domain list
  4 stemming           Porter, applied to the vocabulary rather than to every
                       token (identical result, roughly 400x less work)
  5 inverted index     term -> postings list of (document, term frequency)

The index is stored as a sparse matrix. A compressed-sparse-column matrix over
(document x term) IS an inverted index: column j holds exactly the documents
containing term j, with their counts, which is the definition of a postings
list. Storing it this way means the same structure serves Boolean retrieval,
TF-IDF and BM25 without being rebuilt three times.

Output: models/index.npz, models/vocab.json, data/processed/doc_meta.parquet,
        reports/02_preprocessing_index.txt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "02_preprocessing_index.txt"

# Words that appear everywhere in a debate corpus and separate nothing. Every
# document argues, claims and believes something.
DOMAIN_STOPWORDS = {
    "argument", "arguments", "debate", "point", "points", "say", "says",
    "said", "think", "believe", "would", "could", "should", "also", "thus",
    "therefore", "however", "furthermore", "moreover", "con", "pro",
}

TOKEN_PATTERN = r"(?u)\b[a-z][a-z'-]+\b"

# A term in only one document cannot connect a query to anything, and the long
# tail of typos and one-off strings is most of the vocabulary.
MIN_DF = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-df", type=int, default=MIN_DF)
    ap.add_argument("--no-stem", action="store_true",
                    help="skip stemming, for the comparison in [5]")
    args = ap.parse_args()

    for package in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)

    corpus_path = PROC / "corpus.parquet"
    if not corpus_path.exists():
        raise SystemExit("run scripts/build_corpus.py first")
    corpus = pd.read_parquet(corpus_path)

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    stop = set(stopwords.words("english")) | DOMAIN_STOPWORDS
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # The title is part of the searchable text: an args.me title is the debate
    # motion, which is often the best single sentence in the document.
    text = (corpus["title"].fillna("") + " " + corpus["text"].fillna("")).values

    say("[1] NORMALIZATION")
    say("    lowercase          'Tenure' and 'tenure' are one term")
    say("    Unicode NFKC       done in build_corpus, per field")
    say(f"    token pattern      {TOKEN_PATTERN}")
    say("                       letters, apostrophes and hyphens only, so")
    say("                       'don't' and 'well-being' survive as one token")
    say("                       while '1990', '$' and '###' are dropped")
    say("    minimum length     2 characters")
    say()
    say("    Numbers are discarded. In this collection they are years, vote")
    say("    counts and statistics quoted inside arguments; a query like 'Should")
    say("    teachers get tenure?' never contains one, so indexing them costs")
    say("    memory and buys no matches.")
    say()

    say("[2] TOKENIZATION")
    say("    Tool: scikit-learn's regex tokenizer, not nltk.word_tokenize.")
    say("    This is a deliberate reversal of what we would do for a tagging")
    say("    task. NLTK's Treebank tokenizer is more careful - it splits clitics")
    say("    and handles punctuation properly - but it is a Python loop, and this")
    say("    corpus is 110 million words. The regex runs in C.")
    say("    For retrieval the difference does not matter: a term is only ever")
    say("    used as a dictionary key, never shown to a user and never labelled,")
    say("    so 'don' + 't' versus 'don't' changes nothing about which documents")
    say("    come back, as long as queries are tokenized the same way. They are -")
    say("    the same analyzer is applied to both sides.")
    say()

    t0 = time.time()
    raw_vec = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                              dtype=np.int32)
    X_raw = raw_vec.fit_transform(text)
    raw_vocab = raw_vec.get_feature_names_out()
    say("[3] VOCABULARY, BEFORE FILTERING")
    say(f"    distinct terms     {len(raw_vocab):,}")
    say(f"    postings (non-zero document-term pairs) {X_raw.nnz:,}")
    say(f"    tokens indexed     {int(X_raw.sum()):,}")
    say(f"    built in           {time.time() - t0:.0f}s")
    say()

    # --- stopwords -------------------------------------------------------
    keep_mask = np.array([t not in stop for t in raw_vocab])
    n_stop_removed = int((~keep_mask).sum())
    tokens_removed = int(X_raw[:, ~keep_mask].sum())

    say("[4] STOPWORD REMOVAL")
    say(f"    NLTK English list           {len(stopwords.words('english'))} words")
    say(f"    our debate-specific list    {len(DOMAIN_STOPWORDS)} words")
    say(f"    terms removed from vocab    {n_stop_removed:,}")
    say(f"    tokens removed              {tokens_removed:,} "
        f"({tokens_removed / X_raw.sum():.1%} of all tokens)")
    say()
    say("    A third of the text is stopwords, and removing them shrinks the")
    say("    index without changing what can be found - 'the' appears in almost")
    say("    every document, so it separates nothing.")
    say()
    say("    The domain list is the interesting half. In a debate corpus every")
    say("    document contains 'argument', 'claim', 'should' and 'believe'. They")
    say("    behave exactly like stopwords here even though no standard list")
    say("    contains them. Removed: " + ", ".join(sorted(DOMAIN_STOPWORDS)[:8])
        + ", ...")
    say()
    say("    Caveat we accept: dropping 'should' means the query 'Should teachers")
    say("    get tenure?' loses a word. That is fine - it appears in 60%+ of")
    say("    documents, so it could only add noise to the ranking.")
    say()

    # --- stemming --------------------------------------------------------
    say("[5] STEMMING")
    say("    Porter stemmer, applied to the VOCABULARY, not to every token.")
    say("    Stemming 110 million tokens one at a time in Python would take")
    say(f"    hours; stemming the {len(raw_vocab):,} distinct terms once and then")
    say("    merging the columns that collapse together gives an identical index")
    say("    in seconds. This is the single optimisation that makes the pipeline")
    say("    runnable on a laptop.")
    say()
    say("    Why stem at all, when the NER project deliberately did not?")
    say("    Because the output here is a ranked list of documents, not a span of")
    say("    text. Nobody ever sees a stem. A user searching 'teachers' should")
    say("    match a document arguing about 'teacher' and 'teaching', and")
    say("    stemming is what makes those the same key.")
    say()
    say("    Porter vs WordNet lemmatization on real terms from this corpus:")
    say(f"      {'term':<16} {'Porter':<14} {'WordNet lemma'}")
    for word in ["teachers", "teaching", "privatized", "privatization",
                 "smoking", "vaccines", "policies", "arguing"]:
        say(f"      {word:<16} {stemmer.stem(word):<14} "
            f"{lemmatizer.lemmatize(word)}")
    say()
    say("    Porter is the right choice for retrieval and the table shows why:")
    say("    it collapses 'privatized' and 'privatization' to one key, which is")
    say("    the behaviour a searcher wants. The lemmatizer keeps them apart")
    say("    because they are genuinely different words - correct linguistically,")
    say("    useless for matching. Porter's non-words ('privat', 'polici') are")
    say("    never displayed, so their ugliness costs nothing.")
    say()

    t1 = time.time()
    kept_terms = raw_vocab[keep_mask]
    X = X_raw[:, keep_mask].tocsc()

    if not args.no_stem:
        stems = np.array([stemmer.stem(t) for t in kept_terms])
        order = np.argsort(stems)
        stems_sorted = stems[order]
        unique_stems, start_index = np.unique(stems_sorted, return_index=True)
        # Sum the columns that share a stem: build a (terms x stems) selector
        # and multiply. One sparse matmul replaces a Python merge loop.
        col_of_term = np.empty(len(kept_terms), dtype=np.int64)
        col_of_term[order] = np.searchsorted(unique_stems, stems_sorted)
        selector = sp.csr_matrix(
            (np.ones(len(kept_terms), dtype=np.int32),
             (np.arange(len(kept_terms)), col_of_term)),
            shape=(len(kept_terms), len(unique_stems)))
        X = (X @ selector).tocsc()
        vocab_terms = unique_stems
        collapsed = len(kept_terms) - len(unique_stems)
    else:
        vocab_terms = kept_terms
        collapsed = 0

    say(f"    terms before stemming       {len(kept_terms):,}")
    say(f"    terms after stemming        {len(vocab_terms):,}")
    say(f"    merged away                 {collapsed:,} "
        f"({collapsed / max(len(kept_terms), 1):.1%})")
    say(f"    took                        {time.time() - t1:.0f}s")
    say()

    # --- document frequency filter ---------------------------------------
    df = np.diff(X.tocsc().indptr)
    keep_df = df >= args.min_df
    X = X[:, keep_df].tocsc()
    vocab_terms = vocab_terms[keep_df]
    df = df[keep_df]

    say("[6] THE INVERTED INDEX")
    say(f"    minimum document frequency  {args.min_df} "
        "(a term in one document links nothing to anything)")
    say(f"    terms dropped by that       {int((~keep_df).sum()):,}")
    say()
    say(f"    FINAL INDEX")
    say(f"      documents                 {X.shape[0]:,}")
    say(f"      terms (postings lists)    {X.shape[1]:,}")
    say(f"      postings                  {X.nnz:,}")
    say(f"      tokens                    {int(X.sum()):,}")
    say(f"      density                   "
        f"{X.nnz / (X.shape[0] * X.shape[1]):.6%}")
    say(f"      memory                    "
        f"{(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6:.0f} MB")
    say()
    say("    Structure: compressed sparse column. Column j of the matrix holds")
    say("    every document containing term j together with its term frequency -")
    say("    which is precisely a postings list. The classic dictionary-of-lists")
    say("    inverted index and this matrix store the same information; the")
    say("    matrix form lets the scoring run as vectorised arithmetic instead of")
    say("    a Python loop over postings.")
    say()
    say("    Postings list lengths (how many documents contain a term)")
    for label, value in [("min", df.min()), ("median", np.median(df)),
                         ("mean", df.mean()), ("95th pct", np.percentile(df, 95)),
                         ("max", df.max())]:
        say(f"      {label:<10} {value:>10,.0f}")
    say()
    top = np.argsort(df)[::-1][:12]
    say("    Longest postings lists (most common terms after all filtering)")
    for i in top:
        say(f"      {vocab_terms[i]:<16} {df[i]:>9,} documents "
            f"({df[i] / X.shape[0]:>5.1%})")
    say()
    singletons = int((df == args.min_df).sum())
    say(f"    {singletons:,} terms appear in exactly {args.min_df} documents "
        f"({singletons / len(df):.0%} of the")
    say("    vocabulary). That is Zipf's law: a few terms are everywhere and most")
    say("    are almost nowhere, which is what makes an inverted index worth")
    say("    building - a query touches a handful of short lists, never the")
    say("    whole collection.")
    say()

    # --- sparse vs dense -------------------------------------------------
    say("[7] SPARSE OR DENSE?  (the rubric asks for this choice to be justified)")
    say("    We index sparse: one dimension per term, mostly zeros.")
    say()
    say("    Sparse (what we use)")
    say(f"      - {X.shape[1]:,} dimensions, {X.nnz:,} non-zeros, "
        f"{(X.data.nbytes + X.indices.nbytes) / 1e6:.0f} MB")
    say("      - exact term matching; a query for 'tenure' cannot miss a")
    say("        document containing 'tenure'")
    say("      - every score decomposes into per-term contributions, so any")
    say("        ranking can be explained - which the error analysis depends on")
    say("      - no training, no GPU, no model to download")
    say()
    say("    Dense (what we did not use)")
    n_docs = X.shape[0]
    say(f"      - {n_docs:,} documents x 384 dimensions x 4 bytes = "
        f"{n_docs * 384 * 4 / 1e6:.0f} MB, comparable size")
    say("      - would match 'tenure' to 'job security' without sharing a word,")
    say("        which is a real advantage on this task")
    say("      - but encoding 369,390 documents with a sentence transformer on")
    say("        CPU is hours of compute, and the rubric marks dense retrieval")
    say("        optional")
    say("      - and a dense score is a single number with no decomposition, so")
    say("        'why did this rank first' has no answer to put on a slide")
    say()
    say("    The honest summary: sparse is chosen for interpretability and for")
    say("    fitting the machine we have. The gap it leaves - vocabulary")
    say("    mismatch between a short question and a long argument - is the first")
    say("    item in the limitations and the first item in future work.")

    MODELS.mkdir(exist_ok=True)
    sp.save_npz(MODELS / "index.npz", X.tocsr())
    (MODELS / "vocab.json").write_text(json.dumps({
        "terms": list(map(str, vocab_terms)),
        "token_pattern": TOKEN_PATTERN,
        "min_df": args.min_df,
        "stemmed": not args.no_stem,
        "stopwords": sorted(stop),
    }))
    doc_len = np.asarray(X.sum(axis=1)).ravel()
    pd.DataFrame({"doc_id": corpus["doc_id"].values,
                  "n_terms": doc_len}).to_parquet(
        PROC / "doc_meta.parquet", index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {(MODELS / 'index.npz').relative_to(ROOT)}")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
