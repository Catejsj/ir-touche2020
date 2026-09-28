# Information Retrieval — Boolean, TF-IDF and BM25 on argument search

Text Mining project. Three classic retrieval models built over one inverted
index and scored by one evaluation function, on a collection where the
relevance judgments are **graded** and include documents a human explicitly
rejected.

**Dataset:** [Webis-Touché 2020](https://huggingface.co/datasets/BeIR/webis-touche2020),
Task 1 (argument retrieval), distributed as part of BEIR. Qrels:
[webis-touche2020-qrels](https://huggingface.co/datasets/BeIR/webis-touche2020-qrels).
Documents are arguments from the args.me crawl of five public debate portals.
Licence CC BY-SA 4.0.

## Research question

**Does BM25 actually beat Boolean matching and TF-IDF at finding good debate
arguments?**

Yes — nDCG@10 0.525 against 0.289 and 0.183. *Where* and *why* each model fails
is the explanation behind that answer, not a second question.

## Results

49 queries, 60,000 documents, relevance threshold level >= 1:

| model | P@10 | R@10 | F1@10 | MRR | MAP | **nDCG@10** |
|---|---|---|---|---|---|---|
| Boolean | 0.292 | 0.190 | 0.219 | 0.491 | 0.194 | 0.289 |
| VSM (TF-IDF + cosine) | 0.210 | 0.134 | 0.158 | 0.413 | 0.149 | 0.183 |
| **BM25** | **0.516** | **0.324** | **0.380** | **0.753** | **0.408** | **0.525** |

**The finding worth presenting:** the vector space model scores *below* an
unweighted Boolean match. The cause is measurable — the median document VSM puts
in a top 10 is **16 words** long, against 333 for BM25 and 127 for the
collection. Cosine similarity divides by document length, so a three-word
argument mentioning the query term once gets a near-perfect score. BM25's length
normalisation is the difference, and this is a concrete demonstration of the
thing BM25 exists to fix.

## Setup

```bash
python3.11 -m venv .venv
./.venv/bin/pip install pandas pyarrow numpy scipy scikit-learn nltk matplotlib
```

Download the data (about 345 MB):

```bash
mkdir -p data/raw && curl -sL -o data/raw/corpus.parquet "https://huggingface.co/datasets/BeIR/webis-touche2020/resolve/main/corpus/corpus-00000-of-00001.parquet" && curl -sL -o data/raw/queries.parquet "https://huggingface.co/datasets/BeIR/webis-touche2020/resolve/main/queries/queries-00000-of-00001.parquet" && curl -sL -o data/raw/qrels_test.tsv "https://huggingface.co/datasets/BeIR/webis-touche2020-qrels/resolve/main/test.tsv"
```

## Pipeline — run in this order

| # | Command | Produces |
|---|---|---|
| 1 | `./.venv/bin/python scripts/build_corpus.py` | pooled subset, `reports/01_corpus.txt` |
| 2 | `./.venv/bin/python scripts/preprocess_index.py` | `models/index.npz`, `reports/02_preprocessing_index.txt` |
| 3 | `./.venv/bin/python scripts/retrieve.py --model all` | `data/processed/run_*.parquet`, `reports/03_retrieval.txt` |
| 4 | `./.venv/bin/python scripts/evaluate.py` | `reports/04_evaluation.txt` |
| 5 | `./.venv/bin/python scripts/make_figures.py` | `figures/fig1-5*.png` (results) |
| 6 | `cd scripts && ../.venv/bin/python make_figures_person4.py` | `figures/fig6-7*.png` (how the models work) |

The whole pipeline runs in about two minutes. Add `--sweep` to step 3 for the
BM25 parameter sweep.

## Design decisions worth knowing

**We search a pooled subset of 60,000 documents, not all 369,390.** Indexing the
full collection exhausted memory and the process was killed by the kernel. The
subset keeps **every judged document** plus a random sample of the rest, so no
query loses a relevant document and no measure changes definition — but the task
is easier and the numbers are an upper bound. `--max-docs 0` restores the full
collection on a machine with more memory.

**Stemming is applied to the vocabulary, not to every token.** Porter-stemming
110 million tokens in Python takes hours; stemming the 169,313 distinct terms
once and merging the columns that collapse together is identical and takes
seconds.

**The index is a sparse matrix, and that matrix *is* the inverted index.**
Column *j* of a CSC matrix holds exactly the documents containing term *j* with
their frequencies — the definition of a postings list. One structure serves all
three retrieval models.

**Boolean uses OR, ranked by coordination level.** With 2-3 term queries, AND
returns almost nothing. Real Boolean retrieval returns an unordered set, so
ranking it at all is a concession we state rather than hide.

**Relevance threshold is level >= 1, stated explicitly**, with the strict
variant (level == 2 only) reported alongside as a sensitivity check. The model
ordering does not change.

**BM25 parameters are the published defaults (k1 = 1.2, b = 0.75), not tuned.**
Touché has 49 queries and no validation split, so tuning on it would be fitting
the test set. The sweep is reported for transparency only.

## Layout

```
data/raw/          the three files as downloaded
data/processed/    pooled corpus, queries, qrels, run files, per-query metrics
models/            the inverted index and its vocabulary
reports/           numbered plain-text reports, one per stage
figures/           the five slide figures
docs/              presentation structure and speaker material
scripts/           the pipeline
```

## Documents

- **`docs/PRESENTATION_STRUCTURE.md`** — the rubric's 7 criteria mapped onto 6
  speakers, each anchored to the script, report and figure that back it. This is
  the document people pick their part from.
- **`docs/PARAMETERS_AND_WHY.md`** — every parameter in the project written as a
  spoken answer, for question prep.
- **`docs/SLIDES_PERSON4_RETRIEVAL.md`** — the retrieval-models part in full:
  slide text, spoken script, both figures explained, and Q&A.
