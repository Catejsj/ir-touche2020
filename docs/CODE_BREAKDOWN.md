# The code, explained line by line

For the question "what did you actually run, and why".

## First, the honest answer to the obvious question

**There is no training code in this project. Nothing is fitted.**

| | is anything learned? |
|---|---|
| Boolean | **No.** Counts how many query terms a document contains |
| VSM (TF-IDF) | **No.** A formula over counts the index already holds |
| BM25 | **No.** A different formula over the same counts |

Search `scripts/` for `.fit(` and you find exactly one hit —
`CountVectorizer.fit_transform` in `preprocess_index.py` — and that is not
learning either. It is building a dictionary of which word is column number
what, and counting. No gradient, no loss function, no parameters adjusted to fit
data.

If a teacher asks where the model is, the answer is:

> "There isn't one, and that is the point of these three approaches. All the
> knowledge is in the index and in two formulas. The only numbers we did not
> compute from the collection are k1 and b, and we took those from the
> literature rather than fitting them."

That is a strength, not a gap. It means every score can be taken apart and
explained, which is what makes the error analysis possible. Compare with the NER
project, where the CRF really was trained and its 595,404 weights could only be
inspected, not derived.

**What IS computed from the data** — `retrieve.py::build_stats`, lines 119-141:

- `df` — how many documents contain each term
- `doc_len` — how many terms each document has
- `avg_len` — the collection average
- `idf`, `idf_bm25` — two different rarity weightings
- `doc_norm` — each document's TF-IDF vector length

Those are **derived statistics**, not learned parameters. Run it twice on the
same collection and you get identical numbers; there is no randomness and no
optimisation anywhere.

---

# Part 1 — Building the index (`preprocess_index.py`)

This is where almost all the work happens. Everything after it is arithmetic.

## Step 1 — count everything (line 118)

```python
raw_vec = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                          dtype=np.int32)
X_raw = raw_vec.fit_transform(text)
```

One call produces a matrix with **60,000 rows (documents) × 169,313 columns
(distinct words)**, where cell (i, j) is how many times word j appears in
document i. Runs in about 4 seconds because the loop is in C, not Python.

`dtype=np.int32` matters at this size — the default is int64 and would double
the memory for counts that never exceed a few thousand.

## Step 2 — drop stopwords (columns, not tokens)

```python
keep_mask = np.array([t not in stop for t in raw_vocab])
X = X_raw[:, keep_mask].tocsc()
```

Note what this does **not** do: it never touches the text. Removing a stopword
is deleting a *column* from the matrix. That is the whole operation, and it is
why 49.6% of tokens disappear in milliseconds.

## Step 3 — stemming, the one clever bit (lines 191-203)

The naive version is to stem every token as you read it. That is 110 million
calls to the Porter stemmer in Python, which takes hours. Instead:

```python
stems = np.array([stemmer.stem(t) for t in kept_terms])   # 169,104 calls
...
selector = sp.csr_matrix((ones, (arange(n_terms), col_of_term)),
                         shape=(n_terms, n_stems))
X = (X @ selector).tocsc()                                # one matmul
```

Stem the **vocabulary** once — 169,104 calls instead of 110 million — then build
a matrix that says "term 5 and term 12 both became stem 3" and multiply. The
multiplication sums the columns that collapsed together.

**The result is identical** to stemming every token. It takes 18 seconds. This
single trick is why the pipeline runs on a laptop, and it is the part of the
code I would point at if asked what I am pleased with.

## Step 4 — drop rare terms, save

```python
df = np.diff(X.tocsc().indptr)     # documents per term, for free
keep_df = df >= args.min_df        # min_df = 2
X = X[:, keep_df].tocsc()
sp.save_npz(MODELS / "index.npz", X.tocsr())
```

`np.diff(indptr)` is worth knowing: in a compressed-sparse-column matrix,
`indptr` already records where each column starts, so the difference between
consecutive entries **is** the number of documents containing that term. No
counting loop needed.

Final index: 60,000 × 50,045, 5,302,673 postings, **43 MB**.

## Why this matrix is an inverted index

The classic structure is a dictionary:

```
"tenur"   -> [(doc 41, 3 times), (doc 902, 1 time), ...]
"teacher" -> [(doc 7, 12 times), (doc 41, 63 times), ...]
```

A CSC matrix stores exactly that. Column *j* holds the row indices of the
documents containing term *j*, and the matching counts. Same information, same
lookup, but the scoring can be done with array arithmetic instead of a Python
loop over postings.

`postings()` in `retrieve.py` (line 74) is the proof — it *is* the dictionary
lookup, in three lines:

```python
start, end = X.indptr[col], X.indptr[col + 1]
return X.indices[start:end], X.data[start:end]
```

---

# Part 2 — The three scoring functions (`retrieve.py`)

All three have the same shape: start with an array of 60,000 zeros, walk the
postings list of each query term, add something to the documents on that list.
Only the "something" differs.

## Query processing first (line 64)

```python
def analyse(text, vocab, term_id, stemmer):
    tokens = re.findall(vocab["token_pattern"], text.lower())
    tokens = [t for t in tokens if t not in stop]
    tokens = [stemmer.stem(t) for t in tokens]
    return [term_id[t] for t in tokens if t in term_id]
```

**This is the most important function in the file**, because it is the one that
can silently ruin everything. If the query were cleaned differently from the
documents — a different pattern, a different stopword list, no stemming — terms
would simply never match and there would be no error message, just bad results.

It reads the settings from `vocab.json`, which `preprocess_index.py` wrote. The
two sides cannot drift apart because there is only one copy of the settings.

*"Should teachers get tenure?"* → `teacher`, `get`, `tenur`.

## Boolean (line 80)

```python
hits = np.zeros(n_docs, dtype=np.int32)
for col in set(cols):
    docs, _ = postings(X, col)
    hits[docs] += 1
return hits
```

Note `_` — the term frequencies are fetched and **thrown away**. Boolean does
not care how often a word appears, only whether it does. `set(cols)` means a
repeated query word counts once.

The result is the coordination level: how many distinct query terms this
document has.

## Vector Space (line 89)

```python
for col, q_tf in counts.items():
    docs, tf = postings(X, col)
    w_d = (1.0 + np.log(tf)) * idf[col]       # document weight
    w_q = (1.0 + np.log(q_tf)) * idf[col]     # query weight
    scores[docs] += w_d * w_q
scores[nonzero] /= doc_norm[nonzero]          # <- the cosine
```

Three ideas in four lines. `1 + log(tf)` damps repetition. `idf[col]` rewards
rare terms. And the last line is the cosine: **dividing by the document's own
vector length.**

That division is the whole model's weakness in one operation. A very short
document has a tiny `doc_norm`, so dividing by it makes the score large. That is
what Person 5's measurement is about, and it is *this line* they are talking
about.

## BM25 (line 109)

```python
for col in set(cols):
    docs, tf = postings(X, col)
    denom = tf + k1 * (1.0 - b + b * doc_len[docs] / avg_len)
    scores[docs] += idf_bm25[col] * (tf * (k1 + 1.0)) / denom
```

One line contains both fixes.

- **Saturation**: `tf` appears on top *and* bottom. As tf grows, the fraction
  approaches `k1 + 1` and stops. It cannot run away.
- **Length**: `doc_len[docs] / avg_len` compares this document to the collection
  average. `b` controls how much that matters — at `b = 0` the term vanishes and
  length is ignored entirely; at `b = 1` it is fully applied.

Compare with the VSM version: there, length is a *division applied afterwards*.
Here it is *inside the denominator*, mixed with term frequency. That difference
is the whole reason the two models rank differently.

## The two idf formulas are genuinely different (lines 126, 131)

```python
idf      = np.log(n_docs / np.maximum(df, 1)) + 1.0
idf_bm25 = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
```

Not a cosmetic variant. The first is always positive. The second comes from the
probabilistic derivation and can go **negative** for a term appearing in more
than half the collection — which correctly says "this term is evidence against
relevance". The `0.5` terms keep it defined when `df` is 0 or N.

## Getting the top 100 without sorting 60,000 things (lines 157-160)

```python
hit = np.flatnonzero(scores > 0)
if len(hit) > top_k:
    part = hit[np.argpartition(-scores[hit], top_k)[:top_k]]
order = part[np.argsort(-scores[part], kind="stable")]
```

`argpartition` finds the top 100 without ordering the rest — much cheaper than a
full sort. Then only those 100 are sorted properly. `kind="stable"` means ties
break by document id rather than randomly, so the run is reproducible.

---

# Part 3 — Scoring the results (`evaluate.py`)

The only place a decision about *relevance* is made, in one line (line 50):

```python
rel_ids = {d for d, lv in level.items() if lv >= threshold}
```

`threshold = 1` means level 1 and level 2 both count as relevant. Change it to 2
and the strict variant is produced — which is exactly how the sensitivity check
in section 4 of the report is computed, with no other change to the code.

**nDCG is the only metric that uses the grade rather than the yes/no:**

```python
gains = np.array([2 ** level.get(d, 0) - 1 for d in ranked[:k]])
```

`2^level - 1` gives 0, 1, 3 for levels 0, 1, 2 — so a highly relevant document
is worth three times a relevant one. Precision and recall cannot express that,
because they were handed a set, not grades.

---

# The five things to remember

1. **Nothing is trained.** The knowledge is in the index plus two formulas.
2. **The matrix IS the inverted index** — a column is a postings list.
3. **Stemming runs on the vocabulary**, not the tokens. 169,104 calls, not 110
   million.
4. **`analyse()` must match the indexing pipeline exactly** or nothing matches
   and nothing warns you.
5. **The cosine's division by `doc_norm` is the line** that makes VSM prefer
   tiny documents.
