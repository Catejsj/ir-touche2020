# Every parameter, and why — question prep

Written as spoken answers, not documentation. Each row is something a teacher
can ask and a reply you can give out loud.

The safe general answer when you are stuck:

> "We did not tune that one — it is the standard value, and we checked the
> result was not sensitive to it."

That is a good answer. Claiming everything was optimised is a bad answer,
because the follow-up is "show me the tuning".

---

# 1 — Corpus (`scripts/build_corpus.py`)

| Setting | Value | Why |
|---|---|---|
| Collection | Webis-Touché 2020, Task 1 | Graded relevance (0/1/2) and explicit non-relevant judgments — both rare, and both used in the evaluation |
| `MIN_DOC_WORDS` | **3** | Below that a document is empty or a single punctuation mark; it cannot be retrieved meaningfully and it distorts the length statistics |
| `--max-docs` | **60,000** | Memory. See below |
| `SEED` | 42 | So the distractor sample is reproducible |
| Fields indexed | title + text | An args.me title is the debate motion, often the best single sentence in the document |

**Why only 60,000 of 369,390 documents?**
Straight answer: indexing the full collection exhausted memory and the process
was killed by the kernel. We search a **pooled subset** instead — every judged
document (2,095) plus 57,905 random unjudged distractors. This is standard
practice when a full collection is out of reach.

**Doesn't that invalidate the evaluation?**
No, and it is worth being precise about why. A metric can only be computed over
documents a human judged, and every one of those is still in the collection. No
query loses a relevant document and no measure changes definition. What *does*
change is difficulty: six times fewer distractors means six times fewer chances
to rank an irrelevant document above a relevant one. So our scores are an
**upper bound**, and they are not comparable to published Touché numbers. Both
statements are on the slide.

**Why not just use a smaller dataset instead?**
Because the judgments are the valuable part. Touché's graded scale and its
explicitly-rejected documents are what make our error analysis possible; a
smaller collection with binary judgments would have cost us the best section of
the project.

---

# 2 — Preprocessing and indexing (`scripts/preprocess_index.py`)

| Setting | Value | Why |
|---|---|---|
| Token pattern | `\b[a-z][a-z'-]+\b` | Letters, apostrophes, hyphens. Keeps *don't* and *well-being* whole; drops *1990* and *$* |
| Minimum token length | 2 characters | Single letters carry no retrieval signal |
| Tokenizer | scikit-learn regex, **not** NLTK | A C-level regex over 110 million words; NLTK's Treebank tokenizer is a Python loop |
| Stopwords | NLTK English (198) + our 21 | A third list would not help; these remove 49.6% of tokens |
| Stemming | **Porter**, on the vocabulary | See below |
| `MIN_DF` | **2** | A term in one document cannot connect a query to anything |
| Index format | CSC sparse matrix | A column *is* a postings list |

**Why a regex tokenizer when the NER project used NLTK's?**
Because the use is different. In NER a token is labelled and shown to a user, so
the split has to be linguistically right. In retrieval a term is only ever a
dictionary key — never displayed, never labelled — so it only has to be
*consistent* between documents and queries. It is: `retrieve.py::analyse` runs
the same analyser on both sides.

**Why stem for retrieval when the NER project deliberately did not?**
Same reason inverted. The output of NER is a span of the original text, so
`privat` would be nonsense. The output of retrieval is a ranked list of
documents — nobody ever sees a stem. A user searching *teachers* should match a
document arguing about *teacher* and *teaching*, and stemming is what makes
those one key.

**Why Porter rather than WordNet lemmatization?**
The report shows it on real terms: *privatized* and *privatization* both become
`privat` under Porter but stay separate under the lemmatizer. The lemmatizer is
linguistically correct and useless for matching. Porter's non-words (`polici`,
`privat`) never reach a screen.

**You stemmed 110 million tokens in 18 seconds?**
No — we stemmed the **vocabulary**. 169,104 distinct terms stemmed once, then
the matrix columns that collapse to the same stem are summed with one sparse
multiply. The result is identical to stemming every token and it is roughly 400
times less work. This is the single optimisation that makes the pipeline
runnable on a laptop.

**Half your tokens are stopwords. Isn't removing them risky?**
For this corpus, no, and the domain list is the interesting half. Every document
in a debate corpus contains *argument*, *claim*, *should* and *believe* — they
behave exactly like stopwords even though no standard list holds them. The cost
we accept: *"Should teachers get tenure?"* retrieves on two terms, `teacher` and
`tenur`. On very short queries that is a thin basis, and it is in the
limitations.

**In what sense is a matrix an inverted index?**
Column *j* of a compressed-sparse-column matrix holds exactly the documents
containing term *j*, with their term frequencies. That is the definition of a
postings list. The classic dictionary-of-lists and this matrix store the same
information; the matrix form lets scoring run as vectorised arithmetic instead
of a Python loop, and one structure serves all three retrieval models without
being rebuilt.

---

# 3 — Retrieval (`scripts/retrieve.py`)

| Setting | Value | Why |
|---|---|---|
| Boolean operator | **OR** | AND over 60,000 documents with a 2-4 term query returns a handful or nothing |
| Boolean ranking | coordination level | The minimum needed to compare an unranked model against ranked ones |
| VSM document weight | `(1 + log tf) × idf` | Sub-linear tf; idf rewards rare terms |
| VSM similarity | cosine | Divides by the document's own vector length |
| BM25 `k1` | **1.2** | How fast term frequency saturates |
| BM25 `b` | **0.75** | How hard long documents are penalised |
| BM25 idf | `log(1 + (N - df + 0.5)/(df + 0.5))` | The probabilistic form, not the VSM one |
| `TOP_K` | 100 | Deep enough to measure recall honestly; cuts at 5/10/20 |

**Why OR and not AND? Isn't AND what Boolean retrieval means?**
Both are Boolean. With queries of 2-4 terms after stopword removal, AND returns
almost nothing, and a system that returns nothing cannot be evaluated at all. We
use OR and rank by coordination level — how many distinct query terms the
document contains.

**Then it isn't really Boolean retrieval any more.**
Correct, and we say so rather than pretend otherwise. Real Boolean retrieval
returns an *unordered set*; every ranking metric assumes an order. Ranking it by
coordination level is a concession made so the comparison can happen, and it is
listed in the limitations.

**How did you pick k1 and b?**
We did not. They are the published defaults from Robertson and Zaragoza. Touché
has 49 queries and no validation split, so choosing them by test score would be
fitting the test set — we would be reporting how well we tuned, not how well
BM25 works. `--sweep` prints the grid for transparency only.

**What is the difference between VSM's idf and BM25's?**
Not cosmetic — they come from different derivations. VSM uses
`log(N/df) + 1`, always positive. BM25's can go **negative** for a term in more
than half the collection, which correctly says "this term is evidence against
relevance". The +0.5 smoothing keeps it finite when df is 0 or N.

**Why does BM25 beat VSM when both weight terms?**
Length handling. VSM's cosine divides length away completely; BM25 compares each
document to the collection average and saturates term frequency. On a corpus
running from 3 to 16,162 words that is the whole difference — see section 5.

---

# 4 — Evaluation (`scripts/evaluate.py`)

| Setting | Value | Why |
|---|---|---|
| Relevance threshold | **level ≥ 1** | Stated explicitly because it is a choice |
| Strict variant | level == 2 only | Reported as a sensitivity check |
| Unjudged documents | counted **not relevant** | Standard TREC convention |
| Cutoffs | 5, 10, 20 | 10 is the standard; 5 and 20 show the trend |
| Run depth | 100 | For recall |
| Bootstrap | 2,000 resamples | Confidence intervals over 49 queries |
| Matching | exact document id | No partial credit, no text overlap |

**How is relevance judged?** — the rubric asks this directly.
Level 2 (highly relevant) and level 1 (relevant) count as relevant. Level 0 is a
document a judge looked at and **rejected**. Unjudged documents were never shown
to a judge and are counted not relevant, which is the TREC convention. Matching
is by document id — a returned document either *is* the judged document or it is
not.

**Why report precision, recall and F1 if you say they are limited?**
Because the rubric asks for them and they are the standard first answer. The
limitation is real though: a query has 19 relevant documents on average, so a
*perfect* top-10 reaches recall 0.53 and F1 inherits that ceiling. We say that
before someone reads a low recall as a failure.

**Why nDCG as the headline?**
It is the only measure here that can see the difference between a level-2 and a
level-1 document. Precision, recall and F1 need a yes/no and throw the grading
away. Since graded relevance is why we chose this collection, the measure that
uses it should carry the argument.

**Isn't treating unjudged documents as wrong unfair?**
It is the standard convention and it is also our largest caveat. Between 35% and
62% of every top 10 was never seen by a judge. Those are scored as misses, so
every precision number is a **lower bound** — some of those documents may well
be relevant. Section 5 of the evaluation report breaks this out rather than
burying it.

**49 queries is very few.**
Yes. That is why we report bootstrap confidence intervals rather than bare
means: Boolean [0.233, 0.341] and VSM [0.133, 0.234] overlap, so those two are
not reliably separable. BM25's interval [0.448, 0.588] does not overlap either,
which is the only ordering claim we make with confidence.

---

# 5 — The finding, and how to defend it

**BM25 0.525, Boolean 0.289, VSM 0.183 (nDCG@10).**

**Why does an unweighted Boolean match beat TF-IDF?**
Because cosine similarity has a length bias on this corpus, and we measured it:

| | median words in top 10 | % under 50 words |
|---|---|---|
| Boolean | 843 | 8% |
| **VSM** | **16** | **79%** |
| BM25 | 333 | 15% |
| the collection | 127 | 31% |

Cosine divides by the document's vector length, so a three-word argument
containing *tenure* once has a vector pointing almost exactly at the query. It
scores near-perfectly on a document that says nothing. **79% of what VSM returns
is under 50 words.**

**And that is not a bug in your implementation?**
No — it is the textbook behaviour of cosine normalisation meeting a corpus with
extreme length variance (3 to 16,162 words). BM25 was designed for exactly this:
it does not normalise length away, it compares each document against the
collection average, and it saturates term frequency. Our result is a measured
demonstration of the problem BM25 exists to solve.

**Does the conclusion depend on your relevance threshold?**
No. Re-scored with only level-2 documents counting as relevant, the ordering is
unchanged — BM25 nDCG@10 moves 0.525 → 0.527. That check is in the report
precisely so the answer to this question is "we tested it".

---

# 6 — The numbers to have memorised

| | |
|---|---|
| **60,000** | documents searched (pooled subset of 369,390) |
| **49** | queries; **2,210** relevance judgments; **19** relevant per query |
| **50,045** | terms in the index; 5,302,673 postings; 43 MB |
| **0.525 / 0.289 / 0.183** | nDCG@10 for BM25 / Boolean / VSM |
| **16 words** | median document VSM puts in a top 10 — the headline |
| **49.6%** | of tokens removed as stopwords |

---

# 7 — If you genuinely do not know

> "We did not set that one — it is the library default. I can tell you what it
> controls and why it did not change our result."

or

> "That is in the repo, section 3 of the indexing report. I can show you after."

Both beat guessing. A wrong number said confidently is the only answer that
actually costs marks.
