# Presentation structure

Maps the rubric's 7 criteria onto 7 parts, each backed by a specific
script/report/figure in this repo. Numbers are from the generated reports
(`reports/01_corpus.txt` … `reports/04_evaluation.txt`) — re-check them if the
pipeline is re-run before the talk, since a re-run can shift them slightly.

Total: ~20-25 min, 3-4 min per person + buffer. Order follows the pipeline's
data flow: corpus → indexing → retrieval models → evaluation → limitations.

---

## Person 1 — Problem Definition & Motivation (rubric: 5 pts)

No script — pure framing, sets up the vocabulary the rest of the talk depends
on.

- **The task**: given a query, return a ranked list of documents, best first.
  Unlike classification there is no label per document — relevance is a
  relationship between a query and a document, and it is judged by people.
- **RQ1**: how do the three classic retrieval models — Boolean, vector space,
  probabilistic — compare on one collection under one scoring function?
- **RQ2**: where does each one fail, and why? Answered in Person 5's section,
  and it is the question that makes this more than a leaderboard.
- **Why argument retrieval**: the query is a controversial question a person
  might really type (*"Should teachers get tenure?"*) and a relevant document is
  one that helps them take a side. Relevance is therefore about argument quality
  and stance, not topical aboutness — which is why the judges used a 0/1/2 scale
  instead of yes/no. That grading is what makes Person 5's evaluation richer
  than a binary one.
- **Unit of analysis**: one argument = one document; one query = one ranked list;
  one (query, document) pair = one relevance judgment.
- **Scope**: sparse lexical retrieval only. Dense/neural retrieval is marked
  optional in the rubric and is out of scope for reasons Person 3 gives and
  Person 6 revisits.

**Slides:** 1 — the task, the two RQs, one example query.

---

## Person 2 — Corpus Collection & Legal/Ethical Compliance (rubric: 10 pts)

**Source:** `scripts/build_corpus.py` → `reports/01_corpus.txt`.

**Collection**: Webis-Touché 2020, Task 1, distributed via BEIR
(`huggingface.co/datasets/BeIR/webis-touche2020`). Documents are the args.me
crawl of five public debate portals — debatewise, idebate, debatepedia,
debate.org, and the Canadian parliament record.

**Cleaning pipeline** (drop counts):

- documents as distributed: 382,545
- under 3 words: -13,155
- duplicate id: -0
- kept: 369,390 (96.6%)

**The pooled subset — say this before anyone asks.** Indexing all 369,390
documents exhausted memory on the machine available to us and the process was
killed by the kernel. We therefore search a **pooled subset**: every judged
document (2,095, carrying 2,210 judgments) plus 57,905 random unjudged
distractors, 60,000 total, seed 42. No query loses a relevant document and no
measure changes definition, so the evaluation stays valid — but the task is
easier and our scores are an **upper bound**, not comparable to published Touché
numbers. `--max-docs 0` restores the full collection.

**Corpus description**:

- 60,000 documents searched, 18,008,977 words
- document length: median 127 words, mean 300, min 3, max 16,162 — heavily
  right-skewed, and this is the single fact that decides Person 5's headline
  result
- 49 queries, mean 6.6 words, e.g. *"Is vaping with e-cigarettes safe?"*

**Relevance judgments** — the reason this collection was chosen:

| level | judgments | share | meaning |
|---|---|---|---|
| 0 | 1,278 | 57.8% | judged NOT relevant |
| 1 | 296 | 13.4% | relevant |
| 2 | 636 | 28.8% | highly relevant |

45.1 documents judged per query on average; 19.0 of them relevant. Two
consequences, both of which shape Person 5's section: relevance is **graded**,
so nDCG can see a difference precision cannot; and 1,278 documents were
**explicitly rejected by a human**, so a failure can be a judged "no" rather
than merely an absence.

**Legal/ethical compliance** (10 of 100 rubric points):

- Licence: CC BY-SA 4.0 — reuse, modification and redistribution permitted with
  attribution and share-alike. Touché and BEIR both cited.
- Provenance: we use the published research corpus, not our own crawl, so no
  portal's terms of service are engaged.
- Personal data: arguments were posted publicly under usernames. The BEIR
  distribution carries **no author, no username, no timestamp** — only id, title
  and text. There is nothing to anonymise and we add nothing back.
- We do not identify authors, link arguments to people, or aggregate anyone's
  positions across documents.
- Content: debate material on abortion, gun control, religion, immigration.
  Some arguments are offensive. They are retrieved and scored, never endorsed;
  slides carry a content note.
- The system ranks arguments by match to a query. It does **not** judge whether
  an argument is true and must not be presented as doing so.

**Slides:** 2 — source, size and the pooled-subset caveat; the graded judgment
table plus the legal/ethical bullets.

---

## Person 3 — Text Preprocessing & Indexing (rubric: 25 pts)

**Source:** `scripts/preprocess_index.py` → `reports/02_preprocessing_index.txt`.

1. **Normalization** — lowercase, Unicode NFKC, token pattern
   `\b[a-z][a-z'-]+\b`. Letters, apostrophes and hyphens only, so *don't* and
   *well-being* survive as one token while *1990* and *$* are dropped. Numbers
   are discarded deliberately: in this corpus they are years and vote counts
   inside arguments, and a 6-word debate question never contains one.
2. **Tokenization** — scikit-learn's regex tokenizer, **not** `nltk.word_tokenize`.
   This is a deliberate reversal of what the NER project did. The Treebank
   tokenizer is more careful but it is a Python loop over 110 million words. For
   retrieval a term is only ever a dictionary key — never displayed, never
   labelled — so the split only has to be *consistent* between documents and
   queries, and it is: the same analyser runs on both sides
   (`retrieve.py::analyse`).
3. **Stopword removal** — NLTK English (198) plus a 21-word debate list
   (*argument*, *debate*, *claim*, *believe*, *should*, *con*, *pro*). Removed
   209 terms, which is **49.6% of all tokens**. The domain half is the
   interesting part: every document in a debate corpus contains "argument" and
   "should", so they behave exactly like stopwords even though no standard list
   holds them. Accepted cost: *"Should teachers get tenure?"* loses two of its
   words and retrieval rests on `teacher` + `tenur`.
4. **Stemming** — Porter, applied to the **vocabulary** rather than to every
   token: 169,104 distinct terms stemmed once, then the columns that collapse
   together are merged with one sparse matrix multiply. Identical result, 18
   seconds instead of hours. 38,287 terms merged away (22.6%).
   Porter over WordNet lemmatization, and the report shows why:
   *privatized* and *privatization* both become `privat` under Porter but stay
   apart under the lemmatizer — linguistically correct, useless for matching.
   Porter's non-words (`polici`, `privat`) are never displayed.
5. **The inverted index** — `min_df = 2` drops 80,772 terms that appear in a
   single document. Final index: **60,000 documents x 50,045 terms, 5,302,673
   postings, 43 MB**, density 0.18%.
   Structure: compressed sparse column. **Column j holds every document
   containing term j with its term frequency — that is a postings list.** The
   classic dictionary-of-lists index and this matrix store the same information;
   the matrix form lets scoring run as vectorised arithmetic rather than a Python
   loop, and one structure serves all three of Person 4's models.
   Postings lengths: median 4, mean 106, max 22,608 (`one`). Zipf in one line —
   a query touches a handful of short lists, never the whole collection, which
   is the entire reason an inverted index is worth building.
6. **Sparse vs dense, justified** (the rubric asks for this explicitly):
   sparse gives exact term matching, a score that decomposes into per-term
   contributions — which Person 5's error analysis depends on — and no GPU.
   Dense would handle the vocabulary mismatch that Person 5 identifies as the
   main failure mode, but encoding 60,000 documents on CPU is hours, dense
   retrieval is optional in the rubric, and a dense score is one number with no
   explanation to put on a slide. The gap it leaves is Person 6's first future
   work item.

**Slides:** 2 — the five-stage pipeline with the numbers each stage moved;
the index structure with the postings-list picture and the sparse/dense
justification.

---

## Person 4 — Retrieval Model Design (rubric: 25 pts)

**Source:** `scripts/retrieve.py` → `reports/03_retrieval.txt`.

Three of the four approaches the rubric lists; the fourth (dense) is optional
and out of scope per Person 3.

**Query processing first.** A query runs through the analyser the documents went
through — same lowercase, same pattern, same stopwords, same stemmer. Any
mismatch and terms silently fail to meet. After analysis queries are **4.0 terms
on average** (min 2, max 8) and **no query is left empty**.

**Boolean** — set membership, no weighting. We use **OR, not AND**: with 2-4
term queries, AND over 60,000 documents returns a handful or nothing, and a
system returning nothing cannot be evaluated. To compare it against ranked
models we order by **coordination level** — how many distinct query terms are
present. State this as the concession it is: real Boolean retrieval returns an
*unordered set*, and every ranking metric assumes an order.

**Vector Space Model** — document weight `(1 + log tf) x idf`, query weight the
same, similarity by **cosine**. The log dampens term frequency, idf rewards rare
terms, and the cosine divides by the document's own vector length so a long
document has no built-in advantage. That last property is exactly what goes
wrong — Person 5 has the measurement.

**BM25** — `k1 = 1.2`, `b = 0.75`,
`idf = log(1 + (N - df + 0.5) / (df + 0.5))`. Term frequency **saturates** (the
tenth mention of *tenure* adds far less than the second) and length is corrected
**against the collection average** rather than normalised away.

**Parameters are the published defaults and were NOT tuned.** Touché has 49
queries and no validation split, so choosing k1 and b by test score would be
fitting the test set. `--sweep` reports the grid for transparency only. Saying
this out loud is worth marks; quietly reporting a tuned best would cost them.

Runs are stored 100 deep for all 49 queries; evaluation cuts at 5, 10 and 20.
All three models score in under 0.05 seconds per run.

**Slides:** 2 — query processing plus Boolean and VSM; BM25 with the
saturation/length-normalisation contrast and the "we did not tune" statement.

---

## Person 5 — Retrieval Evaluation & Interpretation (rubric: 25 pts)

**Source:** `scripts/evaluate.py` → `reports/04_evaluation.txt`;
`figures/fig1`-`fig5`.

**How relevance is judged — state this first, the rubric asks for it
explicitly:**

| | |
|---|---|
| level 2 | highly relevant → **counted relevant** |
| level 1 | relevant → **counted relevant** |
| level 0 | judged NOT relevant → counted not relevant |
| unjudged | never seen by a judge → counted not relevant (TREC convention) |

Binary threshold is **level >= 1**, and it is a choice, not a given. Matching is
by document id — exact, no partial credit.

**Set-based measures** (`fig2`):

| model | P@10 | R@10 | F1@10 |
|---|---|---|---|
| Boolean | 0.292 | 0.190 | 0.219 |
| VSM | 0.210 | 0.134 | 0.158 |
| **BM25** | **0.516** | **0.324** | **0.380** |

Recall is low **by construction, not by failure**: a query has 19 relevant
documents on average, so a perfect top-10 reaches recall 0.53 and F1 inherits
that ceiling. Say this before someone reads it as a bad result.

**Ranking measures** (`fig1`):

| model | Hit@10 | MRR | MAP | **nDCG@10** | R@100 |
|---|---|---|---|---|---|
| Boolean | 0.898 | 0.491 | 0.194 | 0.289 | 0.550 |
| VSM | 0.776 | 0.413 | 0.149 | 0.183 | 0.550 |
| **BM25** | **0.980** | **0.753** | **0.408** | **0.525** | **0.762** |

95% bootstrap CIs on nDCG@10 over 49 queries: Boolean [0.233, 0.341], VSM
[0.133, 0.234], BM25 [0.448, 0.588]. Wide, because 49 queries is few — report
them rather than implying more precision than exists.

**Sensitivity check**: re-scored with only level 2 counting as relevant, the
ordering is unchanged (BM25 nDCG@10 0.527 vs 0.525). The conclusion does not
depend on where the threshold sits.

**THE HEADLINE FINDING — why VSM loses to unweighted Boolean** (`fig4`):

| | median words in top 10 | % under 50 words |
|---|---|---|
| Boolean | 843 | 8% |
| **VSM** | **16** | **79%** |
| BM25 | 333 | 15% |
| the collection | 127 | 31% |

Cosine similarity divides by document length, so a three-word argument
containing *tenure* once has a vector pointing almost exactly at the query and
scores near-perfectly. **79% of what VSM returns is under 50 words.** BM25 does
not normalise length away — it compares each document to the collection average
and saturates term frequency — and on a corpus running from 3 to 16,162 words
that difference decides the ranking. This is a concrete, measured demonstration
of the thing BM25 exists to fix, and it is the best thing in the project.

**Query-level error analysis** (`fig3`) — every document in a top 10 is one of
three things, and this collection can tell them apart:

| model | relevant | judged WRONG | unjudged |
|---|---|---|---|
| Boolean | 143 (29%) | 41 (8%) | 306 (62%) |
| VSM | 103 (21%) | 115 (23%) | 272 (56%) |
| BM25 | 253 (52%) | 65 (13%) | 172 (35%) |

"Judged wrong" is the honest failure count — a human looked and said no.
"Unjudged" is the uncertainty, and it is the main caveat on every precision
number above.

**A failure looked at closely.** BM25 scores nDCG@10 = **0.000** on *"Should
abortion be legal?"* — its top 5 are topically perfect (*"Abortion should be
made legal."*) but four are unjudged and one was explicitly rejected, while 21
relevant documents sit outside the top 100. Per-query spread (`fig5`): BM25
ranges from 0.000 to 0.889 across the 49 queries. Averages hide this.

**Slides:** 3 — the relevance definition plus `fig1`; `fig2` and the recall
ceiling; `fig4` as the headline finding, with `fig3` and the worked failure.

---

## Person 6 — Limitation and Future Studies (rubric: 5 pts)

**Limitations** — every one is specific to this project, not generic:

- **We searched 60,000 of 369,390 documents** (Person 2). A memory bound, not a
  design choice. Six times fewer distractors means six times fewer chances to
  rank an irrelevant document above a relevant one, so every number is an upper
  bound and none is comparable to published Touché results.
- **Unjudged documents are scored as wrong.** 35-62% of every top 10 was never
  seen by a judge (Person 5). Standard TREC practice, but it means precision is
  a lower bound and the true figure is unknowable without more judging.
- **49 queries is few.** The bootstrap intervals overlap for Boolean and VSM,
  and a single query swings the mean noticeably.
- **Vocabulary mismatch is unaddressed.** Sparse retrieval can only match terms
  it shares. The worst failures are queries whose relevant arguments use
  different words, and no amount of tuning fixes that.
- **Boolean was ranked to be measurable** (Person 4). Coordination level is not
  part of the Boolean model; a pure Boolean system returns an unordered set and
  is not really comparable to ranked models at all.
- **Stopword removal cost real query terms.** *"Should teachers get tenure?"*
  retrieves on two terms. On shorter queries that is a thin basis.
- **One domain, one language.** Debate portals in English. Nothing here
  transfers to a corpus without a strong lexical overlap between query and
  document.

**Future studies**, paired with the limitation each addresses:

- Re-run at full scale (`--max-docs 0`) on a machine with more memory and report
  how far the scores fall — directly measures the subset's effect.
- **Dense retrieval** (sentence embeddings) as the fourth approach the rubric
  marks optional, aimed squarely at the vocabulary mismatch.
- **Hybrid BM25 + dense**, since the two fail differently — the cheapest real
  gain available.
- **Query expansion / pseudo-relevance feedback**: take the top k BM25
  documents, extract their strongest terms, re-run. Recovers some of what
  stopword removal cost, with no new model.
- Judge a sample of the unjudged top-10 documents ourselves, with two annotators
  and a kappa, to put a bound on how much the unjudged column hides.

**Slides:** 1 — limitations and future work as paired lines, so each limitation
visibly has an answer.

---

## Person 7 — Presentation & Report Clarity (rubric: 5 pts)

Owns the deck, the references and the closing, and cross-checks the other six
parts for consistency.

**Before the talk:**

- One template — same font, same colours, same title position on every slide.
- Collect bullets from this document. Bullets only; nobody pastes a paragraph.
- Check every chart is readable from the back of the room. `fig4` is the one
  that matters most — make sure the "16" is legible.
- One timed rehearsal; cut whatever runs long.

**Consistency checks across parts:**

- Terminology: a **document** is one argument; a **query** is one debate
  question; a **judgment** is one (query, document, level) row; **relevant**
  always means level >= 1 unless explicitly stated otherwise.
- Numbers that must match everywhere they appear: 60,000 documents, 49 queries,
  2,210 judgments, 19 relevant per query.
- Figure ownership: `fig1`-`fig5` all belong to Person 5. Nobody else shows a
  chart, so nothing is duplicated.
- Every number traces to a report file and section — this document's citations
  are the map for that check.

**Closing slide:**

- 49 debate questions, 60,000 arguments, three classic retrieval models, one
  index and one scorer.
- **BM25 wins on every measure — nDCG@10 0.525 against 0.289 and 0.183.**
- The reason is length normalisation, and we measured it: VSM's median returned
  document is 16 words.
- Honest limits: a pooled subset, unjudged documents counted as wrong, 49
  queries.

**References slide:**

- Bondarenko et al. (2020), *Overview of Touché 2020: Argument Retrieval*, CLEF
- Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models*, NeurIPS
- Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and
  Beyond*
- Salton, Wong & Yang (1975), *A Vector Space Model for Automatic Indexing*
- Porter (1980), *An Algorithm for Suffix Stripping*
- Dataset: `huggingface.co/datasets/BeIR/webis-touche2020`, CC BY-SA 4.0
- Tools: scikit-learn, SciPy, NLTK, pandas, matplotlib
- Code: this repository

**Slides:** 2 — conclusion; references.

---

## Timing notes

- Persons 3, 4 and 5 carry the 25-point criteria — three quarters of the marks.
  Most rehearsal time and all the Q&A prep belongs there.
- Person 2's legal/ethics section is short in slide count but is 10 rubric
  points, and the pooled-subset caveat lives there. Deliberate pacing, not a
  speed-run.
- Person 5 has the only genuinely surprising result. Give that part the extra
  minute if the deck runs long elsewhere.
- Hand-offs follow the pipeline: Person 2 → 3 (60,000 documents and 2,210
  judgments arrive), Person 3 → 4 (one index, 50,045 postings lists, serves all
  three models), Person 4 → 5 (three runs, 100 deep, one scorer), Person 5 → 6
  (what the unjudged column and the vocabulary mismatch leave open).
