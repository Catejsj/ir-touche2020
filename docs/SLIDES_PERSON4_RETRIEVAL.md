# Slides — Person 4, Retrieval Model Design (Seth)

Rubric criterion 4, **25 points**. Two slides.

- **Part A** — for whoever builds the slides. Type the bullets, nothing else.
- **Part B** — what I say out loud.
- **Part C** — my two pictures, explained.
- **Part D** — the words explained.
- **Parts E–F** — questions I might get, and what is not mine.

**My job in one line:** build the three ways of ranking documents that the
rubric asks for, and explain how each one decides which document goes first.

**The one idea that holds it together — what each model thinks "relevant" means:**

| model | its idea of relevant |
|---|---|
| **Boolean** | the document *contains* your words |
| **VSM (TF-IDF)** | the document *points the same direction* as your query |
| **BM25** | the document contains your words *unusually often for its length* |

Say that at the start and the rest has somewhere to sit.

**The scores belong to Person 5.** I set the models up and hand over.

---

# PART A — for the slide maker

## Slide 1 — TWO COLUMNS

**Slide title:** `Two ways to match: does it contain the words, or point the same way?`

**Small line under the title (all three slides share it):**
`Query: "Should teachers get tenure?"  →  after cleaning: teacher, get, tenur`

### LEFT COLUMN

**Heading:** `Model 1 — Boolean (does it contain the words?)`

- A document either **has** your query words or it does not. No weighting
- We use **OR**, not AND — with 2–4 word queries, AND returns almost nothing
- Ranked by **how many** query words are present (coordination level)
- **Honest note:** real Boolean returns an *unordered set*. Ranking it at all is
  a concession we make so it can be compared

### RIGHT COLUMN

**Heading:** `Model 2 — Vector Space (does it point the same way?)`

- Every document and the query become a **list of numbers**, one per word
- Word weight = **how often it appears here × how rare it is overall** (TF-IDF)
- Similarity = the **angle** between the two lists (cosine)
- The angle ignores length — **and that turns out to be the problem** (Person 5)

**Picture (bottom half, full width):** `figures/fig6_scoring_walkthrough.png`
One real query and one real document, scored by all three models. It is the
slide — the bullets are just labels for it.

---

## Slide 2 — FULL WIDTH

**Slide title:** `Model 3 — BM25: how we actually rank`

- Same idea as TF-IDF, but with **two fixes**
- **Fix 1 — saturation:** the 10th "tenure" adds far less than the 2nd. A
  document is not 10× better for repeating a word 10 times
- **Fix 2 — length:** compares each document to the **average length in the
  collection**, instead of dividing length away
- Settings: **k1 = 1.2** (how fast saturation kicks in), **b = 0.75** (how hard
  long documents are penalised)
- **We did NOT tune these.** 49 queries, no validation set — tuning on the test
  set would mean reporting how well we tuned, not how well BM25 works

**Picture (bottom half, full width):** `figures/fig7_bm25_two_fixes.png`
Two panels — saturation on the left, length normalisation on the right. This is
the picture that makes "saturation" and "length normalisation" mean something.

**Small monospace line under the picture, if it fits:**

```
49 queries, 60,000 documents:  Boolean 0.01s   VSM 0.04s   BM25 0.02s
```

---

# PART B — what I say out loud

About 3–4 minutes. This is the shape, not a script to memorise.

## Opening — the map (15 seconds)

> My part is the three ranking models. The difference between them is simply
> **what each one thinks "relevant" means**.
>
> The first says a document is relevant if it *contains* your words.
> The second says it is relevant if it *points the same direction* as your query.
> The third says it is relevant if it contains your words *unusually often for a
> document of its length*.
>
> This slide has the first two. The next slide has the one that wins.

## Slide 1, left — Boolean (60 seconds)

> Boolean is the oldest method and the simplest. A document either contains your
> query words or it does not. There is no weighting and no idea of "how well" —
> just yes or no.
>
> One decision I want to be upfront about. There are two ways to do Boolean: AND,
> where a document needs *all* your words, and OR, where it needs *any* of them.
> We use OR. The reason is that after we remove stop words, "Should teachers get
> tenure?" becomes just two words — `teacher` and `tenur`. Asking for documents
> containing *both*, out of sixty thousand, returns a handful or sometimes
> nothing at all, and a system that returns nothing cannot be scored.
>
> There is a second honesty point here. Real Boolean retrieval gives you an
> unordered **set** of documents — it does not rank. But every measure we use
> assumes an order. So we order by how many of the query words each document
> contains, which is called coordination level. That is a concession we made so
> the comparison could happen, and it is in our limitations rather than hidden.

## Slide 1, right — Vector Space (75 seconds)

> The second model is the vector space model, and this is where weighting starts.
>
> The idea is to turn every document into a list of numbers — one number per word
> in our vocabulary, fifty thousand of them. The query becomes a list the same
> shape. Then relevance is just: how similar are these two lists.
>
> Each number is TF-IDF, which is two simple ideas multiplied. TF is how often
> the word appears in this document — if an argument says "tenure" five times it
> is probably about tenure. IDF is how rare the word is across the whole
> collection — "tenure" appears in few documents so it is informative, while a
> word appearing everywhere tells you nothing. Frequent here, rare overall, means
> a high weight.
>
> Then we compare the two lists using cosine similarity, which measures the
> **angle** between them rather than their size. That is deliberate: it means a
> four-thousand-word argument and a forty-word one are judged on direction, not
> on length, so a long document cannot win just by containing more words.
>
> That sounds like exactly what you want. Remember it, because [Person 5] is
> about to show you that this is the single thing that makes this model lose.

## Slide 2 — BM25 (90 seconds)

> The third model is BM25, and it is the one that wins.
>
> Start from TF-IDF and apply two fixes.
>
> The first is **saturation**. In TF-IDF, a word appearing ten times counts ten
> times as much as once. That is not how relevance actually works — if an
> argument mentions tenure twice you know it is about tenure, and the tenth
> mention tells you almost nothing new. BM25 makes the score flatten out. The
> `k1` setting, which we leave at 1.2, controls how quickly that flattening
> happens.
>
> The second fix is **length**, and this is the important one. The vector space
> model removes length completely by using the angle. BM25 does something
> different: it compares each document to the **average length in the
> collection**. A document longer than average gets a penalty, a shorter one gets
> a small boost, but length is never erased. The `b` setting, at 0.75, controls
> how strong that correction is.
>
> On our corpus that matters enormously, because the documents run from three
> words to sixteen thousand.
>
> One last thing, and I think it is worth saying out loud. We did **not** tune
> k1 and b. They are the published defaults. Touché gives us forty-nine queries
> and no separate validation set, so if we had tried many values and reported the
> best, we would be reporting how well we tuned on the test set — not how well
> BM25 works. We ran the sweep and put it in the report for transparency, but the
> numbers you are about to see use the defaults.
>
> [Person 5] will now tell you how the three actually scored.

---

# PART C — my two pictures, explained

Read this before presenting. If someone points at either one, the answer is
here.

## Picture 1 — `fig6_scoring_walkthrough.png` (slide 1)

**What it is:** one real query and one real document, put through all three
models, so you can see what each one actually computes. Nothing here is invented
— the numbers come from the same functions that produced our results.

**Reading it top to bottom:**

- **QUERY** — the raw question, then what survives cleaning: `teacher`, `get`,
  `tenur`. Three terms from five words.
- **DOCUMENT** — a real argument from the collection, *"There should not be a
  teacher tenure."* It is **842 terms** long against a collection average of
  **147**, so it is a long document — which matters for the third box.
- **WHAT THE INDEX TELLS US** — the two facts every model needs. How often each
  term appears *in this document*, and how many documents in the whole
  collection contain it.
- **The three boxes** — each model's answer.

**The one thing to point at:** look at the `get` row. It appears in **14,084 of
60,000** documents — almost a quarter of the collection. `tenur` appears in
**67**. That is the whole idea of IDF on one line: `tenur` tells you a great deal
about what this document is about, `get` tells you almost nothing, and the
weighting has to reflect that.

**If someone asks why the three scores are so different (3, 2.5, 22.5):** they
are on different scales and are never compared to each other. Each one only
ranks documents *within its own model*. A BM25 score of 22 does not mean BM25 is
"seven times better" than VSM's 2.5 — it means nothing at all across models.

## Picture 2 — `fig7_bm25_two_fixes.png` (slide 2)

**What it is:** BM25's two changes to TF-IDF, drawn as curves. These are the
formulas themselves plotted, not measurements of our results — so they are true
regardless of what our data did.

**Left panel — saturation.**
The x-axis is how many times a word appears in a document. The y-axis is how
much that adds to the score, relative to appearing once.

- **Grey dashed** — raw counting. Ten mentions count ten times. It runs off the
  top of the chart.
- **Orange** — TF-IDF's `1 + log tf`. Already damped, but still climbing at 20.
- **Green** — BM25. Flattens out fast: by about the fifth mention it has nearly
  stopped rising.

Say: *"if an argument mentions tenure twice, you know it is about tenure. The
tenth mention tells you almost nothing new, and BM25 is the only one of the
three that knows that."*

**Right panel — length normalisation.**
The x-axis is a document's length divided by the collection average, so 1.0 is
an average-length document. The y-axis is the multiplier applied to its score.

- **Grey dashed, flat at 1.0 — `b = 0`** — length ignored completely. **This is
  effectively what the vector space model does**, and it is the setup for
  Person 5's finding.
- **Blue dotted — `b = 1`** — fully corrected. Short documents get boosted hard.
- **Green solid — `b = 0.75`, ours** — a compromise between the two, which is
  why it is the standard default.

**If someone asks why you chose 0.75:** we did not choose it, it is the
published default, and the curve shows why it is a reasonable one — it corrects
length without the extreme boost that `b = 1` gives very short documents.

---

# PART D — the words explained

**Query** — the debate question someone types, e.g. *"Should teachers get
tenure?"*

**Document** — one argument from the collection. We have 60,000.

**Term** — one word after cleaning, e.g. `tenur`. It is only ever a lookup key;
nobody ever sees it.

**Inverted index** — for each term, the list of documents containing it. Person 3
built it. It is why we can search 60,000 documents in a hundredth of a second.

**Coordination level** — how many different query words a document contains.
Query has 2 words, document has both → level 2.

**TF (term frequency)** — how often a word appears in this document.

**IDF (inverse document frequency)** — how rare a word is across the collection.
Rare = informative.

**TF-IDF** — the two multiplied. High when a word is frequent here and rare
elsewhere.

**Vector** — the list of numbers standing for a document. One slot per word in
the vocabulary, mostly zeros.

**Cosine similarity** — the angle between two vectors. Close to 1 means pointing
the same way. Ignores length by design.

**Saturation** — extra repetitions counting less and less. BM25 does this; TF-IDF
does not.

**k1** — BM25's saturation speed. Ours is 1.2.

**b** — BM25's length correction strength. Ours is 0.75. At b = 0 length is
ignored entirely; at b = 1 it is fully corrected.

**Run** — our output: for each query, the top 100 documents in order.

---

# PART E — questions I might get

**Why OR and not AND? Isn't AND what Boolean means?**
Both are Boolean. But after stop word removal our queries are two to four words,
and AND over 60,000 documents returns a handful or nothing. A system that returns
nothing cannot be evaluated at all, so the comparison would be impossible.

**Then it is not really Boolean retrieval any more.**
Fair, and we say so ourselves rather than wait to be caught. Real Boolean returns
an unordered set; ranking by coordination level is something we added so ranking
metrics could be computed. It is written in our limitations.

**How did you choose k1 and b?**
We did not — they are the published defaults from Robertson and Zaragoza. With
49 queries and no validation split, tuning on this data would be fitting the test
set. We ran the sweep for transparency and report the defaults.

**Isn't it lazy not to tune?**
The opposite. Tuning on the test set and reporting the best number is the easy
thing to do and it inflates your result. Choosing not to is the more defensible
decision, and we explain it rather than hoping nobody checks.

**What is the actual difference between TF-IDF's IDF and BM25's?**
They come from different derivations, not just different formulas. TF-IDF's is
always positive. BM25's can go **negative** for a word appearing in more than
half the collection — which correctly says that word is evidence *against*
relevance. The 0.5 smoothing keeps it defined when a term is in every document or
none.

**Why does cosine similarity ignore length if length matters?**
Because normally it should. If two documents argue the same thing and one is
longer, you do not want the longer one to win automatically. The problem is that
our collection contains three-word documents, and for those the normalisation
overcorrects. That is Person 5's measurement.

**How long does it take to run?**
Under a twentieth of a second for all 49 queries. The inverted index means a
query only touches the postings lists for its own two or three terms, never the
whole collection.

**Did you try dense or neural retrieval?**
No. The rubric marks it optional, encoding 60,000 documents on CPU is hours, and
a dense score is a single number that cannot be broken down — which would have
cost us the error analysis. It is our first future work item.

---

# PART F — what is NOT mine

Do not present these. If asked, name the person who owns them.

- **All the scores.** P@10, MRR, MAP, nDCG, the comparison table — **Person 5**.
  I finish by handing over, not by announcing a winner.
- **`fig1`–`fig5` belong to Person 5.** Those are the *results* figures. Mine
  are `fig6` and `fig7`, which show how the models *work*, not how well they
  did. No overlap, and nothing is shown twice.
- **The length-bias finding** (VSM's median document is 16 words) is Person 5's
  headline. I *set it up* by explaining that cosine ignores length — I do not
  give away the punchline.
- **The index and preprocessing** — stop words, stemming, the postings lists —
  belong to Person 3.
- **The pooled subset and the licence** belong to Person 2.

One sentence to hand over on, and nothing more:

> "All three now produce a ranked list of 100 documents per query. [Person 5]
> will tell you how good those lists actually are."
