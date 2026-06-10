---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python (NLP Project)
    language: python
    name: nlp-project
---

# Combined Review Analysis System: Summarization, Keyword Extraction & Sentiment Analysis

**Academic NLP Project**

**Authors:**
- *Part A — Text Summarization & Keyword Extraction:* Stanisław
- *Part B — Sentiment Analysis:* Partner

**Dataset:** Amazon Fine Food Reviews (SNAP)

---

## 1. Introduction & Goal

The explosion of user-generated content on e-commerce platforms means that a
single popular product can accumulate thousands of textual reviews. For a human
reader — whether a prospective customer or a product manager — it is impossible
to read all of them. The goal of this project is to build a **combined,
empirical review-analysis system** that automatically distills the essential
information from raw product reviews.

We decompose the problem into three complementary NLP tasks:

1. **Text Summarization** — condense long reviews into short, abstractive
   summaries that preserve the key message.
2. **Keyword Extraction** — surface the most representative terms and phrases,
   giving a quick, scannable signal of what a review is about.
3. **Sentiment Analysis** — classify the emotional polarity (positive /
   negative) of each review, enabling aggregate opinion mining.

Together these three modules form a pipeline that turns an unstructured wall of
text into a structured, decision-ready overview: *what* the review says
(summary), *which aspects* it talks about (keywords), and *how* the author feels
(sentiment).

This notebook focuses on the empirical implementation and qualitative
evaluation of each component on a real-world dataset, and discusses the
strengths, weaknesses, and practical lessons learned.

---

## 2. Examples in Science & Practice

This section motivates each task with references to the scientific literature
and to real-world business applications.

### 2.1 Text Summarization

**Science.** Automatic text summarization is a long-standing NLP problem. Two
broad families exist: *extractive* summarization (selecting and stitching
together existing sentences, e.g. the classic TextRank algorithm of Mihalcea &
Tarau, 2004) and *abstractive* summarization (generating new text). The
transformer revolution — starting with the original *Attention Is All You Need*
(Vaswani et al., 2017) — enabled high-quality abstractive models. **BART**
(Lewis et al., 2020) is a denoising sequence-to-sequence autoencoder that, when
fine-tuned on the CNN/DailyMail news-summarization corpus, produces fluent
abstractive summaries, and **PEGASUS** (Zhang et al., 2020) pushed the state of
the art further with a summarization-specific pre-training objective. More
recently, **instruction-tuned** models such as **Flan-T5** (Chung et al., 2022)
— which extends the T5 text-to-text framework (Raffel et al., 2020) with
multi-task instruction tuning — can summarize directly from a natural-language
prompt, without task-specific fine-tuning. For our experiments we use the
`google/flan-t5-xl` checkpoint, prompting it to summarize each review.

**Practice.** Summarization is everywhere in industry: news aggregators (e.g.
the now-classic *inshorts* app condensing articles to 60 words), e-commerce
"review highlights" (Amazon's own AI-generated review summaries), customer-
support ticket triage, meeting-note generation (Otter.ai, Zoom AI Companion),
and legal/medical document summarization. For our use case — food-product
reviews — a summary lets a shopper grasp the gist of a 250-word review in a
single sentence.

### 2.2 Keyword Extraction

**Science.** Keyword/keyphrase extraction has evolved from statistical methods
such as **TF-IDF** (Spärck Jones, 1972) and graph-based **TextRank** /
**RAKE** (Rose et al., 2010) toward embedding-based approaches. **KeyBERT**
(Grootendorst, 2020) leverages contextual BERT embeddings: it embeds the whole
document and candidate n-grams into the same vector space and selects the
phrases whose embeddings are most cosine-similar to the document embedding.
This captures *semantic* relevance rather than mere frequency.

**Practice.** Keyword extraction powers search-engine optimization (SEO),
automatic tagging of articles and support tickets, faceted product search,
aspect-based opinion mining ("battery", "shipping", "flavor"), and trend
detection in social media. For food reviews, extracted keyphrases like
*"great coffee flavor"* or *"arrived stale"* give an instant, aspect-level
signal of what reviewers care about.

### 2.3 Sentiment Analysis

> **TODO (Partner):** Opisać przykłady naukowe i biznesowe dla analizy
> sentymentu (np. lexicon-based VADER, modele transformerowe; zastosowania:
> monitoring marki, analiza opinii klientów, finanse). *Placeholder.*

---

## 3. Libraries, Functions, and Parameters

A short description of the key libraries used and the most important parameters
we tune.

### 3.1 Hugging Face `transformers` (Summarization)

The [Hugging Face `transformers`](https://huggingface.co/docs/transformers)
library provides a unified API to thousands of pre-trained models. In
`transformers` 5.x the dedicated `pipeline("summarization")` task was removed, so
we load the model with the **`AutoTokenizer`** and **`AutoModelForSeq2SeqLM`**
classes and drive generation via **`model.generate()`** — the standard
lower-level API for sequence-to-sequence models.

We use **`google/flan-t5-xl`**, an *instruction-tuned* T5 model (~3B
parameters). Because it is instruction-tuned, we steer it with a natural-language
prompt (*"Summarize the following Amazon product review in one concise
sentence"*) rather than relying on a single fixed fine-tuning task. This gives
more relevant, controllable summaries on the review domain than a news-only
checkpoint.

Key parameters we control:

- **`model`** — the checkpoint, here `google/flan-t5-xl` (instruction-tuned T5).
- **`torch_dtype`** — `float16` on GPU to halve the memory footprint (~6 GB
  VRAM), `float32` on CPU.
- **`max_length` / `min_length`** — upper/lower bound on the length of the
  generated summary (in tokens).
- **`num_beams`** — beam-search width; `4` is a common quality/speed trade-off.
- **`do_sample`** — `False` gives deterministic (greedy/beam) decoding, which is
  preferable for reproducible, faithful summaries.
- **`truncation`** — Flan-T5 was trained with a 512-token context, so longer
  prompts are truncated at tokenization time.

### 3.2 `KeyBERT` (Keyword Extraction)

[KeyBERT](https://github.com/MaartenGr/KeyBERT) wraps a sentence-transformer to
extract keywords. The core method is **`extract_keywords()`** with these
parameters:

- **`keyphrase_ngram_range`** — the `(min_n, max_n)` size of candidate phrases.
  `(1, 1)` yields single words; `(1, 2)` yields words *and* bigrams, which is
  often more informative for reviews (e.g. *"dog food"* vs. *"dog"* + *"food"*).
- **`stop_words`** — removes uninformative function words (`'english'`).
- **`top_n`** — how many keyphrases to return.
- **`use_mmr` / `diversity`** — Maximal Marginal Relevance to reduce redundancy
  among returned keyphrases.

### 3.3 VADER (Sentiment Analysis)

> **TODO (Partner):** Opisać bibliotekę `vaderSentiment` —
> `SentimentIntensityAnalyzer`, metoda `polarity_scores()` (klucze `neg`, `neu`,
> `pos`, `compound`) oraz typowy próg `compound >= 0.05` dla klasy pozytywnej.
> *Placeholder.*

---

## 4. Setup & Imports

```python
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42  # reproducible sampling
np.random.seed(RANDOM_STATE)

pd.set_option("display.max_colwidth", 200)
```

---

## 5. Dataset Characteristics

We use the **Amazon Fine Food Reviews** dataset (SNAP), which contains ~568k
reviews of fine-food products on Amazon, including the review text, a 1–5 star
score, and helpfulness votes.

### 5.1 Loading the data

```python
import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
csv_path = os.path.join(path, "Reviews.csv")
df = pd.read_csv(csv_path)
```

```python
print(f"Raw dataset shape: {df.shape}")
df.head(3)
```

```python
# Quick look at the columns we care about
df[["Score", "Summary", "Text"]].head(3)
```

### 5.2 Data Preparation (Shared)

This preparation step is **common** to all three tasks (Part A and Part B), so
that everyone works on the same filtered, sampled subset.

We:
1. compute the word count of each review,
2. keep only reviews of *moderate* length (≥ 50 and ≤ 300 words) — very short
   reviews carry too little signal to summarize, while very long ones blow past
   BART's input limit and slow everything down,
3. take a random sample of ~1,500 reviews for tractable experimentation.

```python
# 1) Word count per review
df["word_count"] = df["Text"].astype(str).str.split().str.len()

# Keep a copy of the unfiltered counts for the "before" plot
word_count_before = df["word_count"].copy()

# 2) Filter by length: between 50 and 300 words
MIN_WORDS, MAX_WORDS = 50, 300
mask = (df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)
df_filtered = df[mask].copy()

print(f"Reviews before filtering: {len(df):,}")
print(f"Reviews after length filter [{MIN_WORDS}-{MAX_WORDS} words]: {len(df_filtered):,}")

# 3) Random sample of ~1500 reviews for the analysis
SAMPLE_SIZE = 1500
df_sample = df_filtered.sample(
    n=min(SAMPLE_SIZE, len(df_filtered)),
    random_state=RANDOM_STATE,
).reset_index(drop=True)

print(f"Working sample size: {len(df_sample):,}")
df_sample[["Score", "word_count", "Text"]].head(3)
```

### 5.3 Review-length distribution — before vs. after filtering

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before: full dataset (clip the x-axis so the long tail doesn't crush the plot)
axes[0].hist(word_count_before.clip(upper=600), bins=60, color="#4C72B0")
axes[0].axvline(MIN_WORDS, color="red", linestyle="--", label=f"min={MIN_WORDS}")
axes[0].axvline(MAX_WORDS, color="red", linestyle="--", label=f"max={MAX_WORDS}")
axes[0].set_title("Review length BEFORE filtering (full dataset)")
axes[0].set_xlabel("Word count (clipped at 600)")
axes[0].set_ylabel("Number of reviews")
axes[0].legend()

# After: filtered sample
axes[1].hist(df_sample["word_count"], bins=40, color="#55A868")
axes[1].set_title("Review length AFTER filtering (working sample)")
axes[1].set_xlabel("Word count")
axes[1].set_ylabel("Number of reviews")

plt.tight_layout()
plt.show()
```

```python
# Descriptive statistics of the working sample
df_sample["word_count"].describe()
```

```python
# Distribution of star ratings in the working sample (useful context for
# the sentiment task as well)
plt.figure(figsize=(7, 4))
sns.countplot(x="Score", data=df_sample, palette="viridis")
plt.title("Star-rating distribution in the working sample")
plt.xlabel("Score (1-5 stars)")
plt.ylabel("Count")
plt.show()
```

---

## 6. Empirical Analysis & Results

### Part A — Text Summarization & Keyword Extraction *(Stanisław)*

#### 6.1 Text Summarization with `facebook/bart-large-cnn`

We load BART through the high-level summarization pipeline. The model is
automatically placed on a GPU if one is available, otherwise on CPU.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Flan-T5 XL: instruction-tuned model seq2seq (~3B parametrów), sterowany
# promptem. Ładujemy go bezpośrednio przez klasy Auto* (transformers 5.x nie
# udostępnia już pipeline("summarization")).
#
# Na GPU ładujemy wagi w fp16 (~6 GB VRAM), na CPU wracamy do fp32 (fp16 na CPU
# bywa nieobsługiwane i wolne).
MODEL_NAME = "google/flan-t5-xl"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device} (dtype={dtype})")

summ_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
summ_model.eval()
```

```python
# Flan-T5 jest instruction-tuned, więc streszczanie sterujemy promptem.
# Prompt jawnie prosi o parafrazę i zakazuje kopiowania zdań — bez tego model
# często po prostu przepisuje pierwsze zdanie recenzji.
SUMMARY_PROMPT = (
    "Write a short summary of the following Amazon product review in your own "
    "words, capturing the reviewer's overall opinion. Do not copy sentences "
    "from the review.\n\nReview: {text}"
)


@torch.no_grad()
def summarize_text(text, max_length=60, min_length=15):
    """Generate an abstractive summary of a single review with Flan-T5.

    Flan-T5 was trained with a 512-token context, so we truncate the prompt at
    512 tokens. Deterministic beam-search decoding (do_sample=False) keeps
    results reproducible. To prevent the model from simply copying the source,
    `encoder_no_repeat_ngram_size` forbids reusing any 4-gram from the input,
    which forces genuine rephrasing.
    """
    prompt = SUMMARY_PROMPT.format(text=text)
    inputs = summ_tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    summary_ids = summ_model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=3,          # brak powtórzeń wewnątrz streszczenia
        encoder_no_repeat_ngram_size=4,  # zakaz kopiowania 4-gramów ze źródła -> parafraza
        length_penalty=1.0,
    )
    return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

```python
# Qualitative analysis: summarize a handful of example reviews.
# We pick a few reviews from the working sample.
N_EXAMPLES = 4
examples = df_sample.sample(n=N_EXAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

summaries = []
for i, row in examples.iterrows():
    original = row["Text"]
    summary = summarize_text(original)
    summaries.append(summary)

    print("=" * 100)
    print(f"EXAMPLE {i + 1}  |  Score: {row['Score']}  |  Original length: {row['word_count']} words")
    print("-" * 100)
    print("ORIGINAL (truncated to 600 chars for display):")
    print(original[:600] + ("..." if len(original) > 600 else ""))
    print("-" * 100)
    print("BART SUMMARY:")
    print(summary)
    print()

examples["bart_summary"] = summaries
```

```python
# Side-by-side comparison table: original word count vs. summary word count
comparison = pd.DataFrame({
    "score": examples["Score"],
    "orig_words": examples["word_count"],
    "summary_words": examples["bart_summary"].str.split().str.len(),
    "bart_summary": examples["bart_summary"],
})
comparison["compression_ratio"] = (
    comparison["summary_words"] / comparison["orig_words"]
).round(2)
comparison
```

**Qualitative observation.** BART produces fluent, grammatical, single-paragraph
summaries that capture the dominant sentiment and the main product aspect of
each review, typically compressing the text to 15–25% of its original length.
On clearly-written reviews the summaries are faithful; on rambling or
multi-topic reviews BART tends to focus on whichever aspect appears first.

#### 6.2 Keyword Extraction with `KeyBERT`

```python
from keybert import KeyBERT

# KeyBERT defaults to the 'all-MiniLM-L6-v2' sentence-transformer — small, fast
# and well suited for short documents like reviews.
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
```

```python
def extract_keywords(text, ngram_range=(1, 2), top_n=5, use_mmr=False, diversity=0.5):
    """Extract the top-n keyphrases from a review using KeyBERT.

    `ngram_range` controls phrase length:
      (1, 1) -> single words only
      (1, 2) -> single words AND bigrams (often more descriptive)
    """
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
    )
```

```python
# Demonstrate the effect of ngram_range on the SAME reviews used above.
for i, row in examples.iterrows():
    text = row["Text"]

    kw_unigram = extract_keywords(text, ngram_range=(1, 1), top_n=5)
    kw_bigram = extract_keywords(text, ngram_range=(1, 2), top_n=5)

    print("=" * 100)
    print(f"EXAMPLE {i + 1}  |  Score: {row['Score']}")
    print("-" * 100)
    print("BART summary:        ", row["bart_summary"])
    print("Keywords (1,1) unigrams:", [kw for kw, score in kw_unigram])
    print("Keywords (1,2) uni+bigrams:", [kw for kw, score in kw_bigram])
    print()
```

```python
# Show the cosine-similarity scores too, so we can see KeyBERT's confidence.
sample_text = examples.iloc[0]["Text"]
print("Review:\n", sample_text[:400], "...\n")

print("ngram_range=(1, 1):")
for kw, score in extract_keywords(sample_text, ngram_range=(1, 1)):
    print(f"   {score:.4f}  {kw}")

print("\nngram_range=(1, 2):")
for kw, score in extract_keywords(sample_text, ngram_range=(1, 2)):
    print(f"   {score:.4f}  {kw}")

print("\nngram_range=(1, 3) with MMR diversity:")
for kw, score in extract_keywords(sample_text, ngram_range=(1, 3), use_mmr=True, diversity=0.7):
    print(f"   {score:.4f}  {kw}")
```

**Qualitative observation.** Switching `ngram_range` from `(1, 1)` to `(1, 2)`
markedly improves interpretability: single tokens like *"coffee"* / *"flavor"*
become coherent phrases like *"great coffee flavor"*. Enabling MMR
(`use_mmr=True`) with high `diversity` reduces near-duplicate phrases and yields
a broader aspect coverage. Because KeyBERT is embedding-based, it also surfaces
semantically central terms that are not necessarily the most frequent.

#### 6.3 Combined Part-A output

Bringing summarization and keyword extraction together — the "review at a
glance" view that Part A contributes to the combined system.

```python
combined = []
for i, row in examples.iterrows():
    combined.append({
        "score": row["Score"],
        "summary": row["bart_summary"],
        "keywords": [kw for kw, _ in extract_keywords(row["Text"], ngram_range=(1, 2), top_n=4)],
    })

combined_df = pd.DataFrame(combined)
combined_df
```

#### 6.3.1 Export results to CSV (easy side-by-side review in Excel)

To make the qualitative comparison easy, we run summarization + keyword
extraction over a batch of reviews and export everything to a single CSV. The
file is written with **`utf-8-sig`** encoding so Excel opens it with correct
special characters, and `QUOTE_ALL` so that commas/newlines inside reviews don't
break the columns.

```python
import csv

# How many reviews to export. BART on CPU is slow, so keep it modest;
# bump this up (or run on GPU) for a larger sample.
N_EXPORT = 25

batch = df_sample.sample(n=min(N_EXPORT, len(df_sample)), random_state=RANDOM_STATE).reset_index(drop=True)

records = []
for i, row in batch.iterrows():
    text = row["Text"]
    summary = summarize_text(text)
    keywords = extract_keywords(text, ngram_range=(1, 2), top_n=5)

    records.append({
        "id": int(row.name),
        "score": int(row["Score"]),
        "orig_word_count": int(row["word_count"]),
        "original_text": text,
        "bart_summary": summary,
        "summary_word_count": len(summary.split()),
        # join keyphrases into one readable cell, e.g. "great coffee flavor; rich aroma"
        "keywords": "; ".join(kw for kw, _ in keywords),
    })
    print(f"Processed {i + 1}/{len(batch)}")

results_df = pd.DataFrame(records)
results_df.head()
```

```python
# Write the Excel-friendly CSV next to the notebook.
OUTPUT_CSV = "summarization_results.csv"

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",   # makes Excel render accents/special chars correctly
    quoting=csv.QUOTE_ALL,  # wrap every field so commas/newlines stay inside cells
    lineterminator="\n",
)

print(f"Saved {len(results_df)} rows to '{OUTPUT_CSV}'.")
print("Open it in Excel: columns are original_text | bart_summary | keywords for easy comparison.")
```

---

### Part B — Sentiment Analysis *(Partner)*

> This section is a **scaffold**. The implementation is the partner's
> responsibility.

#### 6.4 Sentiment Analysis with VADER

```python
# TODO: Implementacja Sentiment Analysis z VADER
# - zaimportować SentimentIntensityAnalyzer z vaderSentiment
# - policzyć polarity_scores() (compound) dla każdej recenzji w df_sample
# - zmapować compound -> etykieta (positive / negative / neutral)
```

```python
# TODO: Stworzyć "ground truth" z kolumny 'Score'
# (np. Score >= 4 -> positive, Score <= 2 -> negative) i porównać z VADER
```

#### 6.5 Evaluation metrics

```python
# TODO: Metryki Accuracy, F1, Confusion Matrix
# - sklearn.metrics: accuracy_score, f1_score, classification_report
# - sklearn.metrics: confusion_matrix + seaborn heatmap do wizualizacji
```

---

## 7. Strengths and Weaknesses

This section evaluates the methods used in **Part A** (Summarization & Keyword
Extraction). *(Part B / sentiment strengths & weaknesses — TODO partner.)*

### 7.1 Text Summarization (Flan-T5 XL)

**Strengths**

- **Abstractive & fluent:** Flan-T5 generates natural, grammatical sentences
  rather than copying fragments, so summaries read like human-written highlights.
- **Instruction-tuned & steerable:** because it follows natural-language prompts,
  we can control the summary (length, style, "one concise sentence") without
  retraining — and adapt it to the review domain instead of news.
- **Strong multi-task pre-training:** Flan-T5's instruction tuning across many
  tasks gives robust zero-shot generalization to product reviews.

**Weaknesses**

- **Hallucination:** like all abstractive models, it can invent details not
  present in the source (e.g. a brand name or a fact), which is dangerous for
  faithful review summarization.
- **Computational cost:** Flan-T5 XL has ~3B parameters; it really wants a GPU
  (in fp16 it needs ~6 GB VRAM), and on CPU summarizing many reviews is very
  slow.
- **Input-length limit:** the 512-token training context is *more* restrictive
  than BART's 1024, so very long reviews are truncated more aggressively.
- **Prompt sensitivity:** output quality depends on the wording of the prompt;
  a poorly phrased instruction can degrade the summary.

### 7.2 Keyword Extraction (KeyBERT)

**Strengths**

- **Semantic, not just frequency-based:** captures contextually relevant terms
  that TF-IDF would miss.
- **Flexible n-grams:** `ngram_range` lets us choose between single words and
  multi-word phrases; MMR controls diversity.
- **Lightweight default model:** `all-MiniLM-L6-v2` is fast and runs comfortably
  on CPU.

**Weaknesses**

- **No notion of importance beyond similarity:** a phrase can be central to the
  document embedding yet not be what a *human* would consider the key point.
- **Sensitive to preprocessing:** stop-word choice and casing affect results;
  noisy text can yield odd phrases.
- **Candidate generation cost:** for long documents with a wide n-gram range,
  the number of candidate phrases (and embeddings) grows quickly.
- **No built-in deduplication unless MMR is enabled** — without it, returned
  keyphrases can be near-duplicates ("great coffee", "coffee great").

---

## 8. Good Practices & Insights

Practical lessons from implementing the Part-A methods:

1. **Filter by length before summarizing.** Removing <50-word reviews avoids
   "summaries longer than the source", and capping at 300 words keeps inputs
   within the model's token budget — this single preprocessing step improved both
   speed and output quality.
2. **Always set `truncation=True`** at tokenization. Forgetting it causes hard
   errors on long inputs; with it, the model degrades gracefully when a review
   exceeds the 512-token context.
3. **Craft the instruction prompt deliberately.** With an instruction-tuned model
   like Flan-T5, the prompt *is* the task spec — being explicit ("in one concise
   sentence") measurably tightens the summaries.
4. **Prefer deterministic decoding (`do_sample=False`)** for reproducible
   academic results — sampling makes qualitative comparisons unreliable.
5. **Use `(1, 2)` n-grams as a sensible default for KeyBERT** on reviews:
   bigrams capture aspect+opinion phrases ("great flavor") that single words
   cannot, at negligible extra cost.
6. **Turn on MMR (`use_mmr=True`) when you need diverse keywords.** Without it,
   the top-n list is often crowded with synonyms of the single dominant theme.
7. **Cache / load heavy models once.** Loading a 3B-parameter model and the
   `KeyBERT` model is expensive; create them in a dedicated cell and reuse the
   objects, rather than re-instantiating inside a loop.
8. **Load big models in fp16 on GPU.** `torch_dtype=torch.float16` roughly halves
   VRAM use and speeds up generation; keep fp32 as the CPU fallback. A simple
   `torch.cuda.is_available()` check selects the device automatically.
9. **Validate qualitatively, not just quantitatively.** ROUGE alone hides
   hallucinations; reading a handful of original-vs-summary pairs (as we do in
   §6.1) is essential to catch unfaithful outputs.

---

## 9. Conclusion

Part A delivers two working modules — abstractive summarization with
`facebook/bart-large-cnn` and semantic keyword extraction with `KeyBERT` — that
together turn long Amazon food reviews into a concise "summary + keyphrases"
view. Combined with the partner's sentiment-analysis module (Part B), the system
forms a complete pipeline for automated, structured review analysis. The
qualitative evaluation confirms that the methods are effective while also
exposing their limitations (hallucination, compute cost), which we document for
honest, reproducible academic reporting.
