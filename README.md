# Provenance Guard

An AI-authorship detection API for creative platforms. Submitted text is analyzed by four independent signals and returned with a transparency label, a confidence score, and an appeal pathway.

---

## Architecture Overview

A submission travels through five stages before a label is returned:

```
POST /submit
     │
     ▼
[Rate Limiter] ── 429 if > 10 req/min per IP
     │
     ▼
[Request Controller] ── assigns content_id (UUID)
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
[Perplexity Signal]              [Burstiness Signal]
[Vocabulary Signal]              [N-Gram Signal]
     └──────────────────────────────────────┘
                       │
                       ▼
          [Confidence Scoring Engine]
          weighted average → float [0.0–1.0]
                       │
                       ▼
           [Label Generator]
           score → classification + disclosure text
                       │
                       ▼
           [Audit Log] ── append-only JSONL entry
                       │
                       ▼
           JSON response to caller
```

The four signals run independently on the same text. Each produces a normalized `[0.0, 1.0]` score where `1.0 = likely AI-generated`. These are combined by the Confidence Scoring Engine into a single weighted score, which the Label Generator maps to one of five transparency label bands. The full record — text signals, combined score, label, timestamp — is written to `audit_log.jsonl` before the response is sent.

Appeals follow a separate path: `POST /appeal` validates the `content_id` against the audit log, appends an appeal record, and returns a confirmation. The original classification entry is never overwritten.

---

## Detection Signals

### Signal A — Perplexity (weight: 0.45)

**What it measures:** How statistically predictable the word choices are. Implemented as a Groq LLM judge: the model scores the text on a `[0.0, 1.0]` scale where `1.0` means the text matches the high-probability patterns typical of instruction-tuned LLM output.

**Why it works:** LLMs generate text by selecting statistically likely tokens. The result is coherent but gravitates toward the mathematical center of language. Human writers deviate through personal idiom, unexpected diction, and non-standard phrasing that an LLM would not naturally produce.

**What it misses:** Formal academic prose and legal writing hug the statistical center by design — not because they are AI-generated, but because academic register is trained toward precision and predictability. This signal consistently over-scores these texts.

---

### Signal B — Burstiness (weight: 0.35)

**What it measures:** The standard deviation of sentence lengths (in word-tokens) across the submission. Low standard deviation means uniform sentences; high standard deviation means the author mixes long complex sentences with short punchy ones.

**Why it works:** LLMs optimize for stable flow and produce structurally uniform output. Human writers modulate pace based on emotion and emphasis, producing natural length variation.

**What it misses:** Short submissions (fewer than 4–5 sentences) cannot produce a meaningful variance reading. A text with two sentences of coincidentally similar length will score as AI-like even if it is clearly human-authored. The signal defaults to `0.5` (neutral) when fewer than two sentences are detected.

---

### Signal C — Vocabulary Diversity / MATTR (weight: 0.10)

**What it measures:** Moving Average Type-Token Ratio — the average fraction of unique words in a sliding 50-token window. Low MATTR means repetitive vocabulary; high MATTR means rich lexical variety.

**Why it works:** LLMs repeat common filler and transitional words more than humans over extended text. The sliding window normalizes for document length, preventing long texts from artificially lowering the score.

**What it misses:** On short texts (under 50 tokens), MATTR falls back to plain TTR — at which point almost every word is unique regardless of authorship. This signal contributes meaningfully only on submissions of 100+ words.

---

### Signal D — N-Gram Marker Frequency (weight: 0.10)

**What it measures:** The density of LLM-characteristic words and phrases per 100 tokens. The marker lexicon includes single words (`delve`, `testament`, `furthermore`, `meticulously`, `underscore`, `nuanced`, `pivotal`, `robust`, `stakeholders`, `transformative`, `actionable`, `multifaceted`) and multi-word phrases (`it is important to note`, `it is worth noting`, `testament to`, `in today's`).

**Why it works:** Modern LLMs trained on RLHF instruction datasets converge on a polite, structured register that over-uses a predictable set of transitional and hedging phrases at measurably elevated rates compared to natural human writing.

**What it misses:** A human writer who naturally uses formal academic or corporate vocabulary can trigger false positives. The marker list was tuned for RLHF-style output; it may not generalize well to other LLM families or future model generations.

---

## Confidence Scoring

The four signal scores are combined via a weighted average:

```
confidence = 0.45 × perplexity
           + 0.35 × burstiness
           + 0.10 × vocabulary
           + 0.10 × ngram
```

The weights reflect confidence in each signal's reliability: perplexity receives the highest weight because it is semantically grounded (the LLM judge understands context), burstiness second because it is structurally independent of vocabulary, and vocabulary and n-gram share the remaining weight as strong corroborating signals that are noisier on short text.

**Validation:** The function was verified against synthetic inputs spanning the full range before wiring into the endpoint. Key boundary checks:

| perplexity | burstiness | vocabulary | ngram | confidence | band |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 1.0 | 1.00 | `definitely_ai` |
| 0.88 | 0.85 | 0.5 | 1.0 | 0.87 | `definitely_ai` |
| 0.72 | 0.70 | 0.5 | 0.5 | 0.68 | `likely_ai` |
| 0.55 | 0.35 | 0.0 | 0.0 | 0.28 | `likely_human` |
| 0.0  | 0.0  | 0.0 | 0.0 | 0.00 | `definitely_human` |

**Example — high-confidence AI submission** (confidence: `0.74`):

> *"In today's rapidly evolving technological landscape, it is essential to underscore the importance of leveraging data-driven insights to enhance organizational efficiency. Furthermore, the implementation of robust frameworks enables stakeholders to navigate complex challenges with precision. It is important to note that these multifaceted solutions demonstrate a testament to meticulous planning and nuanced strategic thinking."*

```json
{
  "signals": { "perplexity": 0.92, "burstiness": 0.85, "vocabulary": 0.03, "ngram": 1.0 },
  "confidence_score": 0.7419,
  "attribution": "likely_ai",
  "label": { "classification": "Likely AI-Generated", "confidence_display": "74% AI likelihood" }
}
```

**Example — lower-confidence borderline submission** (confidence: `0.50`):

> *"The relationship between syntactic complexity and reading comprehension has been extensively studied in psycholinguistics. Prior work suggests that embedded relative clauses impose a measurable processing cost. This study extends those findings to naturalistic reading conditions using eye-tracking methodology."*

```json
{
  "signals": { "perplexity": 0.80, "burstiness": 0.88, "vocabulary": 0.0, "ngram": 0.0 },
  "confidence_score": 0.4994,
  "attribution": "uncertain",
  "label": { "classification": "Authorship Uncertain", "confidence_display": "Inconclusive (50% AI likelihood)" }
}
```

The second example is a documented false-positive risk: it is human-authored academic writing, but perplexity scores it high because formal academic register is statistically predictable. The n-gram signal correctly returns `0.0` (no LLM marker words), which drags the confidence down from `likely_ai` into `uncertain` — the system refuses to make a confident attribution rather than mislabeling it.

---

## Transparency Label

The label returned by `/submit` includes three fields: `classification`, `confidence_display`, and `disclosure`.

---

### Variant 1 — AI-Generated Content (score ≥ 0.85)

```
classification:      AI-Generated Content
confidence_display:  90% AI likelihood

disclosure:
  This submission was analyzed by Provenance Guard. Multiple
  independent signals — including statistical predictability,
  sentence structure uniformity, vocabulary compression, and
  language model marker phrases — converged to indicate this
  text was likely generated by an AI system. This label does
  not constitute a final determination. The author may submit
  an appeal if this classification is incorrect.
```

---

### Variant 2 — Likely AI-Generated (score 0.65–0.84)

```
classification:      Likely AI-Generated
confidence_display:  74% AI likelihood

disclosure:
  This submission was analyzed by Provenance Guard. The
  detection signals lean toward AI generation, though not with
  high confidence. Patterns such as predictable phrasing,
  uniform structure, or AI-characteristic vocabulary were
  detected. The author may submit an appeal if this
  classification is incorrect.
```

---

### Variant 3 — Authorship Uncertain (score 0.36–0.64)

```
classification:      Authorship Uncertain
confidence_display:  Inconclusive (50% AI likelihood)

disclosure:
  This submission was analyzed by Provenance Guard. The
  detection signals produced mixed results and the system
  cannot confidently attribute this text to a human or AI
  author. This may indicate a hybrid workflow, heavily edited
  AI output, or content that falls outside the system's
  reliable detection range. The author may submit additional
  context via an appeal.
```

---

### Variant 4 — Likely Human-Authored (score 0.16–0.35)

```
classification:      Likely Human-Authored
confidence_display:  71% human likelihood

disclosure:
  This submission was analyzed by Provenance Guard. The text
  shows patterns consistent with human authorship — including
  variable sentence rhythm, diverse vocabulary, and low
  reliance on AI-characteristic phrasing. This label does not
  constitute a final determination.
```

---

### Variant 5 — Human-Authored Content (score < 0.16)

```
classification:      Human-Authored Content
confidence_display:  92% human likelihood

disclosure:
  This submission was analyzed by Provenance Guard. The text
  shows strong patterns consistent with human authorship —
  including variable sentence rhythm, diverse vocabulary, and
  an absence of AI-characteristic phrasing. This label does not
  constitute a final determination.
```

---

## Rate Limiting

**`POST /submit`: 10 requests per minute per IP**
**All other routes (`/log`, `/appeal`)**: 100 requests per minute per IP

### Reasoning

10/min on `/submit` was chosen to reflect the realistic ceiling of a human creator submitting their own work — a prolific writer checking several pieces before publishing would never need more than a handful of submissions per minute. The limit is tight enough to prevent three specific abuse patterns:

1. **Threshold probing** — an attacker trying to reverse-engineer the scoring boundaries by flooding the endpoint with variations of the same text cannot run more than 10 attempts per minute per IP.
2. **Bulk pipeline abuse** — a script submitting thousands of AI-generated pieces to obtain "human" labels at scale is blocked at the network layer before the Groq API is called.
3. **API cost amplification** — every `/submit` call hits the Groq API. A 429 on the 11th request caps runaway spend from a single source.

`/log` and `/appeal` are left at 100/min because they do not call external APIs and are used at low frequency by design.

### Test evidence

12 rapid requests sent to `POST /submit` (limit: 10/minute):

```
200  ← request 1
200  ← request 2
200  ← request 3
200  ← request 4
200  ← request 5
200  ← request 6
200  ← request 7
200  ← request 8
200  ← request 9
200  ← request 10
429  ← request 11 (rate limit exceeded)
429  ← request 12 (rate limit exceeded)
```

---

## Known Limitations

### 1. Formal academic and legal prose

The system consistently mis-scores formally written human text as AI-generated. A peer-reviewed methods section, a legal brief, or a philosophy dissertation is statistically predictable by design — academic register demands terminological precision and avoids idiomatic variation. The perplexity signal scores these texts above 0.80, and the burstiness signal scores them high because disciplined paragraph structure produces uniform sentence lengths. The n-gram signal is the only correction: if the text contains no LLM marker phrases, it returns 0.0 and drags the confidence back toward the uncertain band. In testing, a three-sentence human-authored academic paragraph scored `0.50` (`Authorship Uncertain`) rather than being mislabeled — but the system cannot definitively clear it either. Writers with formal academic backgrounds are more likely to receive uncertain labels than confident human-authored ones, and may need to file appeals.

### 2. Short-form text (under 100 tokens)

Burstiness is statistically meaningless on fewer than 4–5 sentences, and MATTR cannot complete a full 50-token window. Both signals default to `0.5` (neutral) on short submissions, which means the confidence score is effectively driven by perplexity and n-gram alone. A two-sentence submission will receive a noisier, less reliable classification than a multi-paragraph one. Platforms using this API to evaluate social media posts, taglines, or short creative prompts should treat all results as uncertain regardless of the reported confidence score.

---

## Spec Reflection

### One way the spec helped

Writing out the exact transparency label text before building any code was the most valuable constraint the spec imposed. When it came time to implement `labels.py`, there was no ambiguity — the `classification` string, `confidence_display` format, and `disclosure` paragraph were already decided. This also forced a decision about what to say to users who receive an uncertain result (specifically: acknowledge it might be a hybrid workflow or a false positive, and point toward the appeal), rather than leaving that as a UI concern to resolve later.

### One way implementation diverged from the spec

The spec defined Signal A (Perplexity) as a mathematical calculation using a local GPT-2 model via the `transformers` library — raw token log-likelihoods, normalized through a sigmoid function. The actual implementation uses Groq as an LLM judge instead. The switch was driven by the project's existing dependency set: `groq==0.15.0` was already in `requirements.txt`, while `transformers` and `torch` would have added several gigabytes of dependencies with no other use in the project.

The tradeoff is meaningful: the Groq approach is not mathematically equivalent to perplexity. True perplexity is a deterministic function of a fixed model; the LLM judge is a semantic assessment that can vary with prompt phrasing. In practice the signal produces good separation between clear AI and clear human text, but it introduces variability that a log-likelihood calculation would not have. For a production system, this would be worth revisiting.

---

## AI Usage

### Instance 1 — Flask skeleton and first signal

**What I directed the AI to do:** Generate a Flask app skeleton with a `POST /submit` route accepting `text` and `creator_id`, assign a UUID tracking ID, and implement `compute_perplexity_score(text: str) -> float` using the GPT-2 transformers pipeline with the sigmoid normalization from the spec.

**What I revised:** The generated code correctly implemented the sigmoid formula and lazy model loading, but the implementation used `transformers` + `torch` which were not in the project's dependency set. I overrode the approach entirely — switching from a local GPT-2 model to a Groq LLM judge that returns structured JSON. The function signature, output contract (`float` in `[0.0, 1.0]`), and normalization direction (1.0 = AI-like) stayed the same; only the mechanism changed.

### Instance 2 — Burstiness signal as a composite metric

**What I directed the AI to do:** Implement `compute_burstiness_score(text: str) -> float` as a composite of two stylometric sub-metrics — sentence-length variance and type-token ratio (TTR) — combined with equal weighting.

**What I revised:** After testing, I found that TTR was systematically reducing burstiness scores on AI-generated text because AI texts tend to use diverse vocabulary (many unique academic and corporate words), so TTR would return a human-leaning score and cancel out the sentence-uniformity signal. More importantly, TTR was already being implemented as its own standalone signal (Signal C — Vocabulary Diversity via MATTR). Having the same metric in two signals is redundant and creates correlated scores that distort the weighted average. I reverted burstiness to pure sentence-length standard deviation, which is what the planning spec described, and moved all vocabulary-based measurement exclusively to Signal C.

---

## Endpoints

| Method | Route | Limit | Purpose |
|---|---|---|---|
| POST | `/submit` | 10/min | Analyze text, return label + confidence score |
| POST | `/appeal` | 100/min | Dispute a classification by `content_id` |
| GET | `/log` | 100/min | Return the 50 most recent audit log entries |

### Running the server

```bash
python app.py
```

Requires a `.env` file with `GROQ_API_KEY=<your key>`.

---

## Audit Log

Every submission writes a structured JSON entry to `audit_log.jsonl` (one record per line, append-only):

```json
{
  "timestamp": "2026-06-25T22:12:57.999582Z",
  "content_id": "5bb99365-b047-4efe-87eb-2a04d0d5ae59",
  "creator_id": "demo-user-1",
  "attribution": "likely_ai",
  "confidence": 0.7419,
  "signals": {
    "perplexity": 0.92,
    "burstiness": 0.85,
    "vocabulary": 0.03,
    "ngram": 1.0
  },
  "label": "Likely AI-Generated",
  "status": "classified",
  "appeal_filed": false
}
```

When an appeal is filed, a second record is appended referencing the same `content_id`. The `GET /log` endpoint merges appeal status back onto the original classification entry at read time:

```json
{
  "timestamp": "2026-06-25T22:12:57.999582Z",
  "content_id": "5bb99365-b047-4efe-87eb-2a04d0d5ae59",
  "attribution": "likely_ai",
  "confidence": 0.7419,
  "signals": { "perplexity": 0.92, "burstiness": 0.85, "vocabulary": 0.03, "ngram": 1.0 },
  "label": "Likely AI-Generated",
  "status": "classified",
  "appeal_filed": true,
  "appeal_status": "under_review",
  "appeal_id": "eab2b050-..."
}
```
