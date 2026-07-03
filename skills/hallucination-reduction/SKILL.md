---
name: hallucination-reduction
description: >
  Reduce LLM hallucination in question-answering, summarization, and RAG
  systems using prompting techniques, chain-of-thought/chain-of-verification,
  temperature settings, and RAG system design. Use when building or debugging
  any LLM feature where factual reliability matters — "the model is making
  things up", "wrong numbers in answers", "answers questions it shouldn't",
  "add citations/grounding", "design a RAG pipeline", or "measure
  hallucination". Provider-agnostic: works on top of any LLM (Claude, GPT,
  Llama, etc.).
---

# Hallucination Reduction on Top of Any LLM

Hallucination is not one problem. Attack it in layers, cheapest first:

1. **Don't let the model answer what it can't know** (retrieval + abstention gate)
2. **Constrain how it answers** (grounding contract, citations, low temperature)
3. **Check what it answered** (claim decomposition + chain-of-verification)
4. **Repair or remove what fails** (selective repair, downgrade to abstention)
5. **Measure what remains** (groundedness score, hallucination-rate benchmarks)

A reference implementation of this whole recipe (the VERITAS pipeline) lives in
this repo: `src/veritas/`. Copy the prompts from `src/veritas/prompts.py`.

---

## 1. Prompting techniques

### The grounding contract

The single highest-leverage prompt change for RAG-style tasks. Put ALL of
these elements in — each closes a distinct hallucination path:

```text
Answer the question using ONLY the evidence chunks below.

Rules:
1. Every sentence of your answer MUST end with citations of the chunks that
   support it, e.g. "Water boils at 100 C at sea level. [c2]"
2. Use only facts that appear in the evidence. Do not add background
   knowledge, estimates, or plausible-sounding details.
3. If the evidence does not contain the answer, reply exactly:
   "I don't have enough evidence in the provided documents to answer this."
4. Be concise: 1-4 sentences.

EVIDENCE:
[c1] ...
[c2] ...

QUESTION: ...
```

Why each rule works:

- **Mandatory per-sentence citations** force the model to bind every sentence
  to evidence; sentences it cannot bind become conspicuous to the model
  itself and to any downstream checker. Number the chunks (`[c1]`, `[c2]`) —
  citing "the context" is too easy to fake.
- **Explicit permission to abstain** matters more than the prohibition.
  Models are heavily biased toward being helpful; without a sanctioned "I
  don't know" escape hatch, the prohibition alone loses to that bias. Give
  the *exact* refusal string so you can detect abstention programmatically.
- **Evidence before question** (not after): the model conditions on the
  evidence while reading the question, and long-context attention favors
  the end of the prompt — put the question, the most volatile part, last.
- **"Do not add background knowledge"** names the failure mode precisely.
  Vague negatives ("don't hallucinate", "be accurate") do almost nothing;
  the model doesn't *know* it's hallucinating.

### System-prompt framing

Set the persona to value abstention over helpfulness:

```text
You are a rigorously grounded assistant. You answer ONLY from the evidence
provided. You never guess. Saying "I don't know" is always better than an
unsupported answer.
```

### Prompting pitfalls

- **Don't threaten or SHOUT.** `CRITICAL: NEVER make things up!!` doesn't add
  signal; on strongly instruction-following models it causes over-refusal
  instead. State the contract calmly and precisely.
- **Don't ask for confidence scores as your only defense.** Self-reported
  confidence is poorly calibrated. Verify claims against evidence instead
  (§3).
- **Few-shot the abstention.** If you use examples, include at least one
  where the correct output is the refusal string — otherwise the examples
  teach "always answer".
- **Beware summarization-induced hallucination.** "Summarize" invites
  smoothing-over. Prefer "Extract the claims the document makes about X;
  quote numbers exactly."

## 2. Chain-of-thought and chain-of-verification

Plain CoT ("think step by step") helps *reasoning* errors but can **amplify**
confabulation on factual recall: each reasoning step can introduce an
unsupported premise that later steps treat as fact. Use structured
verification instead:

### Chain-of-Verification (CoVe) — the 4-step loop

1. **Draft** an answer (grounded prompt, §1).
2. **Decompose** the draft into atomic claims — one self-contained factual
   statement each, pronouns resolved, temperature 0:

   ```text
   Break the answer below into atomic factual claims. One claim per line,
   numbered. Each claim must be a single self-contained factual statement.
   Keep citation markers attached to their claim.
   ```

3. **Verify each claim independently** against the evidence, temperature 0.
   Verification is *easier* than generation — checking one claim against
   given text is a reading task, not a recall task — so even the same model
   that hallucinated the draft catches most of its own fabrications when
   asked this way:

   ```text
   You are verifying one factual claim against evidence. Think step by step:
   1. What does the claim assert (entities, numbers, relations)?
   2. Which evidence sentences are relevant?
   3. Does the evidence entail the claim, partially support it, or not
      support it?
   Output the final line as exactly one of:
   VERDICT: SUPPORTED | VERDICT: PARTIAL | VERDICT: UNSUPPORTED
   ```

   Key details: verify claims **one at a time** (batch verification lets one
   strong claim carry weak neighbors), in a **fresh context** that contains
   only the claim and the evidence (so the model can't be anchored by its own
   draft), and force a **machine-parseable verdict line**.

4. **Revise**: rewrite unsupported claims from the evidence, or delete them.
   If most of the draft failed, abstain entirely — a mostly-wrong answer is
   worse than no answer.

### Add a model-free second judge

Pair the LLM verifier with a deterministic lexical-entailment check (content
-token overlap + strict number matching between claim and evidence). It's
crude, but it cannot be sweet-talked, and fabricated **numbers** — the most
damaging hallucination class — light it up instantly. Fuse conservatively:
a claim counts as SUPPORTED only if *both* judges agree; any disagreement
demotes it. (Implementation: `src/veritas/verification.py`.)

### When to use which

| Task | Technique |
|---|---|
| Multi-step math/logic | Plain CoT is fine (errors are reasoning, not recall) |
| Factual QA, RAG | Grounding contract + CoVe; avoid free-form CoT before the answer |
| Long-form generation | Generate, then CoVe the factual claims only |
| Classification/extraction | No CoT; strict output schema, temperature 0 |

## 3. Temperature settings

Sampling temperature controls how far the model strays from its most-probable
continuation. Hallucination-sensitive stages should run cold:

| Task | Temperature | Why |
|---|---|---|
| Claim verification, fact checking | **0.0** | The checker must not introduce entropy of its own |
| Claim decomposition, extraction, classification | **0.0** | Deterministic transforms; any creativity is noise |
| Grounded QA / RAG answering | **0.0–0.2** | Slight variation tolerable; keep near-greedy |
| Faithful summarization | **0.0–0.3** | Wording freedom without inviting new "facts" |
| Repair / correction passes | **0.0** | Corrections must come from evidence only |
| Brainstorming, creative writing | 0.7–1.0 | Hallucination is the feature, not the bug |

Practical rules:

- **Schedule temperature per stage, not per app.** One global temperature is
  a design smell: generate at 0.1, verify at 0.0 (see
  `STAGE_TEMPERATURES` in `src/veritas/prompts.py`).
- **Set temperature OR top_p, not both.** They interact multiplicatively and
  several APIs reject the combination outright.
- **Temperature 0 does not stop hallucination.** It removes sampling noise;
  the most-probable continuation can still be a confident fabrication. Cold
  sampling is a floor-raiser, not a fix — you still need §1 and §2.
- **Some models don't expose sampling knobs.** Claude Opus 4.7+/Sonnet 5/
  Fable 5 reject `temperature`/`top_p`/`top_k` (HTTP 400). Omit the
  parameters there and rely on the prompting contract — an adapter should
  drop the knob per model family (see `_NO_SAMPLING_PREFIXES` in
  `src/veritas/llm.py`), not per call site.

## 4. RAG system design

Retrieval quality upstream bounds hallucination downstream: the model can only
be as grounded as the evidence you hand it.

### Chunking
- Sentence-aware windows of ~2–4 sentences with 1 sentence of overlap.
  Too-large chunks bury the answer in noise; too-small chunks orphan facts
  from their context (pronouns, units).
- Give every chunk a stable id — those ids are the citation vocabulary.

### Retrieval
- Hybrid retrieval (lexical BM25 + a semantic signal) beats either alone;
  lexical matching is what keeps rare entities and exact numbers findable.
- Retrieve a small top-k (3–6). More context ≠ more grounding; irrelevant
  chunks give the model raw material to misattribute.

### The abstention gate (the biggest single win)
The largest class of RAG hallucination is answering questions the corpus
cannot support. Before calling the LLM at all, compute an evidence-confidence
signal — e.g. what fraction of the query's content terms the retrieved set
covers, blended with the best retrieval score — and **abstain below a
threshold**. Tune the threshold on a dev set with both answerable and
unanswerable questions: push it up until false abstentions on answerable
questions just start to appear, then back off. (Implementation:
`HybridRetriever.confidence` in `src/veritas/retrieval.py`.)

### The verification loop
Wire §2's CoVe between generation and the user: decompose → verify (two
judges) → repair-or-remove → **downgrade to abstention when > ~50% of the
draft's claims fail**. Ship a groundedness score (fraction of supported
claims) with every answer so callers can gate on it.

### Measure it
You cannot reduce what you don't measure. Build a benchmark with three
question types — answerable, unanswerable, and adversarial (mentions your
entities but asks for details the corpus lacks, or has a false premise) — and
track at minimum:

- **hallucination rate** (questions whose answer contains ≥1 unsupported claim,
  or that answered an unanswerable question)
- **unsupported claim rate** (claim-level)
- **abstention recall** on unanswerable + **false abstention rate** on answerable
- **answer accuracy** on answerable (so grounding gains aren't bought with
  useless refusals)
- **citation precision** (does the cited chunk actually support the claim?)

Grade with a judge that is independent of the generator (a model-free lexical
judge, a different model, or humans). Reference harness:
`benchmarks/run_benchmark.py` in this repo.

## Quick checklist

- [ ] Evidence-only contract with numbered-chunk citations, evidence before question
- [ ] Explicit, exact-string permission to abstain
- [ ] Retrieval confidence gate that abstains *before* generation
- [ ] Generation at temperature ≤ 0.2 (or knob omitted where unsupported)
- [ ] CoVe: decompose → verify per-claim at temperature 0 in fresh context → repair/remove
- [ ] Model-free second judge with strict number matching
- [ ] Downgrade mostly-unsupported answers to abstention
- [ ] Groundedness score attached to every answer
- [ ] Benchmark with answerable / unanswerable / adversarial splits before and after changes
