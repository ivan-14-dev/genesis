# GENESIS — an experimental cognitive architecture that learns from (almost) nothing

GENESIS is a research prototype. It is **not** a chatbot and does **not** use any external or pretrained model:
every model, tokenizer and weight it has is built and trained in this repository, on data its administrator
approved (see [Governance](#governance-the-admin-sets-the-rules) and
[In-house language model](#in-house-language-model-built-here-nothing-pretrained)). It starts
with a few innate computational primitives: integer `+ − × mod`, equality/conditional selection, and the
constants 0–3. From there it builds its own model of small worlds whose rules are sampled from a seed and
hidden from it. It works by observing, predicting, choosing informative experiments, and remembering what it
learned along with the evidence for it. A designer-provided bilingual grammar and symbolic inference mechanism
also let it acquire simple facts from teaching; these external assertions are never labelled as verified observations.

- Architecture, scientific rationale, interfaces, memory model, curriculum, roadmap and risks:
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Measured results on held-out seeds: [`docs/RESULTS.md`](docs/RESULTS.md)

## Quickstart

Requires Python ≥ 3.10. Runtime dependencies: none (stdlib + SQLite). pytest is needed for the tests.

```bash
python -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m genesis demo                    # one agent develops through stages 0–7 (~50 s)
.venv/bin/python -m genesis inspect --db runs/demo.db --status refuted
.venv/bin/python -m genesis inspect --db runs/demo.db --explain 12   # why is belief #12 held?
.venv/bin/python -m genesis dashboard --db runs/demo.db             # read-only monitor: http://127.0.0.1:8765
.venv/bin/python -m genesis chat --db runs/demo.db                  # ChatGPT-style entry point: http://127.0.0.1:8766
.venv/bin/python -m genesis chat --db runs/demo.db --terminal       # same, in the terminal
.venv/bin/python -m genesis improve --db runs/demo.db               # stage 7: propose + evaluate, never deploys
.venv/bin/python -m genesis approve --db runs/demo.db --version 2 --by <name>   # human-only deployment
.venv/bin/python -m genesis rollback --db runs/demo.db
.venv/bin/python -m genesis benchmark --seeds 100-109                 # full suite (~15 min, CPU)
.venv/bin/python -m genesis grounded --seed 100 --db runs/objects.db # new file only; object-concept experiment
.venv/bin/python -m pytest -q
```

Administration (all signed with an Ed25519 admin key generated locally; see [Governance](#governance-the-admin-sets-the-rules)):

```bash
G=".venv/bin/python -m genesis"
$G admin keygen                                   # ~/.genesis/admin.key (0600) + admin.pub
$G admin pin --db runs/demo.db                    # pin the admin public key in this memory
$G admin policy-template > policy.json            # edit roles, domains, limits, blocked terms, users
$G admin token --name alice --role user           # token for alice + the hash to paste into policy.json
$G admin policy-install --db runs/demo.db --file policy.json
$G corpus add --db runs/demo.db notes/*.txt       # documents the language model may learn from (pending)
$G corpus list --db runs/demo.db
$G corpus approve --db runs/demo.db --by ivan 1 2 3
$G lm train --db runs/demo.db --kind ngram        # or --kind gpt --dim 64 --n-layers 2 --steps 300
$G lm approve 1 --db runs/demo.db --by ivan       # signed activation (must beat the unigram baseline)
$G lm eval 1 --db runs/demo.db                    # final, audited measurement on the sealed test split
$G audit seal --db runs/demo.db && $G audit verify --db runs/demo.db --pub ~/.genesis/admin.pub
$G chat --db runs/demo.db --admin-pub ~/.genesis/admin.pub   # refuses to start under another admin key
```

## Chat entry point

`python -m genesis chat` serves a ChatGPT-style page: a sidebar with persistent conversation history, message
bubbles, suggestions, FR/EN. **No external language model is behind it.** GENESIS is born with an **innate
bilingual faculty** (`language.py`): a hand-written French/English lexicon (~350 word forms, ~30 concepts) about
its own interface and cognition, plus grammatical cues: elisions and contractions, interrogatives, verb inversion,
negation, light stemming and one-typo tolerance. It is declared at birth in `meta.innate_language` (versioned
digest; upgrades are logged in `innate_language_history`, never silent), and audited to contain **no world
knowledge** (no digits, no rule vocabulary such as "plus", "double", "fibonacci", no environment terms). You can talk
naturally ("Qu'as-tu appris jusqu'ici ?", "How far along is your development?", "c'est quoi une règle ?",
"n'explore pas"). Questions about anything else are parsed and answered honestly ("I understand you are asking about
« capital », « France », but I know nothing about it"). On 38 held-out sentences written after the parser was frozen:
38/38 intents correct, language detection 100%, 0 out-of-scope questions answered as if known (`docs/RESULTS.md`).
What the faculty maps sentences to:

- a sequence of ≥ 4 integers (`3, 7, 11, 15, 19`) is fed term by term through the real cognitive cycle
  (`Genesis.live`): the agent predicts each term *before* reading it, then reports the rule it found (validated,
  candidate or unpredictable), its confidence, its next terms and its prediction trace;
- `continue 23 27` / a few numbers alone add evidence to the last sequence; `predict 5` rolls it out;
- `explain 12` / `why` cites the episodes supporting a belief; `#12` links in replies do the same;
- `what do you know?`, `stage`, `limitations`, `questions`, `explore 100` (autonomous experimentation in a new
  sandboxed hidden world) answer from long-term memory only.

**Words outside the innate faculty are learned, not guessed** (`lexicon.py`): slang, jargon, invented words. GENESIS first says it doesn't understand and asks
which command was meant. It then associates the sentence's words with that command (cross-situational learning,
naive-Bayes combination). The next time, it *proposes* the interpretation ("do you mean `stage`?"). It acts on
its own only once the word→command mappings have been validated by ≥ 2 confirmed predictions, says so, and
accepts `no` as a correction. Repeated errors refute a mapping. Each learned word is a knowledge item
(`kind="lexical"`, provenance `user-taught`) backed by episodes, so `explain` and `vocabulary` show why it is held.
These learned word mappings alone add no new facts. The semantic teaching channel below acquires relations
separately. Ordinary commands still handle one request per message. Numbers typed by the user are
stored as episodes with provenance `user-provided`, kept distinct from the agent's own observations.

`complète : …` / `complete: …` (explicit prefix only) continues a text with the in-house language model, if the
admin approved one. The reply is labelled *unverified, not knowledge*, filtered (personal data, blocked terms),
bounded by the policy (`max_generate_tokens`, time budget) and never stored as a belief.

The server binds to localhost and holds the memory DB for its lifetime (don't run `demo`/`approve` on the same DB
at the same time). Defences, in order: Host allowlist on the bound port (defeats DNS rebinding; add names with
`--allow-host`), same-origin JSON POSTs only (Origin / `Sec-Fetch-Site`), validated `Content-Length` ≤ 16 KB,
socket timeouts, bearer-token identity from the signed policy (`/?token=…` is moved into an HttpOnly SameSite=Strict
cookie), per-identity rate limit, per-message time budget, conversations visible only to their owner, a strict
hash-pinned Content-Security-Policy, and opaque 500 errors (details only in the server log). If the stored policy
no longer verifies, the server answers 503 instead of guessing.

### Teach facts, then ask a new question

The semantic memory (`semantic.py`) starts empty. In the chat, send:

```text
Nala est un chat; Les chats sont des animaux
Nala est-elle un animal ?
pourquoi ?
Milo est un chat
Milo est-il un animal ?
que sais-tu de Nala ?
```

GENESIS derives membership using the taught relations and cites the premise IDs. Neither the names nor their
categories are supplied at birth. English equivalents work: `Nala is a cat; All cats are animals`, `Is Nala an
animal?`, `why?`, `what do you know about Nala?`. The logic of inheritance is innate, **not learned**; this is
symbolic acquisition and reuse of teachings, not unsupervised language learning or human-level understanding.

- Teach several simple assertions separated by periods or semicolons. Negation belongs to each assertion:
  `Nala est un chat; Milo n'est pas un chat`. Mixed unsupported clauses are rejected without partial teaching.
- `Elle est un animal ?` / `Is she an animal?` uses the last unambiguous semantic subject in that conversation,
  even after restart. Multiple subjects require an explicit name.
- `Nala est dans Paris` / `Nala is in Paris` stores a spatial relation; no unprovided spatial rules are inferred.
- `Nala n'est pas un chat` preserves a conflict instead of silently replacing the old assertion. A conflicted
  premise cannot support downstream conclusions.
- `correction : Nala n'est pas un chat` explicitly refutes the opposite teaching. `retire le fait 3` /
  `retract fact 3` retracts a teaching or inclusion rule by its ID. Neither deletes history or episodes.
- Inferences are recomputed from current premises; old answers remain in the transcript with their proofs.
  Correcting a premise does not leave an obsolete conclusion active.

Each teaching records source, timestamp, conversation, supporting/contradicting episodes and the prior model's
agreement or disagreement. It remains **unverified** regardless of repetition. Questions record conditional
inference episodes, not new observations. Unknown is distinct from false; conflicting evidence is reported.
The existing knowledge view, `explain N`, CLI inspection and dashboard expose these records.

**Bounds:** 512 distinct assertion/polarity pairs per memory; no eviction of original experiences. Labels are
single words (hyphens allowed), case/accent-folded, with simple `-s` / French `-aux` plural normalization.
There is no automatic translation of concepts, irregular morphology, arbitrary predicates, quantifier scope,
negative universal rules or SMS interpretation. Ask questions separately from teachings.

Reproduce the transfer/retention experiment (invented labels; no target conclusions taught):

```bash
.venv/bin/python -c 'from genesis.experiments import run_semantic_learning; print(run_semantic_learning(100))'
```

## Governance: the admin sets the rules

GENESIS learns in any domain it is given, **but only under rules its administrator defines** (`governance.py`):

- **Admin key.** `admin keygen` creates an Ed25519 key pair (RFC 8032, implemented in `crypto.py`, stdlib only);
  `admin pin` stores the public key in the memory. Every admin decision (policy, corpus approval, model
  activation, version deployment, audit seal) is a statement signed with it and bound to that memory's random
  id, so it cannot be forged by editing the database or replayed on another memory.
- **Policy** (`admin policy-template` → edit → `admin policy-install`): roles (`admin`, `teacher`, `user`,
  `auditor` by default, or your own) granting actions such as `semantic.teach`, `sequence.learn`,
  `objects.*`, `lm.generate`, `corpus.submit`; domains the admin can switch off (`semantic`, `objects`, `world`,
  `lm`, ...); limits (message size, sequence length, exploration budget, seconds per message, generated tokens,
  requests per minute) that can be lowered but never raised above frozen ceilings; blocked terms; users with
  hashed bearer tokens; whether authentication is required. Unknown keys are rejected, unknown actions denied by
  default, and the serial must increase (no rollback to an older policy).
- **Enforcement** happens inside the engine before any write: a refused request is answered ("I can't do that
  (`semantic.teach`): your role does not allow it; nothing was learned or changed"), and a message containing a
  blocked term is not even stored in the transcript.
- **Audit chain.** Every write and every refusal is appended to `audit_log`, each entry hashing the previous one.
  `audit verify` detects edits, deletions and (with `audit seal`) truncation; it also re-checks corpus documents,
  model weights and the policy signature. The DB file and checkpoints are created with mode 0600.
- Without an installed policy, the built-in default applies (local user with the `teacher` role, no auth), which
  preserves the original single-user behaviour.

## In-house language model (built here, nothing pretrained)

The GPT-like part is written from scratch in this repository, with no external model, tokenizer, weights or
dataset, and no dependency beyond the standard library:

- `lm/bpe.py`: byte-level BPE tokenizer whose merges are learned from the approved documents;
- `lm/ngram.py`: n-gram language model with interpolated Kneser–Ney smoothing (+ unigram baseline);
- `nn/`: tensors with reverse-mode autograd, AdamW, a decoder-only Transformer (`TinyGPT`: learned positions,
  causal self-attention, LayerNorm, GELU MLP, tied input/output embeddings), deterministic training loops, a
  pickle-free weight format, and C kernels (`nn/_kernels.c`: tiled/AVX matmul with optional 2-thread split,
  attention, layer norm, GELU, cross-entropy, AdamW) compiled locally on first use, self-tested against a
  pure-Python fallback, bit-for-bit deterministic whatever the SIMD level or thread count
  (`GENESIS_NN_THREADS`);
- `corpus.py`: what the model may learn from. Submitted documents are normalised, scrubbed of personal data
  (e-mails, phone numbers, IBANs, Luhn-valid card numbers, IP addresses), deduplicated, split train/valid/test
  by hash, and **used only once the admin has signed their exact SHA-256**; the test split is read only for final
  evaluation, and each read is audited;
- `lm/registry.py`: training records full provenance (document hashes, tokenizer digest, code and kernel digests,
  seed, `pretrained: null`) and reports validation perplexity and bits per byte next to a unigram baseline. A model
  becomes active only by admin signature, and only if it beats that baseline (unless `--force`). The weights are
  re-hashed and the signature re-checked every time the model is loaded.

Scale is deliberately modest (2 CPUs, no GPU): an n-gram model trains in well under a second on thousands of
sentences; an ~80 k-parameter TinyGPT training step (4 windows × 64 tokens) takes ~13 ms with the C kernels
versus ~1.6 s in pure Python. On the synthetic test corpus
(subject–verb–object sentences), the 3-gram model reaches a validation perplexity of ≈2.6 versus ≈19 for the
unigram baseline. This is a small text predictor, not a general assistant: its output is shown as unverified text
and never enters the agent's knowledge, whose beliefs still come only from evidence.

## What the demo shows (seed 0)

### A separate grounded object experiment

`python -m genesis grounded --seed 100 --db runs/objects.db` starts with **no learned object concepts**.
It experiments on simulated specimens with five opaque sensor channels and two opaque actions. Hidden laws are
sampled by the simulator; the learner receives only sensor readings and action outcomes through the sandbox.

The learner discovers which sensor subsets suffice to predict each action and groups objects by those learned
projections. Predictions are made before actions. Novel-input forward checks and one-channel interventions
support validation; repeated identical examples cannot validate an abstraction. Concept records cite the original
episodes in ordinary GENESIS memory, alongside historical refutations and model revisions.

The command reports accuracy on unseen object IDs with new distractor values, comparison with rote lookup and the
existing symbolic learner, one-action goal success, persistence, retention across another world and adaptation to
a changed hidden rule. Evaluation outcomes are not fed back into training. It preserves the database and refuses
to overwrite an existing file:

```bash
.venv/bin/python -m genesis inspect --db runs/objects.db --status validated
.venv/bin/python -m genesis dashboard --db runs/objects.db
```

In the measured ten-seed experiment, final transfer and goal success were 100%, with no measured forgetting.
**The existing symbolic learner also reached 100%**. Active exploration was not consistently better than random
at smaller budgets. See `docs/RESULTS.md` for the controls and limitations.

This adds learned, task-relative representations, **not human-like general comprehension**. Sensor formats,
feature-subset search and the notion of an action effect are innate. No word such as "fragile" is assigned to the
learned groups in this experiment; the object workshop below adds explicit lexical teaching.
The world is deterministic and categorical, with only
one-step goals. Unknown relevant sensor values lead to abstention. Each world retains at most 512 experimental
transitions; exceeding the limit stops explicitly without deleting experiences. No pretrained model, neural
network dependency or external service is used.

### Learning words from objects in the chat

Start `python -m genesis chat --db runs/words.db`, then send these messages separately:

```text
atelier objets
explore objets 80
observe specimen-0
teste a0
cet objet est dax
```

`dax` has **no preset meaning**. Choose a consistent naming convention from the displayed sensor readings or
observed effects, and teach contrasting examples on different specimens: `cet objet est dax` or
`cet objet n'est pas dax`. For example, you can choose to label objects whose observed `a0` effect is 1;
use the negative sentence when that effect is 0, rather than blindly copying the positive example above.
The suggested next specimen prioritizes ambiguity. One label or repeated labels on the same configuration
cannot validate a word. At least three correct predictions on new configurations and both label polarities
are required after the compatible meanings stabilize.

After teaching two words, ask `cet objet est-il dax ?`, `montre un objet dax et pas mip`, or
`pourquoi le mot dax ?`. English frames include `object workshop`, `explore objects 80`,
`this object is dax`, `this object is not dax`, `Is this object dax?`, and
`show an object dax and not mip`. `correction : cet objet est dax` explicitly supersedes earlier annotations
for that configuration without deleting their history.

The learner links labels to sensor-value or action-effect predicates, **not command names or specimen IDs**.
It retains competing meanings, abstains on uncertainty, and selects objects only using validated word
interpretations. New-object effect predictions additionally require a validated action model.
Annotations remain external teachings; validation measures agreement with the teacher, not universal meaning.
Evidence and vocabulary persist in SQLite, shared within a world; the selected object is conversation-local.
An observed change in the world's effects requires new lexical grounding.
`vocabulaire` / `vocabulary` lists these interpretations separately from word-to-command associations.

In the fixed ten-seed experiment, 48 labels following 80 world experiments yielded 1,280/1,280 correct
held-out word classifications and 40/40 exact composed selections. This is a small, deliberately constrained
test, **not human-level general understanding**. Objects are structured sensor vectors, not images;
the original sentence frames and logical composition operators are supplied. The additional construction
learner below learns new surface forms, not those underlying operators. See `docs/RESULTS.md`.

```bash
.venv/bin/python -c 'from genesis.experiments import run_grounded_word_learning; print(run_grounded_word_learning(100))'
```

### Learning new sentence constructions

After grounding the words `dax`, `mip`, `zup` and `tov` with object examples in the workshop, teach:

```text
apprends phrase : prends dax sans mip => dax et pas mip
apprends phrase : prends mip sans dax => mip et pas dax
apprends phrase : prends zup sans tov => zup et pas tov
apprends phrase : prends tov sans zup => tov et pas zup
```

GENESIS induces a reusable construction with word slots, including which slot is negated. It can then interpret
`prends dax sans tov` without having been taught that sentence. Neither `prends ... sans ...` nor its negation
assignment is installed as a built-in construction. English teaching uses
`learn phrase: fetch dax without mip => dax and not mip`.

The right side is an explicit teacher-provided meaning, not a meaning discovered autonomously from raw text.
All target words must already have object examples. Three correct **prior** predictions on new slot bindings,
with variation in every slot, are required before automatic selection. Repetition does not validate.
Conflicting constructions abstain; `correction : apprends phrase : ... => ...` supersedes the earlier
demonstration of that sentence, preserves history and resets validation. `vocabulaire` and `explain <id>` expose
the learned forms and evidence. Existing explicit commands retain priority; learned forms can only select objects.

**This does not deliver human-level general understanding.** Flat slot substitution and a small set of logical
operators remain innate. No recursive syntax, quantifier scope, general reasoning algorithm or open-domain
semantics is learned. Completely eliminating programmed learning mechanisms is not a meaningful acceptance
criterion; learning new constructions is a measurable intermediate step, not a solution to general intelligence.

```bash
.venv/bin/python -c 'from genesis.experiments import run_construction_learning; print(run_construction_learning(100))'
```

### Original developmental demonstration

- **Stage 0:** the memory starts empty, with no domain knowledge and no pretrained models. This is audited and
  stored at birth.
- **Stage 1:** curiosity chooses which of 10 integer streams to sample. The agent rediscovers rules such as
  `x[t-1] + x[t-2]`, `x[t-1] * -2` and `t*t + 9` from 17–21 observations per stream, and extrapolates them
  correctly to unseen terms. It judges the two noise streams *unpredictable* rather than inventing a rule.
- **Stage 2:** watching passively gives 100% accuracy on naturally occurring events but only about 20% on
  arbitrary configurations, because a confounded variable mimics the true cause.
- **Stage 3:** about 70 experiments chosen by expected information gain identify every hidden rule. This
  includes the confounded one, where the wrong belief `if p0 == …` is refuted by a specific episode and the
  correct cause is validated. Held-out accuracy reaches 100%.
- **Stage 4:** after a restart and after learning a second world, accuracy on the first world is unchanged.
- **Stage 5:** planning over the learned model reaches unseen goals in every task.
- **Stage 6:** given four new worlds (one pure noise) and one budget, the agent decides where to spend it by
  measured learning progress and sets its own objectives (34 of 38 achieved here). Controls with fresh agents
  are shown honestly: on this seed random allocation scores 86.6% vs 84.7% for progress and 80.6% for
  round-robin.
- **Stage 7:** it lists its own limitations, then runs a sandboxed self-improvement cycle. `validation_k` 4 → 3
  wins on trial seeds (+0.05) but not on holdout (76% vs 76%), so it is *rejected*. A variant that passes would be
  stored as *pending approval*, **never** deployed automatically; only a named human can approve it, and rollback
  is available.

Every belief is stored with provenance, confidence, status history, and the episodes that induced, supported
or refuted it (`inspect --explain`).

## Headline results (10 held-out seeds, see docs/RESULTS.md)

- Sequences: 100% extrapolation accuracy on hidden rules (baselines 7.6% / 17%); 20/20 noise streams judged
  unpredictable, 0 false validations; 0.23 s CPU per seed (12× less than v0.2.0, C search kernel).
- Robustness: with 10% corrupted observations, 100% of rule streams validated; with 20%, 85% (98.8% accuracy of
  validated rules). 7-node rules: 98.75% validated with deepening vs 37.5% without. 0 false validations on noise
  in every condition.
- Causal world: 0% → 100% held-out accuracy; 100% on out-of-range values (tabular baseline 46%); confounded
  causes found 0% by watching vs 100% after experiments; curiosity needs 24.5 experiments vs 51.7 for random.
- Memory: retention 1.0 across sessions, forgetting 0.0 after a second world. Planning: 100% vs 78.5% random.
- Tight budget (60 observations): curiosity with budget-aware focus validates 60% of rule streams (random 24%,
  round-robin 0%). Remembered rule shapes cut search 16.5× (CPU −17%); on 7-node worlds they also cut
  observations to validation by 25% (11.6 → 8.8).
- Stage 6: learning-progress allocation beats round-robin at budget 150 (0.849 vs 0.792, half the noise) and
  stops itself at 300 with equal knowledge.
- Negative results, also reported: stage 6 loses to round-robin at budget 100 (0.702 vs 0.723); on standard
  worlds remembered shapes do not reduce observations to validation; curiosity spends slightly more on noise and
  only ties round-robin at budget 100; at 20% noise, geometric streams overflow before they can be learned; stage 7
  finds no variant that survives holdout, and most tunable parameters have no effect.

## Layout

```
genesis/
  innate.py          innate primitives, constants, frozen safety policy, manifest
  expressions.py     program representation + enumeration
  induction.py       learning engine: online Bayesian program induction (noise-tolerant, deepening)
  search.py          program search: Python reference + C kernel (_search.c), deepening, shape instantiation
  native.py          hardened local compilation/loading of the C kernels (private cache, digest-checked)
  perception.py      features, novelty
  attention.py       salience
  memory.py          working memory + SQLite long-term memory (5 memory systems + evidence)
  curiosity.py       expected information gain, knowledge gaps/questions
  metacognition.py   calibration, learning progress, developmental curriculum gates
  consolidation.py   sleep-like replay, refutation, generalisation, causal vs spurious associations
  adapters.py        bridges between the cognitive cycle and each environment
  agent.py           the 12-step cognitive cycle, persistence, planning
  environments/      sandbox + SequenceWorld + CausalWorld + ObjectWorld (hidden rules)
  autonomy.py        stage 6: learning-progress allocation across worlds, self-generated objectives
  self_improvement.py stage 7: sandboxed evaluation of config variants, human approval, rollback, limitations
  dashboard.py       read-only monitoring web UI + JSON API (stdlib only)
  language.py        innate FR/EN faculty: lexicon about itself + grammar cues; declared, versioned, audited
  chat.py            chat engine: maps parses to intents, user sequences learned via the cognitive cycle
  semantic.py        taught relations, conditional inference, correction history and dialogue subject
  grounding.py       learned sensor projections, experimental concepts, revisions and future validation
  grounded_language.py object/effect word meanings learned from examples; bilingual object workshop
  constructions.py   induced sentence frames, word slots and negation assignments with prospective validation
  lexicon.py         learned vocabulary: word→command associations taught by the user, validated by predictions
  chat_server.py     ChatGPT-style web UI + JSON API over chat.py (stdlib only), hardened (see above)
  crypto.py          Ed25519 signatures (RFC 8032) and signed statements, stdlib only
  governance.py      admin policy, roles, limits, guard, tamper-evident audit chain, signed approvals
  corpus.py          admin-approved training corpus: hygiene, PII redaction, dedup, splits, signatures
  lm/                BPE tokenizer, Kneser–Ney n-gram LM, model registry (train/approve/load/generate)
  nn/                from-scratch autograd, AdamW, TinyGPT Transformer, C kernels, safe weight format
  cli_admin.py       `admin`, `audit`, `corpus`, `lm` commands
  experiments.py     benchmarks, baselines and controls
  demo.py, __main__.py
tests/               pytest suite (incl. a static check that agent code never reads hidden world internals)
```

## Honest scope

GENESIS learns only what its hypothesis spaces can express: small integer programs, conditionals,
bounded categorical sensor projections, grounded word predicates, and explicitly taught relations.
Within these bounds, learning is genuine.
The experimental world rules are sampled per seed, never given to the agent, and recovered from its own
interactions. Beyond the supported spaces, the agent reports uncertainty or abstains. The current zero-forgetting
results come from separate per-context models; they are not
a general solution to catastrophic forgetting. Self-improvement is limited to a few allowlisted parameters and
never touches the safety policy or deploys without a human. The in-house language model is small and only as good
as the approved corpus; it predicts text, it does not reason, and its output is never treated as knowledge.
Governance assumes the admin key stays secret and the host is trusted: someone with write access to the DB can
still delete data (detected by `audit verify`, not prevented) and delete seals, so keep the last sealed head hash
outside the DB. Limitations and next steps (noise-tolerant conditionals, rules beyond 7 nodes,
sub-program library learning, neural perception) are listed in `docs/ARCHITECTURE.md`.
