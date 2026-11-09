# GENESIS — Scientific and Technical Architecture

This document covers deliverables A–H of the first assignment. Deliverables I and J (the prototype and its
reproducible demonstration) are the `genesis/` package, `python -m genesis demo`, and
`python -m genesis benchmark` (results in [`RESULTS.md`](RESULTS.md)).

Notation: **innate** = present before any experience (fixed by the designers, recorded in the manifest);
**acquired** = constructed by the agent from its own episodes (recorded with provenance and evidence).

---

## A. Feasibility analysis

### What is scientifically realistic now

| Goal | Feasibility | Why |
|---|---|---|
| Discover hidden rules of small symbolic worlds from interaction, starting with only arithmetic/logic primitives | **High** – demonstrated here | Bayesian program induction over a compositional hypothesis space is well understood (Solomonoff induction, Bayesian Occam's razor, DreamCoder-style program synthesis). With small worlds the search is tractable on a CPU. |
| Curiosity that is *not* random (information-gain–driven experiments) | **High** – demonstrated here | Expected information gain over an explicit hypothesis posterior is computable exactly; it ignores unlearnable noise ("noisy-TV" immunity) because hypotheses agree there. |
| Distinguish correlation from causation | **High in sandboxed worlds** – demonstrated here | The agent can intervene (`do(state)`), so confounds in passive data can be broken by experiments (Pearl's interventional rung). |
| Persistent, auditable, non-forgetting memory | **High** for symbolic knowledge – demonstrated here | Per-context symbolic rules with provenance do not interfere with each other. This does **not** solve catastrophic forgetting in distributed neural representations. |
| Perception of raw images/audio from scratch | **Medium** | Requires learned neural encoders (PyTorch). Self-supervised representation learning works, but needs data and compute, and the learned features must be bridged to the symbolic layer. Not implemented yet. |
| Open-ended language understanding from scratch | **Low** within any reasonable budget | Humans need years of grounded multimodal experience. Without a pretrained LLM (which the spec forbids, rightly, for learning claims) this is a long-term research question, not an engineering task. |
| Unrestricted recursive self-improvement | **Deliberately excluded** | Unsafe and unmeasurable. Only bounded, validated self-tuning (priors, strategies) is allowed. |

### Core scientific bets

1. **Learning = constructing programs that predict experience.** The agent's knowledge is a set of small
   programs (rules) composed from innate primitives, scored by a Bayesian posterior (Occam prior × likelihood).
   This makes learning *measurable* (prediction accuracy on unseen data), *inspectable* (rules are readable) and
   *honest* (each rule cites the episodes that induced, supported or refuted it).
2. **Prediction error drives everything.** Surprise triggers re-induction; forward (out-of-sample) prediction
   success is required before a belief is called validated; the metacognitive layer tracks calibration.
3. **Curiosity = expected information gain** about the agent's own hypotheses, plus a decaying novelty bonus.
4. **Developmental gating.** Stages are unlocked by measured performance (curriculum gates), not time.
5. **Start symbolic and small; add neural modules only when a measured need appears** (e.g. raw perception).

### Main risks (see also §H)

- Combinatorial explosion of the hypothesis space (mitigated: observational-equivalence pruning, budgets, size
  limits, meet-in-the-middle deepening, library learning of invented primitives).
- The hypothesis space is an inductive bias chosen by the designers. Rules outside it (e.g. requiring
  division or loops) cannot be learned yet; the agent then reports them as *unpredictable* rather than inventing
  an answer.
- Black-swan induction: a rule can be validated before the evidence that would refute it has appeared. Severe
  validation (§E) withholds a rule until its own predictions, not its prior or the observations it was fitted
  to, separate it from every rival within reach (5 nodes; 7 for rules that err); a rival beyond reach can still
  be the true rule (see §H).
- Noise robustness is bounded by identifiability: a sequence rule must be right far more often than chance, not
  more often than wrong, so a stream with half of its observations corrupted is learned, but not one whose rule
  is right barely more often than a coincidence would be. In the lab, new rules are proposed only below 40%
  mismatches.

---

## B. Cognitive architecture

| Module | File | Scientific principle | Status |
|---|---|---|---|
| Innate core | `innate.py` | Core-knowledge hypothesis (Spelke): minimal domain-general primitives; explicit innate/acquired split | implemented |
| Perception engine | `perception.py`, `adapters.py` | Feature extraction, novelty as inverse familiarity (habituation) | implemented (numeric/symbolic); neural encoders planned |
| Working memory | `memory.py: WorkingMemory` | Limited capacity (Miller/Cowan), salience × recency decay eviction | implemented |
| Long-term memory | `memory.py: LongTermMemory` | Multiple memory systems (Tulving/Squire): episodic, semantic, procedural, associative, metacognitive | implemented (SQLite) |
| Attention engine | `attention.py` | Salience from novelty, surprise and reward (Itti & Baldi "Bayesian surprise") | implemented |
| Learning engine | `induction.py`, `expressions.py` | Bayesian program induction; Occam prior; online posterior update; prediction-error–triggered re-induction | implemented |
| Reasoning engine | `induction.py` (induction), `agent.py` (deduction via rule application, BFS planning), `consolidation.py` (generalisation, causal vs spurious) | Hypothetico-deductive cycle; means-end planning | implemented (basic) |
| Curiosity engine | `curiosity.py` | Intrinsic motivation as information gain (Schmidhuber, Oudeyer, active learning / optimal experiment design) | implemented |
| Decision engine | `agent.py: live`, `adapters.py` | Utility maximisation over candidate experiments/actions; planning over learned model | implemented |
| Metacognitive engine | `metacognition.py` | Calibration (Brier, ECE), learning progress, knowing-what-you-don't-know | implemented |
| Consolidation engine | `consolidation.py` | Systems consolidation / hippocampal replay during sleep (McClelland et al. CLS) | implemented |
| Library learning (sleep) | `library.py` | Wake-sleep program learning (DreamCoder, Ellis et al.): MDL promotion of recurring sub-programs to invented primitives; dreams from procedural memory train a recognition model | implemented (sequences) |
| Environments + sandbox | `environments/` | Controlled micro-worlds with hidden, seed-sampled rules | implemented (sequences, causal lab) |
| Developmental curriculum | `metacognition.py: CURRICULUM` | Piaget-inspired staged development with measurable gates | stages 0–7 gated |
| Autonomous exploration | `autonomy.py: Explorer` | Learning-progress–driven allocation (Oudeyer & Kaplan, intelligent adaptive curiosity); self-generated objectives | implemented (stage 6) |
| Self-improvement | `self_improvement.py` | Bounded, reproducible A/B evaluation of allowlisted parameters on trial + holdout seeds; human approval; rollback | implemented (stage 7) |
| Innate language faculty | `language.py` | Hand-written FR/EN lexicon about the agent itself + grammatical cues → `Parse` (language, concepts, question type, negation, unknown topic words, typo corrections). Like the arithmetic primitives, it is given, declared and audited, not learned. No pretrained model | implemented (59 + 38 test sentences, see RESULTS) |
| Entry point (chat) | `chat.py`, `chat_server.py` | Interaction channel: the innate language faculty's parse is mapped to an intent that routes user input to the cycle (`live` on user-provided sequences) or to memory queries (`explain`, knowledge, stage, limitations, curiosity gaps). Unknown sentences are taught by the user and learned as word→command associations (`lexicon.py`, cross-situational learning; Siskind 1996, Yu & Smith 2007), executed autonomously only after validation by confirmed predictions | implemented (no LLM; stdlib HTTP + vanilla JS, ChatGPT-style UI, persistent conversations) |
| Semantic teaching | `semantic.py` | Explicit membership, inclusion and location; conditional forward inference with premise IDs, conflict blocking, correction history and conversational references | implemented, empty at birth; external teachings stay unverified |
| Grounded representation learning | `grounding.py` | Minimal consistent sensor projections, learned action-relative groups, forward checks and one-channel contrasts; revised on incompatible repeated states | implemented as a bounded object experiment, not general language grounding |
| Observability | `dashboard.py` | Read-only inspection of memory; every element traceable to DB rows (`explain`) | implemented (stdlib HTTP + vanilla JS) |
| Governance | `governance.py`, `crypto.py` | Admin-defined rules: signed policy (roles → actions, domains, limits, blocked terms, hashed tokens), default-deny guard before every write, hash-chained audit log with signed seals, signed approvals bound to one memory | implemented (Ed25519 from RFC 8032, stdlib) |
| Training corpus | `corpus.py` | Data governance: normalisation, PII redaction, dedup, hash-based train/valid/test split, admin signature over each document's hash, sealed test split | implemented |
| In-house language model | `lm/`, `nn/` | Statistical language modelling built here: byte-level BPE, interpolated Kneser–Ney n-grams, decoder-only Transformer with autograd + AdamW, C kernels (tiled matmul, attention, layer norm, GELU, cross-entropy, AdamW); provenance, baselines, signed activation (`lm/registry.py`) | implemented at small scale; output unverified, never knowledge |

## C. Module diagram

```mermaid
flowchart LR
    subgraph World["Sandboxed environment (hidden rules)"]
        W[(SequenceWorld / CausalWorld)]
    end
    SB[Sandbox<br/>whitelist + budget + log]
    W <--> SB

    subgraph Agent["GENESIS agent"]
        IN[Innate core<br/>primitives, constants,<br/>safety policy]
        PER[Perception<br/>features, novelty]
        ATT[Attention<br/>salience]
        WM[Working memory<br/>capacity 9]
        LTM[(Long-term memory<br/>episodic · semantic ·<br/>procedural · associative ·<br/>metacognitive)]
        LRN[Learning engine<br/>Bayesian program induction]
        RSN[Reasoning<br/>prediction · planning ·<br/>generalisation]
        CUR[Curiosity<br/>expected information gain]
        DEC[Decision<br/>choose experiment/action]
        META[Metacognition<br/>calibration · progress ·<br/>stage gates]
        CON[Consolidation<br/>replay · refute · associate]
        LIB[Library learning sleep<br/>invented primitives · dreams ·<br/>recognition model]
        EXP[Autonomy<br/>learning-progress<br/>world selection · objectives]
    end
    SI[Self-improvement sandbox<br/>isolated trial agents]
    HUM([Human approval])
    DASH[Dashboard<br/>read-only]
    CHAT([Chat entry point<br/>command grammar · no LLM])
    GROUND[Grounded explorer<br/>learned sensor projections]
    WORDS[Grounded lexicon<br/>competing word meanings]

    SB -- observation --> PER --> ATT --> WM
    WM <--> LTM
    WM --> LRN
    IN --> LRN
    LRN -- hypotheses --> RSN
    RSN -- predictions --> CUR
    CUR -- utilities --> DEC
    DEC -- action/experiment --> SB
    SB -- outcome --> LRN
    LRN -- prediction error --> META
    LRN -- episodes --> LTM
    META -- stage, reliability --> DEC
    CON <--> LTM
    CON -- refutations, generalisations --> LRN
    LTM -- procedural prior --> LRN
    CON -- validated rules, shapes --> LIB
    LIB -- inventions, recognition prior --> LRN
    META -- progress --> EXP
    EXP -- which world --> DEC
    SI -- pending version --> HUM
    HUM -- approve / rollback --> LTM
    LTM -. read only .-> DASH
    CHAT -- user-provided sequences --> SB
    CHAT <-- explain · knowledge · gaps --> LTM
    CHAT -- object annotations --> WORDS
    WORDS <-- observations · learned effects --> GROUND
    GROUND <-- experiments --> SB
    GROUND <--> LTM
    WORDS <--> LTM
```

### The 12-step cycle (`Genesis.live`)

1. **Observe** – the adapter enumerates candidate situations (streams to sample, experiments to run).
2. **Perceive** – features + novelty signature (`Perception`).
3. **Remember** – recent context from working memory, falling back to long-term memory.
4. **Predict** – each relevant learner's predictive distribution.
5. **Explore** – curiosity utility = EIG + novelty + review-due bonus.
6. **Act** – execute the highest-utility situation through the sandbox (or random / round-robin for controls).
7. **Observe outcome**.
8. **Compare** – correctness, surprise (bits), calibration record.
9. **Learn** – posterior update; re-induction on surprise.
10. **Store** – episode (before learning, so knowledge can cite it), learner state.
11. **Consolidate** – every 50 interactions and at session end.
12. **Adapt** – spaced-review schedule, procedural prior, budget-aware focus (finish the cheapest open problem
    when the budget cannot cover all of them), idle stop when nothing is left to learn. Across worlds, the
    `Explorer` (stage 6) decides where the next 10 interactions go; across versions, stage 7 proposes
    configuration changes that a human must approve.

### Interfaces

- **World → agent:** only through `Sandbox.call(method, ...)`; whitelisted methods per world (`AGENT_METHODS`).
  Hidden members (`_rules`, `_true_next`, `_future`, …) are reachable only by the evaluation harness; a test
  statically checks that agent code never references them.
- **Adapter protocol** (`adapters.py`): `candidates() -> [Situation]` (each with the learner `Query`s it
  informs), `execute(situation) -> Outcome`, `after(situation, outcome)`, `var_names(key)`. New environments plug
  in via a new adapter.
- **Learner protocol** (`RuleLearner`): `observe(inputs, y, episode_id) -> LearnResult`, `predict(inputs) ->
  Prediction`, `best()`, `status()`, `to_state()/from_state()`.
- **Memory API** (`LongTermMemory`): `record_episode`, `upsert_knowledge`, `add_evidence`, `explain(id)`,
  `record_shape`, `upsert_association`, `record_metric`, `checkpoint/restore`.

### Data structures

- `Expr(op, args, value)` – immutable program tree; `shape()` abstracts variables/constants for procedural memory.
- `Hypothesis(expr, log_prior, n_ok, n_bad, fwd_ok, fwd_bad, born_n, probe_sig)` – `fwd_*` count predictions made
  *before* seeing the outcome (true out-of-sample evidence).
- `Prediction(value, confidence, p_null, distribution, hypothesis)`.
- `KnowledgeItem(id, key, kind, statement, expr, confidence, status, provenance, context, ts, ...)`.
- Episodes: `(id, session, ts, env_id, context, provenance ∈ {observation, experiment, goal-directed}, action,
  observation, prediction, outcome, error, surprise)`.

---

## D. Initial knowledge specification

Recorded verbatim at birth in `meta.innate_manifest` (with a SHA-256 digest; the agent refuses to wake if it
changes) and audited in `meta.stage0_audit`.

| Innate (given) | Not given (must be acquired) |
|---|---|
| Integer `add`, `sub`, `mul`, `mod` (positive modulus) | Any rule of any environment |
| `eq`, `if_eq` (conditional selection) | Which variables matter, which are noise |
| Constants 0, 1, 2, 3 | Larger constants (e.g. 7, 5, −2) – proposed from data |
| Mechanisms: enumeration of compositions, Occam prior, likelihood, surprise, novelty counts, reward signal, working/long-term memory | Concepts, categories, causal associations, which actions do what |
| Safety policy (frozen) | Which rule *shapes* tend to work (procedural memory) |
| | Whether something is learnable at all (judged "unpredictable" from evidence) |
| Bilingual FR/EN language faculty (`language.py`): ~350 word forms about GENESIS itself → ~30 concepts, grammar cues (elision, contraction, interrogatives, inversion, negation, stemming, one-typo tolerance) | Any meaning outside that vocabulary (slang, jargon, world topics) – taught by the user (`lexicon.py`) |
| Symbolic teaching grammar and conditional class inheritance (`semantic.py`, declared in the language manifest) | Named entities, membership, inclusion and spatial relations; no example facts are installed at birth |
| Bounded predictive feature abstraction (`grounding.py`, separate `innate_grounding` manifest) | Relevant channels, projected groups, effect tables and empirical validation evidence |
| Sensor/effect predicate space and conjunction/local-negation grammar (`grounded_language.py`, declared in `innate_grounding`) | All object-word meanings, acquired through external examples linked to experiments |

The language faculty is recorded separately in `meta.innate_language` (own digest, version, session installed);
a designer upgrade is allowed but logged in `meta.innate_language_history`. It is interface knowledge, not world
knowledge: `language.audit()` rejects digits, rule vocabulary (`WORLD_TERMS`: "plus", "double", "modulo",
"fibonacci", "arithmetic"...) and environment terms (the tests feed it the hidden rule families and kinds).

Stage-0 audit checks (all must pass): empty memory, `domain_knowledge == []`, `pretrained_models == []`, no
learners, language faculty free of world knowledge. No pretrained model, embedding, dataset or LLM is used anywhere.
The in-house language model (§I) does not change this: it is born empty, learns its tokenizer and weights here
from admin-approved documents only, and records `pretrained: null` with the hashes of everything it read.

### Semantic teaching and reasoning

`SemanticDialogue.respond(cid, text, lang)` runs before command/lexical routing. A full-match, bounded grammar
produces `Fact(subject, relation, object, negative)` values. Supported relations are instance membership,
universal class inclusion and location. The grammar distinguishes assertions from questions; only whole,
supported teaching messages are stored. Several assertions can be separated by periods or semicolons.
References resolve to one conversational subject stored in message metadata; ambiguous references ask for a name.

`SemanticMemory.teach` records an external teaching episode and a `knowledge(kind="semantic")` row using the
existing evidence/history tables. All premises are `user-taught`, never `observation` or `validated`. The schema's
numeric confidence is fixed at 0.5 as an **uncalibrated placeholder**, not an empirical probability; the UI labels
it unverified. Repeating a claim adds provenance, not confidence. Prior agreement is recorded before teaching,
but an error here means disagreement with a user assertion, not an independently measured factual error.

`answer(Fact)` performs bounded forward closure over active inclusion rules. From `x instance A` and
`A subclass B`, it may infer `x instance B`; it never confuses an instance with a universal rule, infers the
converse, or assumes arbitrary relations to be transitive. Missing evidence yields `unknown`, not false.
Opposite premises yield `conflict`; conflicting intermediates are excluded from further propagation. Stable ID
ordering and a visited-class set give deterministic, cycle-safe proofs. Each query records its premise IDs and
statements in an `inference` episode.

Explicit corrections/refutations preserve original teachings and history. Conclusions are computed on demand,
not stored as independent truths, so retracting an inclusion rule immediately invalidates its downstream answers.
Knowledge survives restarts; dialogue subjects are scoped to conversations. A hard limit of 512 distinct
assertion/polarity pairs bounds closure without silently evicting knowledge. This is a deliberately small,
inspectable symbolic learner, not a model of human memory or an open-domain language system.

## E. Learning algorithms per developmental stage

### Object-grounded word acquisition

`GroundedLexicon` connects a taught word to sensor readings or experimentally observed action effects,
not to a command name. The learner maintains a version space of simple meanings: "sensor X equals V" or
"action A produces V". This hypothesis language and the FR/EN sentence frames are innate; word meanings are
not. Positive/negative examples eliminate incompatible meanings. Equivalent meanings can remain ambiguous;
agreement on an object permits a tentative prediction, not a claim that the ambiguity is resolved.

Teaching episodes must retain both the external annotation and the observed object/effect evidence. Validation
requires correct predictions made before new annotations, on distinct sensor configurations; repetition is not
validation. Queries on unseen objects may use validated learned effect predictions, while teaching uses actual recorded
experiments. An effect-model revision requires lexical revalidation. Contradictory annotations are retained;
only an explicit correction supersedes the earlier annotation.

Acceptance criteria: no preloaded word meanings; transfer to new object IDs and irrelevant sensor values;
composition of independently learned words with conjunction and local negation; abstention for unknown words or
unsupported effects; retained evidence, corrections and reload behavior. A simulated teacher is harness-only.
These are tests of constrained lexical grounding, not learned grammar or human-level general comprehension.

`ObjectDialogue` routes explicit workshop sentences before semantic teaching and ordinary command parsing.
The CLI supplies a sandbox factory; the language module never imports a concrete world or reads hidden rules.
`meta["object-dialogue/<conversation>"]` retains the selected specimen/world, while
`meta["grounded-language/<environment>"]` retains examples and validation signatures.
`knowledge(kind="grounded-word")` links external annotation episodes and actual experiment episodes, with
`supports` reserved for prior predictions on novel configurations. A word needs three correct future checks,
no failures for its current interpretation, and both positive/negative labels. Equally compatible meanings
can remain unresolved even when predictions validate. Confidence is empirical annotation agreement, not truth.

Fixed bounds are 64 words, 512 annotations and 128 candidate predicates per world, and four terms per query.
Changing any action revision conservatively invalidates all words in that world, including sensory words.
Explicit corrections require renewed checks if the meaning changes; original annotations remain available.
This lexical module alone learns no syntax; translation, noisy-label models, relational word induction and
visual perception remain unsupported.

### Induced sentence constructions

`constructions.py` adds supervised flat construction induction, not general grammar acquisition. Given a
sentence paired with a conjunction of signed, already grounded word predicates, it aligns each target word
with its unique occurrence and replaces it with a numbered slot. Literal tokens, slot order and the polarity
assigned to each slot are learned from demonstrations. Tokenization, alignment, substitution, conjunction and
Boolean negation are the declared innate mechanisms; the initial construction inventory is empty.

`ConstructionLearner.teach` records its interpretation **before** incorporating a demonstration. A form needs
three correct prior predictions on new binding tuples and at least two different words in every slot.
Contradictory assignments or overlapping forms with different meanings cause abstention. Corrections deactivate
earlier demonstrations of that sentence, retain original episodes and reset prospective validation. Queries
never fit. Only validated constructions can select objects, through the existing grounded lexicon's validation
gates. They cannot invoke actions, change safety policy, or override explicit workshop commands.

`meta["constructions/<environment>"]` stores examples and validation state, never `learner_state`.
`knowledge(kind="construction")` exposes learned forms, status history and annotation evidence. Mechanism
installation is versioned inside the grounding manifest, preserving the frozen core birth manifest.
Bounds: 256 demonstrations, 64 forms, 24 alphabetic tokens per sentence and four distinct target words.
Multiword lexical units, recursion, morphology, missing arguments and general semantic parsing are unsupported.
Grammar survives effect-model revisions, but execution still requires currently grounded word meanings.

Acceptance: no supplied forms; exact interpretation and object selection for untrained word bindings;
retention on reload; no validation from repetition; abstention on unknown or conflicting forms; no fitting on
held-out sentences/objects. The teacher supplies meanings using the explicit interface grammar. This acquires
small interpretation programs, **not the learning algorithm or a general reasoning mechanism**.

### Grounded object experiment: predictive abstractions

The next experiment learns a **representation**, not just a sentence-to-command mapping. Objects expose opaque
integer sensor channels and opaque actions. The agent observes effects only after acting on disposable simulated
specimens. The simulator samples hidden action laws; neither relevant channels nor object categories are supplied
to the learner.

The selected CPU-only mechanism is predictive feature abstraction: enumerate bounded subsets of observed sensor
channels, retain those consistent with experience, and prefer the smallest. Objects sharing a projected state are
tentatively equivalent *for predicting an action's effect*. The number of groups and their memberships are learned.
This is a categorical representation learner, not a neural model and not a claim that the innate subset-search
bias was itself learned. It is compared with the existing symbolic `RuleLearner`, full-state lookup, object-ID
lookup and a majority-outcome baseline using the same training transitions.

Experiments are chosen using competing abstractions' disagreement/unknown predictions, novelty and controlled
one-channel contrasts. Predictions precede outcomes. Candidate concepts require future successful predictions on
new full sensor configurations and intervention evidence before validation. An incompatible repeated full state
starts a new action-model revision, preserving the old episodes; this is a deterministic-world revision policy,
not a reliable change detector in stochastic worlds.

Acceptance criteria: abstain before experience; at least 90% accuracy on unseen object IDs with new distractor
values; exceed rote lookup coverage; demonstrate one-action goal selection from the learned model; retain answers
after reload and learning another environment; recover after a hidden rule change. Report all failures and
baselines, not just averages. Held-out effects must never enter fitting or experiment selection.

Technical risks: the smallest consistent feature subset can be spurious under confounding; novel relevant values
require abstention, not extrapolation; feature-subset search is exponential and must be capped; pure outcome
equivalence is task-relative, not human semantics. There is no open-domain language learning, physical realism,
pretrained model or autonomous modification of safety constraints in this experiment.

Implementation: `GroundedExplorer(agent, sandbox)` reads only the observation/action schema and specimen catalog.
`Projection` searches subsets in increasing size and preserves ambiguity between equally small consistent models.
`predict(features, action)` abstains when models disagree or lack the projected input; `choose_action` uses the
learned effects for a one-step goal. `run(steps)` prioritizes uncertainty, novelty and one-channel changes.
The benchmark also runs `strategy="random"` at the same budgets; no superiority is assumed.

`meta["grounding/<environment>"]` retains training transitions, action-specific revisions and forward-validation
state. `knowledge(kind="grounded-concept")` stores representations with ordinary episode evidence and status
history. Validation needs four correct future predictions on new full configurations, a unique minimal
projection and contrasting intervention evidence for every retained feature. Confidence is the observed forward
success fraction, not a calibrated probability. Old revisions are refuted, not erased; no learned parameters or
experiences are added to the frozen core innate manifest. Mechanism installation/upgrades are recorded separately
in `innate_grounding` / `innate_grounding_history`, preserving old database compatibility.

Fixed bounds are eight sensors, four actions, 64 candidate specimens and 512 retained experimental transitions
per world, in addition to the existing agent/sandbox interaction budgets. The current simulator uses five binary
training sensors and two actions; held-out irrelevant sensors take a new value. The harness alone accesses
hidden rules for scoring, which is not an additional training channel. The last report is stored as
`grounded_evaluation` and shown on the read-only dashboard.

| Stage | Algorithm(s) | Gate to pass (measured) |
|---|---|---|
| 0 Initialization | Manifest + audit | `stage0_audit_passed ≥ 1` |
| 1 Pattern discovery | Online Bayesian program induction over sequence features `x[t-1..3], t`; on-manifold rollout probes define behavioural equivalence; curiosity selects which stream to sample | extrapolation accuracy on 10 unseen terms ≥ 0.8 and false-validation rate on noise ≤ 0.05 |
| 2 Relationships & associations | Same induction over object properties; frame generalisation ("action a can only change {…}"); contrast-based associations Δ = P(change\|q=v) − P(change\|q≠v) | passive prediction accuracy on natural events ≥ 0.8 |
| 3 Causal discovery | Interventional experiments chosen by expected information gain; conditional rules `if q==c then e1 else e2`; one-variable-change experiment designs; association labelled *causal* only if the interventional contrast survives | held-out accuracy ≥ 0.95 on unseen configurations, and every confounded rule's true cause identified |
| 4 Memory & knowledge | Persistence, consolidation/replay, refutation tracking, spaced review | retention after reload ≥ 0.99, forgetting after a new world ≤ 0.02 |
| 5 Reasoning | Forward simulation with learned rules + breadth-first planning to unseen goals | planning success ≥ 0.9 |
| 6 Autonomous exploration | Learning-progress allocation of 10-interaction chunks across worlds (2 probe chunks each, then progress + 0.1/√chunks optimism; stop when no world shows progress ≥ 0.05); knowledge gaps → stored objectives | knowledge gain vs round-robin ≥ 0.02 and noise share ≤ 0.25 |
| 7 Continuous development | One-parameter variants of allowlisted settings, evaluated in isolated agents on trial then holdout seeds (strict gain, no extra false validations); human approval; self-identified limitations | evaluation reproducible, safety policy unchanged, no holdout regression, ≥ 1 limitation identified |

Details of the core learner (`RuleLearner`):

- **Prior:** `−2·ln(size) − ln N(size)` (N = number of distinct programs of that size) + procedural bonus
  `shape_bias·ln(1+successes(shape))`; conditionals pay `−ln(#conditions) − ln 4`.
- **Likelihood:** `(1−ε)` if the program predicts the observation, `ε/R` otherwise. A **null hypothesis**
  ("unpredictable", uniform over R outcomes) competes with every program.
- **Proposal:** bottom-up enumeration with observational-equivalence pruning; constants proposed from data
  (frequent values, differences, ratios, contrasts, the per-input contrast of the two most frequent residuals,
  the three largest values + 1, and moduli never observed, see below) besides innate ones; bias fitting `e + r`
  where `r` is the majority residual, so a few corrupted observations do not prevent the fit.
- **Search kernel (`search.py`, `_search.c`):** the enumeration, scoring and bias fits run in C (compiled locally
  with the same hardened loader as the NN kernels, `native.py`), 10–40× faster than Python. The pure-Python
  version is the specification: both give identical results (fuzzed, and checked by a self-test at load);
  values beyond ±2⁶¹ always use the exact Python path. `GENESIS_SEARCH_BACKEND=python` forces it.
- **Deepening (iterative, up to 9 nodes):** when no enumerated rule (≤ 5 nodes) mispredicts at most 25% of the
  observations, the kernel looks for deeper rules without enumerating them, at 7 nodes then 9
  (`deepen_size`, `deepen_max_size`): bias fits and `mod c` on the largest level, a meet-in-the-middle step (for
  each candidate A and operator, the output B would need on a window of observations is looked up in a hash table
  of enumerated signatures) and an affine step `(A op B) + c`. Windows are 4 observations, up to 8 of them, most
  recent first, so a short noisy stream still contains a clean one. Every hit is checked on all observations.
  Deep searches back off geometrically (next one at n·1.5), and the candidate budget is unchanged, so the
  frozen safety policy and its digest are untouched. A best rule whose errors are *structured* (they share one
  residual: a missed cycle, a wrap-around) does not stop deepening, however few its errors; a commutative pair
  whose left side is larger than any partner could be is skipped before its lookup (C and Python alike).
- **Exact sparse linear induction (`linear.py`):** on the same schedule, and only when the enumerated rules
  leave too many observations unexplained, rules `y = a1·f1 + … + ak·fk + c` are *solved for* instead of
  enumerated: features are the perceived variables and at most one pairwise product (counting as two terms),
  k ≤ 4, small non-zero integer coefficients (|a| ≤ 16). Each subset is solved exactly (fraction-free
  Gauss-Jordan elimination) on every k rows of short windows of k + 1 rows, so a corrupted row is stepped over; a
  solution is kept only if it explains at least 3 observations beyond those it was solved on (a square system
  always has a solution). This reaches 11- and 13-node rules such as `3·x[t−1] − 2·x[t−2] + 5·t − 4`. Solutions
  are ordinary expressions over innate primitives, scored with the Occam prior of their size; when one fits,
  the kernel's deepening phase only looks for *shorter* rules (`t·(t+3)` beats `t·t + 3·t`).
  The same loop runs in C (`gn_linear`), with results identical to the Python reference (fuzzed, and checked by a
  self-test at load):
  - Each system is solved modulo 31-bit primes, and the solution is substituted back exactly.
  - A system is proven singular when the product of the primes it is singular modulo exceeds Hadamard's bound.
  - Systems the kernel cannot settle are handed back to Python one at a time, and the kernel then resumes.
- **Unseen moduli:** for `y = (a·x + c) mod m`, every `y − a·x` lies in one residue class, so the gcd of their
  differences over 5 consecutive observations reveals m although m exceeds every observed value. A modulus must
  exceed its window's outputs and recur in at least two windows (a corrupted observation spoils only the
  windows containing it); at most two are proposed.
- **Noise tolerance: beat chance, not 50%.** A rule right on half of a stream's observations is still about 50
  times better than chance. Passive sequence streams (`genesis.agent.default_configs`) therefore:
  - keep a hypothesis while it is right on a quarter of the observations (`max_mismatch_frac` = 0.75; default
    0.5), so a burst of corruption pauses it instead of refuting it, and propose rules up to that tolerance
    (`search_mismatch_frac` = 0.75);
  - score hypotheses with a corruption rate that may be unknown: with prior weight 0.01
    (`noise_unknown_weight`), the rate is uniform on (0, 1) and integrated out (a Beta function of the counts),
    otherwise it is the nominal 2%. A rule right on half of the observations then gains about log(R / 2) nats
    per observation over the null instead of losing log(R / noise) per error; at low error rates the posterior is
    unchanged;
  - measure chance on the stream itself (`learned_chance`): under the unknown-rate component, corrupted values
    and the null follow a learned value law, a Bayes mixture of uniform(100) and a uniform law on an unknown
    interval (scale-invariant prior). Its current probability per value is the chance of a coincidental hit:
    about 1/10 on digits, 1/100 on unbounded streams. Otherwise a rule right on 4 of 11 digits would look like a
    1-in-100⁴ coincidence. Chance is capped at 1/2 (a law concentrated on one value is a constant rule, not
    chance) and the clean component keeps uniform(100), so a constant stream is still a rule;
  - validate through an anytime-valid test against chance (`validation_chance` = 0.1, `_beats_chance`): the
    forward record's likelihood ratio of "right at an unknown rate above 10%" (uniform prior) to "right at 10%"
    is a nonnegative supermartingale under chance, so by Ville's inequality it ever reaches 1/α (α = 0.01) with
    probability ≤ α, however long the record (checked exactly in the tests). Six right and six wrong forward
    predictions pass; two right in eight do not. The same ratio above 1 keeps a rule *promising*;
  - challenge every rule about to be validated (`_challenge`): the search keeps one program per behaviour on the
    observations, so a rival that differs only on inputs never observed (`(x − 1) mod 10` next to
    `x mod 10 − 1` when no multiple of 10 was seen) can be missing from the pool. Once per rule, the search
    reruns on the observations plus the rule's own probes, labelled with a value no program reaches, so the
    tolerance admits exactly the programs that fit the observations as well (or with one more mismatch, below).
    A rule that already errs also faces a deeper search, up to `deepen_size` = 7 nodes, for rules that explain
    some of its errors: errors that a longer rule predicts are not corruption (`−2·x[t−2] + 8` errs on one term
    in five of the cycle 8, 4, 3, 0, 2, which `((x[t−2] − x[t−1]) mod 11) − 1` predicts);
  - validate a rule only through **severe** tests (`severe_validation`, `_untested_rival`, after Popper and
    Mayo): the evidence must separate it from each rival the challenge found that predicts another value on its
    probes. Two things cannot separate them:
    - *the prior*: it ranks rules, it does not test them. `x[t−3] + 1` fits five terms of a counter modulo 7 as
      well as `(x[t−1] − 2) mod 7` and is 2 nodes shorter, so the posterior alone gave it 99% before the
      wrap-around (a longer rival weighs about 10⁻³ of a shorter one);
    - *the observations the rule was fitted to* (before `born_n`): they are not tests it could have failed. On
      `x[t−2] − 4` with its first term corrupted, `−21 + x[t−1] mod 38` was found after that term, fits it by
      coincidence and predicts every later term as `x[t−2] − 4` does, until t = 14. A rival may therefore err
      once more than the rule, on those observations (`_unseparated`); two more errors would be a second
      coincidence.

    The rule waits for the decisive observation: either the rival errs there (the rule passed a test it could
    have failed), or the rule does. A rule with no such rival is validated as soon as before. A rival that is
    the rule plus a clause no observation exercised is not waited for (`_idle_modulus`): `x[t−1] + x[t−2] mod 27`
    on a Fibonacci stream whose `x[t−2]` stayed below 27 is an exception added beyond the data, which any rule
    can be given, not another explanation of the data. Once an observation exercises its modulus, it is one
    (the counter modulo 7 above wraps in the data). In a harness that checks every validation against the
    noiseless continuation (10 seeds, 11 conditions from clean to 50% noise and 13-node rules), wrong transient
    validations fell from 9 to 0. The lab keeps the old gate (only erring 4–5-node rules are challenged);
  - tell an *incomplete* rule from a corrupted one (`_incomplete_ids`). Corruption strikes at independent times,
    so a corrupted rule's hit/miss record is a sequence of independent draws, and another rule hits a corrupted
    value only by coincidence. Two anytime-valid tests, at level α/2 each, refute this:
    - *pattern* (`_pattern_log_ratio`): a Bayes mixture of Markov models of orders 1–6 (Krichevsky–Trofimov
      estimates) of the hit/miss record, against the record's best fit as independent draws. Under independent
      errors at any rate, the ratio is below a nonnegative martingale, so it reaches 2/α with probability
      ≤ α/2. A rule right on one phase of a cycle (every fourth term) is caught within a few cycles;
    - *rival*: another pooled rule hits its errors far more often than the learned chance (the chance test
      above, level shared among the rivals). Only the errors observed after the rival joined the pool count: a
      rival found by searching those very observations was chosen to hit them. Every rule with errors is
      tested, whatever its error rate: `x mod 30 − 1` on `(x − 1) mod 10` is right on most inputs and wrong on
      a whole region of them, and once `4 + x mod 5` keeps hitting those errors it is refuted.

    An incomplete rule keeps only the fixed-rate component, cannot be *promising* through the chance clause and
    is never validated. A stream that only incomplete rules reach (a third-order recurrence beyond the size
    limit, whose 7-node approximations are each right on one phase in four) is then reported *unpredictable*
    (after 22 observations in the test) instead of staying *unknown*. Likewise, while the best rule is right no more
    often than wrong, a re-induction counts as failed and the search backs off exponentially, instead of
    rerunning after each of its errors.

  Two clamps keep the cost down:
  - A deep hit must err strictly less often than the best enumerated rule, in both backends.
  - A shape's bias fit must be the residual of a strict majority.

  The `posterior()` memo is keyed on the pool list itself (counts change only when `observe()` replaces it), and
  `_merge()` recomputes probe signatures only for new hypotheses while the probes are unchanged.
  The lab keeps the defaults (survival 0.5, proposals below 40% mismatches, fixed corruption rate, no chance
  test): its outcomes are a few small shifts, so chance is high, and half-right conditionals would divert the
  experiments that settle them. Noise is
  *unstructured*: a rule whose errors mostly share one residual (a missed wrap-around, an off-by-one) is
  incomplete and is never validated, whatever its error rate. Conditionals `if q == c then e1 else e0` tolerate
  noise too: each branch takes its exact blocks when there are any, otherwise the blocks with the fewest
  mismatches within that branch's tolerance (mismatch sets are bitmasks, one popcount per block and branch).
- **Exact beliefs, bounded search:** the search kernels use bounded arithmetic (±10¹⁸), but beliefs are
  evaluated exactly (`Expr.evaluate(env, exact=True)`, big integers capped at 4096 bits). Observations beyond the
  bound are left out of the search and every proposal is re-scored on them exactly, so a geometric stream keeps
  being predicted after it overflows.
- **Library learning (wake-sleep, `library.py`):** at each consolidation (*sleep*), sub-programs that recur in
  validated rules of at least two problems are promoted to *invented primitives* when they save description
  length (`uses·(size−1) − size − 2 > 0`, greedy, ≤ 6 inventions; behaviourally equal sub-programs such as
  `x·2` and `x+x` count as one). During *wake*, when the innate language finds nothing that fits at a deepening
  point, the same search runs once more with the inventions as extra leaves (one node each, deepening up to
  `deepen_size`); results are inlined, so beliefs keep using innate primitives only and must still be validated
  by forward predictions. Dreams (validated shapes instantiated with fresh variables and constants, rolled out by
  the domain) and replays of validated problems train a naive-Bayes recognition model `q(shape | coarse series
  features)`; it adds a task-conditional bonus to the prior of familiar shapes and decides which shapes are
  tried when procedural memory exceeds the per-search cap. Everything is recomputed from validated knowledge at
  every sleep, so an invention disappears with the rules that supported it.
- **Recognizing noise:** the recognition model has two reserved classes, dreamt at every sleep: `~noise`
  (uniform random draws) and `~program` (random programs of the base language, 3–11 nodes, so that an
  *unfamiliar* rule family is not mistaken for noise). Features are coarse (monotone, few distinct differences,
  smooth, bounded…), so the recognizer only sets the *prior* log-odds of noise. The hypothesis pool's evidence
  can only lower them: `+ min(0, log p(data | null) − log Σ p(data | rules))` (`RuleLearner._likely_noise`).
  When the series still looks rule-less with p ≥ 0.9 after ≥ 10 observations, a due deep search (deep, linear
  and invention stages) is postponed by n observations: delayed, never forbidden, and validation is unaffected.
  A rule with no salient feature is not postponed once a hypothesis explains most observations; noise keeps an
  empty pool. Dream statistics record their feature set and are re-dreamt when it changes.
- **Validation:** equivalence-class posterior ≥ 0.9 **and** ≥ 4 correct forward predictions **and** no
  structured errors **and** either ≤ 25% forward errors (`validation_max_error`) or, after a burst of corruption,
  Wald's sequential test (SPRT) accepting "errs at 25%" against "errs at 50%" (a rule no better than a coin) at
  level 0.01: a long forward record can carry a few more errors. Sequences also accept the anytime-valid test
  against chance (above), so a heavily corrupted rule can be validated while it errs more often than not. **Unpredictable:** p(null) ≥ 0.9 with
  ≥ 8 observations, unless the best rule is *promising* (≥ 2 forward successes and more successes than
  failures, a 1/100 coincidence each on noise, or, for sequences, a forward record favouring "right more often
  than chance"): a long rule pays its Occam prior back only over several more observations.
- **Procedural memory:** remembered shapes (up to 7 nodes) are instantiated first; a C kernel evaluates every
  instance and only those within tolerance become expressions (identical results to the former Python loop).
- **Re-induction** on surprise with exponential backoff; previously proposed hypotheses keep their statistics.
- **Curiosity:** EIG of the next observation about the hypothesis variable; novelty bonus `1/(1+n)`; resolved
  questions are revisited on a spaced schedule (10, 20, 40 interactions, then graduate).

## F. Persistent memory model (SQLite, `memory.py`)

| Table | Memory system | Key fields |
|---|---|---|
| `episodes` | Episodic | session, ts, env, context, provenance (observation/experiment/goal-directed), action, observation, prediction, outcome, error, surprise |
| `knowledge` + `knowledge_history` | Semantic | key, kind (rule/meta/generalisation), statement, expr, confidence, status (candidate/validated/dormant/refuted/unpredictable), provenance, context, timestamps; every status change is logged |
| `evidence` | Provenance links | knowledge_id ↔ episode_id with relation induced_from / supports / contradicts / refuted_by |
| `procedures` | Procedural | rule shape → successes / failures (biases future induction) |
| `associations` | Associative | a, b, kind (correlational/causal/spurious/inconclusive), weight, n, contrast details |
| `metacog` | Metacognitive | session, metric, value, context (calibration, replay consistency, assessments) |
| `learner_state` | Model state | serialised learners (data, hypotheses, statistics) to resume across sessions |
| `knowledge` (`kind=lexical`) | Learned vocabulary | `lexicon/<word>` → intent, P(intent\|word), support/contra counts, forward predictions right/wrong, status; evidence = `language` episodes (taught/confirmed/rejected) |
| `knowledge` (`kind=semantic`) | External teachings | Membership, inclusion and location; unverified source, conflict/retraction history, supporting episodes |
| `knowledge` (`kind=grounded-concept`) | Experimental abstractions | Learned sensor projections and effect groups; future prediction checks and controlled intervention evidence |
| `knowledge` (`kind=grounded-word`) | Grounded lexical hypotheses | Compatible sensor/effect predicates, external annotations, experiment provenance, novel-input prediction checks |
| `knowledge` (`kind=construction`) | Learned interpretation forms | Literal/slot patterns, signed predicates, demonstrated meanings, prospective checks; state in `meta["constructions/<env>"]` |
| `conversations` + `chat_messages` | Interaction log | conversation title/lang; messages (role, text, meta JSON with intent, knowledge id, status). Data typed by the user enters `episodes` with provenance `user-provided` (external information, distinct from own observation) |
| `meta` | Identity and bounded object models | Core/language/grounding manifests and installation histories, safety digest, stage-0 audit, session counter, review schedule; `grounding/<env>` transitions and revisions, `grounded_evaluation` report; `grounded-language/<env>` lexical state, `object-dialogue/<conversation>` selection; `library` (invented primitives, recognition model, per-shape dream statistics) |

Consolidation never deletes episodes; refuted beliefs are kept with their refuting evidence.
`checkpoint()` / `restore()` provide rollback.

## G. First artificial learning environments

1. **SequenceWorld(seed)** – 10 integer streams. 8 follow rules sampled from the seed across families: arithmetic,
   geometric, additive recurrence, period-2/3, quadratic in t, modular counter, lag-2 arithmetic. 2 are pure
   noise. The agent may only call `stream_ids()` and `next_term(sid)`; it chooses which stream to sample.
2. **CausalWorld(seed)** – 4 objects' properties `p0..p3 ∈ {0..5}` and 4 actions. Each action has a hidden rule
   (add / set / copy / sum), possibly gated by a condition `if pq == c`. At least one condition is
   **confounded** in passive observation (another property mirrors the true cause), so watching alone cannot
   identify the cause. Agent-callable methods: `spec`, `passive_observation`, `experiment(state, action)`,
   `begin_task`, `act`.

Rules are sampled per seed, so they cannot be hard-coded. External information sources are not connected; any
future source must be tagged with provenance `external` and kept separate from experience.

## H. Implementation roadmap and risks

### Roadmap

| Phase | Content | Acceptance criterion |
|---|---|---|
| 1 ✅ | Innate core, induction, memory, curiosity, metacognition, consolidation, sequence + causal worlds, benchmarks, demo | stages 0–5 gates pass on held-out seeds |
| 2 ✅ | **Noise-robust rules**: stochastic likelihoods for conditionals, probabilistic transitions | ≥ 0.9 held-out accuracy at 5% observation noise. *Met:* sequences (majority-residual fits, structured-error check, deep hits that must err less than shorter rules, sequential validation test, exact beliefs, anytime-valid test against chance): 100% at 10%, 20% and 30% innovation noise, 97.5% at 50%; causal lab with noisy conditionals: 100% held-out accuracy and every confounded cause found at 5%, 10% and 20% outcome noise; 0 false validations (RESULTS, robustness). Probabilistic transitions remain open |
| 3 ✅ | **Library learning** (DreamCoder-style): promote recurring validated sub-programs to new primitives | measurable reduction in observations-to-validation on new worlds vs no library. *Met:* library-first search over remembered shapes cuts discovery search 16× on standard worlds (the rival search of severe validation, the same with or without memory, brings the total to 2×) and observations to validation by 25% on 7-node worlds; sleep-time sub-program promotion and dreaming (`library.py`) let a 7-node search budget learn later 9-node rules (curriculum, RESULTS) |
| 4 ✅ | **Stage 6**: multi-world autonomy, learning-progress–based world selection, self-generated goals | beats round-robin world allocation in knowledge per interaction (met at budget 150 and 300; **not** at 100) |
| 5 | **Neural perception** (PyTorch, CPU): self-supervised encoders for grid images/audio emitting discrete symbols to the learner | rule discovery in a pixel-rendered CausalWorld within 2× the symbolic sample count |
| 6 ✅ | **Stage 7 / self-improvement sandbox**: the agent proposes config variants (priors, budgets), evaluated on frozen benchmark seeds in a copy of memory; promoted only after human approval | improvement on held-out seeds, safety digest unchanged |
| 7 ✅ | **Observability service**: read-only API over the memory DB + dashboard (stage, curves, beliefs, `explain`). Built with stdlib `http.server` + vanilla JS instead of FastAPI/React: no measured need yet justifies the dependencies | every dashboard element traceable to DB rows |
| 8 (started) | Simulated physics world; text interaction grounded in learned concepts. *Done so far:* bilingual chat, learned word mappings, semantic teaching with conditional inference, predictive object abstractions, object-grounded word meanings and flat sentence-construction induction (see RESULTS). Ordinary commands still handle one intent; semantic messages can contain several assertions with local negation. No open-domain language learning, general syntax induction, SMS handling or realistic physics | defined when phase 5 is complete |

Dependencies are added only when their phase begins and a measured need justifies them (PyTorch at phase 5;
FastAPI/React only if the stdlib dashboard becomes insufficient; Docker when there are services to orchestrate).

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Search explosion for larger rules | cannot learn | budgets (candidates, seconds) in the safety policy; meet-in-the-middle deepening; library learning (phase 3) |
| Inductive black swan (validated too early) | false knowledge | forward-validation, severe validation against rivals within reach, spaced review, refutation tracking; residual risk documented |
| Designer bias in the hypothesis space | "learning" limited to what designers anticipated | explicit manifest; report "unpredictable" honestly; expand primitives only via documented versions |
| Benchmark overfitting during development | inflated results | report held-out seeds (100–109) not used during development |
| Symbolic-neural bridge | perception may not yield usable symbols | phase 5 has its own measurable criterion |
| Unsafe self-modification | loss of control | frozen safety digest; self-improvement only in sandbox + human approval (signed once an admin key is pinned) |
| Learning outside the admin's rules | unwanted knowledge or behaviour | signed policy checked before every write; default deny; refusals answered and audited |
| Poisoned or personal training data | model reproduces it | corpus gate: admin signature over each document hash, PII redaction, blocked terms, output filtering |
| Local web attacks (DNS rebinding, CSRF, DoS) | memory written by a hostile page, server stalled | Host allowlist, same-origin JSON only, Content-Length validation, timeouts, rate limit, time budget, CSP |

### Known limitations of the current prototype

- Hypothesis space: integer arithmetic/logic, size ≤ 5 enumerated plus sizes 7 and 9 by deepening (sequences),
  or ≤ 3 + one conditional (lab, no deepening). Deepening only finds deep rules of the shapes it composes (bias,
  `mod c`, `A op B`, `(A op B) + c`). Beyond 9 nodes, only *linear* rules are reached directly (≤ 4 terms, at most
  one product, |coefficients| ≤ 16); other 11+-node rules need invented primitives or a raised
  `deepen_max_size`.
- Sequence rules tolerate innovation noise up to 30% (100% validated) and 50% (97.5%), with 0 false validations:
  a rule must beat the chance measured on its own stream, not be right more often than wrong. A stream whose rule
  is right barely more often than a coincidence would be (few distinct values and very heavy corruption) stays
  unvalidated. The tolerance has costs:
  - +20% CPU on clean worlds (0.26 vs 0.22 s per seed), because searches keep candidates with up to 75%
    mismatches;
  - spurious rules keep noise streams open longer (unpredictable after 8.65 observations instead of 8.0), so
    curiosity spends more observations on noise;
  - the structure tests that tell an incomplete rule from a corrupted one need evidence, so an incomplete rule
    whose errors still look independent can be validated before a pattern refutes it, or a rival beyond reach
    of the rival search (5 nodes, 7 for rules that err, or trimmed from the pool) does (black swan).
- Severe validation makes a rule with an untested rival wait for the decisive observation: 9.0 observations to
  validation on clean worlds instead of 8.45, about one more on procedural worlds, and a rival search per rule
  (about 38,000 programs per rule stream, which remembered shapes do not reduce). A rival that only adds a
  modulus no observation exercised does not hold validation back, so a counter observed before its first
  wrap-around is validated without one (as without severity); the wrap-around refutes it.

  The lab keeps the majority requirement (proposals below 40% mismatches).
- The search is bounded by ±10¹⁸ (beliefs are exact): proposals are found on the in-range part of a stream, and
  once values exceed ±2⁶¹ the search falls back to the slower Python path.
- The noise recognizer's features are coarse: some rules look like noise from their features alone (e.g.
  `x[t−1] − x[t−3] + 4`, p ≈ 0.97). The pool's evidence vetoes the postponement once a hypothesis explains most
  observations, but a rule that no hypothesis fits yet can still have its deep searches postponed. The
  recognition model's effect on observations to validation is marginal; it mainly ranks shapes and postpones deep
  searches on likely noise.
- Moduli are found from data (largest values + 1, differences, contrasts) and, for affine modular rules
  `(a·x + c) mod m` with a small multiplier, from the gcd of residue differences; other unseen moduli may be
  missed. The search kernels stay bounded (no `__int128`): beliefs are exact integers, so only the search, not
  prediction, is limited by ±10¹⁸.
- Zero forgetting is a property of per-context symbolic knowledge, not a general solution to catastrophic
  forgetting.
- Equivalence for sequence rules is judged on rollouts of the current best rule (on-manifold), so two rules that
  agree on the trajectory but differ elsewhere are treated as the same belief.
- Stage 6 allocation needs ≥ 2 probe chunks per world, so it loses to round-robin when the budget is very small
  (100 interactions for 4 worlds).
- Stage 7 tunes only a few allowlisted parameters; most have no measurable effect. Since the v0.3 learner
  improvements, no variant survives holdout on the benchmark seeds. It cannot change algorithms or architecture.
- Curiosity spends slightly more observations on noise than round-robin.
- Neural perception is not implemented yet.
- The in-house language model is small (n-gram or ~10⁵-parameter Transformer on 2 CPUs) and only as good as the
  approved corpus. It predicts text; it does not reason, and the chat never stores its output as knowledge.
- Governance trusts the host and the admin key: DB write access can still delete rows and seals (detected by
  `audit verify`, not prevented); keep the last sealed head outside the DB. There is a single admin key (no
  quorum, no rotation command yet).
- The chat server runs one engine behind a lock: a message that uses its whole time budget
  (`max_seconds_per_message`, 30 s by default) delays other clients by that long.

## I. Governance, security and the in-house language model

### Trust model

The admin holds an Ed25519 private key (`admin keygen`, file mode 0600, refused if group/world-readable). Its
public key is pinned in `meta.admin_pubkey`, together with a random `meta.memory_id`. Every admin decision is a
statement `{…, memory_id, ts, by}` signed over `"genesis-v1/" + purpose + "\n" + canonical JSON`, so a signature
cannot be reused for another purpose, another target or another memory:

| Purpose | Signed target | Checked when |
|---|---|---|
| `policy` | the whole normalised policy (serial must increase) | every load; the chat server re-checks on change and answers 503 if it fails |
| `corpus-approval` | document id + SHA-256 + decision | every time training or evaluation reads the document |
| `lm-approval` | model id + SHA-256 of the weights blob | every time the active model is loaded |
| `version-approval` | self-improvement version id + config | deployment (`approve --key`) |
| `audit-seal` | last audit entry id + head hash | `audit verify` (detects truncation) |

### Policy and guard

`validate_policy` rejects unknown keys, unknown actions, limits above the frozen ceilings (`HARD_LIMITS`) and
reserved user names. `Guard.require(action)` is called by the chat engine, the semantic and object dialogues, the
corpus and the model registry **before** anything is written. It denies unknown actions, actions missing from
the caller's role, and actions in a disabled domain. Allowed writes and all refusals go to `audit_log`. Each entry
stores `sha256(prev_hash + canonical(entry))`, so editing or deleting an entry breaks the chain. Identity comes
from bearer tokens whose SHA-256 is listed in the policy, compared in constant time.

### Language-model pipeline

```
documents ──submit──▶ normalise · redact PII · blocked terms · dedup · split by hash ──▶ pending
pending ──admin signs sha256──▶ approved ──(train split only)──▶ BPE merges ──▶ n-gram / TinyGPT
model ──▶ valid PPL + bits/byte vs unigram ──admin signs weights sha256──▶ active ──▶ `complète : …` in chat
test split ──(audited read)──▶ final evaluation only
```

Weights are stored as a zlib-compressed blob (`GLM1` header + JSON tokenizer/n-gram tables + `GNNW1` tensors, no
pickle). Decompression is capped (256 MiB), and the deserialiser accepts only whitelisted model classes. Training
is deterministic: the same corpus, config and seed give the same weights hash. Generation is seeded by the
prompt's hash, bounded by `max_generate_tokens` and the per-message time budget, filtered for PII and blocked
terms, and labelled as unverified text.
