# GENESIS — an experimental cognitive architecture that learns from (almost) nothing

GENESIS is a research prototype. It is **not** a chatbot and does **not** use any pretrained model. It starts
with a few innate computational primitives: integer `+ − × mod`, equality/conditional selection, and the
constants 0–3. From there it builds its own model of small worlds whose rules are sampled from a seed and
hidden from it. It works by observing, predicting, choosing informative experiments, and remembering what it
learned along with the evidence for it.

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
.venv/bin/python -m pytest -q
```

## Chat entry point

`python -m genesis chat` serves a ChatGPT-style page: a sidebar with persistent conversation history, message
bubbles, suggestions, FR/EN. **There is no language model behind it.** GENESIS is born with an **innate
bilingual faculty** (`language.py`): a hand-written French/English lexicon (~350 word forms, ~30 concepts) about
its own interface and cognition, plus grammatical cues: elisions and contractions, interrogatives, verb inversion,
negation, light stemming and one-typo tolerance. It is declared at birth in `meta.innate_language` (versioned
digest; upgrades are logged in `innate_language_history`, never silent), and audited to contain **no world
knowledge** (no digits, no rule vocabulary such as "plus", "double", "fibonacci", no environment terms). You can talk
naturally ("Qu'as-tu appris jusqu'ici ?", "How far along is your development?", "c'est quoi une règle ?",
"n'explore pas"). Questions about anything else are parsed and answered honestly ("I understand you are asking about
« capital », « France », but I know nothing about it"). On 38 held-out sentences written after the parser was frozen:
38/38 intents correct, language detection 100%, 0 out-of-scope questions answered as if known (`RESULTS.md`).
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
Learned words are word→command associations only; the innate faculty handles one request per message and has no
open-domain meaning. Numbers typed by the user are
stored as episodes with provenance `user-provided`, kept distinct from the agent's own observations. The server
binds to localhost, accepts only same-origin JSON POSTs ≤ 16 KB, and holds the memory DB for its lifetime (don't
run `demo`/`approve` on the same DB at the same time).

## What the demo shows (seed 0)

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
  are shown honestly: on this seed random allocation scores 86.2% vs 84.7% for progress and 80.6% for
  round-robin.
- **Stage 7:** it lists its own limitations, then runs a sandboxed self-improvement cycle
  (`validation_k` 4 → 3: holdout 72% vs 71%). The change is stored as *pending approval* and is **not**
  deployed; only a named human can approve it, and rollback is available.

Every belief is stored with provenance, confidence, status history, and the episodes that induced, supported
or refuted it (`inspect --explain`).

## Headline results (10 held-out seeds, see docs/RESULTS.md)

- Sequences: 98.75% extrapolation accuracy on hidden rules (baselines 7.6% / 16%); 20/20 noise streams judged
  unpredictable, 0 false validations; 1 failure (an unproposed modulus).
- Causal world: 0% → 100% held-out accuracy; 100% on out-of-range values (tabular baseline 46%); confounded
  causes found 0% by watching vs 100% after experiments; curiosity needs 24.5 experiments vs 51.7 for random.
- Memory: retention 1.0 across sessions, forgetting 0.0 after a second world. Planning: 100% vs 78.5% random.
- Tight budget (60 observations): curiosity with budget-aware focus validates 60% of rule streams (random 24%,
  round-robin 0%). Remembered rule shapes cut search 16.7× (CPU −6%).
- Stage 6: learning-progress allocation beats round-robin at budget 150 (0.849 vs 0.792, half the noise) and
  stops itself at 300 with equal knowledge.
- Negative results, also reported: stage 6 loses to round-robin at budget 100 (0.702 vs 0.723); remembered
  shapes do not reduce observations to validation; curiosity spends slightly more on noise; the stage 7 gain is
  small (+0.007 on holdout) and most tunable parameters have no effect.

## Layout

```
genesis/
  innate.py          innate primitives, constants, frozen safety policy, manifest
  expressions.py     program representation + enumeration
  induction.py       learning engine: online Bayesian program induction
  perception.py      features, novelty
  attention.py       salience
  memory.py          working memory + SQLite long-term memory (5 memory systems + evidence)
  curiosity.py       expected information gain, knowledge gaps/questions
  metacognition.py   calibration, learning progress, developmental curriculum gates
  consolidation.py   sleep-like replay, refutation, generalisation, causal vs spurious associations
  adapters.py        bridges between the cognitive cycle and each environment
  agent.py           the 12-step cognitive cycle, persistence, planning
  environments/      sandbox + SequenceWorld + CausalWorld (hidden rules)
  autonomy.py        stage 6: learning-progress allocation across worlds, self-generated objectives
  self_improvement.py stage 7: sandboxed evaluation of config variants, human approval, rollback, limitations
  dashboard.py       read-only monitoring web UI + JSON API (stdlib only)
  language.py        innate FR/EN faculty: lexicon about itself + grammar cues; declared, versioned, audited
  chat.py            chat engine: maps parses to intents, user sequences learned via the cognitive cycle
  lexicon.py         learned vocabulary: word→command associations taught by the user, validated by predictions
  chat_server.py     ChatGPT-style web UI + JSON API over chat.py (stdlib only)
  experiments.py     benchmarks, baselines and controls
  demo.py, __main__.py
tests/               pytest suite (incl. a static check that agent code never reads hidden world internals)
```

## Honest scope

GENESIS learns only what its hypothesis space can express: small integer programs, plus one conditional for
the causal world. Within that space, learning is genuine. The specific rules are sampled per seed, never given
to the agent, and recovered from its own interactions. Beyond that space, the agent reports
"unpredictable". The current zero-forgetting result comes from separate per-context symbolic rules; it is not
a general solution to catastrophic forgetting. Self-improvement is limited to a few allowlisted parameters and
never touches the safety policy or deploys without a human. Limitations and next steps (noise-robust rules,
sub-program library learning, neural perception) are listed in `docs/ARCHITECTURE.md`.
