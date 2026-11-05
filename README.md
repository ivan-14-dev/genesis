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
.venv/bin/python -m genesis demo                    # one agent develops through stages 0–5 (~10 s)
.venv/bin/python -m genesis inspect --db runs/demo.db --status refuted
.venv/bin/python -m genesis inspect --db runs/demo.db --explain 12   # why is belief #12 held?
.venv/bin/python -m genesis benchmark --seeds 100-109                 # full suite (~10 min, CPU)
.venv/bin/python -m pytest -q
```

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
- **Stage 6:** development stops here explicitly, because that gate is not implemented yet.

Every belief is stored with provenance, confidence, status history, and the episodes that induced, supported
or refuted it (`inspect --explain`).

## Headline results (10 held-out seeds, see docs/RESULTS.md)

- Sequences: 98.75% extrapolation accuracy on hidden rules (baselines 7.6% / 16%); 20/20 noise streams judged
  unpredictable, 0 false validations; 1 failure (an unproposed modulus).
- Causal world: 0% → 100% held-out accuracy; 100% on out-of-range values (tabular baseline 46%); confounded
  causes found 0% by watching vs 100% after experiments; curiosity needs 24.5 experiments vs 58.5 for random.
- Memory: retention 1.0 across sessions, forgetting 0.0 after a second world. Planning: 100% vs 78.5% random.
- Negative results, also reported: curiosity loses to random at a very tight budget (60 observations), and the
  procedural prior has no measurable effect yet.

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
  experiments.py     benchmarks, baselines and controls
  demo.py, __main__.py
tests/               pytest suite (incl. a static check that agent code never reads hidden world internals)
```

## Honest scope

GENESIS learns only what its hypothesis space can express: small integer programs, plus one conditional for
the causal world. Within that space, learning is genuine. The specific rules are sampled per seed, never given
to the agent, and recovered from its own interactions. Beyond that space, the agent reports
"unpredictable". The current zero-forgetting result comes from separate per-context symbolic rules; it is not
a general solution to catastrophic forgetting. Limitations and planned next steps (noise-robust rules, library
learning, stage 6–7 autonomy, neural perception, dashboard) are listed in `docs/ARCHITECTURE.md`.
