# GENESIS — Measured results (deliverable J)

Reproduce:

```bash
python -m genesis benchmark --seeds 100-109 --out results/benchmark-seeds100-109.json
```

- **Seeds and data:** 10 held-out seeds (100–109), not used during development. Every world's rules are
  sampled from its seed and never shown to the agent. The raw output is in
  [`../results/benchmark-seeds100-109.json`](../results/benchmark-seeds100-109.json).
- **Runtime:** about 11 minutes wall time on one CPU core for the whole suite (three strategies, budget sweep,
  robustness conditions including 9-, 11- and 13-node worlds, noisy causal labs, ablations, the library-learning curriculum,
  stage 6 allocation, one stage 7 self-improvement cycle).
- **Determinism:** results do not depend on `PYTHONHASHSEED` (a set ordering that did was fixed; regression
  test `test_behaviour_independent_of_hash_seed`).
- **Accuracy definition:** every accuracy below is measured against the *hidden* rules, on inputs the agent has
  not experienced.

## Why this demonstrates acquired, not given, knowledge

1. **Nothing is known before experience.** Accuracy before any experience is **0.0**. The manifest lists no
   domain knowledge, and the stage-0 audit verifies that memory is empty.
2. **Rules cannot be hard-coded.** Rules are sampled per seed, and these seeds were unseen during development.
3. **The agent cannot see the rules.** It reaches the world only through a whitelisting sandbox, and a static
   test (`test_cognitive_modules_never_touch_world_internals`) verifies that agent code never references hidden
   members.
4. **It generalises rather than memorises.** It predicts configurations it never experienced, including
   **out-of-range values** it never saw, where a memorising (tabular) baseline fails.

## Stage 1 — pattern discovery (10 streams per seed: 8 rule streams + 2 noise streams)

| Metric | GENESIS | Repeat-last baseline | Linear-extrapolation baseline |
|---|---|---|---|
| Accuracy extrapolating 10 unseen terms (rule streams) | **100%** | 7.6% | 16.8% |
| Rule streams validated | **80 / 80** | – | – |
| Accuracy of *validated* rules | **100%** | – | – |
| Noise streams judged "unpredictable" | **20 / 20** | – | – |
| False validations on noise | **0** | – | – |

Cost per seed:
- **Observations per stream to validation:** 8.45 on average.
- **Interactions:** 194 in total, including spaced reviews.
- **Compute:** **0.31 s CPU** (2.74 s in v0.2.0, 9× less) for about 1.6M candidate programs evaluated; the
  search runs in a C kernel whose results are identical to the Python reference. The 9-node deepening, the
  linear solver and library learning are paid mostly on the two noise streams, which a learner cannot tell from
  an out-of-reach rule until it has looked deeper; the kernel's pruning and the postponement of deep searches on
  series that look like noise brought the cost back from 0.32 s to the level before those features (0.23 s).
  Learning half-corrupted streams (below) raised it again to 0.31 s: sequence searches now keep candidates
  with up to 75% mismatches.
- **Memory:** about 240 KB.
- **Calibration:** ECE 0.068 (0.065 before half-corrupted streams were learnable).

The former single failure (seed 104, stream S6, a modular counter whose modulus was never proposed) is fixed:
constants now include the three largest observed values + 1, not only the maximum + 1.

### Robustness: noisy observations and deeper rules (`run_robustness_benchmark`)

Same seeds, same budget per stream (40 observations). *Noise p* replaces each observed term, with probability
p, by a random value; the stream keeps evolving from the true value (innovation noise). The true rules are
evaluated on clean terms. "Deep" worlds draw rules that need **7 nodes** (e.g. `y = x·(x+3)−2`,
`y = 2·x[t−1] − x[t−2] + 3`, `y = (3·x + 5) mod 17`), beyond the 5-node enumeration. "Deeper" worlds draw
rules that need **9 nodes**: `y = 3·x[t−1] − 2·x[t−2] + c` (or 2 and 3 swapped) and `y = 2·x[t−1] ± 3·t + c`.
"Deepest" worlds draw rules that need **11 or 13 nodes**: `y = x[t−1] − x[t−2] + x[t−3] + k·t + c`,
`y = 2·x[t−1] − x[t−2] + t·t + c` and `y = 3·x[t−1] − 2·x[t−2] + k·t + c`.
In parentheses: for validation, the previous v0.3.0 draft (commit `dd14c19`); for CPU, the commit before
half-corrupted streams became learnable (`267aa4f`); only when different.

| Condition | Rule streams validated | Extrapolation accuracy | Accuracy of validated rules | False validations on noise | CPU / seed |
|---|---|---|---|---|---|
| Clean | **100%** | 100% | 100% | **0 / 20** | 0.27 s (0.22 s) |
| Noise 10% | **100%** | 100% | 100% | **0 / 20** | 0.59 s (0.44 s) |
| Noise 20% | **100%** (92.5%) | 100% (96.3%) | 100% | **0 / 20** | 0.97 s (1.57 s) |
| Noise 30% | **100%** | 100% | 100% | **0 / 20** | 1.37 s |
| Noise 50% | **97.5%** | 97.5% | 100% | **0 / 20** | 2.47 s |
| Noise 50%, a rule must be right more often than wrong (ablation) | 7.5% | 24.0% | 100% | 0 / 20 | 5.77 s |
| Noise 20%, 2 rules + 8 noise streams per seed | **100%** | 100% | 100% | **0 / 80** | 0.82 s (0.88 s) |
| Noise 50%, 2 rules + 8 noise streams per seed | **100%** | 100% | 100% | **0 / 80** | 1.18 s |
| Deep (7-node rules) | **100%** (98.75%) | 100% (99.0%) | 100% | **0 / 20** | 0.31 s (0.26 s) |
| Deep + noise 10% | **100%** (98.75%) | 100% (97.5%) | 100% (98.6%) | **0 / 20** | 0.84 s (0.72 s) |
| Deep, deepening disabled (ablation) | 37.5% | 38.1% | 100% | 0 / 20 | 0.37 s (0.34 s) |
| Deeper (9-node rules) | **100%** | 100% | 100% | **0 / 20** | 0.45 s (0.37 s) |
| Deeper + noise 10% | **100%** (93.75%) | 100% (93.75%) | 100% | **0 / 20** | 0.98 s (1.00 s) |
| Deeper, 9-node level and linear solver disabled (ablation) | 37.5% | 37.5% | 100% | 0 / 20 | 0.61 s (0.47 s) |
| Deepest (11- and 13-node rules) | **100%** | 100% | 100% | **0 / 20** | 0.93 s (0.84 s) |
| Deepest + noise 10% | **100%** | 100% | 100% | **0 / 20** | 1.59 s (1.54 s) |
| Deepest, linear solver disabled (ablation) | 11.25% | 11.5% | 100% | 0 / 20 | 1.47 s (0.89 s) |

How it works (see `docs/ARCHITECTURE.md`, learner): biases are fitted on the *majority* residual. A sequence rule
is proposed and survives while it is right on at least a quarter of the observations (in the lab, new rules are
only proposed below 40% mismatches and survive up to 50%). A rule is validated only when its forward record
passes one of three tests: ≤ 25% forward errors; after a burst of corruption, Wald's sequential test accepting
"errs at 25%" against "errs no better than a coin"; or, for sequences, an anytime-valid test that it beats chance.
Its errors must also be unstructured: a rule whose errors share one residual, follow a pattern, or are hit by
another rule is incomplete, not noisy. Deep rules are found by an iterative meet-in-the-middle search (7 nodes, then
9) on windows of 4 observations, most recent first, which backs off geometrically; linear rules of any size up to
4 terms (one product allowed, |coefficients| ≤ 16) are *solved for* exactly on short windows (`linear.py`);
moduli never observed are proposed from the gcd of residue differences. Beliefs are evaluated with exact
integers, so a geometric stream that leaves the ±10¹⁸ search bound is still predicted.

The two deepening ablations isolate each mechanism: without the linear solver, 11- and 13-node rules are out of
reach (11%, the streams whose constants happen to give a shorter rule); without the 9-node level *and* the
solver, 9-node rules too (37.5%).

**Half-corrupted streams: a rule must beat chance, not be right more often than wrong.** With half of the
observations corrupted, no rule is right more often than wrong, yet a rule right on half of them is still about
50 times better than chance (corrupted values hit by coincidence about once in 100, once in 10 on digits).
Passive sequence streams therefore:

- keep and propose rules up to 75% mismatches (`max_mismatch_frac`, `search_mismatch_frac`);
- score them with a corruption rate that may be unknown, integrated out under a uniform prior (weight 0.01);
- measure chance on the stream itself (a learned law over values, capped at 1/2);
- validate through an anytime-valid test against chance at level 0.01 (Ville's inequality), on forward
  predictions only.

Since a tolerant rule could be *incomplete* rather than corrupted, two safeguards come with it:

- **Structure tests** on every rule that has errors: a Markov-mixture test of its hit/miss record (corruption
  strikes at independent times) and a test that no other pooled rule hits its errors more often than chance.
- **Rival search:** before a rule that already errs is validated, the search reruns on the observations plus the
  rule's probes, to find rules that fit as well but differ on inputs never seen.

The ablation that restores the old rule (proposals and survival at 50%, a fixed corruption rate, no chance
test) validates only 7.5% of the rule streams at 50% noise, against 97.5% now: one arithmetic and one lag-2
stream are left unvalidated, and none is validated wrongly. At 20% noise the streams are now validated sooner
(43.9 corrupted observations seen per seed instead of 66.8), so CPU fell from 1.57 to 0.97 s.

**Costs, compared with `267aa4f`:**

- CPU is up on clean and lightly noisy worlds: +24% clean, +35% at 10% noise, +17–30% on deep worlds and up to
  +64% in the ablations. Searches at 75% tolerance keep more candidates. Proposing only below 50% would save
  that, but validated just 75 of 96 rule streams at 50% noise in development runs (seeds 1–12).
- Spurious rules on noise streams survive longer: noise is judged unpredictable after 8.65 observations instead
  of 8.0, and 4 of 60 noise streams were reopened. Curiosity therefore spends more on noise (below), and random
  allocation in stage 6 lost knowledge at budget 300 (0.949 vs 0.981).
- A structure test needs evidence: an incomplete rule whose errors still look independent can be validated
  before a rival or a pattern refutes it. This is the black-swan risk of every inductive validation.

Earlier in v0.3.0, three changes made noisy searches cheaper:

- The exact linear solver runs in C, with results identical to the Python reference. It uses modular elimination
  over 12 primes, a Hadamard bound that proves singularity, and exact back-substitution.
- Search tables are reused across calls.
- A search keeps only the pool's best instances.

Two clamps also keep tolerant searches small:

- A deep hit is kept only when it errs strictly less often than the best enumerated rule; a deeper rule that errs
  as often as a shorter one is less probable anyway.
- A shape's bias fit must be the residual of a strict majority, so it stays in the C kernel.

**Recognizing noise.** During sleep, the recognition model also learns two reserved classes from dreams:
uniform noise and random programs of the base language (3–11 nodes). When a series looks rule-less with
p ≥ 0.9 after 10 observations, its next deep search (deep, linear, invention stages) is postponed by n
observations. On held-out series of rule families never seen in training (n ≥ 12), every noise series is
flagged and no rule series is. In procedural transfer (seeds 4–8 × 6 worlds, measured in pairs), the agent
evaluates 7.5–11% fewer expressions (CPU −7% to −40%) with **identical** validation and accuracy in all four
conditions (standard, deep, 10% noise, deep + 10% noise). The recognizer only sets the *prior* odds of noise. The evidence of the hypothesis pool can only lower them:
the log-odds become `log-odds(features) + min(0, log p(data | null) − log Σ p(data | rules))`. A series for which
some rule explains most observations is therefore never postponed, whatever its features. For example, `x[t−2] + 2`
started from 23, 15 with two corrupted terms has p(noise) ≥ 0.9 from its features alone; it is no longer delayed
once a hypothesis fits it. Noise series keep empty pools, so they are postponed as before.

At 20% noise, the previous rule wrongly postponed 15 of 144 rule streams. With the evidence update, 0 rule streams
are postponed, while noise postponements, validations and observations to validation are unchanged and CPU stays
within ±5% (10 seeds, best of 2 runs).

### Attention / curiosity allocation (fraction of rule streams validated within a fixed budget)

| Budget (observations) | Curiosity | Round-robin | Random |
|---|---|---|---|
| 60 | **58.8%** | 0.0% | 23.8% |
| 100 | **92.5%** | 91.3% | 73.8% |
| 150 | **100%** | 97.5% | 91.3% |

**Fix of the earlier negative result (60 observations: curiosity 0%).** Curiosity used to spread samples
breadth-first, so no stream reached the validation threshold. The agent now applies *budget-aware focus*
(`Genesis._focus`): when the remaining budget cannot cover the expected cost of every open problem, it finishes
the cheapest open problem first. The expected cost is the median number of observations its own past problems
needed (learned, default 8 before any experience). Problems that already took more than twice that are set
aside.

Remaining negatives: at budget 100 curiosity leads round-robin by only one stream. Curiosity spends more
observations on noise (15.8 vs 12 for round-robin at budget 60; 25.9 vs 20 at 100; 43.9 vs 30 at 150). Noise must
be observed at least 8 times before it can be judged unpredictable, and since half-corrupted streams became
learnable (previously 15.2, 22.2 and 33.1), spurious rules that beat chance for a while keep some noise streams
open longer.

## Stages 2–3 — relationships and causal discovery (CausalWorld: 4 properties, 4 actions, ≥ 1 confounded rule)

| Condition | Held-out accuracy (unseen configurations) |
|---|---|
| Before any experience | 0.0% |
| After 60 passive observations, on the natural distribution | 98.9% |
| After 60 passive observations, under intervention | **27.8%** (the confound misleads) |
| After experiments (curiosity) | **100%** |
| Transfer to out-of-range values never experienced | **100%** |
| Tabular memorisation baseline | 46.4% (transfer: 46.0%) |
| "Nothing changes" baseline | 44.6% |

Learning curve (mean held-out accuracy on unseen configurations, 10 seeds):

| Experiments | 0 | 5 | 10 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|---|---|
| Curiosity (EIG) | 27.2% | **99.0%** | 99.3% | 99.3% | 99.3% | 99.7% | 100% |
| Random experiments | 27.2% | 74.3% | 83.6% | 96.6% | 97.3% | 99.2% | 99.6% |

Experiments needed to reach 100%: **24.5** with curiosity (10/10 runs) vs 51.7 with random experiments
(only 9/10 runs reach 100% within 120 experiments).

### Confounded rules

| Measure | Result |
|---|---|
| Confounded rules with the true cause identified after passive watching | 0% |
| Confounded rules with the true cause identified after experiments | **100%** |
| Associations labelled causal / spurious | 7 / 6 |
| Spurious labels that coincide with a real rule condition | 0 |

### Noisy experiments (`run_lab_robustness`)

Same labs and seeds, but each experiment outcome is corrupted with probability p (the target property is shifted
by 1–3). Held-out sets use the true physics. Conditional rules are learned with noise-tolerant branches: each
branch keeps its exact blocks when there are any, otherwise the blocks with the fewest mismatches within its
tolerance.

| Outcome noise | Held-out accuracy (unseen) | Out-of-range transfer | Runs reaching 100% | Confounded rules: correct cause | Wrong cause validated | CPU / run |
|---|---|---|---|---|---|---|
| 5% | **100%** | 100% | 10 / 10 | **18 / 18** | **0** | 1.4 s |
| 10% | **100%** | 100% | 10 / 10 | **18 / 18** | **0** | 2.7 s |
| 20% | **100%** | 100% | 10 / 10 | **18 / 18** | **0** | 4.4 s |

In v0.3.0 each branch had to match all of its observations exactly, so one corrupted outcome in a branch was
enough to block that conditional.

## Stage 4 — memory

| Metric | Value |
|---|---|
| Accuracy on world A after a restart (new session) | 100% (retention 1.0) |
| Accuracy on world B learned afterwards | 100% |
| Forgetting of world A after learning world B | **0.0** |
| Replay consistency of stored beliefs vs episodes | 1.0 |

Caveat: knowledge is stored as separate symbolic rules per context, so interference is structurally
impossible here. This is not evidence about catastrophic forgetting in neural learners.

## Stage 5 — planning to unseen goals

| Policy | Success | Mean steps when successful |
|---|---|---|
| BFS planning over the learned model | **100%** | 1.29 |
| Random actions (8 steps max) | 78.5% | 3.35 |

Caveat: the current tasks are easy (random actions often succeed). Harder multi-step goals are needed to
measure reasoning more seriously.

## Meta-learning (procedural memory of successful rule shapes)

Six worlds in sequence, later worlds 4–6, 5 seeds. With the prior, shapes that worked before are instantiated
first (*library-first search*); full enumeration is skipped when one of them fits the data exactly.

| Metric (later worlds) | With procedural memory | Without |
|---|---|---|
| Candidate programs evaluated per rule stream | **2,528** | 41,229 (16.3×) |
| CPU seconds per world | **0.343** | 0.527 (−35%) |
| Observations to validation | 8.11 | 8.25 |
| Extrapolation accuracy | 100% | 100% |

Same protocol on **deep worlds** (7-node rules; remembered shapes now go up to 7 nodes and are instantiated by
a C kernel):

| Metric (later worlds, deep) | With procedural memory | Without |
|---|---|---|
| Candidate programs evaluated per rule stream | **14,704** | 160,325 (11×) |
| CPU seconds per world | **0.411** | 0.625 (−34%; +3% at `267aa4f`, +25% in the previous draft) |
| Observations to validation | **8.74** | 11.56 (−24%) |
| Extrapolation accuracy | 100% | 100% |

"With procedural memory" includes library learning (below) and the recognition of noise. Memory now saves CPU
on standard worlds and no longer costs CPU on deep worlds:

- On standard worlds (−35%; −19% at `267aa4f`), likely noise no longer triggers the deep and invention stages
  at every deepening point. Searches at 75% tolerance (half-corrupted streams) cost more without memory, where
  every rule stream is found by enumeration.
- On deep worlds, memory cost 25% more in the previous draft (0.542 vs 0.434 s): the linear solver had made
  the learner *without* memory cheap, while instantiating remembered shapes and the invention stage still cost
  about 0.1 s per world. Most of that cost was the invention stage's deep phase on noise streams: it runs at 6
  and 9 observations, before the recognizer can judge a series. Deep hits must now beat the enumerated rules, so
  that phase is cheap. The library search also keeps only the pool's best instances, and the shape kernel reuses
  its tables.

The single benchmark run above (5 seeds) is within the ±10% timing noise of the 2 shared cores. A more robust
measurement at `267aa4f` (seeds 100–109, best of 2 interleaved runs) gave 0.347 s per late deep world with memory
vs 0.365 s without (−5%), and 8.47 vs 11.60 observations to validation. Memory thus saves search (11×) and observations
(−25%) at no CPU cost. Accuracy on deep worlds is 100% either way (98.0% before).

**Fixed:** remembering rule shapes gives a large, measured reduction in search. On deep worlds it also cuts
**observations** by a quarter: without memory, a 7-node rule is found only after the deepening threshold
(≥ 6 observations, then backoff), while a remembered shape is tried at once. **Still small:** on standard worlds
observations barely change, because validation requires 4 correct *forward* predictions whatever the search
does. In v0.2.0 the shape loop was in Python (2.6 s per world with memory); it is now 11× faster with identical
results.

### Library learning: invented primitives and dreams (`library.py`, `run_library_curriculum`)

A curriculum of six worlds per seed (seeds 100–102): three worlds of 7-node rules, then three worlds of 9-node
rules. During each sleep, sub-programs shared by validated rules of different problems are promoted to invented
primitives (MDL gain), dreams of validated shapes train a recognition model, and the next searches may use the
inventions as single leaves when the innate language finds nothing. Measured on the three late worlds.

The late rules are linear, so the exact linear solver also reaches them; it is switched off in the first four
rows to isolate what the library contributes, and the last row shows the solver alone.

| Search budget | Library learning | Rule streams validated | Observations to validation | CPU / world |
|---|---|---|---|---|
| Deepening up to 7 nodes | **on** | **100%** | 12.1 | 1.04 s |
| Deepening up to 7 nodes | off | 34.7% | 11.0 (validated streams only) | 0.82 s |
| Deepening up to 9 nodes | on | 100% | 10.3 | 0.59 s |
| Deepening up to 9 nodes | off | 100% | 10.4 | 0.70 s |
| Deepening up to 7 nodes, linear solver on | off | **100%** | 12.1 | 0.97 s |

Inventions found, for example: `x[t−1] + (x[t−1] − x[t−2])` (a second-order step), `x[t−1] + (t + t)`,
`2·(x[t−1] + t)`. With them, `3·x[t−1] − 2·x[t−2] + c` and `2·x[t−1] ± 3·t + c` become small programs over the
extended language, so a learner that can search only 7-node rules learns the 9-node worlds; without the library
it validates only the late streams whose rules happen to fit in 7 nodes. Validated beliefs are always inlined
into innate primitives and still need 4 correct forward predictions; dreams never become knowledge.

**Honest costs:** when the 9-node level is available, the library brings nothing here (same accuracy and
observations; CPU 0.59 s vs 0.70 s per world here, 0.58 s vs 0.50 s at `267aa4f`: the difference is within timing
noise). The
recognition model's effect on rule shapes is a tie-breaker: on deep procedural worlds (development seeds 2–4, equal CPU), observations to
validation go from 8.34 to 8.31, 8.48 to 8.29 and 8.59 to 8.59 when it is enabled.
`library_learning=False` disables the whole mechanism.

## Stage 6 — autonomous exploration (`autonomy.py`)

Four worlds, one shared budget, no instruction on where to look: a sequence world, a pure-noise world and two
causal labs. The explorer allocates chunks of 10 interactions. Each world is probed twice, then chosen by
measured learning progress (|Δ accuracy| per step + newly validated rules; "unpredictable" verdicts don't count)
plus a small optimism bonus 0.1/√chunks. It stops when no world shows progress ≥ 0.05. Knowledge gaps become
stored objectives (achieved / abandoned as unpredictable). Knowledge score = mean held-out accuracy over the
three learnable worlds.

| Budget | Learning progress | Round-robin | Random |
|---|---|---|---|
| 100 | 0.705 (noise share 20%) | **0.725** (30%) | 0.661 (31%) |
| 150 | **0.848** (13%) | 0.786 (28%) | 0.767 (32%) |
| 300 | **0.979** (18%, stopped itself after 226) | 0.968 (40%, 300) | 0.949 (45%, 300) |

- At 150 it beats round-robin by +6.2 points and spends half as much on noise.
- At 300 it reaches slightly more knowledge (0.979 vs 0.968) using 25% fewer interactions, by stopping itself.
- Random allocation at 300 fell from 0.981 to 0.949 when half-corrupted streams became learnable: spurious rules
  keep noise streams open longer, and it spends 45% of its budget on noise (41% before).
- **Negative:** at 100 it loses to round-robin. The two probe chunks per world cost 80 of the 100 interactions,
  so there is almost nothing left to exploit.
- On demo seed 0 at 150, random allocation won (86.2% vs 84.7%); per-seed variance is large.

## Stage 7 — controlled self-improvement (`self_improvement.py`)

Proposes one-parameter variants from an allowlist (safety parameters cannot be changed:
`PermissionError`). Evaluates each in isolated temporary agents with an *observable* score: predict 5 terms,
then observe them. It refuses to run if the safety digest has changed, and checks reproducibility by evaluating
the baseline twice. A variant must gain ≥ 0.01 on trial seeds and be **strictly better** on holdout seeds with no
more false validations. It is then stored as `pending-approval`; only a named human can `approve` it, and
`rollback` restores the previous configuration.

| Item | Result |
|---|---|
| Evaluation reproducible | yes |
| Baseline on trial seeds | 0.753 (0.747 at `267aa4f`, 0.713 in v0.2.0) |
| `sequence.validation_k` 4 → 3 / 4 → 5 | 0.740 / 0.747: worse than the baseline (both 0.780 at `267aa4f`) |
| Decision | **no variant improved the trial score** (at `267aa4f`: 4 → 3 passed the trial gate, then `rejected-holdout` at 0.773 vs 0.773) |
| Other variants (`shape_bias`, `default_resolution_cost`, `novelty_weight`) | exactly 0 effect at this budget |

**Honest reading:** in v0.2.0 the variant `validation_k` 4 → 3 was accepted for human review with a tiny holdout
gain (0.747 vs 0.740). The improved learner no longer needs it: it was rejected on holdout at `267aa4f`, and with
the current learner it no longer helps even on the trial seeds.
Nothing is proposed to the admin. This is the gate working as intended, not a new capability. Limitations found in its own records
(stuck problems, streams judged unpredictable, blocked gates) are reported by `limitations()`.

## Innate FR/EN language faculty (`language.py`, `experiments.run_language_understanding`)

This measures a **designer-provided** faculty, not learning: no number here is evidence that GENESIS acquired
language. Each sentence is sent to a fresh conversation (after a sequence when the intent needs one), and the
routed intent is compared with the expected one.

| Set | n | Intent accuracy | Language detection | Out-of-scope answered as if known |
|---|---|---|---|---|
| Development set (`LANGUAGE_BENCH`, used to find and fix 5 parser bugs, so **not** held-out) | 59 | 100% (91.5% before the fixes) | 100% | 0 |
| Held-out set (`LANGUAGE_HELDOUT`, written after the fixes, measured once, no tuning afterwards) | 38 | **100%** | 100% | 0 |

The held-out set has 15 FR + 15 EN paraphrases, 2 negations, 2 typos and 4 out-of-scope questions. Both sets were
written by the same designers as the lexicon, so this measures coverage of *their* phrasing. Failures observed on
harder probes, kept as known limits of the original command parser (the semantic teaching channel below has
separate, bounded handling of multiple assertions, local negation and subject references):
- one intent per message ("explore puis dis-moi tes limites" → explore only);
- negation scope is the whole sentence ("pas besoin d'explorer, dis-moi ce que tu sais" → help);
- no reference to earlier turns or time ("la suite de tout à l'heure"), no idioms outside the list ("figure
  out");
- no SMS spelling ("t es ki");
- no open-domain meaning: "tu es nul" is only "a sentence about « nul »".

## Word learning in the chat (`lexicon.py`, `experiments.run_word_learning`)

Since the innate faculty understands ordinary French, this experiment uses **slang/jargon outside it** ("balance
tes trouvailles", "ton palier", "tes galères", "je suis paumé"...). A simulated user sends 15 training sentences
(5 commands × 3 paraphrases) in random order over 4 passes. It answers questions truthfully and picks the intended
command when GENESIS doesn't understand. Evaluation uses 15 **unseen** paraphrases, read without learning, plus 6
unrelated control sentences. The figures are the means over seeds 100–109; seed 100's per-pass trace is:

| Pass | not understood | proposed, wrong | proposed, correct | acted alone, correct | acted alone, wrong |
|---|---|---|---|---|---|
| 1 | 8 | 6 | 1 | 0 | 0 |
| 2 | 0 | 0 | 14 | 1 | 0 |
| 3 | 0 | 0 | 13 | 2 | 0 |
| 4 | 0 | 0 | 1 | 14 | 0 |

| Metric (seeds 100–109) | Value |
|---|---|
| Vocabulary learned / validated mappings | 35 words / 26 |
| Unseen paraphrases: top interpretation correct | **100%** (chance 9.1%), min over seeds 100% |
| Unseen paraphrases: would act alone / precision | 67% / **100%** |
| Control sentences acted on / asked about | 0 / 0 |
| Wrong autonomous actions during training | 0 |
| User exchanges for the whole run | 108 |

Limits: the unseen paraphrases share content words with the training ones by construction. Learned words are
word→command associations, combined statistically; they add no new meaning beyond the innate commands.

## Acquiring semantic relations without a pretrained model

`experiments.run_semantic_learning(seed)` uses 30 invented labels per seed, with no facts installed at birth.
Six independent groups each receive two class-inclusion teachings and one example entity. A new entity in
each group is then introduced with membership only. The six target memberships require **two inference steps**;
none is supplied as a teaching. Six cross-group questions are negative controls. Rules of deduction and the
sentence grammar are designer-provided, not learned. The acquired content is the particular relation graph.

Measured over seeds 100–109:

| Metric | Result |
|---|---|
| Initial semantic/world knowledge | 0 items |
| Target answers before teaching | 60/60 unknown |
| Training assertions | 24 per run |
| New-entity target answers after teaching | 60/60 supported, each with 3 premise IDs |
| Unsupported positive answers on negative controls | 0/60 |
| Retention after unrelated teaching / restart | 100% / 100% |
| Opposing teachings detected as a conflict | 10/10 |
| Obsolete conclusions removed after correction | 10/10 |
| Unaffected knowledge retained after correction | 100% |
| Recorded interaction episodes | 52 per run |
| SQLite size at measurement | 155,648 bytes per run |
| Wall time on this CPU, isolated measurement | 0.161–0.209 seconds per run |

This is evidence of **symbolic acquisition, transfer and persistence**, not general intelligence, open-domain
comprehension or independent discovery of logic. User assertions are not independently verified. Prior agreement
recorded before each teaching measures consistency with the user, not truth. Repeated assertions do not increase
confidence. The tests additionally cover bilingual teaching, explanation, API use, pronouns across restarts,
ambiguity, cycle termination, bounded storage, retraction, preservation of old evidence and non-transitivity of
spatial relations. Unknown and contradicted are distinct outcomes.

Reproduce without running the longer world benchmarks:

```bash
.venv/bin/python -c 'from genesis.experiments import run_semantic_learning; print([run_semantic_learning(s) for s in range(100, 110)])'
```

## Grounded object representations (`grounding.py`)

After development on seeds 0–2 and tests on seed 7, the fixed learner was evaluated on world seeds 100–109.
Each run begins with empty learned knowledge and performs 80 initial experiments on a catalog of 32 structured
specimens. The five binary sensor channels and two action tokens are opaque. The hidden effect laws are simple
seed-sampled functions of one or two channels. The innate learner knows neither the functions nor their relevant
channels; it searches for minimal consistent predictive projections.

Evaluation uses 64 new object IDs per run, each tested under both actions. All objects have previously unseen
**full** sensor vectors because globally irrelevant channels take value 2. Relevant values remain in the trained
binary domain. This specifically tests invariance to distractors, **not extrapolation to new causal values**.
The world oracle is used only by the evaluation harness. The held-out curves never affect training choices,
budgets, stopping or fitted tables.

| Metric across ten seeds | Result |
|---|---|
| Initial prediction coverage | 0% (abstention, no supplied facts) |
| Final held-out accuracy / coverage | 100% / 100% in every run |
| Correct minimal feature sets / validated action representations | 2 / 2 per run |
| Object-ID / full-state lookup coverage | 0% / 0% |
| Majority-outcome baseline accuracy | 53.9–68.0% |
| Existing `RuleLearner`, same initial observations | 100% in every run |
| One-step goals on new specimens | 640/640 successes |
| Uniform random action, expected goal success | 69.5–77.3% per run |
| Predictions retained after reload / learning a second world | 100% / 100% |
| Changed-action accuracy immediately after hidden inversion | 0% |
| Changed-action accuracy after 80 further experiments | 100% |
| Unchanged action accuracy after revision | 100% |
| Abstention when a relevant channel takes an unseen value | 100% |
| Main-memory interaction episodes | 304 (80 initial, 64 evaluation goals, 80 other-world, 80 adaptation) |
| SQLite size at measurement | 303,104–311,296 bytes |
| Measured total wall time, including baselines | 0.91–1.73 s per run |

Same learner, different experiment selection (held-out accuracy ranges over seeds):

| Interaction budget | Active selection | Random selection |
|---|---|---|
| 16 | 83.6–93.8% | 50.0–90.6% |
| 32 | 82.8–90.6% | 50.0–100% |
| 48 | 100% | 50.0–100% |
| 64 / 80 | 100% | 100% |

**Honest interpretation:** learned projections outperform rote lookup on the chosen distractor-shift test, but
do not beat the existing symbolic learner at the final budget. Active exploration is not consistently better
than random at every intermediate budget. The main addition is an inspectable learned representation and its
validation/revision lifecycle, not evidence of human-like understanding.

The concept check uses future predictions on novel full configurations plus controlled one-channel contrasts.
This protects against validating from repeated identical examples, but does not guarantee validity outside the
sampled world. Confounded evidence can leave multiple candidate projections. Incompatible repeated states
trigger model revision; under noisy dynamics this could mistake noise for a changed law. Sensor formats, search
bias and the definition of an action effect remain innate. No word meanings, new sensors, neural features,
multi-step planning or general physical laws are learned here. Retention across worlds uses separate environment
contexts, not interference-free learning in a shared neural representation.

```bash
.venv/bin/python -m genesis grounded --seed 100 --db runs/objects.db
.venv/bin/python -m genesis dashboard --db runs/objects.db
# Multiple seeds, isolated temporary memories:
.venv/bin/python -c 'from genesis.experiments import run_grounded_learning; print([run_grounded_learning(s) for s in range(100, 110)])'
```

The explicit database path must not already exist. The report, concepts, revisions and original episodes remain
inspectable there. Goal-evaluation outcomes are logged but never fitted; a separate temporary memory is used for
the random-exploration comparison.

## Learning word meanings from objects (`grounded_language.py`)

After development on seed 0 and a regression experiment on seed 3, the fixed procedure was measured on
seeds 100–109. Each fresh memory explores its world for 80 experiments, then receives 24 positive/negative
examples for each of two arbitrary words. The simulated teacher assigns words to action-effect predicates;
word/action/value assignments vary by seed. The teacher and evaluation oracle live only in `experiments.py`.
The learner receives labels linked to actual experiment records, never those predicate definitions.

Evaluation classifies 64 unseen specimens for both words and selects the exact sets matching four
previously untaught conjunction/negation combinations. New IDs and irrelevant sensor values prevent
object-ID/full-configuration memorization. Relevant values remain in the trained domain. No test annotation
or test effect is fitted; the world's 80 training transitions remain unchanged throughout the language test.

| Metric across ten seeds | Result |
|---|---|
| Initial word prediction coverage | 0% |
| World experiments / initial word annotations | 80 / 48 per run |
| Held-out word classification accuracy / coverage | 1,280/1,280 correct / 100% |
| Exact composed reference sets | 40/40 |
| Validated words / correct novel-input checks | 2 per run / 14–22 per word |
| Object-ID label lookup coverage | 0% |
| Objects selected by negating an unknown word | 0 |
| Retention after reload | 100% |
| Contradictory annotations cause abstention | 10/10 |
| Explicit correction + eight further examples restore validation | 10/10 |
| Post-correction accuracy / original incorrect episodes preserved | 100% / 10/10 |
| Total episodes, including contradiction/correction checks | 138 per run |
| SQLite size at measurement | 233,472 bytes per run |
| Measured wall time on this CPU | 0.482–0.542 s per run |

These are **learned lexical predicates in a restricted world**, not evidence of human general comprehension.
Sensor and effect representations are supplied as a data format; the predicate search space, sentence frames,
conjunction and negation are designer-provided. A human or simulated teacher supplies the naming convention.
Observationally equivalent sensor/effect meanings can remain ambiguous despite accurate predictions.
Agreement with annotations does not establish a uniquely correct or universal definition. No photographs,
learned grammar, spontaneous language emergence or general physical reasoning are demonstrated.

The tests also cover refusal on unseen relevant values or unvalidated action models, no fitting during queries,
annotation/resource bounds, bilingual HTTP dialogue, conversation-local selection, correction history and
invalidation after an observed world revision. Repetition alone cannot validate a meaning.

```bash
.venv/bin/python -c 'from genesis.experiments import run_grounded_word_learning; print([run_grounded_word_learning(s) for s in range(100, 110)])'
```

## Inducing sentence constructions (`constructions.py`)

After development on seed 0 and a regression run on seed 3, the fixed procedure was evaluated on seeds 100–109.
Each run first performs 80 world experiments and receives 144 object-label annotations for six words.
Eight additional utterance/meaning demonstrations teach two forms with different slot orders. Literal markers,
word-effect assignments and negation assignments vary by seed; no forms are installed in the learner.
Four words appear in construction training; two further grounded words appear only in held-out substitutions.

Evaluation uses 36 new sentences per seed and the exact selected object sets on 64 specimens with unseen IDs
and irrelevant sensor values. Neither held-out meanings nor effects enter training. The meaning supplied by the
teacher is a conjunction of signed lexical predicates, **not raw unsupervised language experience**.

| Metric across ten seeds | Result |
|---|---|
| Recognized probe sentences before construction teaching | 0 |
| New-sentence exact interpretations / validated coverage | 360/360 / 100% |
| Exact object selections on unseen specimens | 360/360 sets |
| Exact-sentence lookup baseline coverage | 0% |
| Unknown-form abstention / restart retention | 100% / 100% |
| Episodes per run | 232 = 80 experiments + 144 labels + 8 construction demonstrations |
| SQLite size after reload | 335,872–348,160 bytes |
| Measured total wall time on this CPU | 1.625–2.837 s per run |

These are narrow tests of systematic slot substitution, not general grammar, human-level understanding or
learning a reasoning algorithm. The hypothesis space and interpretation operators remain designer-provided.
The benchmark does not test recursive syntax, compositional depth, noisy demonstrations or natural discourse.
Unit tests additionally check ambiguous overlapping forms, contradictory demonstrations, correction and renewed
validation, repetition, unvaried slots, resource/provenance checks, bilingual dialogue and query non-learning.

```bash
.venv/bin/python -c 'from genesis.experiments import run_construction_learning; print([run_construction_learning(s) for s in range(100, 110)])'
```

## Developmental gating

On the demo run (seed 0), the agent passes the gates for stages 0–7 one after another, each based on measured
metrics. The stage 6 gate requires a gain ≥ 0.02 over round-robin and ≤ 25% of the budget spent on noise. The
stage 7 gate requires a reproducible evaluation, an unchanged safety policy, no holdout regression and at least
one self-identified limitation. It does **not** require any change to be deployed.
