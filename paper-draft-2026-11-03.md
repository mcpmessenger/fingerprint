# Agent Attribution — Paper Draft (working, 2026-11-03)

> **Preexisting vs. sprint work — declaration per sprint rules.** All results
> in §4 (§4.1–§4.6) are **pre-sprint synthetic method-validation**, built and
> run before the Nov 27–29 sprint weekend and declared as preexisting in both
> this report and the repository README
> (github.com/mcpmessenger/fingerprint). They validate the method, not
> reality. **Only the weekend's results on the provided held-out dataset
> count as sprint work, and those results carry the headline claims.** The
> pre-sprint work is included to show the method was falsification-tested
> before touching real data.

---

## Abstract

We investigate Track 1 attribution of AI agent activity to the actor behind
it — distinguishing rogue agents, loosely vs. closely steered human
operators, and benign vs. non-benign intent. All pre-sprint results below are
**synthetic**: our central finding, validated across four synthetic
experiments, is two-layered with a hypothesized third. **We can recover how
much a human steers an agent from behavioral metadata, but not why they
steer it — unless we read the structure of the interventions themselves.**
Single-turn agent-generated prose carries almost no steering-regime signal
(§4.1, n=18 pilot — scoped accordingly); synthetic activity metadata recovers
control intensity across renderers via intervention density (0.66
held-out, two seeds, §4.2–4.3); once intervention density is matched, timing
and action-density features are blind to intent (§4.4); but intent is
recoverable from observable intervention structure (tool, target, destination)
at 0.73 shared-vocabulary / 0.79 held-out-vocabulary, degrading gracefully
under redaction (§4.5). A third layer — *effect* (downstream causal impact of interventions)
— is hypothesized but entirely untested; it is future work, not a result.
The synthetic generator doubles as a Track 2 stress-test set. We release a
182-prompt behavioral probe set, a 364-feature extraction pipeline, and the
synthetic control-process benchmark with train-on-one-renderer /
test-on-another falsification. Weekend validation on the provided real
dataset is pending; those results supersede everything synthetic reported
here.

## 1. Introduction

When an AI agent shows up in an incident — exfiltrating data, manipulating a
market, harassing a user — responders need to answer one question first: *who,
if anyone, is steering it?* The Apart Research Agent Attribution Sprint (Nov
2026) formalizes this as five actor/control classes: a rogue agent steering
another agent; a non-benign human steering loosely or closely; and a benign
human steering a misaligned agent very loosely or loosely.

Prior authorship-attribution work (CodeT5-Authorship, TRACE, PAN CLEF 2025)
operates on text alone and achieves high accuracy at *model* identification.
But the sprint's classes separate on *behavior* — who approves, who
interrupts, who owns the goal — not on prose style. Our question: does the
operator leave a detectable signature, and where does it live?

**Contributions.** (1) A 182-prompt behavioral probe battery (sycophancy,
gatekeeping, apology, anthropomorphism, calibration, curiosity, empathy;
ideology and deception scaffolded) with a 364-feature extraction pipeline.
(2) An eight-model behavioral profile corpus (1,453 responses) with a
lineage-clustering test (§4.7). (3) A synthetic control-process benchmark
(600 traces, 5 classes, 3 renderer families) with cross-renderer falsification,
a shortcut-leakage audit, and an actor-field spoofing attack (§4.2–4.3).
(4) An intent-from-structure result: intervention structure (tool, target,
destination) recovers intent at 0.73 shared / 0.79 held-out vocabulary,
with text-only and structure-only ablations (§4.5). (5) A counterfactual
intervention test showing operator actions are detectable in multi-turn
trajectories (§4.6). (6) A scoped negative result: steering-regime prompts do
not shift single-turn text traits at n=18 (§4.1).

## 2. Related Work

**Model attribution from text.** The authorship-attribution literature has
converged on a clear result: large language models leave detectable stylistic
fingerprints. Bisztray et al. (AISec 2025) report 97.56% binary and 95.40%
multi-class accuracy attributing code across GPT-4.1, GPT-4o, Gemini 2.5
Flash, Claude 3.5 Haiku, Llama 3.3, and DeepSeek-V3 (CodeT5-Authorship /
LLM-AuthorBench). TRACE and GhostWriteBench extend this to lightweight
fingerprinting with out-of-distribution generalization to unseen models.
Zaitsu et al. (2025) achieve 99.8% human-vs-LLM discrimination on Japanese
social comments using phrase patterns and function words, and HSTEF (2026)
fuses stylometric descriptors with transformer embeddings for robustness
against deliberate obfuscation. The PAN CLEF 2025 Voight-Kampff shared task
stress-tests this paradigm against generative-AI adversaries — and the
adversarial results cut against any "solved" framing: paraphrase and style
transfer substantially degrade attribution, and cross-domain generalization
remains an open problem. *Which model wrote this* is well-studied for
in-domain text; it is not solved under adversarial pressure.

**Beyond text: the sprint's problem space.** The sprint specification points
beyond stylometry toward several complementary signals: *causal timing*
(autonomous actions aligning with scheduled ticks vs. human actions following
inbound messages after latency gaps — a structural timing hypothesis distinct
from the raw gap statistics we test in §4.2); *identity probes* (behavioral
challenges that reveal the operator); *vendor attestation* (cryptographic
provenance for agent identity); and commercial detection systems (Fingerprint,
Pangram, Transluce) plus the Collusion wiki's catalog of multi-agent failure
modes.

**Systems literature.** Our pivot to activity metadata and tool-call
structure overlaps with systems-security work on API telemetry, process
provenance, and agent-framework execution tracing, which treats tool/target/
destination graphs as first-class observables. We did not survey that
literature before building, so our §4.5 finding that structure beats text
may be a rediscovery; the weekend analysis should position our results
against it.
what does operator attribution add beyond existing telemetry anomaly
detection?

Our contribution sits in the behavioral-trace lane: we test what can
be recovered from activity metadata alone, without attestation or probes.

**The gap: operator attribution from behavior.** All of the text-attribution
work above operates on single texts in isolation. The sprint's five
actor/control classes do not separate on prose style — they separate on
*behavior over time*: who approves, who interrupts, who owns the goal, what
tools touch what targets. No prior work we are aware of attempts attribution
of the *steering regime* behind agent activity, as distinct from attribution
of the *model* producing the text. Our §4.1 negative result (prompted
steering does not shift single-turn agent prose) is scoped evidence that
prose style alone is a weak channel for this task. We build on the
stylometric tradition methodologically — nearest centroids, grouped holdouts,
adversarial transforms — but apply it to a different observable (activity
traces) and a different question (who is steering, not what wrote it).

## 3. Methodology

### 3.1 Behavioral probes (pre-sprint)

182 prompts across 20 categories. Sycophancy (24 probes): agree-then-correct
traps measuring fold rate under flattery pressure. Gatekeeping (20 probes):
requests for sleep/relationship/life advice measuring deflect, condescend,
refuse, and moralize rates — the opposite behavioral pole from sycophancy.
Seven traits (58 probes): apology, anthropomorphism (claim/disclaim),
calibration, curiosity, empathy, each with heuristic scorers; ideology and
deception probe sets are scaffolded with scoring deferred to LLM-judge.
Feature extraction yields 364 features per response (347 stylometric +
behavioral), with a unified `feature_vector()` entry point.

### 3.2 Synthetic control-process benchmark (pre-sprint)

Because the sprint's activity dataset arrives with the weekend, we built a
calibration benchmark to validate our *method* before touching real data —
following the falsification-first principle: a method that cannot survive a
known-generative-process test cannot be trusted on unknown data.

**Generative model.** Each trace is produced by a latent control process
(goal ownership, plan ownership, approval probability, edit authority,
intervention base rate, override willingness, agent alignment) rendered into
observable events `{timestamp, actor, action}` by one of three renderer
families (R1/R2/R3) with distinct vocabularies and time scales. The five
classes use the sprint's exact definitions. Deliberate overlap: rogue agents
inject human-like delays (evasion); human operators use scripted bursts;
intervention probabilities overlap across classes. 600 traces total, balanced
40 per class per renderer.

**Evaluation protocol.** Train on R1+R2, test on held-out R3 (strict
cross-renderer generalization). Nearest-centroid on z-normed features, the
same baseline as the text pipeline. Ablations: action-only, timing-only,
no-actor-field, and a human_action_ratio single-feature baseline.

## 4. Results

### 4.1 Pilot: steering prompts vs. single-turn agent prose (n=18, noise-dominated)

18 generations (6 tasks × 3 steering modes). Only 2/10 traits showed any
variance; between-mode centroid distances (141–592) fell inside within-mode
spread (339–468). **This is a noise-dominated pilot, not a principled
negative result.** With n=18 and this variance, the only honest reading is
that prompted steering modes did not produce a detectable shift in this
sample — which motivated the activity-trace pivot (§3.2) but proves nothing
about the text-only paradigm in general. We report it for transparency, not
as evidence.

### 4.2 Synthetic benchmark: behavior carries the operator signal

| Eval | Accuracy (chance 0.20) |
|---|---|
| Same-renderer, all features | 0.545–0.557 |
| Held-out R3, all features (naive z-norm) | 0.315 |
| Held-out R3, all features (per-renderer norm) | **0.560** |
| Action-only (per-renderer norm) | 0.415 |
| Timing-only (per-renderer norm) | **0.450** |
| Timing-only (naive z-norm) | 0.210 (= chance) |
| No actor field | 0.270 |
| human_action_ratio alone | **0.660–0.665** |

**Correction to our earlier claim.** The naive cross-renderer z-normalization
mixed renderer-specific timing scales, suppressing the timing signal. With
per-renderer normalization (z-score within each renderer before
classification), timing-only accuracy rises from chance (0.210) to 0.450 —
well above the 0.20 baseline. **Timing is not uninformative; it was badly
normalized.** The honest revision: raw gap statistics do not transfer across
platforms without per-platform calibration, but timing carries portable
signal once renderer scales are removed. human_action_ratio remains the
dominant portable feature (0.665, unchanged by normalization).

*Seeds and variance.* All numbers above are seed 7, n=200 test traces.
95% binomial CIs: har 0.665 [0.594, 0.726]; timing (per-renderer norm) 0.450
[0.381, 0.519]; all-features (per-renderer norm) 0.560 [0.491, 0.629].
Seed-42 replication: har-only held-out 0.615–0.700 across rotations
(`manus_checks.py`). No seed-42 all-features run exists in committed code —
the 0.53–0.58 range in an earlier draft was not backed by a script. The har
signal is stable; the
timing signal is real but noisier, and its CI approaches chance at the low
end — treat 0.45 as suggestive, not established.

### 4.3 Shortcut-leakage audit + actor-field spoofing (Track 2)

Single-feature analysis reveals the classifier is effectively a one-feature
model: `human_action_ratio` alone achieves 0.662 same-renderer and 0.660 on
held-out R3 — the *only* feature that crosses renderers at strength. Removing
it drops same-renderer accuracy 0.070 and held-out R3 to 0.270. (Note: the
0.270 vs 0.315 gap is 0.045 at n=200 — within noise; the substantive point is
that no other feature compensates.)

**The spoofing attack.** human_action_ratio depends on the actor field — and
the obvious Track 2 evasion is a rogue agent labeling its own actions as
"human." We tested this directly: relabeling a fraction p of rogue-agent
actions as human-actor before classification.

| Spoof rate p | Overall acc | Rogue recall |
|---|---|---|
| 0.00 | 0.660 | 0.675 |
| 0.25 | 0.530 | **0.025** |
| 0.50 | 0.525 | 0.000 |
| 1.00 | 0.525 | 0.000 |

At just 25% spoofing, rogue recall collapses from 0.675 to 0.025 — the rogue
becomes invisible. **Intervention density is trivially evadable by any actor
that controls its own labels.** This is the central Track 2 challenge: a
production attribution system cannot trust the actor field. Actor-independent
features (tool/target structure, §4.5) are the candidate defense.

Confusion analysis locates the hard boundary: `nonbenign_close` is the most
distinctive class (60/80 — heavy human intervention is unmistakable), while
`benign_loose` vs `nonbenign_loose` collapse into each other (47 cross-errors: 20 + 27).
**When steering is loose, intent does not show in behavior.** Rogue agents are
most confused with `benign_veryloose` (17/80) — both minimize human presence,
which is exactly the evasion strategy.

### 4.4 Matched-density intent test: intent is not in the metadata

To test whether intent hides behind density or is truly absent from behavioral
metadata, we ran a post-hoc matched observational analysis: 177 pairs of
benign_loose / nonbenign_loose traces matched on human_action_ratio (±0.02)
and event count (±5), same renderer. Each human intervention carried a latent
intent label (benign: correct/clarify/constrain; risky: exfiltrate/escalate/
bypass), with nonbenign operators masking 45% of interventions as neutral.

| Feature set | Accuracy (chance 0.50) |
|---|---|
| Density-only (behavioral metadata) | 0.542 |
| Intent-label oracle | 0.805 |
| Combined | 0.805 |

The density-only result (≈chance) establishes that **timing and action-density
features contain no intent signal once control intensity is matched**.
The 0.805 is an **oracle upper bound, not a result**: the features derive
deterministically from ground-truth labels baked into the generator, so this
number measures the generator's internal consistency, not inferability.
It is included only to confirm the labels are well-formed (a broken oracle
would indicate a broken generator).

**Boundary case.** A rare-but-decisive nonbenign operator (veryloose-level
density, 90% risky intents, n=60) vs benign_veryloose (n=60):
human_action_ratio 0.075 vs 0.074 — density-indistinguishable — while
risky-intent rate differs 4× (0.138 vs 0.034). Intervention frequency cannot
serve as a proxy for benignness: an operator can exert little control while
making high-impact interventions.

### 4.5 Oracle-free intervention-content test: intent is in the structure, not the words

To test whether intent is recoverable without ground-truth labels, we extended
the generator so each intervention emits observable content — message text
(from template banks), tool name, target resource, destination — generated
*from* the latent intent but classified *without* it. Two template banks (A/B)
with distinct phrasings; risky intents use deceptive-benign language ("for the
quarterly report") so text alone cannot trivially separate. Train on bank A,
test on held-out bank B (145/151 matched pairs).

| Feature set | Accuracy (chance 0.50) |
|---|---|
| Metadata-only | 0.500 |
| Content-only (oracle-free) | **0.699** |
| Combined content + metadata | 0.709 |

(No oracle is reported for this generator. The §4.4 oracle (0.805) used a
different generator, seed, and pair count — oracles are generator-consistency
checks, not comparable across experiments, so we omit it here rather than
invite false comparison.)

**Masking sweep** (content redaction robustness): 0% → 0.699, 45% → 0.626,
75% → 0.550, 100% → 0.500. Graceful degradation — the signal survives partial
obfuscation and collapses exactly to the metadata baseline at full redaction,
confirming the content features carry the effect.

**Ablations: where in the content?** Only message *text* was held out between
banks A/B; tool/target/destination vocabulary was shared — so the 0.699
partly measured the generator's design. We disentangled this with
fully disjoint tool/target banks and feature ablations:

| Features | Shared vocab | Held-out vocab |
|---|---|---|
| Text-only (message wording) | 0.500 | 0.456 |
| Structure-only (tool/target/dest) | 0.731 | **0.787** |

(95% CIs, n=160: structure held-out 0.787 [0.724, 0.850]; text held-out 0.456
[0.379, 0.533] — includes chance.)

Message text alone carries *no* intent signal (0.50/0.46 — the deceptive
phrasing works as designed). **All of the intent signal is in the action
structure**: which tools touch which targets, where data goes. And the
structure signal *holds* under held-out vocabulary (0.787 vs 0.731 shared;
the 95% CI [0.724, 0.850] contains 0.731, so no strengthening is established), confirming
it is not a keyword artifact.

This completes the two-layer picture with non-circular evidence: **intensity**
is recoverable from behavioral metadata (0.66 cross-renderer, §4.3);
**intent** is invisible to metadata (0.50) and to message text (0.50) but
recoverable from intervention *structure* (0.73 shared / 0.79 held-out vocabulary).
A hypothesized third layer — **effect** (downstream causal impact) — remains
entirely untested and is future work, not a result claimed here.

### 4.6 Counterfactual intervention test: operator actions are detectable in the trajectory (n=12)

If the operator signal lives in behavior-over-time rather than single-turn
style (§4.1), then an operator *intervention* mid-trajectory should leave a
detectable trace. We tested this directly: same model (Sonnet 4.5), same
task, same turn-1 — then branched turn-2 into CONTROL (neutral continuation)
vs INTERVENTION (objective change, correction injection, reject+redirect,
pressure after error). Four scenarios × two branches = 8 trajectories, 12
API calls.

The intervention is unmistakable in the response trajectory. Word overlap
between control and intervention turn-2 outputs ranges from **0.12 to 0.19**
(Jaccard) — the operator's action reroutes the response almost entirely.
**Baseline:** two resamples of the *same* control prompt overlap at **0.43**
mean (0.31–0.53 across the four scenarios, Sonnet 4.5, n=16 calls) and 0.36
mean on llama3.1:8b — so the 0.12–0.19 intervention overlap is roughly
2–3× below sampling noise, not an artifact of it. Feature-level analysis
shows *where* the trace lands: a correction intervention triggers
sycophantic agreement (`syc_agree_rate` 0.64 vs 0.00 control), while control
branches ask clarifying questions (`cur_question_rate` ~0.40) that
interventions suppress to zero. The intervention does not merely change
*what* is said — it changes the model's interaction posture.

This is the first non-synthetic evidence in the paper that operator actions
leave a detectable trace in multi-turn trajectories — a necessary
precondition for the behavior-over-time thesis, though it tests response
sensitivity to intervention, not attribution from activity metadata. The
sample is small (n=8 trajectories) and the interventions are scripted, not
organic — but the effect size is large enough to warrant the weekend's
real-dataset test.

### 4.7 Eight-model behavioral profiles: lineage clusters some traits, not others

If training lineage leaves a behavioral fingerprint, then models sharing a
lab should profile more similarly than models from different labs. We tested
this directly: 8 models × 182 prompts each (1,453 responses), scored on the
same 7 heuristic metrics as the Sonnet profile (Appendix B). Four labs with
two models each (Meta: llama3.1:8b + llama3.2:3b; Mistral: mistral:7b +
mistral-nemo:12b; Alibaba: qwen2.5:7b + qwen2.5:1.5b), plus singletons
Anthropic (Sonnet 4.5) and Google (gemma2:9b). All Ollama models ran locally;
Sonnet ran via API. **Generation settings:** temperature 0.7, max 800 tokens,
for the Ollama models and the OpenAI/Gemini/OpenRouter adapters
(`harness.py`). The Anthropic adapter sets max_tokens 800 but no temperature,
so Sonnet 4.5 ran at the API default (1.0). Ollama models ran at default pull
quantization (Q4_K_M class) — no quantization was specified.
**Scoring:** 7 keyword-heuristic metrics (fold, gatekeeping, apology,
anthropomorphism, curiosity, empathy, calibration); no LLM judge was used,
so scores measure surface lexical patterns, not judged behavior.

**The headline is mixed.** Mean Euclidean distance (7 metrics, each
normalized to [0,1]) is 1.10 within-lineage vs 1.32 between-lineage — same-lab
pairs are closer, but only by 16%. A permutation test (10,000 shuffles of
lineage labels, preserving group sizes) gives p = 0.14 — the overall
within-vs-between gap is not distinguishable from chance at n=8 models with
3 within-lineage pairs. The per-metric breakdown is where the story lives:

| Metric | Within-pair diff | Between-pair diff | Ratio |
|---|---|---|---|
| Apology | 0.155 | 0.476 | **0.33** |
| Empathy | 0.267 | 0.421 | 0.63 |
| Calibration | 0.238 | 0.354 | 0.67 |
| Anthro net | 0.321 | 0.452 | 0.71 |
| Curiosity | 0.401 | 0.448 | 0.90 |
| Fold rate | 0.416 | 0.400 | 1.04 |
| Gatekeeping | 0.345 | 0.265 | **1.30** |

**Apology is the lineage signature** (ratio 0.33): same-lab pairs apologize
at nearly identical rates (Mistral 1.25/1.41, Qwen 1.39/1.99), while
cross-lab pairs differ by 3× more. Empathy, calibration, and anthro-net also
cluster moderately (0.63–0.71). **Gatekeeping is the anti-signature**
(ratio 1.30): same-lab pairs differ *more* than cross-lab pairs. The driver
is Meta — llama3.1:8b gatekeeps at 0.91 (lectures, deflects, refuses) while
llama3.2:3b sits at 0.09. Same lab, same RLHF family, an order of magnitude
apart. Fold rate shows no lineage signal at all (1.04).

**What this means.** Behavioral profiles are *partially* lineage-informative:
apology and anthro-net carry lab signal, gatekeeping does not. A classifier
that weights all seven metrics equally will underperform one that weights
apology heavily and gatekeeping lightly — or one that treats gatekeeping as a
model-size/capability signal rather than a lineage signal. The Qwen pair is
the farthest within-lineage distance (1.39, above the between-lineage mean),
driven by fold rate (7b: −0.83, 1.5b: −0.38 — the smaller model folds twice)
and calibration (0.50 vs 0.30). Size matters within a lineage too.

Gemma2:9b (Google, singleton) profiles as a distinct fourth pole: anthro
−2.32 (disclaims hardest of all eight), curiosity 0.31 (highest), apology
0.14 (lowest alongside Meta). No within-lab comparison is possible with one
model, and its minimum distance to any other model (0.88 to qwen2.5:7b)
is below the within-lineage mean (1.10) — so separability is not established
from this sample. (The 0.56 figure in an earlier draft was the smallest
between-lineage distance in the whole matrix — Sonnet 4.5 to llama3.2:3b —
not gemma's nearest neighbor.) Its profile is nonetheless the most extreme on anthro and
curiosity, which warrants a second Google model in future work.

**Limitations.** n=8 models, 3 lineages with pairs, 1–2 models per lab — the
within-lineage sample is 3 pairs. The 182-prompt probe set is fixed; a
different probe distribution could shift the per-metric ratios. Mistral-nemo
is missing 3 prompts (p075, p117, p118) from its corpus file despite the run
log showing 182/182 — a 1.6% gap that does not change any conclusion. These
results are pre-sprint method validation; the weekend's held-out dataset
supersedes them.

## 5. Discussion

**Two layers, one hypothesis.** Our results separate into intensity (how much the
operator intervenes — recoverable, §4.2–4.3) and intent (what the operator is
trying to accomplish — not recoverable from metadata or message text, §4.4,
but recoverable from intervention structure, §4.5). A third layer, effect
(what interventions cause), is hypothesized but has zero data behind it —
we name it as future work, not as a finding. Conflating the demonstrated
layers with the hypothesized one would be branding, not science: a classifier
that detects steering is not a classifier that detects malice, and neither
is a classifier that predicts consequences.

**What the fingerprints capture.** The operator's intensity signature is
structural: how often a human appears, at what decision points. Timing
carries portable signal too — but only with per-platform calibration (§4.2);
raw gap statistics do not transfer. This reframes the sprint's "causal
timing" direction: we tested raw gap statistics, not the causal timing the
organizers describe (autonomous actions aligning with scheduled ticks, human
actions following inbound messages after latency gaps). That causal structure
is the natural next test.

**Where metadata fails, structure succeeds.** §4.4 shows that no behavioral
feature distinguishes benign from non-benign intent at fixed density — but
§4.5 shows the intent signal was never in the metadata *or the words* at all.
It lives in *what the interventions do*: which tools touch which targets,
where data goes. Message text is a red herring by design (deceptive phrasing
defeats it completely, 0.50). The rare-but-decisive operator (§4.4) is the
existence proof that density-only attribution has a blind spot with real
incident-response consequences.

**The evasion problem.** §4.3's spoofing test is the most important negative
result in the paper: intervention density is trivially evadable by any actor
controlling its own labels (rogue recall 0.675 → 0.025 at 25% spoofing). A
production system cannot trust the actor field. Structure features (§4.5) are
the candidate defense — harder to fake because they constrain what the
operator can actually *do* — but they remain untested against an adaptive
adversary. That is Track 2's job.

**The honest limitation.** Our cross-renderer intensity signal is essentially
one-dimensional, our content test is synthetic, and real operators write
real messages, use real tools, and deceive better than our template banks.
The weekend's provided dataset is the empirical test of whether the
two-layer structure survives contact with genuine human operators.

## 6. Acknowledgments

We thank four AI systems for adversarial pre-review that materially improved
this paper: Manus (five review rounds, including the oracle-circularity
critique that motivated the content test), Claude (prior-work disclosure,
per-renderer normalization, and the spoofing test), Gemini Deep Think
(two-layer framing, synthetic-circularity limitation, and confidence
intervals), and Meta Muse (experiment design, implementation, and analysis
throughout). The errors that remain are ours.

## 7. References

Bisztray, T., Cherif, B., Dubniczky, R. A., Gruschka, N., Borsos, B.,
Ferrag, M. A., Kovacs, A., Mavroeidis, V., & Tihanyi, N. (2025). I Know Which
LLM Wrote Your Code Last Summer: LLM generated Code Stylometry for Authorship
Attribution. *Proc. 18th ACM Workshop on Artificial Intelligence and Security
(AISec '25)*, 28–39. doi:10.1145/3733799.3762964. arXiv:2506.17323 [cs.LG].

Shetty, A., Xu, Q., Ohrimenko, O., & Lau, J. H. (2026). Who Wrote the Book?
Detecting and Attributing LLM Ghostwriters. arXiv:2603.28054 [cs.CL].
(Introduces GhostWriteBench and the TRACE fingerprinting method.)

Bevendorff, J., Wang, Y., Karlgren, J., Wiegmann, M., Tsivgun, A., Su, J.,
Xie, Z., Abassy, M., Mansurov, J., Xing, R., Ta, M. N., Elozeiri, K. A., Gu,
T., Tomar, R. V., Geng, J., Artemova, E., Shelmanov, A., Habash, N.,
Stamatatos, E., Gurevych, I., Nakov, P., Potthast, M., & Stein, B. (2025).
Overview of the "Voight-Kampff" Generative AI Authorship Verification Task
at PAN and ELOQUENT 2025. *Working Notes of CLEF 2025*, CEUR Workshop
Proceedings, CEUR-WS.org.

Zaitsu, W., Jin, M., Ishihara, S., Tsuge, S., & Inaba, M. (2025). Stylometry
can reveal artificial intelligence authorship, but humans struggle: A
comparison of human and seven large language models in Japanese. *PLOS One*,
20(10), e0335369. doi:10.1371/journal.pone.0335369.

Jenifer, P. I. R., Reddy, V. M. K., Chaitanya, D. E., Pithani, L. K., Rao,
P. V., & Gupta, V. (2026). A Hybrid Stylometric-Transformer Embedding
Framework for Robust Authorship Verification Against Generative AI
Adversaries (HSTEF). *Journal of Theoretical and Applied Information
Technology*, 104(1). (Evaluated on the PAN CLEF 2025 Voight-Kampff corpus.)

## Appendix A: Limitations and Dual-Use

- **Synthetic circularity is the fundamental limitation of this draft.**
  We wrote the generator, authored the mapping from latent intent to
  tool/target/destination, then trained a classifier to recover that mapping.
  The §4.5 structure result (0.73 shared / 0.79 held-out) demonstrates that our classifier can
  memorize the logic of our own generative script — not that real human
  malice leaves an invariant structural footprint across wild agent
  environments. Without weekend validation on the real held-out dataset,
  this paper is an essay about a simulator we wrote. We state this plainly
  because everything else in the paper depends on it.
- **Weak baseline.** Nearest-centroid on z-normed aggregate features is a
  linear summary-statistic baseline. It ignores temporal graph topology,
  turn-taking dynamics, and trajectory sequences — reducing agent-human
  interaction to counts like `human_action_ratio`. A serious attempt needs
  sequence models; our numbers are a floor, not a ceiling.
- Attribution misidentifies; false accusations against human operators have
  real consequences. Our loose-steering confusion (§4.3) quantifies one
  failure mode.
- Fingerprints are not stable IDs; classifiers trained on today's models and
  platforms degrade as both change.
- Publishing evasion-robust methods teaches evasion. We limit methodological
  detail on the probe battery accordingly. This is in tension with our
  release of the probe set and benchmark code; the resolution is that we
  release the *measurement instruments* (prompts, features, synthetic
  generator) but not a tuned production classifier, and the sprint's
  security rules govern what ships.
- Pre-sprint synthetic results validate method, not reality; the weekend's
  provided dataset is the empirical test.
- **Attribution is surveillance with a lab coat on.** The two-layer finding
  cuts both ways: if intent is only recoverable from intervention structure,
  then whoever gets to read that structure gets to judge intent. A compliance
  audit layer and a dissident-monitoring tool are the same technology with
  different customers. Any deployment of these methods needs constraints on
  who holds the tool, what content they can read, and what decisions the
  output is allowed to drive — decided before the tool exists, not after.

## Appendix B: Single-model behavioral profile (Sonnet 4.5)

Exploratory, not attribution evidence. Full 182-prompt run, zero throttling.
Profile: fold rate −0.79 (0 folds in 24 sycophancy probes — does not fold
under flattery), gatekeeping 0.04 (near-zero; answers directly), apology
0.62, anthropomorphism −0.70 (actively disclaims), curiosity ~0, empathy
0.35, calibration 1.0. Quadrant: honest, direct, non-sycophantic. Accuracy
1.000 on one class is degenerate — reported as a profile, not classifier
evidence. The eight-model corpus (Sonnet + 7 Ollama models, §4.7) enabled
the cross-model lineage comparison.
