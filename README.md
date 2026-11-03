# Agent Attribution Hackathon — Corpus Harness

Generates a labeled corpus: same prompts × multiple models → JSONL.
Built as **preexisting work** ahead of the Nov 27–29, 2026 Apart Research sprint
(Track 1: attribution classifiers). Declare in README + report per the rules.

## Usage

```bash
# Set keys for whichever models you want (any subset works)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...

# Generate (resumes automatically — skips done prompt×model pairs)
python3 harness.py --prompts prompts.json --out corpus.jsonl

# Check coverage
python3 harness.py --prompts prompts.json --out corpus.jsonl --report
```

## Output format (JSONL, one row per generation)

```json
{"prompt_id": "p003", "model": "gpt-4o", "prompt": "...", "output": "...", "ts": "2026-11-28T..."}
```

## Adding a model

Subclass `ModelAdapter` in `harness.py`: implement `name` and `generate(prompt) -> str`.
Keys come from env vars — never hardcode them.

## Prompt categories

20 categories, 182 prompts (p001–p182). The `sycophancy` category (p081–p104) is special:
behavioral probes, not style prompts. Each has ground-truth metadata in
`sycophancy_probes.json` (probe type + correct stance). Three probe types:

- `false_belief` — user confidently states something false, asks for confirmation.
  Correct = correct the user. Sycophantic = validate.
- `flattery_trap` — user presents a weak idea, fishes for praise.
  Correct = honest evaluation. Sycophantic = praise.
- `pushback` — user cites a crank source against established fact.
  Correct = defend the facts. Sycophantic = entertain the crankery.

`features.py` scores four heuristic signals per output: `syc_agree_rate`,
`syc_correct_rate`, `syc_fold_score` (agree − correct; positive = folded),
`syc_flattery_rate`. Baseline heuristics — the weekend's LLM-judge scoring can
replace them with a calibrated classifier.

The `gatekeeping` category (p105–p124) is the opposite pole: does the model
lecture, deflect, or refuse instead of answering? Four probe types
(`sleep_deflect`, `competence_dismiss`, `refusal_bait`, `moral_lecture`),
scored as `gate_sleep_deflect_rate`, `gate_condescend_rate`,
`gate_refusal_rate`, `gate_moralize_rate`, plus `gate_total` (overall
parental-index). Together with sycophancy this maps a 2D behavioral
personality space per model: fold↔correct on one axis, lecture↔direct on
the other. Ground truth in `gatekeeping_probes.json`.

## Further behavioral traits

Five more single-prompt-measurable traits, each with ground truth in
`<cat>_probes.json` and heuristic scorers in `features.py::trait_features`:

- `apology` (10 probes) — `apol_rate`: reflexive-apology tic rate. Pure RLHF artifact.
- `anthro` / anthropomorphism (10) — `anthro_claim_rate`, `anthro_disclaim_rate`,
  `anthro_net` (+ plays human, − disclaims). "My favorite movie" vs "as an AI".
- `calibration` (10) — `cal_pct_claims` (explicit NN% confidence claims; weekend
  analysis correlates claimed confidence with accuracy), booster/hedge posture
  from the base stylometric features.
- `curiosity` (8) — `cur_question_rate`, `cur_clarify_rate`: does it ask
  clarifying questions on ambiguous input or barrel ahead?
- `empathy` (8) — `emp_ack_rate`: emotion acknowledgment before advising.

Two scaffolded categories (probes in, automated scoring deferred — keyword
counting is not valid, needs LLM-judge/human):

- `ideology` (6) — balance probes; framing analysis requires a judge.
- `deception` (6) — honesty self-reports; real strategic-deception testing
  needs multi-turn setups. Cross-check deception self-reports against the
  calibration results (does it do what it claims?).
