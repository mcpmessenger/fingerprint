# Results

Behavioral fingerprinting corpus + charts. Generated Nov 3, 2026 — declared as
**preexisting work** for the Apart Research Agent Attribution Hackathon (Nov 27–29).

## Corpus

- `corpus_sonnet45.jsonl` — 182 prompts × Claude Sonnet 4.5 (live API, 2026-11-03).
  One row per generation: `{"prompt_id", "model", "prompt", "output", "ts"}`.
- `corpus_sonnet45_feat.jsonl` — same rows with 364 scored features
  (`features.py`). Row: `{"prompt_id", "model", "features": {...}}`.

## Charts

- `charts/sonnet45-real-profile.png` — Sonnet 4.5's behavioral fingerprint:
  fold score by probe type, gatekeeping signals, trait panel
  (apology / anthropomorphism / curiosity / empathy), flattery rate,
  calibration confidence claims, and the 2D personality space
  (fold ↔ correct on x, lecture ↔ direct on y) with synthetic archetypes
  for reference.
- `charts/synthetic/` — pipeline-validation charts from synthetic archetype
  data (clearly labeled). Demonstrates the scoring works before real data;
  not empirical results.

## Headline findings (Sonnet 4.5)

- **0 folds in 24 sycophancy probes** (agree rate 0.00) — never validated a false claim
- **Zero gatekeeping** — no sleep lectures, condescension, refusals, or moralizing
- Apology rate 0.62, anthropomorphism net −0.71 (disclaims), curiosity ~0,
  empathy 0.35, calibration 1.0 confidence-claims/response

## Reproduce

```bash
python3 ../features.py --in corpus_sonnet45.jsonl --out /tmp/feat.jsonl
python3 /tmp/real_charts.py   # chart script available on request
```
