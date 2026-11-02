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
