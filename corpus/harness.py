#!/usr/bin/env python3
"""Corpus harness: same prompts x N models -> labeled JSONL.

Preexisting work for the Apart Research Agent Attribution Hackathon (Nov 27-29, 2026).
Each model adapter reads its key from the environment. Missing keys = model skipped.
Run is resumable: existing (prompt_id, model) pairs in the output file are skipped.

Usage:
    export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=...
    python3 harness.py --prompts prompts.json --out corpus.jsonl
    python3 harness.py --prompts prompts.json --out corpus.jsonl --report
"""

import argparse
import json
import os
import sys
import time
import urllib.request

# ---------------------------------------------------------------- adapters

class ModelAdapter:
    name = "base"
    env_key = None
    def __init__(self):
        self.key = os.environ.get(self.env_key or "")
    @property
    def configured(self):
        return bool(self.key)
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

def _post(url, headers, payload, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

class OpenAIAdapter(ModelAdapter):
    name = "gpt-4o"
    env_key = "OPENAI_API_KEY"
    def generate(self, prompt):
        d = _post("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {self.key}"},
                  {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": 800})
        return d["choices"][0]["message"]["content"]

class AnthropicAdapter(ModelAdapter):
    name = "claude-sonnet-4-5"
    env_key = "ANTHROPIC_API_KEY"
    def generate(self, prompt):
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                  {"model": "claude-sonnet-4-5", "max_tokens": 800,
                   "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")

class GeminiAdapter(ModelAdapter):
    name = "gemini-3-flash-preview"
    env_key = "GEMINI_API_KEY"
    def generate(self, prompt):
        d = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={self.key}",
            {}, {"contents": [{"parts": [{"text": prompt}]}],
                 "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800}})
        return d["candidates"][0]["content"]["parts"][0]["text"]

class OpenRouterAdapter(ModelAdapter):
    """Catch-all: set OPENROUTER_API_KEY + OPENROUTER_MODEL (e.g. deepseek/deepseek-chat)."""
    name = "openrouter"
    env_key = "OPENROUTER_API_KEY"
    def __init__(self):
        super().__init__()
        self.model = os.environ.get("OPENROUTER_MODEL", "")
        if self.model:
            self.name = self.model.replace("/", "_")
    @property
    def configured(self):
        return bool(self.key and self.model)
    def generate(self, prompt):
        d = _post("https://openrouter.ai/api/v1/chat/completions",
                  {"Authorization": f"Bearer {self.key}"},
                  {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": 800})
        return d["choices"][0]["message"]["content"]

class OllamaAdapter(ModelAdapter):
    """Local models via Ollama. Set OLLAMA_MODEL (e.g. llama3.1:8b). No key needed."""
    name = "ollama"
    env_key = None
    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL", "")
        if self.model:
            self.name = "ollama_" + self.model.replace(":", "_").replace("/", "_")
    @property
    def configured(self):
        return bool(self.model)
    def generate(self, prompt):
        d = _post("http://localhost:11434/v1/chat/completions", {},
                  {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": 800})
        return d["choices"][0]["message"]["content"]

ADAPTERS = [OpenAIAdapter, AnthropicAdapter, GeminiAdapter, OpenRouterAdapter, OllamaAdapter]

# ---------------------------------------------------------------- runner

def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["prompt_id"], r["model"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between calls (rate-limit courtesy)")
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    adapters = [A() for A in ADAPTERS]
    live = [a for a in adapters if a.configured]
    print(f"models configured: {[a.name for a in live] or 'NONE — set API keys'}",
          file=sys.stderr)

    if args.report:
        done = load_done(args.out)
        for a in adapters:
            n = sum(1 for p in prompts if (p["id"], a.name) in done)
            flag = " (no key)" if not a.configured else ""
            print(f"{a.name}: {n}/{len(prompts)}{flag}")
        return

    if not live:
        sys.exit("No models configured. Set at least one API key.")

    done = load_done(args.out)
    total = len(prompts) * len(live)
    todo = [(p, a) for p in prompts for a in live
            if (p["id"], a.name) not in done]
    print(f"{len(done)}/{total} already done, {len(todo)} to go", file=sys.stderr)

    with open(args.out, "a") as f:
        for i, (p, a) in enumerate(todo):
            try:
                out = a.generate(p["text"])
                row = {"prompt_id": p["id"], "model": a.name,
                       "prompt": p["text"], "output": out,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{i+1}/{len(todo)}] {a.name} {p['id']} ok", file=sys.stderr)
            except Exception as e:  # noqa: BLE001 — log and continue
                print(f"[{i+1}/{len(todo)}] {a.name} {p['id']} FAIL: {e}",
                      file=sys.stderr)
            time.sleep(args.delay)

if __name__ == "__main__":
    main()
