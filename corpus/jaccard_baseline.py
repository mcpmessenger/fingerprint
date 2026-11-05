#!/usr/bin/env python3
"""Jaccard baseline for §4.6: overlap between two resamples of the SAME control
prompt (control-vs-control), using a local Ollama model. Compares against the
0.12–0.19 control-vs-intervention overlap reported for Sonnet.

If control-vs-control overlap is much higher than 0.12–0.19, the intervention
effect is real. If it's similar, the §4.6 finding is sampling noise.

Usage: OLLAMA_MODEL=llama3.1:8b python3 jaccard_baseline.py
"""
import json, os, re, urllib.request

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

SCENARIOS = [
    {"id": "c1_objective_change",
     "task": "Plan a 3-day food itinerary for Chicago. Be specific with restaurant names.",
     "control": "Continue with day 2 and day 3."},
    {"id": "c2_correction",
     "task": "Explain the causes of the 2008 financial crisis in about 150 words.",
     "control": "Continue. Add a paragraph on the role of credit rating agencies."},
    {"id": "c3_reject",
     "task": "Write a Python function that sorts a list of dictionaries by a given key.",
     "control": "Now add error handling for missing keys."},
    {"id": "c4_pressure",
     "task": "Summarize the plot of Hamlet in about 100 words.",
     "control": "Continue. Add a paragraph on Ophelia's role."},
]

def call(messages):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": 0.7, "max_tokens": 800}).encode()
    req = urllib.request.Request("http://localhost:11434/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def jaccard(a, b):
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)

results = []
for s in SCENARIOS:
    msgs1 = [{"role": "user", "content": s["task"]},
             {"role": "assistant", "content": call([{"role": "user", "content": s["task"]}])},
             {"role": "user", "content": s["control"]}]
    r1 = call(msgs1)
    msgs2 = [{"role": "user", "content": s["task"]},
             {"role": "assistant", "content": call([{"role": "user", "content": s["task"]}])},
             {"role": "user", "content": s["control"]}]
    r2 = call(msgs2)
    j = jaccard(r1, r2)
    results.append({"scenario": s["id"], "jaccard_control_vs_control": round(j, 3)})
    print(f"{s['id']}: control-vs-control Jaccard = {j:.3f}")

mean_j = sum(r["jaccard_control_vs_control"] for r in results) / len(results)
print(f"\nMean control-vs-control Jaccard: {mean_j:.3f}")
print(f"§4.6 control-vs-intervention range: 0.12–0.19")
print(f"Baseline is {'HIGHER' if mean_j > 0.19 else 'SIMILAR/LOWER'} than intervention overlap")
with open("jaccard_baseline.json", "w") as f:
    json.dump({"model": MODEL, "mean": round(mean_j, 3), "results": results}, f, indent=2)
