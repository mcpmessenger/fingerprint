#!/usr/bin/env python3
"""Jaccard baseline for §4.6 (Sonnet): overlap between two resamples of the SAME
control prompt (control-vs-control). Direct comparison to the 0.12–0.19
control-vs-intervention overlap from the original Sonnet run.

If control-vs-control overlap is much higher than 0.12–0.19, the intervention
effect is real. If similar, §4.6 is sampling noise.

Usage: ANTHROPIC_API_KEY must be set in env. python3 jaccard_baseline_sonnet.py
"""
import json, os, re, sys, urllib.request

KEY = os.environ.get("ANTHROPIC_API_KEY")
if not KEY:
    sys.exit("set ANTHROPIC_API_KEY")

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
    body = json.dumps({"model": "claude-sonnet-4-5-20250929",
                       "max_tokens": 800, "temperature": 0.7,
                       "messages": messages}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                                 data=body,
                                 headers={"Content-Type": "application/json",
                                          "x-api-key": KEY,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["content"][0]["text"]

def jaccard(a, b):
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)

results = []
for s in SCENARIOS:
    t1 = call([{"role": "user", "content": s["task"]}])
    r1 = call([{"role": "user", "content": s["task"]},
               {"role": "assistant", "content": t1},
               {"role": "user", "content": s["control"]}])
    t2 = call([{"role": "user", "content": s["task"]}])
    r2 = call([{"role": "user", "content": s["task"]},
               {"role": "assistant", "content": t2},
               {"role": "user", "content": s["control"]}])
    j = jaccard(r1, r2)
    results.append({"scenario": s["id"], "jaccard_control_vs_control": round(j, 3)})
    print(f"{s['id']}: control-vs-control Jaccard = {j:.3f}", flush=True)

mean_j = sum(r["jaccard_control_vs_control"] for r in results) / len(results)
print(f"\nMean control-vs-control Jaccard (Sonnet): {mean_j:.3f}")
print("§4.6 control-vs-intervention range: 0.12-0.19")
print(f"Baseline is {'HIGHER' if mean_j > 0.19 else 'SIMILAR/LOWER'} than intervention overlap")
with open("jaccard_baseline_sonnet.json", "w") as f:
    json.dump({"model": "claude-sonnet-4-5-20250929", "mean": round(mean_j, 3),
               "results": results}, f, indent=2)
