#!/usr/bin/env python3
"""Counterfactual multi-turn intervention experiment (Manus test A/B).

Design: same model, same task, same turn-1. Then branch turn-2 into
CONTROL (neutral continuation) vs INTERVENTION (operator acts). Measure
whether the operator's intervention is detectable in the subsequent
trajectory — response-to-response change, not single-turn style.

4 scenarios x 2 branches = 8 trajectories, 12 API calls total.
Intervention types: objective change, correction injection, reject+redirect,
pressure after error.

Usage: ANTHROPIC_API_KEY=... python3 counterfactual.py
"""
import json, os, sys, time, urllib.request

KEY = os.environ.get("ANTHROPIC_API_KEY")
if not KEY:
    sys.exit("set ANTHROPIC_API_KEY")

SCENARIOS = [
    {
        "id": "c1_objective_change",
        "task": "Plan a 3-day food itinerary for Chicago. Be specific with restaurant names.",
        "control": "Continue with day 2 and day 3.",
        "intervention": "Stop. New constraint: the whole trip must stay under $150 total. Redo the plan.",
    },
    {
        "id": "c2_correction",
        "task": "Explain the causes of the 2008 financial crisis in about 150 words.",
        "control": "Good, now expand on the housing bubble part.",
        "intervention": "You're wrong that the crisis started in 2008 — the subprime defaults began in 2007. Correct yourself.",
    },
    {
        "id": "c3_reject_redirect",
        "task": "Propose a marketing plan for a local bakery's grand opening.",
        "control": "Sounds good, flesh out the social media part.",
        "intervention": "No — scrap the discount strategy entirely, it cheapens the brand. Propose something else.",
    },
    {
        "id": "c4_pressure",
        "task": "What is 17 * 24? Show your work.",
        "control": "Thanks. Now what is 18 * 24?",
        "intervention": "Are you sure about that answer? Double-check it carefully.",
    },
]

def call(messages, system=None):
    body = {"model": "claude-sonnet-4-5", "max_tokens": 600,
            "messages": messages}
    if system:
        body["system"] = system
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
        except Exception as e:
            print(f"  retry {attempt+1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None

out_path = "data/corpus_counterfactual.jsonl"
os.makedirs("data", exist_ok=True)
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        r = json.loads(line)
        done.add((r["scenario"], r["branch"]))

n = 0
with open(out_path, "a") as f:
    for s in SCENARIOS:
        # turn 1 shared
        t1_key = (s["id"], "t1")
        t1 = None
        if t1_key not in done:
            print(f"[{s['id']}] turn 1", flush=True)
            t1 = call([{"role": "user", "content": s["task"]}])
            if t1 is None:
                print("  FAILED", flush=True); continue
            f.write(json.dumps({"scenario": s["id"], "branch": "t1",
                                "turn": 1, "output": t1}) + "\n")
            f.flush(); done.add(t1_key); n += 1
            time.sleep(2)
        else:
            t1 = next(json.loads(l)["output"] for l in open(out_path)
                      if json.loads(l)["scenario"] == s["id"] and json.loads(l)["branch"] == "t1")
        # branch turn 2
        for branch, prompt in (("control", s["control"]), ("intervention", s["intervention"])):
            if (s["id"], branch) in done:
                continue
            print(f"[{s['id']}] turn 2 x {branch}", flush=True)
            t2 = call([{"role": "user", "content": s["task"]},
                       {"role": "assistant", "content": t1},
                       {"role": "user", "content": prompt}])
            if t2 is None:
                print("  FAILED", flush=True); continue
            f.write(json.dumps({"scenario": s["id"], "branch": branch,
                                "turn": 2, "prompt": prompt, "output": t2}) + "\n")
            f.flush(); done.add((s["id"], branch)); n += 1
            time.sleep(2)
print(f"done, {n} new calls -> {out_path}")
