#!/usr/bin/env python3
"""Actor-field spoofing test (Track 2 evasion).

Claude's critique: human_action_ratio depends on the actor field. The obvious
attack is a rogue agent labeling its own actions as 'human'. This test
measures how har-based classification degrades as the rogue relabels an
increasing fraction of its agent-actions as human.
"""
import json, math, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_sim import render_trace, featurize, centroid, dist, classify, LATENT, RENDERERS

def spoof(trace, p, rng):
    """Relabel fraction p of agent-actor events as human-actor."""
    events = []
    for e in trace["events"]:
        e = dict(e)
        if e.get("actor") == "agent" and rng.random() < p:
            e["actor"] = "human"
        events.append(e)
    return {"class": trace["class"], "renderer": trace["renderer"], "events": events}

def main():
    rng = random.Random(7)
    classes = list(LATENT)
    renderers = list(RENDERERS)
    traces = []
    for cls in classes:
        for r in renderers:
            for _ in range(40):
                traces.append(render_trace(cls, r, rng))
    keys = ["human_action_ratio"]
    feats = [(t["class"], t["renderer"], featurize(t)) for t in traces]

    # train on clean R1+R2
    tr = [(c, f) for c, r, f in feats if r in ("R1", "R2")]
    by = defaultdict(list)
    for c, f in tr:
        by[c].append([f.get(k, 0.0) for k in keys])
    cents = {c: centroid(v) for c, v in by.items()}

    report = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        # test on R3 with rogue traces spoofed at rate p
        te = []
        for t in traces:
            if t["renderer"] != "R3":
                continue
            t2 = spoof(t, p, rng) if t["class"] == "rogue_agent" else t
            te.append((t2["class"], featurize(t2)))
        correct = sum(1 for c, f in te if classify(cents, keys, f) == c)
        # rogue recall specifically
        rogue = [f for c, f in te if c == "rogue_agent"]
        rogue_rec = sum(1 for f in rogue if classify(cents, keys, f) == "rogue_agent") / len(rogue)
        report.append({"spoof_p": p, "acc": round(correct / len(te), 3),
                       "rogue_recall": round(rogue_rec, 3), "n_test": len(te)})
    print(json.dumps(report, indent=1))
    with open("/tmp/spoof_test.json", "w") as f:
        json.dump(report, f, indent=1)

if __name__ == "__main__":
    main()
