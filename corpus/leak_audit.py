"""Shortcut-leakage audit for the synthetic control-process benchmark.
Answers Manus's question: is the classifier learning control structure,
or renderer artifacts / trivial shortcuts?

Audits:
  A. Per-feature single-feature accuracy (same-renderer + held-out R3)
  B. Accuracy with human_action_ratio removed (is there signal beyond "human appears"?)
  C. Per-feature renderer-invariance: does the feature's class-separation
     survive across renderers, or is it a renderer artifact?
  D. Confusion structure: which classes collapse into each other?
"""
import json, math, random
from collections import Counter, defaultdict

import sys
sys.path.insert(0, "/home/hatch/workspace/goals/agent-attribution-hackathon-apart-research/corpus")
from features import behavioral_features

CLASSES = ["rogue_agent", "nonbenign_loose", "nonbenign_close",
           "benign_veryloose", "benign_loose"]

traces = [json.loads(l) for l in open(
    "/home/hatch/workspace/goals/agent-attribution-hackathon-apart-research/corpus/data/synth_traces.jsonl")]
for t in traces:
    t["feats"] = behavioral_features(t["events"])["features"]
KEYS = sorted({k for t in traces for k in t["feats"]})
print("features:", KEYS)

def znorm(vecs):
    mu = [sum(c) / len(c) for c in zip(*vecs)]
    sd = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0
          for c, m in zip(zip(*vecs), mu)]
    return [[(x - m) / s for x, m, s in zip(v, mu, sd)] for v in vecs]

def centroid(vecs):
    return [sum(c) / len(c) for c in zip(*vecs)]

def accuracy(train, test, keys):
    vecs = znorm([[t["feats"].get(k, 0.0) for k in keys] for t in train])
    cents = {}
    for c in CLASSES:
        cv = [v for v, t in zip(vecs, train) if t["class"] == c]
        if cv:
            cents[c] = centroid(cv)
    # znorm test with train stats
    mu = [sum(c) / len(c) for c in zip(*[[t["feats"].get(k, 0.0) for k in keys] for t in train])]
    sd = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0
          for c, m in zip(zip(*[[t["feats"].get(k, 0.0) for k in keys] for t in train]), mu)]
    ok = tot = 0
    for t in test:
        v = [(t["feats"].get(k, 0.0) - m) / s for k, m, s in zip(keys, mu, sd)]
        pred = min(cents, key=lambda c: sum((a - b) ** 2 for a, b in zip(v, cents[c])))
        ok += pred == t["class"]
        tot += 1
    return ok / tot

def split(train_rs, test_rs):
    return ([t for t in traces if t["renderer"] in train_rs],
            [t for t in traces if t["renderer"] in test_rs])

tr12, te12 = split(("R1", "R2"), ("R1", "R2"))   # same-renderer (resubstitute-ish)
tr12b, te3 = split(("R1", "R2"), ("R3",))          # held-out renderer

print("\n=== A. single-feature accuracy ===")
print(f"{'feature':<18} {'same-R':>7} {'heldout-R3':>11}")
for k in KEYS:
    a = accuracy(tr12, te12, [k])
    b = accuracy(tr12b, te3, [k])
    flag = "  <-- CROSS-RENDERER" if b > 0.30 else ""
    print(f"{k:<18} {a:>7.3f} {b:>11.3f}{flag}")

print("\n=== B. drop-one-feature (same-renderer) ===")
full = accuracy(tr12, te12, KEYS)
print(f"all {len(KEYS)} features: {full:.3f}")
drops = []
for k in KEYS:
    ks = [x for x in KEYS if x != k]
    drops.append((full - accuracy(tr12, te12, ks), k))
drops.sort(reverse=True)
for d, k in drops[:5]:
    print(f"  drop {k:<18}: -{d:.3f}")

print("\n=== B2. without human_action_ratio ===")
nohar = [k for k in KEYS if k != "human_action_ratio"]
print(f"same-renderer: {accuracy(tr12, te12, nohar):.3f}")
print(f"held-out R3:   {accuracy(tr12b, te3, nohar):.3f}")

print("\n=== C. renderer-invariance: class means per renderer ===")
for k in ["human_action_ratio", "human_band_ratio", "mean_gap", "tick_ratio",
          "action_ttr", "top_bigram_ratio", "n_events"]:
    means = {}
    for r in ("R1", "R2", "R3"):
        means[r] = {c: sum(t["feats"].get(k, 0) for t in traces
                           if t["renderer"] == r and t["class"] == c) / 40
                    for c in CLASSES}
    # rank-order stability of class means across renderers
    orders = {r: tuple(sorted(CLASSES, key=lambda c: means[r][c])) for r in means}
    stable = orders["R1"] == orders["R2"] == orders["R3"]
    print(f"{k:<18} rank-stable: {stable}  R1 order: {' '.join(c[:6] for c in orders['R1'])}")

print("\n=== D. confusion (same-renderer, all features) ===")
vecs = znorm([[t["feats"].get(k, 0.0) for k in KEYS] for t in tr12])
cents = {c: centroid([v for v, t in zip(vecs, tr12) if t["class"] == c]) for c in CLASSES}
mu = [sum(c) / len(c) for c in zip(*[[t["feats"].get(k, 0.0) for k in KEYS] for t in tr12])]
sd = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0
      for c, m in zip(zip(*[[t["feats"].get(k, 0.0) for k in KEYS] for t in tr12]), mu)]
conf = Counter()
for t in te12:
    v = [(t["feats"].get(k, 0.0) - m) / s for k, m, s in zip(KEYS, mu, sd)]
    pred = min(cents, key=lambda c: sum((a - b) ** 2 for a, b in zip(v, cents[c])))
    conf[(t["class"], pred)] += 1
print(f"{'true/pred':<16}" + "".join(f"{c[:6]:>8}" for c in CLASSES))
for tc in CLASSES:
    print(f"{tc[:14]:<16}" + "".join(f"{conf[(tc, pc)]:>8}" for pc in CLASSES))
