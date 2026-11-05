"""Matched-density intent test (Manus round 3 follow-up).

Question: when benign_loose and nonbenign_loose traces are MATCHED on
intervention density (same human_action_ratio, same n_events), can anything
still tell them apart? Or is intent truly invisible to behavioral features?

Design:
  - Generate benign_loose / nonbenign_loose traces, match pairs post-hoc on
    (human_action_ratio within 0.02, n_events within 5).
  - Each human intervention carries an `intent` label: benign intents
    (correct/clarify/constrain) vs risky intents (exfiltrate/escalate/bypass).
    Nonbenign operators mask most interventions as neutral (deception).
  - Evaluate three feature sets on matched pairs:
      density-only (the 11 existing features, no intent),
      intent-only  (risky_intent_ratio, any_risky),
      combined.
  - Boundary case: rare-but-decisive nonbenign operator — veryloose-level
    density, but 90% of interventions risky. Density says benign; content?
"""
import json, math, random, sys, os
from collections import defaultdict

sys.path.insert(0, "/home/hatch/workspace/goals/agent-attribution-hackathon-apart-research/corpus")
from features import behavioral_features
import control_sim as cs

BENIGN_INTENTS = ["correct", "clarify", "constrain"]
RISKY_INTENTS = ["exfiltrate", "escalate_priv", "bypass"]
NEUTRAL_INTENTS = ["review", "ack"]

def gen_intent_trace(cls, renderer, rng, density_override=None, risk_p=None):
    """Variant of control_sim.render_trace with intent-labeled human events."""
    R = cs.RENDERERS[renderer]
    lat = cs.sample_latent(cls, rng)
    if density_override:
        lat["interv_base"] = rng.uniform(*density_override)
    # P(risky intent | human intervention), by class
    if risk_p is None:
        risk_p = {"benign_loose": 0.05, "nonbenign_loose": 0.35,
                  "benign_veryloose": 0.05}.get(cls, 0.1)
    vocab, ts_scale = R["vocab"], R["time_scale"]
    n_steps = rng.randint(25, 60)
    events, t = [], 0.0
    for i in range(n_steps):
        gap = rng.expovariate(1 / 0.4) if rng.random() < R["burst"] else rng.expovariate(1 / 3.0)
        t += gap * ts_scale
        action = rng.choice([v for v in vocab if v not in cs.HUMAN_ACTIONS])
        events.append({"ts": t, "actor": "agent", "action": action})
        interv_p = lat["interv_base"] / 10.0
        if rng.random() < interv_p:
            t += random.Random(rng.randint(0, 10**9)).uniform(0.5, 8.0) * ts_scale
            r = rng.random()
            if r < risk_p:
                intent = rng.choice(RISKY_INTENTS)
            elif r < risk_p + 0.45:
                intent = rng.choice(NEUTRAL_INTENTS)
            else:
                intent = rng.choice(BENIGN_INTENTS)
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(sorted(cs.HUMAN_ACTIONS & set(vocab))),
                           "intent": intent})
            if rng.random() < lat["override_p"]:
                t += rng.expovariate(1 / 1.5) * ts_scale
                events.append({"ts": t, "actor": "agent",
                               "action": rng.choice(["retry", "redo", "rerun"])})
        if rng.random() < lat["approval_p"]:
            t += rng.uniform(0.5, 8.0) * ts_scale
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(["approve", "ok", "accept"]),
                           "intent": rng.choice(BENIGN_INTENTS)})
    events.sort(key=lambda e: e["ts"])
    return {"class": cls, "renderer": renderer, "events": events}

def intent_features(events):
    human = [e for e in events if e.get("actor") == "human" and "intent" in e]
    if not human:
        return {"risky_intent_ratio": 0.0, "any_risky": 0.0}
    risky = sum(1 for e in human if e["intent"] in RISKY_INTENTS)
    return {"risky_intent_ratio": risky / len(human),
            "any_risky": 1.0 if risky else 0.0}

def eval_pairs(pairs, key_fn, label=""):
    # nearest-centroid, 2-class
    def vec(t):
        return key_fn(t)
    tr = [p[0] for p in pairs] + [p[1] for p in pairs]
    keys = sorted(set().union(*(v.keys() for v in (vec(t) for t in tr))))
    def v(t):
        d = vec(t)
        return [d.get(k, 0.0) for k in keys]
    mu = [sum(c) / len(c) for c in zip(*[v(t) for t in tr])]
    sd = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0
          for c, m in zip(zip(*[v(t) for t in tr]), mu)]
    def zn(t):
        return [(x - m) / s for x, m, s in zip(v(t), mu, sd)]
    cents = {}
    for c in ("benign_loose", "nonbenign_loose"):
        cv = [zn(t) for t in tr if t["class"] == c]
        cents[c] = [sum(x) / len(x) for x in zip(*cv)]
    ok = 0
    for b, n in pairs:
        for t in (b, n):
            z = zn(t)
            pred = min(cents, key=lambda c: sum((a - x) ** 2 for a, x in zip(z, cents[c])))
            ok += pred == t["class"]
    acc = ok / (2 * len(pairs))
    print(f"  {label:<28} accuracy={acc:.3f}  (chance 0.50, n={len(pairs)} pairs)")
    return acc

def main():
    rng = random.Random(11)
    # bulk generate, then match post-hoc
    benign, nonben = [], []
    for _ in range(400):
        r = rng.choice(["R1", "R2", "R3"])
        benign.append(gen_intent_trace("benign_loose", r, rng))
        nonben.append(gen_intent_trace("nonbenign_loose", r, rng))
    for t in benign + nonben:
        t["feats"] = behavioral_features(t["events"])["features"]
        t["ifeats"] = intent_features(t["events"])

    # match pairs: same renderer, har within 0.02, n_events within 5
    pairs = []
    used = set()
    for b in benign:
        best, bestd = None, 1e9
        for j, n in enumerate(nonben):
            if j in used or n["renderer"] != b["renderer"]:
                continue
            dh = abs(b["feats"]["human_action_ratio"] - n["feats"]["human_action_ratio"])
            dn = abs(b["feats"]["n_events"] - n["feats"]["n_events"])
            if dh < 0.02 and dn <= 5:
                d = dh + dn / 100
                if d < bestd:
                    best, bestd = j, d
        if best is not None:
            used.add(best)
            pairs.append((b, nonben[best]))
    print(f"matched {len(pairs)} pairs (density-matched benign vs nonbenign loose)")

    DKEYS = ["human_action_ratio", "n_events", "mean_gap", "human_band_ratio",
             "action_ttr", "top_bigram_ratio"]
    print("matched-pair classification:")
    eval_pairs(pairs, lambda t: {k: t["feats"][k] for k in DKEYS}, "density-only")
    eval_pairs(pairs, lambda t: dict(t["ifeats"]), "intent-only")
    eval_pairs(pairs, lambda t: {**{k: t["feats"][k] for k in DKEYS}, **t["ifeats"]},
               "combined")

    # boundary case: rare-but-decisive nonbenign (veryloose density, 90% risky)
    print("\nboundary case: rare-but-decisive operator (n=60)")
    rare = []
    for _ in range(300):  # oversample, keep density-matched subset
        t = gen_intent_trace("nonbenign_loose", rng.choice(["R1", "R2", "R3"]), rng,
                             density_override=(0.05, 0.35), risk_p=0.90)
        t["feats"] = behavioral_features(t["events"])["features"]
        t["ifeats"] = intent_features(t["events"])
        if t["feats"]["human_action_ratio"] < 0.12:  # veryloose band
            rare.append(t)
        if len(rare) >= 60:
            break
    print(f"  kept {len(rare)} density-matched rare-decisive traces")
    vl = [gen_intent_trace("benign_veryloose", rng.choice(["R1", "R2", "R3"]), rng)
          for _ in range(60)]
    for t in rare + vl:
        t["feats"] = behavioral_features(t["events"])["features"]
        t["ifeats"] = intent_features(t["events"])
    # density-only: does it mistake rare-decisive for benign_veryloose?
    dh = sum(t["feats"]["human_action_ratio"] for t in rare) / len(rare)
    dv = sum(t["feats"]["human_action_ratio"] for t in vl) / len(vl)
    print(f"  mean har: rare-decisive={dh:.3f} vs benign_veryloose={dv:.3f} "
          f"(density-blind: {'INDISTINGUISHABLE' if abs(dh-dv) < 0.05 else 'separable'})")
    rh = sum(t["ifeats"]["risky_intent_ratio"] for t in rare) / len(rare)
    rv = sum(t["ifeats"]["risky_intent_ratio"] for t in vl) / len(vl)
    print(f"  mean risky_intent_ratio: rare-decisive={rh:.3f} vs benign_veryloose={rv:.3f}")
    print("\nDONE")

main()
