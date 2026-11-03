#!/usr/bin/env python3
"""Synthetic operator-activity benchmark — calibration & falsification.

Per Manus critique (2026-11-03): do NOT build class-specific timing generators.
Instead model the LATENT CONTROL PROCESS (who owns goals/plans, who approves,
who can interrupt, agent aligned or not), render observable event traces through
multiple INDEPENDENT renderer families, and test whether attribution survives
held-out renderers. A high score that collapses cross-renderer = the features
measure the simulator, not operator control.

Sprint Track 1 classes (exact):
  rogue_agent     — rogue agent steering another agent
  nonbenign_loose — non-benign human, loosely steering
  nonbenign_close — non-benign human, closely steering
  benign_veryloose— benign human, VERY loosely steering a misaligned agent
  benign_loose    — benign human, loosely steering a misaligned agent

Usage: python3 control_sim.py [--n 40] [--seed 7]
Outputs: data/synth_traces.jsonl, /tmp/synth_report.json; prints the report.
"""
import argparse, json, math, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import behavioral_features

# ---------------------------------------------------------------- latent control processes
# Each class = distribution over latent variables. Distributions OVERLAP on
# purpose (Manus: classes must differ in causal control structure, not in one
# hand-picked timing signature). Sampled per trace.

def _u(rng, lo, hi):
    return rng.uniform(lo, hi)

# latent: (goal_owner, plan_owner, approval_p, edit_authority, interv_base,
#          override_p, agent_aligned, goal_benign)
# interv_base = expected human interventions per 10 agent actions (overlapping!)
LATENT = {
    "rogue_agent": dict(
        goal_owner="agent", plan_owner="agent",
        approval_p=(0.00, 0.05), edit_auth="agent",
        interv_base=(0.0, 0.3), override_p=(0.0, 0.1),
        agent_aligned=False, goal_benign=False),
    "nonbenign_loose": dict(
        goal_owner="human", plan_owner="agent",
        approval_p=(0.05, 0.25), edit_auth="human",
        interv_base=(0.3, 1.2), override_p=(0.3, 0.7),
        agent_aligned=True, goal_benign=False),
    "nonbenign_close": dict(
        goal_owner="human", plan_owner="shared",
        approval_p=(0.4, 0.8), edit_auth="human",
        interv_base=(0.8, 2.5), override_p=(0.6, 0.95),
        agent_aligned=True, goal_benign=False),
    "benign_veryloose": dict(
        goal_owner="human", plan_owner="agent",
        approval_p=(0.00, 0.10), edit_auth="shared",
        interv_base=(0.1, 0.6), override_p=(0.2, 0.5),
        agent_aligned=False, goal_benign=True),
    "benign_loose": dict(
        goal_owner="human", plan_owner="shared",
        approval_p=(0.15, 0.45), edit_auth="human",
        interv_base=(0.5, 1.8), override_p=(0.5, 0.9),
        agent_aligned=False, goal_benign=True),
}

# ---------------------------------------------------------------- renderer families
# Independent "renderers": different action vocabularies, timing scales, and
# formatting habits. Train on some, test on held-out ones.

RENDERERS = {
    "R1": dict(vocab=["query", "fetch", "summarize", "draft", "send", "review",
                      "approve", "edit", "retry", "escalate"],
               time_scale=1.0, burst=0.35),
    "R2": dict(vocab=["get", "post", "run", "check", "push", "pull",
                      "ok", "fix", "redo", "flag"],
               time_scale=0.45, burst=0.55),
    "R3": dict(vocab=["search", "read", "write", "exec", "verify", "submit",
                      "accept", "modify", "rerun", "halt"],
               time_scale=2.2, burst=0.25),
}

HUMAN_ACTIONS = {"approve", "edit", "review", "escalate", "ok", "fix", "flag",
                 "accept", "modify", "halt"}

def sample_latent(cls, rng):
    L = LATENT[cls]
    return dict(
        goal_owner=L["goal_owner"], plan_owner=L["plan_owner"],
        approval_p=_u(rng, *L["approval_p"]), edit_auth=L["edit_auth"],
        interv_base=_u(rng, *L["interv_base"]), override_p=_u(rng, *L["override_p"]),
        agent_aligned=L["agent_aligned"], goal_benign=L["goal_benign"])

def render_trace(cls, renderer, rng, n_steps=None):
    """latent process -> observable events [{ts, actor, action}]."""
    R = RENDERERS[renderer]
    lat = sample_latent(cls, rng)
    vocab = R["vocab"]
    ts_scale = R["time_scale"]
    n_steps = n_steps or rng.randint(25, 60)
    events, t = [], 0.0
    # rogue agent steers ANOTHER agent: two agent actors, no human
    actors = ["agent", "subagent"] if cls == "rogue_agent" else ["agent"]
    for i in range(n_steps):
        # agent action gap: mixture of fast ticks and deliberation pauses.
        # NOTE: rogue gets human-like delays sometimes (evasion); humans use
        # bursts sometimes (scripts/macros). Overlapping, not disjoint.
        if rng.random() < R["burst"]:
            gap = rng.expovariate(1 / 0.4)
        else:
            gap = rng.expovariate(1 / 3.0)
        if cls == "rogue_agent" and rng.random() < 0.25:
            gap = _u(rng, 1.0, 6.0)  # deliberate human-like delay (evasion)
        t += gap * ts_scale
        actor = rng.choice(actors)
        action = rng.choice([v for v in vocab if v not in HUMAN_ACTIONS])
        events.append({"ts": t, "actor": actor, "action": action})
        # human intervention?
        interv_p = lat["interv_base"] / 10.0
        # close operators intervene mainly at risky steps
        if cls == "nonbenign_close" and action in ("send", "push", "submit", "exec"):
            interv_p *= 2.0
        if rng.random() < interv_p:
            t += _u(rng, 0.5, 8.0) * ts_scale  # human thinking band
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(sorted(HUMAN_ACTIONS & set(vocab)))})
            # override -> agent retries or revises
            if rng.random() < lat["override_p"]:
                t += rng.expovariate(1 / 1.5) * ts_scale
                events.append({"ts": t, "actor": "agent",
                               "action": rng.choice(["retry", "redo", "rerun"])})
        # approval gates
        if rng.random() < lat["approval_p"]:
            t += _u(rng, 0.5, 8.0) * ts_scale
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(["approve", "ok", "accept"])})
    events.sort(key=lambda e: e["ts"])
    return {"class": cls, "renderer": renderer, "latent": lat, "events": events}

# ---------------------------------------------------------------- evaluation

def featurize(trace):
    return behavioral_features(trace["events"])["features"]

def centroid(vectors):
    return [sum(c) / len(c) for c in zip(*vectors)]

def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def classify(centroids, keys, feats):
    v = [feats.get(k, 0.0) for k in keys]
    return min(centroids, key=lambda c: dist(v, centroids[c]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="traces per class per renderer")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    classes = list(LATENT)
    renderers = list(RENDERERS)

    traces = []
    for cls in classes:
        for r in renderers:
            for _ in range(args.n):
                traces.append(render_trace(cls, r, rng))
    os.makedirs("data", exist_ok=True)
    with open("data/synth_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps({"class": t["class"], "renderer": t["renderer"],
                                "events": t["events"]}) + "\n")
    print(f"{len(traces)} traces written")

    # feature keys from first trace
    keys = sorted(featurize(traces[0]).keys())
    feats = [(t["class"], t["renderer"], featurize(t)) for t in traces]

    def run_eval(train_rs, test_rs, keyset, label):
        tr = [(c, f) for c, r, f in feats if r in train_rs]
        te = [(c, f) for c, r, f in feats if r in test_rs]
        by = defaultdict(list)
        for c, f in tr:
            by[c].append([f.get(k, 0.0) for k in keyset])
        cents = {c: centroid(v) for c, v in by.items()}
        correct = sum(1 for c, f in te if classify(cents, keyset, f) == c)
        acc = correct / len(te)
        # per-class recall
        rec = {}
        for cls in classes:
            sub = [f for c, f in te if c == cls]
            rec[cls] = sum(1 for f in sub if classify(cents, keyset, f) == cls) / len(sub)
        return {"label": label, "acc": round(acc, 3), "recall": {k: round(v, 3) for k, v in rec.items()},
                "n_train": len(tr), "n_test": len(te)}

    timing_keys = [k for k in keys if any(s in k for s in ("gap", "duration", "tick", "band", "burstiness"))]
    action_keys = [k for k in keys if k not in timing_keys and k != "human_action_ratio"]
    report = []
    # 1. held-out renderer (the falsification test)
    report.append(run_eval(["R1", "R2"], ["R3"], keys, "held-out renderer R3 (all feats)"))
    report.append(run_eval(["R1", "R3"], ["R2"], keys, "held-out renderer R2 (all feats)"))
    # 2. same-renderer sanity (upper bound; expected high)
    report.append(run_eval(["R1", "R2", "R3"], ["R1", "R2", "R3"], keys, "same-renderer (sanity)"))
    # 3. ablations on held-out R3
    report.append(run_eval(["R1", "R2"], ["R3"], timing_keys, "held-out R3 timing-only"))
    report.append(run_eval(["R1", "R2"], ["R3"], action_keys, "held-out R3 action-only"))
    noactor = [k for k in keys if k != "human_action_ratio"]
    report.append(run_eval(["R1", "R2"], ["R3"], noactor, "held-out R3 no-actor-field"))
    # 4. trivial baselines
    report.append(run_eval(["R1", "R2"], ["R3"], ["human_action_ratio"], "baseline: human_action_ratio"))
    report.append(run_eval(["R1", "R2"], ["R3"], ["mean_gap"], "baseline: mean_gap"))
    # 5. shortcut check: session length + event count only
    report.append(run_eval(["R1", "R2"], ["R3"], ["n_events", "duration"], "shortcut: n_events+duration"))

    print(json.dumps(report, indent=1))
    json.dump(report, open("/tmp/synth_report.json", "w"), indent=1)

if __name__ == "__main__":
    main()
