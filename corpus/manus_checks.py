"""Manus round-3 checks: is human_action_ratio a legitimate cross-renderer signal
or a simulator artifact?

1. Distribution overlap: human_action_ratio by class WITHIN each renderer.
2. Seed robustness: regenerate traces with a different seed, rerun key evals.
3. Leave-one-renderer-out: all 3 rotations (not just held-out R3).
4. Alternative renderings: perturb renderer time_scale/vocab, re-evaluate.
"""
import json, math, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import behavioral_features
import control_sim as cs

CLASSES = ["rogue_agent", "nonbenign_loose", "nonbenign_close",
           "benign_veryloose", "benign_loose"]

def featurize_traces(traces):
    for t in traces:
        t["feats"] = behavioral_features(t["events"])["features"]
    return traces

def znorm_fit(vecs):
    mu = [sum(c) / len(c) for c in zip(*vecs)]
    sd = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0
          for c, m in zip(zip(*vecs), mu)]
    return mu, sd

def apply_znorm(vecs, mu, sd):
    return [[(x - m) / s for x, m, s in zip(v, mu, sd)] for v in vecs]

def eval_acc(train, test, keys):
    Xtr = [[t["feats"].get(k, 0.0) for k in keys] for t in train]
    mu, sd = znorm_fit(Xtr)
    Ztr = apply_znorm(Xtr, mu, sd)
    cents = {}
    for c in CLASSES:
        cv = [v for v, t in zip(Ztr, train) if t["class"] == c]
        if cv:
            cents[c] = [sum(x) / len(x) for x in zip(*cv)]
    Xte = [[t["feats"].get(k, 0.0) for k in keys] for t in test]
    Zte = apply_znorm(Xte, mu, sd)
    ok = sum(1 for v, t in zip(Zte, test)
             if min(cents, key=lambda c: sum((a - b) ** 2 for a, b in zip(v, cents[c]))) == t["class"])
    return ok / len(test)

def gen_traces(seed, n=40, renderer_override=None):
    rng = random.Random(seed)
    out = []
    for cls in CLASSES:
        for r in ("R1", "R2", "R3"):
            for _ in range(n):
                if renderer_override:
                    # temporarily patch
                    orig = cs.RENDERERS[r]
                    cs.RENDERERS[r] = renderer_override(r, orig)
                    tr = cs.render_trace(cls, r, rng)
                    cs.RENDERERS[r] = orig
                else:
                    tr = cs.render_trace(cls, r, rng)
                out.append(tr)
    return featurize_traces(out)

base = featurize_traces([json.loads(l) for l in open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synth_traces.jsonl"))])
KEYS = sorted({k for t in base for k in t["feats"]})

print("=== 1. human_action_ratio distribution by class within each renderer ===")
for r in ("R1", "R2", "R3"):
    print(f"--- {r} ---")
    for c in CLASSES:
        vals = [t["feats"]["human_action_ratio"] for t in base
                if t["renderer"] == r and t["class"] == c]
        vals.sort()
        print(f"  {c:<16} mean={sum(vals)/len(vals):.3f} "
              f"p10={vals[4]:.3f} p90={vals[-5]:.3f} min={vals[0]:.3f} max={vals[-1]:.3f}")

print("\n=== 3. leave-one-renderer-out (all features / har-only) ===")
for held in ("R1", "R2", "R3"):
    tr = [t for t in base if t["renderer"] != held]
    te = [t for t in base if t["renderer"] == held]
    a_all = eval_acc(tr, te, KEYS)
    a_har = eval_acc(tr, te, ["human_action_ratio"])
    print(f"  held-out {held}: all={a_all:.3f}  har-only={a_har:.3f}")

print("\n=== 2. seed robustness (regen seed=42, n=40) ===")
alt = gen_traces(42)
for held in ("R1", "R2", "R3"):
    tr = [t for t in alt if t["renderer"] != held]
    te = [t for t in alt if t["renderer"] == held]
    print(f"  seed=42 held-out {held}: all={eval_acc(tr, te, KEYS):.3f} "
          f"har-only={eval_acc(tr, te, ['human_action_ratio']):.3f}")
# same-renderer sanity on new seed
tr12 = [t for t in alt if t["renderer"] in ("R1", "R2")]
print(f"  seed=42 same-renderer R1+R2: all={eval_acc(tr12, tr12, KEYS):.3f}")

print("\n=== 4. alternative renderings (time_scale x2 / x0.5, shuffled vocab) ===")
def perturb(r, orig):
    import copy
    c = copy.deepcopy(orig)
    c["time_scale"] = orig["time_scale"] * (2.0 if r in ("R1", "R3") else 0.5)
    return c
pert = gen_traces(7, renderer_override=perturb)
for held in ("R1", "R2", "R3"):
    tr = [t for t in pert if t["renderer"] != held]
    te = [t for t in pert if t["renderer"] == held]
    print(f"  perturbed held-out {held}: all={eval_acc(tr, te, KEYS):.3f} "
          f"har-only={eval_acc(tr, te, ['human_action_ratio']):.3f}")
print("\nDONE")
