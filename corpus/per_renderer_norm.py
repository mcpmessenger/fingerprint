#!/usr/bin/env python3
"""Per-renderer normalization test (fair timing test).

Claude's critique: the full model's z-normed centroids mix renderer-specific
timing scales, so 0.315 cross-renderer may reflect a poorly specified model,
not uninformative timing. This test z-scores every feature WITHIN each
renderer before cross-renderer classification — the fair test of whether
timing carries portable signal.
"""
import json, math, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_sim import render_trace, featurize, centroid, dist, classify, LATENT, RENDERERS

def main():
    rng = random.Random(7)
    classes = list(LATENT)
    renderers = list(RENDERERS)
    traces = []
    for cls in classes:
        for r in renderers:
            for _ in range(40):
                traces.append(render_trace(cls, r, rng))
    keys = sorted(featurize(traces[0]).keys())
    feats = [(t["class"], t["renderer"], featurize(t)) for t in traces]

    # per-renderer z-scoring: compute mean/std of each feature within each renderer
    stats = {}
    for r in renderers:
        vecs = [[f.get(k, 0.0) for k in keys] for c, rr, f in feats if rr == r]
        n = len(vecs)
        mean = [sum(c) / n for c in zip(*vecs)]
        var = [sum((v[i] - mean[i]) ** 2 for v in vecs) / n for i in range(len(keys))]
        std = [math.sqrt(v) or 1.0 for v in var]
        stats[r] = (mean, std)

    def znorm(r, f):
        mean, std = stats[r]
        return {k: (f.get(k, 0.0) - mean[i]) / std[i] for i, k in enumerate(keys)}

    timing_keys = [k for k in keys if any(s in k for s in ("gap", "duration", "tick", "band", "burstiness"))]
    action_keys = [k for k in keys if k not in timing_keys and k != "human_action_ratio"]
    har_keys = ["human_action_ratio"]

    def run_eval(keyset, label):
        tr = [(c, znorm(r, f)) for c, r, f in feats if r in ("R1", "R2")]
        te = [(c, znorm(r, f)) for c, r, f in feats if r == "R3"]
        by = defaultdict(list)
        for c, f in tr:
            by[c].append([f.get(k, 0.0) for k in keyset])
        cents = {c: centroid(v) for c, v in by.items()}
        correct = sum(1 for c, f in te if classify(cents, keyset, f) == c)
        return {"label": label, "acc": round(correct / len(te), 3), "n_test": len(te)}

    report = [
        run_eval(keys, "per-renderer-norm, all feats, held-out R3"),
        run_eval(timing_keys, "per-renderer-norm, timing only, held-out R3"),
        run_eval(action_keys, "per-renderer-norm, action only, held-out R3"),
        run_eval(har_keys, "per-renderer-norm, har only, held-out R3"),
    ]
    print(json.dumps(report, indent=1))
    with open("/tmp/per_renderer_norm.json", "w") as f:
        json.dump(report, f, indent=1)

if __name__ == "__main__":
    main()
