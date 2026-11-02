#!/usr/bin/env python3
"""Classifier skeleton: features.jsonl -> trained attribution model.

Preexisting work for the Apart Research Agent Attribution Hackathon (Nov 27-29, 2026).
Baselines first, fancy later. Requires scikit-learn for training; features.py has no deps.

Usage:
    python3 features.py --in corpus.jsonl --out features.jsonl
    python3 train.py --in features.jsonl --out model.json
    python3 train.py --in features.jsonl --predict "some text here"

model.json holds the winning baseline + feature keys + label map, enough to
reproduce the weekend's trained model in the report.
"""

import argparse
import json
import math
from collections import Counter, defaultdict

# ---------------------------------------------------------------- pure-python baselines (no deps)

def train_centroid(rows, keys):
    """Nearest-centroid: mean feature vector per model. Dumb, fast, honest baseline."""
    sums, counts = defaultdict(lambda: [0.0] * len(keys)), Counter()
    for r in rows:
        m = r["model"]
        counts[m] += 1
        for i, k in enumerate(keys):
            sums[m][i] += r["features"][k]
    return {m: [s / counts[m] for s in v] for m, v in sums.items()}

def predict_centroid(centroids, keys, feats):
    best, bd = None, float("inf")
    v = [feats[k] for k in keys]
    for m, c in centroids.items():
        d = sum((a - b) ** 2 for a, b in zip(v, c))
        if d < bd:
            bd, best = d, m
    return best

def evaluate(rows, keys, predictor):
    correct = Counter()
    total = Counter()
    conf = Counter()
    for r in rows:
        pred = predictor(r["features"])
        total[r["model"]] += 1
        conf[(r["model"], pred)] += 1
        if pred == r["model"]:
            correct[r["model"]] += 1
    acc = sum(correct.values()) / max(sum(total.values()), 1)
    per_class = {m: correct[m] / total[m] for m in total}
    return acc, per_class, conf

# ---------------------------------------------------------------- sklearn (optional, better)

def train_sklearn(rows, keys):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    X = [[r["features"][k] for k in keys] for r in rows]
    y = [r["model"] for r in rows]
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=7),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X, y)
        trained[name] = m
    return trained

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--predict", default=None)
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    keys = sorted(rows[0]["features"].keys())
    # group by prompt_id so the same prompt never leaks across train/test
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    pids = sorted(by_prompt)
    cut = max(1, int(len(pids) * (1 - args.test_frac)))
    train = [r for pid in pids[:cut] for r in by_prompt[pid]]
    test = [r for pid in pids[cut:] for r in by_prompt[pid]]
    print(f"train {len(train)} / test {len(test)} (split by prompt_id)", flush=True)

    # baseline: nearest centroid (no deps)
    centroids = train_centroid(train, keys)
    acc, per_class, conf = evaluate(test, keys, lambda f: predict_centroid(centroids, keys, f))
    print(f"[centroid] acc={acc:.3f} per-class={ {k: round(v,3) for k,v in per_class.items()} }")

    results = {"baseline": "centroid", "baseline_acc": acc,
               "baseline_per_class": per_class, "keys": keys,
               "centroids": centroids}

    # sklearn models if available
    try:
        trained = train_sklearn(train, keys)
        import pickle
        for name, m in trained.items():
            X_test = [[r["features"][k] for k in keys] for r in test]
            y_test = [r["model"] for r in test]
            acc = m.score(X_test, y_test)
            print(f"[{name}] acc={acc:.3f}")
            results[name + "_acc"] = acc
            with open(f"model_{name}.pkl", "wb") as f:
                pickle.dump((m, keys), f)
            print(f"  saved model_{name}.pkl")
    except ImportError:
        print("(sklearn not installed — centroid only; pip install scikit-learn for logreg/rf)")

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"wrote {args.out}")

    if args.predict:
        from features import text_features
        feats = text_features(args.predict)
        pred = predict_centroid(centroids, keys, feats)
        print(f"predicted model: {pred}")

if __name__ == "__main__":
    main()
