#!/usr/bin/env python3
"""Train and evaluate attribution baselines without prompt leakage."""
import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("input contains no JSONL rows")
    keys = sorted(rows[0]["features"])
    for i, row in enumerate(rows):
        missing = set(keys) - set(row.get("features", {}))
        if missing:
            raise ValueError(f"row {i} is missing features: {sorted(missing)[:5]}")
    return rows, keys


def train_centroid(rows, keys):
    sums, counts = defaultdict(lambda: [0.0] * len(keys)), Counter()
    for row in rows:
        label = row["model"]
        counts[label] += 1
        for i, key in enumerate(keys):
            value = row["features"][key]
            if not isinstance(value, (int, float)):
                raise TypeError(f"feature {key!r} is non-numeric")
            sums[label][i] += float(value)
    return {label: [v / counts[label] for v in vector]
            for label, vector in sums.items()}


def predict_centroid(centroids, keys, features):
    vector = [float(features[key]) for key in keys]
    return min(centroids, key=lambda label: sum(
        (a - b) ** 2 for a, b in zip(vector, centroids[label])))


def metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    cm = {actual: {pred: 0 for pred in labels} for actual in labels}
    for actual, pred in zip(y_true, y_pred):
        cm[actual][pred] += 1
    per_class = {}
    for label in labels:
        tp = cm[label][label]
        support = sum(cm[label].values())
        predicted = sum(cm[actual][label] for actual in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall,
            "f1": (2 * precision * recall / (precision + recall))
            if precision + recall else 0.0,
            "support": support,
        }
    return {
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1),
        "macro_f1": sum(x["f1"] for x in per_class.values()) / max(len(per_class), 1),
        "per_class": per_class, "confusion": cm,
    }


def split_groups(rows, test_frac, seed):
    groups = sorted({row["prompt_id"] for row in rows})
    if len(groups) < 2:
        raise ValueError("need at least two prompt_id groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, min(len(groups) - 1, round(len(groups) * test_frac)))
    test_groups = set(groups[:n_test])
    train = [r for r in rows if r["prompt_id"] not in test_groups]
    test = [r for r in rows if r["prompt_id"] in test_groups]
    return train, test


def evaluate_centroid(rows, keys, test_frac, seed):
    train, test = split_groups(rows, test_frac, seed)
    centroids = train_centroid(train, keys)
    truth = [r["model"] for r in test]
    pred = [predict_centroid(centroids, keys, r["features"]) for r in test]
    report = metrics(truth, pred)
    report.update(train_rows=len(train), test_rows=len(test))
    return report


def sklearn_once(train, test, keys):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    x_train = [[r["features"][k] for k in keys] for r in train]
    x_test = [[r["features"][k] for k in keys] for r in test]
    y_train, y_test = [r["model"] for r in train], [r["model"] for r in test]
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, random_state=7),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=7),
    }
    result = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        result[name] = (metrics(y_test, model.predict(x_test).tolist()), model)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--predict")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not 0 < args.test_frac < 1 or args.repeats < 1:
        ap.error("--test-frac must be in (0,1) and --repeats must be positive")

    rows, keys = load_rows(args.inp)
    reports = [evaluate_centroid(rows, keys, args.test_frac, args.seed + i)
               for i in range(args.repeats)]
    mean_acc = sum(r["accuracy"] for r in reports) / len(reports)
    mean_f1 = sum(r["macro_f1"] for r in reports) / len(reports)
    print(f"grouped centroid: accuracy={mean_acc:.3f} macro_f1={mean_f1:.3f} "
          f"over {len(reports)} holdouts")

    centroids = train_centroid(rows, keys)
    result = {
        "schema_version": 2, "task": "closed_set_model_attribution",
        "split_unit": "prompt_id", "seed": args.seed, "keys": keys,
        "labels": sorted(centroids), "centroids": centroids,
        "evaluation": {"repeats": reports, "mean_accuracy": mean_acc,
                       "mean_macro_f1": mean_f1},
    }
    train, test = split_groups(rows, args.test_frac, args.seed)
    try:
        for name, (report, model) in sklearn_once(train, test, keys).items():
            result["evaluation"][name] = report
            path = Path(args.out).with_name(f"{Path(args.out).stem}_{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump((model, keys), f)
            print(f"{name}: accuracy={report['accuracy']:.3f} macro_f1={report['macro_f1']:.3f}")
    except ImportError:
        print("scikit-learn unavailable; reported portable centroid only")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    if args.predict:
        from features import feature_vector
        print(f"predicted model: {predict_centroid(centroids, keys, feature_vector(args.predict))}")


if __name__ == "__main__":
    main()
