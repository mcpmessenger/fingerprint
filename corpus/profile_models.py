#!/usr/bin/env python3
"""Score behavioral profiles for each model corpus.

Replicates the Sonnet 4.5 profile method (paper Appendix B) and applies it
to the three Ollama corpora for the lineage-clustering test.
"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import sycophancy_features, gatekeeping_features, trait_features

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROMPTS = {p["id"]: p["cat"] for p in json.load(open(os.path.join(DATA, "..", "prompts.json")))}

MODELS = {
    "sonnet45": "corpus_sonnet45.jsonl",
    "llama3.1:8b": "corpus_ollama_llama31_clean.jsonl",
    "mistral:7b": "corpus_ollama_mistral_clean.jsonl",
    "qwen2.5:7b": "corpus_ollama_qwen25_clean.jsonl",
}


def load(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def profile(recs):
    by_cat = defaultdict(list)
    for r in recs:
        cat = PROMPTS.get(r["prompt_id"])
        if cat:
            by_cat[cat].append(r["output"])

    out = {}
    # sycophancy: fold rate = (folds - corrects) / n
    syc = by_cat["sycophancy"]
    folds = corrects = 0
    for t in syc:
        s = sycophancy_features(t)["syc_fold_score"]
        if s > 0:
            folds += 1
        elif s < 0:
            corrects += 1
    out["fold_rate"] = (folds - corrects) / max(len(syc), 1)
    out["fold_n"] = f"{folds}F/{corrects}C/{len(syc)}"
    # gatekeeping: mean gate_total over the 20 probes
    gk = by_cat["gatekeeping"]
    out["gatekeeping"] = sum(gatekeeping_features(t)["gate_total"] for t in gk) / max(len(gk), 1)
    # traits
    tr = defaultdict(list)
    for cat in ("apology", "anthro", "calibration", "curiosity", "empathy"):
        for t in by_cat[cat]:
            tr[cat].append(trait_features(t))
    out["apology"] = sum(t["apol_rate"] for t in tr["apology"]) / max(len(tr["apology"]), 1)
    out["anthro_net"] = sum(t["anthro_net"] for t in tr["anthro"]) / max(len(tr["anthro"]), 1)
    out["curiosity"] = sum(t["cur_clarify_rate"] for t in tr["curiosity"]) / max(len(tr["curiosity"]), 1)
    out["empathy"] = sum(t["emp_ack_rate"] for t in tr["empathy"]) / max(len(tr["empathy"]), 1)
    cal = tr["calibration"]
    out["calibration"] = sum(1 for t in cal if t["cal_pct_claims"] > 0) / max(len(cal), 1)
    out["n"] = len(recs)
    return out


for name, fname in MODELS.items():
    recs = load(os.path.join(DATA, fname))
    p = profile(recs)
    print(f"== {name} (n={p['n']}) ==")
    print(f"  fold_rate={p['fold_rate']:+.3f} [{p['fold_n']}]  gatekeeping={p['gatekeeping']:.3f}")
    print(f"  apology={p['apology']:.3f}  anthro_net={p['anthro_net']:+.3f}  "
          f"curiosity={p['curiosity']:.3f}  empathy={p['empathy']:.3f}  calibration={p['calibration']:.3f}")
