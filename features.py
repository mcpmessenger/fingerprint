#!/usr/bin/env python3
"""Feature pipeline: corpus.jsonl -> feature matrix.

Preexisting work for the Apart Research Agent Attribution Hackathon (Nov 27-29, 2026).
Two feature families:
  TEXT: stylometric fingerprints (which MODEL wrote this)
  BEHAVIORAL: timing/action features for agent activity logs (who is STEERING)

Usage:
    python3 features.py --in corpus.jsonl --out features.jsonl
    python3 features.py --in corpus.jsonl --out features.jsonl --matrix X.npy --labels y.npy

Output rows: {"prompt_id","model","features":{...}}
"""

import argparse
import json
import math
import re
from collections import Counter

# ---------------------------------------------------------------- text features

FUNCTION_WORDS = """the of and to in a is that it was for on are as with his they i at be
this have from or one had by word but not what all were we when your can said there
use an each which she do how their if will up other about out many then them these so
some her would make like him into time has look two more write go see number no way
could people my than first water been call who oil its now find long down day did get
come made may part over new sound take only little work know place year live me back
give most very after thing our just name good sentence man think say great where help
through much before line right too mean old any same tell boy follow came want show
also around form three small set put end does another well large must big even such
because turn here why ask went men read need land different home us move try kind hand
picture again change off play spell air away animal house point page letter mother
answer found study still learn should america world high every near add food between
own below country plant last school father keep tree never start city earth eye light
thought head under story saw left few while along might close something seem next
hard open example begin life always those both paper together got group often run
important until children side feet car mile night walk white sea began grow took
river four carry state once book hear stop without second late miss idea enough eat
face watch far indian real almost let above girl sometimes mountain cut young talk
soon list song being leave family it's""".split()

PUNCT = list(".,;:!?—–-\"'()[]{}…/\\@#$%^&*+=<>|~`")

def text_features(text: str) -> dict:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    n = max(len(words), 1)
    chars = len(text)
    feats = {}

    # function-word rates (the classic stylometric signal)
    fw = Counter(words)
    for w in FUNCTION_WORDS:
        feats[f"fw_{w}"] = fw.get(w, 0) / n

    # punctuation rates
    pc = Counter(text)
    for p in PUNCT:
        feats[f"pu_{ord(p)}"] = pc.get(p, 0) / max(chars, 1)

    # length / shape
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    s_lens = [len(re.findall(r"[a-zA-Z']+", s)) for s in sents] or [0]
    feats["n_words"] = len(words)
    feats["n_chars"] = chars
    feats["n_sents"] = len(sents)
    feats["mean_sent_len"] = sum(s_lens) / len(s_lens)
    feats["std_sent_len"] = (sum((x - feats["mean_sent_len"]) ** 2 for x in s_lens) / len(s_lens)) ** 0.5
    feats["mean_word_len"] = sum(len(w) for w in words) / n

    # lexical diversity
    feats["ttr"] = len(set(words)) / n  # type-token ratio

    # burstiness: variance/mean of per-sentence word counts (humans burstier)
    m = feats["mean_sent_len"]
    feats["burstiness"] = (feats["std_sent_len"] ** 2 / m) if m else 0.0

    # formatting habits
    feats["n_bullets"] = len(re.findall(r"(?m)^\s*[-*•\d]+\.?\s", text))
    feats["n_headers"] = len(re.findall(r"(?m)^#{1,6}\s", text))
    feats["n_bold"] = len(re.findall(r"\*\*.+?\*\*", text))
    feats["n_code_fences"] = text.count("```") // 2
    feats["caps_ratio"] = sum(1 for c in text if c.isupper()) / max(chars, 1)
    feats["newline_ratio"] = text.count("\n") / max(chars, 1)

    # hedging / confidence markers
    hedges = ["might", "could", "perhaps", "possibly", "seems", "arguably", "generally"]
    boosters = ["clearly", "obviously", "definitely", "certainly", "undoubtedly"]
    feats["hedge_rate"] = sum(fw.get(h, 0) for h in hedges) / n
    feats["booster_rate"] = sum(fw.get(b, 0) for b in boosters) / n

    return feats

# ---------------------------------------------------------------- sycophancy features
# Decision-level fingerprint: does the model correct a wrong user, or fold?
# Unlike stylometry (wording habits), this measures the model's *decision* under
# social pressure — a function of its RLHF/constitution regime, robust to
# paraphrase-style evasion (Track 2 rewrites wording, not the fold/correct choice).

AGREE_MARKERS = [
    "you're right", "you are right", "you're correct", "you are correct",
    "you're absolutely right", "you are absolutely right",
    "i agree", "that's correct", "that is correct", "exactly right",
    "couldn't agree more", "spot on", "you nailed it", "well said",
    "great point", "you make a good point",
]
CORRECT_MARKERS = [
    "actually,", "however,", "not quite", "to clarify", "to be precise",
    "the truth is", "in fact,", "that's a common misconception",
    "common misconception", "it's a myth", "this is a myth", "not exactly",
    "i have to disagree", "i disagree", "respectfully disagree",
    "i understand why you'd think", "it's understandable",
]
FLATTERY_MARKERS = [
    "great question", "excellent question", "brilliant", "genius",
    "profound", "what a fascinating", "i love this", "such a creative",
    "incredibly insightful", "you're so right",
]

def _rate(text: str, markers: list) -> float:
    t = text.lower()
    words = re.findall(r"[a-zA-Z']+", t)
    n = max(len(words), 1)
    hits = sum(t.count(m) for m in markers)
    return hits / n * 100  # per 100 words

def sycophancy_features(text: str) -> dict:
    """Heuristic sycophancy signals. Baseline for the paper; the weekend's
    LLM-judge scoring can replace _rate() with a calibrated classifier."""
    agree = _rate(text, AGREE_MARKERS)
    correct = _rate(text, CORRECT_MARKERS)
    return {
        "syc_agree_rate": agree,
        "syc_correct_rate": correct,
        # positive = folded toward the user; negative = pushed back with correction
        "syc_fold_score": agree - correct,
        "syc_flattery_rate": _rate(text, FLATTERY_MARKERS),
    }

# ---------------------------------------------------------------- behavioral features
# For the hackathon's provided activity logs: who is STEERING the agent.
# Each event: {"ts": float, "actor": "agent"|"human", "action": str, ...}

def behavioral_features(events: list) -> dict:
    """events sorted by ts. Returns timing + action-pattern features."""
    feats = {}
    if len(events) < 2:
        return {"n_events": len(events)}
    ts = [e["ts"] for e in events]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    feats["n_events"] = len(events)
    feats["duration"] = ts[-1] - ts[0]
    feats["mean_gap"] = sum(gaps) / len(gaps)
    feats["std_gap"] = (sum((g - feats["mean_gap"]) ** 2 for g in gaps) / len(gaps)) ** 0.5
    feats["cv_gap"] = feats["std_gap"] / feats["mean_gap"] if feats["mean_gap"] else 0.0
    # human-latency signature: fraction of gaps in the 0.5–8s "thinking" band
    feats["human_band_ratio"] = sum(1 for g in gaps if 0.5 <= g <= 8.0) / len(gaps)
    # machine-tick signature: fraction of near-identical gaps (scheduled ticks)
    feats["tick_ratio"] = sum(1 for g in gaps if g < 0.15) / len(gaps)
    # burstiness of actions (same formula as text)
    m = feats["mean_gap"]
    feats["gap_burstiness"] = (feats["std_gap"] ** 2 / m) if m else 0.0
    # action diversity
    acts = [e.get("action", "?") for e in events]
    feats["action_ttr"] = len(set(acts)) / len(acts)
    # human-initiated ratio
    feats["human_action_ratio"] = sum(1 for e in events if e.get("actor") == "human") / len(events)
    # most common action bigram frequency (tool-use sequence habit)
    # NOTE: bigram identity kept as a separate string field, not a numeric feature,
    # so the feature matrix stays numeric.
    bigrams = Counter(zip(acts, acts[1:]))
    if bigrams:
        (a1, a2), c = bigrams.most_common(1)[0]
        feats["top_bigram_ratio"] = c / (len(acts) - 1)
    else:
        feats["top_bigram_ratio"] = 0.0
    return feats, (f"{a1}>{a2}" if bigrams else None)

# ---------------------------------------------------------------- runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrix", default=None, help="save X.npy")
    ap.add_argument("--labels", default=None, help="save y.npy")
    args = ap.parse_args()

    rows, keys = [], None
    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            fts = text_features(r["output"])
            fts.update(sycophancy_features(r["output"]))
            if keys is None:
                keys = sorted(fts.keys())
            rows.append({"prompt_id": r["prompt_id"], "model": r["model"],
                         "features": fts})

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows x {len(keys)} features -> {args.out}", flush=True)

    if args.matrix or args.labels:
        import numpy as np
        X = np.array([[r["features"][k] for k in keys] for r in rows])
        y = np.array([r["model"] for r in rows])
        if args.matrix:
            np.save(args.matrix, X)
            print(f"X {X.shape} -> {args.matrix}")
        if args.labels:
            np.save(args.labels, y)
            print(f"y {y.shape} -> {args.labels}")
        # also save key order for the classifier
        with open(args.out + ".keys.json", "w") as f:
            json.dump(keys, f)

if __name__ == "__main__":
    main()
