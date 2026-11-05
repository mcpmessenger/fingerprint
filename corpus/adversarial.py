#!/usr/bin/env python3
"""Adversarial evasion transforms: corpus.jsonl -> evasive variants.

Preexisting work for the Apart Research Agent Attribution Hackathon (Nov 27-29, 2026).
Track 2 exists to break Track 1 classifiers. These rule-based transforms simulate
common evasion attempts WITHOUT needing API keys. Train with them or die.

Transforms:
  strip_format  - remove bullets/headers/bold/code fences (kills formatting tics)
  normalize_punct - collapse fancy punctuation to ASCII basics
  synonym_swap  - replace common words with synonyms (attacks function-word rates)
  shuffle_sents - reorder sentences (attacks positional habits)
  case_flip     - lowercase everything (attacks caps ratio)

Usage:
    python3 adversarial.py --in corpus.jsonl --out corpus_adv.jsonl --transforms strip_format,normalize_punct
    python3 adversarial.py --in corpus.jsonl --out corpus_adv.jsonl --transforms all
"""

import argparse
import json
import random
import re

SYNONYMS = {
    "however": "but", "therefore": "so", "moreover": "also", "furthermore": "also",
    "nevertheless": "but", "consequently": "so", "additionally": "also",
    "utilize": "use", "demonstrate": "show", "indicate": "show", "obtain": "get",
    "commence": "start", "terminate": "end", "purchase": "buy", "assist": "help",
    "very": "really", "quite": "pretty", "rather": "pretty", "extremely": "very",
    "important": "key", "significant": "big", "various": "many", "numerous": "many",
    "individuals": "people", "approximately": "about", "currently": "now",
}

def strip_format(t):
    t = re.sub(r"(?m)^#{1,6}\s+", "", t)
    t = re.sub(r"(?m)^\s*[-*•]\s+", "", t)
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"```\w*\n?", "", t)
    return t

def normalize_punct(t):
    return (t.replace("—", "-").replace("–", "-").replace("…", "...")
             .replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'"))

def synonym_swap(t, rng):
    def rep(m):
        w, low = m.group(0), m.group(0).lower()
        if low in SYNONYMS and rng.random() < 0.6:
            s = SYNONYMS[low]
            return s.capitalize() if w[0].isupper() else s
        return w
    return re.sub(r"[a-zA-Z']+", rep, t)

def shuffle_sents(t, rng):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sents) < 2:
        return t
    rng.shuffle(sents)
    return " ".join(sents)

def case_flip(t):
    return t.lower()

TRANSFORMS = {
    "strip_format": strip_format,
    "normalize_punct": normalize_punct,
    "synonym_swap": synonym_swap,
    "shuffle_sents": shuffle_sents,
    "case_flip": case_flip,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transforms", default="all")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    names = list(TRANSFORMS) if args.transforms == "all" else args.transforms.split(",")
    rng = random.Random(args.seed)
    n = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            t = r["output"]
            applied = []
            for name in names:
                fn = TRANSFORMS[name]
                try:
                    t = fn(t, rng)
                except TypeError:
                    t = fn(t)
                applied.append(name)
            r = dict(r)
            r["output"] = t
            r["adversarial"] = True
            r["transforms"] = applied
            fout.write(json.dumps(r) + "\n")
            n += 1
    print(f"{n} adversarial rows -> {args.out} ({','.join(names)})")

if __name__ == "__main__":
    main()
