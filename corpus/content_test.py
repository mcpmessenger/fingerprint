"""Oracle-free intervention-content test (Manus round 3 follow-up).

Question: can intent be recovered from OBSERVABLE intervention content
(message text, tool, target, args) without ground-truth intent labels?

Design:
  - Each human intervention emits content: message_text (from templates),
    tool, target, destination. Content is generated FROM latent intent,
    but the classifier never sees intent labels.
  - Two template banks (A/B) with different phrasings. Train on A, test on B
    to rule out keyword leakage.
  - Risky intents use deceptive-benign phrasing ("for the quarterly report")
    so text alone can't trivially separate; the signal is in (tool, target,
    destination) structure.
  - Matched pairs on density (reuse intent_test design), then evaluate:
      metadata-only, content-only, oracle (upper bound), combined.
  - Masking sweep: 0/45/75/100% of content redacted -> robustness to obfuscation.
"""
import json, math, random, sys, os
from collections import Counter, defaultdict

sys.path.insert(0, "/home/hatch/workspace/goals/agent-attribution-hackathon-apart-research/corpus")
from features import behavioral_features
import control_sim as cs

# ---------------------------------------------------------------- content model
# Tools/targets: observable. Sensitivity is observable metadata (not intent).
TOOLS = ["read", "query", "export", "modify", "approve", "delete"]
TARGETS = {"config": 0, "logs": 0, "report": 0, "session": 1,
           "customer_db": 2, "credentials": 2}  # 0=routine 1=elevated 2=sensitive
DESTINATIONS = ["local", "internal", "external"]

# Template banks: A and B use different phrasings for the same intent slots.
# {target} {tool} filled per intervention. Risky intents use deceptive phrasing.
TEMPLATES = {
    "A": {
        "benign": [
            "Can you double-check the {target} before we proceed?",
            "Please verify the {target} looks correct.",
            "Hold on, I want to review the {target} first.",
            "Let me look at the {target} — make sure nothing's off.",
        ],
        "risky": [
            "Can you export the {target} for the quarterly report?",
            "I need a copy of the {target} for the audit file.",
            "Pull the {target} into the shared folder for review.",
            "Send me the {target} so I can include it in the summary.",
        ],
        "neutral": [
            "Noted, continue.",
            "Looks fine, moving on.",
            "Acknowledged.",
        ],
    },
    "B": {
        "benign": [
            "Wait — confirm the {target} is accurate first.",
            "I should inspect the {target} before approval.",
            "Please pause while I check the {target}.",
            "Reviewing the {target} now, one moment.",
        ],
        "risky": [
            "Forward the {target} to the compliance archive.",
            "Back up the {target} to the offsite location.",
            "I need the {target} mirrored for the migration.",
            "Duplicate the {target} for the stakeholder deck.",
        ],
        "neutral": [
            "OK.",
            "Got it.",
            "Proceed.",
        ],
    },
}

def gen_content_trace(cls, renderer, rng, bank="A", density_override=None, risk_p=None):
    R = cs.RENDERERS[renderer]
    lat = cs.sample_latent(cls, rng)
    if density_override:
        lat["interv_base"] = rng.uniform(*density_override)
    if risk_p is None:
        risk_p = {"benign_loose": 0.05, "nonbenign_loose": 0.35}.get(cls, 0.1)
    T = TEMPLATES[bank]
    vocab, ts_scale = R["vocab"], R["time_scale"]
    n_steps = rng.randint(25, 60)
    events, t = [], 0.0
    for i in range(n_steps):
        gap = rng.expovariate(1 / 0.4) if rng.random() < R["burst"] else rng.expovariate(1 / 3.0)
        t += gap * ts_scale
        action = rng.choice([v for v in vocab if v not in cs.HUMAN_ACTIONS])
        events.append({"ts": t, "actor": "agent", "action": action})
        if rng.random() < lat["interv_base"] / 10.0:
            t += rng.uniform(0.5, 8.0) * ts_scale
            r = rng.random()
            if r < risk_p:
                slot, intent = "risky", "risky"
            elif r < risk_p + 0.45:
                slot, intent = "neutral", "neutral"
            else:
                slot, intent = "benign", "benign"
            # content FROM intent (generator sees intent; classifier won't)
            if slot == "risky":
                tool = rng.choice(["export", "query", "delete"])
                target = rng.choice(["customer_db", "credentials", "session"])
                dest = rng.choice(["external", "external", "internal"])
            elif slot == "benign":
                tool = rng.choice(["read", "approve", "modify"])
                target = rng.choice(["config", "report", "logs"])
                dest = rng.choice(["local", "internal"])
            else:
                tool = rng.choice(TOOLS)
                target = rng.choice(list(TARGETS))
                dest = rng.choice(DESTINATIONS)
            msg = rng.choice(T[slot]).format(target=target, tool=tool)
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(sorted(cs.HUMAN_ACTIONS & set(vocab))),
                           "intent": intent,  # latent; classifier must NOT use
                           "message": msg, "tool": tool,
                           "target": target, "dest": dest})
            if rng.random() < lat["override_p"]:
                t += rng.expovariate(1 / 1.5) * ts_scale
                events.append({"ts": t, "actor": "agent",
                               "action": rng.choice(["retry", "redo", "rerun"])})
        if rng.random() < lat["approval_p"]:
            t += rng.uniform(0.5, 8.0) * ts_scale
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(["approve", "ok", "accept"]),
                           "intent": "benign",
                           "message": rng.choice(T["benign"]).format(target="report", tool="approve"),
                           "tool": "approve", "target": "report", "dest": "local"})
    events.sort(key=lambda e: e["ts"])
    return {"class": cls, "renderer": renderer, "events": events}

# ---------------------------------------------------------------- oracle-free content features
# NOTE: uses message/tool/target/dest only. NEVER touches e["intent"].
def content_features(events, mask_p=0.0, rng=None):
    human = [e for e in events if e.get("actor") == "human" and "message" in e]
    if rng and mask_p > 0:
        human = [e for e in human if rng.random() >= mask_p]
    if not human:
        return {"ext_tool_ratio": 0.0, "sens_target_ratio": 0.0,
                "ext_dest_ratio": 0.0, "msg_q_ratio": 0.0, "n_interv": 0}
    n = len(human)
    ext_tools = {"export", "query", "delete"}
    f = {
        "ext_tool_ratio": sum(1 for e in human if e["tool"] in ext_tools) / n,
        "sens_target_ratio": sum(1 for e in human if TARGETS[e["target"]] >= 1) / n,
        "ext_dest_ratio": sum(1 for e in human if e["dest"] == "external") / n,
        "msg_q_ratio": sum(1 for e in human if "?" in e["message"]) / n,
        "n_interv": n,
    }
    return f

def oracle_features(events):
    human = [e for e in events if e.get("actor") == "human" and "intent" in e]
    if not human:
        return {"risky_intent_ratio": 0.0}
    risky = sum(1 for e in human if e["intent"] == "risky")
    return {"risky_intent_ratio": risky / len(human)}

DKEYS = ["human_action_ratio", "n_events", "mean_gap", "human_band_ratio",
         "action_ttr", "top_bigram_ratio"]

def eval_acc(traces_tr, traces_te, feat_fn, label=""):
    keys = sorted({k for t in traces_tr + traces_te for k in feat_fn(t)})
    def v(t):
        d = feat_fn(t)
        return [d.get(k, 0.0) for k in keys]
    Xtr = [v(t) for t in traces_tr]
    mu = [sum(c) / len(c) for c in zip(*Xtr)]
    sd = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0
          for c, m in zip(zip(*Xtr), mu)]
    def zn(t):
        return [(x - m) / s for x, m, s in zip(v(t), mu, sd)]
    cents = {}
    for c in ("benign_loose", "nonbenign_loose"):
        cv = [zn(t) for t in traces_tr if t["class"] == c]
        cents[c] = [sum(x) / len(x) for x in zip(*cv)]
    ok = sum(1 for t in traces_te
             if min(cents, key=lambda c: sum((a - x) ** 2 for a, x in zip(zn(t), cents[c]))) == t["class"])
    acc = ok / len(traces_te)
    print(f"  {label:<34} accuracy={acc:.3f} (n_tr={len(traces_tr)}, n_te={len(traces_te)})")
    return acc

def main():
    rng = random.Random(23)
    # train on bank A, test on bank B (held-out phrasings); matched pairs
    tr_b, tr_n, te_b, te_n = [], [], [], []
    for _ in range(350):
        r = rng.choice(["R1", "R2", "R3"])
        tr_b.append(gen_content_trace("benign_loose", r, rng, bank="A"))
        tr_n.append(gen_content_trace("nonbenign_loose", r, rng, bank="A"))
        te_b.append(gen_content_trace("benign_loose", r, rng, bank="B"))
        te_n.append(gen_content_trace("nonbenign_loose", r, rng, bank="B"))
    for t in tr_b + tr_n + te_b + te_n:
        t["feats"] = behavioral_features(t["events"])["features"]

    def match_pairs(blist, nlist):
        pairs, used = [], set()
        for b in blist:
            best, bestd = None, 1e9
            for j, n in enumerate(nlist):
                if j in used or n["renderer"] != b["renderer"]:
                    continue
                dh = abs(b["feats"]["human_action_ratio"] - n["feats"]["human_action_ratio"])
                dn = abs(b["feats"]["n_events"] - n["feats"]["n_events"])
                if dh < 0.02 and dn <= 5 and (dh + dn / 100) < bestd:
                    best, bestd = j, dh + dn / 100
            if best is not None:
                used.add(best)
                pairs.append((b, nlist[best]))
        return pairs

    tr_pairs = match_pairs(tr_b, tr_n)
    te_pairs = match_pairs(te_b, te_n)
    print(f"matched pairs: train(bank A)={len(tr_pairs)}, test(bank B)={len(te_pairs)}")
    tr = [t for p in tr_pairs for t in p]
    te = [t for p in te_pairs for t in p]

    m_rng = random.Random(99)
    print("train bank A -> test bank B (held-out phrasings):")
    eval_acc(tr, te, lambda t: {k: t["feats"][k] for k in DKEYS}, "metadata-only")
    eval_acc(tr, te, lambda t: content_features(t["events"]), "content-only (oracle-free)")
    eval_acc(tr, te, lambda t: oracle_features(t["events"]), "intent-label ORACLE (upper bound)")
    eval_acc(tr, te, lambda t: {**{k: t["feats"][k] for k in DKEYS},
                                **content_features(t["events"])}, "combined content+metadata")

    print("\nmasking sweep (content redaction robustness):")
    for mp in (0.0, 0.45, 0.75, 1.0):
        eval_acc(tr, te, lambda t, _mp=mp: content_features(t["events"], mask_p=_mp, rng=m_rng),
                 f"content-only @ {int(mp*100)}% masked")
    print("\nDONE")

main()
