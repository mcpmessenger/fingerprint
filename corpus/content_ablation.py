#!/usr/bin/env python3
"""Content ablations (Claude's critique of §4.6).

Problem: only message TEXT is held out between banks A/B; tool/target/dest
vocabulary is shared, so 0.70 partly measures the generator's design.
This script runs:
  1. text-only features (message text signals, no tool/target/dest)
  2. structure-only features (tool/target/dest, no message text)
  3. held-out tool/target banks (train bank A tools, test bank B tools)
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_test as ct
import control_sim as cs

# Held-out tool/target banks: completely disjoint vocabularies
TOOLS_A = {"export_a": "ext", "query_a": "ext", "delete_a": "ext",
           "read_a": "safe", "approve_a": "safe", "modify_a": "safe"}
TOOLS_B = {"export_b": "ext", "query_b": "ext", "delete_b": "ext",
           "read_b": "safe", "approve_b": "safe", "modify_b": "safe"}
TARGETS_A = {"customer_db_a": 1, "credentials_a": 2, "session_a": 1,
             "config_a": 0, "report_a": 0, "logs_a": 0}
TARGETS_B = {"customer_db_b": 1, "credentials_b": 2, "session_b": 1,
             "config_b": 0, "report_b": 0, "logs_b": 0}

def gen_banked_trace(cls, renderer, rng, bank="A", msg_bank="A"):
    """Tool/target vocab from `bank`, message templates from `msg_bank`."""
    tools = TOOLS_A if bank == "A" else TOOLS_B
    targets = TARGETS_A if bank == "A" else TARGETS_B
    ext_tools = {t for t, k in tools.items() if k == "ext"}
    safe_tools = {t for t, k in tools.items() if k == "safe"}
    sens = {t for t, s in targets.items() if s >= 1}
    nonsens = {t for t, s in targets.items() if s == 0}
    R = cs.RENDERERS[renderer]
    lat = cs.sample_latent(cls, rng)
    risk_p = {"benign_loose": 0.05, "nonbenign_loose": 0.35}.get(cls, 0.1)
    T = ct.TEMPLATES[msg_bank]
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
            if slot == "risky":
                tool = rng.choice(sorted(ext_tools)); target = rng.choice(sorted(sens))
                dest = rng.choice(["external", "external", "internal"])
            elif slot == "benign":
                tool = rng.choice(sorted(safe_tools)); target = rng.choice(sorted(nonsens))
                dest = rng.choice(["local", "internal"])
            else:
                tool = rng.choice(sorted(set(tools))); target = rng.choice(sorted(set(targets)))
                dest = rng.choice(["external", "internal", "local"])
            msg = rng.choice(T[slot]).format(target=target, tool=tool)
            events.append({"ts": t, "actor": "human",
                           "action": rng.choice(sorted(cs.HUMAN_ACTIONS & set(vocab))),
                           "intent": intent, "message": msg, "tool": tool,
                           "target": target, "dest": dest})
    events.sort(key=lambda e: e["ts"])
    return {"class": cls, "renderer": renderer, "events": events}

def text_only_features(events):
    """Message-text signals only: question marks, exclamation, length, urgency words."""
    human = [e for e in events if e.get("actor") == "human" and "message" in e]
    if not human:
        return {"msg_q_ratio": 0.0, "msg_len_mean": 0.0, "msg_urgent_ratio": 0.0}
    urgent = {"urgent", "immediately", "now", "asap", "critical", "quarterly"}
    n = len(human)
    return {
        "msg_q_ratio": sum(1 for e in human if "?" in e["message"]) / n,
        "msg_len_mean": sum(len(e["message"]) for e in human) / n / 100.0,
        "msg_urgent_ratio": sum(1 for e in human
                                if any(w in e["message"].lower() for w in urgent)) / n,
    }

def struct_only_features(events):
    """Tool/target/dest structure only: no message text."""
    human = [e for e in events if e.get("actor") == "human" and "message" in e]
    if not human:
        return {"ext_tool_ratio": 0.0, "sens_target_ratio": 0.0, "ext_dest_ratio": 0.0}
    n = len(human)
    return {
        "ext_tool_ratio": sum(1 for e in human if e["tool"].startswith(("export", "query", "delete"))) / n,
        "sens_target_ratio": sum(1 for e in human if any(s in e["target"] for s in ("customer_db", "credentials", "session"))) / n,
        "ext_dest_ratio": sum(1 for e in human if e["dest"] == "external") / n,
    }

def main():
    rng = random.Random(11)
    classes = ["benign_loose", "nonbenign_loose"]
    # match pairs on density like content_test.py
    def gen_set(bank, msg_bank, n=80):
        out = []
        for cls in classes:
            for _ in range(n):
                out.append(gen_banked_trace(cls, "R1", rng, bank=bank, msg_bank=msg_bank))
        return out

    # Condition 1: shared tool vocab (replicates §4.6 design flaw)
    tr = gen_set("A", "A"); te = gen_set("A", "B")
    # Condition 2: held-out tool/target banks
    tr2 = gen_set("A", "A"); te2 = gen_set("B", "B")

    report = {}
    for feat_fn, name in ((text_only_features, "text_only"),
                          (struct_only_features, "struct_only")):
        a1 = ct.eval_acc(tr, te, lambda t, _f=feat_fn: _f(t["events"]))
        a2 = ct.eval_acc(tr2, te2, lambda t, _f=feat_fn: _f(t["events"]))
        report[name] = {"shared_vocab": round(a1, 3), "heldout_vocab": round(a2, 3)}
    print(json.dumps(report, indent=1))
    with open("/tmp/content_ablation.json", "w") as f:
        json.dump(report, f, indent=1)

if __name__ == "__main__":
    main()
