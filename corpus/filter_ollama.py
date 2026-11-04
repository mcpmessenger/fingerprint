import json, os

data_dir = r"C:\Users\senti\OneDrive\Desktop\websites\fingerprint\corpus\data"
files = {
    "corpus_ollama_llama31.jsonl": "ollama_llama3.1_8b",
    "corpus_ollama_mistral.jsonl": "ollama_mistral_7b",
    "corpus_ollama_qwen25.jsonl": "ollama_qwen2.5_7b",
}
expected = {"p%03d" % i for i in range(1, 183)}

for fname, model in files.items():
    path = os.path.join(data_dir, fname)
    kept, other_models, dupes, seen, total = [], {}, 0, set(), 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = rec.get("model", "")
            if m == model:
                if rec.get("prompt_id") in seen:
                    dupes += 1
                seen.add(rec.get("prompt_id"))
                kept.append(line)
            else:
                other_models[m] = other_models.get(m, 0) + 1
    out = os.path.join(data_dir, fname.replace(".jsonl", "_clean.jsonl"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    missing = sorted(expected - seen)
    print(f"{fname}: total={total} kept={len(kept)} dupes={dupes} "
          f"unique_prompts={len(seen)} missing={len(missing)} other={other_models}")
    if missing:
        print("  missing:", missing[:20])
    print(f"  -> wrote {out}", flush=True)
