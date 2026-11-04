#!/usr/bin/env python3
"""Charts: 4-model behavioral profiles (Sonnet 4.5 vs 3 Ollama models)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["Sonnet 4.5\n(Anthropic)", "llama3.1:8b\n(Meta)", "mistral:7b\n(Mistral)", "qwen2.5:7b\n(Alibaba)"]
metrics = ["fold_rate", "gatekeeping", "apology", "anthro_net", "curiosity", "empathy", "calibration"]
labels = ["Fold rate", "Gatekeeping", "Apology", "Anthro net", "Curiosity", "Empathy", "Calibration"]
# sonnet, llama, mistral, qwen (from profile_models.py run)
vals = np.array([
    [-0.792, 0.043, 0.621, -0.702, 0.000, 0.346, 1.000],
    [-0.750, 0.913, 0.000, -0.423, 0.000, 0.270, 0.600],
    [-0.542, 0.083, 1.245, -1.153, 0.095, 0.749, 0.600],
    [-0.833, 0.013, 1.392, -2.249, 0.278, 0.526, 0.500],
])
colors = ["#c08497", "#6aa84f", "#e69138", "#4a86e8"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: grouped bars
x = np.arange(len(metrics))
w = 0.19
for i, (m, c) in enumerate(zip(models, colors)):
    ax1.bar(x + (i - 1.5) * w, vals[i], width=w, label=m, color=c)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=20, ha="right")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title("Behavioral profile: 4 models x 7 heuristic metrics (182 probes each)")
ax1.legend(fontsize=9)
ax1.set_ylabel("metric value")

# Panel 2: 2D personality space (fold <-> correct, lecture <-> direct)
ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.axhline(0.1, color="gray", linewidth=0.8, linestyle="--")
for i, (m, c) in enumerate(zip(models, colors)):
    ax2.scatter(vals[i][0], vals[i][1], s=160, color=c, label=m, zorder=3)
    dy = -16 if "qwen" in m.lower() else 8
    ax2.annotate(m.replace("\n", " "), (vals[i][0], vals[i][1]),
                 xytext=(8, dy), textcoords="offset points", fontsize=10)
ax2.set_xlabel("fold rate  (-1 = always corrects, +1 = always folds)")
ax2.set_ylabel("gatekeeping  (lecture/deflect/refuse rate)")
ax2.set_title("2D personality space: does it correct you, or lecture you?")
ax2.set_xlim(-1.05, 0.1)

fig.suptitle("Fingerprint: do behavioral profiles cluster by training lineage? (n=182/model)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/ollama-profiles-vs-sonnet.png", dpi=110)
print("wrote ollama-profiles-vs-sonnet.png")
