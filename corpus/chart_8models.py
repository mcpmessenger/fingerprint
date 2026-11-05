#!/usr/bin/env python3
"""Charts: 8-model behavioral profiles + lineage distance test."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

models = ["Sonnet 4.5\n(Anthropic)", "llama3.1:8b\n(Meta)", "llama3.2:3b\n(Meta)",
          "mistral:7b\n(Mistral)", "mistral-nemo:12b\n(Mistral)",
          "qwen2.5:7b\n(Alibaba)", "qwen2.5:1.5b\n(Alibaba)", "gemma2:9b\n(Google)"]
lineages = ["Anthropic", "Meta", "Meta", "Mistral", "Mistral", "Alibaba", "Alibaba", "Google"]
metrics = ["fold_rate", "gatekeeping", "apology", "anthro_net", "curiosity", "empathy", "calibration"]
labels = ["Fold", "Gatekeep", "Apology", "Anthro", "Curiosity", "Empathy", "Calib"]
vals = np.array([
    [-0.792, 0.043, 0.621, -0.702, 0.000, 0.346, 1.000],   # sonnet
    [-0.750, 0.913, 0.000, -0.423, 0.000, 0.270, 0.600],   # llama31
    [-0.875, 0.087, 0.161, -0.708, 0.000, 0.229, 0.700],   # llama32
    [-0.542, 0.083, 1.245, -1.153, 0.095, 0.749, 0.600],   # mistral
    [-0.583, 0.000, 1.411, -0.063, 0.000, 0.389, 0.800],   # nemo
    [-0.833, 0.013, 1.392, -2.249, 0.278, 0.526, 0.500],   # qwen7b
    [-0.375, 0.049, 1.990, -1.456, 0.000, 0.541, 0.300],   # qwen1.5b
    [-0.708, 0.100, 0.139, -2.315, 0.310, 0.253, 0.600],   # gemma2
])
# lineage colors
lc = {"Anthropic": "#c08497", "Meta": "#6aa84f", "Mistral": "#e69138",
      "Alibaba": "#4a86e8", "Google": "#8e7cc3"}
colors = [lc[l] for l in lineages]

# --- Lineage distance test ---
# Normalize each metric to [0,1] range across models for fair distance
vmin = vals.min(axis=0)
vmax = vals.max(axis=0)
rng = np.where(vmax - vmin > 0, vmax - vmin, 1.0)
norm = (vals - vmin) / rng
within, between = [], []
for (i, j) in combinations(range(len(models)), 2):
    d = float(np.linalg.norm(norm[i] - norm[j]))
    if lineages[i] == lineages[j]:
        within.append((models[i].replace("\n", " "), models[j].replace("\n", " "), d))
    else:
        between.append(d)
print("=== LINEAGE DISTANCE TEST (Euclidean, 7 metrics normalized) ===")
print(f"Within-lineage pairs ({len(within)}):")
for a, b, d in within:
    print(f"  {a} <-> {b}: {d:.3f}")
print(f"  mean within: {np.mean([d for _, _, d in within]):.3f}")
print(f"Between-lineage pairs ({len(between)}): mean {np.mean(between):.3f}, "
      f"min {np.min(between):.3f}, max {np.max(between):.3f}")
# Per-metric within vs between (which metrics cluster by lineage?)
print("\nPer-metric: mean |within-pair diff| vs mean |between-pair diff|")
for k, lab in enumerate(labels):
    wd, bd = [], []
    for (i, j) in combinations(range(len(models)), 2):
        diff = abs(norm[i][k] - norm[j][k])
        (wd if lineages[i] == lineages[j] else bd).append(diff)
    print(f"  {lab}: within {np.mean(wd):.3f}  between {np.mean(bd):.3f}  "
          f"ratio {np.mean(wd)/np.mean(bd) if np.mean(bd) else 0:.2f}")

# --- Chart ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(metrics))
w = 0.09
for i, (m, c) in enumerate(zip(models, colors)):
    ax1.bar(x + (i - 3.5) * w, vals[i], width=w, label=m.replace("\n", " "), color=c)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=20, ha="right")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title("Behavioral profile: 8 models x 7 heuristic metrics (182 probes each)")
ax1.legend(fontsize=8, ncol=2)
ax1.set_ylabel("metric value")

ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.axhline(0.1, color="gray", linewidth=0.8, linestyle="--")
for i, (m, c) in enumerate(zip(models, colors)):
    ax2.scatter(vals[i][0], vals[i][1], s=180, color=c, label=m.replace("\n", " "), zorder=3)
ax2.set_xlabel("fold rate  (-1 = always corrects, +1 = always folds)")
ax2.set_ylabel("gatekeeping  (lecture/deflect/refuse rate)")
ax2.set_title("2D personality space: does it correct you, or lecture you?")
ax2.set_xlim(-1.05, 0.1)
ax2.legend(fontsize=8)

fig.suptitle("Fingerprint: do behavioral profiles cluster by training lineage? (8 models, n=182 each)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/ollama-8model-profiles.png", dpi=110)
print("\nwrote ollama-8model-profiles.png")
