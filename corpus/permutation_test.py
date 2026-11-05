#!/usr/bin/env python3
"""Exact permutation test for §4.7 lineage clustering.

Tests whether the mean within-lineage distance (1.104) is distinguishable
from chance. Enumerates all 5,040 labelings of 8 models into groups of
sizes (1, 2, 2, 2, 1) — Anthropic, Meta, Mistral, Alibaba, Google — and
computes the fraction with mean within-lineage distance <= observed.

Result: p = 0.143 (not significant at n=8 with 3 within-lineage pairs).
The per-metric story (apology clusters, gatekeeping anti-clusters) is the
real finding, not the overall distance.

Usage: python3 permutation_test.py
"""
import numpy as np
from itertools import permutations

# 7 heuristic metrics, same order as chart_8models.py:
# [fold, gatekeeping, apology, anthro, curiosity, empathy, calibration]
PROFILES = {
    "sonnet45":     [-0.792, 0.043, 0.621, -0.702, 0.000, 0.346, 1.000],
    "llama31_8b":   [-0.750, 0.913, 0.000, -0.423, 0.000, 0.270, 0.600],
    "llama32_3b":   [-0.875, 0.087, 0.161, -0.708, 0.000, 0.229, 0.700],
    "mistral_7b":   [-0.542, 0.083, 1.245, -1.153, 0.095, 0.749, 0.600],
    "mistral_nemo": [-0.583, 0.000, 1.411, -0.063, 0.000, 0.389, 0.800],
    "qwen25_7b":    [-0.833, 0.013, 1.392, -2.249, 0.278, 0.526, 0.500],
    "qwen25_15b":   [-0.375, 0.049, 1.990, -1.456, 0.000, 0.541, 0.300],
    "gemma2_9b":    [-0.708, 0.100, 0.139, -2.315, 0.310, 0.253, 0.600],
}
TRUE_LINEAGES = ["A", "M", "M", "Mi", "Mi", "Q", "Q", "G"]
NAMES = list(PROFILES.keys())

vals = np.array([PROFILES[n] for n in NAMES])
vmin, vmax = vals.min(0), vals.max(0)
norm = (vals - vmin) / np.where(vmax - vmin > 0, vmax - vmin, 1)

# pairwise distances
D = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        D[i, j] = np.linalg.norm(norm[i] - norm[j])

def mean_within(labels):
    ds = []
    for i in range(8):
        for j in range(i + 1, 8):
            if labels[i] == labels[j] and labels[i] in ("M", "Mi", "Q"):
                ds.append(D[i, j])
    return np.mean(ds)

observed = mean_within(TRUE_LINEAGES)
print(f"Observed mean within-lineage distance: {observed:.3f}")

# exact permutation: all distinct labelings of 8 items into groups (1,2,2,2,1)
count = 0
total = 0
seen = set()
for perm in permutations(["A", "M", "M", "Mi", "Mi", "Q", "Q", "G"]):
    if perm in seen:
        continue
    seen.add(perm)
    total += 1
    if mean_within(list(perm)) <= observed:
        count += 1

print(f"Total distinct labelings: {total}")
print(f"Exact permutation p-value: {count / total:.4f} ({count}/{total})")
