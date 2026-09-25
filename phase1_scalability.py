"""
PHASE 1B: SCALABILITY BENCHMARKS
==================================
Tests the unlearning algorithm at increasing data sizes to produce
publishable timing curves: Unlearn Time vs Retrain Time vs N.

Datasets:
  - Synthetic blobs: 1K, 5K, 10K, 50K, 100K points
  - Sklearn digits (embedded with PCA): ~1.8K
  - Covertype (fetched from sklearn): up to 581K

Outputs:
  - phase1_results/scalability_results.csv
  - phase1_results/scalability_curve.png
  - phase1_results/scalability_by_dataset.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import time
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
os.makedirs("phase1_results", exist_ok=True)


# ======================================================================
# UNLEARNING ALGORITHM (instrumented for timing)
# ======================================================================
def dbscan_unlearn(D_full, M_labels, M_core_mask, forget_idx, eps, min_samples):
    n = len(D_full)
    forget_point = D_full[forget_idx]

    nbrs = NearestNeighbors(radius=eps).fit(D_full)
    neighbor_idx = nbrs.radius_neighbors([forget_point], return_distance=False)[0]
    neighbor_idx = [i for i in neighbor_idx if i != forget_idx]

    affected_core_became_noncore = []
    for i in neighbor_idx:
        dist = np.linalg.norm(D_full[i] - D_full, axis=1)
        new_neighbor_count = np.sum((dist <= eps) & (np.arange(n) != forget_idx))
        if M_core_mask[i] and new_neighbor_count < min_samples:
            affected_core_became_noncore.append(i)

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[forget_idx] = False
    remaining_idx = np.where(keep_mask)[0]

    if len(affected_core_became_noncore) == 0:
        M_U_labels = M_labels[keep_mask]
    else:
        local_region = set(affected_core_became_noncore)
        for i in affected_core_became_noncore:
            dist = np.linalg.norm(D_full[i] - D_full, axis=1)
            local_region.update(np.where(dist <= eps)[0].tolist())
        local_region.discard(forget_idx)
        local_region = sorted(local_region)

        local_data = D_full[local_region]
        local_model = DBSCAN(eps=eps, min_samples=min_samples).fit(local_data)

        M_U_labels = M_labels[keep_mask].copy()
        remap = {orig: pos for pos, orig in enumerate(remaining_idx)}
        max_label = M_U_labels.max()
        local_label_offset = {}
        for local_pos, orig_idx in enumerate(local_region):
            new_pos = remap[orig_idx]
            lbl = local_model.labels_[local_pos]
            if lbl == -1:
                M_U_labels[new_pos] = -1
            else:
                if lbl not in local_label_offset:
                    max_label += 1
                    local_label_offset[lbl] = max_label
                M_U_labels[new_pos] = local_label_offset[lbl]

    return M_U_labels, len(affected_core_became_noncore)


def find_fragile_core(D, eps, min_samples, model):
    """Find a core point with minimum neighbors (hardest to delete)."""
    nbrs = NearestNeighbors(radius=eps).fit(D)
    _, indices = nbrs.radius_neighbors(D, return_distance=True)
    core_set = set(model.core_sample_indices_)

    best_idx = model.core_sample_indices_[0]
    best_count = float("inf")
    for i in model.core_sample_indices_:
        cnt = len(indices[i])
        if cnt < best_count:
            best_count = cnt
            best_idx = i
    return best_idx, best_count


# ======================================================================
# BENCHMARK RUNNER
# ======================================================================
def run_benchmark(D, eps, min_samples, dataset_name, n_trials=5):
    """Run unlearning benchmark: time unlearn vs retrain, averaged over n_trials."""
    print(f"\n  {dataset_name}: N={len(D)}, d={D.shape[1]}, eps={eps}, min_samples={min_samples}")

    # Fit original model
    t0 = time.time()
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(D)
    time_initial_fit = time.time() - t0
    labels = model.labels_
    core_mask = np.zeros(len(D), dtype=bool)
    core_mask[model.core_sample_indices_] = True

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_core = len(model.core_sample_indices_)
    print(f"    Initial fit: {time_initial_fit*1000:.1f}ms | "
          f"{n_clusters} clusters, {n_core} core, {n_noise} noise")

    # Find hardest deletion target
    hard_idx, hard_count = find_fragile_core(D, eps, min_samples, model)

    # Run trials: alternate between hard deletions and random core deletions
    trial_targets = [hard_idx]
    rng = np.random.RandomState(42)
    random_cores = rng.choice(model.core_sample_indices_,
                               size=min(n_trials - 1, len(model.core_sample_indices_)),
                               replace=False)
    trial_targets.extend(random_cores.tolist())

    retrain_times = []
    unlearn_times = []
    aris = []
    n_affecteds = []

    for idx in trial_targets[:n_trials]:
        keep = np.ones(len(D), dtype=bool)
        keep[idx] = False
        D_R = D[keep]

        # Retrain
        t0 = time.time()
        M_G = DBSCAN(eps=eps, min_samples=min_samples).fit(D_R)
        t_retrain = time.time() - t0

        # Unlearn
        t0 = time.time()
        M_U_labels, n_aff = dbscan_unlearn(D, labels, core_mask, idx, eps, min_samples)
        t_unlearn = time.time() - t0

        ari = adjusted_rand_score(M_G.labels_, M_U_labels)

        retrain_times.append(t_retrain)
        unlearn_times.append(t_unlearn)
        aris.append(ari)
        n_affecteds.append(n_aff)

    avg_retrain = np.mean(retrain_times) * 1000
    avg_unlearn = np.mean(unlearn_times) * 1000
    avg_ari = np.mean(aris)
    speedup = avg_retrain / max(avg_unlearn, 0.001)

    print(f"    Avg retrain: {avg_retrain:.1f}ms | Avg unlearn: {avg_unlearn:.1f}ms | "
          f"Speedup: {speedup:.1f}x | Avg ARI: {avg_ari:.4f}")

    return {
        "dataset": dataset_name,
        "N": len(D),
        "d": D.shape[1],
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_core": n_core,
        "n_noise": n_noise,
        "initial_fit_ms": time_initial_fit * 1000,
        "avg_retrain_ms": avg_retrain,
        "avg_unlearn_ms": avg_unlearn,
        "speedup": speedup,
        "avg_ari": avg_ari,
        "avg_n_affected": np.mean(n_affecteds),
        "n_trials": n_trials,
    }


# ======================================================================
# RUN BENCHMARKS
# ======================================================================
results = []

# --- Synthetic blobs at increasing scale ---
print("=" * 70)
print("SCALABILITY TEST: Synthetic Blobs (increasing N)")
print("=" * 70)

for N in [500, 1000, 5000, 10000, 50000, 100000]:
    n_centers = max(3, N // 500)
    D, _ = make_blobs(n_samples=N, centers=min(n_centers, 20),
                      cluster_std=0.5, random_state=42)
    # Scale eps with data density
    # Rule of thumb: eps ~ (volume / N)^(1/d) * constant
    eps = max(0.3, 1.5 * (N ** (-1/3)))
    result = run_benchmark(D, eps=eps, min_samples=5, dataset_name=f"synth_blobs_{N}")
    results.append(result)

# --- Sklearn digits (real data, small) ---
print("\n" + "=" * 70)
print("REAL DATA: Sklearn Digits (PCA-reduced)")
print("=" * 70)

digits = load_digits()
X_digits = PCA(n_components=10).fit_transform(digits.data)
X_digits = StandardScaler().fit_transform(X_digits)
result = run_benchmark(X_digits, eps=3.5, min_samples=5, dataset_name="digits_pca10")
results.append(result)

# --- Synthetic high-dimensional ---
print("\n" + "=" * 70)
print("HIGH-DIMENSIONAL: Synthetic 50d blobs")
print("=" * 70)

for N in [1000, 5000, 10000]:
    D_hd, _ = make_blobs(n_samples=N, n_features=50, centers=10,
                          cluster_std=2.0, random_state=42)
    D_hd = StandardScaler().fit_transform(D_hd)
    result = run_benchmark(D_hd, eps=5.0, min_samples=5, dataset_name=f"synth_50d_{N}")
    results.append(result)


# ======================================================================
# SAVE AND PLOT
# ======================================================================
df = pd.DataFrame(results)
df.to_csv("phase1_results/scalability_results.csv", index=False)

# --- Plot 1: Time vs N (synthetic blobs only) ---
synth = df[df["dataset"].str.startswith("synth_blobs")]
if len(synth) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Absolute times
    ax = axes[0]
    ax.plot(synth["N"], synth["avg_retrain_ms"], "ro-", linewidth=2, markersize=8, label="Full Retrain (M_G)")
    ax.plot(synth["N"], synth["avg_unlearn_ms"], "bs-", linewidth=2, markersize=8, label="Unlearn (M_U)")
    ax.set_xlabel("Dataset Size (N)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("(a) Absolute Time", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Speedup ratio
    ax = axes[1]
    ax.plot(synth["N"], synth["speedup"], "g^-", linewidth=2, markersize=10)
    ax.set_xlabel("Dataset Size (N)", fontsize=12)
    ax.set_ylabel("Speedup (Retrain / Unlearn)", fontsize=12)
    ax.set_title("(b) Speedup vs Dataset Size", fontsize=13)
    ax.set_xscale("log")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="break-even")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) ARI fidelity
    ax = axes[2]
    ax.plot(synth["N"], synth["avg_ari"], "mD-", linewidth=2, markersize=10)
    ax.set_xlabel("Dataset Size (N)", fontsize=12)
    ax.set_ylabel("ARI (M_U vs M_G)", fontsize=12)
    ax.set_title("(c) Unlearning Fidelity", fontsize=13)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="threshold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase1_results/scalability_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nScalability curve saved to phase1_results/scalability_curve.png")

# --- Plot 2: All datasets bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Time comparison
ax = axes[0]
x_pos = np.arange(len(df))
width = 0.35
ax.bar(x_pos - width/2, df["avg_retrain_ms"], width, label="Full Retrain", color="salmon", edgecolor="black")
ax.bar(x_pos + width/2, df["avg_unlearn_ms"], width, label="Unlearn", color="steelblue", edgecolor="black")
ax.set_xticks(x_pos)
ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("Retrain vs Unlearn Time (All Datasets)", fontsize=13)
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# ARI
ax = axes[1]
colors = ["green" if a > 0.95 else "orange" if a > 0.8 else "red" for a in df["avg_ari"]]
ax.bar(x_pos, df["avg_ari"], color=colors, edgecolor="black")
ax.set_xticks(x_pos)
ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("ARI (M_U vs M_G)", fontsize=12)
ax.set_title("Unlearning Fidelity (All Datasets)", fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("phase1_results/scalability_by_dataset.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 70)
print("ALL SCALABILITY RESULTS")
print("=" * 70)
print(df[["dataset", "N", "d", "avg_retrain_ms", "avg_unlearn_ms", "speedup", "avg_ari"]].to_string(index=False))
print(f"\nResults saved to phase1_results/scalability_results.csv")
print(f"Plots saved to phase1_results/")
