# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
PHASE 1A: STRESS-TEST — Deliberately Hard Deletions
=====================================================
Your current robustness test never triggered the hard case (n_affected=0 every time).
This script:
  1. Identifies "fragile core" points — core points with EXACTLY min_samples neighbors
     (removing 1 neighbor guarantees core→non-core transition)
  2. Deletes THOSE points and measures unlearning quality
  3. Tests on chain-shaped data (make_moons, make_circles) where cascades actually happen
  4. Compares M_U vs M_G for every hard deletion
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import time
import os

np.random.seed(42)
os.makedirs("phase1_results", exist_ok=True)

# ======================================================================
# CORE UNLEARNING ALGORITHM (same as yours, with minor instrumentation)
# ======================================================================
def dbscan_unlearn(D_full, M_labels, M_core_mask, forget_idx, eps, min_samples):
    """
    Decremental DBSCAN unlearning — local update only.
    Returns: (M_U_labels, n_affected, affected_indices)
    """
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

    return M_U_labels, len(affected_core_became_noncore), affected_core_became_noncore


# ======================================================================
# HELPER: find fragile core points (exactly min_samples neighbors)
# ======================================================================
def find_fragile_cores(D, eps, min_samples):
    """Find core points with exactly min_samples neighbors within eps.
    Deleting ANY of their neighbors will make them lose core status."""
    nbrs = NearestNeighbors(radius=eps).fit(D)
    distances, indices = nbrs.radius_neighbors(D, return_distance=True)

    model = DBSCAN(eps=eps, min_samples=min_samples).fit(D)
    core_set = set(model.core_sample_indices_)

    fragile = []
    for i in range(len(D)):
        if i in core_set:
            n_neighbors = len(indices[i])  # includes self
            if n_neighbors == min_samples:
                fragile.append(i)
    return fragile, model


def find_bridge_points(D, labels, eps, min_samples, max_check=50):
    """Find points that, if deleted, would split a cluster into two.
    These are the hardest possible deletions for DBSCAN unlearning.
    Uses scipy sparse graph for fast connectivity checks.
    max_check: limit how many core points to check (prioritize smallest-degree cores)."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components

    model = DBSCAN(eps=eps, min_samples=min_samples).fit(D)
    core_indices = model.core_sample_indices_
    core_set = set(core_indices)

    if len(core_indices) < 3:
        return []

    # Build sparse adjacency matrix among core points (once)
    nbrs = NearestNeighbors(radius=eps).fit(D[core_indices])
    core_adj = nbrs.radius_neighbors_graph(D[core_indices], mode='connectivity')

    # Map: core position in core_indices array -> original index
    core_to_pos = {idx: pos for pos, idx in enumerate(core_indices)}

    # Group core positions by cluster
    cluster_cores = {}
    for pos, idx in enumerate(core_indices):
        cid = labels[idx]
        if cid == -1:
            continue
        cluster_cores.setdefault(cid, []).append(pos)

    # Only check cores with smallest degree (most likely to be bridges)
    degrees = np.array(core_adj.sum(axis=1)).flatten()
    # Sort by degree ascending, check up to max_check
    sorted_positions = np.argsort(degrees)

    bridges = []
    checked = 0
    for pos in sorted_positions:
        if checked >= max_check:
            break
        idx = core_indices[pos]
        cid = labels[idx]
        if cid == -1:
            continue

        cluster_pos_list = cluster_cores.get(cid, [])
        if len(cluster_pos_list) < 3:
            continue

        # Check if removing this core disconnects its cluster
        remaining = [p for p in cluster_pos_list if p != pos]
        sub_adj = core_adj[remaining][:, remaining]
        n_comp, _ = connected_components(sub_adj, directed=False)
        if n_comp > 1:
            bridges.append(idx)
        checked += 1

    return bridges



def plot_stress_test(D, labels, forget_idx, title, fname, affected=None):
    """Plot with the forgotten point highlighted."""
    plt.figure(figsize=(6, 5))
    unique_labels = set(labels) if forget_idx is None else set(np.delete(labels, forget_idx) if forget_idx < len(labels) else labels)

    for lbl in set(labels):
        mask = labels == lbl
        color = "lightgray" if lbl == -1 else None
        marker = "x" if lbl == -1 else "o"
        plt.scatter(D[mask, 0], D[mask, 1], c=color, marker=marker, s=20, alpha=0.6,
                    label=f"C{lbl}" if lbl != -1 else "noise")

    if forget_idx is not None and forget_idx < len(D):
        plt.scatter(D[forget_idx, 0], D[forget_idx, 1], c="red", marker="*",
                    s=300, zorder=10, edgecolors="black", linewidths=1.5, label="DELETED")

    if affected:
        for ai in affected:
            if ai < len(D):
                plt.scatter(D[ai, 0], D[ai, 1], c="orange", marker="D",
                            s=80, zorder=9, edgecolors="black", linewidths=1)

    plt.title(title, fontsize=10)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(f"phase1_results/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# DATASET 1: Original Gaussian blobs (find the HARD cases this time)
# ======================================================================
print("=" * 70)
print("DATASET 1: Gaussian Blobs — Finding Fragile Core Points")
print("=" * 70)

blob1 = np.random.normal(loc=[0, 0], scale=0.3, size=(40, 2))
blob2 = np.random.normal(loc=[5, 5], scale=0.3, size=(40, 2))
blob3 = np.random.normal(loc=[0, 5], scale=0.3, size=(40, 2))
noise = np.random.uniform(low=-3, high=8, size=(10, 2))
D_blobs = np.vstack([blob1, blob2, blob3, noise])

EPS_B, MIN_B = 0.6, 5
fragile_blobs, model_blobs = find_fragile_cores(D_blobs, EPS_B, MIN_B)
labels_blobs = model_blobs.labels_
core_mask_blobs = np.zeros(len(D_blobs), dtype=bool)
core_mask_blobs[model_blobs.core_sample_indices_] = True

print(f"Total points: {len(D_blobs)}")
print(f"Core points: {len(model_blobs.core_sample_indices_)}")
print(f"Fragile core points (exactly {MIN_B} neighbors): {len(fragile_blobs)}")
print(f"Fragile indices: {fragile_blobs}")

# Also find bridge points
bridges_blobs = find_bridge_points(D_blobs, labels_blobs, EPS_B, MIN_B)
print(f"Bridge points (deletion splits cluster): {len(bridges_blobs)}")

# Test ALL fragile + bridge points
hard_targets_blobs = sorted(set(fragile_blobs + bridges_blobs))
if len(hard_targets_blobs) == 0:
    # If no fragile points, target core points with fewest neighbors
    nbrs = NearestNeighbors(radius=EPS_B).fit(D_blobs)
    _, indices = nbrs.radius_neighbors(D_blobs, return_distance=True)
    core_neighbor_counts = [(i, len(indices[i])) for i in model_blobs.core_sample_indices_]
    core_neighbor_counts.sort(key=lambda x: x[1])
    hard_targets_blobs = [x[0] for x in core_neighbor_counts[:10]]
    print(f"No fragile/bridge points found. Using 10 core points with fewest neighbors.")

results_blobs = []
for idx in hard_targets_blobs:
    keep = np.ones(len(D_blobs), dtype=bool)
    keep[idx] = False
    D_R = D_blobs[keep]

    # Gold model
    t0 = time.time()
    M_G = DBSCAN(eps=EPS_B, min_samples=MIN_B).fit(D_R)
    t_retrain = time.time() - t0

    # Unlearned model
    t0 = time.time()
    M_U_labels, n_aff, aff_list = dbscan_unlearn(D_blobs, labels_blobs, core_mask_blobs, idx, EPS_B, MIN_B)
    t_unlearn = time.time() - t0

    ari = adjusted_rand_score(M_G.labels_, M_U_labels)
    nmi = normalized_mutual_info_score(M_G.labels_, M_U_labels)

    was_core = "CORE" if core_mask_blobs[idx] else "border/noise"
    status = "[OK]" if ari > 0.95 else "[WARN] DEGRADED" if ari > 0.8 else "[FAIL] FAILED"

    print(f"  [{status}] idx={idx:3d} ({was_core}) | affected={n_aff} | "
          f"ARI={ari:.4f} | NMI={nmi:.4f} | speedup={t_retrain/max(t_unlearn,1e-9):.1f}x")

    results_blobs.append({
        "dataset": "gaussian_blobs", "forget_idx": idx, "point_type": was_core,
        "n_affected": n_aff, "ari": ari, "nmi": nmi,
        "time_retrain_ms": t_retrain * 1000, "time_unlearn_ms": t_unlearn * 1000,
        "status": status
    })

plot_stress_test(D_blobs, labels_blobs, hard_targets_blobs[0] if hard_targets_blobs else None,
                 "Blobs: Fragile Core Deletion", "blobs_fragile_deletion",
                 affected=hard_targets_blobs[:5])


# ======================================================================
# DATASET 2: make_moons — chain-shaped clusters
# ======================================================================
print("\n" + "=" * 70)
print("DATASET 2: make_moons — Chain-Shaped Clusters (Cascade Test)")
print("=" * 70)

X_moons, _ = make_moons(n_samples=500, noise=0.06, random_state=42)
EPS_M, MIN_M = 0.15, 5

model_moons = DBSCAN(eps=EPS_M, min_samples=MIN_M).fit(X_moons)
labels_moons = model_moons.labels_
core_mask_moons = np.zeros(len(X_moons), dtype=bool)
core_mask_moons[model_moons.core_sample_indices_] = True

n_clusters_moons = len(set(labels_moons)) - (1 if -1 in labels_moons else 0)
print(f"Moons: {len(X_moons)} points, {n_clusters_moons} clusters, "
      f"{list(labels_moons).count(-1)} noise")

fragile_moons, _ = find_fragile_cores(X_moons, EPS_M, MIN_M)
bridges_moons = find_bridge_points(X_moons, labels_moons, EPS_M, MIN_M)
print(f"Fragile core points: {len(fragile_moons)}")
print(f"Bridge points: {len(bridges_moons)}")

# Also target the "neck" of each crescent -- where the two moons are closest
# These are the most structurally critical points
dists_between = np.linalg.norm(X_moons[:250, np.newaxis] - X_moons[250:], axis=2)
i_close, j_close = np.unravel_index(np.argmin(dists_between), dists_between.shape)
neck_points = [i_close, j_close + 250]
print(f"Neck points (closest between moons): {neck_points}")

hard_targets_moons = sorted(set(fragile_moons + bridges_moons + neck_points))
if len(hard_targets_moons) > 30:
    hard_targets_moons = hard_targets_moons[:30]  # cap at 30

results_moons = []
for idx in hard_targets_moons:
    keep = np.ones(len(X_moons), dtype=bool)
    keep[idx] = False
    D_R = X_moons[keep]

    t0 = time.time()
    M_G = DBSCAN(eps=EPS_M, min_samples=MIN_M).fit(D_R)
    t_retrain = time.time() - t0

    t0 = time.time()
    M_U_labels, n_aff, aff_list = dbscan_unlearn(X_moons, labels_moons, core_mask_moons, idx, EPS_M, MIN_M)
    t_unlearn = time.time() - t0

    ari = adjusted_rand_score(M_G.labels_, M_U_labels)
    nmi = normalized_mutual_info_score(M_G.labels_, M_U_labels)

    was_core = "CORE" if core_mask_moons[idx] else "border/noise"
    status = "[OK]" if ari > 0.95 else "[WARN] DEGRADED" if ari > 0.8 else "[FAIL] FAILED"

    print(f"  [{status}] idx={idx:3d} ({was_core}) | affected={n_aff} | "
          f"ARI={ari:.4f} | NMI={nmi:.4f}")

    results_moons.append({
        "dataset": "make_moons", "forget_idx": idx, "point_type": was_core,
        "n_affected": n_aff, "ari": ari, "nmi": nmi,
        "time_retrain_ms": t_retrain * 1000, "time_unlearn_ms": t_unlearn * 1000,
        "status": status
    })

plot_stress_test(X_moons, labels_moons, bridges_moons[0] if bridges_moons else fragile_moons[0] if fragile_moons else 0,
                 "Moons: Bridge Point Deletion", "moons_bridge_deletion")


# ======================================================================
# DATASET 3: make_circles — concentric ring clusters
# ======================================================================
print("\n" + "=" * 70)
print("DATASET 3: make_circles - Concentric Ring Clusters")
print("=" * 70)

X_circles, _ = make_circles(n_samples=600, noise=0.05, factor=0.5, random_state=42)
EPS_C, MIN_C = 0.12, 5

model_circles = DBSCAN(eps=EPS_C, min_samples=MIN_C).fit(X_circles)
labels_circles = model_circles.labels_
core_mask_circles = np.zeros(len(X_circles), dtype=bool)
core_mask_circles[model_circles.core_sample_indices_] = True

n_clusters_circles = len(set(labels_circles)) - (1 if -1 in labels_circles else 0)
print(f"Circles: {len(X_circles)} points, {n_clusters_circles} clusters, "
      f"{list(labels_circles).count(-1)} noise")

fragile_circles, _ = find_fragile_cores(X_circles, EPS_C, MIN_C)
bridges_circles = find_bridge_points(X_circles, labels_circles, EPS_C, MIN_C)
print(f"Fragile core points: {len(fragile_circles)}")
print(f"Bridge points: {len(bridges_circles)}")

hard_targets_circles = sorted(set(fragile_circles + bridges_circles))
if len(hard_targets_circles) > 30:
    hard_targets_circles = hard_targets_circles[:30]
if len(hard_targets_circles) == 0:
    # Fallback: pick core points with lowest neighbor counts
    nbrs = NearestNeighbors(radius=EPS_C).fit(X_circles)
    _, indices = nbrs.radius_neighbors(X_circles, return_distance=True)
    core_neighbor_counts = [(i, len(indices[i])) for i in model_circles.core_sample_indices_]
    core_neighbor_counts.sort(key=lambda x: x[1])
    hard_targets_circles = [x[0] for x in core_neighbor_counts[:15]]

results_circles = []
for idx in hard_targets_circles:
    keep = np.ones(len(X_circles), dtype=bool)
    keep[idx] = False
    D_R = X_circles[keep]

    t0 = time.time()
    M_G = DBSCAN(eps=EPS_C, min_samples=MIN_C).fit(D_R)
    t_retrain = time.time() - t0

    t0 = time.time()
    M_U_labels, n_aff, aff_list = dbscan_unlearn(X_circles, labels_circles, core_mask_circles, idx, EPS_C, MIN_C)
    t_unlearn = time.time() - t0

    ari = adjusted_rand_score(M_G.labels_, M_U_labels)
    nmi = normalized_mutual_info_score(M_G.labels_, M_U_labels)

    was_core = "CORE" if core_mask_circles[idx] else "border/noise"
    status = "[OK]" if ari > 0.95 else "[WARN] DEGRADED" if ari > 0.8 else "[FAIL] FAILED"

    print(f"  [{status}] idx={idx:3d} ({was_core}) | affected={n_aff} | "
          f"ARI={ari:.4f} | NMI={nmi:.4f}")

    results_circles.append({
        "dataset": "make_circles", "forget_idx": idx, "point_type": was_core,
        "n_affected": n_aff, "ari": ari, "nmi": nmi,
        "time_retrain_ms": t_retrain * 1000, "time_unlearn_ms": t_unlearn * 1000,
        "status": status
    })

plot_stress_test(X_circles, labels_circles,
                 bridges_circles[0] if bridges_circles else hard_targets_circles[0],
                 "Circles: Hard Deletion", "circles_hard_deletion")


# ======================================================================
# COMBINED SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("COMBINED STRESS-TEST SUMMARY")
print("=" * 70)

all_results = results_blobs + results_moons + results_circles
df = pd.DataFrame(all_results)
df.to_csv("phase1_results/stress_test_results.csv", index=False)

# Summary stats per dataset
for dataset_name in ["gaussian_blobs", "make_moons", "make_circles"]:
    subset = df[df["dataset"] == dataset_name]
    if len(subset) == 0:
        continue
    n_total = len(subset)
    n_perfect = len(subset[subset["ari"] > 0.99])
    n_good = len(subset[(subset["ari"] > 0.95) & (subset["ari"] <= 0.99)])
    n_degraded = len(subset[(subset["ari"] > 0.8) & (subset["ari"] <= 0.95)])
    n_failed = len(subset[subset["ari"] <= 0.8])
    avg_ari = subset["ari"].mean()
    avg_affected = subset["n_affected"].mean()

    print(f"\n  {dataset_name}:")
    print(f"    Tests run:        {n_total}")
    print(f"    Perfect (>0.99):  {n_perfect}")
    print(f"    Good (0.95-0.99): {n_good}")
    print(f"    Degraded (<0.95): {n_degraded}")
    print(f"    Failed (<0.80):   {n_failed}")
    print(f"    Avg ARI:          {avg_ari:.4f}")
    print(f"    Avg affected:     {avg_affected:.1f}")

# Highlight worst cases
print("\n  WORST CASES (lowest ARI):")
worst = df.nsmallest(10, "ari")
for _, row in worst.iterrows():
    print(f"    {row['dataset']:15s} idx={row['forget_idx']:3.0f} "
          f"({row['point_type']:6s}) affected={row['n_affected']:2.0f} "
          f"ARI={row['ari']:.4f}  {row['status']}")

print(f"\nAll results saved to phase1_results/stress_test_results.csv")
print(f"Plots saved to phase1_results/")
