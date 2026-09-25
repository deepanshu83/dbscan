# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
PHASE 1C: ENGINEERED HARD CASES
=================================
The standard datasets (blobs, moons, circles) with typical eps values are
TOO DENSE -- deleting a single point never causes structural changes.

This script engineers datasets WHERE single-point deletions WILL break things:
  1. Thin bridge between two clusters (only 1-2 points connecting them)
  2. Sparse chain clusters (barely connected)
  3. Cluster at exact density threshold (every core has exactly min_samples neighbors)
  4. Real-world datasets with tuned eps to hit the critical regime
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import time
import os

np.random.seed(42)
os.makedirs("phase1_results", exist_ok=True)

results = []


# ======================================================================
# UNLEARNING ALGORITHM
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


def run_deletion_test(D, eps, min_samples, dataset_name, target_indices=None):
    """Run unlearning on specific target indices and report results."""
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(D)
    labels = model.labels_
    core_mask = np.zeros(len(D), dtype=bool)
    core_mask[model.core_sample_indices_] = True

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_core = len(model.core_sample_indices_)
    print(f"  Model: {n_clusters} clusters, {n_core} core, {n_noise} noise, N={len(D)}")

    if target_indices is None:
        # Find fragile cores
        nbrs = NearestNeighbors(radius=eps).fit(D)
        _, indices = nbrs.radius_neighbors(D, return_distance=True)
        core_counts = [(i, len(indices[i])) for i in model.core_sample_indices_]
        core_counts.sort(key=lambda x: x[1])
        target_indices = [x[0] for x in core_counts[:20]]

    local_results = []
    for idx in target_indices:
        keep = np.ones(len(D), dtype=bool)
        keep[idx] = False
        D_R = D[keep]

        t0 = time.time()
        M_G = DBSCAN(eps=eps, min_samples=min_samples).fit(D_R)
        t_retrain = time.time() - t0

        t0 = time.time()
        M_U_labels, n_aff = dbscan_unlearn(D, labels, core_mask, idx, eps, min_samples)
        t_unlearn = time.time() - t0

        ari = adjusted_rand_score(M_G.labels_, M_U_labels)
        nmi = normalized_mutual_info_score(M_G.labels_, M_U_labels)
        was_core = "CORE" if core_mask[idx] else "border/noise"

        status = "[OK]" if ari > 0.95 else "[WARN]" if ari > 0.8 else "[FAIL]"
        affected_str = f"affected={n_aff}" if n_aff > 0 else "affected=0"

        print(f"    {status} idx={idx:3d} ({was_core}) | {affected_str} | "
              f"ARI={ari:.4f} | NMI={nmi:.4f}")

        local_results.append({
            "dataset": dataset_name, "forget_idx": idx, "point_type": was_core,
            "n_affected": n_aff, "ari": ari, "nmi": nmi,
            "time_retrain_ms": t_retrain * 1000, "time_unlearn_ms": t_unlearn * 1000,
        })

    return local_results, labels


# ======================================================================
# SCENARIO 1: Two blobs connected by a thin bridge (1-3 points)
# Deleting a bridge point MUST cause a structural change
# ======================================================================
print("=" * 70)
print("SCENARIO 1: Two Blobs Connected by a Thin Bridge")
print("=" * 70)

blob_A = np.random.normal(loc=[0, 0], scale=0.3, size=(30, 2))
blob_B = np.random.normal(loc=[4, 0], scale=0.3, size=(30, 2))
# Bridge: exactly 3 points connecting them, spaced so each is within eps of the next
bridge = np.array([[1.3, 0.0], [2.0, 0.0], [2.7, 0.0]])
D_bridge = np.vstack([blob_A, blob_B, bridge])
bridge_indices = list(range(60, 63))  # indices of bridge points

EPS_BR = 0.8
MIN_BR = 3
print(f"  eps={EPS_BR}, min_samples={MIN_BR}")
print(f"  Bridge point indices: {bridge_indices}")

r, labels_br = run_deletion_test(D_bridge, EPS_BR, MIN_BR, "thin_bridge", bridge_indices)
results.extend(r)

# Plot
plt.figure(figsize=(8, 4))
for lbl in set(labels_br):
    mask = labels_br == lbl
    color = "lightgray" if lbl == -1 else None
    plt.scatter(D_bridge[mask, 0], D_bridge[mask, 1], c=color, s=30,
                label=f"C{lbl}" if lbl != -1 else "noise")
for bi in bridge_indices:
    plt.scatter(D_bridge[bi, 0], D_bridge[bi, 1], c="red", marker="*",
                s=200, zorder=10, edgecolors="black")
plt.title("Thin Bridge Between Two Clusters (red stars = bridge points)")
plt.legend(fontsize=7)
plt.savefig("phase1_results/scenario1_thin_bridge.png", dpi=150, bbox_inches="tight")
plt.close()


# ======================================================================
# SCENARIO 2: Sparse chain cluster (barely above density threshold)
# Every point has exactly min_samples neighbors
# ======================================================================
print("\n" + "=" * 70)
print("SCENARIO 2: Sparse Chain (Barely Above Density Threshold)")
print("=" * 70)

# Create a 1D chain of points spaced just under eps apart
# Each point has neighbors on both sides, giving exactly min_samples neighbors
EPS_CH = 0.5
MIN_CH = 3
n_chain = 50
# Space points at eps * 0.9 apart so each has 2-3 neighbors within eps
spacing = EPS_CH * 0.8
chain_x = np.arange(n_chain) * spacing
chain_y = np.random.normal(0, 0.05, n_chain)  # tiny y-jitter
D_chain = np.column_stack([chain_x, chain_y])

print(f"  eps={EPS_CH}, min_samples={MIN_CH}, chain of {n_chain} points")
print(f"  Spacing: {spacing:.2f} (eps={EPS_CH})")

# Target every 5th point in the chain
chain_targets = list(range(0, n_chain, 5))
r, labels_ch = run_deletion_test(D_chain, EPS_CH, MIN_CH, "sparse_chain", chain_targets)
results.extend(r)

plt.figure(figsize=(10, 3))
for lbl in set(labels_ch):
    mask = labels_ch == lbl
    color = "lightgray" if lbl == -1 else None
    plt.scatter(D_chain[mask, 0], D_chain[mask, 1], c=color, s=30,
                label=f"C{lbl}" if lbl != -1 else "noise")
plt.title("Sparse Chain Cluster (barely connected)")
plt.legend(fontsize=7)
plt.savefig("phase1_results/scenario2_sparse_chain.png", dpi=150, bbox_inches="tight")
plt.close()


# ======================================================================
# SCENARIO 3: Moons with critical eps (barely enough to connect)
# Tune eps so the moons are JUST connected -- deletions at the sparse
# ends or narrow neck will cause structural breaks
# ======================================================================
print("\n" + "=" * 70)
print("SCENARIO 3: Moons at Critical Density (eps tuned to barely connect)")
print("=" * 70)

X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

# Find the critical eps where cluster count transitions
for test_eps in np.arange(0.1, 0.5, 0.02):
    test_model = DBSCAN(eps=test_eps, min_samples=5).fit(X_moons)
    test_labels = test_model.labels_
    nc = len(set(test_labels)) - (1 if -1 in test_labels else 0)
    nn = list(test_labels).count(-1)
    if nc == 2 and nn < 15:
        EPS_CRIT = test_eps
        print(f"  Critical eps found: {EPS_CRIT:.2f} -> {nc} clusters, {nn} noise")
        break
else:
    EPS_CRIT = 0.2

MIN_CRIT = 5
print(f"  Using eps={EPS_CRIT}, min_samples={MIN_CRIT}")

# Find the sparsest points (fewest neighbors)
model_crit = DBSCAN(eps=EPS_CRIT, min_samples=MIN_CRIT).fit(X_moons)
labels_crit = model_crit.labels_
core_mask_crit = np.zeros(len(X_moons), dtype=bool)
core_mask_crit[model_crit.core_sample_indices_] = True

nbrs_crit = NearestNeighbors(radius=EPS_CRIT).fit(X_moons)
_, indices_crit = nbrs_crit.radius_neighbors(X_moons, return_distance=True)
core_counts = [(i, len(indices_crit[i])) for i in model_crit.core_sample_indices_]
core_counts.sort(key=lambda x: x[1])

# Target the 15 sparsest core points
sparse_targets = [x[0] for x in core_counts[:15]]
print(f"  Targeting {len(sparse_targets)} sparsest core points")
print(f"  Their neighbor counts: {[x[1] for x in core_counts[:15]]}")

r, _ = run_deletion_test(X_moons, EPS_CRIT, MIN_CRIT, "moons_critical_eps", sparse_targets)
results.extend(r)


# ======================================================================
# SCENARIO 4: Cluster with a "waist" (hourglass shape)
# Two dense regions connected by a narrow waist -- delete at the waist
# ======================================================================
print("\n" + "=" * 70)
print("SCENARIO 4: Hourglass Cluster (narrow waist)")
print("=" * 70)

top = np.random.normal(loc=[0, 2], scale=0.5, size=(40, 2))
bottom = np.random.normal(loc=[0, -2], scale=0.5, size=(40, 2))
# Waist: only 4 points connecting top and bottom
waist = np.array([[0, 0.8], [0, 0.3], [0, -0.3], [0, -0.8]])
D_hourglass = np.vstack([top, bottom, waist])
waist_indices = list(range(80, 84))

EPS_HG = 0.7
MIN_HG = 4
print(f"  eps={EPS_HG}, min_samples={MIN_HG}")
print(f"  Waist point indices: {waist_indices}")

r, labels_hg = run_deletion_test(D_hourglass, EPS_HG, MIN_HG, "hourglass_waist", waist_indices)
results.extend(r)

plt.figure(figsize=(5, 7))
for lbl in set(labels_hg):
    mask = labels_hg == lbl
    color = "lightgray" if lbl == -1 else None
    plt.scatter(D_hourglass[mask, 0], D_hourglass[mask, 1], c=color, s=30,
                label=f"C{lbl}" if lbl != -1 else "noise")
for wi in waist_indices:
    plt.scatter(D_hourglass[wi, 0], D_hourglass[wi, 1], c="red", marker="*",
                s=200, zorder=10, edgecolors="black")
plt.title("Hourglass: Narrow Waist (red stars = waist points)")
plt.legend(fontsize=7)
plt.savefig("phase1_results/scenario4_hourglass.png", dpi=150, bbox_inches="tight")
plt.close()


# ======================================================================
# SCENARIO 5: Multiple deletions in sequence (cumulative stress test)
# Delete 10 points one-by-one, each time unlearning from the current model
# ======================================================================
print("\n" + "=" * 70)
print("SCENARIO 5: Sequential Multi-Deletion (cumulative)")
print("=" * 70)

blob1 = np.random.normal(loc=[0, 0], scale=0.4, size=(50, 2))
blob2 = np.random.normal(loc=[3, 3], scale=0.4, size=(50, 2))
D_seq = np.vstack([blob1, blob2])
EPS_SEQ, MIN_SEQ = 0.6, 4

# Sequential deletion: delete point, rebuild, delete next, etc.
D_current = D_seq.copy()
seq_results = []
delete_order = [0, 1, 2, 50, 51, 52, 25, 75, 10, 60]

for step, del_idx in enumerate(delete_order):
    if del_idx >= len(D_current):
        continue

    model_cur = DBSCAN(eps=EPS_SEQ, min_samples=MIN_SEQ).fit(D_current)
    labels_cur = model_cur.labels_
    core_mask_cur = np.zeros(len(D_current), dtype=bool)
    core_mask_cur[model_cur.core_sample_indices_] = True

    # Gold: retrain without the point
    keep = np.ones(len(D_current), dtype=bool)
    keep[del_idx] = False
    D_R = D_current[keep]
    M_G = DBSCAN(eps=EPS_SEQ, min_samples=MIN_SEQ).fit(D_R)

    # Unlearn
    M_U_labels, n_aff = dbscan_unlearn(D_current, labels_cur, core_mask_cur, del_idx, EPS_SEQ, MIN_SEQ)
    ari = adjusted_rand_score(M_G.labels_, M_U_labels)

    status = "[OK]" if ari > 0.95 else "[WARN]" if ari > 0.8 else "[FAIL]"
    print(f"  Step {step+1}: delete idx={del_idx} | N={len(D_current)} | "
          f"affected={n_aff} | ARI={ari:.4f} {status}")

    results.append({
        "dataset": "sequential_deletion", "forget_idx": del_idx,
        "point_type": "CORE" if core_mask_cur[del_idx] else "border/noise",
        "n_affected": n_aff, "ari": ari, "nmi": 0,
        "time_retrain_ms": 0, "time_unlearn_ms": 0,
    })

    # Actually remove the point for next iteration
    D_current = D_R


# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("ENGINEERED HARD CASES -- SUMMARY")
print("=" * 70)

df = pd.DataFrame(results)
df.to_csv("phase1_results/hard_cases_results.csv", index=False)

for ds in df["dataset"].unique():
    subset = df[df["dataset"] == ds]
    n_total = len(subset)
    n_affected = len(subset[subset["n_affected"] > 0])
    avg_ari = subset["ari"].mean()
    min_ari = subset["ari"].min()
    print(f"\n  {ds}:")
    print(f"    Tests: {n_total} | With structural changes: {n_affected}")
    print(f"    Avg ARI: {avg_ari:.4f} | Min ARI: {min_ari:.4f}")

# Highlight ALL cases where ARI < 1.0
degraded = df[df["ari"] < 0.999]
if len(degraded) > 0:
    print(f"\n  >>> CASES WHERE UNLEARNING WAS IMPERFECT (ARI < 1.0): {len(degraded)} <<<")
    for _, row in degraded.iterrows():
        print(f"    {row['dataset']:25s} idx={row['forget_idx']:3.0f} "
              f"affected={row['n_affected']:2.0f} ARI={row['ari']:.4f}")
else:
    print("\n  All cases achieved ARI=1.0 (perfect unlearning)")

print(f"\nResults saved to phase1_results/hard_cases_results.csv")
