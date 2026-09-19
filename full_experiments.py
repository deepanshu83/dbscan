"""
FULL DBSCAN EXPERIMENT SUITE
============================
Covers everything for the meeting:
  EXPERIMENT 1: baseline DBSCAN on full data
  EXPERIMENT 2: changing eps / min_samples (this is NOT unlearning, just re-running
                 DBSCAN with different hyperparameters on the SAME full data)
  EXPERIMENT 3: deleting a point + retraining from scratch (this is D_R -> M_G,
                 the "gold model" / brute-force way, NOT unlearning either)
  EXPERIMENT 4: actual UNLEARNING algorithm U (local update, no full retrain)
  EXPERIMENT 5: compare M_U vs M_G across MULTIPLE forgotten points (robustness check)

Outputs:
  - results_summary.csv   (table of every experiment's result)
  - plots/*.png           (visual clusters for each experiment)
  - console printout       (explanation of each step, good for narrating in meeting)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import time, os

np.random.seed(42)
os.makedirs("plots", exist_ok=True)

# ----------------------------------------------------------------------
# DATA: D_full = D_R + D_F  (full dataset before forgetting anything)
# ----------------------------------------------------------------------
blob1 = np.random.normal(loc=[0, 0], scale=0.3, size=(40, 2))
blob2 = np.random.normal(loc=[5, 5], scale=0.3, size=(40, 2))
blob3 = np.random.normal(loc=[0, 5], scale=0.3, size=(40, 2))
noise = np.random.uniform(low=-3, high=8, size=(10, 2))
D_full = np.vstack([blob1, blob2, blob3, noise])
N = len(D_full)
results = []

def plot_clusters(data, labels, title, fname):
    plt.figure(figsize=(5, 5))
    unique_labels = set(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        color = "black" if lbl == -1 else None
        marker = "x" if lbl == -1 else "o"
        plt.scatter(data[mask, 0], data[mask, 1], c=color, marker=marker, s=25, label=f"cluster {lbl}" if lbl != -1 else "noise")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.savefig(f"plots/{fname}.png", dpi=110, bbox_inches="tight")
    plt.close()

# ========================================================================
# EXPERIMENT 1: BASELINE  (M = DBSCAN on full data, default hyperparams)
# ========================================================================
print("=" * 60)
print("EXPERIMENT 1: Baseline DBSCAN (this is model M)")
print("=" * 60)
EPS_BASE, MIN_BASE = 0.6, 5
M = DBSCAN(eps=EPS_BASE, min_samples=MIN_BASE).fit(D_full)
M_labels = M.labels_
n_clusters_base = len(set(M_labels)) - (1 if -1 in M_labels else 0)
print(f"eps={EPS_BASE}, min_samples={MIN_BASE} -> {n_clusters_base} clusters, "
      f"{list(M_labels).count(-1)} noise points")
plot_clusters(D_full, M_labels, f"Baseline (eps={EPS_BASE}, min={MIN_BASE})", "exp1_baseline")
results.append(dict(experiment="1_baseline", eps=EPS_BASE, min_samples=MIN_BASE,
                     n_clusters=n_clusters_base, note="original model M"))

# ========================================================================
# EXPERIMENT 2: CHANGING eps / min_samples  (NOT unlearning - just re-running
# DBSCAN with different hyperparameters on the SAME full dataset)
# ========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: Varying eps / min_samples (hyperparameter sweep)")
print("NOTE: this is just re-running DBSCAN, it has nothing to do with")
print("      forgetting data - it's included to show the DIFFERENCE")
print("      between 'changing hyperparams' and 'unlearning'.")
print("=" * 60)
for eps_test, min_test in [(0.4, 5), (0.8, 5), (0.6, 3), (0.6, 8)]:
    model = DBSCAN(eps=eps_test, min_samples=min_test).fit(D_full)
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  eps={eps_test}, min_samples={min_test} -> {n_clusters} clusters, "
          f"{list(labels).count(-1)} noise points")
    plot_clusters(D_full, labels, f"eps={eps_test}, min={min_test}", f"exp2_eps{eps_test}_min{min_test}")
    results.append(dict(experiment="2_hyperparam_sweep", eps=eps_test, min_samples=min_test,
                         n_clusters=n_clusters, note="just re-running DBSCAN, NOT unlearning"))

# ========================================================================
# EXPERIMENT 3: DELETE A POINT + FULL RETRAIN  (D_R -> M_G, the "gold"/brute
# force approach - correct, but SLOW because it retrains on all of D_R)
# ========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: Delete one point, retrain from scratch on D_R -> M_G")
print("=" * 60)
forget_idx = 0
keep_mask = np.ones(N, dtype=bool)
keep_mask[forget_idx] = False
D_R = D_full[keep_mask]

t0 = time.time()
M_G = DBSCAN(eps=EPS_BASE, min_samples=MIN_BASE).fit(D_R)
time_retrain = time.time() - t0
M_G_labels = M_G.labels_
n_clusters_G = len(set(M_G_labels)) - (1 if -1 in M_G_labels else 0)
print(f"Forgot point idx={forget_idx}. Retrained M_G on D_R ({len(D_R)} points) "
      f"in {time_retrain*1000:.3f} ms -> {n_clusters_G} clusters")
plot_clusters(D_R, M_G_labels, "Gold model M_G (retrained on D_R)", "exp3_gold_model")
results.append(dict(experiment="3_gold_retrain", eps=EPS_BASE, min_samples=MIN_BASE,
                     n_clusters=n_clusters_G, time_ms=time_retrain * 1000,
                     note="brute-force retrain on D_R, this is the benchmark M_G"))

# ========================================================================
# EXPERIMENT 4: THE ACTUAL UNLEARNING ALGORITHM U -> produces M_U
# (local update only, does NOT retrain on all of D_R)
# ========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 4: Unlearning algorithm U -> produces M_U (local update only)")
print("=" * 60)

M_core_mask = np.zeros(N, dtype=bool)
M_core_mask[M.core_sample_indices_] = True

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

t0 = time.time()
M_U_labels, n_affected = dbscan_unlearn(D_full, M_labels, M_core_mask, forget_idx, EPS_BASE, MIN_BASE)
time_unlearn = time.time() - t0

ari = adjusted_rand_score(M_G_labels, M_U_labels)
nmi = normalized_mutual_info_score(M_G_labels, M_U_labels)
n_clusters_U = len(set(M_U_labels)) - (1 if -1 in M_U_labels else 0)

print(f"Unlearning done in {time_unlearn*1000:.3f} ms | affected core points: {n_affected}")
print(f"M_U clusters: {n_clusters_U} | ARI(M_U, M_G) = {ari:.4f} | NMI = {nmi:.4f}")
plot_clusters(D_R, M_U_labels, "Unlearned model M_U", "exp4_unlearned_model")
results.append(dict(experiment="4_unlearning", eps=EPS_BASE, min_samples=MIN_BASE,
                     n_clusters=n_clusters_U, time_ms=time_unlearn * 1000,
                     ari_vs_gold=ari, nmi_vs_gold=nmi,
                     note="local unlearning algorithm U, compared against M_G"))

# ========================================================================
# EXPERIMENT 5: ROBUSTNESS - repeat unlearning for MULTIPLE different
# forgotten points, to show it's not a one-off lucky result
# ========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 5: Repeat unlearning for 10 different forgotten points")
print("=" * 60)
test_indices = np.random.choice(N, size=10, replace=False)
for idx in test_indices:
    keep_mask_i = np.ones(N, dtype=bool)
    keep_mask_i[idx] = False
    D_R_i = D_full[keep_mask_i]
    M_G_i = DBSCAN(eps=EPS_BASE, min_samples=MIN_BASE).fit(D_R_i).labels_
    M_U_i, n_aff_i = dbscan_unlearn(D_full, M_labels, M_core_mask, idx, EPS_BASE, MIN_BASE)
    ari_i = adjusted_rand_score(M_G_i, M_U_i)
    print(f"  forget_idx={idx:3d} | affected_core={n_aff_i} | ARI(M_U, M_G)={ari_i:.4f}")
    results.append(dict(experiment="5_robustness", eps=EPS_BASE, min_samples=MIN_BASE,
                         forget_idx=int(idx), n_affected=n_aff_i, ari_vs_gold=ari_i,
                         note="repeated unlearning test"))

# ========================================================================
# SAVE EVERYTHING TO CSV
# ========================================================================
df_results = pd.DataFrame(results)
df_results.to_csv("results_summary.csv", index=False)
print("\n" + "=" * 60)
print("All results saved to results_summary.csv")
print("All plots saved to plots/ folder")
print("=" * 60)
