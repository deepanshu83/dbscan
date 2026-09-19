"""
DBSCAN UNLEARNING - full demo
==============================
Terms:
  M    = original model, trained on FULL data (D_R + D_F)
  D_F  = data to forget (the point(s) we delete)
  D_R  = data retained (rest of the data)
  M_G  = gold model = DBSCAN trained FRESH on D_R only (this is the "correct answer")
  U    = unlearning algorithm -> takes M and D_F, produces M_U WITHOUT retraining on D_R
  M_U  = unlearned model (output of U). Goal: M_U should be almost identical to M_G
  ARI  = Adjusted Rand Index -> similarity between two clusterings (1.0 = identical)

Idea: instead of retraining DBSCAN on the whole D_R (slow, "relearning" not "unlearning"),
we do a LOCAL update: only the neighbors of the deleted point are affected, so we only
recompute clustering in that small local region.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import time

np.random.seed(42)

# ----------------------------------------------------------------------
# STEP 0: create some sample data (D_R + D_F combined = full dataset)
# ----------------------------------------------------------------------
# 3 blobs of points, plus a few scattered noise points
blob1 = np.random.normal(loc=[0, 0], scale=0.3, size=(40, 2))
blob2 = np.random.normal(loc=[5, 5], scale=0.3, size=(40, 2))
blob3 = np.random.normal(loc=[0, 5], scale=0.3, size=(40, 2))
noise = np.random.uniform(low=-3, high=8, size=(10, 2))

D_full = np.vstack([blob1, blob2, blob3, noise])   # this is D_R + D_F together
print(f"Total points in D_full: {len(D_full)}")

EPS = 0.6
MIN_SAMPLES = 5

# ----------------------------------------------------------------------
# STEP 1: train original model M on the FULL dataset
# ----------------------------------------------------------------------
M = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(D_full)
M_labels = M.labels_.copy()
M_core_mask = np.zeros(len(D_full), dtype=bool)
M_core_mask[M.core_sample_indices_] = True

print(f"\nM (original model) found {len(set(M_labels)) - (1 if -1 in M_labels else 0)} clusters")

# ----------------------------------------------------------------------
# STEP 2: pick the point(s) to FORGET  -> D_F
# ----------------------------------------------------------------------
forget_idx = 0   # let's forget the very first point (index 0, from blob1)
print(f"\nForgetting point at index {forget_idx}: {D_full[forget_idx]}")

# D_R = everything except the forgotten point
keep_mask = np.ones(len(D_full), dtype=bool)
keep_mask[forget_idx] = False
D_R = D_full[keep_mask]

# ----------------------------------------------------------------------
# STEP 3: build the GOLD MODEL M_G -> retrain DBSCAN FROM SCRATCH on D_R
#         (this is our "ground truth" to check against, NOT the method itself)
# ----------------------------------------------------------------------
t0 = time.time()
M_G = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(D_R)
time_retrain = time.time() - t0
M_G_labels = M_G.labels_

# ----------------------------------------------------------------------
# STEP 4: the actual UNLEARNING ALGORITHM U
#         Takes M (original model) + forget_idx -> produces M_U
#         WITHOUT retraining on the whole D_R
# ----------------------------------------------------------------------
def dbscan_unlearn(D_full, M_labels, M_core_mask, forget_idx, eps, min_samples):
    """
    Decremental DBSCAN unlearning.
    Only touches points affected by removing `forget_idx`.
    """
    n = len(D_full)
    forget_point = D_full[forget_idx]

    # 4a. find eps-neighbors of the forgotten point (points that lose a neighbor)
    nbrs = NearestNeighbors(radius=eps).fit(D_full)
    neighbor_idx = nbrs.radius_neighbors([forget_point], return_distance=False)[0]
    neighbor_idx = [i for i in neighbor_idx if i != forget_idx]
    print(f"  -> forgotten point had {len(neighbor_idx)} eps-neighbors, they lose 1 neighbor-count each")

    # 4b. recompute neighbor counts ONLY for those neighbors (not the whole dataset)
    affected_core_became_noncore = []
    for i in neighbor_idx:
        # count i's neighbors within eps, excluding the forgotten point
        dist = np.linalg.norm(D_full[i] - D_full, axis=1)
        new_neighbor_count = np.sum((dist <= eps) & (np.arange(n) != forget_idx))
        was_core = M_core_mask[i]
        is_core_now = new_neighbor_count >= min_samples
        if was_core and not is_core_now:
            affected_core_became_noncore.append(i)

    print(f"  -> {len(affected_core_became_noncore)} points changed from core to non-core")

    # 4c. build M_U labels: start as copy of M_labels, drop forgotten point,
    #     then LOCALLY re-cluster only around the affected region
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[forget_idx] = False
    remaining_idx = np.where(keep_mask)[0]
    D_remaining = D_full[remaining_idx]

    if len(affected_core_became_noncore) == 0:
        # nothing structurally changed -> just drop the forgotten point's label
        M_U_labels = M_labels[keep_mask]
    else:
        # local region = affected points + their neighbors (small area only)
        local_region = set(affected_core_became_noncore)
        for i in affected_core_became_noncore:
            dist = np.linalg.norm(D_full[i] - D_full, axis=1)
            local_region.update(np.where(dist <= eps)[0].tolist())
        local_region.discard(forget_idx)
        local_region = sorted(local_region)

        # re-run DBSCAN ONLY on this small local region (fast!)
        local_data = D_full[local_region]
        local_model = DBSCAN(eps=eps, min_samples=min_samples).fit(local_data)

        # map local labels back into the full (remaining) label array
        M_U_labels = M_labels[keep_mask].copy()
        # re-index: local_region indices -> position in D_remaining
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

    return M_U_labels

t0 = time.time()
M_U_labels = dbscan_unlearn(D_full, M_labels, M_core_mask, forget_idx, EPS, MIN_SAMPLES)
time_unlearn = time.time() - t0

# ----------------------------------------------------------------------
# STEP 5: VERIFY -> compare M_U with M_G  (|M_U - M_G| < epsilon check)
# ----------------------------------------------------------------------
ari = adjusted_rand_score(M_G_labels, M_U_labels)
nmi = normalized_mutual_info_score(M_G_labels, M_U_labels)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"ARI (M_U vs M_G):  {ari:.4f}   (1.0 = perfect match)")
print(f"NMI (M_U vs M_G):  {nmi:.4f}   (1.0 = perfect match)")
print(f"Time - full retrain (M_G):   {time_retrain*1000:.3f} ms")
print(f"Time - unlearning U (M_U):   {time_unlearn*1000:.3f} ms")
print(f"Speedup: {time_retrain/time_unlearn:.2f}x faster" if time_unlearn > 0 else "instant")

if ari > 0.95:
    print("\n✅ Unlearning successful: M_U is (almost) identical to M_G")
else:
    print("\n⚠️ M_U differs from M_G - check eps/min_samples or affected region logic")
