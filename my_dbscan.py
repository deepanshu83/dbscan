import numpy as np

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# =========================
# DATASET
# =========================

X, _ = make_moons(
    n_samples=500,
    noise=0.05,
    random_state=42
)

eps = 0.1
min_samples = 5


# =========================
# ORIGINAL DBSCAN
# =========================

model_original = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

labels_original = model_original.fit_predict(X)

clusters_original = len(set(labels_original)) - (
    1 if -1 in labels_original else 0
)

noise_original = list(labels_original).count(-1)


print("ORIGINAL")
print("Points:", len(X))
print("Clusters:", clusters_original)
print("Noise:", noise_original)


# =========================
# FIND CORE POINTS
# =========================

# core_indices = model_original.core_sample_indices_

# print("\nCORE POINTS:")
# print("Number of core points:", len(core_indices))
# print("First 20 core indices:", core_indices[:20])


# =========================
# DELETE ONE POINT
# =========================

delete_index = 250

print("\nDELETING POINT:")
print("Index:", delete_index)
print("Coordinates:", X[delete_index])

X_deleted = np.delete(
    X,
    delete_index,
    axis=0
)

print("Points after deletion:", len(X_deleted))


# =========================
# DBSCAN AFTER DELETION
# =========================

model_deleted = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

labels_deleted = model_deleted.fit_predict(X_deleted)

clusters_deleted = len(set(labels_deleted)) - (
    1 if -1 in labels_deleted else 0
)

noise_deleted = list(labels_deleted).count(-1)


print("\nAFTER DELETING POINT")
print("Clusters:", clusters_deleted)
print("Noise:", noise_deleted)


# =========================
# VISUAL COMPARISON
# =========================

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels_original
)

plt.title("Original - 500 Points")
plt.xlabel("X")
plt.ylabel("Y")


plt.subplot(1, 2, 2)

plt.scatter(
    X_deleted[:, 0],
    X_deleted[:, 1],
    c=labels_deleted
)

plt.title("After Deletion - 499 Points")
plt.xlabel("X")
plt.ylabel("Y")


plt.show()