import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch, coherence
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------
# 1. LOAD DATA (your exact loader)
# ---------------------------------------------------------
def load_signals(base_dir):
    files = [
        "body_acc_x_train.txt",
        "body_acc_y_train.txt",
        "body_acc_z_train.txt",
        "body_gyro_x_train.txt",
        "body_gyro_y_train.txt",
        "body_gyro_z_train.txt",
        "total_acc_x_train.txt",
        "total_acc_y_train.txt",
        "total_acc_z_train.txt",
    ]
    
    if "test" in base_dir.lower():
        files = [f.replace("train", "test") for f in files]

    signals = [np.loadtxt(os.path.join(base_dir, f)) for f in files]
    signals = np.stack(signals, axis=-1)   # (N, 128, 9)
    return signals


print("Loading UCI-HAR signals...")
X = load_signals("UCI HAR Dataset/train/Inertial Signals")
X = np.transpose(X, (0, 2, 1))   # (N, 9, 128)
N, C, T = X.shape

print(f"Train shape: {X.shape} (N, C, T)")


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------
OUT_DIR = "similarity_results_v2"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. CORRELATION SIMILARITY
# ---------------------------------------------------------
def correlation_similarity(X):
    """
    Pearson correlation across channels.
    Result = |corr| to reflect magnitude-based similarity.
    """
    N, C, T = X.shape
    vecs = X.transpose(1, 0, 2).reshape(C, -1)  # (C, N*T)
    corr = np.corrcoef(vecs)
    corr = np.nan_to_num(corr)
    corr = np.abs(corr)       # magnitude of correlation as similarity
    return corr


print("Computing correlation similarity...")
corr_sim = correlation_similarity(X)
np.save(f"{OUT_DIR}/corr_sim.npy", corr_sim)


# ---------------------------------------------------------
# 3. SPECTRAL SIMILARITY
# ---------------------------------------------------------
def spectral_similarity(X, fs=50):
    """
    Spectral similarity using PSD distance → RBF kernel.
    """
    N, C, T = X.shape

    psds = []
    nperseg = min(128, T)
    nfft = 256  # forces consistent PSD size

    for c in range(C):
        all_psd = []
        for i in range(N):
            f, Pxx = welch(
                X[i, c],
                fs=fs,
                nperseg=nperseg,
                nfft=nfft,
                noverlap=nperseg // 2,
            )
            all_psd.append(Pxx)
        avg_psd = np.mean(all_psd, axis=0)
        psds.append(np.log1p(avg_psd + 1e-12))

    psds = np.stack(psds, axis=0)  # (C, F)

    dist = squareform(pdist(psds, metric="euclidean"))
    sigma = np.median(dist) + 1e-9
    sim = np.exp(-(dist**2) / (2 * sigma**2))
    return sim


print("Computing spectral similarity...")
spec_sim = spectral_similarity(X)
np.save(f"{OUT_DIR}/spec_sim.npy", spec_sim)


# ---------------------------------------------------------
# 4. COHERENCE SIMILARITY
# ---------------------------------------------------------
def coherence_similarity(X, fs=50):
    """
    Coherence: average magnitude-squared coherence across frequencies.
    True channel-to-channel similarity measure capturing frequency correlation.
    """
    N, C, T = X.shape
    coh_mat = np.zeros((C, C))

    # compute pairwise coherence
    for i in range(C):
        for j in range(C):
            if i == j:
                coh_mat[i, j] = 1.0
                continue

            # average coherence across samples for stability
            all_coh = []
            for n in range(min(N, 100)):  # limit samples for speed (100 is enough)
                f, Cxy = coherence(
                    X[n, i],
                    X[n, j],
                    fs=fs,
                    nperseg=min(128, T),
                )
                all_coh.append(Cxy)

            mean_coh = np.mean(all_coh, axis=0)
            coh_mat[i, j] = np.mean(mean_coh)

    return coh_mat


print("Computing coherence similarity...")
coh_sim = coherence_similarity(X)
np.save(f"{OUT_DIR}/coh_sim.npy", coh_sim)


# ---------------------------------------------------------
# 5. COMBINE SIMILARITIES
# ---------------------------------------------------------
def combine_similarities(corr, spec, coh,
                         w_corr=0.4, w_spec=0.3, w_coh=0.3):
    """
    Weighted combination of true similarity matrices.
    All matrices must be (C, C) and in [0,1].
    """

    combined = (
        w_corr * corr +
        w_spec * spec +
        w_coh  * coh
    )

    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
    return combined


print("Combining similarities...")
combined = combine_similarities(corr_sim, spec_sim, coh_sim)
np.save(f"{OUT_DIR}/combined_similarity.npy", combined)

# ---------------------------------------------------------
# 6. AUTOMATIC k SELECTION + CLUSTERING
# ---------------------------------------------------------
def choose_optimal_k(Z, C):
    """
    Given the linkage matrix Z (shape (C-1, 4)),
    find the optimal number of clusters by detecting the
    largest jump in merge distances.
    """
    merge_distances = Z[:, 2]          # distances at each merge step
    jumps = np.diff(merge_distances)   # jump between merges

    if len(jumps) == 0:
        return 1  # degenerate case

    # index of largest jump
    best_idx = np.argmax(jumps)

    # number of clusters = total merges remaining after the big jump
    # Example: C=9 channels, Z has 8 merges -> best_idx=6 -> k = 9-6 = 3
    best_k = C - best_idx - 1

    # sanity: at least 2 clusters
    best_k = max(2, min(best_k, C))

    return best_k


def cluster_channels_auto(sim_mat, out_dir=OUT_DIR):
    """
    Clusters channels using automatic k-detection.
    """
    C = sim_mat.shape[0]
    dist = 1 - sim_mat

    # Hierarchical clustering (full dendrogram)
    condensed = squareform(dist, checks=False)
    Z = hierarchy.linkage(condensed, method="complete")

    # ---- Plot dendrogram ----
    plt.figure(figsize=(8, 4))
    hierarchy.dendrogram(Z, labels=[f"ch{c}" for c in range(C)])
    plt.title("Dendrogram: Channel Clustering")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/dendrogram.png", dpi=150)
    plt.close()

    # ---- Optimal k detection ----
    optimal_k = choose_optimal_k(Z, C)
    print(f"Optimal number of clusters automatically chosen: k = {optimal_k}")

    # ---- Perform agglomerative clustering with best k ----
    cluster = AgglomerativeClustering(
        n_clusters=optimal_k,
        metric="precomputed",
        linkage="complete"
    )
    labels = cluster.fit_predict(dist)

    groups = {}
    for c in range(C):
        groups.setdefault(int(labels[c]), []).append(c)

    # Save groups
    with open(f"{out_dir}/groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    return groups, labels, Z, optimal_k

print("Clustering channels...")
groups, labels, Z, optimal_k = cluster_channels_auto(combined)
print("Groups:", groups)

print("Groups:", groups)

with open(f"{OUT_DIR}/groups.json", "w") as f:
    json.dump(groups, f, indent=2)


# ---------------------------------------------------------
# 7. SAVE HEATMAP
# ---------------------------------------------------------
order = hierarchy.leaves_list(Z)
ordered_sim = combined[np.ix_(order, order)]

plt.figure(figsize=(6, 5))
sns.heatmap(
    ordered_sim,
    xticklabels=[f"ch{i}" for i in order],
    yticklabels=[f"ch{i}" for i in order],
    cmap="viridis",
    vmin=0, vmax=1
)
plt.title("Combined Similarity (cluster-ordered)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_similarity_heatmap.png", dpi=150)
plt.close()

print("All results saved to:", OUT_DIR)
