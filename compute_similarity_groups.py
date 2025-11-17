#!/usr/bin/env python3
"""
Compute channel similarity matrices and statistical groups for HAR inertial data.

Loads HAR using the user's function.

Saves all plots and matrices in similarity_results/.
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch, periodogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from scipy.cluster import hierarchy


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
X = load_signals("UCI HAR Dataset/train/Inertial Signals")  # (N, 128, 9)
X_test = load_signals("UCI HAR Dataset/test/Inertial Signals")

y = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int)

N, T, C = X.shape  # (N samples, 128 length, 9 channels)
X = np.transpose(X, (0, 2, 1))       # → (N, 9, 128)
X_test = np.transpose(X_test, (0, 2, 1))

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------
OUT_DIR = "similarity_results"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. CORRELATION MATRIX
# ---------------------------------------------------------
def correlation_matrix(X):
    # X: [N, C, T]
    N, C, T = X.shape
    # Flatten each channel across samples & time
    vecs = X.transpose(1, 0, 2).reshape(C, -1)   # shape [C, N*T]
    corr = np.corrcoef(vecs)
    corr = np.nan_to_num(corr)
    return corr    # shape [C, C]

print("Computing correlation matrix...")
corr_mat = correlation_matrix(X)
np.save(f"{OUT_DIR}/corr.npy", corr_mat)


# ---------------------------------------------------------
# 3. SPECTRAL SIMILARITY MATRIX
# ---------------------------------------------------------
def spectral_similarity(X, fs=50):
    """
    Computes spectral similarity between channels.
    Ensures consistent PSD length by:
         - fixing nperseg
         - fixing nfft
         - forcing same frequency bins
    """

    N, C, T = X.shape
    nperseg = min(128, T)
    nfft = 256  # fixed for consistency

    psds = []

    for c in range(C):
        all_psd = []
        for i in range(N):
            f, Pxx = welch(
                X[i, c],
                fs=fs,
                nperseg=nperseg,
                nfft=nfft,     # 👈 this ensures consistent PSD length
                noverlap=nperseg//2
            )
            all_psd.append(Pxx)

        avg_psd = np.mean(all_psd, axis=0)
        psds.append(np.log1p(avg_psd + 1e-12))

    psds = np.stack(psds, axis=0)  # now shape = (C, F) correctly

    dist = squareform(pdist(psds, metric='euclidean'))
    sigma = np.median(dist) + 1e-9
    sim = np.exp(-(dist**2) / (2 * sigma**2))

    return sim

print("Computing spectral similarity...")
spec_sim = spectral_similarity(X, fs=50)
np.save(f"{OUT_DIR}/spec_sim.npy", spec_sim)


# ---------------------------------------------------------
# 4. PCA CONTRIBUTION SCORES
# ---------------------------------------------------------
def pca_channel_contribution(X, n_components=10):
    N, C, T = X.shape
    V = X.reshape(N, C*T)
    Vc = V - V.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(n_components, C*T))
    pca.fit(Vc)
    
    comps = pca.components_  # [k, C*T]
    per_channel_scores = np.zeros(C)
    
    for k in range(comps.shape[0]):
        comp = comps[k].reshape(C, T)
        energy = np.linalg.norm(comp, axis=1)  # [C]
        per_channel_scores += energy * pca.explained_variance_ratio_[k]
    
    # Normalize
    per_channel_scores = (
        per_channel_scores - per_channel_scores.min()
    ) / (per_channel_scores.max() - per_channel_scores.min() + 1e-12)
    
    return per_channel_scores


print("Computing PCA channel contributions...")
pca_scores = pca_channel_contribution(X, n_components=10)
np.save(f"{OUT_DIR}/pca_scores.npy", pca_scores)


# ---------------------------------------------------------
# 5. MUTUAL INFORMATION SCORES
# ---------------------------------------------------------
def mutual_info_per_channel(X, y):
    N, C, T = X.shape
    scores = []
    
    for c in range(C):
        # simple features for MI
        f_mean = X[:, c].mean(axis=1)
        f_std  = X[:, c].std(axis=1)
        
        dom_freqs = []
        for i in range(N):
            f, Pxx = periodogram(X[i, c])
            dom_freqs.append(f[np.argmax(Pxx)])
        dom_freqs = np.array(dom_freqs)
        
        feats = np.stack([f_mean, f_std, dom_freqs], axis=1)
        mi = mutual_info_classif(feats, y, random_state=0)
        
        scores.append(np.mean(mi))
    
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return scores


print("Computing mutual information scores...")
mi_scores = mutual_info_per_channel(X, y)
np.save(f"{OUT_DIR}/mi_scores.npy", mi_scores)


# ---------------------------------------------------------
# 6. COMBINE ALL INTO FINAL SIMILARITY MATRIX
# ---------------------------------------------------------
def build_combined_similarity(corr_mat, spec_sim, pca_scores, mi_scores,
                              w_corr=0.3, w_spec=0.3, w_pca=0.2, w_mi=0.2):
    C = corr_mat.shape[0]
    
    corr_norm = (corr_mat - corr_mat.min()) / (corr_mat.max() - corr_mat.min() + 1e-12)
    
    pca_pair = 0.5 * (pca_scores[:, None] + pca_scores[None, :])
    mi_pair  = 0.5 * (mi_scores[:, None] + mi_scores[None, :])
    
    for M in [pca_pair, mi_pair]:
        M -= M.min()
        M /= (M.max() - M.min() + 1e-12)
    
    combined = (
        w_corr * corr_norm +
        w_spec * spec_sim +
        w_pca * pca_pair +
        w_mi  * mi_pair
    )
    
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
    return combined


print("Building combined similarity matrix...")
combined = build_combined_similarity(corr_mat, spec_sim, pca_scores, mi_scores)
np.save(f"{OUT_DIR}/combined_similarity.npy", combined)


# ---------------------------------------------------------
# 7. CLUSTER CHANNELS (statistical groups)
# ---------------------------------------------------------
def cluster_channels(sim_mat, n_clusters=4, out_dir=OUT_DIR):
    C = sim_mat.shape[0]
    dist = 1 - sim_mat
    
    condensed = squareform(dist, checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    
    # Save dendrogram
    plt.figure(figsize=(8, 4))
    hierarchy.dendrogram(Z, labels=[f"ch{c}" for c in range(C)])
    plt.title("Dendrogram: Statistical Channel Clustering")
    plt.savefig(f"{out_dir}/dendrogram.png", dpi=150)
    plt.close()
    
    # Agglomerative clustering
    cluster = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="complete"
    )
    labels = cluster.fit_predict(dist)
    
    groups = {}
    for c in range(C):
        groups.setdefault(int(labels[c]), []).append(c)
    
    return groups, labels, Z


print("Clustering channels...")
groups, labels, Z = cluster_channels(combined, n_clusters=4)
print("Groups found:", groups)

with open(f"{OUT_DIR}/groups.json", "w") as f:
    json.dump(groups, f, indent=2)


# ---------------------------------------------------------
# 8. SAVE COMBINED SIMILARITY HEATMAP
# ---------------------------------------------------------
order = hierarchy.leaves_list(Z)  # order in dendrogram
ordered_sim = combined[np.ix_(order, order)]

plt.figure(figsize=(6, 5))
sns.heatmap(
    ordered_sim,
    xticklabels=[f"ch{i}" for i in order],
    yticklabels=[f"ch{i}" for i in order],
    cmap="viridis",
    vmin=0, vmax=1
)
plt.title("Combined Similarity Matrix (Cluster Ordered)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_similarity_heatmap.png", dpi=150)
plt.close()

print("All similarity matrices and plots saved in:", OUT_DIR)
