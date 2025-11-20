import os, json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
GROUPS_FILE = "similarity_results_v2/groups.json"
OUT_DIR = "subspace_results"
EXPLAINED_VARIANCE_THRESHOLD = 0.85   # choose d_k so cumulative var >= this
MAX_D_PER_GROUP = 5                   # cap d_k to avoid large bases
MIN_D_PER_GROUP = 1
PLOT_SCREE = True
# ----------------------------------------

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Load groups ---
with open(GROUPS_FILE, "r") as f:
    groups = json.load(f)   # keys are strings
# Convert to sorted list-of-lists
group_list = [groups[k] for k in sorted(groups.keys(), key=int)]
print("Groups:", group_list)

# --- Load X (ensure shape N,C,T) ---
# If you already have X in memory, skip this load and use it.
# Replace this loader with your existing X if needed.
def load_signals(base_dir="UCI HAR Dataset/train/Inertial Signals"):
    files = [
        "body_acc_x_train.txt","body_acc_y_train.txt","body_acc_z_train.txt",
        "body_gyro_x_train.txt","body_gyro_y_train.txt","body_gyro_z_train.txt",
        "total_acc_x_train.txt","total_acc_y_train.txt","total_acc_z_train.txt",
    ]
    signals = [np.loadtxt(os.path.join(base_dir,f)) for f in files]
    S = np.stack(signals, axis=-1)   # (N, 128, 9)
    S = np.transpose(S, (0, 2, 1))   # -> (N, C, T)
    return S

X = load_signals()   # comment out if X already present
N, C, T = X.shape
print("Loaded X with shape:", X.shape)

# --- Per-group PCA ---
results = {}
col_total = 0
for gi, G in enumerate(group_list):
    G = [int(x) for x in G]  # ensure ints
    gk = len(G)
    print(f"\nProcessing group {gi}: channels {G} (size {gk})")
    # Build data matrix V: shape (N, gk*T)
    V = np.stack([X[i, G, :].ravel() for i in range(N)], axis=0)  # (N, gk*T)
    mu = V.mean(axis=0)         # (gk*T,)
    Vc = V - mu
    # Fit PCA with full components (up to cap)
    max_components = min(Vc.shape[1], max(2, min(20, Vc.shape[0])))  # safe cap
    pca = PCA(n_components=max_components)
    pca.fit(Vc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # choose d_k by threshold
    d_k = int(np.searchsorted(cumvar, EXPLAINED_VARIANCE_THRESHOLD) + 1)
    d_k = max(MIN_D_PER_GROUP, min(d_k, MAX_D_PER_GROUP))
    # refit pca with desired d_k to get stable components
    pca = PCA(n_components=d_k)
    pca.fit(Vc)
    Uk = pca.components_.T   # shape (gk*T, d_k)
    explained = pca.explained_variance_ratio_
    print(f"  chosen d_k = {d_k}, explained_var (per comp) = {explained}, cumulative = {explained.sum():.3f}")
    # Save
    results[f"group_{gi}"] = {
        "channels": G,
        "gk": gk,
        "mu_shape": mu.shape,
        "Uk_shape": Uk.shape,
        "explained_variance_ratio": explained.tolist(),
        "d_k": int(d_k)
    }
    np.save(os.path.join(OUT_DIR, f"group_{gi}_mu.npy"), mu)
    np.save(os.path.join(OUT_DIR, f"group_{gi}_Uk.npy"), Uk)
    # optional scree plot per group
    if PLOT_SCREE:
        plt.figure(figsize=(5,3))
        all_pca = PCA().fit(Vc)
        plt.plot(np.arange(1, len(all_pca.explained_variance_ratio_)+1), all_pca.explained_variance_ratio_, 'o-')
        plt.axvline(d_k, color='red', linestyle='--', label=f"chosen d={d_k}")
        plt.title(f"Group {gi} scree (channels {G})")
        plt.xlabel("component")
        plt.ylabel("explained variance ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"group_{gi}_scree.png"), dpi=150)
        plt.close()

# Save summary
with open(os.path.join(OUT_DIR, "group_pca_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved per-group PCA results to", OUT_DIR)
