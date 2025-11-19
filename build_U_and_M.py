import os
import json
import numpy as np
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
GROUPS_FILE     = "similarity_results_v2/groups.json"
PCA_DIR         = "subspace_results"
OUT_DIR         = "subspace_basis"
D_PER_GROUP     = 5    # <-- chosen manually
# -------------------------

Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

# -------------------------
# Load groups
# -------------------------
with open(GROUPS_FILE, "r") as f:
    groups = json.load(f)

# Convert to sorted list of lists
group_list = [groups[k] for k in sorted(groups.keys(), key=int)]
print("Groups:", group_list)

# -------------------------
# Load X to determine C,T
# -------------------------
def load_signals(base_dir="UCI HAR Dataset/train/Inertial Signals"):
    files = [
        "body_acc_x_train.txt","body_acc_y_train.txt","body_acc_z_train.txt",
        "body_gyro_x_train.txt","body_gyro_y_train.txt","body_gyro_z_train.txt",
        "total_acc_x_train.txt","total_acc_y_train.txt","total_acc_z_train.txt",
    ]
    signals = [np.loadtxt(os.path.join(base_dir, f)) for f in files]
    S = np.stack(signals, axis=-1)   # (N, 128, 9)
    S = np.transpose(S, (0, 2, 1))   # -> (N, C, T)
    return S

X = load_signals()
N, C, T = X.shape
print("Loaded X:", X.shape)

# -------------------------
# Build block-diagonal U
# -------------------------
Uk_list = []
mu_list = []
dims = []
block_row_start = 0

for gi, G in enumerate(group_list):
    G = [int(x) for x in G]
    gk = len(G)

    # load the PCA matrices
    mu_k_file = f"{PCA_DIR}/group_{gi}_mu.npy"
    Uk_file   = f"{PCA_DIR}/group_{gi}_Uk.npy"

    mu_k = np.load(mu_k_file)
    Uk_full = np.load(Uk_file)              # (gk*T, d_orig)

    # manually take first 5 columns
    Uk = Uk_full[:, :D_PER_GROUP]           # (gk*T, 5)

    mu_list.append(mu_k)
    Uk_list.append(Uk)
    dims.append(D_PER_GROUP)

# global total dimension
D = sum(dims)
print("Total subspace dimension D =", D)

# Create global block-diagonal U
U = np.zeros((C * T, D), dtype=np.float32)

row_start = 0
col_start = 0
for gi, G in enumerate(group_list):
    gk = len(G)
    rows = gk * T           # number of rows for this block
    cols = dims[gi]         # 5

    Uk = Uk_list[gi]

    U[row_start: row_start+rows, col_start: col_start+cols] = Uk

    row_start += rows
    col_start += cols

print("U shape:", U.shape)

# save U
np.save(f"{OUT_DIR}/U.npy", U)

# -------------------------
# Build global mean vector μ
# -------------------------
# For reconstructing any flattened (C,T) we need block concatenation of mu_k
# in the same block-order used to build U.
mu_global = np.zeros((C*T,), dtype=np.float32)

row_start = 0
for gi, G in enumerate(group_list):
    gk = len(G)
    rows = gk * T
    mu_k = mu_list[gi]
    mu_global[row_start:row_start+rows] = mu_k
    row_start += rows

np.save(f"{OUT_DIR}/mu_global.npy", mu_global)
print("mu_global shape:", mu_global.shape)

# -------------------------
# Compute reprojection matrix M = (UᵀU)^(-1) Uᵀ
# -------------------------
UtU = U.T @ U
print("Condition number of UᵀU:", np.linalg.cond(UtU))

M = np.linalg.inv(UtU) @ U.T   # shape: (D, C*T)

np.save(f"{OUT_DIR}/M.npy", M)
print("M shape:", M.shape)

# -------------------------
# Sanity check: projection → reconstruction
# -------------------------
print("\nRunning reconstruction sanity check...")

# take a random sample
i = np.random.randint(0, N)
x = X[i].reshape(-1)               # shape C*T
x_centered = x - mu_global

# project
alpha = M @ x_centered             # (D,)
# reconstruct
x_hat = U @ alpha + mu_global

rel_err = np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-12)
print("Relative reconstruction error:", rel_err)
print("=== U + M construction complete ===")
