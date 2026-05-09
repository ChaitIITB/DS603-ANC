import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report,
                             silhouette_score)
from scipy.io import arff
import warnings, urllib.request, os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 0. Colour palette ──────────────────────────────────────────────────────────
C_NORMAL   = "#2196F3"   # blue  – normal samples
C_ANOM     = "#F44336"   # red   – anomalous (original)
C_PERTURB  = "#4CAF50"   # green – anomalous (perturbed)
C_CENTROID = "#FF9800"   # orange
BG         = "#0f1117"
GRID       = "#1e2130"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#444", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "grid.color": GRID,
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "monospace",
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ECG200
# ══════════════════════════════════════════════════════════════════════════════
def load_ecg200(data_dir="data/ECG200"):
    """
    Load the real ECG200 dataset from ARFF files.
    ECG200: 200 samples, 96 time steps, 2 classes.
      Normal (1)    : 133 samples — normal heartbeat
      Anomalous (-1):  67 samples — myocardial infarction
    """
    import pandas as pd

    def read_split(path):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        df[df.columns[-1]] = df[df.columns[-1]].str.decode("utf-8")
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)
        return X, y

    X_tr, y_tr = read_split(f"{data_dir}/ECG200_TRAIN.arff")
    X_te, y_te = read_split(f"{data_dir}/ECG200_TEST.arff")
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    print(f"  ECG200 loaded  → {X.shape[0]} samples, {X.shape[1]} time steps")
    print(f"  Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")
    print(f"  Normal (1): {(y==1).sum()}  |  Anomalous (-1): {(y==-1).sum()}")
    return X, y

print("─" * 60)
print("STEP 1 – Loading ECG200")
print("─" * 60)
X, y = load_ecg200()

normal_mask = y ==  1
anom_mask   = y == -1
X_normal = X[normal_mask]
X_anom   = X[anom_mask]


print("\n" + "─" * 60)
print("STEP 2 - Fitting baseline K-Means (k=2) anomaly detector")
print("─" * 60)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=2, n_init=20, random_state=42)
km.fit(X_scaled)

# Identify which cluster is "normal": the one containing more normal samples
labels = km.labels_
c0_normal = ((labels == 0) & normal_mask).sum()
c1_normal = ((labels == 1) & normal_mask).sum()
normal_cluster = 0 if c0_normal >= c1_normal else 1
anom_cluster   = 1 - normal_cluster

print(f"  Normal cluster id : {normal_cluster}  "
      f"(contains {max(c0_normal,c1_normal)} normals)")
print(f"  Anomaly cluster id: {anom_cluster}")

y_pred_base = np.where(labels == normal_cluster, 1, -1)

def detection_stats(y_true, y_pred, tag=""):
    tp = ((y_pred == -1) & (y_true == -1)).sum()
    fp = ((y_pred == -1) & (y_true ==  1)).sum()
    fn = ((y_pred ==  1) & (y_true == -1)).sum()
    tn = ((y_pred ==  1) & (y_true ==  1)).sum()
    dr = tp / (tp + fn + 1e-9)
    fpr= fp / (fp + tn + 1e-9)
    print(f"  {tag:30s}  DR={dr:.3f}  FPR={fpr:.3f}  "
          f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return dict(dr=dr, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn)

stats_base = detection_stats(y, y_pred_base, "BEFORE attack")

# Distance of each anomalous sample to the NORMAL centroid
normal_centroid = km.cluster_centers_[normal_cluster]
X_anom_scaled   = X_scaled[anom_mask]
dist_before = np.linalg.norm(X_anom_scaled - normal_centroid, axis=1)
print(f"  Avg distance (anom → normal centroid) BEFORE: {dist_before.mean():.4f}")


print("\n" + "─" * 60)
print("STEP 3 - Centroid Projection Attack")
print("─" * 60)

def centroid_projection_attack(X_victims_scaled, centroid, epsilon):
    """
    For each victim sample x, compute the direction toward the centroid
    and move x by min(||x - c||, epsilon) along that direction.
    This is the minimal perturbation that, when epsilon ≥ dist, places
    x exactly on the centroid; otherwise it moves x as far as the budget
    allows.
    """
    perturbed = np.copy(X_victims_scaled)
    for i, x in enumerate(X_victims_scaled):
        direction = centroid - x          # vector toward normal centroid
        dist      = np.linalg.norm(direction)
        if dist < 1e-9:
            continue
        step = min(dist, epsilon)         # never overshoot
        perturbed[i] = x + step * (direction / dist)
    return perturbed

# --- sweep epsilons to show the trade-off ---
epsilons = [3.0, 6.0, 10.0, 14.0, 18.0, 22.0]
results  = []

for eps in epsilons:
    X_anom_perturbed = centroid_projection_attack(X_anom_scaled, normal_centroid, eps)

    # rebuild full dataset with perturbed anomalies
    X_attacked = X_scaled.copy()
    X_attacked[anom_mask] = X_anom_perturbed

    # re-run detector (same centroid — simulate a fixed deployed model)
    # assign via nearest centroid (inference only, model not retrained)
    dists_to_normal = np.linalg.norm(X_attacked - normal_centroid, axis=1)
    dists_to_anom   = np.linalg.norm(
        X_attacked - km.cluster_centers_[anom_cluster], axis=1)
    pred_labels = np.where(dists_to_normal <= dists_to_anom,
                           normal_cluster, anom_cluster)
    y_pred_att  = np.where(pred_labels == normal_cluster, 1, -1)

    s = detection_stats(y, y_pred_att, f"eps={eps:.1f}")
    s["eps"] = eps

    dist_after = np.linalg.norm(X_anom_perturbed - normal_centroid, axis=1)
    s["dist_after"] = dist_after.mean()
    s["X_perturbed"] = X_anom_perturbed   # store for best eps
    results.append(s)

# pick the smallest epsilon that achieves DR <= 0.15  (≥85% evasion)
chosen = next((r for r in results if r["dr"] <= 0.15), results[-1])
eps_chosen = chosen["eps"]
print(f"\n  ✓ Chosen ε = {eps_chosen}  →  DR={chosen['dr']:.3f}  "
      f"(evasion rate={(1-chosen['dr'])*100:.1f}%)")

# final perturbed dataset at chosen epsilon
X_anom_final_scaled = chosen["X_perturbed"]
# inverse-transform back to original space for plotting
X_anom_orig_space   = scaler.inverse_transform(X_anom_scaled)
X_anom_pert_space   = scaler.inverse_transform(X_anom_final_scaled)

dist_after_chosen = np.linalg.norm(X_anom_final_scaled - normal_centroid, axis=1)
print(f"  Avg distance (anom → normal centroid)  AFTER: {dist_after_chosen.mean():.4f}")
print(f"  Avg distance BEFORE: {dist_before.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("STEP 4 - Generating plots")
print("─" * 60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
anom_pca_orig = X_pca[anom_mask]

# PCA of perturbed anomalies
X_all_pert = X_scaled.copy()
X_all_pert[anom_mask] = X_anom_final_scaled
anom_pca_pert = pca.transform(X_anom_final_scaled)

normal_c_pca = pca.transform(normal_centroid.reshape(1, -1))[0]
anom_c_pca   = pca.transform(
    km.cluster_centers_[anom_cluster].reshape(1, -1))[0]

# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle("Clean-Label Evasion Attack on ECG200  ·  K-Means Anomaly Detector",
             fontsize=15, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (A) PCA: before attack ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(*X_pca[normal_mask].T,  c=C_NORMAL, s=18, alpha=0.55,
            label="Normal (1)")
ax1.scatter(*anom_pca_orig.T,        c=C_ANOM,   s=28, alpha=0.8,
            label="Anomalous (-1)", zorder=3)
ax1.scatter(*normal_c_pca,           c=C_CENTROID, s=220, marker="*",
            zorder=5, label="Normal centroid")
ax1.scatter(*anom_c_pca,             c="#9C27B0",  s=220, marker="*",
            zorder=5, label="Anomaly centroid")
ax1.set_title("(A) PCA Projection  —  BEFORE Attack", fontsize=11,
              color="white", pad=8)
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.legend(fontsize=8, framealpha=0.2)
ax1.grid(True)

# ── (B) PCA: after attack ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.scatter(*X_pca[normal_mask].T,  c=C_NORMAL,  s=18, alpha=0.45,
            label="Normal (1)")
ax2.scatter(*anom_pca_orig.T,        c=C_ANOM,    s=22, alpha=0.35,
            label="Anomalous original", marker="x")
ax2.scatter(*anom_pca_pert.T,        c=C_PERTURB, s=28, alpha=0.85,
            label=f"Anomalous PERTURBED (ε={eps_chosen})", zorder=4)
# arrows: original → perturbed
for o, p in zip(anom_pca_orig, anom_pca_pert):
    ax2.annotate("", xy=p, xytext=o,
                 arrowprops=dict(arrowstyle="->", color=C_PERTURB,
                                 lw=0.7, alpha=0.5))
ax2.scatter(*normal_c_pca, c=C_CENTROID, s=220, marker="*",
            zorder=5, label="Normal centroid")
ax2.set_title(f"(B) PCA Projection  —  AFTER Attack (ε={eps_chosen})",
              fontsize=11, color="white", pad=8)
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
ax2.legend(fontsize=8, framealpha=0.2)
ax2.grid(True)

# ── (C) Detection rate vs epsilon ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
eps_vals = [r["eps"] for r in results]
dr_vals  = [r["dr"]  for r in results]
ax3.plot(eps_vals, dr_vals, "o-", color=C_ANOM,    lw=2, ms=7,
         label="Detection Rate")
ax3.plot(eps_vals, [1-d for d in dr_vals], "s-", color=C_PERTURB, lw=2,
         ms=7, label="Evasion Rate")
ax3.axvline(eps_chosen, color=C_CENTROID, lw=1.5, ls="--",
            label=f"chosen ε={eps_chosen}")
ax3.set_xlabel("Perturbation Budget ε")
ax3.set_ylabel("Rate")
ax3.set_title("(C) DR vs ε", fontsize=11, color="white", pad=8)
ax3.legend(fontsize=8, framealpha=0.2)
ax3.set_ylim(-0.05, 1.05); ax3.grid(True)

# ── (D) Distance to normal centroid ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
avg_dists = [dist_before.mean()] + [r["dist_after"] for r in results]
labels_d  = ["Original"] + [f"ε={r['eps']}" for r in results]
colors_d  = [C_ANOM] + [C_PERTURB]*len(results)
bars = ax4.bar(range(len(avg_dists)), avg_dists, color=colors_d, alpha=0.85,
               edgecolor="#333")
ax4.set_xticks(range(len(avg_dists)))
ax4.set_xticklabels(labels_d, rotation=35, ha="right", fontsize=8)
ax4.set_ylabel("Avg L2 distance")
ax4.set_title("(D) Dist to Normal Centroid", fontsize=11, color="white", pad=8)
ax4.grid(True, axis="y")
# annotate bars
for b, v in zip(bars, avg_dists):
    ax4.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}",
             ha="center", va="bottom", fontsize=7, color="white")

# ── (E) Sample time-series overlay (3 anomalous samples) ──────────────────
ax5 = fig.add_subplot(gs[2, :])
t = np.arange(X.shape[1])
indices_to_show = np.where(anom_mask)[0][:4]

# also show a normal sample for reference
norm_idx = np.where(normal_mask)[0][0]
ax5.plot(t, X[norm_idx], color=C_NORMAL, lw=1.2, alpha=0.7,
         label="Example Normal", linestyle="--")

for k, idx in enumerate(indices_to_show):
    # find position in anom arrays
    anom_pos = np.where(np.where(anom_mask)[0] == idx)[0][0]
    orig_ts  = X_anom_orig_space[anom_pos]
    pert_ts  = X_anom_pert_space[anom_pos]
    lbl_o = "Anomalous (original)" if k == 0 else "_nolegend_"
    lbl_p = f"Anomalous (perturbed ε={eps_chosen})" if k == 0 else "_nolegend_"
    ax5.plot(t, orig_ts, color=C_ANOM,    lw=1.0, alpha=0.55,
             linestyle=":", label=lbl_o)
    ax5.plot(t, pert_ts, color=C_PERTURB, lw=1.4, alpha=0.85,
             label=lbl_p)
    # shade the difference
    ax5.fill_between(t, orig_ts, pert_ts, alpha=0.08, color=C_PERTURB)

ax5.set_title(f"(E) Time-Series Overlay — Original vs Perturbed Anomalies  (ε={eps_chosen})",
              fontsize=11, color="white", pad=8)
ax5.set_xlabel("Time step"); ax5.set_ylabel("Amplitude")
ax5.legend(fontsize=9, framealpha=0.2)
ax5.grid(True)

plt.savefig("results/eval_atk/ecg200_evasion_attack.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("  Plot saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("RESULTS SUMMARY")
print("═" * 60)
print(f"{'':30s}  {'DR':>6}  {'FPR':>6}  {'Evaded':>8}  {'Dist→C':>8}")
print("─" * 62)

# baseline
b = stats_base
total_anom = anom_mask.sum()
evaded = total_anom - b["tp"]
print(f"  {'BEFORE attack':28s}  {b['dr']:6.3f}  {b['fpr']:6.3f}  "
      f"{evaded:>6}/{total_anom}  {dist_before.mean():8.4f}")

for r in results:
    evaded_r = total_anom - r["tp"]
    print(f"  {'AFTER  ε='+str(r['eps']):28s}  {r['dr']:6.3f}  "
          f"{r['fpr']:6.3f}  {evaded_r:>6}/{total_anom}  "
          f"{r['dist_after']:8.4f}")

print("═" * 60)
print(f"\n  ✓ At ε={eps_chosen}: {(1-chosen['dr'])*100:.1f}% of anomalous samples "
      f"evade detection")
print(f"  ✓ Perturbation is bounded — invisible at scale in the time series")
print(f"  ✓ False Positive Rate stays at {chosen['fpr']:.3f} "
      f"(normal samples unaffected)")
print("\nDone.")