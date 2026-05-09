"""
Clean-Label Evasion: ECG200 vs IForest | LOF | DBSCAN
======================================================
Attack: Projected Gradient Descent (PGD) with batched finite-difference
        gradient estimation — O(2F) score calls per step, F=96 features.

For each anomalous sample x_0, we solve:
    min_{x} score_k(x)   s.t.  ||x - x_0||_2 <= eps
via projected gradient descent:
    x_{t+1} = Pi_{B(x_0, eps)} [ x_t - lr * grad_score(x_t) ]

The gradient is estimated via central finite differences, batching all
N=67 samples in a single vectorised call per dimension, giving O(2*F*steps)
total score evaluations — tractable for F=96, N=67.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, pandas as pd
from scipy.io import arff
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Palette ───────────────────────────────────────────────────────────────────
DET_COLORS = {
    "IForest": "#FF9800",
    "LOF":     "#9C27B0",
    "DBSCAN":  "#00BCD4",
}
C_NORMAL   = "#2196F3"
C_ANOM     = "#F44336"
C_CENTROID = "#FFD700"
BG, GRID   = "#0d1117", "#1a1f2e"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor": BG,
    "axes.edgecolor":  "#2a2f3e", "axes.labelcolor": "#b0b8cc",
    "xtick.color": "#8890a0",  "ytick.color": "#8890a0",
    "text.color":  "#dde3f0",  "grid.color": GRID,
    "grid.linestyle": "--",    "grid.alpha": 0.5,
    "font.family": "monospace","legend.framealpha": 0.15,
    "legend.edgecolor": "#333",
})

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_ecg200(d="data/ECG200"):
    def read(p):
        data, _ = arff.loadarff(p)
        df = pd.DataFrame(data)
        df[df.columns[-1]] = df[df.columns[-1]].str.decode("utf-8")
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(int)
    Xt, yt = read(f"{d}/ECG200_TRAIN.arff")
    Xe, ye = read(f"{d}/ECG200_TEST.arff")
    return np.vstack([Xt, Xe]), np.concatenate([yt, ye])

print("=" * 65)
print("  GRADIENT-BASED EVASION  ·  ECG200  ·  IForest | LOF | DBSCAN")
print("=" * 65)

X_raw, y = load_ecg200()
scaler   = StandardScaler()
X        = scaler.fit_transform(X_raw)

nm, am   = (y == 1), (y == -1)
X_normal = X[nm]
X_anom   = X[am]           # (67, 96) — the victims
n_a, n_f = X_anom.shape
print(f"\n  Samples: {len(y)}  |  Normal: {nm.sum()}  |  Anomalous: {am.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DETECTORS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fitting detectors ──────────────────────────────────────────────")

# Isolation Forest — tree-based, non-parametric
ifo = IsolationForest(n_estimators=100, contamination=0.335,
                      max_samples="auto", random_state=42)
ifo.fit(X)

# LOF — density-ratio based, semi-supervised on normal samples
# Using novelty=True so we can score unseen points during the attack
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.335, novelty=True)
lof.fit(X_normal)

# DBSCAN — fit on full data; anomaly score = dist to nearest core point
db = DBSCAN(eps=3.5, min_samples=5).fit(X)
core_pts = X[db.core_sample_indices_]
nn_core  = NearestNeighbors(n_neighbors=1).fit(core_pts)

# ── Anomaly score functions (higher → more anomalous) ─────────────────────────
def s_ifo(Xq): return -ifo.decision_function(Xq)
def s_lof(Xq): return -lof.decision_function(Xq)
def s_db(Xq):  return nn_core.kneighbors(Xq)[0].ravel()

DETECTORS = {"IForest": s_ifo, "LOF": s_lof, "DBSCAN": s_db}

# ── Calibrate thresholds at 33.5% contamination ───────────────────────────────
THRESH = {n: np.percentile(fn(X), 100 * 0.665) for n, fn in DETECTORS.items()}

def flagged(Xq, name):
    """Boolean mask: True if sample is detected as anomalous."""
    return DETECTORS[name](Xq) > THRESH[name]

def dr(Xq, name):
    """Detection rate: fraction of anomalous samples caught."""
    return flagged(Xq, name).mean()

print(f"\n  {'Detector':10s}  {'Threshold':>10s}  {'Baseline DR':>12s}")
for n in DETECTORS:
    print(f"  {n:10s}  {THRESH[n]:10.4f}  {dr(X_anom, n):12.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GRADIENT ESTIMATION — BATCHED FINITE DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════
def batch_fd_gradient(score_fn, Xv, delta=0.15):
    """
    Estimate ∂score/∂x for all N samples simultaneously.

    For each feature dimension i (out of F=96):
        Xp = Xv with column i shifted +delta
        Xm = Xv with column i shifted -delta
        G[:, i] = (score(Xp) - score(Xm)) / (2*delta)

    Cost: 2*F score calls total (NOT 2*F*N).
    For F=96, N=67: 192 calls per gradient step.
    """
    N, F = Xv.shape
    G    = np.zeros((N, F))
    for i in range(F):
        Xp = Xv.copy(); Xp[:, i] += delta
        Xm = Xv.copy(); Xm[:, i] -= delta
        G[:, i] = (score_fn(Xp) - score_fn(Xm)) / (2 * delta)
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PROJECTED GRADIENT DESCENT (PGD) ATTACK
# ══════════════════════════════════════════════════════════════════════════════
def pgd_attack(score_fn, X_victims, eps,
               lr=0.5, steps=25, lr_decay=0.96, delta=0.15, verbose=False):
    """
    Minimise score_fn(x) subject to ||x - x0||_2 <= eps.

    Algorithm:
      1. Estimate gradient G = ∂score/∂x via batched FD
      2. Take a normalised gradient step (sign-of-gradient style)
      3. Project back onto the L2 epsilon-ball around x0

    Normalising the step by the gradient norm makes the effective
    learning rate invariant to score magnitude — important when
    comparing across detectors with different score scales.
    """
    X0 = X_victims.copy()     # anchor — never changes
    Xp = X_victims.copy()     # current iterate

    score_history = []

    for step in range(steps):
        G  = batch_fd_gradient(score_fn, Xp, delta=delta)
        Gn = np.linalg.norm(G, axis=1, keepdims=True).clip(min=1e-9)
        Xp = Xp - lr * G / Gn           # normalised gradient descent step

        # L2 ball projection: if ||x - x0|| > eps, rescale to boundary
        diff = Xp - X0
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        Xp   = np.where(dist > eps,
                        X0 + eps * diff / dist,
                        Xp)

        lr  *= lr_decay
        avg_score = score_fn(Xp).mean()
        score_history.append(avg_score)

        if verbose and (step % 5 == 0 or step == steps - 1):
            print(f"    step {step+1:3d}  avg_score={avg_score:.4f}  lr={lr:.4f}")

    return Xp, score_history


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RUN ATTACKS
# ══════════════════════════════════════════════════════════════════════════════
EPS        = 10.0
ATK_STEPS  = 20

print(f"\n── PGD Attack  (ε={EPS}, {ATK_STEPS} steps, batched FD gradient) ─────────")
print(f"  {'Attack target':15s}  " +
      "  ".join(f"{'DR→'+n:>10s}" for n in DETECTORS))
print("  " + "-" * 55)

X_attacked = {}      # keyed by target detector name
score_histories = {}

for target, score_fn in DETECTORS.items():
    Xp, hist = pgd_attack(score_fn, X_anom, eps=EPS,
                           lr=0.5, steps=ATK_STEPS, verbose=False)
    X_attacked[target] = Xp
    score_histories[target] = hist
    row = "  ".join(f"{dr(Xp, n):10.3f}" for n in DETECTORS)
    print(f"  Gradient→{target:10s}  {row}")

print()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ε-SWEEP  (per detector, to produce trade-off curves)
# ══════════════════════════════════════════════════════════════════════════════
print("── ε-sweep (15 steps each, all detectors) ─────────────────────────")
EPS_VALS = [2, 5, 8, 10, 16]

# dr_sweep[target_name][detector_name] = list of DRs over eps_vals
dr_sweep = {t: {d: [] for d in DETECTORS} for t in DETECTORS}

for target, score_fn in DETECTORS.items():
    print(f"  Sweeping target={target} …", end="", flush=True)
    for eps in EPS_VALS:
        Xp, _ = pgd_attack(score_fn, X_anom, eps=eps,
                            lr=0.5, steps=10, verbose=False)
        for det in DETECTORS:
            dr_sweep[target][det].append(dr(Xp, det))
    print(" done")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DETECTION RATE SUMMARY  (lower = better evasion)")
print("=" * 65)
print(f"  {'Attack':22s}" + "".join(f"  {n:>10s}" for n in DETECTORS))
print("  " + "-" * 57)

summary_rows = {"Baseline": {n: dr(X_anom, n) for n in DETECTORS}}
for t in DETECTORS:
    summary_rows[f"Gradient→{t}"] = {n: dr(X_attacked[t], n) for n in DETECTORS}

for rname, rdata in summary_rows.items():
    vals = "".join(f"  {rdata[n]:10.3f}" for n in DETECTORS)
    print(f"  {rname:22s}{vals}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Generating figures ──────────────────────────────────────────────")

pca = PCA(n_components=2, random_state=42).fit(X)
P   = lambda Xq: pca.transform(Xq)

pn   = P(X_normal)
pa   = P(X_anom)

# ─── FIGURE 1: Main results (3×3) ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle(
    "Gradient-Based Clean-Label Evasion  ·  ECG200  ·  IForest | LOF | DBSCAN",
    fontsize=14, fontweight="bold", color="white", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.32)

# ── Helper: PCA scatter panel ──────────────────────────────────────────────────
def pca_panel(ax, Xpert, det_name, atk_label, title):
    pp = P(Xpert)
    ax.scatter(*pn.T, c=C_NORMAL, s=13, alpha=0.35, label="Normal")
    ax.scatter(*pa.T, c=C_ANOM,   s=18, alpha=0.30, marker="x",
               label="Anom (orig)")
    ax.scatter(*pp.T, c=DET_COLORS[det_name], s=26, alpha=0.88,
               label=atk_label, zorder=4)
    for o, p in zip(pa, pp):
        ax.annotate("", xy=p, xytext=o,
                    arrowprops=dict(arrowstyle="->",
                                    color=DET_COLORS[det_name],
                                    lw=0.55, alpha=0.38))
    ax.set_title(title, fontsize=10, color="white", pad=6)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True)
    # DR annotations as subtitle
    dr_str = "  ".join(f"{n}:{dr(Xpert, n):.2f}" for n in DETECTORS)
    ax.set_xlabel(f"DR after attack:  {dr_str}", fontsize=7.5)

# Row 0: Baseline + score convergence curves
ax_base = fig.add_subplot(gs[0, 0])
ax_base.scatter(*pn.T, c=C_NORMAL, s=13, alpha=0.4,  label="Normal")
ax_base.scatter(*pa.T, c=C_ANOM,   s=22, alpha=0.88, label="Anomalous")
ax_base.set_title("(A) Baseline — No Attack", fontsize=10, color="white", pad=6)
ax_base.legend(fontsize=8)
ax_base.grid(True)
bl_str = "  ".join(f"{n}:{dr(X_anom, n):.2f}" for n in DETECTORS)
ax_base.set_xlabel(f"DR baseline:  {bl_str}", fontsize=7.5)

# Score convergence curves (all three targets)
ax_conv = fig.add_subplot(gs[0, 1])
for target, hist in score_histories.items():
    ax_conv.plot(range(1, len(hist)+1), hist,
                 color=DET_COLORS[target], lw=2, label=f"→{target}")
ax_conv.set_title("(B) Anomaly Score During PGD", fontsize=10, color="white", pad=6)
ax_conv.set_xlabel("PGD step"); ax_conv.set_ylabel("Avg anomaly score")
ax_conv.legend(fontsize=8); ax_conv.grid(True)

# DR Heatmap
ax_heat = fig.add_subplot(gs[0, 2])
rlabels  = list(summary_rows.keys())
dlabels  = list(DETECTORS.keys())
mat      = np.array([[summary_rows[r][n] for n in dlabels] for r in rlabels])
im = ax_heat.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax_heat.set_xticks(range(len(dlabels)))
ax_heat.set_xticklabels(dlabels, fontsize=9)
ax_heat.set_yticks(range(len(rlabels)))
ax_heat.set_yticklabels(rlabels, fontsize=8)
ax_heat.set_title("(C) DR Heatmap  (green=evaded)", fontsize=10,
                   color="white", pad=6)
for i in range(len(rlabels)):
    for j in range(len(dlabels)):
        ax_heat.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=9,
                     color="white" if mat[i,j] > 0.5 else "black",
                     fontweight="bold")
plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

# Row 1: PCA scatter per target detector
for col, target in enumerate(DETECTORS):
    ax = fig.add_subplot(gs[1, col])
    pca_panel(ax, X_attacked[target], target,
              f"Perturbed (→{target})",
              f"(D{col+1}) Gradient→{target}  ε={EPS}")

# Row 2: ε trade-off curves (one per target detector)
for col, target in enumerate(DETECTORS):
    ax = fig.add_subplot(gs[2, col])
    for det in DETECTORS:
        ax.plot(EPS_VALS, dr_sweep[target][det], "o-",
                color=DET_COLORS[det], lw=2, ms=6,
                label=det,
                linestyle="--" if det != target else "-",
                alpha=0.6 if det != target else 1.0)
    ax.axvline(EPS, color="white", lw=1, ls=":", alpha=0.5, label=f"ε={EPS}")
    ax.set_title(f"(E{col+1}) ε-Trade-off  target={target}",
                  fontsize=10, color="white", pad=6)
    ax.set_xlabel("Perturbation Budget ε"); ax.set_ylabel("Detection Rate")
    ax.legend(fontsize=8); ax.grid(True); ax.set_ylim(-0.05, 1.0)
    # annotate targeted DR at chosen eps
    idx_chosen = EPS_VALS.index(EPS)
    targeted_val = dr_sweep[target][target][idx_chosen]
    ax.annotate(f"  targeted\n  DR={targeted_val:.2f}",
                xy=(EPS, targeted_val),
                xytext=(EPS + 1, targeted_val + 0.15),
                fontsize=8, color=DET_COLORS[target],
                arrowprops=dict(arrowstyle="->",
                                color=DET_COLORS[target], lw=1))

plt.savefig("results/grad_evasion/ecg200_gradient_evasion.png",
            dpi=145, bbox_inches="tight", facecolor=BG)
print("  Main figure saved.")


# ─── FIGURE 2: Signal-level view ──────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig2.suptitle(
    "ECG Signal — Original Anomalies vs Gradient-Perturbed  (5 samples each)",
    fontsize=12, color="white")
fig2.patch.set_facecolor(BG)

t_axis   = np.arange(X_raw.shape[1])
X_anom_r = scaler.inverse_transform(X_anom)
X_norm_r = scaler.inverse_transform(X_normal)

for ax, target in zip(axes, DETECTORS):
    Xpert_r = scaler.inverse_transform(X_attacked[target])

    # normal reference
    ax.plot(t_axis, X_norm_r[0], color=C_NORMAL, lw=1.5, ls="--",
            alpha=0.6, label="Normal example")

    for k in range(5):
        ax.plot(t_axis, X_anom_r[k], color=C_ANOM, lw=0.9, alpha=0.30,
                ls=":", label="Anom original" if k == 0 else "_")
        ax.plot(t_axis, Xpert_r[k],  color=DET_COLORS[target], lw=1.2,
                alpha=0.85, label=f"Perturbed (→{target})" if k == 0 else "_")
        ax.fill_between(t_axis, X_anom_r[k], Xpert_r[k],
                         alpha=0.07, color=DET_COLORS[target])

    ax.set_title(f"Gradient→{target}  |  "
                 f"DR: {dr(X_anom, target):.2f}→{dr(X_attacked[target], target):.2f}",
                 fontsize=10, color="white")
    ax.set_xlabel("Time step"); ax.set_ylabel("Amplitude")
    ax.set_facecolor(BG); ax.grid(True); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/grad_evasion/ecg200_signal_overlay.png",
            dpi=145, bbox_inches="tight", facecolor=BG)
print("  Signal overlay figure saved.")
print("\nAll done.")