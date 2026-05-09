"""
Advanced Clean-Label Evasion: ECG200 vs Multi-Detector Anomaly Detection
=========================================================================
Detectors : K-Means | Isolation Forest | LOF | DBSCAN
Attacks   : 1. Centroid Projection  (naive)
            2. Score-Gradient PGD   (per-detector, SPSA gradient estimate)
            3. Ensemble PGD         (simultaneous evasion, all detectors)

Gradient estimated via SPSA (Simultaneous Perturbation Stochastic
Approximation): only 2 score calls per step regardless of dimensionality,
making it tractable for 96-dimensional time series.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)

C = dict(normal="#2196F3", anom="#F44336", proj="#FF9800",
         grad="#9C27B0", ens="#4CAF50", centroid="#FFD700")
BG = "#0d1117"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc", "xtick.color": "#aaa", "ytick.color": "#aaa",
    "text.color": "#eee", "grid.color": "#1e2130", "grid.linestyle": "--",
    "grid.alpha": 0.4, "font.family": "monospace", "legend.framealpha": 0.15,
})

# ── 1. DATA ───────────────────────────────────────────────────────────────────
def load_ecg200(d="data/ECG200"):
    def read(p):
        data, _ = arff.loadarff(p)
        df = pd.DataFrame(data)
        df[df.columns[-1]] = df[df.columns[-1]].str.decode("utf-8")
        return df.iloc[:,:-1].values.astype(float), df.iloc[:,-1].values.astype(int)
    Xt,yt = read(f"{d}/ECG200_TRAIN.arff")
    Xe,ye = read(f"{d}/ECG200_TEST.arff")
    return np.vstack([Xt,Xe]), np.concatenate([yt,ye])

print("Loading ECG200 …")
X_raw, y = load_ecg200()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
nm, am = y==1, y==-1
X_anom = X[am]
n_a, n_f = X_anom.shape   # (67, 96)

# ── 2. DETECTORS ──────────────────────────────────────────────────────────────
print("Fitting detectors …")
km = KMeans(n_clusters=2, n_init=20, random_state=42).fit(X)
nc = 0 if ((km.labels_==0)&nm).sum() >= ((km.labels_==1)&nm).sum() else 1
normal_c = km.cluster_centers_[nc]
anom_c   = km.cluster_centers_[1-nc]

ifo = IsolationForest(n_estimators=200, contamination=0.335, random_state=42)
ifo.fit(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.335, novelty=True)
lof.fit(X[nm])

dbscan = DBSCAN(eps=3.5, min_samples=5).fit(X)
nn_core = NearestNeighbors(n_neighbors=1).fit(X[dbscan.core_sample_indices_])

def s_km(Xq):  return np.linalg.norm(Xq - normal_c, axis=1)
def s_if(Xq):  return -ifo.decision_function(Xq)
def s_lof(Xq): return -lof.decision_function(Xq)
def s_db(Xq):  return nn_core.kneighbors(Xq)[0].ravel()

SCORE_FNS = {"KMeans": s_km, "IForest": s_if, "LOF": s_lof, "DBSCAN": s_db}
THRESH = {n: np.percentile(fn(X), 100*0.665) for n, fn in SCORE_FNS.items()}

def dr(Xq, name): return (SCORE_FNS[name](Xq) > THRESH[name]).mean()

print(f"\n  {'Detector':10s}  {'Baseline DR':>12s}")
for n in SCORE_FNS: print(f"  {n:10s}  {dr(X_anom, n):12.3f}")

# ── 3. ATTACK 1: CENTROID PROJECTION ─────────────────────────────────────────
def proj_attack(Xv, centroid, eps):
    diff = centroid - Xv
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    step = np.minimum(dist, eps)
    return Xv + step * diff / (dist + 1e-9)

EPS = 10.0
X_proj = proj_attack(X_anom, normal_c, EPS)
print(f"\nAttack 1 — Centroid Projection (ε={EPS})")
for n in SCORE_FNS: print(f"  {n:10s}  DR={dr(X_proj,n):.3f}")

# ── 4. SPSA GRADIENT ESTIMATOR ───────────────────────────────────────────────
"""
SPSA estimates the gradient in 2 function calls regardless of dimension:
  ĝ(x) = [f(x+δΔ) − f(x−δΔ)] / (2δ) * 1/Δ
where Δ ∈ {±1}^d is a random Rademacher vector.
This is an unbiased estimator of the true gradient direction.
We average over K=8 SPSA samples per step for stability.
"""

def spsa_grad(score_fn, Xv, delta=0.15, K=8):
    """
    SPSA gradient estimate for N samples simultaneously.
    Returns gradient matrix (N, F). Cost: 2*K score calls.
    """
    N, F = Xv.shape
    G = np.zeros((N, F))
    for _ in range(K):
        Delta = np.sign(np.random.randn(N, F))       # Rademacher perturbation
        Xp = Xv + delta * Delta
        Xm = Xv - delta * Delta
        g_scalar = (score_fn(Xp) - score_fn(Xm)) / (2 * delta)  # (N,)
        G += g_scalar[:, None] * (1.0 / Delta)       # SPSA update
    return G / K

def pgd_spsa(score_fn, Xv, eps, lr=1.2, steps=30, decay=0.95, K=8):
    """Projected Gradient Descent with SPSA gradient estimates."""
    X0 = Xv.copy()
    Xp = Xv.copy()
    for step in range(steps):
        G  = spsa_grad(score_fn, Xp, K=K)
        Gn = np.linalg.norm(G, axis=1, keepdims=True)
        Xp -= lr * G / (Gn + 1e-9)
        # project onto ε-ball
        diff = Xp - X0
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        Xp = np.where(dist > eps, X0 + eps * diff / (dist + 1e-9), Xp)
        lr *= decay
    return Xp

# ── 5. ATTACK 2: PER-DETECTOR GRADIENT PGD ────────────────────────────────────
print(f"\nAttack 2 — SPSA-PGD per detector (ε={EPS}, 30 steps)")
X_grad = {}
for name, fn in SCORE_FNS.items():
    X_grad[name] = pgd_spsa(fn, X_anom, EPS, steps=30)
    row = "  ".join(f"{n}={dr(X_grad[name],n):.2f}" for n in SCORE_FNS)
    print(f"  → {name:10s}  {row}")

# ── 6. ATTACK 3: ENSEMBLE PGD ────────────────────────────────────────────────
WEIGHTS = {"KMeans": 1.0, "IForest": 1.5, "LOF": 1.5, "DBSCAN": 1.0}

def ensemble_score(Xv):
    total = np.zeros(len(Xv))
    for name, fn in SCORE_FNS.items():
        total += WEIGHTS[name] * fn(Xv) / (THRESH[name] + 1e-9)
    return total

print(f"\nAttack 3 — Ensemble SPSA-PGD (ε={EPS}, 35 steps)")
X_ens = pgd_spsa(ensemble_score, X_anom, EPS, lr=1.2, steps=35, K=8)
for n in SCORE_FNS: print(f"  {n:10s}  DR={dr(X_ens,n):.3f}")

# ── 7. ε-SWEEP ────────────────────────────────────────────────────────────────
print("\nε-sweep (ensemble, 20 steps each) …")
eps_vals = [2, 5, 8, 10, 13, 16]
sweep = {n: [] for n in SCORE_FNS}
sweep_avg = []
for eps in eps_vals:
    Xp = pgd_spsa(ensemble_score, X_anom, eps, lr=1.0, steps=20, K=6)
    vals = [dr(Xp, n) for n in SCORE_FNS]
    for n, v in zip(SCORE_FNS, vals): sweep[n].append(v)
    sweep_avg.append(np.mean(vals))
    print(f"  ε={eps:3d}  avg_DR={sweep_avg[-1]:.3f}  "
          + "  ".join(f"{n}={v:.2f}" for n, v in zip(SCORE_FNS, vals)))

# ── 8. SUMMARY TABLE ──────────────────────────────────────────────────────────
attack_rows = {
    "Baseline":        {n: dr(X_anom, n) for n in SCORE_FNS},
    "Centroid Proj":   {n: dr(X_proj, n) for n in SCORE_FNS},
    "Grad→KMeans":     {n: dr(X_grad["KMeans"],  n) for n in SCORE_FNS},
    "Grad→IForest":    {n: dr(X_grad["IForest"], n) for n in SCORE_FNS},
    "Grad→LOF":        {n: dr(X_grad["LOF"],     n) for n in SCORE_FNS},
    "Grad→DBSCAN":     {n: dr(X_grad["DBSCAN"],  n) for n in SCORE_FNS},
    "Ensemble":        {n: dr(X_ens, n) for n in SCORE_FNS},
}

print("\n" + "="*65)
print("  DETECTION RATE SUMMARY  (lower = better evasion)")
print("="*65)
print(f"  {'Attack':20s}" + "".join(f"  {n:>10s}" for n in SCORE_FNS))
print("  " + "-"*62)
for rname, rdata in attack_rows.items():
    print(f"  {rname:20s}" + "".join(f"  {rdata[n]:10.3f}" for n in SCORE_FNS))
print("="*65)

# ── 9. PLOTS ─────────────────────────────────────────────────────────────────
print("\nGenerating plots …")

pca = PCA(n_components=2, random_state=42).fit(X)
def P(Xq): return pca.transform(Xq)

pn = P(X[nm]); pa = P(X_anom); pnc = P(normal_c.reshape(1,-1))[0]

fig = plt.figure(figsize=(20, 15), facecolor=BG)
fig.suptitle(
    "Advanced Clean-Label Evasion · ECG200 · K-Means | IForest | LOF | DBSCAN",
    fontsize=13, fontweight="bold", color="white", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.30)

def scatter(ax, pert, color, title, label, show_dr=None):
    ax.scatter(*pn.T, c=C["normal"], s=12, alpha=0.35, label="Normal")
    ax.scatter(*pa.T, c=C["anom"],   s=18, alpha=0.30, marker="x", label="Anom (orig)")
    pp = P(pert)
    ax.scatter(*pp.T, c=color, s=24, alpha=0.85, label=label, zorder=4)
    for o, p in zip(pa, pp):
        ax.annotate("", xy=p, xytext=o,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.5, alpha=0.35))
    ax.scatter(*pnc, c=C["centroid"], s=200, marker="*", zorder=6)
    ax.set_title(title, fontsize=9, color="white", pad=5)
    ax.legend(fontsize=6, loc="upper right"); ax.grid(True)
    if show_dr:
        ax.set_xlabel("  ".join(f"{n}:{show_dr[n]:.2f}" for n in SCORE_FNS), fontsize=7)

# (A) Baseline
ax = fig.add_subplot(gs[0,0])
ax.scatter(*pn.T, c=C["normal"], s=12, alpha=0.4, label="Normal")
ax.scatter(*pa.T, c=C["anom"],   s=22, alpha=0.85, label="Anomalous")
ax.scatter(*pnc, c=C["centroid"], s=200, marker="*", zorder=6, label="Centroid")
ax.set_title("(A) Baseline — No Attack", fontsize=9, color="white", pad=5)
ax.legend(fontsize=7); ax.grid(True)
ax.set_xlabel("  ".join(f"{n}:{attack_rows['Baseline'][n]:.2f}" for n in SCORE_FNS), fontsize=7)

# (B) Centroid Projection
ax = fig.add_subplot(gs[0,1])
scatter(ax, X_proj, C["proj"], f"(B) Centroid Projection  ε={EPS}", "Proj",
        show_dr=attack_rows["Centroid Proj"])

# (C) Ensemble
ax = fig.add_subplot(gs[0,2])
scatter(ax, X_ens, C["ens"], f"(C) Ensemble Attack  ε={EPS}", "Ensemble",
        show_dr=attack_rows["Ensemble"])

# (D–G) Per-detector gradient
grad_cols  = [C["grad"], "#00BCD4", "#FF5722", "#8BC34A"]
positions  = [(1,0),(1,1),(1,2),(2,0)]
rkeys      = ["Grad→KMeans","Grad→IForest","Grad→LOF","Grad→DBSCAN"]
for name, pos, col, rk in zip(list(SCORE_FNS.keys()), positions, grad_cols, rkeys):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    scatter(ax, X_grad[name], col,
            f"Gradient→{name}  ε={EPS}", f"Grad→{name}",
            show_dr=attack_rows[rk])

# (H) Heatmap
ax_h = fig.add_subplot(gs[2,1])
rlabels = list(attack_rows.keys())
dlabels = list(SCORE_FNS.keys())
mat = np.array([[attack_rows[r][n] for n in dlabels] for r in rlabels])
im = ax_h.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax_h.set_xticks(range(len(dlabels))); ax_h.set_xticklabels(dlabels, fontsize=9)
ax_h.set_yticks(range(len(rlabels))); ax_h.set_yticklabels(rlabels, fontsize=8)
ax_h.set_title("(H) Detection Rate Heatmap  (green = evaded)", fontsize=9,
               color="white", pad=5)
for i in range(len(rlabels)):
    for j in range(len(dlabels)):
        ax_h.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                  fontsize=9, color="white" if mat[i,j] > 0.5 else "black")
plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

# (I) Time-series overlay
ax_ts = fig.add_subplot(gs[2,2])
t = np.arange(X_raw.shape[1])
X_ar = scaler.inverse_transform(X_anom)
X_er = scaler.inverse_transform(X_ens)
X_nr = scaler.inverse_transform(X[nm][:1])
ax_ts.plot(t, X_nr[0], color=C["normal"], lw=1.8, alpha=0.7, ls="--",
           label="Normal example")
for k in range(min(6, n_a)):
    ax_ts.plot(t, X_ar[k], color=C["anom"], lw=0.8, alpha=0.3, ls=":",
               label="Anom original" if k==0 else "_")
    ax_ts.plot(t, X_er[k], color=C["ens"],  lw=1.1, alpha=0.8,
               label="Anom + Ensemble" if k==0 else "_")
    ax_ts.fill_between(t, X_ar[k], X_er[k], alpha=0.05, color=C["ens"])
ax_ts.set_title("(I) ECG Signal: Original vs Ensemble Perturbed", fontsize=9,
                color="white", pad=5)
ax_ts.set_xlabel("Time step", fontsize=8); ax_ts.set_ylabel("Amplitude", fontsize=8)
ax_ts.legend(fontsize=7); ax_ts.grid(True)

plt.savefig("results/eval_advanced/ecg200_advanced_evasion.png",
            dpi=140, bbox_inches="tight", facecolor=BG)
print("  Main figure saved.")

# ── ε-trade-off figure ────────────────────────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig2.suptitle("Ensemble Attack — ε-Budget Trade-off · ECG200", fontsize=12,
              color="white")
fig2.patch.set_facecolor(BG)

det_colors = [C["proj"], C["grad"], C["ens"], "#00BCD4"]
for col, name in zip(det_colors, SCORE_FNS):
    ax1.plot(eps_vals, sweep[name], "o-", color=col, lw=2, ms=6, label=name)
ax1.plot(eps_vals, sweep_avg, "s--", color="white", lw=2.5, ms=8, label="Avg")
ax1.set_facecolor(BG); ax1.set_xlabel("ε"); ax1.set_ylabel("Detection Rate")
ax1.set_title("DR per Detector vs ε", color="white")
ax1.legend(fontsize=9); ax1.grid(True); ax1.set_ylim(-0.05, 1.05)

evasion = [1-d for d in sweep_avg]
ax2.fill_between(eps_vals, evasion, alpha=0.2, color=C["ens"])
ax2.plot(eps_vals, evasion, "o-", color=C["ens"], lw=2.5, ms=8)
for ex, ev in zip(eps_vals, evasion):
    ax2.annotate(f"{ev*100:.0f}%", (ex, ev), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=10, color="white")
ax2.set_facecolor(BG); ax2.set_xlabel("ε"); ax2.set_ylabel("Avg Evasion Rate")
ax2.set_title("Overall Evasion Rate vs ε", color="white")
ax2.grid(True); ax2.set_ylim(-0.05, 1.15)

plt.tight_layout()
plt.savefig("results/eval_advanced/ecg200_epsilon_tradeoff.png",
            dpi=140, bbox_inches="tight", facecolor=BG)
print("  ε trade-off figure saved.")
print("\nAll done.")