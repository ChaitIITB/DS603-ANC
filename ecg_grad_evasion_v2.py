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

DET_COLORS = {"IForest": "#FF9800", "LOF": "#9C27B0", "DBSCAN": "#00BCD4"}
C_NORMAL, C_ANOM = "#2196F3", "#F44336"
BG = "#0d1117"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#2a2f3e",
    "axes.labelcolor": "#b0b8cc", "xtick.color": "#8890a0", "ytick.color": "#8890a0",
    "text.color": "#dde3f0", "grid.color": "#1a1f2e", "grid.linestyle": "--",
    "grid.alpha": 0.5, "font.family": "monospace", "legend.framealpha": 0.15,
})

# ── Reload data & detectors (fast, no attack rerun) ───────────────────────────
def load_ecg200(d="data/ECG200"):
    def read(p):
        data, _ = arff.loadarff(p)
        df = pd.DataFrame(data)
        df[df.columns[-1]] = df[df.columns[-1]].str.decode("utf-8")
        return df.iloc[:,:-1].values.astype(float), df.iloc[:,-1].values.astype(int)
    Xt,yt=read(f"{d}/ECG200_TRAIN.arff"); Xe,ye=read(f"{d}/ECG200_TEST.arff")
    return np.vstack([Xt,Xe]), np.concatenate([yt,ye])

X_raw, y = load_ecg200()
scaler = StandardScaler(); X = scaler.fit_transform(X_raw)
nm, am = y==1, y==-1
X_normal, X_anom = X[nm], X[am]
n_a, n_f = X_anom.shape

ifo = IsolationForest(n_estimators=100, contamination=0.335, random_state=42).fit(X)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.335, novelty=True).fit(X_normal)
db  = DBSCAN(eps=3.5, min_samples=5).fit(X)
nn_core = NearestNeighbors(n_neighbors=1).fit(X[db.core_sample_indices_])

def s_ifo(Xq): return -ifo.decision_function(Xq)
def s_lof(Xq): return -lof.decision_function(Xq)
def s_db(Xq):  return nn_core.kneighbors(Xq)[0].ravel()
DETECTORS = {"IForest": s_ifo, "LOF": s_lof, "DBSCAN": s_db}
THRESH = {n: np.percentile(fn(X), 100*0.665) for n, fn in DETECTORS.items()}
def dr(Xq, name): return (DETECTORS[name](Xq) > THRESH[name]).mean()

# ── Re-run attacks at ε=5 and ε=10 ───────────────────────────────────────────
def batch_fd_gradient(score_fn, Xv, delta=0.15):
    N, F = Xv.shape; G = np.zeros((N, F))
    for i in range(F):
        Xp=Xv.copy(); Xp[:,i]+=delta
        Xm=Xv.copy(); Xm[:,i]-=delta
        G[:,i]=(score_fn(Xp)-score_fn(Xm))/(2*delta)
    return G

def pgd(score_fn, Xv, eps, steps=20, decay=0.96):
    X0=Xv.copy(); Xp=Xv.copy()
    lr = eps/steps*1.5
    hist = []
    for _ in range(steps):
        G=batch_fd_gradient(score_fn, Xp)
        Gn=np.linalg.norm(G,axis=1,keepdims=True).clip(min=1e-9)
        Xp=Xp-lr*G/Gn
        diff=Xp-X0; dist=np.linalg.norm(diff,axis=1,keepdims=True)
        Xp=np.where(dist>eps, X0+eps*diff/dist, Xp)
        lr*=decay; hist.append(score_fn(Xp).mean())
    return Xp, hist

EPS_VALS = [2, 5, 8, 10, 16]
EPS5, EPS10 = 5.0, 10.0

print("Running attacks at ε=5 and ε=10 …")
X_atk5  = {}; X_atk10 = {}; score_histories = {}
for name, fn in DETECTORS.items():
    print(f"  {name} …", end="", flush=True)
    X_atk5[name],  _    = pgd(fn, X_anom, EPS5, steps=100)
    X_atk10[name], hist = pgd(fn, X_anom, EPS10, steps=100)
    score_histories[name] = hist
    print(" done")

# ε-sweep (for trade-off curves)
print("ε-sweep …")
dr_sweep = {t: {d: [] for d in DETECTORS} for t in DETECTORS}
for target, fn in DETECTORS.items():
    for eps in EPS_VALS:
        Xp, _ = pgd(fn, X_anom, eps, steps=100)
        for det in DETECTORS:
            dr_sweep[target][det].append(dr(Xp, det))

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  DETECTION RATE SUMMARY")
print("="*65)
print(f"  {'Attack':25s}" + "".join(f"  {n:>10s}" for n in DETECTORS))
print("  "+"-"*57)
rows = {"Baseline": {n: dr(X_anom, n) for n in DETECTORS}}
for t in DETECTORS:
    rows[f"Grad→{t} ε=5"]  = {n: dr(X_atk5[t],  n) for n in DETECTORS}
    rows[f"Grad→{t} ε=10"] = {n: dr(X_atk10[t], n) for n in DETECTORS}
for rn, rd in rows.items():
    print(f"  {rn:25s}" + "".join(f"  {rd[n]:10.3f}" for n in DETECTORS))
print("="*65)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main 3×3: Baseline | Convergence | Heatmap
#                      PCA ε=10 (×3)
#                      ε-trade-off curves (×3)  with ε=5 AND ε=10 marked
# ══════════════════════════════════════════════════════════════════════════════
pca = PCA(n_components=2, random_state=42).fit(X)
P   = lambda Xq: pca.transform(Xq)
pn  = P(X_normal); pa = P(X_anom)

fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle(
    "Gradient-Based Evasion · ECG200 · IForest | LOF | DBSCAN  [ε=5 & ε=10]",
    fontsize=14, fontweight="bold", color="white", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.32)

def pca_panel(ax, Xpert, det_name, title):
    pp = P(Xpert)
    ax.scatter(*pn.T, c=C_NORMAL, s=13, alpha=0.35, label="Normal")
    ax.scatter(*pa.T, c=C_ANOM,   s=18, alpha=0.30, marker="x", label="Anom (orig)")
    ax.scatter(*pp.T, c=DET_COLORS[det_name], s=26, alpha=0.88,
               label="Perturbed", zorder=4)
    for o, p in zip(pa, pp):
        ax.annotate("", xy=p, xytext=o,
                    arrowprops=dict(arrowstyle="->", color=DET_COLORS[det_name],
                                    lw=0.5, alpha=0.35))
    ax.set_title(title, fontsize=9, color="white", pad=5)
    ax.legend(fontsize=6, loc="upper right"); ax.grid(True)
    dr_str = "  ".join(f"{n}:{dr(Xpert,n):.2f}" for n in DETECTORS)
    ax.set_xlabel(f"DR:  {dr_str}", fontsize=7)

# Row 0 ── baseline | convergence | heatmap
ax = fig.add_subplot(gs[0,0])
ax.scatter(*pn.T, c=C_NORMAL, s=13, alpha=0.4, label="Normal")
ax.scatter(*pa.T, c=C_ANOM,   s=22, alpha=0.88, label="Anomalous")
ax.set_title("(A) Baseline — No Attack", fontsize=10, color="white", pad=6)
ax.legend(fontsize=8); ax.grid(True)
bl = "  ".join(f"{n}:{dr(X_anom,n):.2f}" for n in DETECTORS)
ax.set_xlabel(f"DR:  {bl}", fontsize=7)

ax = fig.add_subplot(gs[0,1])
for target, hist in score_histories.items():
    ax.plot(range(1, len(hist)+1), hist, color=DET_COLORS[target], lw=2,
            label=f"→{target}")
ax.set_title("(B) Anomaly Score During PGD  (ε=10)", fontsize=10,
             color="white", pad=6)
ax.set_xlabel("PGD step"); ax.set_ylabel("Avg anomaly score")
ax.legend(fontsize=8); ax.grid(True)

# Heatmap — include both ε=5 and ε=10 rows
ax = fig.add_subplot(gs[0,2])
rlabels = (["Baseline"] +
           [f"→{t} ε=5"  for t in DETECTORS] +
           [f"→{t} ε=10" for t in DETECTORS])
dlabels = list(DETECTORS.keys())
hmap_rows = {"Baseline": {n: dr(X_anom,n) for n in DETECTORS}}
for t in DETECTORS:
    hmap_rows[f"→{t} ε=5"]  = {n: dr(X_atk5[t],  n) for n in DETECTORS}
    hmap_rows[f"→{t} ε=10"] = {n: dr(X_atk10[t], n) for n in DETECTORS}
mat = np.array([[hmap_rows[r][n] for n in dlabels] for r in rlabels])
im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(dlabels))); ax.set_xticklabels(dlabels, fontsize=9)
ax.set_yticks(range(len(rlabels))); ax.set_yticklabels(rlabels, fontsize=7)
ax.set_title("(C) DR Heatmap  (green=evaded)", fontsize=10, color="white", pad=5)
for i in range(len(rlabels)):
    for j in range(len(dlabels)):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if mat[i,j]>0.5 else "black", fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 1 ── PCA scatter at ε=10 (one per detector)
for col, target in enumerate(DETECTORS):
    ax = fig.add_subplot(gs[1, col])
    pca_panel(ax, X_atk10[target], target,
              f"(D{col+1}) Gradient→{target}  ε=10")

# Row 2 ── ε trade-off curves with BOTH ε=5 and ε=10 marked
EPS_MARKS = {
    5.0:  (":",  0.80, "#aaaaaa"),
    10.0: ("--", 1.00, "white"),
}
for col, target in enumerate(DETECTORS):
    ax = fig.add_subplot(gs[2, col])
    for det in DETECTORS:
        ax.plot(EPS_VALS, dr_sweep[target][det], "o-",
                color=DET_COLORS[det], lw=2, ms=6, label=det,
                linestyle="--" if det != target else "-",
                alpha=0.55 if det != target else 1.0)
    for eps_m, (ls, alpha, wcol) in EPS_MARKS.items():
        if eps_m not in EPS_VALS: continue
        ax.axvline(eps_m, color=wcol, lw=1.2, ls=ls, alpha=alpha)
        idx_m = EPS_VALS.index(eps_m)
        tv = dr_sweep[target][target][idx_m]
        # alternate annotation side to avoid overlap
        dx = 0.6 if eps_m == 5.0 else -3.8
        dy = 0.10
        ax.annotate(f"ε={int(eps_m)}\nDR={tv:.2f}",
                    xy=(eps_m, tv),
                    xytext=(eps_m + dx, tv + dy),
                    fontsize=7.5, color=DET_COLORS[target],
                    arrowprops=dict(arrowstyle="->",
                                    color=DET_COLORS[target], lw=0.9))
    ax.set_title(f"(E{col+1}) ε-Trade-off  target={target}",
                 fontsize=10, color="white", pad=5)
    ax.set_xlabel("ε"); ax.set_ylabel("Detection Rate")
    ax.legend(fontsize=7); ax.grid(True); ax.set_ylim(-0.05, 1.0)

plt.savefig("anomaly_results/grad_evasion/ecg200_gradient_evasion.png",
            dpi=145, bbox_inches="tight", facecolor=BG)
print("\n  Main figure saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Side-by-side: ε=5 vs ε=10 PCA for each detector (2 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig2.suptitle("ε=5 vs ε=10  —  PCA View per Target Detector  (ECG200)",
              fontsize=13, fontweight="bold", color="white")
fig2.patch.set_facecolor(BG)

for col, target in enumerate(DETECTORS):
    for row, (eps_val, X_atk) in enumerate([(EPS5, X_atk5), (EPS10, X_atk10)]):
        ax = axes2[row, col]
        pp = P(X_atk[target])
        ax.scatter(*pn.T, c=C_NORMAL, s=12, alpha=0.30, label="Normal")
        ax.scatter(*pa.T, c=C_ANOM,   s=16, alpha=0.28, marker="x",
                   label="Anom (orig)")
        ax.scatter(*pp.T, c=DET_COLORS[target], s=24, alpha=0.88,
                   label=f"Perturbed ε={int(eps_val)}", zorder=4)
        for o, p in zip(pa, pp):
            ax.annotate("", xy=p, xytext=o,
                        arrowprops=dict(arrowstyle="->",
                                        color=DET_COLORS[target],
                                        lw=0.5, alpha=0.30))
        dr_t = dr(X_atk[target], target)
        dr_all = "  ".join(f"{n}:{dr(X_atk[target],n):.2f}" for n in DETECTORS)
        ax.set_title(f"→{target}  ε={int(eps_val)}  |  targeted DR={dr_t:.2f}",
                     fontsize=10, color="white", pad=5)
        ax.set_xlabel(f"All DRs:  {dr_all}", fontsize=7)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True)
        ax.set_facecolor(BG)

plt.tight_layout()
plt.savefig("anomaly_results/grad_evasion/ecg200_eps5_vs_eps10.png",
            dpi=145, bbox_inches="tight", facecolor=BG)
print("  ε=5 vs ε=10 comparison figure saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Signal overlay: original | ε=5 | ε=10 per detector
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(3, 3, figsize=(18, 13), facecolor=BG)
fig3.suptitle("ECG Signal — Original vs ε=5 vs ε=10 Perturbation  (5 samples each)",
              fontsize=12, color="white")
fig3.patch.set_facecolor(BG)

t_axis   = np.arange(X_raw.shape[1])
X_anom_r = scaler.inverse_transform(X_anom)
X_norm_r = scaler.inverse_transform(X_normal)

for col, target in enumerate(DETECTORS):
    X5r  = scaler.inverse_transform(X_atk5[target])
    X10r = scaler.inverse_transform(X_atk10[target])

    for row, (Xr, eps_val, alpha_fill) in enumerate([
            (X_anom_r, "orig", 0),
            (X5r,  5,  0.10),
            (X10r, 10, 0.10)]):
        ax = axes3[row, col]
        # reference normal
        ax.plot(t_axis, X_norm_r[0], color=C_NORMAL, lw=1.4, ls="--",
                alpha=0.55, label="Normal ref")
        for k in range(5):
            ax.plot(t_axis, X_anom_r[k], color=C_ANOM, lw=0.8,
                    alpha=0.25, ls=":", label="Original" if k==0 else "_")
            if eps_val != "orig":
                ax.plot(t_axis, Xr[k], color=DET_COLORS[target], lw=1.1,
                        alpha=0.85,
                        label=f"ε={eps_val}" if k==0 else "_")
                ax.fill_between(t_axis, X_anom_r[k], Xr[k],
                                alpha=alpha_fill, color=DET_COLORS[target])
        if eps_val == "orig":
            lbl = f"{target} — Original anomalies"
        else:
            lbl = f"{target} — ε={eps_val}  (DR {dr(X_anom,target):.2f}→{dr(X_atk5[target] if eps_val==5 else X_atk10[target], target):.2f})"
        ax.set_title(lbl, fontsize=9, color="white")
        ax.set_facecolor(BG); ax.grid(True)
        ax.legend(fontsize=7)
        if row == 2: ax.set_xlabel("Time step")
        if col == 0: ax.set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig("anomaly_results/grad_evasion/ecg200_signal_overlay.png",
            dpi=145, bbox_inches="tight", facecolor=BG)
print("  Signal overlay figure saved.")
print("\nAll done.")