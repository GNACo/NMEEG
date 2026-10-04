"""
BLR normative model – combined performance & calibration figure.

Layout:
  Row 1 (top):    a) MSLL heatmap   |   b) Spearman ρ heatmap + stars
  Row 2 (bottom): c) Q-Q plots, one per frequency band (z-scores pooled across ROIs)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_config_params.paths import BLR_DIR, FIGURES_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────────
METRICS_PATH = (
    os.path.join(BLR_DIR, "metrics_osc_pw_rel_canonic.csv")
)
RESULTS_PATH = (
    os.path.join(BLR_DIR, "blr_osc_pw_rel_canonic.csv")
)
_DEFAULT_SAVE = (
    FIGURES_DIR
)
SAVE_PATH = os.environ.get("NMEEG_FIGURES_PATH", _DEFAULT_SAVE)

BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
ROIS  = ["F", "C", "P", "O", "PO"]

BAND_LABELS = {
    "theta":  "Theta",
    "alpha1": "Alpha 1",
    "alpha2": "Alpha 2",
    "beta1":  "Beta 1",
    "beta2":  "Beta 2",
    "beta3":  "Beta 3",
    "gamma":  "Gamma",
}
ROI_LABELS = {
    "F":  "Frontal",
    "C":  "Central",
    "P":  "Parietal",
    "O":  "Occipital",
    "PO": "Parieto-\nOccipital",
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_metrics = pd.read_csv(METRICS_PATH)
df_metrics = df_metrics[df_metrics["band"].isin(BANDS)].copy()

df_results = pd.read_csv(RESULTS_PATH)
df_test = (
    df_results[df_results["set"] == "test"]
    .dropna(subset=["z_score"])
    .copy()
)

# ── BUILD HEATMAP MATRICES ────────────────────────────────────────────────────
n_b, n_r = len(BANDS), len(ROIS)
msll_mat = np.full((n_b, n_r), np.nan)
rho_mat  = np.full((n_b, n_r), np.nan)
pval_mat = np.full((n_b, n_r), np.nan)

for i, band in enumerate(BANDS):
    for j, roi in enumerate(ROIS):
        row = df_metrics[(df_metrics["band"] == band) & (df_metrics["roi"] == roi)]
        if len(row) == 1:
            msll_mat[i, j] = row["msll_test"].values[0]
            rho_mat[i, j]  = row["rho_test"].values[0]
            pval_mat[i, j] = row["pvalue_test"].values[0]

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "Arial",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "axes.linewidth":     0.7,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "axes.grid":          False,
})

band_lbls = [BAND_LABELS[b] for b in BANDS]
roi_lbls  = [ROI_LABELS[r]  for r in ROIS]

# ── FIGURE LAYOUT ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), dpi=300)

gs_outer = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.15, 1.0],
    hspace=0.42,
    top=0.93, bottom=0.07, left=0.10, right=0.97,
)

# ── ROW 1 : heatmaps ──────────────────────────────────────────────────────────
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[0],
    wspace=0.38,
)

# ── Panel A : MSLL ────────────────────────────────────────────────────────────
ax_A = fig.add_subplot(gs_top[0])

v_abs  = max(abs(np.nanmin(msll_mat)), abs(np.nanmax(msll_mat)))
norm_A = TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs)
im_A   = ax_A.imshow(msll_mat, aspect="auto", cmap="RdBu", norm=norm_A)

for i in range(n_b):
    for j in range(n_r):
        val = msll_mat[i, j]
        if not np.isnan(val):
            dark = abs(val) > 0.50 * v_abs
            ax_A.text(j, i, f"{val:.3f}", ha="center", va="center",
                      fontsize=7.8,
                      color="white" if dark else "#222222",
                      fontweight="bold" if dark else "normal")

ax_A.set_xticks(range(n_r))
ax_A.set_xticklabels(roi_lbls, fontsize=8.5)
ax_A.set_yticks(range(n_b))
ax_A.set_yticklabels(band_lbls, fontsize=9.5)
ax_A.set_title("Mean Standardized Log Loss (MSLL)", fontsize=10.5,
               fontweight="bold", pad=8, color="#1a1a2e")
ax_A.tick_params(length=0, labelsize=8.5)

cb_A = fig.colorbar(im_A, ax=ax_A, fraction=0.046, pad=0.03, shrink=0.82)
cb_A.ax.tick_params(labelsize=7.5)
cb_A.ax.axhline(0, color="#333333", lw=1.0, ls="--")

ax_A.text(0.5, -0.12,
          "MSLL < 0  →  model outperforms trivial baseline",
          transform=ax_A.transAxes, ha="center", va="top",
          fontsize=8, color="#555555", style="italic")

# ── Panel B : Spearman ρ ──────────────────────────────────────────────────────
ax_B = fig.add_subplot(gs_top[1])

rho_lim = 0.45
norm_B  = TwoSlopeNorm(vmin=-rho_lim, vcenter=0, vmax=rho_lim)
im_B    = ax_B.imshow(rho_mat, aspect="auto", cmap="RdBu_r", norm=norm_B)

for i in range(n_b):
    for j in range(n_r):
        rho  = rho_mat[i, j]
        pval = pval_mat[i, j]
        if not np.isnan(rho):
            stars = sig_stars(pval)
            dark  = abs(rho) > 0.25
            label = f"{rho:.2f}{stars}"
            ax_B.text(j, i, label, ha="center", va="center",
                      fontsize=7.8,
                      color="white" if dark else "#222222",
                      fontweight="bold" if dark else "normal")

ax_B.set_xticks(range(n_r))
ax_B.set_xticklabels(roi_lbls, fontsize=8.5)
ax_B.set_yticks(range(n_b))
ax_B.set_yticklabels(band_lbls, fontsize=9.5)
ax_B.set_title("Spearman ρ  (test set)", fontsize=10.5,
               fontweight="bold", pad=8, color="#1a1a2e")
ax_B.tick_params(length=0, labelsize=8.5)

cb_B = fig.colorbar(im_B, ax=ax_B, fraction=0.046, pad=0.03, shrink=0.82)
cb_B.ax.tick_params(labelsize=7.5)

ax_B.text(0.5, -0.12,
          "* p < 0.05     ** p < 0.01     *** p < 0.001",
          transform=ax_B.transAxes, ha="center", va="top",
          fontsize=8, color="#555555", style="italic")

# ── ROW 2 : Q-Q plots ─────────────────────────────────────────────────────────
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, len(BANDS), subplot_spec=gs_outer[1],
    wspace=0.40,
)

qq_color = "#2C7BB6"

for b_idx, band in enumerate(BANDS):
    ax = fig.add_subplot(gs_bot[b_idx])

    z_pool = df_test[df_test["band"] == band]["z_score"].dropna().values
    if len(z_pool) < 10:
        ax.set_visible(False)
        continue

    z_sorted = np.sort(z_pool)
    n = len(z_sorted)
    theoretical_q = stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))

    ax.scatter(theoretical_q, z_sorted,
               s=4, color=qq_color, alpha=0.45, linewidths=0, zorder=3)

    lim = max(abs(theoretical_q).max(), abs(z_sorted).max()) * 1.06
    ax.plot([-lim, lim], [-lim, lim],
            color="#333333", lw=1.2, ls="--", zorder=4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Theoretical\nquantiles", fontsize=7.5, labelpad=2)
    if b_idx == 0:
        ax.set_ylabel("Sample quantiles\n(z-scores)", fontsize=7.5, labelpad=3)

    ax.set_title(BAND_LABELS[band], fontsize=9.5, fontweight="bold", pad=4,
                 color="#1a1a2e")
    ax.tick_params(labelsize=7)

    ax.text(0.05, 0.95, f"n = {n}", transform=ax.transAxes,
            fontsize=6.5, va="top", color="#555555")

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

# ── PANEL LABELS ──────────────────────────────────────────────────────────────
# Get approximate y positions
top_y   = gs_outer[0].get_position(fig).y1
bot_y   = gs_outer[1].get_position(fig).y1

fig.text(0.005, top_y + 0.005, "a", fontsize=14, fontweight="bold",
         va="bottom", style="italic")
fig.text(0.50,  top_y + 0.005, "b", fontsize=14, fontweight="bold",
         va="bottom", style="italic")
fig.text(0.005, bot_y + 0.005, "c", fontsize=14, fontweight="bold",
         va="bottom", style="italic")

fig.text(0.50, bot_y + 0.003,
         "Q–Q plots: z-scores vs. standard Gaussian  (test set, ROIs pooled per band)",
         ha="center", va="bottom", fontsize=8, color="#555555", style="italic")

# ── MAIN TITLE ────────────────────────────────────────────────────────────────
fig.suptitle(
    "BLR normative model – performance and calibration on held-out test set",
    fontsize=12, fontweight="bold", y=0.975,
)

# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
out_pdf = os.path.join(SAVE_PATH, "blr_performance_figure.pdf")
out_png = os.path.join(SAVE_PATH, "blr_performance_figure.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight", format="png")
print(f"Figure saved:\n  {out_pdf}\n  {out_png}")
plt.close(fig)
