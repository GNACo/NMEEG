"""
Fig 2 — Group z-score deviation heatmaps.

Three side-by-side heatmaps (MCI | AD | ACr PSEN1 E280A).
Each cell: mean z-score (bands × ROIs) + FDR-corrected significance stars.
Per-group colour scale — each group's pattern is fully saturated and readable.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
    r"\blr_osc_pw_rel_canonic.csv"
)
_DEFAULT_SAVE = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
    r"\figures_paper"
)
SAVE_PATH = os.environ.get("NMEEG_FIGURES_PATH", _DEFAULT_SAVE)

BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
ROIS  = ["F", "C", "P", "O", "PO"]

BAND_LABELS = {
    "theta": "Theta", "alpha1": "Alpha 1", "alpha2": "Alpha 2",
    "beta1": "Beta 1", "beta2": "Beta 2",  "beta3": "Beta 3", "gamma": "Gamma",
}
ROI_LABELS = {
    "F": "Frontal", "C": "Central", "P": "Parietal",
    "O": "Occipital", "PO": "Parieto-\nOccipital",
}

CLINICAL_GROUPS = ["MCI", "AD", "ACr"]
GROUP_LABELS    = {"MCI": "MCI", "AD": "AD", "ACr": "ACr (PSEN1 E280A)"}
GROUP_COLORS    = {"MCI": "#7B3F9E", "AD": "#C0392B", "ACr": "#1A7A4A"}
SET_FILTER      = {"HC": "test", "MCI": "mci", "AD": "AD", "ACr": "ACr"}

# ── RCPARAMS ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "Arial",
    "font.size":       9,
    "axes.facecolor":  "white",
    "figure.facecolor":"white",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── LOAD & BUILD Z-SCORE TABLE ─────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)
frames = []
for grp, set_val in SET_FILTER.items():
    sub = df_raw[df_raw["set"] == set_val][
        ["subject", "band", "roi", "z_score"]].copy()
    sub["group"] = grp
    frames.append(sub)
df_z = pd.concat(frames, ignore_index=True)
df_z = df_z[df_z["band"].isin(BANDS) & df_z["roi"].isin(ROIS)]

# ── STATISTICS ────────────────────────────────────────────────────────────────
records = []
for grp in CLINICAL_GROUPS:
    for band in BANDS:
        for roi in ROIS:
            vals = df_z[
                (df_z["group"] == grp) &
                (df_z["band"]  == band) &
                (df_z["roi"]   == roi)
            ]["z_score"].dropna()
            if len(vals) < 5:
                continue
            try:
                _, pval = stats.wilcoxon(vals, alternative="two-sided")
            except Exception:
                pval = np.nan
            records.append({
                "group": grp, "band": band, "roi": roi,
                "n": len(vals), "mean_z": vals.mean(), "pval": pval,
            })

df_stats = pd.DataFrame(records)
if not df_stats.empty and df_stats["pval"].notna().any():
    valid = df_stats["pval"].notna()
    _, pvals_fdr, _, _ = multipletests(
        df_stats.loc[valid, "pval"].values, method="fdr_bh")
    df_stats.loc[valid,  "pval_fdr"] = pvals_fdr
    df_stats.loc[~valid, "pval_fdr"] = np.nan
else:
    df_stats["pval_fdr"] = np.nan

def sig_stars(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""

df_stats["stars"] = df_stats["pval_fdr"].apply(sig_stars)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
n_bands = len(BANDS)
n_rois  = len(ROIS)
n_clin  = len(CLINICAL_GROUPS)

# Taller figure — more height per band row
fig = plt.figure(figsize=(15.5, 8.5), dpi=300)

gs = gridspec.GridSpec(
    1, n_clin, figure=fig,
    wspace=0.52,
    top=0.88, bottom=0.11,
    left=0.10, right=0.97,
)

band_lbls = [BAND_LABELS[b] for b in BANDS]
roi_lbls  = [ROI_LABELS[r]  for r in ROIS]

for g_idx, grp in enumerate(CLINICAL_GROUPS):
    ax    = fig.add_subplot(gs[g_idx])
    color = GROUP_COLORS[grp]

    # ── Build matrices ────────────────────────────────────────────────────────
    mat_mean  = np.full((n_bands, n_rois), np.nan)
    mat_stars = np.full((n_bands, n_rois), "", dtype=object)
    for i, band in enumerate(BANDS):
        for j, roi in enumerate(ROIS):
            row = df_stats[
                (df_stats["group"] == grp) &
                (df_stats["band"]  == band) &
                (df_stats["roi"]   == roi)
            ]
            if not row.empty:
                mat_mean[i, j]  = row["mean_z"].values[0]
                mat_stars[i, j] = row["stars"].values[0]

    # Per-group colour scale (round to nearest 0.25)
    grp_vals = mat_mean[~np.isnan(mat_mean)]
    v_lim    = max(abs(grp_vals).max(), 0.5) if len(grp_vals) else 1.0
    v_lim    = np.ceil(v_lim * 4) / 4

    norm = TwoSlopeNorm(vmin=-v_lim, vcenter=0, vmax=v_lim)
    im   = ax.imshow(mat_mean, aspect="auto", cmap="RdBu_r",
                     norm=norm, interpolation="none")

    # ── Cell text (value + stars on same line) ────────────────────────────────
    for i in range(n_bands):
        for j in range(n_rois):
            val  = mat_mean[i, j]
            star = mat_stars[i, j]
            if np.isnan(val):
                continue
            saturated = abs(val) > 0.45 * v_lim
            txt_color = "white" if saturated else "#111111"
            # Value line
            ax.text(j, i - 0.10, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=txt_color)
            # Stars line (slightly smaller, below value)
            if star:
                ax.text(j, i + 0.28, star,
                        ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=txt_color,
                        alpha=0.90)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(range(n_rois))
    ax.set_xticklabels(roi_lbls, fontsize=9, ha="center")
    ax.tick_params(length=0, pad=5)

    # Band labels: left on first heatmap, right on last
    if g_idx == 0:
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(band_lbls, fontsize=9.5, ha="right", fontweight="bold")
    elif g_idx == n_clin - 1:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(band_lbls, fontsize=9.5, ha="left", fontweight="bold")
        ax.tick_params(axis="y", pad=6)
    else:
        ax.set_yticks([])

    # Horizontal white separators between bands
    for y in np.arange(0.5, n_bands - 0.5, 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)

    # Remove default spines
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Thin outer border around each heatmap
    border = FancyBboxPatch(
        (-0.5, -0.5), n_rois, n_bands,
        boxstyle="square,pad=0", linewidth=1.0,
        edgecolor="#AAAAAA", facecolor="none", zorder=10,
        transform=ax.transData,
    )
    ax.add_patch(border)

    # ── Group title ───────────────────────────────────────────────────────────
    n_grp = df_z[df_z["group"] == grp]["subject"].nunique()
    ax.set_title(GROUP_LABELS[grp],
                 fontsize=13, fontweight="bold",
                 color=color, pad=4)
    # Sample size subtitle (slightly below title, smaller)
    ax.text(0.5, 1.045, f"n = {n_grp}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color=color)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    # ACr (last group) needs extra pad to avoid covering right-side band labels
    cb_pad = 0.18 if g_idx == n_clin - 1 else 0.020
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=cb_pad,
                      shrink=0.88, aspect=22)
    cb.ax.tick_params(labelsize=8, width=0.6, length=3)
    cb.set_label("Mean z-score", fontsize=8.5, labelpad=5)
    # Mark z = 0
    cb.ax.axhline(norm(0), color="#333333", lw=1.2, ls="--", zorder=3)
    # Four clean ticks
    cb.set_ticks([-v_lim, -v_lim / 2, 0, v_lim / 2, v_lim])
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=7.5)

    # ── Panel label on first heatmap ──────────────────────────────────────────
    if g_idx == 0:
        ax.text(-0.26, 1.12, "a",
                transform=ax.transAxes,
                fontsize=15, fontweight="bold", style="italic",
                va="top", ha="left", color="#1a1a2e")

# ── Shared footnote ───────────────────────────────────────────────────────────
fig.text(
    0.50, 0.025,
    "Mean z-score deviation from the normative model.  "
    "* q < 0.05     ** q < 0.01     *** q < 0.001   "
    "(one-sample Wilcoxon signed-rank vs z = 0,  FDR — Benjamini–Hochberg).",
    ha="center", va="bottom",
    fontsize=8, color="#555555", style="italic",
)

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
for fmt in ("pdf", "png"):
    out = os.path.join(SAVE_PATH, f"fig2_group_deviations.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
    print(f"Saved: {out}")
plt.close(fig)
