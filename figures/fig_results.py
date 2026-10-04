"""
Fig results — Unified results figure (single figure for paper).

Panel a  Normative centile model — 7 bands, Occipital ROI (2 x 4 curves + legend)
Panel b  Group z-score deviation heatmaps — MCI | AD | ACr (7 bands x 5 ROIs)
Panel c  Individual normative profiles — gauge cards, one representative per group
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_config_params.paths import BLR_DIR, FIGURES_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = (
    os.path.join(BLR_DIR, "blr_osc_pw_rel_canonic.csv")
)
_DEFAULT_SAVE = (
    FIGURES_DIR
)
SAVE_PATH = os.environ.get("NMEEG_FIGURES_PATH", _DEFAULT_SAVE)

BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
ROIS  = ["F", "C", "P", "O", "PO"]
PANEL_ROI = "O"

BAND_LABELS = {
    "theta": "Theta", "alpha1": "Alpha 1", "alpha2": "Alpha 2",
    "beta1": "Beta 1", "beta2": "Beta 2",  "beta3": "Beta 3", "gamma": "Gamma",
}
ROI_LABELS = {
    "F": "Frontal", "C": "Central", "P": "Parietal",
    "O": "Occipital", "PO": "Parieto-\nOccipital",
}

CLINICAL_GROUPS = ["MCI", "AD", "ACr"]
GROUP_LABELS    = {"MCI": "MCI", "AD": "AD", "ACr": "ACr"}
GROUP_COLORS    = {"MCI": "#7B3F9E", "AD": "#C0392B", "ACr": "#1A7A4A"}
SET_FILTER      = {"HC": "test", "MCI": "mci", "AD": "AD", "ACr": "ACr"}

# Centile curve colours
C_PI   = "#D6E8FA"
C_CI   = "#F6C4A0"
C_LINE = {5: "#9DB7E8", 25: "#4A6FD6", 50: "#0D1B3E", 75: "#6B46C1", 95: "#A68FE0"}
LW_LN  = {5: 1.3,  25: 1.6, 50: 2.8, 75: 1.6, 95: 1.3}
LS_LN  = {5: "--", 25: "-", 50: "-", 75: "-", 95: "--"}

# Gauge-card zone colours
PCTILE_ZONES = [
    (0,   5,  "#C0392B", 0.45, "< 5th  (extremely low)"),
    (5,  25,  "#E67E22", 0.38, "5-25th  (below normal)"),
    (25, 75,  "#27AE60", 0.32, "25-75th  (normal)"),
    (75, 95,  "#E67E22", 0.38, "75-95th  (above normal)"),
    (95, 100, "#C0392B", 0.45, "> 95th  (extremely high)"),
]

# ── RCPARAMS ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_raw   = pd.read_csv(DATA_PATH)
df_range = df_raw[df_raw["set"] == "range"].copy()
df_bands = df_raw[df_raw["band"].isin(BANDS)].copy()

# Z-score table for all groups
frames = []
for grp, set_val in SET_FILTER.items():
    sub = df_raw[df_raw["set"] == set_val][
        ["subject", "band", "roi", "z_score"]].copy()
    sub["group"] = grp
    frames.append(sub)
df_z = pd.concat(frames, ignore_index=True)
df_z = df_z[df_z["band"].isin(BANDS) & df_z["roi"].isin(ROIS)]

# Representative subject per group (median mean-z across all bands)
rep_subjects, rep_ages = {}, {}
for grp, set_val in {"MCI": "mci", "AD": "AD", "ACr": "ACr"}.items():
    sub = df_bands[df_bands["set"] == set_val].dropna(subset=["z_score"])
    if sub.empty:
        continue
    ranked = sub.groupby("subject")["z_score"].mean().sort_values()
    chosen = ranked.index[len(ranked) // 2]
    rep_subjects[grp] = chosen
    rep_ages[grp]     = int(sub[sub["subject"] == chosen]["x_age"].iloc[0])

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS FOR HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
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
# HELPER — percentile position for gauge cards
# ══════════════════════════════════════════════════════════════════════════════
def get_centile_pos(y_true, p1, p5, p25, p50, p75, p95, p99):
    eps = 1e-12
    ap  = np.array([0, 1, 5, 25, 50, 75, 95, 99, 100])
    av  = np.array([
        p1 - (p5 - p1) / max(p5 - p1, eps),
        p1, p5, p25, p50, p75, p95, p99,
        p99 + (p99 - p95) / max(p99 - p95, eps),
    ])
    if not np.all(np.diff(av) >= 0):
        return 50.0
    return float(np.clip(np.interp(y_true, av, ap), 0, 100))

def zone_color(centile):
    for lo, hi, col, *_ in PCTILE_ZONES:
        if lo <= centile < hi:
            return col
    return "#C0392B"

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — centile curves
# ══════════════════════════════════════════════════════════════════════════════
def draw_curve(ax, band, is_first=False):
    rng = df_range[(df_range["band"] == band) & (df_range["roi"] == PANEL_ROI)]
    if rng.empty:
        ax.set_visible(False)
        return
    rng = rng.sort_values("x_age")
    x   = rng["x_age"].values

    #ax.set_facecolor("#F7FAFF")
    ax.fill_between(x, rng["pi_lower"], rng["pi_upper"],
                    color=C_PI, alpha=0.90, lw=0, zorder=1)
    ax.fill_between(x, rng["ci_lower"], rng["ci_upper"],
                    color=C_CI, alpha=0.70, lw=0, zorder=2)
    for c in [5, 25, 50, 75, 95]:
        ax.plot(x, rng[f"centile_{c}"],
                color=C_LINE[c], lw=LW_LN[c], ls=LS_LN[c],
                alpha=0.95, zorder=3, solid_capstyle="round")
    # Centile labels at right edge
    lbl_fw = {5: "normal", 25: "normal", 50: "bold", 75: "normal", 95: "normal"}
    lbl_fs = {5: 10, 25: 11, 50: 12, 75: 11, 95: 10}
    for c in [5, 25, 50, 75, 95]:
        ax.annotate(f"{c}th",
            xy=(x[-1], rng[f"centile_{c}"].values[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=lbl_fs[c], color=C_LINE[c],
            va="center", ha="left", fontweight=lbl_fw[c],
            clip_on=False, annotation_clip=False)

    ax.set_xlim(x[0] - 1, x[-1] + 2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    #ax.grid(axis="y", color="#EBEBEB", ls="-",  lw=0.4, zorder=0)
    #ax.grid(axis="x", color="#EBEBEB", ls="--", lw=0.4, zorder=0)
    ax.set_title(BAND_LABELS[band], fontsize=18, fontweight="bold",
                 color="#1a1a2e", pad=6)

    ax.set_xlabel("Age (years)", fontsize=16, labelpad=4)
    ax.set_ylabel("Power", fontsize=16, labelpad=5)

    # Panel "a" label — only once, top-left of section
    if is_first:
        ax.text(-0.18, 1.10, "a",
                transform=ax.transAxes,
                fontsize=22, fontweight="bold", style="italic",
                va="bottom", ha="left", color="#1a1a2e")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — z-score heatmaps
# ══════════════════════════════════════════════════════════════════════════════
def draw_heatmap(ax, fig, grp, g_idx):
    color  = 'black' #GROUP_COLORS[grp]
    n_b    = len(BANDS)
    n_r    = len(ROIS)

    band_lbls = [BAND_LABELS[b] for b in BANDS]
    roi_lbls  = [ROI_LABELS[r]  for r in ROIS]

    mat_mean  = np.full((n_b, n_r), np.nan)
    mat_stars = np.full((n_b, n_r), "", dtype=object)
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

    grp_vals = mat_mean[~np.isnan(mat_mean)]
    v_lim    = max(abs(grp_vals).max(), 0.5) if len(grp_vals) else 1.0
    v_lim    = np.ceil(v_lim * 4) / 4
    norm     = TwoSlopeNorm(vmin=-v_lim, vcenter=0, vmax=v_lim)
    im       = ax.imshow(mat_mean, aspect="auto", cmap="PuOr_r",
                         norm=norm, interpolation="none")

    # Cell text
    for i in range(n_b):
        for j in range(n_r):
            val  = mat_mean[i, j]
            star = mat_stars[i, j]
            if np.isnan(val):
                continue
            saturated = abs(val) > 0.45 * v_lim
            tc = "white" if saturated else "#111111"
            ax.text(j, i - 0.10, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color=tc)
            if star:
                ax.text(j, i + 0.28, star,
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color=tc, alpha=0.90)

    ax.set_xticks(range(n_r))
    ax.set_xticklabels(roi_lbls, fontsize=13, ha="center")
    ax.tick_params(length=0, pad=5)
    ax.set_xlabel("ROI", fontsize=13, fontweight="bold", labelpad=8)

    # Band labels: always on the left for all heatmaps
    ax.set_yticks(range(n_b))
    ax.set_yticklabels(band_lbls, fontsize=13, ha="right", fontweight="bold")
    ax.set_ylabel("Frequency band", fontsize=13, fontweight="bold", labelpad=8)

    for y in np.arange(0.5, n_b - 0.5, 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    border = FancyBboxPatch(
        (-0.5, -0.5), n_r, n_b,
        boxstyle="square,pad=0", linewidth=1.0,
        edgecolor="#AAAAAA", facecolor="none", zorder=10,
        transform=ax.transData,
    )
    ax.add_patch(border)

    n_grp = df_z[df_z["group"] == grp]["subject"].nunique()
    ax.set_title(rf"{GROUP_LABELS[grp]} - n = {n_grp}",
                 fontsize=18, fontweight="bold", color=color, pad=6)
    # ax.text(0.4, 1.045, f"n = {n_grp}",
    #         transform=ax.transAxes, ha="center", va="bottom",
    #        fontsize=16, color=color)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.020,
                      shrink=0.88, aspect=22)
    cb.set_label("Mean z-score", fontsize=15, labelpad=4)
    cb.ax.axhline(norm(0), color="#333333", lw=1.2, ls="--", zorder=3)
    cb.set_ticks([-v_lim, -v_lim / 2, 0, v_lim / 2, v_lim])
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=11, width=0.6, length=3)

    # Panel "b" label — only once
    if g_idx == 0:
        ax.text(-0.18, 1.10, "b",
                transform=ax.transAxes,
                fontsize=22, fontweight="bold", style="italic",
                va="bottom", ha="left", color="#1a1a2e")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — gauge cards
# ══════════════════════════════════════════════════════════════════════════════
GAUGE_XLIM  = (-26, 102)
GAUGE_BAR_H = 0.80

def draw_gauge(ax, grp, is_first=False):
    color   = GROUP_COLORS[grp]
    subj    = rep_subjects.get(grp)
    df_subj = df_bands[
        (df_bands["subject"] == subj) & (df_bands["roi"] == PANEL_ROI)
    ].copy()
    n_b = len(BANDS)

    for bi, band in enumerate(BANDS):
        y0  = n_b - 1 - bi
        row = df_subj[df_subj["band"] == band]
        if row.empty:
            continue
        r = row.iloc[0]
        try:
            cp = get_centile_pos(r["y_true"], r["p1"], r["p5"], r["p25"],
                                 r["p50"], r["p75"], r["p95"], r["p99"])
        except Exception:
            cp = 50.0

        gap  = (1 - GAUGE_BAR_H) / 2
        y_lo = y0 + gap
        y_hi = y0 + GAUGE_BAR_H + gap

        for lo, hi, col, alpha, _ in PCTILE_ZONES:
            ax.fill_between([lo, hi], y_lo, y_hi,
                            color=col, alpha=alpha, lw=0, zorder=2)
        for bound in [5, 25, 75, 95]:
            ax.plot([bound, bound], [y_lo, y_hi],
                    color="white", lw=1.8, zorder=3)

        ax.scatter([cp], [(y_lo + y_hi) / 2], marker="v",
                   s=130, color=color,
                   edgecolors="white", linewidths=1.2, zorder=6)

        cp_clipped = np.clip(cp, 4, 96)
        ax.text(cp_clipped, y_hi + 0.06, f"{cp:.0f}",
                fontsize=15, ha="center", va="bottom",
                color=zone_color(cp), fontweight="bold", clip_on=False)

        # Band label in left margin
        ax.text(-1, (y_lo + y_hi) / 2, BAND_LABELS[band],
                fontsize=16, ha="right", va="center",
                fontweight="bold", color="#222222", clip_on=False)

    box = FancyBboxPatch(
        (-0.3, -0.08), 100.6, n_b + 0.16,
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#AAAAAA", facecolor="none", zorder=8, clip_on=False,
    )
    ax.add_patch(box)

    ax.set_xlim(*GAUGE_XLIM)
    ax.set_ylim(-0.15, n_b + 0.35)
    ax.set_xticks([5, 25, 50, 75, 95])
    ax.set_xticklabels(["5th", "25th", "50th", "75th", "95th"], fontsize=15)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=3, pad=3, width=0.8)

    age = rep_ages.get(grp, "?")
    ax.set_title(f"{GROUP_LABELS[grp]}  -  Age {age} yr",
                 fontsize=18, fontweight="bold", color='black', pad=7)

    # Panel "c" label — only once
    if is_first:
        ax.text(-0.18, 1.10, "c",
                transform=ax.transAxes,
                fontsize=22, fontweight="bold", style="italic",
                va="bottom", ha="left", color="#1a1a2e")

# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(19, 17), dpi=300)

gs_outer = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[1.45, 1.0, 1.05],
    hspace=0.40,
    top=0.97, bottom=0.05,
    left=0.07, right=0.96,
)

# ── Panel a: centile curves (2 x 4) ──────────────────────────────────────────
gs_a = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=gs_outer[0],
    hspace=0.85, wspace=0.55,
)
for idx, band in enumerate(BANDS):
    r_i, c_i = divmod(idx, 4)
    ax = fig.add_subplot(gs_a[r_i, c_i])
    draw_curve(ax, band, is_first=(idx == 0))

# Legend in 8th slot
ax_leg = fig.add_subplot(gs_a[1, 3])
ax_leg.axis("off")
ax_leg.legend(
    handles=[
        Patch(facecolor=C_PI, edgecolor="#8BAED6", lw=0.6, alpha=0.90,
              label="95% Prediction interval"),
        Patch(facecolor=C_CI, edgecolor="none", alpha=0.70,
              label="95% Confidence interval"),
        Line2D([0], [0], color=C_LINE[50], lw=2.8, label="Median (50th)"),
        Line2D([0], [0], color=C_LINE[25], lw=1.6,
               label="Centiles (25th, 75th)"),
        Line2D([0], [0], color=C_LINE[5],  lw=1.3, ls="--",
               label="Outer centiles (5th, 95th)"),
    ],
    loc="center", fontsize=16, title="Model reference", title_fontsize=17,
    frameon=True, framealpha=0.97, edgecolor="#CCCCCC",
    handlelength=1.8, handletextpad=0.6, labelspacing=0.80,
)

# ── Panel b: heatmaps ─────────────────────────────────────────────────────────
gs_b = gridspec.GridSpecFromSubplotSpec(
    1, len(CLINICAL_GROUPS), subplot_spec=gs_outer[1],
    wspace=0.55,
)
for g_idx, grp in enumerate(CLINICAL_GROUPS):
    ax = fig.add_subplot(gs_b[g_idx])
    draw_heatmap(ax, fig, grp, g_idx)

# ── Panel c: gauge cards ──────────────────────────────────────────────────────
gs_c = gridspec.GridSpecFromSubplotSpec(
    1, len(CLINICAL_GROUPS), subplot_spec=gs_outer[2],
    wspace=0.40,
)
for g_idx, grp in enumerate(CLINICAL_GROUPS):
    ax = fig.add_subplot(gs_c[g_idx])
    draw_gauge(ax, grp, is_first=(g_idx == 0))

# ── Zone colour legend at the bottom ─────────────────────────────────────────
fig.legend(
    handles=[
        Patch(facecolor=col, alpha=alpha + 0.10, edgecolor="none", label=lbl)
        for _, _, col, alpha, lbl in PCTILE_ZONES
    ],
    loc="lower center", bbox_to_anchor=(0.50, 0.002),
    ncol=5, fontsize=14, frameon=False,
    handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
)

# ── Statistical footnote (between c and zone legend) ─────────────────────────
# fig.text(
#     0.50, 0.025,
#     "Heatmap: mean z-score deviation from normative model.  "
#     "* q<0.05   ** q<0.01   *** q<0.001  "
#     "(one-sample Wilcoxon, FDR — Benjamini-Hochberg)",
#     ha="center", va="bottom",
#     fontsize=7.5, color="#555555", style="italic",
# )

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
for fmt in ("pdf", "png"):
    out = os.path.join(SAVE_PATH, f"fig_results.{fmt}")
    try:
        fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
        print(f"Saved: {out}")
    except PermissionError:
        print(f"Skipped (file open elsewhere): {out}")
plt.close(fig)
