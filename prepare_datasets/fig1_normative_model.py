"""
Fig 1 — Normative model & individual profiles.

Panel a  Centile curves — all 7 bands, Occipital ROI (2 × 4 grid).
Panel b  Percentile gauge cards — one representative subject per group
         (MCI, AD, ACr PSEN1 E280A) showing where each patient falls
         in the normative distribution for every frequency band.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
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
BAND_LABELS = {
    "theta": "Theta", "alpha1": "Alpha 1", "alpha2": "Alpha 2",
    "beta1": "Beta 1", "beta2": "Beta 2",  "beta3": "Beta 3", "gamma": "Gamma",
}
PANEL_ROI = "O"
ROI_NAME  = "Occipital"

CLINICAL_GROUPS = ["MCI", "AD", "ACr"]
GROUP_LABELS    = {"MCI": "MCI", "AD": "AD", "ACr": "ACr (PSEN1 E280A)"}
GROUP_COLORS    = {"MCI": "#7B3F9E", "AD": "#C0392B", "ACr": "#1A7A4A"}
SET_FILTER      = {"MCI": "mci", "AD": "AD", "ACr": "ACr"}

# Centile curve colours
C_PI    = "#D6E8FA"
C_CI    = "#F6C4A0"
C_LINE  = {5: "#9DB7E8", 25: "#4A6FD6", 50: "#0D1B3E", 75: "#6B46C1", 95: "#A68FE0"}
LW_LINE = {5: 1.3, 25: 1.6, 50: 2.8, 75: 1.6, 95: 1.3}
LS_LINE = {5: "--", 25: "-", 50: "-", 75: "-", 95: "--"}

# Gauge-card zone colours
PCTILE_ZONES = [
    (0,   5,  "#C0392B", 0.45, "< 5th  (extremely low)"),
    (5,  25,  "#E67E22", 0.38, "5–25th  (below normal)"),
    (25, 75,  "#27AE60", 0.32, "25–75th  (normal)"),
    (75, 95,  "#E67E22", 0.38, "75–95th  (above normal)"),
    (95, 100, "#C0392B", 0.45, "> 95th  (extremely high)"),
]

# ── RCPARAMS ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.9,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_raw   = pd.read_csv(DATA_PATH)
df_range = df_raw[df_raw["set"] == "range"].copy()
df_bands = df_raw[df_raw["band"].isin(BANDS)].copy()

# ── REPRESENTATIVE SUBJECT PER GROUP (median z-score = most typical) ──────────
rep_subjects, rep_ages = {}, {}
for grp, set_val in SET_FILTER.items():
    sub = df_bands[df_bands["set"] == set_val].dropna(subset=["z_score"])
    if sub.empty:
        continue
    ranked = sub.groupby("subject")["z_score"].mean().sort_values()
    chosen = ranked.index[len(ranked) // 2]
    rep_subjects[grp] = chosen
    rep_ages[grp]     = int(sub[sub["subject"] == chosen]["x_age"].iloc[0])

# ── CENTILE POSITION ──────────────────────────────────────────────────────────
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
# PANEL A — centile curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_centile(ax, band, show_xlabel=False, show_ylabel=False, is_first=False):
    rng = df_range[(df_range["band"] == band) & (df_range["roi"] == PANEL_ROI)]
    if rng.empty:
        ax.set_visible(False)
        return
    rng = rng.sort_values("x_age")
    x   = rng["x_age"].values

    ax.set_facecolor("#F7FAFF")

    ax.fill_between(x, rng["pi_lower"], rng["pi_upper"],
                    color=C_PI, alpha=0.90, lw=0, zorder=1)
    ax.fill_between(x, rng["ci_lower"], rng["ci_upper"],
                    color=C_CI, alpha=0.70, lw=0, zorder=2)
    for c in [5, 25, 50, 75, 95]:
        ax.plot(x, rng[f"centile_{c}"],
                color=C_LINE[c], lw=LW_LINE[c], ls=LS_LINE[c],
                alpha=0.95, zorder=3, solid_capstyle="round")

    # Centile labels on the right edge (clip_on=False so they show outside axes)
    lbl_fw = {5: "normal", 25: "normal", 50: "bold", 75: "normal", 95: "normal"}
    lbl_fs = {5: 6, 25: 6.5, 50: 7.5, 75: 6.5, 95: 6}
    for c in [5, 25, 50, 75, 95]:
        ax.annotate(f"{c}th",
            xy=(x[-1], rng[f"centile_{c}"].values[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=lbl_fs[c], color=C_LINE[c],
            va="center", ha="left", fontweight=lbl_fw[c],
            clip_on=False, annotation_clip=False)

    ax.set_xlim(x[0] - 1, x[-1] + 2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.grid(axis="y", color="#EBEBEB", ls="-",  lw=0.4, zorder=0)
    ax.grid(axis="x", color="#EBEBEB", ls="--", lw=0.4, zorder=0)
    ax.set_title(BAND_LABELS[band], fontsize=10, fontweight="bold",
                 color="#1a1a2e", pad=4)

    if show_xlabel:
        ax.set_xlabel("Age (years)", fontsize=9, labelpad=3)
    if show_ylabel:
        ax.set_ylabel("Osc. rel. power", fontsize=9, labelpad=4)

    # Panel label "a" on the very first axes only — above the band title
    if is_first:
        ax.text(-0.12, 1.14, "a",
                transform=ax.transAxes,
                fontsize=15, fontweight="bold", style="italic",
                va="bottom", ha="left", color="#1a1a2e")
        ax.text(0.02, 1.14, f"Normative centile model  —  {ROI_NAME} ROI",
                transform=ax.transAxes,
                fontsize=9.5, va="bottom", ha="left",
                color="#555555", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — gauge card
# ══════════════════════════════════════════════════════════════════════════════
# Data space:  zone bars run 0–100  |  left margin -26–0 reserved for labels
GAUGE_XLIM = (-26, 102)
GAUGE_BAR_H = 0.80    # fraction of one unit occupied by each bar

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

        gap = (1 - GAUGE_BAR_H) / 2
        y_lo = y0 + gap
        y_hi = y0 + GAUGE_BAR_H + gap

        # Zone bars (0–100)
        for lo, hi, col, alpha, _ in PCTILE_ZONES:
            ax.fill_between([lo, hi], y_lo, y_hi,
                            color=col, alpha=alpha, lw=0, zorder=2)
        # White dividers
        for bound in [5, 25, 75, 95]:
            ax.plot([bound, bound], [y_lo, y_hi],
                    color="white", lw=1.8, zorder=3)

        # Subject marker (triangle)
        ax.scatter([cp], [(y_lo + y_hi) / 2], marker="v",
                   s=130, color=color,
                   edgecolors="white", linewidths=1.2, zorder=6)

        # Percentile value just above bar
        cp_clipped = np.clip(cp, 4, 96)
        ax.text(cp_clipped, y_hi + 0.06, f"{cp:.0f}",
                fontsize=7, ha="center", va="bottom",
                color=zone_color(cp), fontweight="bold",
                clip_on=False)

        # Band label in the LEFT margin (data coords, clip_on=False)
        ax.text(-1, (y_lo + y_hi) / 2, BAND_LABELS[band],
                fontsize=9, ha="right", va="center",
                fontweight="bold", color="#222222",
                clip_on=False)

    # Thin box around the zone bars only (x: 0–100, y: full height)
    box = FancyBboxPatch(
        (0 - 0.3, -0.08), 100.6, n_b + 0.16,
        boxstyle="square,pad=0",
        linewidth=0.9, edgecolor="#AAAAAA", facecolor="none",
        zorder=8, clip_on=False,
    )
    ax.add_patch(box)

    ax.set_xlim(*GAUGE_XLIM)
    ax.set_ylim(-0.15, n_b + 0.35)
    ax.set_xticks([5, 25, 50, 75, 95])
    ax.set_xticklabels(["5th", "25th", "50th", "75th", "95th"],
                       fontsize=8)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=3, pad=3, width=0.8)

    # Group title
    age = rep_ages.get(grp, "?")
    ax.set_title(f"{GROUP_LABELS[grp]}   ·   Age {age} yr",
                 fontsize=10.5, fontweight="bold",
                 color=color, pad=6)

    # Panel label "b" on the first gauge card — well above the group title
    if is_first:
        ax.text(-0.18, 1.17, "b",
                transform=ax.transAxes,
                fontsize=15, fontweight="bold", style="italic",
                va="bottom", ha="left", color="#1a1a2e")
        ax.text(-0.02, 1.17, f"Individual normative profiles  —  {ROI_NAME} ROI",
                transform=ax.transAxes,
                fontsize=9.5, va="bottom", ha="left",
                color="#555555", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(19, 15), dpi=300)

# --- outer grid: Panel A (top) / Panel B (bottom) ----------------------------
# hspace kept small so panels sit close together with no dead space
gs_outer = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.35, 1.10],
    hspace=0.22,
    top=0.96, bottom=0.08,
    left=0.07, right=0.95,
)

# ── Panel A: 2 × 4 band curves ────────────────────────────────────────────────
gs_A = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=gs_outer[0],
    hspace=0.55, wspace=0.52,
)

for idx, band in enumerate(BANDS):
    r_i, c_i = divmod(idx, 4)
    ax = fig.add_subplot(gs_A[r_i, c_i])
    plot_centile(ax, band,
                 show_xlabel=(r_i == 1),
                 show_ylabel=(c_i == 0),
                 is_first=(idx == 0))

# Legend in the 8th slot [row 1, col 3]
ax_leg = fig.add_subplot(gs_A[1, 3])
ax_leg.axis("off")
leg_handles = [
    Patch(facecolor=C_PI, edgecolor="#8BAED6", lw=0.6, alpha=0.90,
          label="95% Prediction interval"),
    Patch(facecolor=C_CI, edgecolor="none", alpha=0.70,
          label="95% Confidence interval"),
    Line2D([0], [0], color=C_LINE[50], lw=2.8, label="Median (50th)"),
    Line2D([0], [0], color=C_LINE[25], lw=1.6, label="Centiles (25th, 75th)"),
    Line2D([0], [0], color=C_LINE[5],  lw=1.3, ls="--",
           label="Outer centiles (5th, 95th)"),
]
ax_leg.legend(
    handles=leg_handles, loc="center",
    fontsize=8.5, title="Model reference", title_fontsize=9,
    frameon=True, framealpha=0.97, edgecolor="#CCCCCC",
    handlelength=1.8, handletextpad=0.6, labelspacing=0.80,
)

# ── Panel B: gauge cards ──────────────────────────────────────────────────────
gs_B = gridspec.GridSpecFromSubplotSpec(
    1, len(CLINICAL_GROUPS), subplot_spec=gs_outer[1],
    wspace=0.38,
)

for g_idx, grp in enumerate(CLINICAL_GROUPS):
    ax = fig.add_subplot(gs_B[g_idx])
    draw_gauge(ax, grp, is_first=(g_idx == 0))

# ── Zone colour legend — bottom of figure, inside bottom margin ──────────────
zone_handles = [
    Patch(facecolor=col, alpha=alpha + 0.10, edgecolor="none", label=lbl)
    for _, _, col, alpha, lbl in PCTILE_ZONES
]
fig.legend(
    handles=zone_handles,
    loc="lower center", bbox_to_anchor=(0.50, 0.005),
    ncol=5, fontsize=8, frameon=False,
    handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
)

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
for fmt in ("pdf", "png"):
    out = os.path.join(SAVE_PATH, f"fig1_normative_model.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
    print(f"Saved: {out}")
plt.close(fig)
