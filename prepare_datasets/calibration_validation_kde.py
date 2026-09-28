"""
Calibration validation of the BLR normative model – KDE style.
Analogous to Figure 2 of Zamanzadeh et al. (2026):
  Panel A – KDE distributions of each metric (SMSE, Skewness, Excess Kurtosis,
             Shapiro-Wilk W, MACE) obtained via bootstrap resampling (N=200)
             of the test set, one row per frequency band.
  Panel B – Q-Q plots (one per band), pooling all ROIs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
    r"\blr_osc_pw_rel_canonic.csv"
)
SAVE_PATH = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
)

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
ROI_LABELS = {"F": "Frontal", "C": "Central", "P": "Parietal",
              "O": "Occipital", "PO": "Parieto-Occipital"}

CENTILE_COLS = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
CENTILE_VALS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

N_BOOTSTRAP = 200
RANDOM_SEED = 42

# One colour per ROI (same palette used in first calibration figure)
ROI_COLORS = {
    "F":  "#1F77B4",
    "C":  "#E8782A",
    "P":  "#2CA02C",
    "O":  "#D62728",
    "PO": "#9467BD",
}
ROI_ALPHA_FILL = 0.18   # transparency for KDE fill
ROI_ALPHA_LINE = 0.90   # transparency for KDE line

C_REF = "#333333"  # reference line colour

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df_test = (
    df[df["set"] == "test"]
    .dropna(subset=["z_score", "y_true", "y_pred"] + CENTILE_COLS)
    .copy()
)

# ── MACE helper ───────────────────────────────────────────────────────────────
def compute_mace(subset):
    y = subset["y_true"].values
    errors = [abs(qj - np.mean(y <= subset[col].values))
              for qj, col in zip(CENTILE_VALS, CENTILE_COLS)]
    return float(np.mean(errors))

# ── BOOTSTRAP per band × ROI ──────────────────────────────────────────────────
print("Running bootstrap per band × ROI …")
rng_bs = np.random.default_rng(RANDOM_SEED)

# bootstrap[band][roi][metric] = list of N_BOOTSTRAP values
bootstrap = {
    band: {
        roi: {"smse": [], "skewness": [], "excess_kurtosis": [],
              "shapiro_w": [], "mace": []}
        for roi in ROIS
    }
    for band in BANDS
}

for band in BANDS:
    for roi in ROIS:
        df_br = df_test[(df_test["band"] == band) & (df_test["roi"] == roi)]
        n = len(df_br)
        print(f"  {band} / {roi}: n={n}")
        if n < 8:
            continue
        for _ in range(N_BOOTSTRAP):
            idx = rng_bs.integers(0, n, size=n)
            s = df_br.iloc[idx]
            z  = s["z_score"].values
            y  = s["y_true"].values
            yp = s["y_pred"].values

            bootstrap[band][roi]["smse"].append(
                float(np.mean((y - yp) ** 2) / np.var(y))
            )
            bootstrap[band][roi]["skewness"].append(float(stats.skew(z)))
            bootstrap[band][roi]["excess_kurtosis"].append(
                float(stats.kurtosis(z))
            )
            z_sw = z[:5000] if len(z) > 5000 else z
            bootstrap[band][roi]["shapiro_w"].append(
                float(stats.shapiro(z_sw)[0])
            )
            bootstrap[band][roi]["mace"].append(compute_mace(s))

print("Bootstrap done.")

# ── METRIC METADATA ───────────────────────────────────────────────────────────
METRICS = [
    ("smse",            "SMSE",            1.0,  None),   # (key, label, ref, x_lim_override)
    ("skewness",        "Skewness",        0.0,  None),
    ("excess_kurtosis", "Excess kurtosis", 0.0,  None),
    ("shapiro_w",       "W",               1.0,  (0.85, 1.01)),
    ("mace",            "MACE",            0.0,  (0.0, 0.08)),
]

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.7,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.grid":         False,
})

N_ROWS = len(BANDS)    # 7
N_COLS = len(METRICS)  # 5

# ── BUILD FIGURE ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 17), dpi=300)

gs_outer = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[2.5, 0.10, 1.2],
    hspace=0.0,
    top=0.94, bottom=0.04, left=0.11, right=0.98,
)

# ─── PANEL A : KDE grid ───────────────────────────────────────────────────────
gs_A = gridspec.GridSpecFromSubplotSpec(
    N_ROWS, N_COLS,
    subplot_spec=gs_outer[0],
    hspace=0.30, wspace=0.38,
)

for r, band in enumerate(BANDS):
    for c, (metric_key, metric_label, ref_val, xlim_ov) in enumerate(METRICS):

        ax = fig.add_subplot(gs_A[r, c])

        # Collect global x-range across all ROIs first
        all_vals = []
        for roi in ROIS:
            v = np.array(bootstrap[band][roi][metric_key])
            if len(v):
                all_vals.append(v)

        if not all_vals:
            ax.set_visible(False)
            continue

        all_concat = np.concatenate(all_vals)
        rng_val = all_concat.max() - all_concat.min()
        if rng_val < 1e-10:
            rng_val = 0.1
        x_lo = all_concat.min() - 0.18 * rng_val
        x_hi = all_concat.max() + 0.18 * rng_val
        if xlim_ov:
            x_lo, x_hi = xlim_ov
        xs = np.linspace(x_lo, x_hi, 400)

        # One KDE per ROI
        for roi in ROIS:
            vals = np.array(bootstrap[band][roi][metric_key])
            if len(vals) < 5 or vals.std() < 1e-10:
                continue
            col = ROI_COLORS[roi]
            kde = gaussian_kde(vals, bw_method="scott")
            ys  = kde(xs)
            ax.fill_between(xs, ys, alpha=ROI_ALPHA_FILL, color=col, lw=0)
            ax.plot(xs, ys, color=col, lw=1.4, alpha=ROI_ALPHA_LINE)

            # Rug marks at the bottom
            rug = vals[::4]
            ax.plot(rug, np.zeros_like(rug), "|",
                    color=col, alpha=0.45, ms=3, mew=0.7,
                    transform=ax.get_xaxis_transform(), clip_on=True)

        # Reference line (ideal value)
        ax.axvline(ref_val, color=C_REF, lw=1.2, ls="--", zorder=6)

        ax.set_xlim(x_lo, x_hi)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7.5, pad=2)

        # Column title (top row only)
        if r == 0:
            ax.set_title(metric_label, fontsize=10.5, fontweight="bold", pad=6,
                         color="#1a1a2e")

        # Band label (leftmost column)
        if c == 0:
            ax.annotate(
                BAND_LABELS[band],
                xy=(-0.28, 0.5), xycoords="axes fraction",
                fontsize=9.5, fontweight="bold", color="#1a1a2e",
                ha="right", va="center", rotation=90,
                annotation_clip=False,
            )

        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_visible(False)

# Panel A label
fig.text(0.005, 0.962, "a", fontsize=14, fontweight="bold", va="top",
         style="italic")

# Legend in the spacer row (row index 1)
ax_leg = fig.add_subplot(gs_outer[1])
ax_leg.set_visible(False)
legend_handles = [
    Line2D([0], [0], color=ROI_COLORS[roi], lw=2.5,
           label=ROI_LABELS[roi])
    for roi in ROIS
] + [
    Line2D([0], [0], color=C_REF, lw=1.4, ls="--",
           label="Ideal reference value  (bootstrap n = 200)"),
]
fig.legend(
    handles=legend_handles,
    loc="center",
    bbox_to_anchor=(0.50, ax_leg.get_position().y0 + 0.018),
    fontsize=9, frameon=True, framealpha=0.92,
    edgecolor="#CCCCCC", handlelength=2.2,
    handletextpad=0.6, labelspacing=0.5, ncol=2,
)

# ─── PANEL B : Q-Q plots ──────────────────────────────────────────────────────
gs_B = gridspec.GridSpecFromSubplotSpec(
    1, N_ROWS,
    subplot_spec=gs_outer[2],
    wspace=0.42,
)

for b_idx, band in enumerate(BANDS):
    ax = fig.add_subplot(gs_B[b_idx])

    z_pool = df_test[df_test["band"] == band]["z_score"].dropna().values
    if len(z_pool) < 10:
        ax.set_visible(False)
        continue

    z_sorted = np.sort(z_pool)
    n = len(z_sorted)
    theoretical_q = stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))

    ax.scatter(theoretical_q, z_sorted,
               s=5, color="#2C7BB6", alpha=0.60, linewidths=0, zorder=3)

    lim = max(abs(theoretical_q).max(), abs(z_sorted).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim],
            color="#333333", lw=1.2, ls="--", zorder=4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Theoretical quantiles", fontsize=8, labelpad=3)
    if b_idx == 0:
        ax.set_ylabel("Sample quantiles", fontsize=8, labelpad=3)

    ax.set_title(BAND_LABELS[band], fontsize=9.5, fontweight="bold", pad=5,
                 color="#1a1a2e")
    ax.tick_params(labelsize=7.5)
    ax.text(0.06, 0.93, f"n = {n}", transform=ax.transAxes,
            fontsize=7, va="top", color="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

fig.text(0.005, 0.310, "b", fontsize=14, fontweight="bold", va="top",
         style="italic")

# ── MAIN TITLE ────────────────────────────────────────────────────────────────
fig.suptitle(
    "BLR normative model – calibration on held-out test set",
    fontsize=12, fontweight="bold", y=0.975,
)

# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
out_pdf = os.path.join(SAVE_PATH, "calibration_validation_kde.pdf")
out_png = os.path.join(SAVE_PATH, "calibration_validation_kde.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight", format="png")
print(f"\nFigure saved:\n  {out_pdf}\n  {out_png}")
plt.close(fig)
