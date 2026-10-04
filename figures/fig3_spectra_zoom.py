"""
Fig 3 — Group-average spectra (Unadjusted | Oscillatory Fit | Aperiodic) with
two zoomed call-outs on the Unadjusted Spectra panel (theta/alpha peak region
and the high-frequency 27-40 Hz tail, where ACr separates from the rest).

Variation on the spectra row of figures/graphs_manuscript.ipynb
(cell 1e7ba578): same data, same group colours, same electrodes-of-interest
averaging — but only the 3 spectra panels (no violin plots), plus zoom insets.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_config_params.paths import FOOOF_DB, IAF_ROIS_DIR, BLR_DIR, FIGURES_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = (
    FOOOF_DB
)
# ROI-aggregated, QC'd cohort (TF/IAFp + FOOOF R^2 exclusions already applied)
# that feeds the demographic table and the BLR/classification pipeline. The
# raw per-electrode DATA_PATH file above is pre-QC and has a few extra HC
# subjects (e.g. invalid TF/IAFp landmarks) that never entered any other
# analysis -- restrict to this cohort so Fig. 3's n matches everywhere else.
QC_COHORT_PATH = (
    os.path.join(IAF_ROIS_DIR, "rois_age.feather")
)
_DEFAULT_SAVE = (
    FIGURES_DIR
)
SAVE_PATH = os.environ.get("NMEEG_FIGURES_PATH", _DEFAULT_SAVE)

ELECTRODES  = ["FP1", "FP2", "C3", "C4", "O1", "O2", "P7", "P8"]
GROUP_ORDER = ["HC", "MCI", "ACr", "AD"]
GROUP_PALETTE = {
    "HC":  "#4472C4",
    "MCI": "#ED7D31",
    "ACr": "#70AD47",
    "AD":  "#A02C7F",
}
GROUP_MAPPING = {"DTA": "AD", "A": "AD", "SAN": None, "F": None}

FREQS_REF = np.linspace(1, 40, 200)

CALLOUT_COLOR = "#1A3A6B"

# Zoom windows on the Unadjusted Spectra panel (x in Hz; y auto from data)
ZOOM_WINDOWS = [
    {"xlim": (4.0, 16.0), "label": "peak", "title": "Theta-Alpha Peak (4-16 Hz)"},
    {"xlim": (27.0, 40.0), "label": "tail", "title": "High-Frequency Tail (27-40 Hz)"},
]

plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD + PREP DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_feather(DATA_PATH)
df["group"] = df["group"].replace(GROUP_MAPPING)
df = df[df["group"].isin(GROUP_ORDER)].copy()
# Greece contributes only 3 subjects total (2 HC, 1 AD) -- excluded study-wide.
df = df[df["SITE"] != "Greece"].copy()
df["uid"] = df["subject"].astype(str) + "||" + df["SITE"].astype(str)

_qc = pd.read_feather(QC_COHORT_PATH)
_qc_renombrar = {"G2": "HC", "GU": "HC", "CTR": "HC", "DCL": "MCI", "A": "AD", "DTA": "AD"}
_qc["group"] = _qc["group"].replace(_qc_renombrar)
_qc_uids = set(_qc["subject"].astype(str) + "||" + _qc["SITE"].astype(str))
df = df[df["uid"].isin(_qc_uids)].copy()
# MCI restricted to the sub-cohorts analysed in testblr.py (Seoul + Spain, n=341).
# The 8 MCI from Medellin are in the QC cohort but not in the normative or
# classification analyses, so they are left out here for consistency.
MCI_SITES = ["Seoul", "Spain"]
df = df[(df["group"] != "MCI") | df["SITE"].isin(MCI_SITES)].copy()

df["subject_site"] = df["subject"].astype(str) + "_" + df["SITE"].astype(str)


def compute_group_curves(component):
    """Per-group average spectrum: mean over electrodes-of-interest per subject,
    then mean over subjects. Returns {group: curve_array}, {group: n_subjects}."""
    curves, ns = {}, {}
    for group in GROUP_ORDER:
        gdf = df[df["group"] == group]
        if gdf.empty:
            continue
        idx = gdf.set_index(["subject_site", "Sensors"])
        subject_curves = []
        for subj in idx.index.get_level_values(0).unique():
            electrode_curves = []
            for electrode in ELECTRODES:
                key = (subj, electrode)
                if key not in idx.index:
                    continue
                row = idx.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                freqs    = row["freqs"]
                comp_val = row[component]
                electrode_curves.append(
                    np.interp(FREQS_REF, freqs[:len(comp_val)], comp_val))
            if electrode_curves:
                subject_curves.append(np.mean(electrode_curves, axis=0))
        if subject_curves:
            curves[group] = np.mean(subject_curves, axis=0)
            ns[group]     = len(subject_curves)
    return curves, ns


COMPONENTS = [
    ("psd",             "Unadjusted Spectra", True),
    ("oscilatory_psd",  "Oscillatory Fit",     False),
    ("aperiodic_comp",  "Aperiodic spectra",   False),
]

curve_data = {comp: compute_group_curves(comp) for comp, _, _ in COMPONENTS}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_theta_band(ax):
    ax.axvspan(6, 8.5, color="gray", alpha=0.3, zorder=0)
    ylim = ax.get_ylim()
    ax.text(7.25, ylim[0] + 0.05 * (ylim[1] - ylim[0]), r"$\theta$",
             fontsize=13, ha="center", va="center")


def plot_component(ax, component, log_scale, xlim=(1.5, 40)):
    curves, ns = curve_data[component]
    lines = {}
    for group in GROUP_ORDER:
        if group not in curves:
            continue
        line, = ax.plot(FREQS_REF, curves[group],
                         color=GROUP_PALETTE[group], lw=2.2,
                         label=f"{group} (n={ns[group]})")
        lines[group] = line
    ax.set_xlim(*xlim)
    if log_scale:
        ax.set_yscale("log")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10.5))
gs = gridspec.GridSpec(
    1, 3, figure=fig,
    wspace=0.32,
    top=0.92, bottom=0.60, left=0.07, right=0.98,
)

axes_main = [fig.add_subplot(gs[0, i]) for i in range(3)]
legend_handles = {}

for ax, (component, title, log_scale) in zip(axes_main, COMPONENTS):
    lines = plot_component(ax, component, log_scale)
    for g, ln in lines.items():
        legend_handles.setdefault(g, ln)
    draw_theta_band(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    style_axes(ax)

ax_unadj = axes_main[0]

# ── Legend (top, one entry per group in fixed order) ──────────────────────────
ordered_handles = [legend_handles[g] for g in GROUP_ORDER if g in legend_handles]
fig.legend(ordered_handles, [h.get_label() for h in ordered_handles],
           loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=len(ordered_handles), frameon=False, fontsize=12,
           handlelength=1.8, columnspacing=1.6)

# ── Zoom insets, called out from the Unadjusted Spectra panel ────────────────
# "Balloon" style: two big rounded cards, centred as a pair under the full
# 3-panel row, each connected to its source region by a single bold arrow.
curves_psd, ns_psd = curve_data["psd"]

ZOOM_BOTTOM  = 0.09
ZOOM_HEIGHT  = 0.36
ZOOM_WIDTH   = 0.30
ZOOM_GAP     = 0.07
BORDER_LW    = 2.8
ARROW_LW     = 2.8
ROUNDING     = 0.06

full_left  = ax_unadj.get_position().x0
full_right = axes_main[-1].get_position().x1
pair_width = 2 * ZOOM_WIDTH + ZOOM_GAP
pair_left  = full_left + (full_right - full_left - pair_width) / 2
lefts = [pair_left, pair_left + ZOOM_WIDTH + ZOOM_GAP]


def add_rounded_border(ax, color, lw, rounding):
    for spine in ax.spines.values():
        spine.set_visible(False)
    box = FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=color, facecolor="none",
        clip_on=False, zorder=25, mutation_aspect=1,
    )
    ax.add_patch(box)


for spec, left in zip(ZOOM_WINDOWS, lefts):
    x0, x1 = spec["xlim"]
    xc_data = (x0 + x1) / 2
    ax_zoom = fig.add_axes([left, ZOOM_BOTTOM, ZOOM_WIDTH, ZOOM_HEIGHT])
    mask = (FREQS_REF >= x0) & (FREQS_REF <= x1)

    if spec["label"] == "peak":
        ax_zoom.axvspan(6, 8.5, color="gray", alpha=0.3, zorder=0)

    y_all = []
    for group in GROUP_ORDER:
        if group not in curves_psd:
            continue
        y = curves_psd[group][mask]
        ax_zoom.plot(FREQS_REF[mask], y, color=GROUP_PALETTE[group], lw=2.4)
        y_all.append(y)
    y_all = np.concatenate(y_all)
    # Log-space padding: the callout box is also drawn on ax_unadj, which is
    # log-scaled — linear padding can push y0 <= 0 and break that transform.
    log_y0, log_y1 = np.log10(y_all.min()), np.log10(y_all.max())
    pad = 0.10 * (log_y1 - log_y0)
    y0, y1 = 10 ** (log_y0 - pad), 10 ** (log_y1 + pad)

    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(y0, y1)
    ax_zoom.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax_zoom.set_xticks([x0, round((x0 + x1) / 2), x1])
    ax_zoom.tick_params(labelsize=11, length=0, colors="#222222", pad=6)
    ax_zoom.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_zoom.set_ylabel("Power", fontsize=12)
    ax_zoom.set_title(spec["title"], fontsize=13, fontweight="bold",
                      loc="left", y=1.12)
    add_rounded_border(ax_zoom, CALLOUT_COLOR, BORDER_LW, ROUNDING)

    # Call-out box on the parent (Unadjusted Spectra) panel
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                      transform=ax_unadj.transData, fill=False,
                      edgecolor=CALLOUT_COLOR, lw=2.0, zorder=10)
    ax_unadj.add_patch(rect)

    # Single bold "balloon" arrow: box mid-bottom -> inset top-centre
    con = ConnectionPatch(
        xyA=(xc_data, y0), coordsA=ax_unadj.transData,
        xyB=(0.5, 1.0), coordsB=ax_zoom.transAxes,
        arrowstyle="-|>", mutation_scale=22,
        color=CALLOUT_COLOR, lw=ARROW_LW, zorder=24, shrinkA=0, shrinkB=2)
    fig.add_artist(con)

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
for fmt in ("pdf", "png"):
    out = os.path.join(SAVE_PATH, f"fig3_spectra_zoom.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
    print(f"Saved: {out}")
plt.close(fig)
