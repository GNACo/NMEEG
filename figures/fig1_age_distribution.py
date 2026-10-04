"""
Figure 1 - Age distribution of the pooled sample and by recording site.

Built from the quality-controlled cohort (IAF_FOOOF_ROIS/rois_age.feather), the same
one used by the normative and classification analyses. Greece is excluded.
Participants are identified by (subject, SITE) because subject codes are reused
across sites.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from run_config_params.paths import IAF_ROIS_DIR, FIGURES_DIR

ROIS_AGE = os.path.join(IAF_ROIS_DIR, "rois_age.feather")
GROUPS = ["HC", "ACr", "MCI", "AD"]
COLORS = {"HC": "#2F6FBF", "ACr": "#27A6A0", "MCI": "#7B3FA0", "AD": "#E0A526"}
LABELS = {"HC": "HC", "ACr": "ACr", "MCI": "MCI", "AD": "AD"}
RENAME = {"G2": "HC", "GU": "HC", "CTR": "HC", "DCL": "MCI", "A": "AD", "DTA": "AD"}
EXCLUDED_SITES = ["Greece"]
SITE_ORDER = {
    "HC": ["Argentina", "Chile", "Cuba", "Dortmund", "Medellin_hd", "Medellin_ld",
           "Oslo", "Poland", "Seoul", "Spain"],
    "MCI": ["Medellin_hd", "Medellin_ld", "Seoul", "Spain"],
    "ADACR": ["Medellin_hd", "Medellin_ld", "Seoul"],
}
SITE_DISPLAY = {"Medellin_hd": "Medellín (hd)", "Medellin_ld": "Medellín (ld)"}


def load():
    d = pd.read_feather(ROIS_AGE)
    d["group"] = d["group"].replace(RENAME)
    d = d[d["group"].isin(GROUPS) & ~d["SITE"].isin(EXCLUDED_SITES)].copy()
    d["uid"] = d["subject"].astype(str) + "||" + d["SITE"].astype(str)
    return d.drop_duplicates("uid").reset_index(drop=True)


def violins(ax, datasets, positions, colors):
    parts = ax.violinplot([x for x in datasets], positions=positions, widths=0.8,
                          showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for pos, x, c in zip(positions, datasets, colors):
        if len(x) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(x))
        ax.scatter(pos + jitter, x, s=6, color=c, alpha=0.6, linewidths=0, zorder=3)
        bp = ax.boxplot([x], positions=[pos], widths=0.18, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.6),
                        boxprops=dict(facecolor="white", edgecolor="black", lw=1.1),
                        whiskerprops=dict(color="black", lw=1.0),
                        capprops=dict(color="black", lw=1.0))
        bp["boxes"][0].set_zorder(4)


def pooled_panel(ax, d):
    for i, g in enumerate(GROUPS):
        x = d.loc[d["group"] == g, "age"].values
        ys = np.linspace(x.min() - 3, x.max() + 3, 300)
        dens = gaussian_kde(x)(ys)
        dens = dens / dens.max() * 0.42
        ax.fill_betweenx(ys, i - dens, i + dens, color=COLORS[g], alpha=0.45, lw=0, zorder=1)
        ax.plot(i - dens, ys, color=COLORS[g], lw=0.8, zorder=1)
        ax.plot(i + dens, ys, color=COLORS[g], lw=0.8, zorder=1)
        jitter = np.random.default_rng(1).uniform(0.12, 0.38, size=len(x)) + i
        ax.scatter(jitter, x, s=4, color=COLORS[g], alpha=0.35, linewidths=0, zorder=2)
        bp = ax.boxplot([x], positions=[i], widths=0.16, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4),
                        boxprops=dict(facecolor="white", edgecolor="black", lw=1.0),
                        whiskerprops=dict(color="black", lw=0.9),
                        capprops=dict(color="black", lw=0.9))
        for part in ("boxes", "whiskers", "caps", "medians"):
            for artist in bp[part]:
                artist.set_zorder(4)
        ax.text(i, ys.max() + 6, f"n = {len(x)}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([LABELS[g] for g in GROUPS], fontsize=12)
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_ylabel("Age (years)", fontsize=12)
    ax.set_title("Pooled sample", fontsize=13, fontweight="bold")
    ax.set_ylim(5, 110)


def site_panel(ax, d, groups, sites, title, legend_groups):
    positions, datasets, colors = [], [], []
    tick_pos, tick_labels = [], []
    for k, s in enumerate(sites):
        for j, g in enumerate(groups):
            x = d.loc[(d["group"] == g) & (d["SITE"] == s), "age"].values
            if len(x) == 0:
                continue
            offset = (j - (len(groups) - 1) / 2) * 0.32
            positions.append(k + offset)
            datasets.append(x)
            colors.append(COLORS[g])
        counts = [f"{LABELS[g]}={int(((d['group'] == g) & (d['SITE'] == s)).sum())}"
                  for g in groups if ((d["group"] == g) & (d["SITE"] == s)).any()]
        name = SITE_DISPLAY.get(s, s)
        tick_pos.append(k)
        tick_labels.append(name + chr(10) + chr(10).join(counts))
    violins(ax, datasets, positions, colors)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8.5)
    ax.set_xlim(-0.6, len(sites) - 0.4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Age (years)", fontsize=12)
    ax.set_ylim(5, 110)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLORS[g], label=LABELS[g]) for g in legend_groups]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False)


def main():
    d = load()
    print(d["group"].value_counts().to_string())

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.2], height_ratios=[1, 1, 1], hspace=0.85, wspace=0.18)

    ax0 = fig.add_subplot(gs[:, 0])
    pooled_panel(ax0, d)

    ax1 = fig.add_subplot(gs[0, 1])
    site_panel(ax1, d, ["HC"], SITE_ORDER["HC"], "By site: HC", ["HC"])

    ax2 = fig.add_subplot(gs[1, 1])
    site_panel(ax2, d, ["MCI"], SITE_ORDER["MCI"], "By site: MCI", ["MCI"])

    ax3 = fig.add_subplot(gs[2, 1])
    site_panel(ax3, d, ["ACr", "AD"], SITE_ORDER["ADACR"], "By site: AD and ACr", ["ACr", "AD"])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"fig1_age_distribution.{ext}")
        fig.savefig(out, dpi=250, bbox_inches="tight")
        print("saved", out)


if __name__ == "__main__":
    main()
