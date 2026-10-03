"""
Extract HC / MCI / AD group z-score deviation stats — same methodology as
prepare_datasets/fig_results.py (panel b), but for the HC, MCI, AD groups
requested by the user (ACr excluded).

Method (identical to fig_results.py):
  - z_score per subject/band/roi, one-sample two-sided Wilcoxon signed-rank
    test against 0, FDR (Benjamini-Hochberg) correction across all
    group x band x roi cells, significance stars from q-value.

Output: organized table (band x roi, one block per group) saved as CSV/XLSX
next to the source data, plus printed to stdout.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_config_params.paths import BLR_DIR
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA_PATH = (
    os.path.join(BLR_DIR, "blr_osc_pw_rel_canonic.csv")
)
OUT_DIR = os.path.dirname(DATA_PATH)

BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
ROIS  = ["F", "C", "P", "O", "PO"]
BAND_LABELS = {
    "theta": "Theta", "alpha1": "Alpha 1", "alpha2": "Alpha 2",
    "beta1": "Beta 1", "beta2": "Beta 2", "beta3": "Beta 3", "gamma": "Gamma",
}
ROI_LABELS = {"F": "Frontal", "C": "Central", "P": "Parietal",
              "O": "Occipital", "PO": "Parieto-Occipital"}

# Same set filter as fig_results.py, restricted to the groups requested
GROUPS      = ["HC", "MCI", "AD"]
SET_FILTER  = {"HC": "test", "MCI": "mci", "AD": "AD"}

df_raw = pd.read_csv(DATA_PATH)

frames = []
for grp, set_val in SET_FILTER.items():
    sub = df_raw[df_raw["set"] == set_val][
        ["subject", "band", "roi", "z_score"]].copy()
    sub["group"] = grp
    frames.append(sub)
df_z = pd.concat(frames, ignore_index=True)
df_z = df_z[df_z["band"].isin(BANDS) & df_z["roi"].isin(ROIS)]

records = []
for grp in GROUPS:
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
valid = df_stats["pval"].notna()
_, pvals_fdr, _, _ = multipletests(df_stats.loc[valid, "pval"].values, method="fdr_bh")
df_stats.loc[valid,  "pval_fdr"] = pvals_fdr
df_stats.loc[~valid, "pval_fdr"] = np.nan

def sig_stars(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""

df_stats["stars"] = df_stats["pval_fdr"].apply(sig_stars)
df_stats["cell"] = df_stats.apply(
    lambda r: f"{r['mean_z']:+.2f}{r['stars']}", axis=1)

# n unique subjects per group (headline count, like the figure titles)
n_subj = df_z.groupby("group")["subject"].nunique()

print("N subjects per group (matches figure panel titles):")
for g in GROUPS:
    print(f"  {g}: n = {n_subj.get(g, 0)}")
print()

pivot_tables = {}
for grp in GROUPS:
    sub = df_stats[df_stats["group"] == grp]
    pivot = sub.pivot(index="band", columns="roi", values="cell")
    pivot = pivot.reindex(index=BANDS, columns=ROIS)
    pivot.index = [BAND_LABELS[b] for b in pivot.index]
    pivot.columns = [ROI_LABELS[r] for r in pivot.columns]
    pivot_tables[grp] = pivot
    print(f"=== {grp} (n = {n_subj.get(grp, 0)}) — mean z-score (FDR stars) ===")
    print(pivot.to_string())
    print()

out_xlsx = os.path.join(OUT_DIR, "hc_ad_mci_zscore_stats.xlsx")
with pd.ExcelWriter(out_xlsx) as writer:
    for grp in GROUPS:
        pivot_tables[grp].to_excel(writer, sheet_name=grp)
    df_stats.to_excel(writer, sheet_name="raw_long", index=False)
print(f"Saved: {out_xlsx}")

out_csv = os.path.join(OUT_DIR, "hc_ad_mci_zscore_stats_long.csv")
df_stats.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
