"""
Site harmonization with neuroHarmonize (ComBat-GAM), learn/apply design.

- The site-correction model is learned ONLY on the HC training participants of
  the global split used by the normative model (testblr.py), so no patient
  data and no diagnostic label influence the correction.
- Covariates: SITE (batch) and age, with age modeled non-linearly (smooth_terms).
- The learned model is applied to every participant (HC test, MCI, AD, ACr).

Run with the neuroHarmonize environment (see requirements-harmonize.txt):
    python harmonize/harmonize_neuroharmonize.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from neuroHarmonize import harmonizationLearn, harmonizationApply

from run_config_params.paths import RECOMBAT_DIR, HARM_OUT_DIR, HARM_SUFFIX

ROIS = ["F", "C", "P", "O", "PO"]
FAMILIES = ["osc_pw_rel_canonic", "osc_pw_ab_canonic"]
EXCLUDED_SITES = ["Greece"]
SRC_DIR = os.path.join(RECOMBAT_DIR, "features_osc", "age_group")


def uid(d):
    return d["subject"].astype(str) + "||" + d["SITE"].astype(str)


def hc_training_uids():
    """Same global HC split as testblr.py (rows, order and seed reproduced)."""
    recs = []
    for roi in ROIS:
        path = os.path.join(SRC_DIR, f"osc_pw_rel_canonic_roi{roi}_SITE_age_group_recombat.xlsx")
        d = pd.read_excel(path, sheet_name="harmonizeSITE_age_group")
        d = d[~d["SITE"].isin(EXCLUDED_SITES)]
        recs.append(d[d["group"] == "HC"][["subject", "SITE", "age"]].dropna())
    hcg = pd.concat(recs).reset_index(drop=True)
    hcg["uid"] = uid(hcg)
    hcg = hcg.drop_duplicates(subset="uid").reset_index(drop=True)
    bins = pd.qcut(hcg["age"], q=5, labels=False, duplicates="drop")
    tr, _ = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(hcg, bins))
    return set(hcg.iloc[tr]["uid"].tolist()), hcg


def main():
    os.makedirs(HARM_OUT_DIR, exist_ok=True)
    train_uids, _ = hc_training_uids()
    print(f"HC training participants (global split): {len(train_uids)}")

    for family in FAMILIES:
        for roi in ROIS:
            src = os.path.join(SRC_DIR, f"{family}_roi{roi}_SITE_age_group_recombat.xlsx")
            d = pd.read_excel(src, sheet_name="unharmonizeSITE_age_group")
            d = d[~d["SITE"].isin(EXCLUDED_SITES)].reset_index(drop=True)
            feats = [c for c in d.columns if c.endswith(f"_roi{roi}")]
            assert d[feats].notna().all().all(), f"missing values in {family} roi{roi}"

            X = d[feats].values.astype(float)
            covars = d[["SITE", "age"]].copy()
            covars["SITE"] = covars["SITE"].astype(str)
            is_train = (d["group"] == "HC") & uid(d).isin(train_uids)

            missing_sites = set(covars["SITE"]) - set(covars.loc[is_train, "SITE"])
            assert not missing_sites, f"sites without HC training data: {missing_sites}"

            model, _ = harmonizationLearn(X[is_train.values], covars[is_train.values],
                                          smooth_terms=["age"], eb=True)
            X_h = harmonizationApply(X, covars, model)

            harm = d[["subject", "group", "SITE", "age"]].copy()
            for j, f in enumerate(feats):
                harm[f"harm_{f}"] = X_h[:, j]

            out = os.path.join(HARM_OUT_DIR, f"{family}_roi{roi}_SITE_age_group_{HARM_SUFFIX}.xlsx")
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                d.to_excel(w, sheet_name="unharmonizeSITE_age_group", index=False)
                harm.to_excel(w, sheet_name="harmonizeSITE_age_group", index=False)
            n_train = int(is_train.sum())
            print(f"{family} roi{roi}: learned on {n_train} HC, applied to {len(d)} rows -> {os.path.basename(out)}")

    # Per-site HC training counts, for the manuscript
    d = pd.read_excel(os.path.join(SRC_DIR, f"osc_pw_rel_canonic_roiO_SITE_age_group_recombat.xlsx"),
                      sheet_name="unharmonizeSITE_age_group")
    d = d[~d["SITE"].isin(EXCLUDED_SITES)]
    counts = d[(d["group"] == "HC") & uid(d).isin(train_uids)].groupby("SITE").size()
    print("HC training participants per site:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
