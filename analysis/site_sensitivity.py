"""
Supplementary analyses for HC vs MCI, AD and ACr:

  (a) ACr vs an age-matched HC reference (HC aged 20-45 y, the range of ACr).
  (b) Site-restricted analyses: each comparison repeated within one site, or
      excluding Seoul, to show whether the results depend on the largest cohort.

Design (same modelling as the main analysis, extended to all groups):
- Features: osc_pw_rel_canonic, 7 bands x 5 ROIs (logit scale, bounded relative
  power) plus the aperiodic exponent per ROI (linear scale). 40 features, the
  same set used by testblr.py. Harmonized with neuroHarmonize (HARM_SUFFIX).
- Normative model per feature: BLR with RBF age bases and production settings,
  fit on all HC after an IQR filter on HC (linear scale). MCI, AD and ACr
  z-scores come from this model; no clinical participant is used for fitting.
- HC z-scores are out-of-fold (5 folds, stratified by age quintile), so every
  HC z-score comes from a model that did not see that participant. This makes
  the HC reference the full HC pool, restricted to the site or age range of
  each comparison.
- Hyperparameters (sigma2, alpha2) are optimized once on the full HC sample and
  reused across folds (documented simplification, as in the main analysis).
- Classification uses classify_blr.classify_zscore_all_features: complete-case
  pivot, in-fold undersampling, calibrated linear SVM, logistic regression,
  random forest, decision tree, 50-iteration bootstrap.

Note: this script must be run with the models/BLR.py used for the reported
results (see the commit message that introduced it). Running it with a different
alpha2 convention changes the fitted models.

Output: BLR_DIR/sensitivity_site.csv (one row per comparison and classifier).
"""
import os
import sys
import contextlib
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import logit as logit_fn
from sklearn.model_selection import StratifiedKFold

from models.BLR import BayesianLinearRegression
from classify_blr import classify_zscore_all_features
from run_config_params.paths import HARM_OUT_DIR, BLR_DIR, HARM_SUFFIX

ROIS = ["F", "C", "P", "O", "PO"]
BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
FAMILY = "osc_pw_rel_canonic"
GROUPS = ["HC", "MCI", "AD", "ACr"]
EPS = 1e-3
KP = {"number_of_bases": 30, "length_scale": 1.5, "bases_sampling_method": "KMeans"}
MEDELLIN = ["Medellin_hd", "Medellin_ld"]
EXCLUDED_SITES = ["Greece"]
MCI_SITES = ["Seoul", "Spain"]


def to_model(y, bounded):
    return logit_fn(np.clip(y, EPS, 1 - EPS)) if bounded else y


def zscore(model, x, y):
    y_pred = model.predict_with_samples(x, n_samples=500)[0]
    total_var, _, _ = model.get_variance_decomposition(x)
    total_var = total_var * model.y_std ** 2
    return (y - y_pred) / np.sqrt(total_var)


def iqr_keep(values):
    """Boolean mask of values inside the 1.5 x IQR bounds computed on `values` itself."""
    q1, q3 = np.quantile(values, 0.25), np.quantile(values, 0.75)
    iqr = q3 - q1
    return (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)


def load_roi(roi):
    path = os.path.join(HARM_OUT_DIR, f"{FAMILY}_roi{roi}_SITE_age_group_{HARM_SUFFIX}.xlsx")
    d = pd.read_excel(path, sheet_name="harmonizeSITE_age_group")
    d = d[~d["SITE"].isin(EXCLUDED_SITES) & d["group"].isin(GROUPS)].copy()
    d["subject"] = d["subject"].astype(str)
    # MCI: same sub-cohorts as the normative model (testblr.py)
    d = d[(d["group"] != "MCI") | d["SITE"].isin(MCI_SITES)]
    return d


def features(roi):
    feats = [(f"band:{b}", f"harm_osc_pw_canonic_{b}_roi{roi}", True, b) for b in BANDS]
    feats.append(("exponent", f"harm_exponent_roi{roi}", False, None))
    return feats


# Load all data once to know the site list and the ACr age range
_all = pd.concat([load_roi(r) for r in ROIS])
ALL_SITES = sorted(_all["SITE"].unique())
NON_SEOUL = [s for s in ALL_SITES if s not in ("Seoul",)]
ACR_AGE_MIN = float(_all.loc[_all["group"] == "ACr", "age"].min())
ACR_AGE_MAX = float(_all.loc[_all["group"] == "ACr", "age"].max())

rows = []
for roi in ROIS:
    data_roi = load_roi(roi)
    for name, col, bounded, band in features(roi):
        d = data_roi[["subject", "SITE", "group", "age", col]].dropna(subset=[col, "age"]).reset_index(drop=True)
        is_hc = (d["group"] == "HC").values
        x_all = d[["age"]].values.astype(float)
        y_raw = d[col].values.astype(float)
        y_all = to_model(y_raw, bounded)
        hc_idx = np.where(is_hc)[0]
        x_hc, y_hc_raw, y_hc = x_all[hc_idx], y_raw[hc_idx], y_all[hc_idx]

        # Full-HC model (IQR on HC only), hyperparameters optimized once
        ok = iqr_keep(y_hc_raw)
        full = BayesianLinearRegression(KP, sigma2=0.06, alpha2=5)
        full.fit(x_hc[ok], y_hc[ok])
        with contextlib.redirect_stdout(io.StringIO()):
            full.optimize_hyperparameters()
        s2, a2 = full.sigma2, full.alpha2
        full = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
        full.fit(x_hc[ok], y_hc[ok])

        # Out-of-fold z-scores for HC (IQR bounds from each training fold only)
        bins = pd.qcut(d.loc[hc_idx, "age"], q=5, labels=False, duplicates="drop")
        z_hc = np.full(len(hc_idx), np.nan)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(x_hc, bins):
            tr_ok = tr[iqr_keep(y_hc_raw[tr])]
            m = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
            m.fit(x_hc[tr_ok], y_hc[tr_ok])
            z_hc[te] = zscore(m, x_hc[te], y_hc[te])

        z_all = np.full(len(d), np.nan)
        z_all[hc_idx] = z_hc
        non_hc = np.where(~is_hc)[0]
        z_all[non_hc] = zscore(full, x_all[non_hc], y_all[non_hc])

        rows.append(pd.DataFrame({
            "subject": d["subject"].values,
            "Site": d["SITE"].values,
            "group": d["group"].values,
            "age": d["age"].values,
            "feature": "exponent" if band is None else FAMILY,
            "band": band,
            "roi": roi,
            "z_score": z_all,
        }))
        print(f"done {roi}/{name}: HC={is_hc.sum()} sigma2={s2:.3f} alpha2={a2:.2f}", flush=True)

long = pd.concat(rows, ignore_index=True)
long["set"] = long["group"].map({"HC": "test", "MCI": "mci", "AD": "AD", "ACr": "ACr"})

# Comparisons: reference HC subset, clinical group, and sites used for the clinical group
COMPARISONS = [
    dict(ref="All sites", hc_sites=None, hc_age=None, group="MCI", grp_sites=MCI_SITES),
    dict(ref="All sites", hc_sites=None, hc_age=None, group="AD", grp_sites=None),
    dict(ref="All sites", hc_sites=None, hc_age=None, group="ACr", grp_sites=None),
    dict(ref=f"HC aged {ACR_AGE_MIN:.0f}-{ACR_AGE_MAX:.0f} y", hc_sites=None, hc_age="acr", group="ACr", grp_sites=None),
    dict(ref="Seoul only", hc_sites=["Seoul"], hc_age=None, group="MCI", grp_sites=["Seoul"]),
    dict(ref="Seoul only", hc_sites=["Seoul"], hc_age=None, group="AD", grp_sites=["Seoul"]),
    dict(ref="Medellin only", hc_sites=MEDELLIN, hc_age=None, group="ACr", grp_sites=None),
    dict(ref="Spain only", hc_sites=["Spain"], hc_age=None, group="MCI", grp_sites=["Spain"]),
    dict(ref="Excluding Seoul", hc_sites=NON_SEOUL, hc_age=None, group="MCI", grp_sites=["Spain"]),
    dict(ref="Excluding Seoul", hc_sites=NON_SEOUL, hc_age=None, group="ACr", grp_sites=None),
]

out_rows = []
for c in COMPARISONS:
    hc = long[long["group"] == "HC"]
    if c["hc_sites"] is not None:
        hc = hc[hc["Site"].isin(c["hc_sites"])]
    if c["hc_age"] == "acr":
        hc = hc[hc["age"].between(ACR_AGE_MIN, ACR_AGE_MAX)]
    g = long[long["group"] == c["group"]]
    if c["grp_sites"] is not None:
        g = g[g["Site"].isin(c["grp_sites"])]
    df_cmp = pd.concat([hc, g], ignore_index=True)
    n_hc = hc[["subject", "Site"]].drop_duplicates().shape[0]
    n_g = g[["subject", "Site"]].drop_duplicates().shape[0]
    print(f"{c['ref']} | {c['group']} vs HC: HC={n_hc}, {c['group']}={n_g}", flush=True)
    res, _ = classify_zscore_all_features(df_cmp, sets_incluidos=["test", c["group"] if c["group"] == "ACr" else
                                                                   {"MCI": "mci", "AD": "AD"}[c["group"]]],
                                          n_bootstrap=50)
    res["reference"] = c["ref"]
    res["comparison"] = f"HC vs {c['group']}"
    res["n_HC"] = n_hc
    res[f"n_{c['group']}"] = n_g
    out_rows.append(res)

out = pd.concat(out_rows, ignore_index=True)
out_path = os.path.join(BLR_DIR, "sensitivity_site.csv")
out.to_csv(out_path, index=False)
print(f"saved {out_path}")
print(out[["comparison", "reference", "model", "n_HC", "AUC_mean", "AUC_std", "accuracy", "f1_score", "brier",
           "boot_AUC_mean", "boot_AUC_std"]].round(3).to_string(index=False))
