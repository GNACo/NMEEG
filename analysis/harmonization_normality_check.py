"""
Normality diagnostic for the ComBat-GAM harmonization of relative oscillatory power.

Reviewer concern: ComBat's empirical-Bayes step assumes approximately normal,
homoscedastic residuals per feature. Oscillatory relative power is bounded in
(0,1) and compositional, so this assumption is questionable on the raw scale.

This script fits, for each band x ROI feature and on the HC training sample
(same global split as testblr.py and harmonize_neuroharmonize.py), a simple
age + age^2 + site model, and compares the residual distribution on the raw
(0,1) scale versus the logit scale: skewness, excess kurtosis and the
Shapiro-Wilk normality test. It is a diagnostic for the Methods/supplementary
response, not part of the harmonization itself (harmonize_neuroharmonize.py
already harmonizes band columns on the logit scale).

Run with the analysis environment:
    python analysis/harmonization_normality_check.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import logit as logit_fn
from scipy.stats import shapiro, skew, kurtosis
from sklearn.model_selection import StratifiedShuffleSplit

from run_config_params.paths import RECOMBAT_DIR, BLR_DIR

ROIS = ["F", "C", "P", "O", "PO"]
BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
EXCLUDED_SITES = ["Greece"]
SRC_DIR = os.path.join(RECOMBAT_DIR, "features_osc", "age_group")
LOGIT_EPS = 1e-3


def uid(d):
    return d["subject"].astype(str) + "||" + d["SITE"].astype(str)


def hc_training_uids():
    """Same global HC split as testblr.py / harmonize_neuroharmonize.py."""
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
    return set(hcg.iloc[tr]["uid"].tolist())


def ols_residuals(y, age, site):
    """Residuals of y ~ 1 + age + age^2 + one-hot(site), via least squares."""
    site_dum = pd.get_dummies(site, drop_first=True).values.astype(float)
    X = np.column_stack([np.ones_like(age), age, age ** 2, site_dum])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def diagnostics(resid):
    n = len(resid)
    # Shapiro-Wilk is unreliable/slow far beyond a few thousand points; subsample if needed.
    r = resid if n <= 4000 else np.random.default_rng(0).choice(resid, 4000, replace=False)
    try:
        _, p_sw = shapiro(r)
    except Exception:
        p_sw = np.nan
    return skew(resid), kurtosis(resid), p_sw  # kurtosis() is excess kurtosis (0 = normal)


train_uids = hc_training_uids()
rows = []
for roi in ROIS:
    path = os.path.join(SRC_DIR, f"osc_pw_rel_canonic_roi{roi}_SITE_age_group_recombat.xlsx")
    d = pd.read_excel(path, sheet_name="unharmonizeSITE_age_group")
    d = d[~d["SITE"].isin(EXCLUDED_SITES)].reset_index(drop=True)
    d = d[(d["group"] == "HC") & uid(d).isin(train_uids)].reset_index(drop=True)
    age = d["age"].values.astype(float)
    site = d["SITE"].astype(str).values
    for band in BANDS:
        col = f"osc_pw_canonic_{band}_roi{roi}"
        y_raw = d[col].values.astype(float)
        y_logit = logit_fn(np.clip(y_raw, LOGIT_EPS, 1 - LOGIT_EPS))

        s_raw, k_raw, p_raw = diagnostics(ols_residuals(y_raw, age, site))
        s_log, k_log, p_log = diagnostics(ols_residuals(y_logit, age, site))

        rows.append(dict(roi=roi, band=band, n=len(d),
                         skew_raw=s_raw, kurtosis_raw=k_raw, shapiro_p_raw=p_raw,
                         skew_logit=s_log, kurtosis_logit=k_log, shapiro_p_logit=p_log))
    print(f"done roi {roi}", flush=True)

out = pd.DataFrame(rows)
out_path = os.path.join(BLR_DIR, "harmonization_normality_check.csv")
out.to_csv(out_path, index=False)

n_tot = len(out)
print(f"\n{n_tot} band x ROI features (HC training, n per feature: {out.n.iloc[0]})")
print("Raw scale:   |skew|>1 in", (out.skew_raw.abs() > 1).sum(), f"/{n_tot}",
      " | Shapiro p<.05 in", (out.shapiro_p_raw < 0.05).sum(), f"/{n_tot}")
print("Logit scale: |skew|>1 in", (out.skew_logit.abs() > 1).sum(), f"/{n_tot}",
      " | Shapiro p<.05 in", (out.shapiro_p_logit < 0.05).sum(), f"/{n_tot}")
print(f"\nsaved {out_path}")
print(out.round(3).to_string(index=False))
