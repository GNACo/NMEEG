"""
Sensitivity analysis for HC vs ACr: age-matched healthy reference.

The main HC reference (held-out 20% test split) spans 47-84 years, whereas ACr
spans 20-45 years, so the two groups do not overlap in age. This script checks
how much the ACr classification depends on that choice.

Design:
- Out-of-fold (OOF) z-scores for ALL HC: 5-fold cross-fitting within HC,
  age-stratified. Each HC z-score comes from a model that did not see that
  subject (avoids the in-sample shrinkage of training-set residuals).
- ACr z-scores: from a model fit on all HC (ACr never used for fitting).
- Same preprocessing as testblr.py: ReComBat-harmonized features, Greece
  excluded, IQR outlier filter on HC (linear scale), logit transform for the
  bounded relative-power bands, BLR with production settings.
- Hyperparameters (sigma2, alpha2) are optimized once on the full HC sample
  and reused across folds (documented simplification).
- Classification uses classify_blr.classify_zscore_all_features (same CV,
  balancing, metrics and bootstrap as the main analysis).

Two references are compared:
  (a) all HC (OOF)              -> baseline for this procedure
  (b) HC restricted to 20-45 y  -> age-matched to ACr
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
EPS = 1e-3
KP = {"number_of_bases": 30, "length_scale": 1.5, "bases_sampling_method": "KMeans"}


def to_logit(y):
    return logit_fn(np.clip(y, EPS, 1 - EPS))


def zscore(model, x, y):
    y_pred = model.predict_with_samples(x, n_samples=500)[0]
    total_var, _, _ = model.get_variance_decomposition(x)
    total_var = total_var * model.y_std ** 2
    return (y - y_pred) / np.sqrt(total_var)


def load_roi(roi):
    path = os.path.join(HARM_OUT_DIR, f"{FAMILY}_roi{roi}_SITE_age_group_{HARM_SUFFIX}.xlsx")
    d = pd.read_excel(path, sheet_name="harmonizeSITE_age_group")
    d = d[(d["SITE"] != "Greece") & d["group"].isin(["HC", "ACr"])].copy()
    d["subject"] = d["subject"].astype(str)
    return d


rows = []
for roi in ROIS:
    data_roi = load_roi(roi)
    for band in BANDS:
        feat = f"harm_osc_pw_canonic_{band}_roi{roi}"
        d = data_roi[["subject", "SITE", "group", "age", feat]].dropna(subset=[feat, "age"]).copy()
        hc = d[d["group"] == "HC"].reset_index(drop=True)
        acr = d[d["group"] == "ACr"].reset_index(drop=True)
        x_hc = hc[["age"]].values.astype(float)
        y_hc_raw = hc[feat].values
        y_hc = to_logit(y_hc_raw)

        def train_mask(idx):
            # IQR bounds from the training rows only; never applied to held-out or ACr rows
            v = y_hc_raw[idx]
            q1, q3 = np.quantile(v, 0.25), np.quantile(v, 0.75)
            iqr = q3 - q1
            return (y_hc_raw[idx] >= q1 - 1.5 * iqr) & (y_hc_raw[idx] <= q3 + 1.5 * iqr)

        all_idx = np.arange(len(hc))
        ok = train_mask(all_idx)
        full = BayesianLinearRegression(KP, sigma2=0.06, alpha2=5)
        full.fit(x_hc[ok], y_hc[ok])
        with contextlib.redirect_stdout(io.StringIO()):
            full.optimize_hyperparameters()
        s2, a2 = full.sigma2, full.alpha2
        full = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
        full.fit(x_hc[ok], y_hc[ok])

        bins = pd.qcut(hc["age"], q=5, labels=False, duplicates="drop")
        z_oof = np.full(len(hc), np.nan)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(x_hc, bins):
            tr_ok = tr[train_mask(tr)]
            m = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
            m.fit(x_hc[tr_ok], y_hc[tr_ok])
            z_oof[te] = zscore(m, x_hc[te], y_hc[te])

        z_acr = zscore(full, acr[["age"]].values.astype(float), to_logit(acr[feat].values))

        for df_part, z, name, set_label in [
            (hc, z_oof, "HC", "test"),
            (acr, z_acr, "ACr", "ACr"),
        ]:
            rows.append(pd.DataFrame({
                "subject": df_part["subject"].values,
                "Site": df_part["SITE"].values,
                "group": name,
                "set": set_label,
                "age": df_part["age"].values,
                "feature": FAMILY,
                "band": band,
                "roi": roi,
                "z_score": z,
            }))
        print(f"done {roi}/{band}: HC={len(hc)} ACr={len(acr)} sigma2={s2:.3f} alpha2={a2:.2f}", flush=True)

long = pd.concat(rows, ignore_index=True)
acr_min = long.loc[long["group"] == "ACr", "age"].min()
acr_max = long.loc[long["group"] == "ACr", "age"].max()
print(f"ACr age range: {acr_min}-{acr_max}")

long_full = long.copy()
long_match = long[~((long["group"] == "HC") & ~long["age"].between(acr_min, acr_max))].copy()
n_hc_full = long_full.loc[long_full.group == "HC"].drop_duplicates(["subject", "Site"]).shape[0]
n_hc_match = long_match.loc[long_match.group == "HC"].drop_duplicates(["subject", "Site"]).shape[0]
print(f"HC in age-matched reference: {n_hc_match} | all HC: {n_hc_full}")

res_full, _ = classify_zscore_all_features(long_full, sets_incluidos=["test", "ACr"], n_bootstrap=50)
res_full["reference"] = f"all HC (OOF, n={n_hc_full})"
res_match, _ = classify_zscore_all_features(long_match, sets_incluidos=["test", "ACr"], n_bootstrap=50)
res_match["reference"] = f"HC aged {acr_min:.0f}-{acr_max:.0f} y (OOF, n={n_hc_match})"

out = pd.concat([res_full, res_match], ignore_index=True)
out_path = os.path.join(BLR_DIR, "sensitivity_acr_agematched.csv")
out.to_csv(out_path, index=False)
print(f"saved {out_path}")
print(out[["reference", "model", "AUC_mean", "AUC_std", "accuracy", "f1_score", "brier", "boot_AUC_mean", "boot_AUC_std"]]
      .round(3).to_string(index=False))
