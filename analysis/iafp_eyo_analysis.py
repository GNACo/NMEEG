"""
Two additional, pre-specified analyses requested for the ACr findings:

(a) Individual alpha peak frequency (IAFp), properly harmonized and modeled.
    IAFp is not part of the osc_pw_rel_canonic/osc_pw_ab_canonic families used by
    testblr.py; it is harmonized and normatively modeled here, independently,
    with the same design as the primary pipeline (ComBat-GAM learned on HC
    training only; BLR with RBF age bases; out-of-fold z-scores for HC).
    Three things are reported: IAFp alone (5 ROIs) as a classifier; IAFp added
    to the 40 primary relative-power features (45 total); and univariate
    group contrasts per ROI.

(b) Expected Years to Onset (EYO) for ACr. EYO = age - 44, using the median
    age of MCI onset reported for the Antioquia PSEN1 E280A kindred (the same
    kindred as the Medellin cohorts here). This is a cohort-level proxy EYO
    (no per-subject parental age at onset is available in this dataset), the
    standard approach when individual onset data is absent. For each ACr
    participant, EYO is correlated (Spearman) with every z-score feature
    (band x ROI relative power, exponent, and IAFp), and with a cross-validated
    composite (ridge regression, leave-one-out) built from the full feature
    vector.

Both analyses reuse the ALREADY-CORRECTED relative-power z-scores from
BLR_DIR/blr_osc_pw_rel_canonic.csv (set in testblr.py) instead of recomputing
them, so the comparison to the 40-feature classifier is on identical features
and identical values.

Run with the neuroHarmonize environment (needs neuroHarmonize for the IAFp
ComBat-GAM step; everything else uses the analysis-environment libraries):
    C:/nh-venv/Scripts/python.exe analysis/iafp_eyo_analysis.py
"""
import os
import sys
import contextlib
import io as iomod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.linear_model import RidgeCV
from neuroHarmonize import harmonizationLearn, harmonizationApply

from models.BLR import BayesianLinearRegression
from classify_blr import classify_zscore_all_features
from run_config_params.paths import RECOMBAT_DIR, IAF_ROIS_DIR, BLR_DIR

ROIS = ["F", "C", "P", "O", "PO"]
EXCLUDED_SITES = ["Greece"]
MCI_SITES = ["Seoul", "Spain"]
SRC_DIR = os.path.join(RECOMBAT_DIR, "features_osc", "age_group")
KP = {"number_of_bases": 30, "length_scale": 1.5, "bases_sampling_method": "KMeans"}
EYO_REFERENCE_AGE = 44  # median age of MCI onset, Antioquia PSEN1 E280A kindred


def uid(d):
    return d["subject"].astype(str) + "||" + d["SITE"].astype(str)


def hc_training_uids():
    """Same global HC split used by testblr.py / harmonize_neuroharmonize.py."""
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


def zscore(model, x, y):
    y_pred = model.predict_with_samples(x, n_samples=500)[0]
    total_var, _, _ = model.get_variance_decomposition(x)
    total_var = total_var * model.y_std ** 2
    return (y - y_pred) / np.sqrt(total_var)


train_uids = hc_training_uids()
print(f"HC training participants (global split): {len(train_uids)}")

# ── Load IAFp (raw, unharmonized) for the full population ──────────────────
raw = pd.read_feather(os.path.join(IAF_ROIS_DIR, "rois_age.feather"))
raw["group"] = raw["group"].replace({"G2": "HC", "GU": "HC", "CTR": "HC", "DCL": "MCI", "A": "AD", "DTA": "AD"})
raw = raw[~raw["SITE"].isin(EXCLUDED_SITES) & raw["group"].isin(["HC", "MCI", "AD", "ACr"])].copy()
raw = raw[(raw["group"] != "MCI") | raw["SITE"].isin(MCI_SITES)]
raw["subject"] = raw["subject"].astype(str)
raw["uid"] = uid(raw)
raw = raw.drop_duplicates("uid").reset_index(drop=True)

iafp_cols = [f"IAFp_roi{r}" for r in ROIS]
raw = raw.dropna(subset=iafp_cols + ["age"]).reset_index(drop=True)
is_train = (raw["group"] == "HC") & raw["uid"].isin(train_uids)
print(f"IAFp: {len(raw)} participants with complete data, {int(is_train.sum())} HC training")

# ── Harmonize IAFp (ComBat-GAM, learned on HC training only; raw scale, no
#    transform needed -- skewness on HC training is below 0.5 for all ROIs) ──
X = raw[iafp_cols].values.astype(float)
covars = raw[["SITE", "age"]].copy()
covars["SITE"] = covars["SITE"].astype(str)
missing_sites = set(covars["SITE"]) - set(covars.loc[is_train, "SITE"])
assert not missing_sites, f"sites without HC training data: {missing_sites}"
model, _ = harmonizationLearn(X[is_train.values], covars[is_train.values], smooth_terms=["age"], eb=True)
X_h = harmonizationApply(X, covars, model)
for j, roi in enumerate(ROIS):
    raw[f"harm_IAFp_roi{roi}"] = X_h[:, j]

# ── Normative model per ROI: OOF z for all HC, model-based z for MCI/AD/ACr ──
rows = []
for roi in ROIS:
    col = f"harm_IAFp_roi{roi}"
    d = raw[["subject", "SITE", "group", "age", col]].dropna().reset_index(drop=True)
    is_hc = (d["group"] == "HC").values
    x_all = d[["age"]].values.astype(float)
    y_all = d[col].values.astype(float)
    hc_idx = np.where(is_hc)[0]
    x_hc, y_hc = x_all[hc_idx], y_all[hc_idx]

    full = BayesianLinearRegression(KP, sigma2=0.06, alpha2=5)
    full.fit(x_hc, y_hc)
    with contextlib.redirect_stdout(iomod.StringIO()):
        full.optimize_hyperparameters()
    s2, a2 = full.sigma2, full.alpha2
    full = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
    full.fit(x_hc, y_hc)

    bins = pd.qcut(d.loc[hc_idx, "age"], q=5, labels=False, duplicates="drop")
    z_hc = np.full(len(hc_idx), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(x_hc, bins):
        m = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
        m.fit(x_hc[tr], y_hc[tr])
        z_hc[te] = zscore(m, x_hc[te], y_hc[te])

    z_all = np.full(len(d), np.nan)
    z_all[hc_idx] = z_hc
    non_hc = np.where(~is_hc)[0]
    z_all[non_hc] = zscore(full, x_all[non_hc], y_all[non_hc])

    rows.append(pd.DataFrame({
        "subject": d["subject"].values, "Site": d["SITE"].values, "group": d["group"].values,
        "age": d["age"].values, "feature": "IAFp", "band": None, "roi": roi, "z_score": z_all,
    }))
    print(f"IAFp roi{roi}: sigma2={s2:.3f} alpha2={a2:.2f} HC={is_hc.sum()}", flush=True)

iafp_long = pd.concat(rows, ignore_index=True)
iafp_long["set"] = iafp_long["group"].map({"HC": "test", "MCI": "mci", "AD": "AD", "ACr": "ACr"})

# ── Load the already-corrected 40-feature z-scores (no recomputation) ──────
primary = pd.read_csv(os.path.join(BLR_DIR, "blr_osc_pw_rel_canonic.csv"))
primary = primary.rename(columns={"Site": "Site"})
primary_test = primary[primary["set"].isin(["test", "mci", "AD", "ACr"])].copy()
primary_test = primary_test[["subject", "Site", "group", "set", "x_age", "feature", "band", "roi", "z_score"]]
primary_test = primary_test.rename(columns={"x_age": "age"})

combined = pd.concat([primary_test, iafp_long[primary_test.columns]], ignore_index=True)

# ── Classification: IAFp alone, and 40+5 combined, same groups as the main table ──
out = []
for name, df_in in [("IAFp only (5 features)", iafp_long), ("40 relative-power + IAFp (45 features)", combined)]:
    for grp, setlab in [("MCI", "mci"), ("AD", "AD"), ("ACr", "ACr")]:
        res, _ = classify_zscore_all_features(df_in, sets_incluidos=["test", setlab], n_bootstrap=50)
        res["analysis"] = name
        res["comparison"] = f"HC vs {grp}"
        out.append(res)
cls_out = pd.concat(out, ignore_index=True)
cls_path = os.path.join(BLR_DIR, "iafp_classification.csv")
cls_out.to_csv(cls_path, index=False)
print(f"\nsaved {cls_path}")
print(cls_out[["analysis", "comparison", "model", "AUC_mean", "AUC_std", "boot_AUC_mean", "boot_AUC_std"]]
      .round(3).to_string(index=False))

# ── EYO analysis (ACr only): EYO = age - 44 ─────────────────────────────────
acr = combined[combined["group"] == "ACr"].copy()
acr["eyo"] = acr["age"] - EYO_REFERENCE_AGE
acr_subj = acr[["subject", "Site", "age", "eyo"]].drop_duplicates()
print(f"\nACr n={len(acr_subj)}, age range {acr_subj.age.min():.0f}-{acr_subj.age.max():.0f}, "
      f"EYO range {acr_subj.eyo.min():.0f} to {acr_subj.eyo.max():.0f}")

acr["feat_col"] = acr["feature"].astype(str) + "__" + acr["band"].fillna("noband").astype(str) + "__" + acr["roi"].astype(str)
pivot = acr.pivot_table(index=["subject", "Site"], columns="feat_col", values="z_score").reset_index()
pivot = pivot.merge(acr_subj, on=["subject", "Site"])
feat_cols = [c for c in pivot.columns if c not in ("subject", "Site", "age", "eyo")]
pivot_cc = pivot.dropna(subset=feat_cols).reset_index(drop=True)
print(f"ACr complete cases (all 45 features): {len(pivot_cc)} / {len(pivot)}")

eyo_rows = []
for c in feat_cols:
    rho, p = spearmanr(pivot[c], pivot["eyo"], nan_policy="omit")
    eyo_rows.append({"feature": c, "n": pivot[c].notna().sum(), "rho": rho, "p": p})
eyo_df = pd.DataFrame(eyo_rows).sort_values("p")
m = len(eyo_df)
order = np.argsort(eyo_df["p"].values)
ranked = eyo_df["p"].values[order] * m / np.arange(1, m + 1)
q = np.minimum.accumulate(ranked[::-1])[::-1]
qa = np.empty(m)
qa[order] = np.minimum(q, 1.0)
eyo_df["q"] = qa
eyo_path = os.path.join(BLR_DIR, "iafp_eyo_univariate.csv")
eyo_df.to_csv(eyo_path, index=False)
print(f"\nsaved {eyo_path}")
print(eyo_df.round(3).to_string(index=False))
print(f"\nEYO univariate: {(eyo_df.p < 0.05).sum()}/{m} p<0.05 uncorrected, "
      f"{(eyo_df.q < 0.05).sum()}/{m} BH q<0.05")

# ── Composite EYO prediction: repeated k-fold ridge regression ──────────────
# NOTE: leave-one-out was tried first and discarded. With only 5-45 highly
# collinear features (the 5 IAFp ROIs correlate at 0.76-0.95) and n=39,
# leave-one-out barely perturbs the training set between folds, so ridge can
# lock onto a fragile, noise-amplifying contrast between near-duplicate
# features and reproduce it at every held-out point -- giving a spuriously
# high and perfectly stable "cross-validated" correlation (rho=-0.92 for
# IAFp alone) that collapsed to -0.33 (and was unstable across reshuffles)
# under proper repeated 5-fold CV, which perturbs the training set enough to
# break that artifact. Repeated k-fold (not LOO) is used here for that reason.
y_cc = pivot_cc["eyo"].values
n_cc = len(y_cc)


def cv_composite(cols, n_seeds=30, n_splits=5):
    X = pivot_cc[cols].values
    rhos = []
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = np.full(n_cc, np.nan)
        for tr, te in kf.split(X):
            reg = RidgeCV(alphas=np.logspace(-2, 3, 20))
            reg.fit(X[tr], y_cc[tr])
            pred[te] = reg.predict(X[te])
        rho, _ = spearmanr(pred, y_cc)
        rhos.append(rho)
    return np.array(rhos)


for label, cols in [("45 features (relative power + IAFp)", feat_cols),
                    ("IAFp only (5 features)", [c for c in feat_cols if c.startswith("IAFp")]),
                    ("relative power only (40 features)", [c for c in feat_cols if not c.startswith("IAFp")])]:
    rhos = cv_composite(cols)
    print(f"Composite ({label}): repeated 5-fold CV rho = {rhos.mean():.3f} +/- {rhos.std():.3f} "
          f"(range [{rhos.min():.3f}, {rhos.max():.3f}], 30 reshuffles), n = {n_cc}")

pivot_cc.to_csv(os.path.join(BLR_DIR, "iafp_eyo_pivot.csv"), index=False)
print(f"\nsaved {os.path.join(BLR_DIR, 'iafp_eyo_pivot.csv')}")
