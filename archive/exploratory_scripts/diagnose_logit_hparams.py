"""
Diagnostic (not production): grid-search length_scale / number_of_bases for
the logit-space BLR, scored on HELD-OUT HC fit quality only (MSLL, EV,
Spearman rho) -- never on downstream clinical-group AUC, to avoid leaking
the classification task into model selection.

Uses the train/test rows already produced by the current (fixed) pipeline,
so the HC split and IQR-outlier filtering are identical to production.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit, logit as logit_fn
from scipy.stats import spearmanr
from models.BLR import BayesianLinearRegression
from models.model_metrics import compute_MSLL, explained_var

DATA_PATH = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
    r"\blr_osc_pw_rel_canonic.csv"
)
EPS = 1e-3
SIGMA2_INIT, ALPHA2_INIT = 0.06, 5

df = pd.read_csv(DATA_PATH)

def fit_eval(band, roi, bases, length_scale):
    tr = df[(df["set"] == "train") & (df["band"] == band) & (df["roi"] == roi)]
    te = df[(df["set"] == "test")  & (df["band"] == band) & (df["roi"] == roi)]

    x_tr, x_te = tr[["x_age"]].values.astype(float), te[["x_age"]].values.astype(float)
    y_tr_raw, y_te_raw = tr["y_true"].values, te["y_true"].values
    y_tr = logit_fn(np.clip(y_tr_raw, EPS, 1 - EPS))
    y_te = logit_fn(np.clip(y_te_raw, EPS, 1 - EPS))

    kernel_params = {"number_of_bases": bases, "length_scale": length_scale, "bases_sampling_method": "KMeans"}
    blr = BayesianLinearRegression(kernel_params, sigma2=SIGMA2_INIT, alpha2=ALPHA2_INIT)
    blr.fit(x_tr, y_tr)
    blr.optimize_hyperparameters()
    blr = BayesianLinearRegression(kernel_params, sigma2=blr.sigma2, alpha2=blr.alpha2)
    blr.fit(x_tr, y_tr)

    y_pred_te, _, _, _, _, _, _ = blr.predict_with_samples(x_te, n_samples=500)
    total_var, _, _ = blr.get_variance_decomposition(x_te)
    total_var = total_var * (blr.y_std ** 2)
    train_mean = np.mean(y_tr, keepdims=True)
    train_var = np.var(y_tr, ddof=1, keepdims=True)
    msll = compute_MSLL(y_te, y_pred_te, total_var, train_mean, train_var)

    y_pred_te_orig = expit(y_pred_te)
    ev = explained_var(y_te_raw, y_pred_te_orig)
    rho, _ = spearmanr(y_te_raw, y_pred_te_orig)
    return msll, ev, rho

BANDS_ROI = [("theta", "O"), ("alpha2", "O"), ("alpha1", "O")]
BASES_GRID = [15, 20, 30, 40]
LS_GRID = [0.75, 1.0, 1.5, 2.0, 3.0]

print(f"{'band':8s} {'bases':>6s} {'ls':>5s} {'MSLL':>8s} {'EV':>8s} {'rho':>8s}")
for band, roi in BANDS_ROI:
    results = []
    for bases in BASES_GRID:
        for ls in LS_GRID:
            try:
                msll, ev, rho = fit_eval(band, roi, bases, ls)
            except Exception as e:
                print(f"{band:8s} {bases:6d} {ls:5.2f}  FAILED: {e}")
                continue
            results.append((bases, ls, msll, ev, rho))
            print(f"{band:8s} {bases:6d} {ls:5.2f} {msll:8.4f} {ev:8.4f} {rho:8.4f}")
    best = min(results, key=lambda r: r[2])  # lowest MSLL = best calibrated fit
    cur  = [r for r in results if r[0] == 30 and r[1] == 1.5][0]
    print(f"  -> current (bases=30, ls=1.5): MSLL={cur[2]:.4f} EV={cur[3]:.4f} rho={cur[4]:.4f}")
    print(f"  -> best by MSLL: bases={best[0]} ls={best[1]}  MSLL={best[2]:.4f} EV={best[3]:.4f} rho={best[4]:.4f}")
    print()
