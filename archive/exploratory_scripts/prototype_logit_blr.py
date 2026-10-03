"""
Prototype: logit-transformed BLR to fix negative centiles for bounded
(0,1) oscillatory relative-power features (Theta/O, Alpha2/O).

Quick, standalone comparison against the current (linear/Gaussian) model
already saved in blr_osc_pw_rel_canonic.csv. Does NOT touch testblr.py or
the production pipeline.

NOTE: the current model is fit on ReCombat-*harmonized* values, which can
go slightly negative (harmonization is an additive/multiplicative site
correction that does not preserve the [0,1] bound of the raw feature).
This prototype instead fits on the *raw, pre-harmonization* relative
power (strictly within (0,1) by construction) to test whether a
logit-transformed BLR keeps centiles bounded and looks reasonable, before
deciding whether to redo harmonization itself in logit space.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit as logit_fn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.BLR import BayesianLinearRegression

DATA_PATH = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
    r"\blr_osc_pw_rel_canonic.csv"
)
RAW_PATH = (
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\features_osc\age_group"
    r"\osc_pw_rel_canonic_roiO_SITE_age_group_recombat.xlsx"
)
SAVE_PATH = os.environ.get("NMEEG_FIGURES_PATH",
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
    r"\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
    r"\figures_paper")

BANDS = {"theta": "osc_pw_canonic_theta_roiO", "alpha2": "osc_pw_canonic_alpha2_roiO"}
BASES, LENGTH_SCALE, SIGMA2, ALPHA2 = 30, 1.5, 0.06, 5

# ── Current (harmonized, linear) centile curves — already computed ───────────
blr_csv = pd.read_csv(DATA_PATH)

# ── Raw (pre-harmonization) HC data, restricted to the same train subjects ──
raw = pd.read_excel(RAW_PATH, sheet_name="unharmonizeSITE_age_group")
raw = raw[raw["group"] == "HC"].copy()
raw["uid"] = raw["subject"].astype(str) + "||" + raw["SITE"].astype(str)

train_ids = set(
    (blr_csv[blr_csv["set"] == "train"]["subject"].astype(str)
     + "||" + blr_csv[blr_csv["set"] == "train"]["Site"].astype(str)).tolist()
)
raw_train = raw[raw["uid"].isin(train_ids)].copy()
print(f"Raw HC train subjects matched: {len(raw_train)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (band, col) in zip(axes, BANDS.items()):
    x = raw_train[["age"]].values.astype(float)
    y_raw = raw_train[col].values.astype(float)
    assert (y_raw > 0).all() and (y_raw < 1).all(), "values outside (0,1)!"
    y_logit = logit_fn(y_raw)

    kernel_params = {"number_of_bases": BASES, "length_scale": LENGTH_SCALE,
                      "bases_sampling_method": "KMeans"}
    blr = BayesianLinearRegression(kernel_params, sigma2=SIGMA2, alpha2=ALPHA2)
    blr.fit(x, y_logit)
    blr.optimize_hyperparameters()
    blr = BayesianLinearRegression(kernel_params, sigma2=blr.sigma2, alpha2=blr.alpha2)
    blr.fit(x, y_logit)

    x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    centiles_logit = blr.predict_centiles(x_range, percentiles=[5, 25, 50, 75, 95])
    centiles_orig = {p: expit(v) for p, v in centiles_logit.items()}

    # Current (harmonized, linear) curve for comparison
    cur = blr_csv[(blr_csv["set"] == "range") & (blr_csv["band"] == band) & (blr_csv["roi"] == "O")]
    cur = cur.sort_values("x_age")

    ax.plot(cur["x_age"], cur["centile_5"], color="#C0392B", lw=1.5, ls="--", label="current (harmonized) 5th")
    ax.plot(cur["x_age"], cur["centile_50"], color="#C0392B", lw=2.0, label="current (harmonized) 50th")
    ax.plot(cur["x_age"], cur["centile_95"], color="#C0392B", lw=1.5, ls="--", label="current (harmonized) 95th")

    ax.plot(x_range.flatten(), centiles_orig[5], color="#1A7A4A", lw=1.5, ls="--", label="logit (raw) 5th")
    ax.plot(x_range.flatten(), centiles_orig[50], color="#1A7A4A", lw=2.0, label="logit (raw) 50th")
    ax.plot(x_range.flatten(), centiles_orig[95], color="#1A7A4A", lw=1.5, ls="--", label="logit (raw) 95th")

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"{band} / O  —  min 5th (current)={cur['centile_5'].min():.3f}, "
                 f"min 5th (logit)={centiles_orig[5].min():.3f}", fontsize=10)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Oscillatory relative power")
    ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
os.makedirs(SAVE_PATH, exist_ok=True)
out = os.path.join(SAVE_PATH, "prototype_logit_blr.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
