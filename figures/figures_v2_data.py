"""
Datos para las figuras del manuscrito con los resultados de pipeline_v2.py.

1. blr_flat_full.csv: mismo formato que blr_osc_pw_rel_canonic.csv de testblr.py
   (filas 'range' con centiles, IC e IP; filas por sujeto con y_true, x_age, z_score y
   centiles p1..p99 en su edad), para fig_results_v2.py. Usa exactamente el mismo
   ajuste BLR que pipeline_v2.py (mismos HC de entrenamiento, filtro IQR y semilla).
2. spectra_v2.feather: espectros por canal de la muestra analizada (PSD, ajuste
   aperiódico y espectro aplanado de FOOOF 2-40 Hz) para fig3_spectra_zoom_v2.py.

Uso: python figures/figures_v2_data.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_config_params.paths import RESULTS_ROOT  # noqa: E402
from models.BLR import BayesianLinearRegression  # noqa: E402

V2 = os.path.join(RESULTS_ROOT, "pipeline_v2")
KP = {"number_of_bases": 30, "length_scale": 1.5, "bases_sampling_method": "KMeans"}
PCTS = [1, 5, 25, 50, 75, 95, 99]
CHANNELS = ["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2"]
RENAME = {"G2": "HC", "GU": "HC", "CTR": "HC", "DCL": "MCI", "A": "AD", "DTA": "AD"}
IAF = {"Argentina": "IAF_Argentina", "Chile": "IAF_Chile", "Cuba": "IAF_Cuba", "Dortmund": "IAF_Dortmund",
       "Greece": "IAF_Greece", "Medellin_hd": "IAF_Medellin_hd", "Medellin_ld": "IAF_Medellin_ld",
       "Oslo": "IAF_Oslo", "Poland": "IAF_Poland", "Seoul": "IAF_Seoul", "Spain": "IAF_Spain"}


def fit_blr(x, y):
    m = BayesianLinearRegression(KP, sigma2=0.06, alpha2=5)
    m.fit(x, y)
    m.optimize_hyperparameters()
    s2, a2 = m.sigma2, m.alpha2
    m = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
    m.fit(x, y)
    return m


def blr_full():
    H = pd.read_csv(os.path.join(V2, "features_roi_harmonized.csv"))
    feats = [c for c in H.columns if c.startswith(("flat_", "exponent_", "offset_"))]
    out = []
    for fc in feats:
        tr = H[H["set"] == "train"]
        q1, q3 = tr[fc].quantile([0.25, 0.75])
        iqr = q3 - q1
        tr = tr[tr[fc].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
        m = fit_blr(tr[["age"]].values.astype(float), tr[fc].values)
        base, roi = fc.rsplit("_roi", 1)
        band = base.replace("flat_", "") if base.startswith("flat_") else base
        feature = "flat" if base.startswith("flat_") else base

        # curvas de referencia en el rango de edad de los HC
        xr = np.linspace(H.loc[H.group == "HC", "age"].min(), H.loc[H.group == "HC", "age"].max(), 200).reshape(-1, 1)
        cen = m.predict_centiles(xr, percentiles=PCTS)
        _, _, _, ci_l, ci_u, pi_l, pi_u = m.predict_with_samples(xr, n_samples=500)
        rng = pd.DataFrame({"x_age": xr.ravel(), "set": "range", "group": "HC", "ci_lower": ci_l,
                            "ci_upper": ci_u, "pi_lower": pi_l, "pi_upper": pi_u})
        for p in PCTS:
            rng[f"centile_{p}"] = cen[p]
        interp = {p: interp1d(xr.ravel(), cen[p], bounds_error=False, fill_value="extrapolate") for p in PCTS}

        # sujetos
        x = H[["age"]].values.astype(float)
        mu = m.predict_with_samples(x, n_samples=500)[0]
        tv, _, _ = m.get_variance_decomposition(x)
        z = (H[fc].values - mu) / np.sqrt(tv * m.y_std ** 2)
        subj = pd.DataFrame({"subject": H["subject"], "Site": H["SITE"], "group": H["group"], "set": H["set"],
                             "x_age": H["age"], "y_true": H[fc], "y_pred": mu, "z_score": z})
        for p in PCTS:
            subj[f"p{p}"] = interp[p](H["age"].values)
        d = pd.concat([subj, rng], ignore_index=True)
        d["feature"], d["band"], d["roi"] = feature, band, roi
        out.append(d)
        print("BLR", fc, flush=True)
    pd.concat(out, ignore_index=True).to_csv(os.path.join(V2, "blr_flat_full.csv"), index=False)


def spectra():
    from fooof import FOOOFGroup
    H = pd.read_csv(os.path.join(V2, "features_roi_harmonized.csv"), usecols=["subject", "SITE", "group"])
    keep = set(H["subject"].astype(str) + "||" + H["SITE"])
    grp = dict(zip(H["subject"].astype(str) + "||" + H["SITE"], H["group"]))
    frames = []
    for site, f in IAF.items():
        d = pd.read_feather(os.path.join(RESULTS_ROOT, f"{f}_scorEpochs_zscore_long.feather"),
                            columns=["subject", "Sensors", "freqs", "psd"])
        d["SITE"] = site
        d = d[d["Sensors"].isin(CHANNELS)]
        u = d["subject"].astype(str) + "||" + site
        d = d[u.isin(keep)].copy()
        d["group"] = (d["subject"].astype(str) + "||" + site).map(grp)
        frames.append(d)
    db = pd.concat(frames, ignore_index=True)
    freqs = np.asarray(db["freqs"].iloc[0], float)
    psds = np.vstack(db["psd"].apply(lambda a: np.asarray(a, float)).values)
    fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=8, aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, psds, [2, 40], n_jobs=-1)
    ap, flat = [], []
    for i in range(len(db)):
        fm = fg.get_fooof(ind=i, regenerate=True)
        ap.append(fm._ap_fit)
        flat.append(fm.power_spectrum - fm._ap_fit)
    ff = fg.freqs
    db["freqs_fooof"] = [ff] * len(db)
    db["aperiodic_comp"] = ap
    db["flat_psd"] = flat
    db.to_feather(os.path.join(V2, "spectra_v2.feather"))
    print("spectra", len(db))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "blr"):
        blr_full()
    if what in ("all", "spectra"):
        spectra()
