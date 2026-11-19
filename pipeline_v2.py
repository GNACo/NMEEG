"""
Pipeline v2 (revisión AJJ, nov-2026): re-ejecución completa desde las PSD.

Cambios respecto a la versión del manuscrito
--------------------------------------------
1. FOOOF (fixed, peak_width [1, 8], max_n_peaks 8) ajustado en 2-40 Hz, fuera de la
   banda de transición del filtro pasa-altos de 1 Hz (0.5-1.5 Hz).
2. Feature espectral: potencia del espectro "aplanado" (aperiodic-adjusted):
   media en cada banda canónica de log10 PSD(f) - fit aperiódico(f). No es
   composicional (no depende de las demás bandas), no está ponderada por el ancho
   de banda y no requiere logit. Delta se define en 2-6 Hz.
3. Exclusión por R2 < 0.9 y por criterios TF/IAFp usando (subject, SITE).
4. Muestra única (run_config_params/sample.py): sin Greece ni MCI/AD del kindred PSEN1.
5. Armonización ComBat-GAM (neuroHarmonize) aprendida solo en HC de entrenamiento.
6. Modelo normativo BLR (misma configuración que testblr.py) y diagnóstico de calibración.
7. Pruebas de grupo: z clínico vs z del HC de test de edad comparable (Mann-Whitney, FDR),
   no contra 0.
8. Referencia ACr: HC de test de 20-45 años (edad de ACr); MCI/AD: HC de test >= 47 años.

Entradas (RESULTS_ROOT): IAF_<site>_scorEpochs_zscore_long.feather (PSD por canal
después del control de épocas, antes de los filtros TF/IAFp) y DEMOGRAPHIC_DIR.
Salidas: RESULTS_ROOT/pipeline_v2/

Uso: python pipeline_v2.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_config_params.paths import RESULTS_ROOT  # noqa: E402
from run_config_params.config_local import DEMOGRAPHIC_DIR  # noqa: E402
from run_config_params.sample import analysis_sample, KINDRED_SITES  # noqa: E402
from models.BLR import BayesianLinearRegression  # noqa: E402
from models.model_metrics import compute_MSLL  # noqa: E402

OUT = os.path.join(RESULTS_ROOT, "pipeline_v2" + ("_TFfilter" if os.environ.get("TF_FILTER", "0") == "1" else ""))
os.makedirs(OUT, exist_ok=True)

SITES = {  # sitio: (archivo IAF, archivo demográfico)
    "Argentina": ("IAF_Argentina", "dem_BrainLat_AR_zscore.xlsx"),
    "Chile": ("IAF_Chile", "dem_BrainLat_CL_zscore.xlsx"),
    "Cuba": ("IAF_Cuba", "dem_CHBMP_zscore.xlsx"),
    "Dortmund": ("IAF_Dortmund", "dem_Dortmund.xlsx"),
    "Greece": ("IAF_Greece", "dem_Greece_zscore.xlsx"),
    "Medellin_hd": ("IAF_Medellin_hd", "dem_biomarcadores_zscore.xlsx"),
    "Medellin_ld": ("IAF_Medellin_ld", "dem_portables_zscore.xlsx"),
    "Oslo": ("IAF_Oslo", "dem_SRM_zscore.xlsx"),
    "Poland": ("IAF_Poland", "dem_Polonia_zscore.xlsx"),
    "Seoul": ("IAF_Seoul", "dem_CAUEEG2.xlsx"),
    "Spain": ("IAF_Spain", "dem_Madrid_c3n_zscore.xlsx"),
}
RENAME = {"G2": "HC", "GU": "HC", "CTR": "HC", "DCL": "MCI", "A": "AD", "DTA": "AD"}
GROUPS = ["HC", "MCI", "AD", "ACr"]
CHANNELS = ["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2"]
ROIS = {"F": ["FP1", "FP2"], "C": ["C3", "C4"], "P": ["P7", "P8"], "O": ["O1", "O2"],
        "PO": ["O1", "O2", "P7", "P8"]}
BANDS = {"delta": (2, 6), "theta": (6, 8.5), "alpha1": (8.5, 10.5), "alpha2": (10.5, 12.5),
         "beta1": (12.5, 18.5), "beta2": (18.5, 21), "beta3": (21, 30), "gamma": (30, 40)}
FREQ_RANGE = [2, 40]
R2_MIN = 0.9
# El criterio TF/IAFp (versión del manuscrito) excluía al 51% de los AD frente al 26% de
# los HC y al 97% de Greece: es un sesgo de selección contra el enlentecimiento. Las
# features canónicas no lo necesitan; el control de calidad del ajuste es R2 >= 0.9.
TF_FILTER = os.environ.get("TF_FILTER", "0") == "1"
KP = {"number_of_bases": 30, "length_scale": 1.5, "bases_sampling_method": "KMeans"}
SEED = 42
log = {}


def uid(d):
    return d["subject"].astype(str) + "||" + d["SITE"].astype(str)


# --------------------------------------------------------------------------- 1. datos
def remove_subjects_by_criteria(data, channels):
    """Igual que Functions.functions.remove_subjects_by_criteria: elimina al sujeto si en
    algún canal TF < 4 (TF entre 3 y 4 ya fue corregido a 4 en calculate_IAF) o TF > IAFp."""
    bad = data[(data["TF"] < 4) | (data["TF"] > data["IAFp"])]["subject"].unique()
    return data[~data["subject"].isin(bad)]


def load_sites():
    frames = []
    for site, (iaf, _) in SITES.items():
        d = pd.read_feather(os.path.join(RESULTS_ROOT, f"{iaf}_scorEpochs_zscore_long.feather"))
        d["SITE"] = site
        d["group"] = d["group"].replace(RENAME)
        d = d[d["Sensors"].isin(CHANNELS)]
        n0 = d["subject"].nunique()
        if TF_FILTER:
            d = remove_subjects_by_criteria(d, CHANNELS)  # TF < 4 (tras corrección) o TF > IAFp
        log.setdefault("TF_IAFp", {})[site] = {"antes": int(n0), "despues": int(d["subject"].nunique())}
        frames.append(d[["subject", "SITE", "group", "Sensors", "freqs", "psd"]])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- 2. FOOOF
def fooof_features(db):
    from fooof import FOOOFGroup
    freqs = np.asarray(db["freqs"].iloc[0], float)
    psds = np.vstack(db["psd"].apply(lambda a: np.asarray(a, float)).values)
    fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=8, aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, psds, FREQ_RANGE, n_jobs=-1)
    rows = []
    for i in range(len(db)):
        fm = fg.get_fooof(ind=i, regenerate=True)
        f = fm.freqs
        flat = fm.power_spectrum - fm._ap_fit       # espectro aplanado (log10)
        r = {"exponent": fm.aperiodic_params_[1], "offset": fm.aperiodic_params_[0],
             "r2": fm.r_squared_}
        for b, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f < hi) if b != "gamma" else (f >= lo) & (f <= hi)
            r[f"flat_{b}"] = float(flat[m].mean())
        rows.append(r)
    feats = pd.concat([db[["subject", "SITE", "group", "Sensors"]].reset_index(drop=True),
                       pd.DataFrame(rows)], axis=1)
    return feats


# --------------------------------------------------------------------------- 3. demografía
def load_demographics():
    out = []
    for site, (_, dem) in SITES.items():
        d = pd.read_excel(os.path.join(DEMOGRAPHIC_DIR, dem))
        d.columns = [c.strip() for c in d.columns]
        keep = {"subject": d["subject"].astype(str), "age": d["age"]}
        keep["sex"] = d["sex"] if "sex" in d else np.nan
        ed = [c for c in d.columns if c.lower().startswith("education")]
        keep["education"] = d[ed[0]] if ed else np.nan
        x = pd.DataFrame(keep)
        x["SITE"] = site
        out.append(x)
    dem = pd.concat(out, ignore_index=True).drop_duplicates(["subject", "SITE"])
    return dem


# --------------------------------------------------------------------------- 5. armonización
def hc_split(df):
    hc = df[df["group"] == "HC"][["subject", "SITE", "age"]].drop_duplicates().reset_index(drop=True)
    bins = pd.qcut(hc["age"], q=5, labels=False, duplicates="drop")
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(hc, bins))
    return set(uid(hc.iloc[tr])), set(uid(hc.iloc[te]))


def harmonize(roi_df, feat_cols, train_uids):
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    out = roi_df.copy()
    is_tr = (out["group"] == "HC") & uid(out).isin(train_uids)
    covars = out[["SITE", "age"]].copy()
    covars["SITE"] = covars["SITE"].astype(str)
    # La spline de edad de ComBat-GAM no extrapola: se acota la edad al rango de los HC
    # de entrenamiento solo como covariable de armonización.
    lo, hi = covars.loc[is_tr, "age"].min(), covars.loc[is_tr, "age"].max()
    covars["age"] = covars["age"].clip(lo, hi)
    X = out[feat_cols].values.astype(float)
    model, _ = harmonizationLearn(X[is_tr.values], covars[is_tr.values], smooth_terms=["age"], eb=True)
    out[feat_cols] = harmonizationApply(X, covars, model)
    return out


# --------------------------------------------------------------------------- 6. modelo normativo
def fit_blr(x, y):
    m = BayesianLinearRegression(KP, sigma2=0.06, alpha2=5)
    m.fit(x, y)
    m.optimize_hyperparameters()
    s2, a2 = m.sigma2, m.alpha2
    m = BayesianLinearRegression(KP, sigma2=s2, alpha2=a2)
    m.fit(x, y)
    return m


def zscores(m, x, y):
    mu = m.predict_with_samples(x, n_samples=500)[0]
    tv, _, _ = m.get_variance_decomposition(x)
    tv = tv * m.y_std ** 2
    return (y - mu) / np.sqrt(tv), mu, tv


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    n = ok.sum()
    if n:
        o = np.argsort(p[ok])
        r = p[ok][o] * n / np.arange(1, n + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        t = np.empty(n)
        t[o] = np.minimum(r, 1)
        q[ok] = t
    return q


def main():
    # 1-2. features por canal
    feat_path = os.path.join(OUT, "fooof_channel_features.feather")
    if os.path.exists(feat_path):
        feats = pd.read_feather(feat_path)
    else:
        db = load_sites()
        feats = fooof_features(db)
        feats.to_feather(feat_path)
    feats = feats[feats["group"].isin(GROUPS)]

    # 3. R2 por (subject, SITE)
    feats["uid"] = uid(feats)
    bad = set(feats.loc[feats["r2"] < R2_MIN, "uid"])
    excl = feats[feats["uid"].isin(bad)].drop_duplicates("uid").groupby(["SITE", "group"]).size()
    log["R2_excluidos"] = {f"{k[0]}|{k[1]}": int(v) for k, v in excl.items()}
    log["R2_excluidos_total"] = len(bad)
    feats = feats[~feats["uid"].isin(bad)]

    # 4. ROI
    fcols = [f"flat_{b}" for b in BANDS] + ["exponent", "offset"]
    rois = []
    for roi, chs in ROIS.items():
        r = feats[feats["Sensors"].isin(chs)].groupby(["subject", "SITE", "group"], as_index=False)[fcols].mean()
        r = r.rename(columns={c: f"{c}_roi{roi}" for c in fcols})
        rois.append(r.set_index(["subject", "SITE", "group"]))
    wide = pd.concat(rois, axis=1).reset_index()

    dem = load_demographics()
    wide = wide.merge(dem, on=["subject", "SITE"], how="left")
    log["sin_edad"] = wide[wide["age"].isna()].groupby(["SITE", "group"]).size().astype(int).to_dict()
    log["sin_edad"] = {f"{k[0]}|{k[1]}": v for k, v in log["sin_edad"].items()}
    wide = wide.dropna(subset=["age"])
    wide = analysis_sample(wide)
    log["muestra_final"] = {f"{k[0]}|{k[1]}": int(v) for k, v in wide.groupby(["SITE", "group"]).size().items()}
    wide.to_csv(os.path.join(OUT, "features_roi_unharmonized.csv"), index=False)

    # 5. split y armonización
    train_uids, test_uids = hc_split(wide)
    harm = []
    for roi in ROIS:
        cols = [f"{c}_roi{roi}" for c in fcols]
        harm.append(harmonize(wide[["subject", "SITE", "group", "age"] + cols], cols, train_uids)[cols])
    H = pd.concat([wide[["subject", "SITE", "group", "age", "sex", "education"]]] + harm, axis=1)
    H["uid"] = uid(H)
    H["set"] = np.select([H["uid"].isin(train_uids), H["uid"].isin(test_uids)], ["train", "test"],
                         default=H["group"].map({"MCI": "mci", "AD": "AD", "ACr": "ACr"}))
    H.to_csv(os.path.join(OUT, "features_roi_harmonized.csv"), index=False)

    # 6. BLR por feature
    feat_names = [c for c in H.columns if c.startswith(("flat_", "exponent_", "offset_"))]
    long, metrics = [], []
    for fc in feat_names:
        tr = H[H["set"] == "train"]
        q1, q3 = tr[fc].quantile([0.25, 0.75])
        iqr = q3 - q1
        tr = tr[tr[fc].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
        m = fit_blr(tr[["age"]].values.astype(float), tr[fc].values)
        z, mu, tv = zscores(m, H[["age"]].values.astype(float), H[fc].values)
        base, roi = fc.rsplit("_roi", 1)
        band = base.replace("flat_", "") if base.startswith("flat_") else None
        feature = "flat" if band else base
        long.append(pd.DataFrame({"subject": H["subject"], "Site": H["SITE"], "group": H["group"],
                                  "set": H["set"], "age": H["age"], "feature": feature, "band": band,
                                  "roi": roi, "y": H[fc], "y_pred": mu, "z_score": z}))
        te = H["set"] == "test"
        ztr = zscores(m, tr[["age"]].values.astype(float), tr[fc].values)[0]
        msll = compute_MSLL(H.loc[te, fc].values, mu[te.values], tv[te.values],
                            np.mean(tr[fc].values, keepdims=True), np.var(tr[fc].values, ddof=1, keepdims=True))
        rho, prho = stats.spearmanr(H.loc[te, fc], mu[te.values])
        zt = z[te.values]
        metrics.append({"feature": fc, "msll_test": float(np.mean(msll)), "rho_test": rho, "p_rho": prho,
                        "ev_test": 1 - np.var(H.loc[te, fc].values - mu[te.values]) / np.var(H.loc[te, fc].values),
                        "z_test_mean": zt.mean(), "z_test_sd": zt.std(ddof=1),
                        "z_test_pct_abs_gt_1.96": 100 * np.mean(np.abs(zt) > 1.96),
                        "z_test_skew": stats.skew(zt), "z_train_shapiro_p": stats.shapiro(ztr).pvalue})
        print("BLR", fc, flush=True)
    L = pd.concat(long, ignore_index=True)
    L.to_csv(os.path.join(OUT, "blr_flat.csv"), index=False)
    M = pd.DataFrame(metrics)
    M["rho_q"] = bh(M["p_rho"])
    M.to_csv(os.path.join(OUT, "metrics_flat.csv"), index=False)

    # 7. pruebas de grupo vs HC de test de edad comparable
    ref_rules = {"MCI": ("mci", lambda a: a >= 47), "AD": ("AD", lambda a: a >= 47),
                 "ACr": ("ACr", lambda a: (a >= 20) & (a <= 45))}
    rows = []
    for g, (st, agerule) in ref_rules.items():
        for (feat, band, roi), d in L.groupby(["feature", "band", "roi"], dropna=False):
            zc = d.loc[d["set"] == st, "z_score"]
            zr = d.loc[(d["set"] == "test") & agerule(d["age"]), "z_score"]
            p = stats.mannwhitneyu(zc, zr).pvalue
            rows.append({"group": g, "feature": feat, "band": band, "roi": roi, "n_clin": len(zc), "n_ref": len(zr),
                         "mean_z_clin": zc.mean(), "mean_z_ref": zr.mean(),
                         "delta_z": zc.mean() - zr.mean(),
                         "pct_clin_|z|>1.96": 100 * np.mean(np.abs(zc) > 1.96),
                         "pct_ref_|z|>1.96": 100 * np.mean(np.abs(zr) > 1.96), "p": p})
    G = pd.DataFrame(rows)
    G["q"] = bh(G["p"])
    G.to_csv(os.path.join(OUT, "group_deviations_vs_HCtest.csv"), index=False)

    # 8. clasificación
    from classify_blr import classify_zscore_all_features
    res = []
    for g, (st, agerule) in ref_rules.items():
        sub = L[(L["set"] == st) | ((L["set"] == "test") & agerule(L["age"]))]
        for label, s in [("todas", sub), ("sin_gamma", sub[sub["band"] != "gamma"])]:
            r, imp = classify_zscore_all_features(s, sets_incluidos=["test", st], n_bootstrap=50)
            r["comparison"] = f"HC vs {g}"
            r["features"] = label
            res.append(r)
            if label == "todas":
                imp.to_csv(os.path.join(OUT, f"feature_importance_{g}.csv"), index=False)
    C = pd.concat(res, ignore_index=True)
    C.to_csv(os.path.join(OUT, "classification.csv"), index=False)

    with open(os.path.join(OUT, "log.json"), "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    print(json.dumps(log, indent=1, default=str))
    print(C[["comparison", "features", "model", "AUC_mean", "AUC_std", "brier", "boot_AUC_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
