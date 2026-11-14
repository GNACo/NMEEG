"""
Diagnósticos solicitados en la revisión de coautores (AJJ) — NO re-entrena nada.

Lee los archivos que ya existen en RESULTS_ROOT y genera un Excel con una hoja
por verificación, más un resumen en consola:

  1. muestra_por_sitio      n por sitio x grupo después de excluir Greece
                            (marca los AD/MCI de Medellín = kindred PSEN1).
  2. solapamiento_medellin  códigos de sujeto presentes en Medellin_hd y
                            Medellin_ld (posibles sujetos duplicados).
  3. logit_clipping         valores armonizados de potencia relativa fuera de
                            (1e-3, 1-1e-3) que testblr.py recorta antes del
                            logit (ComBat-GAM no respeta los límites 0-1).
  4. calibracion_z_HC       media, DE, asimetría, curtosis, Shapiro y % |z|>1.96
                            de los z-scores del HC de test (deberían ser ~N(0,1)).
  5. residuos_HC_train      normalidad de los residuos en el HC de entrenamiento.
  6. desviaciones_extremas  % de participantes con |z|>1.96 por grupo/feature
                            vs HC de test (Fisher exacto, FDR) -> resultado
                            típico de modelos normativos, útil para el paper.
  7. exclusiones_QC         reporte de sujetos restantes tras TF/IAFp y R2<0.9
                            (report_IAF_by_sensor.xlsx) -> diagrama de flujo.
  8. r2_codigos_repetidos   códigos de sujeto repetidos entre sitios: el filtro
                            R2<0.9 de filter_spectrals.ipynb elimina por
                            'subject' sin SITE y puede borrar a otra persona.
  9. potencia_periodica     piso de la potencia relativa por ancho de banda
                            (10^G(f) = 1 sin pico) y exceso periódico real
                            sum(10^G - 1) por grupo -> chequeo del hallazgo
                            beta/gamma en ACr.
 10. subtipos_MCI_CAUEEG    columnas/etiquetas de subtipo en la demografía de
                            CAUEEG (si DEMOGRAPHIC_DIR está configurado).

Uso (desde la raíz del repo, entorno de análisis):
    python analysis/reviewer_checks.py
Salida: <BLR_DIR>/reviewer_checks.xlsx
"""
import os
import sys
import ast
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from run_config_params.paths import (RESULTS_ROOT, HARM_OUT_DIR, HARM_SUFFIX,
                                     BLR_DIR, FOOOF_DB)

try:
    from run_config_params.config_local import DEMOGRAPHIC_DIR
except Exception:
    DEMOGRAPHIC_DIR = None

ROIS = ["F", "C", "P", "O", "PO"]
BANDS = ["theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma"]
EXCLUDED_SITES = ["Greece"]
EPS = 1e-3
GROUP_ORDER = ["ACr", "MCI", "AD"]          # orden clínico pedido por AJJ
RENAME = {"G2": "HC", "GU": "HC", "CTR": "HC", "DCL": "MCI", "A": "AD", "DTA": "AD"}
CANONIC = {"delta": (1.5, 6), "theta": (6, 8.5), "alpha1": (8.5, 10.5),
           "alpha2": (10.5, 12.5), "beta1": (12.5, 18.5), "beta2": (18.5, 21),
           "beta3": (21, 30), "gamma": (30, 40)}

OUT_XLSX = os.path.join(BLR_DIR, "reviewer_checks.xlsx")
sheets = {}
summary = []


def say(msg):
    print(msg, flush=True)
    summary.append(msg)


def harm_path(roi, family="osc_pw_rel_canonic"):
    return os.path.join(HARM_OUT_DIR, f"{family}_roi{roi}_SITE_age_group_{HARM_SUFFIX}.xlsx")


def load_harm(roi, sheet="harmonizeSITE_age_group"):
    d = pd.read_excel(harm_path(roi), sheet_name=sheet)
    d["subject"] = d["subject"].astype(str)
    return d


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    n = ok.sum()
    if n == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.minimum(ranked, 1)
    q[ok] = tmp
    return q


def run(name, fn):
    say(f"\n=== {name} ===")
    try:
        fn()
    except Exception as e:
        say(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------------------- 1
def check_sample():
    d = load_harm("O")
    d = d[~d["SITE"].isin(EXCLUDED_SITES)]
    tab = pd.crosstab(d["SITE"], d["group"], margins=True)
    sheets["muestra_por_sitio"] = tab.reset_index()
    say(tab.to_string())
    med = d[d["SITE"].str.contains("Medell", case=False) & d["group"].isin(["AD", "MCI"])]
    if len(med):
        say(f"  ATENCIÓN: {len(med)} AD/MCI de Medellín en los datos armonizados "
            f"({med.groupby('group').size().to_dict()}). testblr.py excluye los MCI de "
            f"Medellín pero NO los AD: si son del kindred PSEN1 hay que excluirlos también.")
        sheets["AD_MCI_medellin"] = med[["subject", "SITE", "group", "age"]]


# --------------------------------------------------------------------------- 2
def check_medellin_overlap():
    d = load_harm("O")
    hd = d[d["SITE"].str.lower().str.contains("hd")][["subject", "group", "age"]]
    ld = d[d["SITE"].str.lower().str.contains("ld")][["subject", "group", "age"]]
    same_code = hd.merge(ld, on="subject", suffixes=("_hd", "_ld"))
    sheets["solapamiento_medellin"] = same_code
    say(f"  Medellin_hd n={len(hd)} | Medellin_ld n={len(ld)} | mismo código en ambos: {len(same_code)}")
    if len(same_code):
        same_code["age_diff"] = (same_code["age_hd"] - same_code["age_ld"]).abs()
        probable = same_code[(same_code["age_diff"] <= 2) & (same_code["group_hd"] == same_code["group_ld"])]
        say(f"  -> con mismo grupo y edad ±2 años (probable misma persona): {len(probable)}")
    # Coincidencia por (grupo, edad exacta) aunque el código difiera
    by_age = hd.merge(ld, on=["group", "age"], suffixes=("_hd", "_ld"))
    sheets["medellin_misma_edad_grupo"] = by_age
    say(f"  pares hd/ld con mismo grupo y misma edad (revisar manualmente con códigos originales): {len(by_age)}")
    say("  Nota: si los códigos de BIDS_PORTABLES_CODIFICADO están recodificados, el cruce "
        "definitivo debe hacerse con la tabla de códigos originales del GNA.")


# --------------------------------------------------------------------------- 3
def check_logit_clipping():
    rows = []
    for roi in ROIS:
        d = load_harm(roi)
        d = d[~d["SITE"].isin(EXCLUDED_SITES)]
        for band in BANDS:
            f = f"harm_osc_pw_canonic_{band}_roi{roi}"
            if f not in d.columns:
                continue
            v = d[f]
            low, high = v <= EPS, v >= 1 - EPS
            for g, sub in d.groupby("group"):
                rows.append({"roi": roi, "band": band, "group": g, "n": sub[f].notna().sum(),
                             "n_<=0": int((sub[f] <= 0).sum()), "n_<=eps": int(low[sub.index].sum()),
                             "n_>=1-eps": int(high[sub.index].sum()),
                             "min": sub[f].min(), "max": sub[f].max()})
    t = pd.DataFrame(rows)
    sheets["logit_clipping"] = t
    tot = t[["n_<=eps", "n_>=1-eps"]].sum().sum()
    say(f"  Valores recortados antes del logit (todas las features/grupos): {tot} "
        f"de {t['n'].sum()} ({100 * tot / max(t['n'].sum(), 1):.3f}%)")
    if tot:
        say("  -> Reportarlo en Métodos o, mejor, transformar (logit/ALR) ANTES de ComBat-GAM.")
        bad = t[(t['n_<=eps'] + t['n_>=1-eps']) > 0]
        say(bad.groupby(['band', 'group'])[['n_<=eps', 'n_>=1-eps']].sum().to_string())


# --------------------------------------------------------------------------- 4-6
def load_blr():
    p = os.path.join(BLR_DIR, "blr_osc_pw_rel_canonic.csv")
    df = pd.read_csv(p, low_memory=False)
    df = df[df["set"] != "range"].copy()
    df["band"] = df["band"].fillna("exponent")
    df["feat"] = df["band"].astype(str) + "_" + df["roi"].astype(str)
    return df


def check_z_calibration():
    df = load_blr()
    rows = []
    for (feat), sub in df[df["set"] == "test"].groupby("feat"):
        z = sub["z_score"].dropna().values
        if len(z) < 8:
            continue
        sh = stats.shapiro(z)
        rows.append({"feature": feat, "n": len(z), "mean": z.mean(), "sd": z.std(ddof=1),
                     "skew": stats.skew(z), "excess_kurtosis": stats.kurtosis(z),
                     "shapiro_W": sh.statistic, "shapiro_p": sh.pvalue,
                     "pct_|z|>1.96": 100 * np.mean(np.abs(z) > 1.96)})
    t = pd.DataFrame(rows)
    t["shapiro_q"] = bh_fdr(t["shapiro_p"])
    sheets["calibracion_z_HC"] = t
    say(f"  HC test: media de medias z = {t['mean'].mean():+.3f}, media de DE = {t['sd'].mean():.3f}, "
        f"% |z|>1.96 promedio = {t['pct_|z|>1.96'].mean():.1f}% (esperado 5%)")
    say(f"  Features con no-normalidad (Shapiro q<0.05): {(t['shapiro_q'] < 0.05).sum()} / {len(t)}")
    bad = t[(t["sd"] > 1.2) | (t["sd"] < 0.8) | (t["mean"].abs() > 0.2)]
    if len(bad):
        say("  Features mal calibradas (|media|>0.2 o DE fuera de 0.8-1.2):")
        say(bad[["feature", "mean", "sd", "skew"]].round(3).to_string(index=False))


def check_train_residuals():
    df = load_blr()
    rows = []
    for feat, sub in df[df["set"] == "train"].groupby("feat"):
        z = sub["z_score"].dropna().values
        if len(z) < 8:
            continue
        rows.append({"feature": feat, "n": len(z), "skew": stats.skew(z),
                     "excess_kurtosis": stats.kurtosis(z),
                     "dagostino_p": stats.normaltest(z).pvalue})
    t = pd.DataFrame(rows)
    t["q"] = bh_fdr(t["dagostino_p"])
    sheets["residuos_HC_train"] = t
    say(f"  HC train: features con residuos no normales (q<0.05): {(t['q'] < 0.05).sum()} / {len(t)}; "
        f"|asimetría| media = {t['skew'].abs().mean():.2f}")
    say("  Si muchas fallan: considerar BLR con warping (SinhArcsinh) o HBR-SHASH (PCNtoolkit) "
        "y llamarlos 'deviation scores' en el texto.")


def check_extreme_deviations():
    df = load_blr()
    ref = df[df["set"] == "test"]
    sets = {"ACr": "ACr", "MCI": "mci", "AD": "AD"}
    rows = []
    for g in GROUP_ORDER:
        clin = df[df["set"] == sets[g]]
        for feat in sorted(df["feat"].unique()):
            zr = ref.loc[ref["feat"] == feat, "z_score"].dropna()
            zc = clin.loc[clin["feat"] == feat, "z_score"].dropna()
            if len(zc) == 0:
                continue
            for direction, fn in [("pos", lambda z: z > 1.96), ("neg", lambda z: z < -1.96)]:
                a, b = int(fn(zc).sum()), len(zc)
                c, d = int(fn(zr).sum()), len(zr)
                _, p = stats.fisher_exact([[a, b - a], [c, d - c]])
                rows.append({"group": g, "feature": feat, "direction": direction,
                             "pct_clin": 100 * a / b, "pct_HC_test": 100 * c / d, "p": p})
    t = pd.DataFrame(rows)
    t["q"] = bh_fdr(t["p"])
    sheets["desviaciones_extremas"] = t
    sig = t[t["q"] < 0.05].sort_values(["group", "q"])
    say(f"  Features con % de desviaciones extremas mayor que HC (q<0.05): {len(sig)}")
    if len(sig):
        say(sig.round(3).head(30).to_string(index=False))


# --------------------------------------------------------------------------- 7
def check_qc_report():
    p = os.path.join(RESULTS_ROOT, "report_IAF_by_sensor.xlsx")
    xls = pd.ExcelFile(p)
    for sh in xls.sheet_names:
        t = pd.read_excel(xls, sheet_name=sh)
        sheets[f"QC_{sh}"[:31]] = t
        say(f"  [{sh}]")
        cols = [c for c in t.columns if c in ("SITE", "Sensor", "remaining_subject", "remaining_fooof")]
        say(t[cols].drop_duplicates(subset=[c for c in ("SITE", "remaining_subject", "remaining_fooof") if c in cols]).to_string(index=False)
            if cols else t.head(20).to_string(index=False))
    say("  Con estas tablas + los QC_*.xlsx (len_epochs<24) se arma el diagrama de flujo de participantes.")
    say("  Ojo: Spain no tiene QC_Spain.xlsx (comentario 'Falta sacarlo' en filter_spectrals.ipynb).")


# --------------------------------------------------------------------------- 8
def check_repeated_codes():
    db = pd.read_feather(FOOOF_DB, columns=["subject", "SITE", "group"])
    db["subject"] = db["subject"].astype(str)
    u = db.drop_duplicates(["subject", "SITE"])
    rep = u.groupby("subject")["SITE"].nunique()
    rep = rep[rep > 1]
    t = u[u["subject"].isin(rep.index)].sort_values(["subject", "SITE"])
    sheets["r2_codigos_repetidos"] = t
    say(f"  Códigos de sujeto que aparecen en >1 sitio (en la base ya filtrada): {len(rep)}")
    say("  filter_spectrals.ipynb (celda 28) elimina por R2<0.9 usando solo 'subject': un mal ajuste "
        "en un sitio borra a todas las personas con ese código en otros sitios. Corregir usando "
        "(subject, SITE) y re-ejecutar desde ese paso. El número real de afectados solo se puede "
        "calcular sobre filters_IAFs_database (antes del filtro), que no está guardado.")


# --------------------------------------------------------------------------- 9
def check_periodic_power():
    cols = ["subject", "SITE", "group", "Sensors", "freqs_fooof", "oscilatory_psd"]
    db = pd.read_feather(FOOOF_DB, columns=cols)
    db["group"] = db["group"].replace(RENAME)
    db = db[db["group"].isin(["HC", "MCI", "AD", "ACr"]) & ~db["SITE"].isin(EXCLUDED_SITES)]
    freqs = np.asarray(db["freqs_fooof"].iloc[0], float)
    masks = {b: (freqs >= lo) & (freqs <= hi) for b, (lo, hi) in CANONIC.items()}
    nb = {b: int(m.sum()) for b, m in masks.items()}
    floor = {b: nb[b] / sum(nb.values()) for b in CANONIC}
    sheets["piso_potencia_relativa"] = pd.DataFrame(
        {"band": list(CANONIC), "n_bins": [nb[b] for b in CANONIC], "rel_power_sin_picos": [floor[b] for b in CANONIC]})
    say("  Potencia relativa que tendría cada banda SIN ningún pico (10^G=1 en todo el rango):")
    say("   " + ", ".join(f"{b}={floor[b]:.3f}" for b in CANONIC))
    say("  -> La medida actual está ponderada por el ancho de banda; un menor pico alfa sube "
        "automáticamente beta/gamma relativos.")

    G = np.vstack(db["oscilatory_psd"].apply(lambda a: np.asarray(a, float)).values)
    lin = 10 ** G
    rows = {}
    for b, m in masks.items():
        rows[f"excess_{b}"] = (lin[:, m] - 1).sum(axis=1)
    ex = pd.concat([db[["subject", "SITE", "group"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    subj = ex.groupby(["subject", "SITE", "group"], as_index=False).mean(numeric_only=True)
    out = []
    hc = subj[subj["group"] == "HC"]
    for g in GROUP_ORDER:
        s = subj[subj["group"] == g]
        for b in CANONIC:
            c = f"excess_{b}"
            p = stats.mannwhitneyu(s[c], hc[c]).pvalue if len(s) else np.nan
            out.append({"group": g, "band": b, "n": len(s),
                        "pct_con_pico(exceso>0.05)": 100 * np.mean(s[c] > 0.05),
                        "pct_HC_con_pico": 100 * np.mean(hc[c] > 0.05),
                        "mediana_exceso": s[c].median(), "mediana_HC": hc[c].median(), "p_MW": p})
    t = pd.DataFrame(out)
    t["q"] = bh_fdr(t["p_MW"])
    sheets["potencia_periodica_exceso"] = t
    say("  Exceso periódico (sin ajustar por edad) en beta1/beta3/gamma, ACr vs todos los HC:")
    say(t[(t["group"] == "ACr") & t["band"].isin(["alpha1", "beta1", "beta3", "gamma"])].round(3).to_string(index=False))
    say("  Si ACr NO tiene más picos en gamma, el 'aumento gamma' viene del denominador (menos alfa), "
        "no de actividad gamma real -> discutirlo o cambiar a potencia periódica sum(10^G-1).")


# --------------------------------------------------------------------------- 10
def check_mci_subtypes():
    if not DEMOGRAPHIC_DIR:
        say("  DEMOGRAPHIC_DIR no configurado en config_local.py; se omite.")
        return
    p = os.path.join(DEMOGRAPHIC_DIR, "dem_CAUEEG2.xlsx")
    d = pd.read_excel(p)
    say(f"  Columnas en dem_CAUEEG2.xlsx: {list(d.columns)}")
    cand = [c for c in d.columns if any(k in c.lower() for k in ("symptom", "diag", "subtype", "label", "class"))]
    rows = []
    for c in cand:
        vals = d[c].dropna().astype(str)
        tokens = []
        for v in vals:
            try:
                x = ast.literal_eval(v)
                tokens.extend(x if isinstance(x, (list, tuple)) else [x])
            except Exception:
                tokens.extend([t.strip() for t in v.replace(";", ",").split(",")])
        vc = pd.Series(tokens).value_counts()
        for k, n in vc.items():
            rows.append({"column": c, "label": k, "n": int(n)})
    t = pd.DataFrame(rows)
    sheets["subtipos_MCI_CAUEEG"] = t
    if len(t):
        say(t.head(40).to_string(index=False))
        say("  Con estas etiquetas: (a) restringir MCI a amnésico/compatibles con EA, o "
            "(b) reportar z-scores estratificados por subtipo (comentario de AJJ).")
    else:
        say("  No se encontraron columnas de subtipo; revisar el annotation.json original de CAUEEG "
            "(campo 'symptom').")


if __name__ == "__main__":
    run("1. Muestra por sitio y grupo", check_sample)
    run("2. Solapamiento Medellín hd / ld", check_medellin_overlap)
    run("3. Valores fuera de (0,1) tras ComBat-GAM", check_logit_clipping)
    run("4. Calibración de z en HC de test", check_z_calibration)
    run("5. Normalidad de residuos en HC de entrenamiento", check_train_residuals)
    run("6. Proporción de desviaciones extremas por grupo", check_extreme_deviations)
    run("7. Exclusiones de QC (TF/IAFp, R2)", check_qc_report)
    run("8. Códigos de sujeto repetidos entre sitios (filtro R2)", check_repeated_codes)
    run("9. Potencia periódica vs relativa (hallazgo ACr)", check_periodic_power)
    run("10. Subtipos de MCI en CAUEEG", check_mci_subtypes)

    os.makedirs(BLR_DIR, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        pd.DataFrame({"resumen": summary}).to_excel(w, sheet_name="resumen", index=False)
        for name, t in sheets.items():
            t.to_excel(w, sheet_name=name[:31], index=False)
    print(f"\nGuardado: {OUT_XLSX}")
