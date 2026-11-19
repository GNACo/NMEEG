"""
Sensibilidad del hallazgo gamma en ACr al umbral de remoción muscular (revisión AJJ).

Re-ejecuta, a partir de los derivados `desc-reject` de APPLEE (Medellín hd y ld),
las últimas etapas del preprocesamiento con dos umbrales de ICA muscular
(0.7 = el usado en el artículo; 0.5 = valor por defecto de MNE) y luego:
scorEpochs (24 mejores épocas de 5 s) -> PSD multitaper -> FOOOF (fixed) en dos
rangos (1-40 Hz como en el artículo, 2-40 Hz fuera de la banda de transición del
filtro de 1 Hz). Por canal guarda el exceso periódico por banda sum(10^G - 1),
el exponente, R2 y el número de componentes musculares removidos.

Uso:
    python analysis/muscle_sensitivity_medellin.py <carpeta_hd> <carpeta_ld> <xlsx_edades> <salida.csv>
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from fooof import FOOOF

sys.path.append(os.environ.get("SCOREPOCHS_DIR", "/home/claude/scorepochs_py"))
from scorepochs import scorEpochs  # noqa: E402

mne.set_log_level("ERROR")
CAN = {"delta": (1.5, 6), "theta": (6, 8.5), "alpha1": (8.5, 10.5), "alpha2": (10.5, 12.5),
       "beta1": (12.5, 18.5), "beta2": (18.5, 21), "beta3": (21, 30), "gamma": (30, 40)}
THRESHOLDS = [0.7, 0.5]
FRANGES = {"1-40": [1, 40], "2-40": [2, 40]}
N_EPOCHS, EP_LEN = 24, 5


def group_of(code):
    if code.startswith(("G1", "GG")):
        return "ACr"
    if code.startswith(("CTR", "G2", "GU")):
        return "HC"
    return None


def remove_muscle(raw, thr):
    ica = ICA(random_state=97, n_components=len(raw.ch_names), method="infomax",
              fit_params=dict(extended=True))
    ica.fit(raw)
    _, scores = ica.find_bads_muscle(raw)
    ica.exclude = np.where(np.asarray(scores) > thr)[0].tolist()
    out = raw.copy()
    ica.apply(out)
    return out, len(ica.exclude)


def best_epochs(raw):
    data = raw.get_data()
    cfg = {"freqRange": [1, 40], "fs": int(raw.info["sfreq"]), "windowL": EP_LEN}
    idx_best, epochs, _ = scorEpochs(cfg, data)
    sel = idx_best[:N_EPOCHS]
    return np.stack([epochs[i] for i in sel]), len(sel)


def psd(ep, fs):
    # z-score por canal (como en el artículo), multitaper NW=4 -> half-bandwidth 0.8 Hz
    x = (ep - ep.mean(axis=-1, keepdims=True)) / ep.std(axis=-1, keepdims=True)
    p, f = mne.time_frequency.psd_array_multitaper(x, fs, fmin=0.5, fmax=45, bandwidth=1.6,
                                                   adaptive=False, normalization="full")
    return f, p.mean(axis=0)  # promedio sobre épocas -> (canales, freqs)


def fooof_feats(f, p_ch, frange):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=8, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, p_ch, frange)
    G = fm._peak_fit
    ff = fm.freqs
    lin = 10 ** G - 1
    row = {"exponent": fm.aperiodic_params_[1], "offset": fm.aperiodic_params_[0], "r2": fm.r_squared_}
    for b, (lo, hi) in CAN.items():
        m = (ff >= lo) & (ff <= hi)
        row[f"exc_{b}"] = lin[m].sum() if m.any() else np.nan
    return row


def main(hd_dir, ld_dir, ages_xlsx, out_csv):
    ages = pd.read_excel(ages_xlsx, sheet_name="harmonizeSITE_age_group")[["subject", "SITE", "age"]]
    files = [(p, "Medellin_hd") for p in glob.glob(os.path.join(hd_dir, "sub-*/ses-V0/eeg/*desc-reject_eeg.fif"))]
    files += [(p, "Medellin_ld") for p in glob.glob(os.path.join(ld_dir, "sub-*/ses-V0/eeg/*desc-reject_eeg.fif"))]
    rows = []
    for k, (path, site) in enumerate(sorted(files)):
        code = os.path.basename(path).split("_")[0].replace("sub-", "")
        grp = group_of(code)
        if grp is None:
            continue
        raw = mne.io.read_raw_fif(path, preload=True)
        raw.set_montage("standard_1005", match_case=False, on_missing="ignore")
        for thr in THRESHOLDS:
            try:
                clean, n_rm = remove_muscle(raw, thr)
                ep, n_ep = best_epochs(clean)
                f, P = psd(ep, raw.info["sfreq"])
            except Exception as e:  # registra y sigue
                print(f"ERROR {site} {code} thr={thr}: {e}", flush=True)
                continue
            for ci, ch in enumerate(raw.ch_names):
                for fr_name, fr in FRANGES.items():
                    try:
                        r = fooof_feats(f, P[ci], fr)
                    except Exception as e:  # canal plano o PSD inválida
                        print(f"FOOOF fallo {site} {code} {ch} thr={thr}: {e}", flush=True)
                        continue
                    r.update(subject=f"sub-{code}", SITE=site, group=grp, channel=ch,
                             muscle_thr=thr, n_muscle_removed=n_rm, n_epochs=n_ep, fooof_range=fr_name)
                    rows.append(r)
        print(f"[{k + 1}/{len(files)}] {site} {code} ({grp}) ok", flush=True)
    df = pd.DataFrame(rows).merge(ages, on=["subject", "SITE"], how="left")
    df.to_csv(out_csv, index=False)
    print(f"Guardado {out_csv}: {len(df)} filas")


if __name__ == "__main__":
    main(*sys.argv[1:5])
