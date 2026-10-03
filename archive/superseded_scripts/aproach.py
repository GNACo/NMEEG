
import pandas as pd
from itertools import product

def process_iaf_fooof_rois(df, save_path=None, save_name=None):
    df_rois = df.copy()

    # Mapas de ROIs a sus electrodos
    roi_pairs = {
        "roiF": ["FP1", "FP2"],  
        "roiC": ["C3", "C4"],  
        "roiP": ["P7", "P8"],
        "roiO": ["O1", "O2"],
        "roiPO": ['FP1','FP2','C3','C4','P7','P8','O1','O2']
    }

    # Posibles prefijos y sufijos de columnas que componen los nombres
    prefixes = [
        "TF", "IAFp",
        "pw_delta", "pw_theta", "pw_BGF1", "pw_BGF2", "pw_BGF3", "pw_beta",
        "pw_ab_delta", "pw_ab_theta", "pw_ab_BGF1", "pw_ab_BGF2", "pw_ab_BGF3", "pw_ab_beta",
        "pw_canonic_delta", "pw_canonic_theta", "pw_canonic_alpha1", "pw_canonic_alpha2",
        "pw_canonic_beta1", "pw_canonic_beta2", "pw_canonic_beta3", "pw_canonic_gamma",
        "pw_ab_canonic_delta", "pw_ab_canonic_theta", "pw_ab_canonic_alpha1", "pw_ab_canonic_alpha2",
        "pw_ab_canonic_beta1", "pw_ab_canonic_beta2", "pw_ab_canonic_beta3", "pw_ab_canonic_gamma",
        "osc_pw_ab_delta", "osc_pw_ab_theta", "osc_pw_ab_BGF1", "osc_pw_ab_BGF2", "osc_pw_ab_BGF3", "osc_pw_ab_beta",
        "osc_pw_delta", "osc_pw_theta", "osc_pw_BGF1", "osc_pw_BGF2", "osc_pw_BGF3", "osc_pw_beta",
        "osc_pw_ab_canonic_delta", "osc_pw_ab_canonic_theta", "osc_pw_ab_canonic_alpha1", "osc_pw_ab_canonic_alpha2",
        "osc_pw_ab_canonic_beta1", "osc_pw_ab_canonic_beta2", "osc_pw_ab_canonic_beta3", "osc_pw_ab_canonic_gamma",
        "osc_pw_canonic_delta", "osc_pw_canonic_theta", "osc_pw_canonic_alpha1", "osc_pw_canonic_alpha2",
        "osc_pw_canonic_beta1", "osc_pw_canonic_beta2", "osc_pw_canonic_beta3", "osc_pw_canonic_gamma"
    ]

    # Casos especiales (que no siguen el patrón general)
    special_features = {
        "exponent": ["exponent"],
        "offset": ["offset"],
        "cf_extalpha": ["CF_extalpha"]
    }

    # Construir el diccionario de columnas a promediar
    groups = {}

    # Generar automáticamente todas las combinaciones posibles
    for prefix, (roi_name, channels) in product(prefixes, roi_pairs.items()):
        columns = [f"{prefix}_{ch}" for ch in channels if f"{prefix}_{ch}" in df.columns]
        if len(columns) == len(channels):  # Solo si ambas columnas existen
            groups[f"{prefix}_{roi_name}"] = columns

    # Agregar especiales manualmente
    for base_name, base_cols in special_features.items():
        for roi_name, channels in roi_pairs.items():
            columns = [f"{col}_{ch}" for col in base_cols for ch in channels if f"{col}_{ch}" in df.columns]
            if len(columns) == len(channels):
                groups[f"{base_name}_{roi_name}"] = columns

    # Calcular los promedios
    df_metrics = pd.DataFrame()
    for key, cols in groups.items():
        df_rois[cols] = df_rois[cols].apply(pd.to_numeric, errors='coerce')
        df_metrics[key] = df_rois[cols].mean(axis=1)

    # Construir DataFrame final
    if save_name == 'rois_age':
        rois = pd.concat([df.iloc[:, :3], df_metrics, df.iloc[:, -1]], axis=1)
    elif save_name == 'rois_age_sex':
        rois = pd.concat([df.iloc[:, :3], df_metrics, df.iloc[:, -2:]], axis=1)
    else:
        rois = pd.concat([df.iloc[:, :3], df_metrics, df.iloc[:, -3:]], axis=1)

    if save_path and save_name:
        rois.to_feather(f'{save_path}/{save_name}.feather')
    print(rois)
    return rois


