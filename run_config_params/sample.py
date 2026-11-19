"""
Definición única de la muestra analizada (revisión AJJ, nov-2026).

Los participantes MCI y AD reclutados en Medellín pertenecen al kindred
PSEN1 E280A (forma autosómica dominante, inicio temprano) y se excluyen de las
comparaciones con MCI/AD esporádicos. Antes solo se excluían los MCI; los 3 AD de
Medellin_hd seguían en el grupo AD. Todos los scripts deben usar esta función.
"""
import pandas as pd

# Greece ya no se excluye: su tamaño (n=3) era consecuencia del filtro TF/IAFp, que
# eliminaba al 97% de sus registros. pipeline_v2.py lo desactiva por defecto.
EXCLUDED_SITES = []
KINDRED_SITES = ["Medellin_hd", "Medellin_ld"]
KINDRED_EXCLUDED_GROUPS = ["MCI", "AD"]


def analysis_sample(df: pd.DataFrame, site_col: str = "SITE", group_col: str = "group") -> pd.DataFrame:
    """Quita Greece y los MCI/AD del kindred PSEN1 de Medellín."""
    site = df[site_col].astype(str)
    drop = site.isin(EXCLUDED_SITES) | (
        site.isin(KINDRED_SITES) & df[group_col].isin(KINDRED_EXCLUDED_GROUPS))
    return df[~drop].copy()
