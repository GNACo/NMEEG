import sys
sys.path.append(r'D:\flujo_portables\portables')
from run_config_params.DATASET import Dortmund, AR, CL, CHBMP, PORTABLES_CE, BIOMARCADORES, CAUEEG_CONCAT, SRM, Poland, GREECE, DUQUE
import pandas as pd

PATH_DEM = {
    #'AR': {'dem': AR['demographic'],'variables_dem':['MoCA_total','ifs_total_score']},
    #'CL': {'dem': CL['demographic'],'variables_dem':['MoCA_total','ifs_total_score']},
    #'CHBMP': {'dem': CHBMP['demographic'], 'variables_dem':['MMSE_total']},
    #'PORTABLES_CE': {'dem': r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_portables.xlsx",'variables_dem':['MMSE_total','MoCA_total']},
    #'BIOMARCADORES': {'dem':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_biomarcadores.xlsx",'variables_dem':['MMSE_total']},
    #'SRM': {'dem': SRM['demographic'],'variables_dem':['ravlt_tot']},
    #'Poland': {'dem': Poland['demographic'],'variables_dem':['CVLT_score_1_5']},
    #'GREECE': {'dem': GREECE['demographic'],'variables_dem':['MMSE']},
    #"Spain": {'dem': r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_Madrid_c3n.xlsx", 'variables_dem': ['MMSE_total', 'MoCA_total']},
    "COL_PRG14-1-02": {'dem': r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_Duque.xlsx", 'variables_dem': ['MMSE_total']},
}

import pandas as pd
from scipy.stats import zscore

# Iterar sobre cada entrada en PATH_DEM
for key, value in PATH_DEM.items():
    # Leer el archivo Excel usando la ruta en 'dem'
    try:
        df = pd.read_excel(value['dem'])
    except FileNotFoundError:
        print(f"El archivo para {key} no se encontró en la ruta {value['dem']}.")
        continue
    # Procesar cada variable en 'variables_dem'
    for var in value['variables_dem']:
        if var in df.columns:
            # Crear una nueva columna con el z-score de la variable
            zscore_column = f"{var}_zscore"
            df[zscore_column] = zscore(df[var], nan_policy='omit')  # Evitar problemas con NaN
        else:
            print(f"La columna {var} no se encontró en el archivo para {key}.")

    # Guardar el DataFrame actualizado
    output_path = value['dem'].replace(".xlsx", "_zscore.xlsx")  # Cambiar nombre de salida
    try:
        df.to_excel(output_path, index=False)
        print(f"Archivo con z-scores guardado en: {output_path}")
    except Exception as e:
        print(f"No se pudo guardar el archivo para {key}. Error: {e}")
