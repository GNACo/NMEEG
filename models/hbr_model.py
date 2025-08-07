import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pcntoolkit as ptk
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm

# =============================================================================
# 1. CONFIGURACIONES Y CARGA DE DATOS
# =============================================================================
# Directorio principal y CSV original
main_directory = input('path data')#r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\neuroharmonize"
csv_file = os.path.join(main_directory, "roi_PO_age.feather")

# Directorio de procesamiento (donde se guardarán todos los archivos intermedios y el modelo)
processing_dir = input('path save model')#r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\NormativeModel\HBR"
os.makedirs(processing_dir, exist_ok=True)
os.chdir(processing_dir)
processing_dir = os.getcwd()

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Cargar datos
df = pd.read_feather(csv_file)

# Definir columnas a excluir y determinar IDPs automáticamente
cols_excluir = ['Unnamed: 0', 'subject', 'group', 'SITE', 'age']
idps= ['avg_exponent_ROI_PO',
       'avg_offset_ROI_PO', 'avg_cf_extalpha_ROI_PO',
       'avg_fooof_power_ab_delta_ROI_PO', 'avg_fooof_power_ab_theta_ROI_PO',
       'avg_fooof_power_ab_BGF1_ROI_PO', 'avg_fooof_power_ab_BGF2_ROI_PO',
       'avg_fooof_power_ab_BGF3_ROI_PO', 'avg_fooof_power_ab_beta_ROI_PO']
#idps = [col for col in df.columns if col not in cols_excluir]
print("Se modelarán las siguientes IDPs:", idps)

# =============================================================================
# 2. ASIGNAR SITIO (sitenum) Y SEPARAR POR GRUPO
# =============================================================================
# Separamos Oslo y otros (para asignar números de sitio)
oslo_df = df.loc[df['SITE'] == 'Oslo'].copy()
others_df = df.loc[df['SITE'] != 'Oslo'].copy()

# Asignamos números a los sitios de "others"
unique_sites = others_df['SITE'].unique()
site_map = {s: i for i, s in enumerate(unique_sites)}
others_df['sitenum'] = others_df['SITE'].map(site_map)
# Para Oslo, asignamos un número distinto (por ejemplo, max+1)
oslo_df['sitenum'] = max(site_map.values()) + 1

# Unir ambos
df_all = pd.concat([others_df, oslo_df], ignore_index=True)

# Separar por grupo: entrenar solo con controles (HC)
df_HC = df_all.loc[df_all['group'] == 'HC'].copy()
df_mci = df_all.loc[df_all['group'] == 'mci'].copy()

# =============================================================================
# 3. DIVIDIR LOS CONTROLES EN TRAIN/TEST (PARTICIÓN INTERNA)
# =============================================================================
mask_tr = np.random.rand(len(df_HC)) > 0.7
mask_te = ~mask_tr
df_HC_tr = df_HC.loc[mask_tr].copy()
df_HC_te = df_HC.loc[mask_te].copy()

# =============================================================================
# 4. GUARDAR LOS DATOS (CSV) PARA USO FUTURO
# =============================================================================
df_HC_tr.to_csv(os.path.join(processing_dir, 'HC_train_tr.csv'), index=False)
df_HC_te.to_csv(os.path.join(processing_dir, 'HC_train_te.csv'), index=False)
df_mci.to_csv(os.path.join(processing_dir, 'mci.csv'), index=False)

# =============================================================================
# 5. GENERAR X, Y, BATCH EFFECTS (Para controles) y guardar en pickle
# =============================================================================
def generar_datos(df_, idps_list):
    # Covariable: edad escalada (edad/100)
    X = (df_['age'] / 100).to_numpy(dtype=float).reshape(-1,1)
    # Respuestas: IDPs
    Y = df_[idps_list].to_numpy(dtype=float)
    # Batch effects: sitenum
    BE = df_[['sitenum']].to_numpy(dtype=int)
    return X, Y, BE

X_train, Y_train, BE_train = generar_datos(df_HC_tr, idps)
X_test, Y_test, BE_test    = generar_datos(df_HC_te, idps)

with open('X_train.pkl','wb') as f:
    pickle.dump(pd.DataFrame(X_train), f)
with open('Y_train.pkl','wb') as f:
    pickle.dump(pd.DataFrame(Y_train, columns=idps), f)
with open('trbefile.pkl','wb') as f:
    pickle.dump(pd.DataFrame(BE_train), f)

with open('X_test.pkl','wb') as f:
    pickle.dump(pd.DataFrame(X_test), f)
with open('Y_test.pkl','wb') as f:
    pickle.dump(pd.DataFrame(Y_test, columns=idps), f)
with open('tsbefile.pkl','wb') as f:
    pickle.dump(pd.DataFrame(BE_test), f)

# =============================================================================
# 6. ENTRENAR EL MODELO NORMATIVO CON HBR (Solo controles)
# =============================================================================
covfile = os.path.join(processing_dir, 'X_train.pkl')
respfile = os.path.join(processing_dir, 'Y_train.pkl')
trbefile = os.path.join(processing_dir, 'trbefile.pkl')
testcovfile = os.path.join(processing_dir, 'X_test.pkl')
testrespfile = os.path.join(processing_dir, 'Y_test.pkl')
tsbefile = os.path.join(processing_dir, 'tsbefile.pkl')

output_path = os.path.join(processing_dir, 'Models')
log_dir     = os.path.join(processing_dir, 'log')
os.makedirs(output_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
outputsuffix = '_estimate'

ptk.normative.estimate(
    covfile=covfile,
    respfile=respfile,
    tsbefile=tsbefile,
    trbefile=trbefile,
    alg='hbr',
    log_path=log_dir,
    binary=True,
    output_path=output_path,
    testcov=testcovfile,
    testresp=testrespfile,
    outputsuffix=outputsuffix,
    savemodel=True
)

print("¡Modelo Normativo HBR entrenado y guardado!")



# =============================================================================
# 8. TRANSFERENCIA: APLICAR EL MODELO A SUJETOS mci
# =============================================================================
# Generar datos para mci
def generar_datos_mci(df_, idps_list):
    X = (df_['age'] / 100).to_numpy(dtype=float).reshape(-1,1)
    Y = df_[idps_list].to_numpy(dtype=float)
    BE = df_[['sitenum']].to_numpy(dtype=int)
    return X, Y, BE

X_mci, Y_mci, BE_mci = generar_datos_mci(df_mci, idps)
with open("X_mci.pkl", "wb") as f:
    pickle.dump(pd.DataFrame(X_mci), f)
with open("Y_mci.pkl", "wb") as f:
    pickle.dump(pd.DataFrame(Y_mci, columns=idps), f)
with open("mci_befile.pkl", "wb") as f:
    pickle.dump(pd.DataFrame(BE_mci), f)

transfer_output_path = os.path.join(processing_dir, "Transfer")
transfer_log_dir = os.path.join(processing_dir, "log_transfer")
os.makedirs(transfer_output_path, exist_ok=True)
os.makedirs(transfer_log_dir, exist_ok=True)
outputsuffix_transfer = "_transfer"

yhat_mci, s2_mci, z_scores_mci = ptk.normative.transfer(
    covfile=os.path.join(processing_dir, "X_mci.pkl"),
    respfile=os.path.join(processing_dir, "Y_mci.pkl"),
    tsbefile=tsbefile,
    trbefile=trbefile,  # Usamos el mismo modelo entrenado
    model_path=output_path,
    alg="hbr",
    log_path=transfer_log_dir,
    binary=True,
    output_path=transfer_output_path,
    testcov=os.path.join(processing_dir, "X_mci.pkl"),
    testresp=os.path.join(processing_dir, "Y_mci.pkl"),
    outputsuffix=outputsuffix_transfer,
    savemodel=True
)

print("Transferencia realizada para mci, resultados guardados en:", transfer_output_path)

# =============================================================================
# 9. GENERAR INFORME CON GRÁFICOS POR IDP
# =============================================================================
# Queremos graficar la curva normativa (obtenida de la estimación en HC) y superponer
# los puntos reales de HC y mci en diferentes colores para ver las desviaciones.
#
# En este ejemplo usaremos los resultados de la estimación del modelo (guardados en Models)
# y los datos completos de HC y mci (sin partición) para graficar la trayectoria normativa.

# Cargar datos completos de HC y mci
df_HC_full = df_HC.copy()  # todos los controles
df_mci_full = df_mci.copy()  # todos los mci

# Para graficar la curva normativa, vamos a utilizar los datos de test de HC.
# Cargamos X_test y las predicciones del modelo
with open(os.path.join(output_path, "yhat_estimate.pkl"), "rb") as f:
    yhat_est = pickle.load(f)
with open(os.path.join(output_path, "ys2_estimate.pkl"), "rb") as f:
    ys2_est = pickle.load(f)

# Convertir a numpy arrays si son DataFrame
if isinstance(yhat_est, pd.DataFrame):
    yhat_est = yhat_est.values
if isinstance(ys2_est, pd.DataFrame):
    ys2_est = ys2_est.values
with open("X_test.pkl", "rb") as f:
    X_test_loaded = pickle.load(f)
if isinstance(X_test_loaded, pd.DataFrame):
    X_test_loaded = X_test_loaded.values

# La covariable es la edad escalada (edad/100); convertimos a edad real
age_HC = X_test_loaded.flatten() * 100

# Calculamos la banda de confianza a partir de la varianza
std_est = np.sqrt(ys2_est)
ci95_est = 1.96 * std_est

# Usaremos LOWESS para suavizar la curva normativa
lowess = sm.nonparametric.lowess
frac_val = 0.3  # Ajusta este parámetro según lo suave que desees

# Creamos un PDF para guardar un gráfico por IDP
pdf_path = os.path.join(processing_dir, "informe_normativo.pdf")
with PdfPages(pdf_path) as pdf:
    for i, idp in enumerate(idps):
        plt.figure(figsize=(10,6))
        # Extraemos la predicción y banda para la IDP i
        yhat_idp = yhat_est[:, i]
        ci95_idp = ci95_est[:, i]
        # Ordenamos por edad
        order = np.argsort(age_HC)
        ages_sorted = age_HC[order]
        yhat_sorted = yhat_idp[order]
        ci95_sorted = ci95_idp[order]
        # Suavizamos la curva normativa usando LOWESS
        smooth = lowess(yhat_sorted, ages_sorted, frac=frac_val, return_sorted=True)
        
        # Graficar la curva normativa suavizada
        plt.plot(smooth[:,0], smooth[:,1], color="black", linewidth=2, label="Curva Normativa (HC)")
        # Graficar la banda de confianza (sin suavizado)
        plt.fill_between(ages_sorted, yhat_sorted - ci95_sorted, yhat_sorted + ci95_sorted,
                         color="gray", alpha=0.3, label="Intervalo 95%")
        
        # Graficar puntos reales:
        # HC en azul
        plt.scatter(df_HC_full["age"], df_HC_full[idp], color="blue", alpha=0.5, label="HC", s=20)
        # mci en rojo
        plt.scatter(df_mci_full["age"], df_mci_full[idp], color="red", alpha=0.5, label="mci", s=20)
        
        plt.xlabel("Edad (años)")
        plt.ylabel(idp)
        plt.title(f"Normative Model para {idp} vs. Edad")
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

print("Informe generado en:", pdf_path)




# Revisar si es mejor predecir o hacer un transfer 
# =============================================================================
# 7. GENERAR LA CURVA NORMATIVA A LO LARGO DE UNA REJILLA DE EDAD
# =============================================================================
# Para generar la curva (y banda) para cada IDP, usaremos la función predict() con archivos dummy.
# Dado que el entrenamiento ya se hizo, no queremos reentrenar; por ello reutilizamos el modelo guardado.

# -- Definir función para predecir en una rejilla de edades (para todos los IDPs) --
# def generar_curva_normativa(age_min, age_max, n_points, site_val, model_path, outputsuffix):
#     """
#     Genera la predicción (y varianza) del modelo normativo en una rejilla de edades,
#     fijando el batch effect a site_val.
#     Devuelve: age_grid (array n_points,1), yhat (n_points, n_features), s2 (n_points, n_features)
#     """
#     # Creamos la rejilla de covariables (edad escalada)
#     age_grid = np.linspace(age_min, age_max, n_points).reshape(-1,1)
#     # Guardar dummy_cov.txt
#     dummy_cov_file = 'dummy_cov.txt'
#     pd.DataFrame(age_grid).to_csv(dummy_cov_file, sep=' ', header=False, index=False)
#     # Crear dummy batch effects (fijado a site_val)
#     dummy_be = np.ones((n_points, 1), dtype=int) * site_val
#     dummy_be_file = 'dummy_be.txt'
#     pd.DataFrame(dummy_be).to_csv(dummy_be_file, sep=' ', header=False, index=False)
#     # Crear dummy resp y dummy mask (con forma adecuada, 0 para resp y 1 para mask)
#     # Se asume que el modelo entrenado tiene tantas columnas como len(idps)
#     n_features = len(idps)
#     dummy_resp = np.zeros((n_points, n_features))
#     dummy_resp_file = 'dummy_resp.txt'
#     pd.DataFrame(dummy_resp).to_csv(dummy_resp_file, sep=' ', header=False, index=False)
#     dummy_mask = np.ones((n_points, n_features))
#     dummy_mask_file = 'dummy_mask.txt'
#     pd.DataFrame(dummy_mask).to_csv(dummy_mask_file, sep=' ', header=False, index=False)
    
#     # Para evitar errores en la evaluación (usando dummy), sobrescribimos compute_MSLL temporalmente
#     import pcntoolkit.util.utils as ptu
#     ptu.compute_MSLL = lambda ytrue, ypred, ypred_var, train_mean, train_var: np.zeros((ypred.shape[0], ypred.shape[1]))
    
#     # Llamamos a predict() con meta_data=False, metrics=[], warp=True
#     yhat, s2, _ = ptk.normative.predict(
#         covfile=dummy_cov_file,
#         respfile=dummy_resp_file,
#         maskfile=dummy_mask_file,
#         befile=dummy_be_file,
#         model_path=model_path,
#         alg='hbr',
#         outputsuffix=outputsuffix,
#         log_path=log_dir,
#         meta_data=False,
#         metrics=[],
#         warp=True
#     )
#     return age_grid, yhat, s2

# # Determinar el rango de edad a partir de los datos (recordar que X_train es edad escalada)
# with open('X_train.pkl','rb') as f:
#     X_train_loaded = pickle.load(f).values
# age_min_train = X_train_loaded.min()  # en escala (edad/100)
# age_max_train = X_train_loaded.max()

# # =============================================================================
# # 8. GENERAR INFORME CON GRÁFICOS PARA CADA IDP
# # =============================================================================
# # Cargar datos de controles y mci para graficar puntos reales
# df_HC_full = df_HC.copy()  # controles (HC) de todo el dataset
# df_mci_full = df_mci.copy()

# # Convertir la edad escalada a edad real en controles y mci
# df_HC_full['age_real'] = df_HC_full['age']
# df_mci_full['age_real'] = df_mci_full['age']

# # Número de puntos en la rejilla para la curva
# n_points = 100
# # Elegimos un valor de sitio (podemos usar, por ejemplo, el valor modal de sitenum en controles)
# site_val = df_HC_full['sitenum'].mode()[0]

# # Generamos la curva normativa para todos los IDPs usando el modelo entrenado
# age_grid, yhat_grid, s2_grid = generar_curva_normativa(age_min_train, age_max_train, n_points, site_val, model_path, outputsuffix)
# std_grid = np.sqrt(s2_grid)
# ci95_grid = 1.96 * std_grid

# # Creamos un PDF para guardar el informe (una página por IDP)
# pdf_path = os.path.join(processing_dir, "informe_modelo_normativo.pdf")
# with PdfPages(pdf_path) as pdf:
#     # Iteramos por cada IDP
#     for i, idp in enumerate(idps):
#         plt.figure(figsize=(10,6))
#         # Graficamos la curva normativa: extraemos la columna i de las predicciones y bandas
#         yhat_idp_grid = yhat_grid[:, i]
#         ci95_idp_grid = ci95_grid[:, i]
#         # Opcional: podemos suavizar la curva con LOWESS
#         lowess = sm.nonparametric.lowess
#         frac_val = 0.3
#         data_for_lowess = np.column_stack((age_grid.flatten()*100, yhat_idp_grid))
#         data_sorted = data_for_lowess[data_for_lowess[:,0].argsort()]
#         ages_sorted = data_sorted[:,0]
#         yhat_sorted = data_sorted[:,1]
#         smooth_result = lowess(yhat_sorted, ages_sorted, frac=frac_val, return_sorted=True)
        
#         # Graficamos la curva suavizada
#         plt.plot(smooth_result[:,0], smooth_result[:,1], color='black', linewidth=2, label='Curva Normativa')
#         # Graficamos la banda de confianza (sin suavizado; o se podría suavizar también)
#         plt.fill_between(age_grid.flatten()*100, (yhat_grid[:, i] - ci95_idp_grid), (yhat_grid[:, i] + ci95_idp_grid),
#                          color='gray', alpha=0.3, label='Intervalo 95%')
        
#         # Graficar puntos reales:
#         # Controles (HC) en azul
#         plt.scatter(df_HC_full['age_real'], df_HC_full[idp], color='blue', alpha=0.5, label='Controles (HC)', s=20)
#         # mci en rojo
#         plt.scatter(df_mci_full['age_real'], df_mci_full[idp], color='red', alpha=0.5, label='mci', s=20)
        
#         plt.xlabel("Edad (años)")
#         plt.ylabel(idp)
#         plt.title(f"Normative Model para {idp} vs. Edad")
#         plt.legend()
#         plt.grid(True)
#         pdf.savefig()  # guarda la figura en el PDF
#         plt.close()

# print(f"Informe generado en: {pdf_path}")
