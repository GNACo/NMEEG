import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')  # O cualquier otro backend compatible
import re 


psd_Dortmund = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Dortmund_EyesClosed_scorEpochs_zscore_long.feather")
psd_AR = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Argentina_rs_scorEpochs_zscore_long.feather")
psd_CL = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Chile_rs_scorEpochs_zscore_long.feather")
psd_cuba = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Cuba_CE_scorEpochs_zscore_long.feather")
psd_Med_ld = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Medellin_ld_CE_scorEpochs_zscore_long.feather")
psd_Med_hd = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Medellin_hd_CE_scorEpochs_zscore_long.feather")
psd_CAUEEG = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_CAUEEG_rs_scorEpochs_zscore_long.feather")
psd_SRM = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Oslo_resteyesc_scorEpochs_zscore_long.feather")
psd_poland = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_Poland_rest_scorEpochs_zscore_long.feather")
psd_greece = pd.read_feather(r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\psd\psd_GREECE_eyesclosed_scorEpochs_zscore_long.feather")

psd_CAUEEG['Sensors'] = psd_CAUEEG['Sensors'].replace({'T5': 'P7', 'T6': 'P8'})
psd_greece['Sensors'] = psd_greece['Sensors'].replace({'T5': 'P7', 'T6': 'P8'})

QA_Dortmund = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Dortmund.xlsx")
QA_AR = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Argentina.xlsx")
QA_CL = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Chile.xlsx")
QA_cuba = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Cuba.xlsx")
QA_Med_ld = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Medellin_ld.xlsx")
QA_Med_hd = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Medellin_hd.xlsx")
QA_CAUEEG = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_CAUEEG.xlsx")
QA_SRM = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Oslo.xlsx")
QA_poland = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_Poland.xlsx")
QA_greece = pd.read_excel(r"D:\MulticentersEEG\Features_2_normativeModel\Quality_data\0.7_scorEpochs\QC_GREECE.xlsx")

subj_drop_Dortmund = QA_Dortmund[QA_Dortmund['len_epochs'] < 24]['subject']
subj_drop_AR = QA_AR[QA_AR['len_epochs'] < 24]['subject']
subj_drop_CL = QA_CL[QA_CL['len_epochs'] < 24]['subject']
subj_drop_cuba = QA_cuba[QA_cuba['len_epochs'] < 24]['subject']
subj_drop_Med_ld = QA_Med_ld[QA_Med_ld['len_epochs'] < 24]['subject']
subj_drop_Med_hd = QA_Med_hd[QA_Med_hd['len_epochs'] < 24]['subject']
subj_drop_CAUEEG = QA_CAUEEG[QA_CAUEEG['len_epochs'] < 24]['subject']
subj_drop_SRM = QA_SRM[QA_SRM['len_epochs'] < 24]['subject']
subj_drop_poland = QA_poland[QA_poland['len_epochs'] < 24]['subject']
subj_drop_greece = QA_greece[QA_greece['len_epochs'] < 24]['subject']

# Filtrar el DataFrame eliminando esos sujetos
psd_Dortmund = psd_Dortmund[~psd_Dortmund['subject'].isin(subj_drop_Dortmund)]
psd_CAUEEG = psd_CAUEEG[~psd_CAUEEG['subject'].isin(subj_drop_CAUEEG)]

def calculate_IAF(data, plot = False, path= None):
    # Inicializar una lista para almacenar los resultados
    results = []

    # Iterar sobre cada fila (sensor) del DataFrame
    for index, row in data.iterrows():
        freqs = np.array(row['freqs']) # Convertir frecuencias de string a array
        psd = np.array(row['psd'])      # Convertir PSD de string a array

        # Calcular TF (mínimo entre 3 y 8 Hz)
        tf_mask = (freqs >= 3) & (freqs <= 8)
        TF = freqs[np.where(tf_mask)[0][np.argmin(psd[tf_mask])]]

        # Calcular alpha-BGF peak (máximo entre 6 y 14 Hz)
        alpha_mask = (freqs >= 6) & (freqs <= 14)
        IAFp = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        
        # # Para que no se presenten casos en los que tome IAFp < TF
        # if IAFp <= TF:
        #    tf_mask = (freqs >= 3) & (freqs <= IAFp)
        #    TF = freqs[np.where(tf_mask)[0][np.argmin(psd[tf_mask])]]
         
        # Calcular bandas individuales
        delta_band = (TF - 4, TF - 2)
        theta_band = (TF - 2, TF)
        bgf1_band = (TF, TF + (IAFp - TF) / 2)
        bgf2_band = (TF + (IAFp - TF) / 2, IAFp)
        bgf3_band = (IAFp, IAFp + 2)

        # Calculate power for each band
        def band_power(freq_range):
            mask = (freqs >= freq_range[0]) & (freqs < freq_range[1])
            return np.sum(psd[mask])
        
        delta_power = band_power(delta_band)
        theta_power = band_power(theta_band)
        alpha1_power = band_power(bgf1_band)
        alpha2_power = band_power(bgf2_band)
        alpha3_power = band_power(bgf3_band)
        
        total_power = delta_power + theta_power + alpha1_power + alpha2_power + alpha3_power

        # Calculate relative power
        delta_rel = delta_power / total_power
        theta_rel = theta_power / total_power
        alpha1_rel = alpha1_power / total_power
        alpha2_rel = alpha2_power / total_power
        alpha3_rel = alpha3_power / total_power
        # Almacenar los resultados
        results.append({
            'subject': row['subject'],
            'Sensors': row['Sensors'],
            'group': row['group'],
            'freqs': row['freqs'],
            'psd': row['psd'],
            'center': row['center'],
            'TF': TF,
            'IAFp': IAFp,
            'Delta Band': delta_band,
            'Theta Band': theta_band,
            'BGF1 Band': bgf1_band,
            'BGF2 Band': bgf2_band,
            'BGF3 Band': bgf3_band,
            'Delta_Rel': delta_rel,
            'Theta_Rel': theta_rel,
            'Alpha1_Rel': alpha1_rel,
            'Alpha2_Rel': alpha2_rel,
            'Alpha3_Rel': alpha3_rel
        })
        
        if plot:
            if row["Sensors"] == 'P7' or row["Sensors"] == 'P8' or row["Sensors"] == 'O1' or row["Sensors"] == 'O2':
                    
                plt.figure(figsize=(10, 6))
                plt.plot(freqs, psd, label='PSD')
                plt.axvline(TF, color='r', linestyle='--', label=f'TF ({TF:.2f} Hz)')
                plt.axvline(IAFp, color='g', linestyle='--', label=f'IAFp ({IAFp:.2f} Hz)')
                # Añadir texto con los valores de TF e IAF en el gráfico
                plt.text(TF, max(psd) * 0.8, f'TF\n{TF:.2f} Hz', color='red', fontsize=9, ha='right', va='center')
                plt.text(IAFp, max(psd) * 0.6, f'IAFp\n{IAFp:.2f} Hz', color='green', fontsize=9, ha='right', va='center')

                plt.fill_between(freqs, psd, where=(freqs >= delta_band[0]) & (freqs < delta_band[1]), color='blue', alpha=0.3, label='Delta Band')
                plt.fill_between(freqs, psd, where=(freqs >= theta_band[0]) & (freqs < theta_band[1]), color='purple', alpha=0.3, label='Theta Band')
                plt.fill_between(freqs, psd, where=(freqs >= bgf1_band[0]) & (freqs < bgf1_band[1]), color='orange', alpha=0.3, label='BGF1 Band')
                plt.fill_between(freqs, psd, where=(freqs >= bgf2_band[0]) & (freqs < bgf2_band[1]), color='red', alpha=0.3, label='BGF2 Band')
                plt.fill_between(freqs, psd, where=(freqs >= bgf3_band[0]) & (freqs < bgf3_band[1]), color='green', alpha=0.3, label='BGF3 Band')
                plt.xlim(0, 20)
                plt.yscale('log')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power')
                plt.title(rf'{row["subject"]} - {row["Sensors"]} - {row["group"]}')
                plt.legend()
                if TF >= 4:
                    path_save = f'{path}/{row["center"]}/{row["Sensors"]}'
                else:
                    path_save = f'{path}/{row["center"]}/TF_below_4/{row["Sensors"]}'
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
                subject_safe = re.sub(r'[\\/*?:"<>|]', '_', row['subject'])
                sensor_safe = re.sub(r'[\\/*?:"<>|]', '_', row['Sensors'])
                path_save = os.path.normpath(path_save)  # Normalizar la ruta para el sistema operativo
                filename = f'{subject_safe}_{sensor_safe}.png'

                plt.savefig(os.path.join(path_save, filename))
                plt.close()
            else:
                continue    
        
            

    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_feather(rf'{path}\IAF_{row["center"]}_scorEpochs_zscore_columns.feather')
    print(f'Done {row["center"]}!')
    return results_df


path_psd= r'D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\psd_graphs'
plot = True 
data_IAF_AR = calculate_IAF(psd_AR,plot= plot, path=path_psd)
data_IAF_CL = calculate_IAF(psd_CL,plot= plot, path=path_psd)
data_IAF_Med_ld = calculate_IAF(psd_Med_ld,plot= plot, path=path_psd)
data_IAF_Med_hd = calculate_IAF(psd_Med_hd,plot= plot, path=path_psd)
data_IAF_cuba = calculate_IAF(psd_cuba,plot= plot, path=path_psd)
data_IAF_Dortmund = calculate_IAF(psd_Dortmund,plot= plot, path=path_psd)
data_IAF_CAUEEG = calculate_IAF(psd_CAUEEG,plot= plot, path=path_psd)
data_IAF_SRM = calculate_IAF(psd_SRM,plot= plot, path=path_psd)
data_IAF_poland = calculate_IAF(psd_poland,plot= plot, path=path_psd)
data_IAF_greece = calculate_IAF(psd_greece,plot= plot, path=path_psd)