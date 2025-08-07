import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import re 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from neuroHarmonize import harmonizationLearn
os.chdir('d:/flujo_portables/NormativeModel_EEG')
from Functions.graphs_functions import plot_pca, plot_umap, plot_distance, plot_spearman_correlation, plot_spearman_scatter_single_winner
from Functions.staticals_test import staticals_features
from reComBat import reComBat



def calculate_IAF(data, plot = False, path= None, canonic_bands = dict):
    # Inicializar una lista para almacenar los resultados
    results = []

    # Iterar sobre cada fila (sensor) del DataFrame
    for index, row in data.iterrows():
        freqs = np.array(row['freqs']) # Convertir frecuencias de string a array
        psd = np.array(row['psd'])      # Convertir PSD de string a array

        # Calcular TF (mínimo entre 3 y 8 Hz)
        tf_mask = (freqs >= 3) & (freqs <= 8)
        original_TF = freqs[np.where(tf_mask)[0][np.argmin(psd[tf_mask])]]
        TF = original_TF
        
        # Calcular alpha-BGF peak (máximo entre 6 y 14 Hz)
        alpha_mask = (freqs >= 6) & (freqs <= 14)
        IAFp = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        
        TF_below_3 = False
        TF_corrected = False
        if TF <4:
            if TF > 3:
                TF_corrected = True
                TF = 4
            else:
                TF_below_3 = True

        # Calcular bandas individuales
        # Limitar la banda delta cuando empieza en 0, por el filtro a 1 hz 
        if TF -4 < 1:
            delta_band = (1, TF - 2)
            
        else:
            delta_band = (TF - 4, TF - 2)
        theta_band = (TF - 2, TF)
        bgf1_band = (TF, TF + (IAFp - TF) / 2)
        bgf2_band = (TF + (IAFp - TF) / 2, IAFp)
        bgf3_band = (IAFp, IAFp + 2)
        beta_band = (IAFp + 2, 30)
        print('beta',beta_band)
        # Calculate pw for each band
        def band_pw(freq_range):
            mask = (freqs >= freq_range[0]) & (freqs < freq_range[1])
            return np.sum(psd[mask])
        
        # Invididual band pw
        delta_pw = band_pw(delta_band)
        theta_pw = band_pw(theta_band)
        alpha1_pw = band_pw(bgf1_band)
        alpha2_pw = band_pw(bgf2_band)
        alpha3_pw = band_pw(bgf3_band)
        beta_pw = band_pw(beta_band)
        
        # Calculate canonical bands pw
        delta_pw_canonic =band_pw(canonic_bands['delta'])
        theta_pw_canonic =band_pw(canonic_bands['theta'])
        alpha1_pw_canonic =band_pw(canonic_bands['alpha1'])
        alpha2_pw_canonic =band_pw(canonic_bands['alpha2'])
        beta1_pw_canonic =band_pw(canonic_bands['beta1'])
        beta2_pw_canonic =band_pw(canonic_bands['beta2'])
        beta3_pw_canonic =band_pw(canonic_bands['beta3'])
        gamma_pw_canonic =band_pw(canonic_bands['gamma'])
        
        total_pw = delta_pw + theta_pw + alpha1_pw + alpha2_pw + alpha3_pw + beta_pw
        total_pw_canonic = delta_pw_canonic + theta_pw_canonic + alpha1_pw_canonic + alpha2_pw_canonic + beta1_pw_canonic + beta2_pw_canonic + beta3_pw_canonic + gamma_pw_canonic
        
        # Calculate relative pw
        delta_rel = delta_pw / total_pw
        theta_rel = theta_pw / total_pw
        alpha1_rel = alpha1_pw / total_pw
        alpha2_rel = alpha2_pw / total_pw
        alpha3_rel = alpha3_pw / total_pw
        beta_rel = beta_pw / total_pw
        
        # Calculate relative pw canonic
        delta_rel_canonic = delta_pw_canonic / total_pw_canonic
        theta_rel_canonic = theta_pw_canonic / total_pw_canonic
        alpha1_rel_canonic = alpha1_pw_canonic / total_pw_canonic
        alpha2_rel_canonic = alpha2_pw_canonic / total_pw_canonic
        beta1_rel_canonic = beta1_pw_canonic / total_pw_canonic
        beta2_rel_canonic = beta2_pw_canonic / total_pw_canonic
        beta3_rel_canonic = beta3_pw_canonic / total_pw_canonic
        gamma_rel_canonic = gamma_pw_canonic / total_pw_canonic
        
        
        # Almacenar los resultados
        results.append({
            'subject': row['subject'],
            'Sensors': row['Sensors'],
            'group': row['group'],
            'freqs': row['freqs'],
            'psd': row['psd'],
            'SITE': row['SITE'],
            'TF_original': original_TF,
            'TF_below_3': TF_below_3,
            'TF_corrected': TF_corrected,
            'TF': TF,
            'IAFp': IAFp,
            'delta_Band': delta_band,
            'theta_Band': theta_band,
            'BGF1_Band': bgf1_band,
            'BGF2_Band': bgf2_band,
            'BGF3_Band': bgf3_band,
            'beta_Band': beta_band,
            'pw_delta': delta_rel,
            'pw_theta': theta_rel,
            'pw_BGF1': alpha1_rel,
            'pw_BGF2': alpha2_rel,
            'pw_BGF3': alpha3_rel,
            'pw_beta': beta_rel,
            'pw_ab_delta': delta_pw,
            'pw_ab_theta': theta_pw,
            'pw_ab_BGF1': alpha1_pw,
            'pw_ab_BGF2': alpha2_pw,
            'pw_ab_BGF3': alpha3_pw,
            'pw_ab_beta': beta_pw,
            'delta_canonic_Band':canonic_bands['delta'],
            'theta_canonic_Band':canonic_bands['theta'],
            'alpha1_canonic_Band':canonic_bands['alpha1'],
            'alpha2_canonic_Band':canonic_bands['alpha2'],
            'beta1_canonic_Band':canonic_bands['beta1'],
            'beta2_canonic_Band':canonic_bands['beta2'],
            'beta3_canonic_Band':canonic_bands['beta3'],
            'gamma_canonic_Band':canonic_bands['gamma'],
            'pw_canonic_delta': delta_rel_canonic,
            'pw_canonic_theta': theta_rel_canonic,
            'pw_canonic_alpha1': alpha1_rel_canonic,
            'pw_canonic_alpha2': alpha2_rel_canonic,
            'pw_canonic_beta1': beta1_rel_canonic,
            'pw_canonic_beta2': beta2_rel_canonic,
            'pw_canonic_beta3': beta3_rel_canonic,
            'pw_canonic_gamma': gamma_rel_canonic,
            'pw_ab_canonic_delta': delta_pw_canonic,
            'pw_ab_canonic_theta': theta_pw_canonic,
            'pw_ab_canonic_alpha1': alpha1_pw_canonic,
            'pw_ab_canonic_alpha2': alpha2_pw_canonic,
            'pw_ab_canonic_beta1': beta1_pw_canonic,
            'pw_ab_canonic_beta2': beta2_pw_canonic,
            'pw_ab_canonic_beta3': beta3_pw_canonic,
            'pw_ab_canonic_gamma': gamma_pw_canonic,
            
        })
        
        if plot:
            #if row["Sensors"] == 'P7' or row["Sensors"] == 'P8' or row["Sensors"] == 'O1' or row["Sensors"] == 'O2':    
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, psd, label='PSD')
            plt.axvline(TF, color='r', linestyle='--', label=f'TF ({TF:.2f} Hz)')
            plt.axvline(IAFp, color='g', linestyle='--', label=f'IAFp ({IAFp:.2f} Hz)')
            # Añadir texto con los valores de TF e IAF en el gráfico
            plt.text(TF, max(psd) * 0.8, f'TF\n{TF:.2f} Hz', color='red', fontsize=9, ha='right', va='center')
            plt.text(IAFp, max(psd) * 0.6, f'IAFp\n{IAFp:.2f} Hz', color='green', fontsize=9, ha='right', va='center')

            plt.fill_between(freqs, psd, where=(freqs >= delta_band[0]) & (freqs < delta_band[1]), color='blue', alpha=0.3, label='delta Band')
            plt.fill_between(freqs, psd, where=(freqs >= theta_band[0]) & (freqs < theta_band[1]), color='purple', alpha=0.3, label='theta Band')
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
                path_save = f'{path}/psd_graphs/{row["SITE"]}/{row["Sensors"]}'
            else:
                path_save = f'{path}/psd_graphs/{row["SITE"]}/TF_below_4/{row["Sensors"]}'
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            subject_safe = re.sub(r'[\\/*?:"<>|]', '_', row['subject'])
            sensor_safe = re.sub(r'[\\/*?:"<>|]', '_', row['Sensors'])
            path_save = os.path.normpath(path_save)  # Normalizar la ruta para el sistema operativo
            filename = f'{subject_safe}_{sensor_safe}.png'

            plt.savefig(os.path.join(path_save, filename))
            plt.close()
  
            
    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_feather(rf'{path}\IAF_{row["SITE"]}_scorEpochs_zscore_long.feather')
    print(f'Done {row["SITE"]}!')
    return results_df


def report_IAF_by_sensor(datasets = dict, path_report = str):
    # Listas para almacenar los informes de cada tipo
    general_reports = []
    controls_reports = []
    patients_reports = []

    # Definir los grupos de interés
    controls_groups = ['HC', 'GU', 'G2', 'CTR']
    patients_groups = ['MCI', 'DCL']

    # Iterar por cada base de datos
    for name, data in datasets.items():
        channels = ['FP1','FP2','C3','C4','P7','P8','O1','O2']
        # Filtrar solo los sensores de interés
        filtered_data = data[data['Sensors'].isin(channels)]

        # Informe General: contar los sujetos por grupo y sensor
        initial_subjects = filtered_data.groupby('Sensors')['subject'].nunique()

        # Filtrar eliminados por condiciones
        removed_tf_low = filtered_data[filtered_data['TF'] < 4]
        removed_tf_high = filtered_data[filtered_data['TF'] > filtered_data['IAFp']]
        remaining_data = filtered_data[(filtered_data['TF'] >= 4) & (filtered_data['TF'] <= filtered_data['IAFp'])]

        # Calcular sujetos únicos que cumplen TF < 4 o TF > IAFp (sin repeticiones)
        tf_low_or_high = filtered_data[
            (filtered_data['TF'] < 4) | (filtered_data['TF'] > filtered_data['IAFp'])
        ]
        unique_tf_low_or_high = tf_low_or_high.groupby('Sensors')['subject'].nunique()

        # Crear resumen para la hoja "Informe General"
        general_report = pd.DataFrame({
            "SITE": name,
            "Sensor": channels,
            "Subjects": [
                initial_subjects.get('FP1', 0),
                initial_subjects.get('FP2', 0),
                initial_subjects.get('C3', 0),
                initial_subjects.get('C4', 0),
                initial_subjects.get('P7', 0),
                initial_subjects.get('P8', 0),
                initial_subjects.get('O1', 0),
                initial_subjects.get('O2', 0)
            ],
            "TF < 4": [
                removed_tf_low[removed_tf_low['Sensors'] == 'FP1']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'FP2']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'C3']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'C4']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'P7']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'P8']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'O1']['subject'].nunique(),
                removed_tf_low[removed_tf_low['Sensors'] == 'O2']['subject'].nunique()
            ],
            "TF > IAFp": [
                removed_tf_high[removed_tf_high['Sensors'] == 'FP1']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'FP2']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'C3']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'C4']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'P7']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'P8']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'O1']['subject'].nunique(),
                removed_tf_high[removed_tf_high['Sensors'] == 'O2']['subject'].nunique()
            ],
            "TF < 4 o TF > IAPf": [
                unique_tf_low_or_high.get('FP1', 0),
                unique_tf_low_or_high.get('FP2', 0),
                unique_tf_low_or_high.get('C3', 0),
                unique_tf_low_or_high.get('C4', 0),
                unique_tf_low_or_high.get('P7', 0),
                unique_tf_low_or_high.get('P8', 0),
                unique_tf_low_or_high.get('O1', 0),
                unique_tf_low_or_high.get('O2', 0)
            ],
            "remaining_subject_by_sensor": [
                remaining_data[remaining_data['Sensors'] == 'FP1']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'FP2']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'C3']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'C4']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'P7']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'P8']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'O1']['subject'].nunique(),
                remaining_data[remaining_data['Sensors'] == 'O2']['subject'].nunique()
            ]
        })

        # Agregar el informe general a la lista
        general_reports.append(general_report)

        # Informe de Controles (sanos): Solo los sujetos de los grupos controles
        controls_data = filtered_data[filtered_data['group'].isin(controls_groups)]
        
        initial_subjects_controls = controls_data.groupby('Sensors')['subject'].nunique()
        removed_tf_low_controls = controls_data[controls_data['TF'] < 4]
        removed_tf_high_controls = controls_data[controls_data['TF'] > controls_data['IAFp']]
        remaining_data_controls = controls_data[(controls_data['TF'] >= 4) & (controls_data['TF'] <= controls_data['IAFp'])]
        
        tf_low_or_high_controls = controls_data[
            (controls_data['TF'] < 4) | (controls_data['TF'] > controls_data['IAFp'])
        ]
        unique_tf_low_or_high_controls = tf_low_or_high_controls.groupby('Sensors')['subject'].nunique()

        # Crear resumen para la hoja "Informe de Controles"
        controls_report = pd.DataFrame({
            "SITE": name,
            "Sensor": channels,
            "Subjects": [
                initial_subjects_controls.get('FP1', 0),
                initial_subjects_controls.get('FP2', 0),
                initial_subjects_controls.get('C3', 0),
                initial_subjects_controls.get('C4', 0),
                initial_subjects_controls.get('P7', 0),
                initial_subjects_controls.get('P8', 0),
                initial_subjects_controls.get('O1', 0),
                initial_subjects_controls.get('O2', 0)
            ],
            "TF < 4": [
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'FP1']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'FP2']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'C3']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'C4']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'P7']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'P8']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'O1']['subject'].nunique(),
                removed_tf_low_controls[removed_tf_low_controls['Sensors'] == 'O2']['subject'].nunique()
            ],
            "TF > IAFp": [
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'FP1']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'FP2']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'C3']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'C4']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'P7']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'P8']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'O1']['subject'].nunique(),
                removed_tf_high_controls[removed_tf_high_controls['Sensors'] == 'O2']['subject'].nunique()
            ],
            "TF < 4 o TF > IAPf": [
                unique_tf_low_or_high_controls.get('FP1', 0),
                unique_tf_low_or_high_controls.get('FP2', 0),
                unique_tf_low_or_high_controls.get('C3', 0),
                unique_tf_low_or_high_controls.get('C4', 0),
                unique_tf_low_or_high_controls.get('P7', 0),
                unique_tf_low_or_high_controls.get('P8', 0),
                unique_tf_low_or_high_controls.get('O1', 0),
                unique_tf_low_or_high_controls.get('O2', 0)
            ],
            "remaining_subject_by_sensor": [
                remaining_data_controls[remaining_data_controls['Sensors'] == 'FP1']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'FP2']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'C3']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'C4']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'P7']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'P8']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'O1']['subject'].nunique(),
                remaining_data_controls[remaining_data_controls['Sensors'] == 'O2']['subject'].nunique()
            ]
        })

        controls_reports.append(controls_report)

        # Informe de Sujetos Enfermos: Solo los sujetos de los grupos MCI, DCL
        patients_data = filtered_data[filtered_data['group'].isin(patients_groups)]
        
        initial_subjects_patients = patients_data.groupby('Sensors')['subject'].nunique()
        removed_tf_low_patients = patients_data[patients_data['TF'] < 4]
        removed_tf_high_patients = patients_data[patients_data['TF'] > patients_data['IAFp']]
        remaining_data_patients = patients_data[(patients_data['TF'] >= 4) & (patients_data['TF'] <= patients_data['IAFp'])]
        
        tf_low_or_high_patients = patients_data[
            (patients_data['TF'] < 4) | (patients_data['TF'] > patients_data['IAFp'])
        ]
        unique_tf_low_or_high_patients = tf_low_or_high_patients.groupby('Sensors')['subject'].nunique()

        # Crear resumen para la hoja "Informe de Sujetos Enfermos"
        patients_report = pd.DataFrame({
            "SITE": name,
            "Sensor": channels,
            "Subjects": [
                initial_subjects_patients.get('FP1', 0),
                initial_subjects_patients.get('FP2', 0),
                initial_subjects_patients.get('C3', 0),
                initial_subjects_patients.get('C4', 0),
                initial_subjects_patients.get('P7', 0),
                initial_subjects_patients.get('P8', 0),
                initial_subjects_patients.get('O1', 0),
                initial_subjects_patients.get('O2', 0)
            ],
            "TF < 4": [
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'FP1']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'FP2']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'C3']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'C4']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'P7']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'P8']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'O1']['subject'].nunique(),
                removed_tf_low_patients[removed_tf_low_patients['Sensors'] == 'O2']['subject'].nunique()
            ],
            "TF > IAFp": [
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'FP1']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'FP2']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'C3']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'C4']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'P7']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'P8']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'O1']['subject'].nunique(),
                removed_tf_high_patients[removed_tf_high_patients['Sensors'] == 'O2']['subject'].nunique()
            ],
            "TF < 4 o TF > IAPf": [
                unique_tf_low_or_high_patients.get('FP1', 0),
                unique_tf_low_or_high_patients.get('FP2', 0),
                unique_tf_low_or_high_patients.get('C3', 0),
                unique_tf_low_or_high_patients.get('C4', 0),
                unique_tf_low_or_high_patients.get('P7', 0),
                unique_tf_low_or_high_patients.get('P8', 0),
                unique_tf_low_or_high_patients.get('O1', 0),
                unique_tf_low_or_high_patients.get('O2', 0)
            ],
            "remaining_subject_by_sensor": [
                remaining_data_patients[remaining_data_patients['Sensors'] == 'FP1']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'FP2']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'C3']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'C4']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'P7']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'P8']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'O1']['subject'].nunique(),
                remaining_data_patients[remaining_data_patients['Sensors'] == 'O2']['subject'].nunique()
            ]
        })

        patients_reports.append(patients_report)

    # Concatenar los informes de las diferentes bases de datos
    general_df = pd.concat(general_reports, ignore_index=True)
    controls_df = pd.concat(controls_reports, ignore_index=True)
    patients_df = pd.concat(patients_reports, ignore_index=True)

    # Guardar todos los informes en un archivo de Excel
    with pd.ExcelWriter(rf'{path_report}') as writer:
        general_df.to_excel(writer, sheet_name='General_report', index=False)
        controls_df.to_excel(writer, sheet_name='Controls_report', index=False)
        patients_df.to_excel(writer, sheet_name='MCI_report', index=False)

    print("¡Informe generado con éxito!")


def remove_subjects_by_criteria(data, channels):
    # Función para eliminar sujetos si al menos uno de sus 4 canales cumple los criterios
    subjects_to_remove = []
        # Iterar sobre cada sujeto
    
    for subject in data['subject'].unique():
        subject_data = data[data['subject'] == subject]
        
        # Revisar los 4 canales de interés (P7, P8, O1, O2)
        for electrode in ['FP1','FP2','C3','C4','P7', 'P8', 'O1', 'O2']:
            electrode_data = subject_data[subject_data['Sensors'] == electrode]
            
            condition = ((electrode_data['TF'] < 4) | (electrode_data['TF']>electrode_data['IAFp'] ))
            if condition.any():
                subjects_to_remove.append(subject)
                break  # Si el sujeto cumple con la condición, no revisar más canales
        
    # Eliminar los sujetos de los datos
    return data[~data['subject'].isin(subjects_to_remove)]

# Función para imprimir la cantidad de sujetos restantes por canal
def print_subjects_remaining(data, name):
    # Contar los sujetos restantes por canal
    remaining_by_channel = data.groupby('Sensors')['subject'].nunique()
    total_remaining = data['subject'].nunique()
    
    print(f"\nResultados para la base de datos {name}:")
    print(f"Total de sujetos restantes: {total_remaining}")
    channels = data.Sensors.unique()
    for Sensor in channels:
        print(f"Sujetos restantes en {Sensor}: {remaining_by_channel.get(Sensor, 0)}")

def append_or_merge_to_existing_sheet(output_file, new_data, sheet_name, merge_on):
    """
    Actualiza o agrega filas a una hoja existente de Excel con base en columnas clave.
    
    output_file : str
        Ruta del archivo Excel existente.
    new_data : DataFrame
        Nuevos datos a añadir o actualizar.
    sheet_name : str
        Nombre de la hoja en la que se hará la operación.
    merge_on : list
        Columnas clave para hacer el merge (e.g., ["Site", "Sensor"]).
    """
    try:
        # Cargar el archivo Excel existente
        book = load_workbook(output_file)
        
        if sheet_name in book.sheetnames:
            # Leer la hoja existente
            existing_data = pd.read_excel(output_file, sheet_name=sheet_name)
        else:
            # Si la hoja no existe, crear una nueva
            existing_data = pd.DataFrame()
        
        # Hacer el merge (outer merge para conservar todo)
        updated_data = pd.merge(
            existing_data,
            new_data,
            on=merge_on,
            how="outer"
        )
        
        # Escribir de vuelta al archivo Excel
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            updated_data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    except Exception as e:
        print(f"Error: {e}")

def IAF2columns(df,name_dataset ,path_save=None, path_demographics=None, variables_dem=None, by_sensor=False, drop_columns =None):
    # Pivotar los datos
    df = df[df['SITE'] == name_dataset]
    pivoted = df.pivot(
        index=["subject", "group", "SITE"], 
        columns="Sensors",          
        values=[col for col in df.columns if col.startswith("pw_") 
                or col == "freqs" or col== "psd" or col == "IAFp" or col == "TF" 
                or col.startswith("pw_ab") or col.startswith("pw_BGF") or col.startswith("pw_") 
                or col == 'CF_extalpha' or col == "exponent" or col == "offset" or col == "r2" or col == "error"
                or col.startswith("osc_pw")]
    )

    # Aplanar el índice de las columnas
    pivoted.columns = [f"{band}_{sensor}" for band, sensor in pivoted.columns]
    df_columns = pivoted.reset_index()
    # Cargar datos demográficos y fusionarlos con el DataFrame
    data_demographics = pd.read_excel(path_demographics)
    data_demographics.replace({'group': {'CTR': 'HC', 'GU': 'HC', 'G2': 'HC','DCL':'MCI','G1':'ACr','GG':'ACr'}}, inplace=True)
    data_demographics.drop(columns=drop_columns, inplace=True)
    data_demographics.reset_index(inplace=True)
    data_demographics.drop(columns=['index'], inplace=True)
    # df_columns['SITE'] = df_columns['SITE'].str.strip().str.lower()
    # data_demographics['SITE'] = data_demographics['SITE'].str.strip().str.lower()

    print(data_demographics.columns)
    print(df.columns)
    df_metric_column = pd.merge(df_columns, data_demographics, on=['subject', 'group','SITE'], how='outer')

    # Guardar el DataFrame si se proporciona un path
    if path_save is not None:
        df_metric_column.to_feather(fr'{path_save}\IAF_{name_dataset}_scoreEpochs_zscore_column.feather')

    # Función para obtener los datos por electrodo
    def get_IAF_data(electrode):
        metrics = [f'IAFp_{electrode}', f'TF_{electrode}', f'freqs_{electrode}', f'psd_{electrode}',
                   f'pw_delta_{electrode}', f'pw_theta_{electrode}', 
                   f'pw_alpha1_{electrode}', f'pw_alpha2_{electrode}', f'pw_alpha3_{electrode}']
        if variables_dem is not None:
            return df_metric_column[['subject', 'group', 'SITE', 'age'] + variables_dem + metrics]
        else:
            return df_metric_column[['subject', 'group', 'SITE', 'age']  + metrics]
      
    
    if by_sensor:
        return get_IAF_data('O1'), get_IAF_data('O2'), get_IAF_data('P7'), get_IAF_data('P8')
    else:
        return df_metric_column


def apply_pca_umap(data, 
                   harmonized_data, 
                   columns_to_use, 
                   columns_demographics,
                   directory, 
                   n_pca=10, 
                   n_neighbors=40, 
                   min_dist=0.4, 
                   filename=None, 
                   method='neuroharmonize',
                   aproach = 'ByChannel', 
                   family_features= str,
                   excel_writer=None, 
                   sheet_name=None,
                   pref_covars = str,
                   roi = str):
    """
    Aplica PCA y UMAP a los datos originales y armonizados, y guarda los resultados en archivos CSV.
    
    Parameters:
    - data: DataFrame con los datos originales.
    - harmonized_data: DataFrame con los datos armonizados.
    - columns_to_use: Lista de columnas a usar en el análisis.
    - columns_harmonaze: DataFrame con las columnas armonizadas.
    - directory: Ruta donde se guardarán los archivos CSV.
    - n_pca: Número de componentes principales a retener.
    - n_neighbors: Número de vecinos para UMAP.
    - min_dist: Distancia mínima para UMAP.
    - filename: Nombre base de los archivos generados.
    """

    # Asegurar que el directorio y subdirectorios existen
    subdirs = ["results_harmonize", f"results_harmonize/{method}/graphs/{aproach}", f"results_harmonize/{method}/statical_tests/{aproach}"]
    for subdir in subdirs:
        os.makedirs(os.path.join(directory, subdir), exist_ok=True)
    data_copy = data.copy()
    # Seleccionar columnas relevantes
    selected_data = data_copy.loc[:, columns_to_use]
    columns_harmonaze = [f'harm_{col}' for col in columns_to_use]
    selected_data_harmonized = harmonized_data.loc[:, columns_harmonaze]   
    y = data_copy['SITE']

    # Estandarizar datos
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(selected_data)

    # Aplicar PCA
    pca_unharmonized = PCA(n_components=n_pca)
    pca_result = pca_unharmonized.fit_transform(standardized_data)
    selected_data['pca-one'], selected_data['pca-two'] = pca_result[:, 0], pca_result[:, 1]
    explained_variance = np.cumsum(pca_unharmonized.explained_variance_ratio_)

    # Repetir el proceso para datos armonizados
    scaler_harmonized = StandardScaler()
    standardized_harmonized = scaler_harmonized.fit_transform(selected_data_harmonized)

    pca_harmonized = PCA(n_components=n_pca)
    harmonized_pca_result = pca_harmonized.fit_transform(standardized_harmonized)
    selected_data_harmonized['harmonized-pca-one'], selected_data_harmonized['harmonized-pca-two'] = harmonized_pca_result[:, 0], harmonized_pca_result[:, 1]
    explained_variance_harmonized = np.cumsum(pca_harmonized.explained_variance_ratio_)

    # Aplicar UMAP
    umap_model_unharmonized = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42, n_jobs=1)
    umap_result = umap_model_unharmonized.fit_transform(pca_result)

    umap_model_harmonized = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42,n_jobs=1)
    umap_harmonized_result = umap_model_harmonized.fit_transform(harmonized_pca_result)

    # Agregar UMAP al DataFrame
    selected_data['umap-one'], selected_data['umap-two'] = umap_result[:, 0], umap_result[:, 1]
    selected_data_harmonized['harmonized-umap-one'], selected_data_harmonized['harmonized-umap-two'] = umap_harmonized_result[:, 0], umap_harmonized_result[:, 1]


    unharmonized_data = pd.concat([
    data_copy.loc[:,columns_demographics].reset_index(drop=True),
    selected_data.reset_index(drop=True)
    ], axis=1)

    harmonized_data = pd.concat([
        data_copy.loc[:,columns_demographics].reset_index(drop=True),
        selected_data_harmonized.reset_index(drop=True)
    ], axis=1)

    
    if excel_writer and sheet_name:
        unharmonized_data.to_excel(excel_writer, sheet_name='unharmonize'+sheet_name, index=False)
        harmonized_data.to_excel(excel_writer, sheet_name='harmonize'+sheet_name, index=False)
    # Generar gráficos
    plot_pca(unharmonized_data, harmonized_data,directory, method, aproach, filename)
    plot_umap(unharmonized_data, harmonized_data, directory, method, aproach, filename)
    # Gráfico de varianza acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Original')
    plt.plot(range(1, len(explained_variance_harmonized) + 1), explained_variance_harmonized, marker='o', label='Harmonized')

    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Explained')
    plt.title('Cumulative Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(directory, "results_harmonize", method,"graphs", aproach, f"EXPV_{filename}.png"), dpi=300)
    plt.close()
    
    plot_distance(unharmonized_data, harmonized_data,pca=True, directory= directory, method=method, approach=aproach, filename=filename)
    plot_distance(unharmonized_data, harmonized_data,pca=False, directory= directory, method=method, approach=aproach, filename=filename)
    
    # # Análisis estadístico

    staticals_features(unharmonized_data, 
                       harmonized_data, 
                       columns_to_use,
                       directory, 
                       method,
                       excel_filename="tests.xlsx",
                       sheet_name="tests", 
                       family_features= family_features,
                       covars = pref_covars,
                       roi = roi)
  
    # # Correlaciones de Spearman
    plot_spearman_scatter_single_winner(
    unharmonized_data, harmonized_data, columns_to_use, 
    directory, method, 
    filename=filename,
    excel_filename="spearman_correlations.xlsx",
    sheet_name="spearman_data", 
    family_features = family_features, 
    covars= pref_covars
    )
    return unharmonized_data, harmonized_data



def harmonize_transform(data, selected_cols = list, electrode='O1', exclude_cols=None, smooth_terms=None, covariates=None, method='neuroharmonize'):
    if exclude_cols is None:
        exclude_cols = [f'r2_{electrode}', f'error_{electrode}', f'CF_extalpha_{electrode}',
                        f'exponent_{electrode}', f'offset_{electrode}']
    
    if smooth_terms is None:
        smooth_terms = ['age']

    final_cols = [col for col in selected_cols if col not in exclude_cols]
    
    # Extraer datos armonizables y no armonizables

    my_data = np.array(data.loc[:, final_cols])
    
    # Crear covariables
    covars = pd.DataFrame()
    for cov in covariates:
        covars[cov] = data[cov].copy()
    
    # Codificar variables categóricas
    for cov in covars.columns:
        if covars[cov].dtype == 'object':
            le = LabelEncoder()
            covars[cov] = le.fit_transform(covars[cov]).astype('float64')
    i
    elif method == 'recombat':
        print(covars)
        model = reComBat(parametric=True,    # use parametric or non-parametric empirical Bayes method. 
                                             # The parametric method is significantly faster, whereas the 
                                             # non-parametric method is more flexible.
                 model='ridge',        # The regression model to be used. 
                                             # In our experience pure ridge regression performs best for singular design matrices 
                                             # and pure linear regression is best for non=singular matrices.
                                             # Ver que pasa si la cambio por elastic_net
                 config={'alpha':1e-4},      # Optional arguments for the regression model. 
                                             # We tend to use a tiny regularisation parameter. 
                                             # This has also been cnfirmed by CV.
                 conv_criterion=1e-6,        # The convergence criterion for the empirical Bayes optimisation.
                                             # This value works well in practise.
                 max_iter=100000,              # The maximum number of iterations to stop if convergence is not reached.
                                             # This may also be useful for smaller convergence criteria.
                 n_jobs=1,                   # This parameter is only useful in non-parametric optimisation.
                                             # The non-parametric optimisation is very slow, but can be parallelised easily.
                                             # Set this to the number of CPUs on your machine for significant speed ups.
                 mean_only=False,            # Adjust the mean of your data only (not the variance).
                                             # This can be useful for single sample batches (where the variance is infinite)
                 optimize_params=True,       # If False no empirical Bayes optimisation is performed.
                 reference_batch=None,       # If a reference batch is present (e.g. a batch which is considered "batch effect free")
                                             # it can be set such that all data is adjuste dwith respect to this reference batch.
                 verbose=True                # Turn log messages on or off
        )
        X_harmonized = model.fit_transform(data.loc[:, final_cols], covars.SITE, X=covars.drop('SITE', axis=1)) # Ojo covars no tiene el sitio, asumo que el segundo parametro de la función es suficiente
        harmonized_data = pd.DataFrame(X_harmonized, columns=final_cols)
    # Renombrar columnas armonizadas con prefijo 'harm_f method == 'neuroharmonize':
        model, data_adj = harmonizationLearn(my_data, covars, smooth_terms=smooth_terms, eb=True)
        harmonized_data = pd.DataFrame(data_adj, columns=final_cols)'
    harmonized_data.columns = [f'harm_{col}' for col in harmonized_data.columns]
    return model, harmonized_data

