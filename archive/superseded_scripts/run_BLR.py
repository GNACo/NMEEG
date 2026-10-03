import numpy as np
import math
import pandas as pd
from matplotlib import  pyplot as plt
from typing import Tuple 
from sklearn.cluster import KMeans 
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
import os
from models.BLR import BayesianLinearRegression
from models.model_metrics import compute_MSLL, explained_var
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score, RocCurveDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

log_psd_issues = []

def log_psd_failure(feature, file_key, tag, min_eigenvalue):
    log_psd_issues.append({
        "file": file_key,
        "feature": feature,
        "set": tag,
        "min_eigenvalue": min_eigenvalue,
        "issue": "V_n not positive semidefinite"
    })

def save_psd_log_if_needed(output_path):
    if log_psd_issues:
        log_path = os.path.join(output_path, "Vn_psd_issues_log.csv")
        pd.DataFrame(log_psd_issues).to_csv(log_path, index=False)
        return log_path
    return None

def prepare_covariates(df, covariates):
    df = df.copy()
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    cols = []
    if 'SITE' in covariates:
        site_encoded = encoder.fit_transform(df[['SITE']])
        site_df = pd.DataFrame(site_encoded, columns=encoder.get_feature_names_out(['SITE']))
        df = pd.concat([df.reset_index(drop=True), site_df.reset_index(drop=True)], axis=1)
    if 'age' in covariates:
        cols.append('age')
    if 'sex' in covariates:
        df['sex'] = df['sex'].map({'M': 0, 'F': 1})
        cols.append('sex')
    if 'SITE' in covariates:
        cols.extend(site_df.columns.tolist())
    return df[cols].values

def compute_BLR_and_project_MCI_compare(
                                        df_unharm, 
                                        df_harm,
                                        covset,
                                        feature_list,
                                        bases=int, 
                                        lenght_scale =int,
                                        sigma2 = int,
                                        alpha2=int,
                                        save_path=None):

    all_records = []
    negative_log = []

    for feat in feature_list:
        print(f'BLR modeling:{feat}')
        if (df_harm[feat] < 0).any():
            neg_subjects = df_harm.loc[df_harm[feat] < 0, 'subject'].tolist() if 'subject' in df_harm.columns else []
            negative_log.append({'feature': feat, 'n_removed': len(neg_subjects), 'subjects': neg_subjects})

        # Si hay heterocedasticidad y el sexo es una covariable, usar 5 columnas
        if 'sex' in covset:
            fig_main, axes = plt.subplots(2, 4, figsize=(18, 10))  # 2 filas, 5 columnas
        else:
            fig_main, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 filas, 3 columnas

        fig_roc, ax_roc = plt.subplots(2, 2, figsize=(12, 10))

        for row_idx, (df, tag) in enumerate(zip([df_unharm, df_harm], ["unharmonized", "harmonized"])):
            # Añade al inicio del bucle por feature:
            age_index = covset.index('age')

            data = df.copy()
            data = data[covset + ['group', feat]].dropna()
            hc = data[data.group == 'HC']
            mci = data[data.group == 'MCI']

            X_full = prepare_covariates(hc, covset) #hc.age.values.reshape(-1, 1)
            y_full = hc[feat].values
            X_mci = prepare_covariates(mci, covset) #mci.age.values.reshape(-1, 1)
            y_mci = mci[feat].values

            # prepare data
            X_train_idx, X_test_idx = train_test_split(hc.index, test_size=0.3, random_state=42)

            # 3. Usamos esos índices para extraer datos y mantener tus nombres
    
            X_train = prepare_covariates(hc.loc[X_train_idx], covset)
            X_test = prepare_covariates(hc.loc[X_test_idx], covset)

            y_train = hc.loc[X_train_idx, feat].values
            y_test = hc.loc[X_test_idx, feat].values

            # Ahora el 100% de MCI se usa solo en el test
            X_mci = prepare_covariates(mci, covset)
            y_mci = mci[feat].values

            # 4. Esto es lo que te interesa para la heterocedasticidad
            
            kernel_params = {"number_of_bases": bases, "length_scale": lenght_scale, "bases_sampling_method": "KMeans"}
            blr = BayesianLinearRegression(kernel_params, sigma2=sigma2, alpha2=alpha2)
            use_hetero = any([c for c in covset if c != 'age'])
            # Primer entrenamiento preliminar
            
            if use_hetero:
                Xv_train = prepare_covariates(hc.loc[X_train_idx], covset)
                Xv_test = prepare_covariates(hc.loc[X_test_idx], covset)
                Xv_mci = prepare_covariates(mci, covset)
                try:
                    blr.set_log_context(feature=feat, file_key=key, tag=tag, logger=log_psd_failure)
                    blr.fit_with_heteroscedasticity_and_optimization(X_train, y_train, Xv_train)
                except RuntimeError as e:
                    print(f"Error en feature: {feat} ({tag})")
                    raise e

            else:
                Xv_train = None
                Xv_test = None
                Xv_mci = None
                blr.fit(X_train, y_train)

            blr.optimize_hyperparameters()
            
            
            x_age = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 200).reshape(-1, 1)

            # Reentrenamiento después de optimizar
            if use_hetero:
                blr.fit_with_heteroscedasticity_and_optimization(X_train, y_train, Xv_train)
            else:
                blr.fit(X_train, y_train)

            
            if use_hetero and 'sex' in covset:
                sex_index = covset.index('sex')
                mean_covariates = np.mean(Xv_train, axis=0)

                x_range_list = []
                Xv_range_list = []
                sex_labels = ['Male (0)', 'Female (1)']

                for sex_value in [0, 1]:  
                    x_covariates = np.tile(mean_covariates, (x_age.shape[0], 1))
                    x_covariates[:, age_index] = x_age.flatten() 
                    x_covariates[:, sex_index] = sex_value
                    x_range_list.append(x_covariates)
                    Xv_range_list.append(x_covariates)
            else:
                x_range_list = [x_age]
                Xv_range_list = [None]
                sex_labels = ['All']

            for x_range, Xv_range, sex_label in zip(x_range_list, Xv_range_list, sex_labels):
                x_plot = x_range[:, age_index] if x_range.ndim > 1 else x_range
                y_mean, y_std, y_samples, ci_low, ci_up, pi_low, pi_up = blr.predict_with_samples(x_range, Xv_test=Xv_range)

                # Training plot
                ax_train = axes[row_idx, 0]
                color = "purple" if sex_label == "Male (0)" else "red" if sex_label == "Female (1)" else "blue"
                if 'sex' in covset:
                    sex_index = covset.index('sex')
                    male_idx = X_train[:, sex_index] == 0
                    female_idx = X_train[:, sex_index] == 1
                    ax_train.scatter(X_train[male_idx, 0], y_train[male_idx], color="purple", s=10, label="Male (0)")
                    ax_train.scatter(X_train[female_idx, 0], y_train[female_idx], color="red", s=10, label="Female (1)")
                else:
                    ax_train.scatter(X_train[:, 0], y_train, color="purple", s=10, label="HC")

                for i in range(y_samples.shape[1]):
                    ax_train.plot(x_plot, y_samples[:, i], color=color, alpha=0.01, label=f"{sex_label}" if i == 0 else "")
                ax_train.plot(x_plot, y_mean, color=color, linewidth=2, label=f"Mean {sex_label}")
                ax_train.fill_between(x_plot, ci_low, ci_up, color="yellow", alpha=0.2)
                ax_train.fill_between(x_plot, pi_low, pi_up, color="coral", alpha=0.1)
                ax_train.set_xlabel("Age")
                ax_train.set_ylabel(feat)
                ax_train.legend(fontsize=6)

                # Predicciones promedio para Test y MCI
                # Verificar que los índices booleanos sean del tamaño adecuado para cada conjunto de datos
                male_idx = X_test[:, sex_index] == 0  # Índices de hombres en X_test
                female_idx = X_test[:, sex_index] == 1  # Índices de mujeres en X_test

                male_idx_mci = X_mci[:, sex_index] == 0  # Índices de hombres en MCI
                female_idx_mci = X_mci[:, sex_index] == 1  # Índices de mujeres en MCI

                # Verificar que las predicciones se calculen solo con los datos seleccionados
                Xv_test_male = Xv_test[male_idx] if use_hetero and Xv_test is not None else None
                Xv_test_female = Xv_test[female_idx] if use_hetero and Xv_test is not None else None

                pred_val_male, std_val_male, y_samples_male, ci_low_male, ci_up_male, pi_low_male, pi_up_male  = blr.predict_with_samples(X_test[male_idx], Xv_test=Xv_test_male)
                pred_val_female, std_val_female, y_samples_female, ci_low_female, ci_up_female, pi_low_female, pi_up_female  = blr.predict_with_samples(X_test[female_idx], Xv_test=Xv_test_female)

                # Predicciones para MCI por sexo
                pred_mci_male, std_mci_male, y_samples_mci_male, ci_low_mci_male, ci_up_mci_male, pi_low_mci_male, pi_up_mci_male = blr.predict_with_samples(X_mci[male_idx_mci], Xv_test=Xv_mci[male_idx_mci] if use_hetero else None)
                pred_mci_female, std_mci_female, y_samples_mci_female, ci_low_mci_female, ci_up_mci_female, pi_low_mci_female, pi_up_mci_female = blr.predict_with_samples(X_mci[female_idx_mci], Xv_test=Xv_mci[female_idx_mci] if use_hetero else None)


                # Graficar las predicciones de Test HC y MCI (por sexo)
                ax_cent = axes[row_idx, 1] if sex_label == "Male (0)" else axes[row_idx, 2]

                # Predicciones de Test y MCI
                ax_cent.scatter(X_test[male_idx, 0], y_test[male_idx], color="blue", s=10, label="Test HC Male")
                ax_cent.scatter(X_test[female_idx, 0], y_test[female_idx], color="red", s=10, label="Test HC Female")
                ax_cent.scatter(X_mci[male_idx_mci, 0], y_mci[male_idx_mci], color="gray", s=10, label="MCI Male")
                ax_cent.scatter(X_mci[female_idx_mci, 0], y_mci[female_idx_mci], color="green", s=10, label="MCI Female")

                # Graficar las bandas de centiles para el conjunto de Test
                bands_test_male = blr.predict_centile_bands(X_test[male_idx])
                bands_test_female = blr.predict_centile_bands(X_test[female_idx])

                for label, (lo, hi) in {"25-75": (25, 75), "5-95": (5, 95), "1-99": (1, 99)}.items():
                    lo_band, hi_band = bands_test_male[lo][0], bands_test_male[hi][1]
                    ax_cent.fill_between(X_test[male_idx, 0], lo_band, hi_band, alpha=0.1, color="blue")
                    ax_cent.plot(X_test[male_idx, 0], lo_band, color="blue", linewidth=0.5)
                    ax_cent.plot(X_test[male_idx, 0], hi_band, color="blue", linewidth=0.5)
                    ax_cent.text(X_test[male_idx, 0][-1], lo_band[-1], f"{lo}%", fontsize=7)
                    ax_cent.text(X_test[male_idx, 0][-1], hi_band[-1], f"{hi}%", fontsize=7)

                    lo_band, hi_band = bands_test_female[lo][0], bands_test_female[hi][1]
                    ax_cent.fill_between(X_test[female_idx, 0], lo_band, hi_band, alpha=0.1, color="red")
                    ax_cent.plot(X_test[female_idx, 0], lo_band, color="red", linewidth=0.5)
                    ax_cent.plot(X_test[female_idx, 0], hi_band, color="red", linewidth=0.5)
                    ax_cent.text(X_test[female_idx, 0][-1], lo_band[-1], f"{lo}%", fontsize=7)
                    ax_cent.text(X_test[female_idx, 0][-1], hi_band[-1], f"{hi}%", fontsize=7)

                # Predicción media de Test para hombres y mujeres
                ax_cent.plot(X_test[male_idx, 0], pred_val_male, color="blue", linewidth=2, label="Mean Test HC Male")
                ax_cent.plot(X_test[female_idx, 0], pred_val_female, color="red", linewidth=2, label="Mean Test HC Female")

                # Predicciones de MCI para hombres y mujeres
                ax_cent.plot(X_mci[male_idx_mci, 0], pred_mci_male, color="gray", linewidth=2, label="Mean MCI Male")
                ax_cent.plot(X_mci[female_idx_mci, 0], pred_mci_female, color="green", linewidth=2, label="Mean MCI Female")

                ax_cent.set_xlabel("Age")
                ax_cent.set_ylabel(feat)
                ax_cent.set_title(f"{sex_label} Centiles")
                ax_cent.legend(fontsize=6)

            # Calcular Z-scores por sexo
            Z_val_male = (y_test[male_idx] - pred_val_male) / std_val_male
            Z_val_female = (y_test[female_idx] - pred_val_female) / std_val_female
            Z_mci_male = (y_mci[male_idx_mci] - pred_mci_male) / std_mci_male
            Z_mci_female = (y_mci[female_idx_mci] - pred_mci_female) / std_mci_female
            
            
            # Actualizar el gráfico para los Z-scores
            ax_z = axes[row_idx, 3]
            ax_z.hist(Z_val_male, bins=20, color="purple", alpha=0.3, label="HC Male")
            ax_z.hist(Z_val_female, bins=20, color="red", alpha=0.3, label="HC Female")
            ax_z.hist(Z_mci_male, bins=20, color="blue", alpha=0.3, label="MCI Male")
            ax_z.hist(Z_mci_female, bins=20, color="pink", alpha=0.3, label="MCI Female")
            ax_z.axvline(0, color="black", linestyle="--")
            ax_z.set_title("Z-score comparison: HC vs MCI by Sex")
            ax_z.set_xlabel("Z-score")
            ax_z.set_ylabel("Count")
            ax_z.legend()
            

               
                


            base_record = {
                'feature': feat,
                'set': tag,
                'roi':feat.split('_')[-1],
                'band': feat.split('_')[-2] if feat.split('_')[-2] in ['theta', 'BGF1', 'BGF2', 'BGF3', 'gamma', 'beta','alpha1','alpha2','beta1','beta2','beta3'] else None,
                'covariates': '_'.join(covset),
                'n_train': len(X_train), 'n_test': len(X_test), 'n_val': len(X_val), 'n_mci': len(X_mci)
            }

            # all_records.append({**base_record, 'model': 'blr',
            #                     'EV_train': ev_train, 'EV_test': ev_test,
            #                     'mse_train': mse_train, 'mse_test': mse_test,
            #                     'msll_train': msll_train, 'msll_test': msll_test,
            #                     'spearman_train': spearman_train, 'pvalue_train': pvalue_train,
            #                     'spearman_test': spearman_test, 'pvalue_test': pvalue_test})

            # labels = np.array([0]*len(Z_val) + [1]*len(Z_mci))
            # scores = np.concatenate([Z_val, Z_mci])

            for model_type, clf, col_idx in zip(["logreg", "svm"], [LogisticRegression(), SVC(probability=True)], [0, 1]):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                tprs, aucs = [], []
                mean_fpr = np.linspace(0, 1, 100)
                ax_model = ax_roc[row_idx, col_idx]
                y_true, y_pred = [], []

                for i, (train_idx, test_idx) in enumerate(skf.split(scores.reshape(-1, 1), labels)):
                    clf.fit(scores[train_idx].reshape(-1, 1), labels[train_idx])
                    probas_ = clf.predict_proba(scores[test_idx].reshape(-1, 1))[:, 1]
                    preds = clf.predict(scores[test_idx].reshape(-1, 1))
                    y_true.extend(labels[test_idx])
                    y_pred.extend(preds)

                    fpr, tpr, _ = roc_curve(labels[test_idx], probas_)
                    roc_auc = roc_auc_score(labels[test_idx], probas_)
                    ax_model.plot(fpr, tpr, lw=1, alpha=0.5, label=f"ROC fold {i+1} (AUC={roc_auc:.2f})")
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(roc_auc)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                ax_model.plot(mean_fpr, mean_tpr, color='black', label=f"Mean ROC (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f})", lw=2)
                ax_model.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax_model.set_title(f"{model_type.upper()} - {tag}")
                ax_model.set_xlabel("FPR")
                ax_model.set_ylabel("TPR")
                ax_model.legend(loc="lower right")

                all_records.append({**base_record, 'model_clasification': model_type,
                                    'EV_train': np.nan, 'EV_test': np.nan,
                                    'mse_train': np.nan, 'mse_test': np.nan,
                                    'msll_train': np.nan, 'msll_test': np.nan,
                                    'spearman_train': np.nan, 'pvalue_train': np.nan,
                                    'spearman_test': np.nan, 'pvalue_test': np.nan,
                                    'auc': np.mean(aucs),
                                    'accuracy': accuracy_score(y_true, y_pred),
                                    'f1_score': f1_score(y_true, y_pred),
                                    'recall': recall_score(y_true, y_pred),
                                    'precision': precision_score(y_true, y_pred, zero_division=0)})

        if save_path:
            fig_main.savefig(os.path.join(save_path, f"BLR_{feat}_{'_'.join(covset)}.png"))
            fig_roc.savefig(os.path.join(save_path, f"ROC_{feat}_{'_'.join(covset)}.png"))
            plt.close(fig_main)
            plt.close(fig_roc)

    if save_path and negative_log:
        log_path = os.path.join(save_path, "negative_subjects_log.csv")
        pd.DataFrame(negative_log).to_csv(log_path, index=False)

    return pd.DataFrame(all_records)


import os
import glob
import pandas as pd

# Ruta principal

base_path = r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\neuroharmonize"
output_path = os.path.join(base_path, "BLR")
os.makedirs(output_path, exist_ok=True)

# CSV de resultados parciales
results_csv = os.path.join(output_path, "BLR_comparison_results.csv")

# Cargar resultados previos si existen
if os.path.exists(results_csv):
    existing_results = pd.read_csv(results_csv)
    completed_keys = set(existing_results["pair"].unique())
else:
    existing_results = pd.DataFrame()
    completed_keys = set()

# Buscar archivos
files = glob.glob(os.path.join(base_path, "*.xlsx"))

# Mapear archivos por clave

# Bucle principal con control de errores y guardado incremental
for key in files:

    if key in completed_keys:
        print(f"Saltando {key}, ya procesado.")
        continue

    try:
        print(f"Procesando {key}...")
        if "SITE_age_neuroharmonize" in key:
            df_unharm = pd.read_excel(key, sheet_name='unharmonizeSITE_age')
            df_harm = pd.read_excel(key, sheet_name='harmonizeSITE_age')
            cov =  ['age']
        elif "SITE_age_sex" in key: 
            df_unharm = pd.read_excel(key, sheet_name='unharmonizeSITE_age_sex')
            df_harm = pd.read_excel(key, sheet_name='harmonizeSITE_age_sex')
            cov = ['age', 'sex']
        # elif "SITE_sex" in key:
        #     df_unharm = pd.read_excel(key, sheet_name="unharmonizeSITE_sex")
        #     df_harm = pd.read_excel(key, sheet_name="harmonizeSITE_sex")
        #     cov = ['SITE', 'sex']
       
        # df_unharm = df_unharm[df_unharm['SITE'] != 'seoul']
        # df_harm = df_harm[df_harm['SITE'] != 'seoul']

        # Renombrar columnas armonizadas (remueve el prefijo harm_)
        df_harm = df_harm.rename(columns={col: col.replace("harm_", "") for col in df_harm.columns if col.startswith("harm_")})

        # Features válidas
        exclude_cols = ['subject', 'group', 'SITE', 'age','sex', 'pca-one', 'pca-two', 'umap-one', 'umap-two']
        exclude_cols += [col for col in df_unharm.columns if col.startswith('offset_')]

        features = [f for f in df_unharm.columns if f not in exclude_cols and not f.startswith("pca") and not f.startswith("umap")]

        # Subcarpeta de salida
        pair_output_path = os.path.join(output_path)
        os.makedirs(pair_output_path, exist_ok=True)
        # Revisar y eliminar sujetos con valores negativos en features específicas de df_harm
        negative_mask = (df_harm[features] < 0).any(axis=1)
        if negative_mask.any():
            print(f"Se eliminarán {negative_mask.sum()} sujetos con valores negativos en {key}.")
            if 'subject' in df_harm.columns:
                removed_subjects = df_harm.loc[negative_mask, 'subject'].tolist()
                print(f"Sujetos eliminados en {key}: {removed_subjects}")
            df_harm = df_harm[~negative_mask].reset_index(drop=True)
            df_unharm = df_unharm[~negative_mask].reset_index(drop=True)


        # Ejecutar análisis
        bases = 15
        length_scale = 1.2
        sigma2 = 0.1
        alpha2 = 5

        results = compute_BLR_and_project_MCI_compare(df_unharm, 
                                                      df_harm,
                                                      cov, 
                                                      features, 
                                                      save_path=pair_output_path, 
                                                      bases=bases, 
                                                      lenght_scale =length_scale,
                                                      sigma2 = sigma2,
                                                      alpha2=alpha2)
        results["pair"] = key

        # Guardar resultados incrementales
        updated_results = pd.concat([existing_results, results], ignore_index=True)
        updated_results.to_csv(results_csv, index=False)
        existing_results = updated_results
        print(f"✅ {key} procesado y guardado.")

    except Exception as e:
        print(f"❌ Error procesando {key}: {str(e)}. Continuando con el siguiente.")


psd_log_file = save_psd_log_if_needed(output_path)
if psd_log_file:
    print(f"⚠️ Problemas detectados con V_n. Revisa: {psd_log_file}")


print("🎉 Comparación completa. Resultados acumulados en:", results_csv)
