import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from models.BLR import BayesianLinearRegression
from models.model_metrics import compute_MSLL, explained_var
from scipy.stats import spearmanr

from sklearn.preprocessing import OneHotEncoder
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

def get_matched_hc(df, group_df, group_name):
    age_min, age_max = group_df['age'].min(), group_df['age'].max()
    sites = group_df['SITE'].unique()
    return df[
        (df['group'] == 'HC') &
        (df['age'] >= age_min) & (df['age'] <= age_max) &
        (df['SITE'].isin(sites))
    ].dropna(subset=[feature] + covset)



def compute_BLR_and_save_outputs(df, feature, covset, bases, length_scale, sigma2, alpha2, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Separar por grupo
    hc = df[df.group == 'HC'].dropna(subset=[feature] + covset)
    mci = df[df.group == 'MCI'].dropna(subset=[feature] + covset)
    ad = df[df.group == 'AD'].dropna(subset=[feature] + covset)
    PD = df[df.group == 'PD'].dropna(subset=[feature] + covset)
    vd = df[df.group == 'VD'].dropna(subset=[feature] + covset)
    ACr = df[df.group == 'ACr'].dropna(subset=[feature] + covset)
    ND = df[df['group'].isin(['AD', 'PD', 'VD'])].copy()  # Enfermedades neurodegenerativas
    ALL = df[df['group'].isin(['AD', 'PD', 'VD', 'MCI'])].copy()  # Todas las patologías
    
    #hc = hc[hc['SITE'].isin(['Seoul']) & (hc['age'] >= 50)]
    mci = mci[mci['SITE'].isin(['Seoul','Madrid']) ]#& (mci['age'] >= 50)]

    # Binning de edad para estratificación
    num_bins = 5
    age_bins = pd.qcut(hc['age'], q=num_bins, labels=False)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(hc, age_bins))

    hc_train = hc.iloc[train_idx].copy()
    hc_test = hc.iloc[test_idx].copy()

    # Filtrar el test set para que tenga edades comparables a MCI
    mci_age_range = (mci['age'].min()-5, mci['age'].max()+5)
    hc_test = hc_test[(hc_test['age'] >= mci_age_range[0])].copy()


    def prepare_inputs(subset):
        X = prepare_covariates(subset, covset)
        y = subset[feature].values
        return X, y

    X_train, y_train = prepare_inputs(hc_train)
    X_test, y_test = prepare_inputs(hc_test)
    X_mci, y_mci = prepare_inputs(mci)
    X_ad , y_ad = prepare_inputs(ad)
    X_pd , y_pd = prepare_inputs(PD)
    X_vd , y_vd = prepare_inputs(vd)
    X_acr, y_acr = prepare_inputs(ACr)
    X_nd, y_nd = prepare_inputs(ND)
    X_all, y_all = prepare_inputs(ALL)

    kernel_params = {"number_of_bases": int(bases), "length_scale": length_scale, "bases_sampling_method": "KMeans"}
    blr = BayesianLinearRegression(kernel_params, sigma2=sigma2, alpha2=alpha2)

    use_hetero = False#any([c for c in covset if c != 'age'])

    if use_hetero:
        Xv_train = prepare_covariates(hc_train, covset)
        blr.fit_with_heteroscedasticity_and_optimization(X_train, y_train, Xv_train)
    else:
        blr.fit(X_train, y_train)

    blr.optimize_hyperparameters()
    
    # ✅ Reentrenar con hiperparámetros óptimos (solo si quieres usarlos de verdad)
    blr = BayesianLinearRegression(kernel_params, sigma2=blr.sigma2, alpha2=blr.alpha2)

    if use_hetero:
        blr.fit_with_heteroscedasticity_and_optimization(X_train, y_train, Xv_train)
    else:
        blr.fit(X_train, y_train)


    def predict_and_store(X, y, subject_ids,site,group, label, age_vals, Xv=None):
        y_pred, y_std, _, ci_lower, ci_upper, pi_lower, pi_upper = blr.predict_with_samples(X, n_samples=500, Xv_test=Xv)
        total_var, aleatoric_var, epistemic_var = blr.get_variance_decomposition(X, Xv_test=Xv)
        total_var *= blr.y_std ** 2
        aleatoric_var *= blr.y_std ** 2

        # Calcular z-scores
        z_total = (y - y_pred) / np.sqrt(total_var)
        z_aleatoric = (y - y_pred) / np.sqrt(aleatoric_var)

        print(f"Z-score (total) mean: {z_total.mean():.3f}, std: {z_total.std():.3f}")
        print(f"Z-score (aleatoric) mean: {z_aleatoric.mean():.3f}, std: {z_aleatoric.std():.3f}")

        return pd.DataFrame({
            "subject": subject_ids,
            "Site": site,
            "group": group,
            "set": label,
            "x_age": age_vals,
            "y_true": y,
            "y_pred": y_pred,
            "y_std": y_std,
            "z_score": z_total,
            "ci_lower": np.nan if label == "train" else ci_lower,
            "ci_upper": np.nan if label == "train" else ci_upper,
            "pi_lower": np.nan if label == "train" else pi_lower,
            "pi_upper": np.nan if label == "train" else pi_upper
        })

    Xv_test = prepare_covariates(hc_test, covset) if use_hetero else None
    Xv_mci = prepare_covariates(mci, covset) if use_hetero else None
    Xv_ad = prepare_covariates(ad, covset) if use_hetero else None
    Xv_pd = prepare_covariates(PD, covset) if use_hetero else None
    Xv_vd = prepare_covariates(vd, covset) if use_hetero else None
    Xv_acr = prepare_covariates(ACr, covset) if use_hetero else None
    Xv_nd = prepare_covariates(ND, covset) if use_hetero else None
    Xv_all = prepare_covariates(ALL, covset) if use_hetero else None
    
    # Guardar predicciones
    df_train = predict_and_store(X_train, y_train, hc_train['subject'].values, hc_train['SITE'].values, "HC", "train",hc_train['age'].values, Xv=Xv_train if use_hetero else None)
    df_test = predict_and_store(X_test, y_test, hc_test['subject'].values, hc_test['SITE'].values,  "HC", "test", hc_test['age'].values, Xv=Xv_test)
    df_mci = predict_and_store(X_mci, y_mci, mci['subject'].values, mci['SITE'].values,"MCI", "mci", mci['age'].values, Xv=Xv_mci)
    df_ad = predict_and_store(X_ad, y_ad, ad['subject'].values, ad['SITE'].values,"AD", "AD", ad['age'].values, Xv=Xv_ad)
    df_pd = predict_and_store(X_pd, y_pd, PD['subject'].values, PD['SITE'].values,"PD", "PD", PD['age'].values, Xv=Xv_pd)
    df_vd = predict_and_store(X_vd, y_vd, vd['subject'].values, vd['SITE'].values,"VD", "VD", vd['age'].values, Xv=Xv_vd)
    df_nd = predict_and_store(X_nd, y_nd, ND['subject'].values, ND['SITE'].values,"ND", "ND", ND['age'].values, Xv=Xv_nd)
    df_all = predict_and_store(X_all, y_all, ALL['subject'].values, ALL['SITE'].values,"ALL", "ALL", ALL['age'].values, Xv=Xv_all)
    df_acr = predict_and_store(X_acr, y_acr, ACr['subject'].values, ACr['SITE'].values,"ACr", "ACr", ACr['age'].values, Xv=Xv_acr)
    # Guardar bandas centiles evaluadas en un rango de edad usando predict_centile_bands
    #x_range = np.linspace(df['age'].min(), df['age'].max(), 200).reshape(-1, 1)
    x_range = np.linspace(hc['age'].min(), mci['age'].max(), 200).reshape(-1, 1)

    if use_hetero:
        mean_covariates = Xv_train.mean(axis=0)
        Xv_range = np.tile(mean_covariates, (x_range.shape[0], 1))
        age_index = covset.index("age")
        Xv_range[:, age_index] = x_range.flatten()
    else:
        Xv_range = None

    centiles = blr.predict_centile_bands(x_range, Xv_test=Xv_range, percentiles=[1, 5, 25, 75, 95, 99], factor=0.5)
    mean = blr.median_prediction(x_range) * blr.y_std + blr.y_mean
    _, _, _, ci_lower, ci_upper, pi_lower, pi_upper = blr.predict_with_samples(x_range, Xv_test=Xv_range)
    

    df_range = pd.DataFrame({
        
        "x_age": x_range.flatten(),
        "y_mean": mean,
        "group": "HC",
        "set": "range",
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper
    })

    for p in [1, 5, 25, 75, 95, 99]:
        df_range[f"centile_{p}_low"] = centiles[p][0]
        df_range[f"centile_{p}_high"] = centiles[p][1]

    df_all = pd.concat([df_train, df_test, df_mci, df_ad,df_pd,df_vd,df_acr,df_nd, df_all, df_range], axis=0)

    train_mean = np.mean(y_train, keepdims=True)
    train_var = np.var(y_train, ddof=1, keepdims=True)
    total_var, _, _ = blr.get_variance_decomposition(X_test, Xv_test=Xv_test)
    
    total_var = total_var * (blr.y_std ** 2)  # <-- escalar a la varianza original

    msll_test = compute_MSLL(y_test, df_test.y_pred.values, total_var, train_mean, train_var)
    print(f"MSLL Test: {msll_test}")

    mll_train = compute_MSLL(y_train, df_train.y_pred.values, df_train.y_std.values)
    mll_test = compute_MSLL(y_test, df_test.y_pred.values, df_test.y_std.values)
    ev_train = explained_var(y_train, df_train.y_pred.values)
    ev_test = explained_var(y_test, df_test.y_pred.values)
    mse_train = np.mean((y_train - df_train.y_pred.values)**2)
    mse_test = np.mean((y_test - df_test.y_pred.values)**2)
    rho_train, pvalue_train = spearmanr(y_train, df_train.y_pred.values)
    rho_test,pvalue_test = spearmanr(y_test, df_test.y_pred.values)

    metrics = pd.DataFrame([{
        "band": feature,
        "msll_test": msll_test,
        "mll_train": mll_train,
        "mll_test": mll_test,
        "ev_train": ev_train,
        "ev_test": ev_test,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "rho_train": rho_train,
        'pvalue_train': pvalue_train,
        "rho_test": rho_test,
        'pvalue_test': pvalue_test,
    }])
    
    return df_all, metrics


from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

def classify_zscore_models(df_all_combined, sets_incluidos=['test', 'MCI']):
    from sklearn.utils import resample
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import label_binarize

    results = []

    model_dict = {
        "SVM": CalibratedClassifierCV(LinearSVC(random_state=42, class_weight='balanced'), cv=5),
        "LogReg": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "DT": DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }

    for (feat, band, roi), df_feat in df_all_combined.groupby(['feature', 'band', 'roi']):
        df_z = df_feat[df_feat['set'].isin(sets_incluidos)].dropna(subset=['z_score'])

        class_counts_original = df_z['group'].value_counts().to_dict()
        unique_classes = df_z['group'].unique()

        if len(unique_classes) < 2:
            print(f"⚠️ Skipping {feat}-{band}-{roi}: Not enough classes ({unique_classes})")
            continue

        min_class_size = df_z['group'].value_counts().min()
        balanced_df_list = []

        for cls in unique_classes:
            df_cls = df_z[df_z['group'] == cls]
            if len(df_cls) < 5:
                print(f"⚠️ Skipping {feat}-{band}-{roi}: class '{cls}' too small ({len(df_cls)})")
                break
            df_sampled = resample(df_cls, replace=False, n_samples=min_class_size, random_state=42)
            balanced_df_list.append(df_sampled)
        else:
            df_z = pd.concat(balanced_df_list)

            y_raw = df_z['group'].values
            class_map = {cls: i for i, cls in enumerate(np.unique(y_raw))}
            labels = np.array([class_map[g] for g in y_raw])
            scores = df_z['z_score'].values.reshape(-1, 1)
            n_classes = len(np.unique(labels))

            dataset_label = ', '.join(sorted(np.unique(y_raw)))

            for model_name, clf in model_dict.items():
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                aucs, accs, precisions, recalls, f1s = [], [], [], [], []

                for train_idx, test_idx in skf.split(scores, labels):
                    clf.fit(scores[train_idx], labels[train_idx])
                    preds = clf.predict(scores[test_idx])
                    probas = clf.predict_proba(scores[test_idx]) if hasattr(clf, "predict_proba") else None

                    if n_classes == 2 and probas is not None:
                        auc = roc_auc_score(labels[test_idx], probas[:, 1])
                    elif probas is not None:
                        y_test_bin = label_binarize(labels[test_idx], classes=np.arange(n_classes))
                        auc = roc_auc_score(y_test_bin, probas, multi_class='ovr', average='macro')
                    else:
                        auc = float('nan')

                    prec = precision_score(labels[test_idx], preds, average='macro', zero_division=0)
                    rec = recall_score(labels[test_idx], preds, average='macro', zero_division=0)
                    f1 = f1_score(labels[test_idx], preds, average='macro', zero_division=0)

                    aucs.append(auc)
                    accs.append(accuracy_score(labels[test_idx], preds))
                    precisions.append(prec)
                    recalls.append(rec)
                    f1s.append(f1)

                results.append({
                    "feature": feat,
                    "band": band,
                    "roi": roi,
                    "model": model_name,
                    "Set": dataset_label,
                    "AUC_mean": np.mean(aucs),
                    "AUC_std": np.std(aucs),
                    "accuracy": np.mean(accs),
                    "precision": np.mean(precisions),
                    "recall": np.mean(recalls),
                    "f1_score": np.mean(f1s),
                    "counts": class_counts_original
                })

    return pd.DataFrame(results)



   

bands = ['theta', 'alpha1','alpha2', 'beta1','beta2','beta3', 'gamma']
#bands = ['theta', 'BGF1', 'BGF2', 'BGF3', 'beta']

rois = ['F', 'C', 'P', 'O', 'PO']
bases = 30            
length_scale = 1.5      
sigma2 = 0.06        
alpha2 = 5            

covset= ['age']

outlier_log = []
save_path = r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\recombat\BLR\bands_age\harmonized"
for family in ['osc_pw_rel_canonic', 'pw_rel_canonic', 'osc_pw_ab_canonic', 'pw_ab_canonic']:
    df_all_list = []
    metrics_list = []
    if family == 'osc_pw_rel_canonic' or family == 'osc_pw_ab_canonic':
        non_band_features = ['exponent']
    else:
        non_band_features = ['IAFp']
    file_name = f'{family}_roi'
    for roi in rois:
        path = fr"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\recombat\{file_name}{roi}_SITE_age_group_recombat.xlsx"
        data_roi = pd.read_excel(path, sheet_name="harmonizeSITE_age_group")
        #harmonize
        # Features con banda
        for band in bands:
            if family == 'osc_pw_rel_canonic':
                family  = 'osc_pw_canonic'
            elif family == 'pw_rel_canonic':
                family  = 'pw_canonic'
            feature = f'harm_{family}_{band}_roi{roi}'
            is_hc = (data_roi['group'] == 'HC')
            total_hc = data_roi[is_hc][feature].notna().sum()
            hc_values = data_roi.loc[is_hc, feature].dropna()
            q1 = hc_values.quantile(0.25)
            q3 = hc_values.quantile(0.75)
            iqr_val = q3 - q1
            lower = q1 - 1.5 * iqr_val
            upper = q3 + 1.5 * iqr_val
            keep_hc = data_roi[feature].between(lower, upper)
            keep_mci = data_roi['group'] != 'HC'
            data_filtered = data_roi[keep_hc | keep_mci].copy()
            remaining_hc = data_filtered[(data_filtered['group'] == 'HC') & (data_filtered[feature].notna())].shape[0]
            n_removed = total_hc - remaining_hc

            # Guardar log
            outlier_log.append({
                "feature": feature,
                "roi": roi,
                "HC_before": total_hc,
                "HC_after": remaining_hc,
                "n_removed": n_removed,
                "pct_removed": round(100 * n_removed / total_hc, 2) if total_hc > 0 else 0
            })

            
            df_all, metrics = compute_BLR_and_save_outputs(
                data_filtered, feature, covset, bases, length_scale, sigma2, alpha2, save_path
            )
            df_all["feature"] = family
            df_all["band"] = band
            df_all["roi"] = roi
            metrics["feature"] = family
            metrics["band"] = band
            metrics["roi"] = roi
            df_all_list.append(df_all)
            metrics_list.append(metrics)

        # Features sin banda (como IAFp)
        from scipy.stats import iqr
        

        for f in non_band_features:
            feature = f"harm_{f}_roi{roi}"

            # Sujetos HC originales
            is_hc = (data_roi['group'] == 'HC')
            total_hc = data_roi[is_hc][feature].notna().sum()

            # Cálculo de IQR para el filtro
            hc_values = data_roi.loc[is_hc, feature].dropna()
            q1 = hc_values.quantile(0.25)
            q3 = hc_values.quantile(0.75)
            iqr_val = q3 - q1
            lower = q1 - 1.5 * iqr_val
            upper = q3 + 1.5 * iqr_val

            # Index para mantener
            keep_hc = data_roi[feature].between(lower, upper)
            keep_mci = data_roi['group'] != 'HC'
            data_filtered = data_roi[keep_hc | keep_mci].copy()

            # Sujetos HC luego del filtro
            remaining_hc = data_filtered[(data_filtered['group'] == 'HC') & (data_filtered[feature].notna())].shape[0]
            n_removed = total_hc - remaining_hc

            # Guardar log
            outlier_log.append({
                "feature": f,
                "roi": roi,
                "HC_before": total_hc,
                "HC_after": remaining_hc,
                "n_removed": n_removed,
                "pct_removed": round(100 * n_removed / total_hc, 2) if total_hc > 0 else 0
            })

            # ENTRENAMIENTO como antes
            df_all, metrics = compute_BLR_and_save_outputs(
                data_filtered, feature, covset, bases, length_scale, sigma2, alpha2, save_path
            )
            df_all["feature"] = f
            df_all["roi"] = roi
            metrics["roi"] = roi
            df_all_list.append(df_all)
            metrics_list.append(metrics)


    df_all_combined = pd.concat(df_all_list, ignore_index=True)
    metrics_combined = pd.concat(metrics_list, ignore_index=True)
    df_all_combined.to_csv(os.path.join(save_path, f"blr_{family}.csv"), index=False)
    metrics_combined.to_csv(os.path.join(save_path, f"metrics_{family}.csv"), index=False)


    # print(family)
    ml_mci = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'mci'])
    ml_mci.to_csv(os.path.join(save_path, f"SVM_{family}_MCI.csv"), index=False)
    print('HC_MCI',ml_mci.AUC_mean.max())
    
    ml_ad = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'AD'])
    ml_ad.to_csv(os.path.join(save_path, f"SVM_{family}_AD.csv"), index=False)
    print('HC_AD',ml_ad.AUC_mean.max())
    
    ml_pd = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'PD'])
    ml_pd.to_csv(os.path.join(save_path, f"SVM_{family}_PD.csv"), index=False)
    print('HC_PD',ml_pd.AUC_mean.max())
    
    ml_vd = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'VD'])
    ml_vd.to_csv(os.path.join(save_path, f"SVM_{family}_VD.csv"), index=False)
    print('HC_VD',ml_vd.AUC_mean.max())
    
    ml_ACr = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'ACr'])
    ml_ACr.to_csv(os.path.join(save_path, f"SVM_{family}_ACr.csv"), index=False)
    print('HC_ACr',ml_ACr.AUC_mean.max())

    ml_mci_ad_pd_vd = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'ALL'])
    print('ALL',ml_mci_ad_pd_vd.AUC_mean.max())
    ml_mci_ad_pd_vd.to_csv(os.path.join(save_path, f"SVM_{family}_MCI_AD_PD_VD.csv"), index=False)
    
    ml_vd_ad_pd = classify_zscore_models(df_all_combined, sets_incluidos = ['test', 'ND'])
    print('ND',ml_vd_ad_pd.AUC_mean.max())
    ml_vd_ad_pd.to_csv(os.path.join(save_path, f"SVM_{family}_HC_VD_AD_PD.csv"), index=False)
    
    
    