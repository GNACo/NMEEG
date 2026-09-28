import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from models.BLR import BayesianLinearRegression
from models.model_metrics import compute_MSLL, explained_var
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

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

from scipy.interpolate import interp1d

def build_centile_interpolators(df_range, percentiles=[1, 5, 25, 50, 75, 95, 99]):
    interpolators = {}
    ages = df_range["x_age"].values

    for p in percentiles:
        vals = df_range[f"centile_{p}"].values
        interpolators[p] = interp1d(
            ages, vals, bounds_error=False, fill_value="extrapolate"
        )

    return interpolators


def locate_percentile_bin(y, age, interpolators):
    p1 = float(interpolators[1](age))
    p5 = float(interpolators[5](age))
    p25 = float(interpolators[25](age))
    p50 = float(interpolators[50](age))
    p75 = float(interpolators[75](age))
    p95 = float(interpolators[95](age))
    p99 = float(interpolators[99](age))

    if y < p1:
        bin_label = "<1"
    elif y < p5:
        bin_label = "1-5"
    elif y < p25:
        bin_label = "5-25"
    elif y <= p75:
        bin_label = "25-75"
    elif y <= p95:
        bin_label = "75-95"
    elif y <= p99:
        bin_label = "95-99"
    else:
        bin_label = ">99"

    return pd.Series({
        "percentile_bin": bin_label,
        "p1": p1,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "delta_vs_p50": y - p50
    })
def add_percentile_info(df_subset, interpolators):
    if df_subset is None or df_subset.empty:
        return df_subset.copy()

    return pd.concat(
        [
            df_subset.reset_index(drop=True),
            df_subset.apply(
                lambda row: locate_percentile_bin(
                    row["y_true"],
                    row["x_age"],
                    interpolators
                ),
                axis=1
            ).reset_index(drop=True)
        ],
        axis=1
    )
def compute_BLR_and_save_outputs(df, feature, covset, bases, length_scale, sigma2, alpha2, save_path,
                                  hc_train_ids=None, hc_test_ids=None):
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
    mci = mci[mci['SITE'].isin(['Seoul', 'Spain'])]

    if hc_train_ids is not None and hc_test_ids is not None:
        # Split global: usar IDs pre-computados (mismo conjunto para todos los features)
        hc_train = hc[hc['subject'].isin(hc_train_ids)].copy()
        hc_test  = hc[hc['subject'].isin(hc_test_ids)].copy()
    else:
        # Fallback: split independiente por feature (comportamiento original)
        num_bins = 5
        age_bins = pd.qcut(hc['age'], q=num_bins, labels=False, duplicates='drop')
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(hc, age_bins))
        hc_train = hc.iloc[train_idx].copy()
        hc_test  = hc.iloc[test_idx].copy()
        if not mci.empty:
            mci_age_range = (mci['age'].min() - 5, mci['age'].max() + 5)
            hc_test = hc_test[hc_test['age'] >= mci_age_range[0]].copy()


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
    df_all_path = predict_and_store(X_all, y_all, ALL['subject'].values, ALL['SITE'].values,"ALL", "ALL", ALL['age'].values, Xv=Xv_all)
    df_acr = predict_and_store(X_acr, y_acr, ACr['subject'].values, ACr['SITE'].values,"ACr", "ACr", ACr['age'].values, Xv=Xv_acr)
    # Guardar bandas centiles evaluadas en un rango de edad usando predict_centile_bands
    #x_range = np.linspace(df['age'].min(), df['age'].max(), 200).reshape(-1, 1)
    #x_range = np.linspace(hc['age'].min(), mci['age'].max(), 200).reshape(-1, 1)
    x_range = np.linspace(hc['age'].min(), hc['age'].max(), 200).reshape(-1, 1)
    if use_hetero:
        mean_covariates = Xv_train.mean(axis=0)
        Xv_range = np.tile(mean_covariates, (x_range.shape[0], 1))
        age_index = covset.index("age")
        Xv_range[:, age_index] = x_range.flatten()
    else:
        Xv_range = None

    #centiles = blr.predict_centile_bands(x_range, Xv_test=Xv_range, percentiles=[1, 5, 25, 75, 95, 99], factor=0.5)
    centiles = blr.predict_centiles(
        x_range,
        Xv_test=Xv_range,
        percentiles=[1, 5, 25, 50, 75, 95, 99]
    )
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

    for p in [1, 5, 25, 50, 75, 95, 99]:
        df_range[f"centile_{p}"] = centiles[p]
        # df_range[f"centile_{p}_low"] = centiles[p][0]
        # df_range[f"centile_{p}_high"] = centiles[p][1]

    interpolators = build_centile_interpolators(df_range)
    dfs_sets = {
    "train": df_train,
    "test": df_test,
    "mci": df_mci,
    "ad": df_ad,
    "pd": df_pd,
    "vd": df_vd,
    "acr": df_acr,
    "nd": df_nd,
    "all_path": df_all_path,
    }
    for key, df_tmp in dfs_sets.items():
        dfs_sets[key] = add_percentile_info(df_tmp, interpolators)
    df_all = pd.concat(
        list(dfs_sets.values()) + [df_range],
        axis=0
    )
    train_mean = np.mean(y_train, keepdims=True)
    train_var = np.var(y_train, ddof=1, keepdims=True)
    total_var, _, _ = blr.get_variance_decomposition(X_test, Xv_test=Xv_test)
    
    total_var = total_var * (blr.y_std ** 2)  # <-- escalar a la varianza original
    msll_test = compute_MSLL(y_test, df_test.y_pred.values, total_var, train_mean, train_var)
    print(f"MSLL Test: {msll_test}")

    ev_train = explained_var(y_train, df_train.y_pred.values)
    ev_test = explained_var(y_test, df_test.y_pred.values)
    mse_train = np.mean((y_train - df_train.y_pred.values)**2)
    mse_test = np.mean((y_test - df_test.y_pred.values)**2)
    rho_train, pvalue_train = spearmanr(y_train, df_train.y_pred.values)
    rho_test,pvalue_test = spearmanr(y_test, df_test.y_pred.values)

    metrics = pd.DataFrame([{
        "band": feature,
        "msll_test": msll_test,
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


def classify_zscore_all_rois(df_all_combined, sets_incluidos=['test', 'mci'], rois=['F', 'C', 'P', 'O', 'PO']):
    """
    Classifies using z_scores from ALL ROIs simultaneously as a feature vector.
    Groups by (feature, band), pivots z_score per ROI, then runs CV classification.
    Also returns per-ROI feature importances for RF and DT, and coefficients for LogReg.
    """
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
    importance_results = []

    for (feat, band), df_feat in df_all_combined.groupby(['feature', 'band']):
        df_z = df_feat[df_feat['set'].isin(sets_incluidos)].dropna(subset=['z_score', 'roi'])

        # Pivot: one row per subject, one column per ROI
        df_pivot = (
            df_z[['subject', 'group', 'roi', 'z_score']]
            .pivot_table(index=['subject', 'group'], columns='roi', values='z_score')
            .reset_index()
        )
        df_pivot.columns.name = None

        roi_cols = [r for r in rois if r in df_pivot.columns]
        if len(roi_cols) == 0:
            continue

        df_pivot = df_pivot.dropna(subset=roi_cols)

        class_counts_original = df_pivot['group'].value_counts().to_dict()
        unique_classes = df_pivot['group'].unique()

        if len(unique_classes) < 2:
            print(f"Skipping {feat}-{band}: Not enough classes ({unique_classes})")
            continue

        min_class_size = df_pivot['group'].value_counts().min()
        balanced_list = []
        skip = False
        for cls in unique_classes:
            df_cls = df_pivot[df_pivot['group'] == cls]
            if len(df_cls) < 5:
                print(f"Skipping {feat}-{band}: class '{cls}' too small ({len(df_cls)})")
                skip = True
                break
            balanced_list.append(resample(df_cls, replace=False, n_samples=min_class_size, random_state=42))
        if skip:
            continue

        df_bal = pd.concat(balanced_list)
        y_raw = df_bal['group'].values
        class_map = {cls: i for i, cls in enumerate(np.unique(y_raw))}
        labels = np.array([class_map[g] for g in y_raw])
        scores = df_bal[roi_cols].values
        n_classes = len(np.unique(labels))
        dataset_label = ', '.join(sorted(np.unique(y_raw)))

        # Limit n_splits so CalibratedClassifierCV inner CV always has ≥1 sample per class
        n_splits = min(5, min_class_size)
        if n_splits < 2:
            print(f"Skipping {feat}-{band}: min_class_size={min_class_size} too small for CV")
            continue

        model_dict = {
            "SVM": CalibratedClassifierCV(LinearSVC(random_state=42, class_weight='balanced'), cv=n_splits),
            "LogReg": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            "RF": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "DT": DecisionTreeClassifier(random_state=42, class_weight='balanced')
        }

        for model_name, clf in model_dict.items():
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs, accs, precisions, recalls, f1s = [], [], [], [], []
            fold_importances = []

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

                aucs.append(auc)
                accs.append(accuracy_score(labels[test_idx], preds))
                precisions.append(precision_score(labels[test_idx], preds, average='macro', zero_division=0))
                recalls.append(recall_score(labels[test_idx], preds, average='macro', zero_division=0))
                f1s.append(f1_score(labels[test_idx], preds, average='macro', zero_division=0))

                # Feature importances / coefficients
                base_clf = clf.estimator if hasattr(clf, 'estimator') else clf
                if hasattr(base_clf, 'feature_importances_'):
                    fold_importances.append(base_clf.feature_importances_)
                elif hasattr(base_clf, 'coef_'):
                    coef = np.abs(base_clf.coef_)
                    fold_importances.append(coef.mean(axis=0) if coef.ndim > 1 else coef)

            results.append({
                "feature": feat,
                "band": band,
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

            if fold_importances:
                mean_imp = np.mean(fold_importances, axis=0)
                for roi_col, imp in zip(roi_cols, mean_imp):
                    importance_results.append({
                        "feature": feat,
                        "band": band,
                        "model": model_name,
                        "roi": roi_col,
                        "importance": imp
                    })

    return pd.DataFrame(results), pd.DataFrame(importance_results)


bands = ['theta', 'alpha1','alpha2', 'beta1','beta2','beta3', 'gamma']
#bands = ['theta', 'BGF1', 'BGF2', 'BGF3', 'beta']

rois = ['F', 'C', 'P', 'O', 'PO']
bases = 30
length_scale = 1.5
sigma2 = 0.06
alpha2 = 5

covset= ['age']

outlier_log = []
save_path = r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"

# ── SPLIT GLOBAL DE HC (una sola vez para todos los features) ─────────────────
# Se carga la primera familia para obtener los sujetos HC y su rango de edad MCI.
_first_family   = 'osc_pw_rel_canonic'
_file_name_ref  = f'{_first_family}_roi'
_base_feat_path = (r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS"
                   r"\AIF_Babiloni\results_harmonize\recombat\features_osc\age_group")

_hc_records = []
for _roi in rois:
    _p = fr"{_base_feat_path}\{_file_name_ref}{_roi}_SITE_age_group_recombat.xlsx"
    _d = pd.read_excel(_p, sheet_name="harmonizeSITE_age_group")
    _hc_records.append(_d[_d['group'] == 'HC'][['subject', 'age']].dropna())

_hc_global = pd.concat(_hc_records).drop_duplicates(subset='subject').reset_index(drop=True)

# Rango de edad del MCI (Seoul + Madrid) para filtrar el test set
_p_ref = fr"{_base_feat_path}\{_file_name_ref}{rois[0]}_SITE_age_group_recombat.xlsx"
_d_ref = pd.read_excel(_p_ref, sheet_name="harmonizeSITE_age_group")
_mci_ages = _d_ref[(_d_ref['group'] == 'MCI') & (_d_ref['SITE'].isin(['Seoul', 'Spain']))]['age'].dropna()
_mci_age_min = float(_mci_ages.min()) - 5

_age_bins_global = pd.qcut(_hc_global['age'], q=5, labels=False, duplicates='drop')
_splitter_global = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_tr_idx, _te_idx = next(_splitter_global.split(_hc_global, _age_bins_global))

_hc_train_ids = set(_hc_global.iloc[_tr_idx]['subject'].tolist())
_hc_test_ids_all = set(_hc_global.iloc[_te_idx]['subject'].tolist())
# Filtro de edad: test HC deben solapar con el rango etario del grupo clínico
_hc_test_ids = set(
    _hc_global[
        _hc_global['subject'].isin(_hc_test_ids_all) &
        (_hc_global['age'] >= _mci_age_min)
    ]['subject'].tolist()
)

print(f"Split global HC: {len(_hc_train_ids)} train | "
      f"{len(_hc_test_ids)} test (edad >= {_mci_age_min:.0f} anos)")

for family in ['osc_pw_rel_canonic','osc_pw_ab_canonic']: # ['osc_pw_rel_canonic', 'pw_rel_canonic', 'osc_pw_ab_canonic', 'pw_ab_canonic']
    df_all_list = []
    metrics_list = []
    if family == 'osc_pw_rel_canonic' or family == 'osc_pw_ab_canonic':
        non_band_features = ['exponent']
    else:
        non_band_features = ['IAFp']
    file_name = f'{family}_roi'
    for roi in rois:
        path = fr"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\recombat\features_osc\age_group\{file_name}{roi}_SITE_age_group_recombat.xlsx"
        data_roi = pd.read_excel(path, sheet_name="harmonizeSITE_age_group")
        #harmonize
        # Features con banda
        original_family = family

        for band in bands:
            feature_family = original_family
            if original_family == 'osc_pw_rel_canonic':
                feature_family = 'osc_pw_canonic'
            elif original_family == 'pw_rel_canonic':
                feature_family = 'pw_canonic'

            feature = f'harm_{feature_family}_{band}_roi{roi}'
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
                data_filtered, feature, covset, bases, length_scale, sigma2, alpha2, save_path,
                hc_train_ids=_hc_train_ids, hc_test_ids=_hc_test_ids
            )

            df_all["feature"] = feature_family
            df_all["band"] = band
            df_all["roi"] = roi
            metrics["feature"] = feature_family
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
                data_filtered, feature, covset, bases, length_scale, sigma2, alpha2, save_path,
                hc_train_ids=_hc_train_ids, hc_test_ids=_hc_test_ids
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
    print(f"Saved blr_{family}.csv and metrics_{family}.csv")

