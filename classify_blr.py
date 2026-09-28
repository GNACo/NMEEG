import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline


RANDOM_STATE = 42


def _make_models(inner_cv):
    return {
        "SVM": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE, class_weight='balanced', max_iter=5000),
                cv=inner_cv
            )
        ),
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
        ),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'),
        "DT":  DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    }


def _safe_n_splits(min_class_size, max_splits=5):
    n = min(max_splits, min_class_size)
    return n if n >= 2 else None


def _inner_cv(n_splits):
    """Inner CV folds for CalibratedClassifierCV: safe given training fold size."""
    train_per_class = (n_splits - 1)  # lower bound multiplier
    return max(2, min(n_splits, train_per_class))


def _balance_inside_fold(X_train, y_train):
    """Undersample majority class inside fold only (no leakage into test)."""
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    idx_balanced = []
    rng = np.random.RandomState(RANDOM_STATE)
    for cls in classes:
        idx_cls = np.where(y_train == cls)[0]
        chosen = rng.choice(idx_cls, size=min_count, replace=False)
        idx_balanced.append(chosen)
    idx_balanced = np.concatenate(idx_balanced)
    rng.shuffle(idx_balanced)
    return X_train[idx_balanced], y_train[idx_balanced]


def _extract_importances(clf, n_features):
    """Extract feature importances from a fitted clf or pipeline."""
    final_clf = clf[-1] if hasattr(clf, '__getitem__') else clf
    if hasattr(final_clf, 'calibrated_classifiers_'):
        coefs = []
        for cal in final_clf.calibrated_classifiers_:
            inner = cal.estimator
            if hasattr(inner, 'coef_'):
                coefs.append(np.abs(inner.coef_).mean(axis=0))
        return np.mean(coefs, axis=0) if coefs else None
    elif hasattr(final_clf, 'feature_importances_'):
        return final_clf.feature_importances_
    elif hasattr(final_clf, 'coef_'):
        coef = np.abs(final_clf.coef_)
        return coef.mean(axis=0) if coef.ndim > 1 else coef.flatten()
    return None


def _run_cv(scores, labels, n_classes, n_splits, inner_cv_k, collect_importances=False, feat_cols=None):
    """
    Stratified CV with balancing INSIDE each fold.
    Returns: fold metrics dict, aggregated confusion matrix, per-fold importances,
             feature vote counts (if collect_importances), OOF probas and true labels.
    """
    model_dict = _make_models(inner_cv_k)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_results = {m: {"aucs": [], "accs": [], "precs": [], "recs": [], "f1s": [], "briers": []}
                   for m in model_dict}
    all_cm = {m: np.zeros((n_classes, n_classes), dtype=int) for m in model_dict}
    fold_importances = {m: [] for m in model_dict}
    feature_votes = {m: np.zeros(scores.shape[1] if scores.ndim > 1 else 1, dtype=int)
                     for m in model_dict}
    oof_probas = {m: [] for m in model_dict}
    oof_true = []

    for train_idx, test_idx in skf.split(scores, labels):
        X_tr_raw, y_tr = scores[train_idx], labels[train_idx]
        X_te,     y_te = scores[test_idx],  labels[test_idx]

        # Balance only training fold
        X_tr, y_tr = _balance_inside_fold(X_tr_raw, y_tr)
        oof_true.extend(y_te.tolist())

        for model_name, clf in model_dict.items():
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            probas = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None

            if n_classes == 2 and probas is not None:
                auc = roc_auc_score(y_te, probas[:, 1])
                brier = brier_score_loss(y_te, probas[:, 1])
                oof_probas[model_name].extend(probas[:, 1].tolist())
            elif probas is not None:
                y_te_bin = label_binarize(y_te, classes=np.arange(n_classes))
                auc = roc_auc_score(y_te_bin, probas, multi_class='ovr', average='macro')
                brier = float('nan')
                oof_probas[model_name].extend(probas.tolist())
            else:
                auc = brier = float('nan')

            all_results[model_name]["aucs"].append(auc)
            all_results[model_name]["accs"].append(accuracy_score(y_te, preds))
            all_results[model_name]["precs"].append(precision_score(y_te, preds, average='macro', zero_division=0))
            all_results[model_name]["recs"].append(recall_score(y_te, preds, average='macro', zero_division=0))
            all_results[model_name]["f1s"].append(f1_score(y_te, preds, average='macro', zero_division=0))
            all_results[model_name]["briers"].append(brier)
            all_cm[model_name] += confusion_matrix(y_te, preds, labels=np.arange(n_classes))

            if collect_importances:
                imp = _extract_importances(clf, scores.shape[1] if scores.ndim > 1 else 1)
                if imp is not None:
                    fold_importances[model_name].append(imp)
                    # Vote: feature was selected if importance > 0
                    voted = (imp > 0).astype(int)
                    feature_votes[model_name] += voted

    return all_results, all_cm, fold_importances, feature_votes, oof_probas, np.array(oof_true)


def bootstrap_stability(scores, labels, n_classes, n_splits, inner_cv_k, n_bootstrap=100):
    """
    Bootstrap OOB AUC to assess stability of the best CV result.
    Returns AUC mean ± std over bootstrap resamples.
    """
    rng = np.random.RandomState(RANDOM_STATE + 99)
    model_dict = _make_models(inner_cv_k)
    boot_aucs = {m: [] for m in model_dict}

    for b in range(n_bootstrap):
        idx_boot = rng.choice(len(labels), size=len(labels), replace=True)
        oob_mask = np.ones(len(labels), dtype=bool)
        oob_mask[idx_boot] = False
        oob_idx = np.where(oob_mask)[0]

        if oob_idx.sum() < 4 or len(np.unique(labels[oob_idx])) < 2:
            continue

        X_boot, y_boot = scores[idx_boot], labels[idx_boot]
        X_oob,  y_oob  = scores[oob_idx],  labels[oob_idx]
        X_boot, y_boot = _balance_inside_fold(X_boot, y_boot)

        for model_name, clf in model_dict.items():
            try:
                clf.fit(X_boot, y_boot)
                probas = clf.predict_proba(X_oob) if hasattr(clf, "predict_proba") else None
                if probas is None:
                    continue
                if n_classes == 2:
                    auc = roc_auc_score(y_oob, probas[:, 1])
                else:
                    y_oob_bin = label_binarize(y_oob, classes=np.arange(n_classes))
                    auc = roc_auc_score(y_oob_bin, probas, multi_class='ovr', average='macro')
                boot_aucs[model_name].append(auc)
            except Exception:
                pass

    return {m: {"auc_mean": float(np.mean(v)), "auc_std": float(np.std(v)), "n": len(v)}
            for m, v in boot_aucs.items() if v}


# =============================================================================
# Per-ROI classification (one z_score per feature×band×roi at a time)
# =============================================================================

def classify_zscore_models(df_all_combined, sets_incluidos=['test', 'MCI'], n_bootstrap=0):
    results = []

    for (feat, band, roi), df_feat in df_all_combined.groupby(['feature', 'band', 'roi']):
        df_z = df_feat[df_feat['set'].isin(sets_incluidos)].dropna(subset=['z_score'])

        class_counts_original = df_z['group'].value_counts().to_dict()
        unique_classes = df_z['group'].unique()

        if len(unique_classes) < 2:
            continue

        min_class_size = df_z['group'].value_counts().min()
        if min_class_size < 5:
            print(f"Skipping {feat}-{band}-{roi}: min class size {min_class_size} < 5")
            continue

        y_raw = df_z['group'].values
        class_map = {cls: i for i, cls in enumerate(np.unique(y_raw))}
        labels = np.array([class_map[g] for g in y_raw])
        scores = df_z['z_score'].values.reshape(-1, 1)
        n_classes = len(np.unique(labels))
        dataset_label = ', '.join(sorted(np.unique(y_raw)))

        n_splits = _safe_n_splits(min_class_size)
        if n_splits is None:
            continue
        inner_cv_k = _inner_cv(n_splits)

        fold_results, all_cm, _, _, oof_probas, oof_true = _run_cv(
            scores, labels, n_classes, n_splits, inner_cv_k, collect_importances=False
        )

        boot_results = {}
        if n_bootstrap > 0:
            boot_results = bootstrap_stability(scores, labels, n_classes, n_splits, inner_cv_k, n_bootstrap)

        for model_name, m in fold_results.items():
            row = {
                "feature": feat,
                "band": band,
                "roi": roi,
                "model": model_name,
                "Set": dataset_label,
                "AUC_mean": float(np.nanmean(m["aucs"])),
                "AUC_std":  float(np.nanstd(m["aucs"])),
                "accuracy":  float(np.nanmean(m["accs"])),
                "precision": float(np.nanmean(m["precs"])),
                "recall":    float(np.nanmean(m["recs"])),
                "f1_score":  float(np.nanmean(m["f1s"])),
                "brier":     float(np.nanmean(m["briers"])),
                "cm":        all_cm[model_name].tolist(),
                "counts":    class_counts_original,
            }
            if model_name in boot_results:
                row["boot_AUC_mean"] = boot_results[model_name]["auc_mean"]
                row["boot_AUC_std"]  = boot_results[model_name]["auc_std"]
            results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# All features classification (todos los ROIs y bandas como vector)
# =============================================================================

def classify_zscore_all_features(df_all_combined, sets_incluidos=['test', 'mci'], n_bootstrap=50):
    """
    Un solo clasificador usando TODOS los ROIs y TODAS las bandas como vector de features.
    Balanceo dentro de cada fold (sin data leakage).
    Incluye: Brier Score, confusion matrix agregada, feature vote stability, bootstrap OOB.
    """
    results = []
    importance_results = []

    df_z = df_all_combined[df_all_combined['set'].isin(sets_incluidos)].dropna(subset=['z_score']).copy()

    df_z['feat_col'] = (
        df_z['feature'].astype(str) + '__'
        + df_z['band'].fillna('noband').astype(str) + '__'
        + df_z['roi'].astype(str)
    )

    df_pivot = (
        df_z[['subject', 'group', 'feat_col', 'z_score']]
        .pivot_table(index=['subject', 'group'], columns='feat_col', values='z_score')
        .reset_index()
    )
    df_pivot.columns.name = None

    feat_cols = [c for c in df_pivot.columns if c not in ('subject', 'group')]

    # Mantener sujetos con ≥50% features; imputar resto con mediana de columna
    min_features = int(np.ceil(len(feat_cols) * 0.5))
    df_pivot = df_pivot[df_pivot[feat_cols].notna().sum(axis=1) >= min_features].copy()
    col_medians = df_pivot[feat_cols].median()
    df_pivot[feat_cols] = df_pivot[feat_cols].fillna(col_medians)

    class_counts_original = df_pivot['group'].value_counts().to_dict()
    unique_classes = df_pivot['group'].unique()

    if len(unique_classes) < 2:
        print(f"Skipping: Not enough classes ({unique_classes})")
        return pd.DataFrame(results), pd.DataFrame(importance_results)

    min_class_size = df_pivot['group'].value_counts().min()
    if min_class_size < 5:
        print(f"Skipping: min class size {min_class_size} < 5")
        return pd.DataFrame(results), pd.DataFrame(importance_results)

    y_raw = df_pivot['group'].values
    class_map = {cls: i for i, cls in enumerate(np.unique(y_raw))}
    labels = np.array([class_map[g] for g in y_raw])
    scores = df_pivot[feat_cols].values.astype(float)
    n_classes = len(np.unique(labels))
    dataset_label = ', '.join(sorted(np.unique(y_raw)))

    n_splits = _safe_n_splits(min_class_size)
    if n_splits is None:
        print(f"Skipping: min_class_size={min_class_size} too small for CV")
        return pd.DataFrame(results), pd.DataFrame(importance_results)
    inner_cv_k = _inner_cv(n_splits)

    fold_results, all_cm, fold_imps, feature_votes, oof_probas, oof_true = _run_cv(
        scores, labels, n_classes, n_splits, inner_cv_k,
        collect_importances=True, feat_cols=feat_cols
    )

    boot_results = {}
    if n_bootstrap > 0:
        boot_results = bootstrap_stability(scores, labels, n_classes, n_splits, inner_cv_k, n_bootstrap)

    majority_thr = n_splits // 2  # feature selected in ≥50% of folds

    for model_name, m in fold_results.items():
        row = {
            "model":     model_name,
            "Set":       dataset_label,
            "AUC_mean":  float(np.nanmean(m["aucs"])),
            "AUC_std":   float(np.nanstd(m["aucs"])),
            "accuracy":  float(np.nanmean(m["accs"])),
            "precision": float(np.nanmean(m["precs"])),
            "recall":    float(np.nanmean(m["recs"])),
            "f1_score":  float(np.nanmean(m["f1s"])),
            "brier":     float(np.nanmean(m["briers"])),
            "cm":        all_cm[model_name].tolist(),
            "counts":    class_counts_original,
        }
        if model_name in boot_results:
            row["boot_AUC_mean"] = boot_results[model_name]["auc_mean"]
            row["boot_AUC_std"]  = boot_results[model_name]["auc_std"]
        results.append(row)

        # Feature importances: media entre folds + vote stability
        if fold_imps[model_name]:
            mean_imp = np.mean(fold_imps[model_name], axis=0)
            votes = feature_votes[model_name]
            for col_name, imp, vote in zip(feat_cols, mean_imp, votes):
                parts = col_name.split('__')
                importance_results.append({
                    "feature":       parts[0],
                    "band":          parts[1] if parts[1] != 'noband' else np.nan,
                    "roi":           parts[2],
                    "model":         model_name,
                    "importance":    imp,
                    "fold_votes":    int(vote),           # en cuántos folds fue importante
                    "stable":        vote >= majority_thr, # True si ≥50% de los folds
                })

    return pd.DataFrame(results), pd.DataFrame(importance_results)


# =============================================================================
# Configuration
# =============================================================================
save_path = r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized"
families = ['osc_pw_rel_canonic', 'osc_pw_ab_canonic']
rois = ['F', 'C', 'P', 'O', 'PO']

comparison_sets = {
    "MCI": ['test', 'mci'],
    "AD":  ['test', 'AD'],
    "PD":  ['test', 'PD'],
    "VD":  ['test', 'VD'],
    "ACr": ['test', 'ACr'],
    "ALL": ['test', 'ALL'],
    "ND":  ['test', 'ND'],
}

# =============================================================================
# Main loop
# =============================================================================
for family in families:
    csv_path = os.path.join(save_path, f"blr_{family}.csv")
    print(f"\nLoading {csv_path}")
    df_all_combined = pd.read_csv(csv_path)

    # Per-ROI (one z_score at a time)
    for label, sets in comparison_sets.items():
        ml = classify_zscore_models(df_all_combined, sets_incluidos=sets, n_bootstrap=0)
        ml.to_csv(os.path.join(save_path, f"SVM_{family}_{label}.csv"), index=False)
        if not ml.empty:
            print(f"  {label}: best AUC = {ml.AUC_mean.max():.3f}  "
                  f"brier = {ml.loc[ml.AUC_mean.idxmax(), 'brier']:.3f}")

    # All features combined
    for label, sets in comparison_sets.items():
        ml_all, imp_all = classify_zscore_all_features(df_all_combined, sets_incluidos=sets, n_bootstrap=50)
        ml_all.to_csv(os.path.join(save_path, f"SVM_allFeatures_{family}_{label}.csv"), index=False)
        imp_all.to_csv(os.path.join(save_path, f"feature_importance_{family}_{label}.csv"), index=False)
        if not ml_all.empty:
            best_row = ml_all.loc[ml_all.AUC_mean.idxmax()]
            boot_str = (f"  boot={best_row['boot_AUC_mean']:.3f}±{best_row['boot_AUC_std']:.3f}"
                        if 'boot_AUC_mean' in best_row else "")
            print(f"  AllFeatures {label}: AUC={best_row['AUC_mean']:.3f}±{best_row['AUC_std']:.3f}"
                  f"  brier={best_row['brier']:.3f}{boot_str}")

# cross validation solo training 
# estandarización solo sobre el conjunto de training
# no hacer inputación 