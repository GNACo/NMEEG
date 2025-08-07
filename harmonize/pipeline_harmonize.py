import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Agrega el path a NormativeModel_EEG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.functions import apply_pca_umap, harmonize_transform

def extract_family_features(data):
    family_features ={
        'pw_rel' : list(data.columns[data.columns.str.startswith('pw') & 
                            ~data.columns.str.startswith('pw_canonic') & 
                            ~data.columns.str.startswith('pw_ab') & 
                            ~data.columns.str.startswith('pw_ab_canonic')]),
        'pw_rel_canonic': list(data.columns[data.columns.str.startswith('pw_canonic_')]),
        'pw_ab': list(data.columns[data.columns.str.startswith('pw_ab')
                        & ~data.columns.str.startswith('pw_ab_canonic')]),
        'pw_ab_canonic': list(data.columns[data.columns.str.startswith('pw_ab_canonic')]),
        'osc_pw_rel': list(data.columns[data.columns.str.startswith('osc_pw')
                    & ~data.columns.str.startswith('osc_pw_canonic')
                    & ~data.columns.str.startswith('osc_pw_ab')
                    & ~data.columns.str.startswith('osc_pw_ab_canonic')]),
        'osc_pw_rel_canonic': list(data.columns[data.columns.str.startswith('osc_pw_canonic')]),
        'osc_pw_ab': list(data.columns[data.columns.str.startswith('osc_pw_ab')
                    & ~data.columns.str.startswith('osc_pw_ab_canonic')]),
        'osc_pw_ab_canonic':list(data.columns[data.columns.str.startswith('osc_pw_ab_canonic')]) ,
    }
    return family_features

directory = r'D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni'

aproach1_age = pd.read_feather(rf'{directory}\IAF_FOOOF_ALL_SENSORS\IAF_FOOOF_age.feather')
renombrar_grupos = {'G2': 'HC', 'GU': 'HC', 'CTR': 'HC', 'DCL': 'MCI','A':'AD','DTA':'AD'}
aproach1_age['group'] = aproach1_age['group'].replace(renombrar_grupos)
goi = ['HC','MCI','AD','PD','VD','ACr','F']
aproach1_age = aproach1_age[aproach1_age['group'].isin(goi)]

dem_cols= list(aproach1_age.columns[:3])+['age']
aproach2_age = pd.read_feather(rf'{directory}\IAF_FOOOF_ROIS\rois_age.feather')
aproach2_age['group'] = aproach2_age['group'].replace(renombrar_grupos)
aproach2_age = aproach2_age[aproach2_age['group'].isin(goi)]

aproach2_age_sex = pd.read_feather(rf'{directory}\IAF_FOOOF_ROIS\rois_age_sex.feather')
aproach2_age_sex['group'] = aproach2_age_sex['group'].replace(renombrar_grupos)
aproach2_age_sex = aproach2_age_sex[aproach2_age_sex['group'].isin(goi)]

methods_harmonize = ['recombat']#'neuroharmonize'

for method in methods_harmonize:
    if method == 'neuroharmonize':
        covariates_list = [
        ['SITE', 'age'],
        ['SITE', 'age', 'sex'],
    ]
    elif method == 'recombat':
        covariates_list = [
        ['SITE', 'age','group'],
        ['SITE', 'age', 'sex','group'],
    ]
    for covariates in covariates_list:
        cov_suffix = "_".join(covariates)
        for roi in  ['roiF','roiC','roiO','roiP','roiPO']: 
            print(roi, cov_suffix, method)
            if covariates == ['SITE', 'age','group'] or covariates == ['SITE', 'age']:
                dem_cols= list(aproach2_age.columns[:3])+['age']
                print(dem_cols)
                data_roi = pd.concat([aproach2_age[dem_cols],aproach2_age[[i for i in aproach2_age.columns if i.endswith(roi)]]],axis=1)
                smoth_terms =['age']
            else:
                dem_cols= list(aproach2_age_sex.columns[:3])+['age','sex']
                data_roi = pd.concat([aproach2_age_sex[dem_cols],aproach2_age_sex[[i for i in aproach2_age_sex.columns if i.endswith(roi)]]],axis=1)
                if covariates == ['SITE','age','sex']:
                    smoth_terms = ['age','sex']                
                elif covariates == ['SITE','age', 'sex','group']:
                    smoth_terms = ['age', 'group','sex']
                elif covariates == ['SITE','sex']:
                    smoth_terms = ['sex']
            family_features = extract_family_features(data_roi)
            for key, features_values in family_features.items():
                results_dir = os.path.join(directory, f"results_harmonize/{method}")
                os.makedirs(results_dir, exist_ok=True)
                filename = f'{key}_{roi}_{cov_suffix}_{method}'
                excel_path = os.path.join(results_dir, f"{filename}.xlsx")
                features_values = list(features_values)  # ← evita mutar el original

                if  key == 'pw_rel' or key =='pw_rel_canonic' or key == 'pw_ab' or key == 'pw_ab_canonic':        
                    features_values = list(features_values) + [f'IAFp_{roi}']
                    features_values = [f for f in features_values if 'delta' not in f.lower()] 

                elif key == 'osc_pw_rel' or key == 'osc_pw_rel_canonic' or key == 'osc_pw_ab' or key == 'osc_pw_ab_canonic':
                    features_values =  [f for f in features_values if 'delta' not in f.lower()] + [ f'exponent_{roi}', f'offset_{roi}']
                if len(features_values) > 1:
                    print(key,features_values)
                    if covariates == ['SITE','sex']:
                        exclude_cols = [f'CF_extalpha_{roi}',f'TF_{roi}','age']
                    else:
                        exclude_cols = [f'CF_extalpha_{roi}',f'TF_{roi}']
                    print('Valores con ceros por columna')
                    # Ver cuántos ceros hay por columna
                    zero_counts = (data_roi == 0).sum()
                    columns_with_zeros = zero_counts[zero_counts > 0]
                    print(columns_with_zeros)
                    
                    model_harmonize, harmonize_data = harmonize_transform(data_roi, selected_cols = features_values, electrode=roi, exclude_cols=exclude_cols, smooth_terms=smoth_terms, covariates=covariates, method=method)
                    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                        unharmonize_age, harmonize_age = apply_pca_umap(data_roi, 
                                                                        harmonize_data, 
                                                                        features_values,
                                                                        dem_cols,  
                                                                        directory, 
                                                                        n_pca=3, 
                                                                        n_neighbors=40, 
                                                                        min_dist=0.4, 
                                                                        filename= filename, 
                                                                        method= method,
                                                                        aproach = 'age_ByROI', 
                                                                        family_features= key,
                                                                        excel_writer=writer, 
                                                                        sheet_name=f"{cov_suffix}",
                                                                        pref_covars = cov_suffix,
                                                                        roi=roi)
                    del model_harmonize, harmonize_data