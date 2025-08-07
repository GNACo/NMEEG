from pingouin import ancova, homoscedasticity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest, levene, kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from xicorpy import compute_xi_correlation
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import os
import statsmodels.formula.api as smf
from scipy.stats import shapiro

def extract_band(feature_name):
    """Extrae la banda de una cadena de texto que contiene el nombre de la característica."""
    fname = feature_name.lower()
    if "delta" in fname:
        return "delta"
    elif "theta" in fname:
        return "theta"
    elif "bgf1" in fname:
        return "BGF1"
    elif "bgf2" in fname:
        return "BGF2"
    elif "bgf3" in fname:
        return "BGF3"
    elif "beta" in fname:
        if "beta1" in fname:
            return "beta1"
        elif "beta2" in fname:
            return "beta2"
        elif "beta3" in fname:
            return "beta3"
        else:
            return "beta"
    elif "alpha1" in fname:
        return "alpha1"
    elif "alpha2" in fname:
        return "alpha2"
    elif "gamma" in fname:
        return "gamma"
    else:
        return feature_name

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def get_multinomial_logit_std_beta(df, feature_col, age_col, site_col):
    """
    Ejecuta una regresión logística multinomial SITE ~ feature + age
    y devuelve el coeficiente estandarizado del feature.
    """
    # Filtrar y quitar NaNs
    data = df[[feature_col, age_col, site_col]].dropna()

    # Filtrar sitios con al menos 10 sujetos
    site_counts = data[site_col].value_counts()
    valid_sites = site_counts[site_counts >= 20].index
    data = data[data[site_col].isin(valid_sites)]
    
    # Estandarizar las variables predictoras
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[[feature_col, age_col]])
    X = sm.add_constant(X_scaled)  # Agregar constante para intercepto

    # Codificar SITE como variable categórica
    y = pd.Categorical(data[site_col]).codes

    try:
        model = sm.MNLogit(y, X).fit(disp=0)
        coef = model.params.iloc[:, 1]  # coeficiente del feature (columna 1 porque 0 = intercepto)
        pval = model.pvalues.iloc[:, 1]  # p-valor del coeficiente

        std = data[feature_col].std()
        beta_std = coef * std
        # Devuelve promedio si hay múltiples clases (una por clase de referencia)
        return beta_std.mean(), pval.mean()
    except Exception:
        return np.nan, np.nan
    

   
def staticals_features(df_unharmonize, df_harmonize, features,
                       directory, method,
                       excel_filename=str,
                       sheet_name=str,
                       family_features= str, covars= str, roi= str):
    
    supuestos = pd.DataFrame(columns=["Feature",
                                    "covars",
                                    "Category",
                                    "roi" ,
                                    "band",
                                    "shap_stat",
                                    "shap_pval",
                                    "shap",
                                    "levene_stat_zscore",
                                    "levene_pval_zscore",
                                    "Levene_zscore",
                                    "shap_Harmonized",
                                    "shap_stat_Harmonized",
                                    "shap_pval_Harmonized",
                                    "levene_stat_Harmonized",
                                    "levene_pval_Harmonized",
                                    "levene_Harmonized",
                                    "kruskal_stat_zscore",
                                    "kruskal_pval_zscore",
                                    "kruskal_zscore",
                                    "kruskal_stat_Harmonized",
                                    "kruskal_pval_Harmonized",
                                    "kruskal_Harmonized",
                                    "xicor_zscore",
                                    "xicor_Harmonized",
                                    "group" ,
                                    "mannwhitney_stat_zscore",
                                    "mannwhitney_pval_zscore",
                                    "mannwhitney_zscore",  
                                    "mannwhitney_stat_Harmonized",
                                    "mannwhitney_pval_Harmonized",
                                    "mannwhitney_Harmonized",
                                    "rspearman_hc_unharmonized",
                                    "pvalue_hc_unharmonized",
                                    "rspearman_mci_unharmonized",
                                    "pvalue_mci_unharmonized",
                                    "rspearman_hc_harmonized",
                                    "pvalue_hc_harmonized",
                                    "rspearman_mci_harmonized",
                                    "pvalue_mci_harmonized",
                                    "moderation_coef",
                                    "moderation_pval",
                                    "moderation_zscore",
                                    # "logit_beta_std_zscore",
                                    # "logit_pval_zscore",
                                    # "logit_beta_std_harmonized",
                                    # "logit_pval_harmonized"
                                    ])
    
    # Recorremos cada característica
    for feature in features:
        feature_harmonize = f'harm_{feature}'
        # Incluir "group" para realizar la prueba Mann-Whitney
        data_unharmonize = df_unharmonize[[feature, 'age', 'SITE', 'group']].dropna()
        data_harmonize = df_harmonize[[feature_harmonize, 'age', 'SITE', 'group']].dropna()
        
        y_unharmonize = data_unharmonize[feature]
        y_harmonize = data_harmonize[feature_harmonize]
        
        # 1. Normality with Kolmogorov-Smirnov
        
        shap_stat, shap_pval = shapiro(y_unharmonize)
        Norm_unharm = "Normal" if shap_pval >= 0.05 else "Not normal"

        shap_stat_harm, shap_pval_harm = shapiro(y_harmonize)
        Norm_harm = "Normal" if shap_pval_harm >= 0.05 else "Not normal"
        
        # 2. Homogeneity of variances with Levene
       
        grouped_data_unharmonize = [data_unharmonize[data_unharmonize['SITE'] == site][feature] 
                                    for site in data_unharmonize['SITE'].unique()]
        levene_stat, levene_pval = levene(*grouped_data_unharmonize)
        Hom_zscore = "Homogeneous" if levene_pval >= 0.05 else "Heterogeneous"
        
        grouped_data_harmonize = [data_harmonize[data_harmonize['SITE'] == site][feature_harmonize] 
                                  for site in data_harmonize['SITE'].unique()]
        levene_stat_harmonized, levene_pval_harmonized = levene(*grouped_data_harmonize)
        Hom_Harmonized = "Homogeneous" if levene_pval_harmonized >= 0.05 else "Heterogeneous"
        
        # 3. Kruskal-Wallis test
        kruskal_stat, kruskal_pval = kruskal(*grouped_data_unharmonize)
        Kruskal_zscore = "Significant differences" if kruskal_pval < 0.05 else "No significant differences"
        
        kruskal_stat_harmonized, kruskal_pval_harmonized = kruskal(*grouped_data_harmonize)
        Kruskal_Harmonized = "Significant differences" if kruskal_pval_harmonized < 0.05 else "No significant differences"
        
        # 4. Xi Correlation between the dependent variable and age
        # Asumimos que compute_xi_correlation está definida y retorna al menos una tupla cuyo primer elemento es el coeficiente
        xicor_zscore = compute_xi_correlation(data_unharmonize['age'].values, y_unharmonize.values)[0]
        xicor_Harmonized = compute_xi_correlation(data_harmonize['age'].values, y_harmonize.values)[0]
        
        # 5. Mann-Whitney U test for comparing HC vs. MCI based on the 'group' column
        
        group_unharm_HC = data_unharmonize[data_unharmonize['group'] == 'HC'][feature]
        group_unharm_MCI = data_unharmonize[data_unharmonize['group'] == 'MCI'][feature]
        try:
            mwu_stat, mwu_pval = mannwhitneyu(group_unharm_HC, group_unharm_MCI, alternative='two-sided')
        except Exception as e:
            mwu_stat, mwu_pval = np.nan, np.nan
        mannwhitney_zscore = "Significant" if mwu_pval < 0.05 else "Not significant"
        
        group_harm_HC = data_harmonize[data_harmonize['group'] == 'HC'][feature_harmonize]
        group_harm_MCI = data_harmonize[data_harmonize['group'] == 'MCI'][feature_harmonize]
        try:
            mwu_stat_h, mwu_pval_h = mannwhitneyu(group_harm_HC, group_harm_MCI, alternative='two-sided')
        except Exception as e:
            mwu_stat_h, mwu_pval_h = np.nan, np.nan
        mannwhitney_Harmonized = "Significant" if mwu_pval_h < 0.05 else "Not significant"
        
        #6. Spearman correlation
        r_unharmonize_hc, p_unharmonize_hc = spearmanr(data_unharmonize[data_unharmonize['group'] == 'HC']['age'], group_unharm_HC)
        r_unharmonize_mci, p_unharmonize_mci = spearmanr(data_unharmonize[data_unharmonize['group'] == 'MCI']['age'], group_unharm_MCI)
        r_harmonize_hc, p_harmonize_hc = spearmanr(data_harmonize[data_harmonize['group'] == 'HC']['age'], group_harm_HC) 
        r_harmonize_mci, p_harmonize_mci = spearmanr(data_harmonize[data_harmonize['group'] == 'MCI']['age'], group_harm_MCI)

        # Preparar datos para moderación
        mod_data = data_unharmonize.copy()
        mod_data['group'] = mod_data['group'].astype('category')  # Asegura codificación correcta
        mod_data['interaction'] = mod_data['age'] * mod_data['group'].cat.codes

        # Modelo lineal con interacción (moderación)
        formula = f"{feature} ~ age * group"
        try:
            model = smf.ols(formula, data=mod_data).fit()
            interaction_pval = model.pvalues.get('age:group[T.MCI]', np.nan)  # cambia T.MCI si tu codificación es diferente
            interaction_coef = model.params.get('age:group[T.MCI]', np.nan)
        except Exception as e:
            interaction_pval = np.nan
            interaction_coef = np.nan
            
        # # Regresión logistica multinomial para considerar la variable de age
        # logit_beta_std_zscore, logit_pval_zscore = get_multinomial_logit_std_beta(df_unharmonize, feature, 'age', 'SITE')
        # logit_beta_std_harmonized, logit_pval_harmonized = get_multinomial_logit_std_beta(df_harmonize, f'harm_{feature}', 'age', 'SITE')


        # Create a DataFrame to store the results
        # Append results to the DataFrame
        supuestos = pd.concat([
            supuestos, 
            pd.DataFrame({
                "Feature": [feature],
                "covars": [covars],
                "Category":[family_features],
                "roi": [roi],
                "band": [extract_band(feature)],
                #KS test Unharmonize
                "shap_stat": [shap_stat],
                "shap_pval": [shap_pval],
                "ks": [Norm_unharm],
                #Levene test Unharmonize
                "levene_stat_zscore": [levene_stat],
                "levene_pval_zscore": [levene_pval],
                "Levene_zscore": [Hom_zscore],
                #KS test Harmonized
                "ks_Harmonized": [Norm_harm],
                "shap_stat_Harmonized": [shap_stat_harm],
                "shap_pval_Harmonized": [shap_pval_harm],
                #Levene test Harmonized
                "levene_stat_Harmonized": [levene_stat_harmonized],
                "levene_pval_Harmonized": [levene_pval_harmonized],
                "levene_Harmonized": [Hom_Harmonized],
                #Kruskal test Unharmonize
                "kruskal_stat_zscore": [kruskal_stat],
                "kruskal_pval_zscore": [kruskal_pval],
                "kruskal_zscore": [Kruskal_zscore],
                #Kruskal test Harmonized
                "kruskal_stat_Harmonized": [kruskal_stat_harmonized],
                "kruskal_pval_Harmonized": [kruskal_pval_harmonized],
                "kruskal_Harmonized": [Kruskal_Harmonized],
                #Xi Correlation
                "xicor_zscore": [xicor_zscore],
                "xicor_Harmonized": [xicor_Harmonized],
                # Mann-Whitney U test Unharmonize
                "group":  ["HC vs MCI"],
                "mannwhitney_stat_zscore": [mwu_stat],
                "mannwhitney_pval_zscore": [mwu_pval],
                "mannwhitney_zscore": [mannwhitney_zscore],
                # Mann-Whitney U test Harmonized    
                "mannwhitney_stat_Harmonized": [mwu_stat_h],
                "mannwhitney_pval_Harmonized": [mwu_pval_h],
                "mannwhitney_Harmonized": [mannwhitney_Harmonized],
                #R Spearman Unharmonize
                "rspearman_hc_unharmonized": [r_unharmonize_hc],
                "pvalue_hc_unharmonized": [p_unharmonize_hc],
                "rspearman_mci_unharmonized": [r_unharmonize_mci],
                "pvalue_mci_unharmonized": [p_unharmonize_mci],
                #R Spearman Harmonized
                "rspearman_hc_harmonized": [r_harmonize_hc],
                "pvalue_hc_harmonized": [p_harmonize_hc],
                "rspearman_mci_harmonized": [r_harmonize_mci],
                "pvalue_mci_harmonized": [p_harmonize_mci],
                # Moderation (OLS interaction)
                "moderation_coef": [interaction_coef],
                "moderation_pval": [interaction_pval],
                "moderation_zscore": ["Significant" if interaction_pval < 0.05 else "Not significant"],
                # Multinomial logistic regression
                # "logit_beta_std_zscore": [logit_beta_std_zscore],
                # "logit_pval_zscore": [logit_pval_zscore],
                # "logit_beta_std_harmonized": [logit_beta_std_harmonized],
                # "logit_pval_harmonized": [logit_pval_harmonized],
            })
        ], ignore_index=True)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    X = df_unharmonize[features + ['age']].dropna()
    X = sm.add_constant(X)

    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vifs)   
    
    out_dir = os.path.join(directory, "results_harmonize", method, "statical_tests")
    os.makedirs(out_dir, exist_ok=True)
    excel_path = os.path.join(out_dir, excel_filename)
    
    if os.path.exists(excel_path):
        # Intenta cargar el archivo existente
        try:
            existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
            supuestos = pd.concat([existing_df, supuestos], ignore_index=True)
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                supuestos.to_excel(writer, sheet_name=sheet_name, index=False)
           
        except IndexError:
            print(f"[!] El archivo '{excel_path}' está corrupto (sin hojas visibles). Se sobrescribirá.")
            os.remove(excel_path)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                supuestos.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Si no existe, lo crea desde cero
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            supuestos.to_excel(writer, sheet_name=sheet_name, index=False)

    return supuestos



