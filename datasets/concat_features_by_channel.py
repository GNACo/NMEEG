import pandas as pd 

def extract_channels_feature(df, feature=str, channels_of_interest=['C3', 'C4', 'FP1', 'FP2', 'O1', 'O2', 'P7', 'P8']):
    
    # Función interna para extraer solo las columnas de un canal de interés específico
    def extract_channel(df, channel, channels_of_interest):
        # Filtrar columnas que empiecen con el nombre del canal de interés
        columns_of_interest = ['subject', 'Task', 'group','Site']  # Mantener siempre estas columnas
        columns_of_interest += [col for col in df.columns if col.startswith(channel)]
        
        # Extraer solo las columnas de interés
        df_filtered = df[columns_of_interest]
        
        # Renombrar las columnas con el prefijo
        df_filtered = df_filtered.rename(columns={col: f"{feature}_{col}" if col.startswith(channel) else col for col in df_filtered.columns})
        
        return df_filtered

    # Extraer los DataFrames de cada canal de interés
    O1_df = extract_channel(df, 'O1', channels_of_interest)
    O2_df = extract_channel(df, 'O2', channels_of_interest)
    P7_df = extract_channel(df, 'P7', channels_of_interest)
    P8_df = extract_channel(df, 'P8', channels_of_interest)

    return O1_df, O2_df, P7_df, P8_df

# path QA
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


# Path features
path_crossfreq= r'D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\zscore_crossfreq'
path_coherence= r'D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\zscore_coherence'
path_entropy = r'D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\zscore_entropy'
path_sl = r'D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\zscore_sl'

crossfreq_Dortmund= pd.read_feather(fr'{path_crossfreq}\crossfreq_Dortmund_EyesClosed_scorEpochs_zscore_columns.feather')
crossfreq_Dortmund.rename(columns={'Site ': 'Site'}, inplace=True)
crossfreq_Dortmund = crossfreq_Dortmund[~crossfreq_Dortmund['subject'].isin(subj_drop_Dortmund)]

crossfreq_AR= pd.read_feather(fr'{path_crossfreq}\crossfreq_Argentina_rs_scorEpochs_zscore_columns.feather')
crossfreq_CL= pd.read_feather(fr'{path_crossfreq}\crossfreq_Chile_rs_scorEpochs_zscore_columns.feather')
crossfreq_cuba= pd.read_feather(fr'{path_crossfreq}\crossfreq_Cuba_CE_scorEpochs_zscore_columns.feather')
crossfreq_Meedellin_ld= pd.read_feather(fr'{path_crossfreq}\crossfreq_Medellin_ld_CE_scorEpochs_zscore_columns.feather')
crossfreq_Meedellin_hd= pd.read_feather(fr'{path_crossfreq}\crossfreq_Medellin_hd_CE_scorEpochs_zscore_columns.feather')

crossfreq_CAUEEG= pd.read_feather(fr'{path_crossfreq}\crossfreq_CAUEEG_rs_scorEpochs_zscore_columns.feather')
crossfreq_CAUEEG.rename(columns={'site': 'Site'}, inplace=True)
crossfreq_CAUEEG = crossfreq_CAUEEG[~crossfreq_CAUEEG['subject'].isin(subj_drop_CAUEEG)]
crossfreq_CAUEEG.columns = (
    crossfreq_CAUEEG.columns
    .str.replace(r'T6', 'P8', regex=True)
    .str.replace(r'T5', 'P7', regex=True)
)

crossfreq_SRM= pd.read_feather(fr'{path_crossfreq}\crossfreq_Oslo_resteyesc_scorEpochs_zscore_columns.feather')
crossfreq_poland= pd.read_feather(fr'{path_crossfreq}\crossfreq_Poland_rest_scorEpochs_zscore_columns.feather')

crossfreq_greece = pd.read_feather(fr'{path_crossfreq}\crossfreq_GREECE_eyesclosed_scorEpochs_zscore_columns.feather')
crossfreq_greece['Site'] = 'Greece'
crossfreq_greece.columns = (
    crossfreq_greece.columns
    .str.replace(r'T6', 'P8', regex=True)
    .str.replace(r'T5', 'P7', regex=True)
)

coh_Dortmund= pd.read_feather(fr'{path_coherence}\cohfreq_Dortmund_EyesClosed_scorEpochs_zscore_columns.feather')
coh_Dortmund.rename(columns={'Site ': 'Site'},inplace=True)
coh_Dortmund.rename(columns={'Site ': 'Site'}, inplace=True)
coh_Dortmund = coh_Dortmund[~coh_Dortmund['subject'].isin(subj_drop_Dortmund)]

coh_AR= pd.read_feather(fr'{path_coherence}\cohfreq_Argentina_rs_scorEpochs_zscore_columns.feather')
coh_CL= pd.read_feather(fr'{path_coherence}\cohfreq_Chile_rs_scorEpochs_zscore_columns.feather')
coh_cuba= pd.read_feather(fr'{path_coherence}\cohfreq_Cuba_CE_scorEpochs_zscore_columns.feather')
coh_Meedellin_ld= pd.read_feather(fr'{path_coherence}\cohfreq_Medellin_ld_CE_scorEpochs_zscore_columns.feather')
coh_Meedellin_hd= pd.read_feather(fr'{path_coherence}\cohfreq_Medellin_hd_CE_scorEpochs_zscore_columns.feather')
coh_CAUEEG= pd.read_feather(fr'{path_coherence}\cohfreq_CAUEEG_rs_scorEpochs_zscore_columns.feather')
coh_CAUEEG.rename(columns={'site': 'Site'}, inplace=True)
coh_CAUEEG = coh_CAUEEG[~coh_CAUEEG['subject'].isin(subj_drop_CAUEEG)]
coh_CAUEEG.columns = (
    coh_CAUEEG.columns
    .str.replace('T6', 'P8', regex=True)
    .str.replace('T5', 'P7', regex=True)
)

coh_SRM= pd.read_feather(fr'{path_coherence}\cohfreq_Oslo_resteyesc_scorEpochs_zscore_columns.feather')
coh_poland= pd.read_feather(fr'{path_coherence}\cohfreq_Poland_rest_scorEpochs_zscore_columns.feather')
coh_greece= pd.read_feather(fr'{path_coherence}\cohfreq_GREECE_eyesclosed_scorEpochs_zscore_columns.feather')
coh_greece['Site'] = 'Greece'
coh_greece.columns = (
    coh_greece.columns
    .str.replace(r'T6', 'P8', regex=True)
    .str.replace(r'T5', 'P7', regex=True)
)



sl_Dortmund= pd.read_feather(fr'{path_sl}\sl_Dortmund_EyesClosed_scorEpochs_zscore_columns.feather')
sl_Dortmund.rename(columns={'Site ': 'Site'},inplace=True)
sl_Dortmund.rename(columns={'Site ': 'Site'}, inplace=True)
sl_Dortmund = sl_Dortmund[~sl_Dortmund['subject'].isin(subj_drop_Dortmund)]

sl_AR= pd.read_feather(fr'{path_sl}\sl_Argentina_rs_scorEpochs_zscore_columns.feather')
sl_CL= pd.read_feather(fr'{path_sl}\sl_Chile_rs_scorEpochs_zscore_columns.feather')
sl_cuba= pd.read_feather(fr'{path_sl}\sl_Cuba_CE_scorEpochs_zscore_columns.feather')
sl_Meedellin_ld= pd.read_feather(fr'{path_sl}\sl_Medellin_ld_CE_scorEpochs_zscore_columns.feather')
sl_Meedellin_hd= pd.read_feather(fr'{path_sl}\sl_Medellin_hd_CE_scorEpochs_zscore_columns.feather')
sl_CAUEEG= pd.read_feather(fr'{path_sl}\sl_CAUEEG_rs_scorEpochs_zscore_columns.feather')
sl_CAUEEG.rename(columns={'site': 'Site'}, inplace=True)
sl_CAUEEG = sl_CAUEEG[~sl_CAUEEG['subject'].isin(subj_drop_CAUEEG)]
sl_CAUEEG.columns = (
    sl_CAUEEG.columns
    .str.replace('T6', 'P8', regex=True)
    .str.replace('T5', 'P7', regex=True)
)

sl_SRM= pd.read_feather(fr'{path_sl}\sl_Oslo_resteyesc_scorEpochs_zscore_columns.feather')
sl_poland= pd.read_feather(fr'{path_sl}\sl_Poland_rest_scorEpochs_zscore_columns.feather')
sl_greece= pd.read_feather(fr'{path_sl}\sl_GREECE_eyesclosed_scorEpochs_zscore_columns.feather')
sl_greece['Site'] = 'Greece'
crossfreq_greece.columns = (
    crossfreq_greece.columns
    .str.replace('T6', 'P8', regex=True)
    .str.replace('T5', 'P7', regex=True)
)

entropy_Dortmund= pd.read_feather(fr'{path_entropy}\entropy_Dortmund_EyesClosed_scorEpochs_zscore_columns.feather')
entropy_Dortmund.rename(columns={'Site ': 'Site'},inplace=True)
entropy_Dortmund.rename(columns={'Site ': 'Site'}, inplace=True)
entropy_Dortmund = entropy_Dortmund[~entropy_Dortmund['subject'].isin(subj_drop_Dortmund)]

entropy_AR= pd.read_feather(fr'{path_entropy}\entropy_Argentina_rs_scorEpochs_zscore_columns.feather')
entropy_CL= pd.read_feather(fr'{path_entropy}\entropy_Chile_rs_scorEpochs_zscore_columns.feather')
entropy_cuba= pd.read_feather(fr'{path_entropy}\entropy_Cuba_CE_scorEpochs_zscore_columns.feather')
entropy_Meedellin_ld= pd.read_feather(fr'{path_entropy}\entropy_Medellin_ld_CE_scorEpochs_zscore_columns.feather')
entropy_Meedellin_hd= pd.read_feather(fr'{path_entropy}\entropy_Medellin_hd_CE_scorEpochs_zscore_columns.feather')
entropy_CAUEEG= pd.read_feather(fr'{path_entropy}\entropy_CAUEEG_rs_scorEpochs_zscore_columns.feather')
entropy_CAUEEG.rename(columns={'site': 'Site'}, inplace=True)
entropy_CAUEEG = entropy_CAUEEG[~entropy_CAUEEG['subject'].isin(subj_drop_CAUEEG)]
entropy_CAUEEG.columns = (
    entropy_CAUEEG.columns
    .str.replace('T6', 'P8', regex=True)
    .str.replace('T5', 'P7', regex=True)
)
entropy_SRM= pd.read_feather(fr'{path_entropy}\entropy_Oslo_resteyesc_scorEpochs_zscore_columns.feather')
entropy_poland= pd.read_feather(fr'{path_entropy}\entropy_Poland_rest_scorEpochs_zscore_columns.feather')
entropy_greece= pd.read_feather(fr'{path_entropy}\entropy_GREECE_eyesclosed_scorEpochs_zscore_columns.feather')
entropy_greece['Site'] = 'Greece'
entropy_greece.columns = (
    entropy_greece.columns
    .str.replace('T6', 'P8', regex=True)
    .str.replace('T5', 'P7', regex=True)
)

# Lista de las métricas que quieres extraer
features = ['sl', 'coh', 'entropy', 'crossfreq']

# Lista de los datasets, donde cada dataset corresponde a una métrica específica
datasets = {
    'entropy':[entropy_Dortmund, entropy_AR,entropy_CL,entropy_cuba, entropy_Meedellin_ld,entropy_Meedellin_hd,entropy_CAUEEG, entropy_SRM, entropy_poland, entropy_greece],
    'coh':[ coh_Dortmund,coh_AR,coh_CL,coh_cuba,coh_Meedellin_ld,coh_Meedellin_hd, coh_CAUEEG, coh_SRM, coh_poland, coh_greece],
    'sl':[sl_Dortmund, sl_AR, sl_CL, sl_cuba, sl_Meedellin_ld, sl_Meedellin_hd, sl_CAUEEG, sl_SRM, sl_poland, sl_greece],
    'crossfreq':[crossfreq_Dortmund,crossfreq_AR,crossfreq_CL,crossfreq_cuba, crossfreq_Meedellin_ld,crossfreq_Meedellin_hd,crossfreq_CAUEEG, crossfreq_SRM, crossfreq_poland, crossfreq_greece]
}

entropy = pd.concat (datasets['entropy'], ignore_index=True)
coh = pd.concat(datasets['coh'], ignore_index=True)
sl = pd.concat(datasets['sl'], ignore_index=True)
crossfreq = pd.concat(datasets['crossfreq'], ignore_index=True)

O1_df, O2_df, P7_df, P8_df = extract_channels_feature(entropy, feature='entropy')
O1_coh, O2_coh, P7_coh, P8_ch = extract_channels_feature(coh, feature='coh')
O1_sl, O2_sl, P7_sl, P8_sl = extract_channels_feature(sl, feature='sl')
O1_crossfreq, O2_crossfreq, P7_crossfreq, P8_crossfreq = extract_channels_feature(crossfreq, feature='crossfreq')
O1_features = pd.merge(O1_df, O1_coh, on=['subject','Site','group','Task'], how='outer').merge(O1_sl, on=['subject','Site','group','Task'], how='outer').merge(O1_crossfreq, on=['subject','Site','group','Task'], how='outer')
O1_features.drop(columns=['Task'], inplace=True)
O1_features.dropna(inplace=True)
O2_features = pd.merge(O2_df, O2_coh, on=['subject','Site','group','Task'], how='outer').merge(O2_sl, on=['subject','Site','group','Task'], how='outer').merge(O2_crossfreq, on=['subject','Site','group','Task'], how='outer')
O2_features.drop(columns=['Task'], inplace=True)
O2_features.dropna(inplace=True)
P7_features = pd.merge(P7_df, P7_coh, on=['subject','Site','group','Task'], how='outer').merge(P7_sl, on=['subject','Site','group','Task'], how='outer').merge(P7_crossfreq, on=['subject','Site','group','Task'], how='outer')
P7_features.drop(columns=['Task'], inplace=True)
P7_features.dropna(inplace=True)
P8_features = pd.merge(P8_df, P8_ch, on=['subject','Site','group','Task'], how='outer').merge(P8_sl, on=['subject','Site','group','Task'], how='outer').merge(P8_crossfreq, on=['subject','Site','group','Task'], how='outer')
P8_features.drop(columns=['Task'], inplace=True)
P8_features.dropna(inplace=True)

