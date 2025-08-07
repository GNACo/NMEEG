'''
@Author: Alberto Jaramillo Jimenez
las convenciones de cada grupo están en el paper de ellos (Tabla 1). 
https://www.sciencedirect.com/science/article/pii/S1053811923002008?via%3Dihub 
'''
import pandas as pd
# Reloading the Excel file, treating the 'serial' column as a string from the start
file_path = r'D:/MulticentersEEG/caueeg-dataset/annotation.xlsx'
# Re-loading the original file with correct columns
df_original = pd.read_excel(file_path, dtype={'serial': str})

df = df_original
#Apply the necessary transformations again
df['participant_id'] = 'sub-' + df['serial']
df = df.rename(columns={'age': 'age'})
df = df[['participant_id', 'age']]  # Keep only participant_id and age for now
df['ad_syndrome'] = ''

# Populate ad_syndrome column

# Step 1: Assign "hc" where "hc_normal" or "cb_normal" are True
df.loc[(df_original['hc_normal'].notnull()) | (df_original['cb_normal'].notnull()), 'ad_syndrome'] = 'hc'

# Step 2: Assign "mci" where any 'mci_ad', 'mci_amnestic', 'mci_amnestic_ef', 'mci_multi_domain' is True (encoding failures were selected vs retrieval failures)
mci_columns = ['mci_ad', 'mci_amnestic', 'mci_amnestic_ef', 'mci_multi_domain'] 
df.loc[df_original[mci_columns].notnull().any(axis=1), 'ad_syndrome'] = 'mci'

# Step 3: Assign "dementia" where any of the dementia-related columns are True
dementia_columns = ['ad', 'load', 'eoad', 'ad_vd_mixed']
df.loc[df_original[dementia_columns].notnull().any(axis=1), 'ad_syndrome'] = 'dementia'

# Step 4: Assign "smc" where "smi" is True
df.loc[df_original['smi'].notnull(), 'ad_syndrome'] = 'smc'

df.to_excel(r'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_CAUEEG.xlsx', index=False)


# Opcion 2
#Estos grupos son los de la variable "level_2" y se formaron así 


# Apply the necessary transformations again
# df['participant_id'] = 'sub-' + df['serial']
# df = df.rename(columns={'age': 'age'})
# df = df[['participant_id', 'age']]  # Keep only participant_id and age for now
# df['level_1'] = ''
# df['level_2'] = ''

# # Populate level_1 column

# # Assign "hc (+smc)" where "normal" is True
# df.loc[df_original['normal'].notnull(), 'level_1'] = 'hc (+smc)'

# # Assign "mci" where "normal" is True
# df.loc[df_original['mci'].notnull(), 'level_1'] = 'mci'

# # Assign "dementia" where "dementia" is True
# df.loc[df_original['dementia'].notnull(), 'level_1'] = 'dementia'


# # Populate level_2 column

# # Assign healthy controls "hc (+smc)" where "normal" is True
# df.loc[df_original['normal'].notnull(), 'level_2'] = 'hc (+smc)'

# # Assign mild cognitive impairment "mci" where "normal" is True
# df.loc[df_original['mci'].notnull(), 'level_2'] = 'mci'

# # Assign alzheimer's disease "ad" where any 'ad', 'eoad', 'load' is True
# ad_related_columns = ['ad', 'eoad', 'load']
# df.loc[df_original[ad_related_columns].notnull().any(axis=1), 'level_2'] = 'ad'

# # Assign "parkinson_synd" where 'parkinson_synd' is True
# df.loc[df_original['parkinson_synd'].notnull(), 'level_2'] = 'parkinson_synd'

# # Assign "ftd" where 'ftd' is True
# df.loc[df_original['ftd'].notnull(), 'level_2'] = 'ftd'

# # Assign vascular dementia "vd" where any 'vd', 'sivd' is True
# vd_related_columns = ['vd', 'sivd']
# df.loc[df_original[vd_related_columns].notnull().any(axis=1), 'level_2'] = 'vd'



# df.to_excel(r'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_CAUEEG2.xlsx', index=False)

