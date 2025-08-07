directory_demographic = rf'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic'
# DATABASES 
BIOMARCADORES={
    'name':'Medellin_hd',
    'input_path':'F:\EEG_MULTICENTER\BIOMARCADORES',
    'layout':{'extension':'.vhdr', 'task':'CE','suffix':'eeg', 'return_type':'filename', 'session':'V0'},
    'args':{'line_freqs':[60]},
    'group_regex':'(.+).{3}',
    'events_to_keep':None,
    'run-label':'restCE',
    'session':'V',
    'demographic':r'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_biomarcadores_zscore.xlsx' 
}

DUQUE={
    'name':'COL_PRG14-1-02',
    'input_path':'F:\EEG_MULTICENTER\DUQUE',
    'layout':{'extension':'.vhdr', 'task':'resting','suffix':'eeg', 'return_type':'filename'},
    'args':{'line_freqs':[60]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'restCE',
    'session':'V',
    'demographic':rf'{directory_demographic}\dem_Duque.xlsx' 
}

PORTABLES_CE={
    'name':'Medellin_ld',
    'input_path':r'F:\EEG_MULTICENTER\BIDS_PORTABLES_CODIFICADO',
    'layout':{'extension':'.vhdr', 'task':'CE','suffix':'eeg', 'return_type':'filename', 'session':'V0'},
    'args':{'line_freqs':[60]},
    'group_regex':'(.+).{3}',
    'events_to_keep':None,
    'run-label':'restCE',
    'session':'V',
    'demographic':r'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_portables_zscore.xlsx'
}

SRM = {
    'name':'Oslo',
    'input_path':r'F:\EEG_MULTICENTER\SRM',
    'layout':{'extension':'.edf', 'task':'resteyesc','suffix':'eeg', 'return_type':'filename', 'session':'t1'},
    'args':{'line_freqs':[50]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'restCE',
    'session':'V',
    'demographic':r'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_SRM_zscore.xlsx'
    }

Dortmund={
    'name':'Dortmund',
    'input_path':r'F:\EEG_MULTICENTER\Dortmund_Vital_Study',
    'layout':{'extension':'.edf', 'task':'EyesClosed','acquisition': 'pre','suffix':'eeg','return_type':'filename','session':'1'},
    'args':{'line_freqs':[50]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'EyesClosed',
    'session':'V',
    'demographic':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_Dortmund.xlsx"
}

CHBMP = {
    'name':'Cuba',
    'input_path':r'F:\EEG_MULTICENTER\CHMP',
    'layout':{'extension':'.edf', 'task':'CE','suffix':'eeg', 'return_type':'filename'}, #'task':'protmap'
    'args':{'line_freqs':[60],},
    'group_regex':None,
    #'events_to_keep':[65],
    'run-label':'restCE',
    'session':None,
    'demographic':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_CHBMP_zscore.xlsx"
}

Poland = {
    'name':'Poland',
    'input_path':r'F:\EEG_MULTICENTER\Polonia',
    'layout':{'extension':'.vhdr', 'task':'rest','suffix':'eeg', 'return_type':'filename'},
    'args':{'line_freqs':[50],},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'rest',
    'session':None,
    'demographic':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_Polonia_zscore.xlsx"
    
}


CL={
    'name':'Chile',
    'input_path':r'F:\EEG_MULTICENTER\BrainLat\CL_BIDS',
    'layout':{'extension':'.vhdr', 'task':'rs','suffix':'eeg', 'return_type':'filename'},
    'args':{'line_freqs':[60]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'rs',
    'session':'V',
    'demographic':r'C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_BrainLat_CL_zscore.xlsx'
}

AR={
    'name':'Argentina',
    'input_path':r'F:\EEG_MULTICENTER\BrainLat\AR_BIDS',
    'layout':{'extension':'.vhdr', 'task':'rs','suffix':'eeg', 'return_type':'filename'},
    'args':{'line_freqs':[60]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'rs',
    'session':'V',
    'demographic':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_BrainLat_AR_zscore.xlsx"
    
}

CAUEEG_CONCAT={
    'name':'CAUEEG',
    'input_path':r'D:\MulticentersEEG\bids_cau_eyes_closed',
    'layout':{'extension':'.vhdr', 'task':'rs','suffix':'eeg', 'return_type':'filename'},
    'args':{'line_freqs':[60]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'rs',
    'session':'V',
    'demographic':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_CAUEEG2.xlsx"
}

GREECE={
    'name':'GREECE',
    'input_path':r'C:\Users\Luisa\Documents\DB_HARMONIZE\Grecia',
    'layout':{'extension':'.set', 'task':'eyesclosed','suffix':'eeg','return_type':'filename'},
    'args':{'line_freqs':[50]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'eyesclosed',
    'demographic':r"C:\Users\Luisa\Documents\DB_HARMONIZE\demographic\dem_Greece_zscore.xlsx"    
}

NMT_NORMAL={
    'name':'NMT_NORMAL',
    'input_path':r'C:\Users\Luisa\Documents\DB_HARMONIZE\nmt_scalp_eeg_dataset\normal_BIDS',
    'layout':{'extension':'.vhdr', 'task':'rs','suffix':'eeg','return_type':'filename'},
    'args':{'line_freqs':[60]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'eyesclosed'    
}


Madrid_c3n={
    'name':'Madrid',
    'input_path':r'F:\EEG_MULTICENTER\BIDS_MUSCA_OO',
    'layout':{'extension':'.vhdr', 'task':'rest','suffix':'eeg','return_type':'filename'},
    'args':{'line_freqs':[50]},
    'group_regex':None,
    'events_to_keep':None,
    'run-label':'rest' ,
    'demographic':rf"{directory_demographic}\dem_Madrid_c3n_zscore.xlsx"
}






