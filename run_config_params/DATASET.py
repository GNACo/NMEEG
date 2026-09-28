from pathlib import Path

try:
    from run_config_params.config_local import EEG_ROOT, DEMOGRAPHIC_DIR
except ImportError:
    from config_local import EEG_ROOT, DEMOGRAPHIC_DIR

_eeg = Path(EEG_ROOT)
_dem = Path(DEMOGRAPHIC_DIR)

# DATABASES
BIOMARCADORES = {
    'name': 'Medellin_hd',
    'input_path': str(_eeg / 'BIOMARCADORES'),
    'layout': {'extension': '.vhdr', 'task': 'CE', 'suffix': 'eeg', 'return_type': 'filename', 'session': 'V0'},
    'args': {'line_freqs': [60]},
    'group_regex': '(.+).{3}',
    'events_to_keep': None,
    'run-label': 'restCE',
    'session': 'V',
    'demographic': str(_dem / 'dem_biomarcadores_zscore.xlsx'),
}

DUQUE = {
    'name': 'COL_PRG14-1-02',
    'input_path': str(_eeg / 'DUQUE'),
    'layout': {'extension': '.vhdr', 'task': 'resting', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'restCE',
    'session': 'V',
    'demographic': str(_dem / 'dem_Duque.xlsx'),
}

PORTABLES_CE = {
    'name': 'Medellin_ld',
    'input_path': str(_eeg / 'BIDS_PORTABLES_CODIFICADO'),
    'layout': {'extension': '.vhdr', 'task': 'CE', 'suffix': 'eeg', 'return_type': 'filename', 'session': 'V0'},
    'args': {'line_freqs': [60]},
    'group_regex': '(.+).{3}',
    'events_to_keep': None,
    'run-label': 'restCE',
    'session': 'V',
    'demographic': str(_dem / 'dem_portables_zscore.xlsx'),
}

SRM = {
    'name': 'Oslo',
    'input_path': str(_eeg / 'SRM'),
    'layout': {'extension': '.edf', 'task': 'resteyesc', 'suffix': 'eeg', 'return_type': 'filename', 'session': 't1'},
    'args': {'line_freqs': [50]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'restCE',
    'session': 'V',
    'demographic': str(_dem / 'dem_SRM_zscore.xlsx'),
}

Dortmund = {
    'name': 'Dortmund',
    'input_path': str(_eeg / 'Dortmund_Vital_Study'),
    'layout': {'extension': '.edf', 'task': 'EyesClosed', 'acquisition': 'pre', 'suffix': 'eeg', 'return_type': 'filename', 'session': '1'},
    'args': {'line_freqs': [50]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'EyesClosed',
    'session': 'V',
    'demographic': str(_dem / 'dem_Dortmund.xlsx'),
}

CHBMP = {
    'name': 'Cuba',
    'input_path': str(_eeg / 'CHMP'),
    'layout': {'extension': '.edf', 'task': 'CE', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'group_regex': None,
    'run-label': 'restCE',
    'session': None,
    'demographic': str(_dem / 'dem_CHBMP_zscore.xlsx'),
}

Poland = {
    'name': 'Poland',
    'input_path': str(_eeg / 'Polonia'),
    'layout': {'extension': '.vhdr', 'task': 'rest', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [50]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'rest',
    'session': None,
    'demographic': str(_dem / 'dem_Polonia_zscore.xlsx'),
}

CL = {
    'name': 'Chile',
    'input_path': str(_eeg / 'BrainLat' / 'CL_BIDS'),
    'layout': {'extension': '.vhdr', 'task': 'rs', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'rs',
    'session': 'V',
    'demographic': str(_dem / 'dem_BrainLat_CL_zscore.xlsx'),
}

AR = {
    'name': 'Argentina',
    'input_path': str(_eeg / 'BrainLat' / 'AR_BIDS'),
    'layout': {'extension': '.vhdr', 'task': 'rs', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'rs',
    'session': 'V',
    'demographic': str(_dem / 'dem_BrainLat_AR_zscore.xlsx'),
}

CAUEEG_CONCAT = {
    'name': 'CAUEEG',
    'input_path': str(_dem.parent / 'bids_cau_eyes_closed'),
    'layout': {'extension': '.vhdr', 'task': 'rs', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'rs',
    'session': 'V',
    'demographic': str(_dem / 'dem_CAUEEG2.xlsx'),
}

GREECE = {
    'name': 'GREECE',
    'input_path': str(_eeg / 'Grecia'),
    'layout': {'extension': '.set', 'task': 'eyesclosed', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [50]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'eyesclosed',
    'demographic': str(_dem / 'dem_Greece_zscore.xlsx'),
}

NMT_NORMAL = {
    'name': 'NMT_NORMAL',
    'input_path': str(_eeg / 'nmt_scalp_eeg_dataset' / 'normal_BIDS'),
    'layout': {'extension': '.vhdr', 'task': 'rs', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'eyesclosed',
}

Madrid_c3n = {
    'name': 'Madrid',
    'input_path': str(_eeg / 'BIDS_MUSCA_OO'),
    'layout': {'extension': '.vhdr', 'task': 'rest', 'suffix': 'eeg', 'return_type': 'filename'},
    'args': {'line_freqs': [50]},
    'group_regex': None,
    'events_to_keep': None,
    'run-label': 'rest',
    'demographic': str(_dem / 'dem_Madrid_c3n_zscore.xlsx'),
}
