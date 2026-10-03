# Copy this file to config_local.py (same folder) and set the paths for your machine.
# config_local.py is git-ignored and must not be committed.

# Root folder with the raw per-site EEG recordings (BIDS layout), used by DATASET.py
EEG_ROOT = r'<path to raw EEG root>'

# Folder with the per-site demographic Excel files, used by DATASET.py
DEMOGRAPHIC_DIR = r'<path to demographic files>'

# Root folder of the processed outputs (FOOOF features, ReCombat harmonization and BLR results).
# Equivalent to the "AIF_Babiloni" folder used throughout the pipeline.
RESULTS_ROOT = r'<path to AIF_Babiloni results root>'
