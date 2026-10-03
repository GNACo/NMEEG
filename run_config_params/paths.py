"""
Derived paths for the analysis pipeline. All machine-specific roots come from
run_config_params/config_local.py (see config_local.example.py).
"""
import os

try:
    from run_config_params.config_local import RESULTS_ROOT
except ImportError:
    from config_local import RESULTS_ROOT

FOOOF_DB = os.path.join(RESULTS_ROOT, "filtered_IAFs_fooof_database.feather")
IAF_ROIS_DIR = os.path.join(RESULTS_ROOT, "IAF_FOOOF_ROIS")
RECOMBAT_DIR = os.path.join(RESULTS_ROOT, "results_harmonize", "recombat")
FEATURES_OSC_DIR = os.path.join(RECOMBAT_DIR, "features_osc", "age_group")
BLR_DIR = os.path.join(RECOMBAT_DIR, "BLR_paper", "bands_age", "harmonized")
FIGURES_DIR = os.path.join(BLR_DIR, "figures_paper")

# Primary harmonization: neuroHarmonize (ComBat-GAM, learned on training HC only)
HARM_SUFFIX = "neuroharmonize"
HARM_OUT_DIR = os.path.join(RECOMBAT_DIR, "features_osc_neuroharmonize", "age_group")
