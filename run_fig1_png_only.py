import os, sys
import matplotlib
matplotlib.use("Agg")

# Override save to PNG only
_SAVE_PATH = os.environ.get("NMEEG_FIGURES_PATH",
    r"D:\MulticentersEEG\Features_2_normativeModel\gamma_40\24_BEST_EPOCHS\AIF_Babiloni\results_harmonize\recombat\BLR_paper\bands_age\harmonized\figures_paper"
)
os.makedirs(_SAVE_PATH, exist_ok=True)

# Run the module but intercept savefig
import importlib.util, types
spec = importlib.util.spec_from_file_location("fig1", "prepare_datasets/fig1_normative_model.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
