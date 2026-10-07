# NMEEG — Normative modeling of resting-state EEG oscillatory power

Code for the manuscript on normative modeling of resting-state EEG oscillatory power
(manuscript title and DOI to be added on publication). Run all commands from the repository root.

## Requirements

- Analysis (Python 3.11): `pip install -r requirements.txt`
- Harmonization (Python 3.11): `pip install -r requirements-harmonize.txt`

## Configuration

Create the local configuration file and set the paths for your machine:

```bash
cp run_config_params/config_local.example.py run_config_params/config_local.py
```

- `RESULTS_ROOT`: folder with the processed outputs (`AIF_Babiloni`). The analysis and figure scripts read and write inside it.
- `EEG_ROOT`, `DEMOGRAPHIC_DIR`: raw recordings and demographic files, used only by the ingestion scripts in `datasets/`.

`config_local.py` is git-ignored.

## Steps

1. **Preprocessing and features**: `filter_spectrals.ipynb`, `FOOOF_Analysis/FOOOF.ipynb`,
   `FOOOF_Analysis/FOOOF_24_scorEpoch.ipynb`, and the ingestion scripts in `datasets/`.
2. **Harmonization** (harmonization environment): `python harmonize/harmonize_neuroharmonize.py`
3. **Normative model**: `python testblr.py`
4. **Classification**: `python classify_blr.py`
5. **Supplementary analyses**: `python analysis/site_sensitivity.py` and
   `python analysis/extract_hc_ad_mci_stats.py`
6. **Figures**: `python run_figures.py`, then `python figures/fig1_age_distribution.py` and
   `python figures/fig1_normative_model.py`
7. **Tables**: the LaTeX files in `tables/` (`table_demographics.tex`, `table_classification.tex`).
