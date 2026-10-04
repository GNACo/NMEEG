# NMEEG — Normative modeling of resting-state EEG oscillatory power

Code accompanying the manuscript on normative modeling of resting-state EEG
oscillatory relative power in healthy controls (HC), mild cognitive impairment (MCI),
Alzheimer's disease (AD) and asymptomatic *PSEN1* E280A mutation carriers (ACr),
pooled from multiple recording centers.

> Manuscript title, authors and DOI: _to be added on publication_.

## Pipeline

| # | Step | Entry point | Output |
|---|---|---|---|
| 1 | EEG preprocessing and spectral parameterization (FOOOF/specparam, individual alpha frequency) | `filter_spectrals.ipynb`, `FOOOF_Analysis/` | Feature database (`FOOOF_DB`) and per-ROI IAF tables (`IAF_ROIS_DIR`) |
| 2 | Per-site dataset ingestion | `datasets/` | Per-site feature files |
| 3 | Site harmonization (primary: neuroHarmonize, learned on training HC only) | `harmonize/harmonize_neuroharmonize.py` | `HARM_OUT_DIR` |
| 4 | Normative model: Bayesian linear regression on age with RBF bases; bounded relative power modeled on the logit scale | `testblr.py` (uses `models/BLR.py`) | `blr_<family>.csv`, `metrics_<family>.csv` in `BLR_DIR` |
| 5 | Classification of HC vs MCI, AD and ACr from z-score deviations | `classify_blr.py` | `SVM_*.csv`, `feature_importance_*.csv` in `BLR_DIR` |
| 6 | Sensitivity analyses (age-matched HC reference for ACr; group statistics) | `analysis/` | `sensitivity_acr_agematched.csv`, `hc_ad_mci_zscore_stats.xlsx` |
| 7 | Figures | `run_figures.py`, `figures/` | PNG and PDF in `FIGURES_DIR` |
| 8 | Tables | `tables/*.tex` | LaTeX tables used in the manuscript |

Key analysis choices (details in `docs/decision_harmonization.md`):

- Harmonization is learned on the training HC participants of a single global split
  (stratified by age). No patient data or diagnostic label is used to fit the site correction.
- Oscillatory relative power is bounded in (0, 1). The normative model is fit on the logit
  scale and back-transformed for predictions and centiles.
- Outlier bounds (IQR) are computed on training HC only and applied to the training
  rows, never to test or clinical participants.
- Classification uses stratified 5-fold cross-validation with in-fold undersampling of the
  majority class, and 50 out-of-bag bootstrap iterations for AUC stability.
- ReComBat harmonization is kept as a sensitivity analysis (`harmonize/pipeline_harmonize.py`).

## Repository layout

| Path | Contents |
|---|---|
| `testblr.py` | Normative model (step 4). |
| `classify_blr.py` | Classification (step 5). Also imported by the sensitivity analysis. |
| `run_figures.py` | Runs the main figure scripts in `figures/` (step 7). |
| `filter_spectrals.ipynb` | EEG preprocessing notebook (step 1). |
| `FOOOF_Analysis/` | Spectral parameterization and IAF extraction (step 1). |
| `datasets/` | Per-site ingestion and feature concatenation (step 2): `CAUEEG/`, `CHBMP/`, `brainlat/`, `concat_features_by_channel.py`. |
| `harmonize/` | neuroHarmonize (primary) and ReComBat (sensitivity) harmonization (step 3). |
| `models/` | `BLR.py` (Bayesian linear regression), `model_metrics.py` (MSLL, explained variance). |
| `analysis/` | Sensitivity analysis for ACr and group statistics (step 6). |
| `figures/` | Figure scripts (`fig1_age_distribution.py`, `fig1_normative_model.py`, `fig2_group_deviations.py`, `fig3_spectra_zoom.py`, `fig_results.py`, `blr_performance_figure.py`) and the demographic and manuscript notebooks. |
| `tables/` | LaTeX tables: `table_demographics.tex`, `table_classification.tex`. |
| `Functions/` | Helper functions for ReComBat and the preprocessing notebook. |
| `run_config_params/` | Dataset definitions (`DATASET.py`), derived paths (`paths.py`) and the local config template. |
| `docs/` | Decision record for the harmonization method. |
| `archive/` | Exploratory notebooks and superseded scripts, kept for transparency; not needed to reproduce the results (see `archive/README.md`). |

## Requirements

Two environments are needed.

- **Analysis** (Python 3.11): `pip install -r requirements.txt`. Pinned to the versions
  used for the reported results. Covers steps 4–8.
- **Harmonization** (Python 3.11): `pip install -r requirements-harmonize.txt`. Covers the
  primary harmonization (step 3). Not pinned yet.
- **ReComBat sensitivity** (Python < 3.10, because the `reCombat` library requires it):
  `pip install -r requirements-recombat.txt`, in a separate environment. Only needed to rerun
  the ReComBat sensitivity analysis.

## Configuration

Machine-specific paths are not stored in the repository. Create the local config:

```bash
cp run_config_params/config_local.example.py run_config_params/config_local.py
```

and set:

- `RESULTS_ROOT`: folder with the processed outputs (the `AIF_Babiloni` folder). All analysis
  and figure scripts derive their input and output paths from it (`run_config_params/paths.py`).
- `EEG_ROOT`, `DEMOGRAPHIC_DIR`: raw recordings and demographic files, used only by the
  ingestion scripts in `datasets/` and by `run_config_params/DATASET.py`.

`config_local.py` is git-ignored.

## Data availability

The individual EEG recordings and participant-level clinical data are not publicly available
because of the consent and ethics restrictions of the contributing centers. Derived,
de-identified data needed to rerun the analysis can be requested from the corresponding
author, subject to data-sharing agreements.

## Reproduction order

Run from the repository root.

1. Preprocessing: `filter_spectrals.ipynb`, `FOOOF_Analysis/FOOOF.ipynb`,
   `FOOOF_Analysis/FOOOF_24_scorEpoch.ipynb`, and the ingestion scripts in `datasets/`.
2. Harmonization (harmonization environment): `python harmonize/harmonize_neuroharmonize.py`.
3. Normative model (analysis environment): `python testblr.py`.
4. Classification: `python classify_blr.py`.
5. Sensitivity analyses: `python analysis/sensitivity_acr_agematched.py` and
   `python analysis/extract_hc_ad_mci_stats.py`.
6. Figures: `python run_figures.py`, then `python figures/fig1_age_distribution.py`
   and `python figures/fig1_normative_model.py`. The demographic and manuscript figure
   notebooks are in `figures/`.
7. Tables: `tables/table_demographics.tex` and `tables/table_classification.tex` are the
   LaTeX tables used in the manuscript.

## Known limitations

- `filter_spectrals.ipynb` imports `Functions.aproach`, which was archived in
  `archive/superseded_scripts/aproach.py`. The import will fail until the notebook is updated.
- Some scripts still contain absolute paths from the original workstation and must be adapted
  before running: `datasets/concat_features_by_channel.py`, `FOOOF_Analysis/IAF.py`,
  `Functions/graphs_functions.py`, and the `sys.path` line in `filter_spectrals.ipynb`.
- Steps 1–2 are partly notebook-based and depend on raw data that is not distributed.
- Sex is not recorded for every site; see the footnote in `tables/table_demographics.tex`.

## License

MIT. See [LICENSE](LICENSE).

## Citation

_To be added on publication (a `CITATION.cff` will be provided with the DOI)._
