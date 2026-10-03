# NMEEG — Normative modeling of resting-state EEG oscillatory power

Code accompanying the manuscript on normative modeling of resting-state EEG
oscillatory relative power across healthy controls, MCI, AD and asymptomatic
*PSEN1* E280A mutation carriers (ACr), pooled from eleven recording centers.

> Manuscript title, authors and DOI: _to be added on publication_.

## What the pipeline does

1. **Preprocessing and feature extraction** — EEG cleaning (`filter_spectrals.ipynb`),
   FOOOF/specparam parameterization and individual alpha frequency (`FOOOF_Analysis/`).
2. **Site harmonization** — ReComBat, with site as batch and age/group as protected
   covariates (`harmonize/pipeline_harmonize.py`, `harmonize/recombat.py`).
3. **Normative modeling** — Bayesian linear regression on age with radial basis functions.
   Bounded features (oscillatory relative power) are modeled on the logit scale and
   back-transformed (`testblr.py`, `models/BLR.py`).
4. **Classification** — z-score deviations as biomarkers, HC vs MCI / AD / ACr
   (`classify_blr.py`).
5. **Figures and tables** — `run_figures.py` (main results, Fig. 2, Fig. 3) and
   `prepare_datasets/fig1_normative_model.py`, `extract_hc_ad_mci_stats.py`.

## Repository layout

| Path | Purpose |
|---|---|
| `testblr.py` | Main normative-model run. Writes `blr_osc_pw_rel_canonic.csv` and `blr_osc_pw_ab_canonic.csv`. |
| `classify_blr.py` | Classification (CV, Brier, bootstrap OOB AUC) from the BLR outputs. |
| `run_figures.py` | Runs the figure scripts in `prepare_datasets/`. |
| `models/` | `BLR.py` (Bayesian linear regression) and `model_metrics.py` (MSLL, explained variance). |
| `harmonize/` | ReComBat harmonization pipeline. |
| `Functions/` | Helper functions used by the harmonization pipeline. |
| `FOOOF_Analysis/` | Spectral parameterization and IAF extraction. |
| `prepare_datasets/` | Dataset-specific ingestion, feature concatenation, figures and tables. |
| `run_config_params/` | Dataset definitions (`DATASET.py`) and paths (`paths.py`, `config_local*.py`). |
| `archive/` | Exploratory notebooks and superseded scripts, kept for transparency (see `archive/README.md`). |

## Requirements

Two Python environments are needed:

- **Analysis** (Python 3.11): `pip install -r requirements.txt`
- **Harmonization** (Python **< 3.10**, required by the `reCombat` library):
  `pip install -r requirements-harmonize.txt`. Versions are not pinned in that file yet.

## Configuration

Machine-specific paths are not stored in the repository. Create the local config:

```bash
cp run_config_params/config_local.example.py run_config_params/config_local.py
```

and set:

- `RESULTS_ROOT` — folder with the processed outputs (the `AIF_Babiloni` folder).
  All analysis scripts derive their input and output paths from it (`run_config_params/paths.py`).
- `EEG_ROOT`, `DEMOGRAPHIC_DIR` — raw recordings and demographic files, used only by
  the dataset ingestion scripts in `prepare_datasets/`.

`config_local.py` is git-ignored.

## Data availability

The individual EEG recordings and participant-level clinical data are not publicly
available because of participant consent and ethics restrictions of the contributing
centers. Derived, de-identified data needed to rerun the analysis can be requested from
the corresponding author, subject to data-sharing agreements.

## Reproduction order

Run from the repository root.

1. Preprocessing and features: `filter_spectrals.ipynb`, `FOOOF_Analysis/FOOOF.ipynb`,
   `FOOOF_Analysis/FOOOF_24_scorEpoch.ipynb`, and the dataset scripts in `prepare_datasets/`.
2. Harmonization (harmonization environment): `python harmonize/pipeline_harmonize.py`.
3. Normative modeling: `python testblr.py`.
4. Classification: `python classify_blr.py`.
5. Figures: `python run_figures.py` (Figs. 2–3 of the manuscript), then
   `python prepare_datasets/fig1_normative_model.py` and
   `python prepare_datasets/extract_hc_ad_mci_stats.py`.
6. Tables: `prepare_datasets/table_classification.tex` and
   `prepare_datasets/table_demographics.tex` are the LaTeX tables used in the manuscript.

Figures are written to `<RESULTS_ROOT>/results_harmonize/recombat/BLR_paper/bands_age/harmonized/figures_paper/`.

## Known limitations of the code

- `prepare_datasets/concat_features_by_channel.py`, `FOOOF_Analysis/IAF.py` and
  `Functions/graphs_functions.py` still contain absolute paths from the original workstation;
  adapt them to your `RESULTS_ROOT` / raw-data folders before running.
- Steps 1 and 2 are partially notebook-based and depend on raw data that is not distributed.

## License

MIT — see [LICENSE](LICENSE).

## Citation

_To be added on publication (a `CITATION.cff` will be provided with the DOI)._
