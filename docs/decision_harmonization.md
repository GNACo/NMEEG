# Decision record: site harmonization

Date: 2026-10-03 (decided before re-running any analysis)

## Decision

Replace the current ReComBat harmonization (fit on all participants, with
diagnostic group as a protected covariate) with neuroHarmonize (ComBat-GAM):

- Learn the site-correction parameters **only on the training HC participants**
  (the same global HC training split used by the normative model).
- Apply the learned parameters to the held-out HC test set and to all clinical
  groups (MCI, AD, ACr). Diagnosis is not a covariate.
- Covariates: site (batch) and age, with age modeled non-linearly (`smooth_terms=['age']`).

ReComBat results are kept as a sensitivity analysis in the supplementary material.

## Reasons

1. **Leakage.** The current fit uses patients' data to estimate site corrections,
   and the same diagnosis labels are later classified. Learning on HC only removes
   this dependence.
2. **Confounding.** ACr come only from the Medellin sites and MCI/AD mostly from
   Seoul and Spain. With group as a protected covariate, site correction for those
   groups rests on very few sites.
3. **Non-linear age.** Site age ranges differ strongly. A linear age term would
   attribute part of the true age effect to site and remove it, which the normative
   model then tries to estimate. ComBat-GAM avoids this.
4. **Validated API.** neuroHarmonize provides `harmonizationLearn` and
   `harmonizationApply`, a published, standard method in normative modeling. It
   avoids a custom, unvalidated apply step.

## Consequences

- Methods must be rewritten (harmonization method, covariates, fitting set, and
  removal of the claim that elastic-net regularization is an advantage here,
  because the design is small).
- All downstream results change: harmonized features, normative model,
  classification, figures, and tables.
- Check the number of HC training participants per site after the split. Chile (13)
  and Spain (17) are small. The empirical Bayes estimator tolerates this, but it
  must be reported.

## Other fixes applied in the same re-run

- Outlier (IQR) filter computed on training HC only and never applied to the
  evaluation sets.
- Classification pivot indexed by (subject, Site), so codes reused across sites
  are not merged.
- Complete-case analysis instead of median imputation.
