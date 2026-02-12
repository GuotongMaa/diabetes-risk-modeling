# Diabetes Risk Modeling

## Objective
Predict type-2 diabetes risk from clinical features and explain the main risk drivers.

## Method
- Tabular preprocessing and baseline classifiers.
- Threshold analysis with ROC-AUC and PR-AUC.
- Feature-importance analysis for interpretability.

## Repository Structure
- `notebooks/` modeling and interpretation workflows.
- `src/` reusable code modules.
- `results/` metrics and plots.
- `assets/` README figures.
- `models/` saved models (optional).
- `data/` local dataset directory (not versioned).

## Data Access
Pima Indians Diabetes dataset.

## Run
1. Put `diabetes.csv` under `data/`.
2. Run notebooks in `notebooks/`.
3. Export final metrics/plots to `results/`.

## Result Artifacts
- ROC and PR curves
- Calibration/threshold plots
- Feature-importance summary

## Validated Baseline Run
- `accuracy`: `0.7143`
- `f1`: `0.5600`
- `roc_auc`: `0.8230`
- `pr_auc`: `0.6933`
- metrics file: `results/metrics_diabetes_baseline.json`
- model file: `results/artifacts/diabetes_logreg.joblib`
