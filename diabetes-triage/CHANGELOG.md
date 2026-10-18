## v0.2 - Ridge Regression

- Added `src/train_v0_2.py` using Ridge Regression with `alpha=1.0`.
- Kept the v0.1 dataset, 80/20 validation split, random seed 42, preprocessing, and RMSE metric unchanged.
- Preserved the v0.1 model and metrics as `models/model_v0.1.joblib` and `models/metrics_v0.1.json`.
- Promoted the v0.2 model to `models/model.joblib` and updated `models/metrics.json` with the v0.2 evaluation and baseline comparison.
- Updated the API health response to report model version `v0.2`.

### Evaluation comparison

| Version | Model | Validation RMSE |
|---|---|---:|
| v0.1 | StandardScaler + LinearRegression | 53.8534 |
| v0.2 | StandardScaler + Ridge (`alpha=1.0`) | 53.7775 |

The v0.2 model improves the validation RMSE by approximately `0.0760` points while using the same dataset, split, seed, preprocessing, and evaluation metric. Ridge is therefore a measurable improvement over the v0.1 baseline on this held-out validation set. This result is limited to the assignment dataset and does not establish clinical usefulness.

## v0.1 - Baseline

- Added the StandardScaler and LinearRegression baseline.
- Added the FastAPI prediction service, API tests, Docker packaging, and MLflow tracking.
