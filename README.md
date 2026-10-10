# Virtual Diabetes Clinic Triage

A small MLOps project that demonstrates how to train, evaluate, package, test, and deploy a machine learning model through an API and Docker. The project uses the diabetes dataset provided by scikit-learn.

> **Important:** This is an educational machine learning project, not a clinical decision-support system. Its predictions must not be used to diagnose diabetes, assess an individual's medical risk, or guide treatment.

## 1. Project Overview

The goal is to build a reproducible machine learning workflow that exposes a trained model through a REST API.

The project is developed incrementally:

- **v0.1 — Baseline:** StandardScaler and Linear Regression, with a trained model, evaluation metrics, FastAPI endpoints, automated tests, and Docker packaging.
- **v0.2 — Model improvement:** Evaluate an alternative model, such as Ridge Regression, and compare its performance with the baseline.
- **CI/CD:** Use GitHub Actions to run code-quality checks, tests, and a training smoke test. A version tag triggers the Docker release workflow.

### Baseline

| Item | Value |
|---|---|
| Model version | v0.1 |
| Algorithm | Linear Regression |
| Preprocessing | StandardScaler |
| Validation strategy | 80% training / 20% validation |
| Random seed | 42 |
| Validation RMSE | 53.8534 |
| API framework | FastAPI |
| Container | Docker |
| Experiment tracking | MLflow |

The reported RMSE is the result of the current fixed train/validation split. It is a baseline for comparison, not evidence of clinical accuracy.

## 2. Repository Structure

The project lives inside the `diabetes-triage/` directory of the `MLOps` GitHub repository.

```text
MLOps/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
└── diabetes-triage/
    ├── app/
    │   └── main.py
    │
    ├── src/
    │   └── train.py
    │
    ├── models/
    │   ├── model.joblib
    │   └── metrics.json
    │
    ├── tests/
    │   └── test_api.py
    │
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    ├── .gitignore
    ├── README.md
    └── CHANGELOG.md
```

### What each component does

| Component | Responsibility |
|---|---|
| `src/train.py` | Loads the dataset, splits the data, trains the pipeline, evaluates RMSE, saves the model and metrics, and logs the run to MLflow. |
| `app/main.py` | Loads the saved model and exposes the prediction and health endpoints. |
| `models/model.joblib` | Serialized trained scikit-learn pipeline, including preprocessing and regression. |
| `models/metrics.json` | Records the model version, model type, RMSE, random seed, and validation split size. |
| `tests/test_api.py` | Tests API health, valid predictions, missing features, unexpected features, and finite predictions. |
| `requirements.txt` | Defines the Python dependencies and their pinned versions. |
| `Dockerfile` | Packages the application, dependencies, and trained model into a container. |
| `.github/workflows/ci.yml` | Runs linting, automated tests, training, and metric artifact upload. |
| `.github/workflows/release.yml` | Builds and smoke-tests a versioned Docker image, publishes it to GHCR, and creates a GitHub Release. |
| `mlruns/` | Local MLflow tracking data generated during training; excluded from Git. |

## 3. Dataset and Prediction Target

The project uses `load_diabetes(as_frame=True)` from scikit-learn.

```python
from sklearn.datasets import load_diabetes

Xy = load_diabetes(as_frame=True)

X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]
```

The dataset provides ten numerical features:

`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, and `s6`.

The target is a quantitative measure of disease progression used in the dataset. It is **not a diabetes diagnosis or a clinically validated individual risk score**.

The feature values are already normalized by the dataset's preprocessing conventions. The API therefore expects the dataset's feature representation, not raw values such as age in years or BMI in its usual units.

### Evaluation metric

The baseline uses Root Mean Squared Error (RMSE):

\[
\mathrm{RMSE} =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE measures the typical prediction error while penalizing larger errors more heavily. Lower values indicate better predictions on the same evaluation data.

## 4. Prerequisites

Install the following before running the project:

- Python 3.11
- Git
- Docker Desktop or another Docker-compatible environment
- A code editor such as Visual Studio Code

Python 3.11 is recommended because it matches the project's Docker base image and dependency environment.

## 5. Local Setup

Open a terminal at the `MLOps` repository root, then enter the project directory:

```bash
cd diabetes-triage
```

### Create and activate a virtual environment

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verify the Python version:

```bash
python --version
```

It should report Python 3.11.x.

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependencies include scikit-learn, pandas, NumPy, FastAPI, Uvicorn, MLflow, joblib, pytest, HTTPX, and Ruff.

## 6. Train the Baseline Model

Run this command from inside `diabetes-triage/`:

```bash
python src/train.py
```

The script:

1. Loads the scikit-learn diabetes dataset.
2. Separates the features and target.
3. Creates a reproducible 80/20 train/validation split using random seed 42.
4. Fits a pipeline containing `StandardScaler` and `LinearRegression`.
5. Evaluates predictions on the validation set.
6. Saves the fitted pipeline to `models/model.joblib`.
7. Saves evaluation information to `models/metrics.json`.
8. Logs the parameters, RMSE, and model to the local MLflow tracking store.

Training should print the validation RMSE and the paths of the saved artifacts.

### View experiments in MLflow

Start the MLflow UI in a second terminal from the project directory, with the virtual environment activated:

```bash
mlflow ui \
  --backend-store-uri ./mlruns \
  --host 127.0.0.1 \
  --port 5002
```

Open:

http://localhost:5002

Select the `diabetes_triage` experiment and inspect the `v0.1_baseline` run.

Stop the UI using `Ctrl+C` when finished.

MLflow tracking is currently local. Each machine or CI runner may have its own tracking data unless a shared tracking server is configured.

## 7. Run the API Locally

The model must exist before the API starts. Run the training command first if `models/model.joblib` is missing.

From `diabetes-triage/`, start the API:

```bash
uvicorn app.main:app --reload
```

The API runs at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### Health endpoint

**Request**

```http
GET /health
```

**Example response**

```json
{
  "status": "ok",
  "model_version": "v0.1"
}
```

This endpoint confirms that the API is running and reports the configured model version. It does not independently validate the model's predictive quality.

### Prediction endpoint

**Request**

```http
POST /predict
Content-Type: application/json
```

**Example request body**

```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
```

These are illustrative feature values in the dataset's normalized representation, not raw clinical measurements.

**Example response structure**

```json
{
  "prediction": 150.0
}
```

The number above illustrates the response format only; the actual prediction depends on the trained model and input values.

### Input validation

The API expects all ten features. It returns HTTP 422 for missing features, invalid input types, or unexpected extra fields.

The prediction is a regression output, not a probability. The current version does not define a high-risk threshold or a clinical triage category.

## 8. Run Automated Tests

From `diabetes-triage/`:

```bash
python -m pytest -v
```

The API tests cover:

- Health endpoint returns HTTP 200 and the expected model version.
- Valid prediction requests return a numerical prediction.
- Missing features return HTTP 422.
- Unexpected extra features return HTTP 422.
- Predictions are finite.

### Run lint checks

```bash
ruff check app src tests
```

Both commands should pass before changes are committed.

## 9. Run with Docker

The Docker image includes the Python dependencies, application code, and saved model. It does not require a separately running MLflow server to serve predictions.

Run these commands from the **`MLOps` repository root**.

### Build the image

```bash
docker build \
  -f diabetes-triage/Dockerfile \
  -t diabetes-triage:v0.1 \
  diabetes-triage
```

### Start the container

```bash
docker run --rm \
  -p 8000:80 \
  --name diabetes-triage \
  diabetes-triage:v0.1
```

The application is exposed on port 8000 of the host and port 80 inside the container.

Open:

http://localhost:8000/docs

Test the health endpoint:

```bash
curl http://localhost:8000/health
```

Stop the container with `Ctrl+C`.

If port 8000 is already occupied, choose another host port, for example `-p 8080:80`.

## 10. GitHub Actions and CI/CD

The workflows are stored at the repository root, under `MLOps/.github/workflows/`, because `MLOps` is the GitHub repository root.

### Continuous integration

`ci.yml` runs on pushes to `main` and pull requests targeting `main`.

It:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs the pinned dependencies.
4. Runs Ruff.
5. Runs pytest.
6. Executes the training script.
7. Uploads `models/metrics.json` as a workflow artifact.

The workflow executes project commands from `diabetes-triage/`.

To inspect the results, open the repository's **Actions** tab and select the relevant run.

### Versioned release

`release.yml` is triggered by a pushed version tag matching `v*`.

The intended release process is:

1. Build the Docker image from the project directory.
2. Start a container and smoke-test `/health` and `/predict`.
3. Confirm that missing-feature validation returns HTTP 422.
4. Publish versioned and `latest` images to GitHub Container Registry (GHCR).
5. Create a GitHub Release with generated release notes.

The release workflow must pass before the published image and release can be considered available.

## 11. Versioning and Model Improvement

The project uses explicit version labels to distinguish model iterations.

### v0.1 — Baseline

- StandardScaler + LinearRegression.
- Fixed train/validation split.
- RMSE evaluation and local MLflow tracking.
- FastAPI prediction service.
- Automated API tests.
- Docker packaging.

### v0.2 — Model improvement

The next iteration will evaluate an alternative regression model, initially Ridge Regression.

The comparison should use the same dataset split and random seed as v0.1. The new version should:

- Record the model configuration and RMSE.
- Compare v0.2 against the baseline.
- Explain the observed metric change in `CHANGELOG.md`.
- Update the model version consistently in the training metadata and API.
- Pass the API tests and CI checks.
- Be released only after its artifacts and Docker image have been verified.

A new model should not automatically replace the baseline merely because it is more complex. The measured evaluation results should guide the decision.

## 12. Reproducibility Notes

The project uses pinned dependencies and a fixed random seed for the baseline split. Training metrics are saved to JSON and logged to MLflow.

For reliable comparisons:

- Use the same dataset, split, seed, and metric when comparing model versions.
- Keep preprocessing inside the saved scikit-learn pipeline.
- Record relevant model parameters and evaluation results.
- Review dependency changes before updating the pinned versions.
- Test the API and container after changing model artifacts.

The current validation result is a single holdout evaluation. More extensive validation, such as cross-validation and analysis of prediction errors, may be useful before drawing stronger conclusions.

## 13. Limitations and Responsible Use

This project is designed to demonstrate software engineering and MLOps practices, not to provide a clinically deployable diabetes service.

Important limitations include:

- The target is a dataset-specific disease-progression measure, not a diagnostic label.
- The features are normalized dataset values and are not accepted as raw clinical measurements.
- RMSE alone does not establish clinical usefulness, safety, or fairness.
- The model has not been clinically validated.
- The API does not provide medical advice or a validated triage classification.

Any future real-world clinical application would require appropriate clinical validation, data governance, privacy and security controls, and expert oversight.

## 14. Further Information

- [scikit-learn diabetes dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [Docker documentation](https://docs.docker.com/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [GitHub Container Registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)


