import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
BASELINE_METRICS_PATH = MODEL_DIR / "metrics_v0.1.json"

MODEL_VERSION = "v0.2"
RANDOM_STATE = 42
TEST_SIZE = 0.2
RIDGE_ALPHA = 1.0


def train():
    # 1. Load the same public dataset used by the v0.1 baseline.
    dataset = load_diabetes(as_frame=True)
    X = dataset.frame.drop(columns=["target"])
    y = dataset.frame["target"]

    # 2. Keep the split identical so the model comparison is meaningful.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # 3. Use the same preprocessing with Ridge Regression as the v0.2 model.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=RIDGE_ALPHA)),
    ])

    # 4. Train the complete pipeline.
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on the same validation data used by v0.1.
    predictions = pipeline.predict(X_val)
    rmse = float(mean_squared_error(y_val, predictions) ** 0.5)

    baseline_metrics = json.loads(
        BASELINE_METRICS_PATH.read_text(encoding="utf-8")
    )
    baseline_rmse = float(baseline_metrics["rmse"])
    rmse_delta = rmse - baseline_rmse

    # Track the experiment locally.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes_triage")

    with mlflow.start_run(run_name="v0.2_ridge"):
        mlflow.log_params({
            "model_type": "Ridge",
            "alpha": RIDGE_ALPHA,
            "scaler": "StandardScaler",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        })
        mlflow.log_metrics({
            "rmse": rmse,
            "baseline_rmse": baseline_rmse,
            "rmse_delta": rmse_delta,
        })
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
        )

    # 6. Save the v0.2 pipeline as the current model artifact.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # 7. Save evaluation information and the comparison with v0.1.
    metrics = {
        "model_version": MODEL_VERSION,
        "model_type": "Ridge",
        "alpha": RIDGE_ALPHA,
        "rmse": rmse,
        "baseline_version": "v0.1",
        "baseline_rmse": baseline_rmse,
        "rmse_delta": rmse_delta,
        "better_than_baseline": rmse < baseline_rmse,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train()
