
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train():
    # 1. Loading public dataset
    dataset = load_diabetes(as_frame=True)
    X = dataset.frame.drop(columns=["target"])
    y = dataset.frame["target"]

    # 2. Splitting data reproducibly with seed 42
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)

    # 3. Put preprocessing and model into one pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])

    # 4. Train the complete pipeline
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on test data
    predictions = pipeline.predict(X_val)
    rmse = float(mean_squared_error(y_val, predictions) ** 0.5)

    # Track the experiment locally
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes_triage")

    with mlflow.start_run(run_name="v0.1_baseline"):
        mlflow.log_params({
            "model_type": "LinearRegression",
            "scaler": "StandardScaler",
            "random_state": 42,
            "test_size": 0.2,
        })

        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
        )

    # 6. Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # 7. Save evaluation metrics
    metrics = {
        "model_version": "v0.1",
        "model_type": "LinearRegression",
        "rmse": rmse,
        "random_state": 42,
        "test_size": 0.2,
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train()