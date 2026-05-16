import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ── Configuration ──────────────────────────────────────────────────

# These are the feature columns IN THE EXACT SAME ORDER as to_feature_vector()
# WARNING: If you change this order, you MUST retrain the model
FEATURE_COLS = [
    "distance_km",
    "cargo_weight_kg",
    "is_rush_hour",
    "hour_of_day",
    "day_of_week",
    "num_stops",
    "traffic_index",
    "vehicle_truck",
    "vehicle_van",
    "vehicle_motorcycle",
]
TARGET_COL = "eta_minutes"
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "eta-predictor-day1"


# ── Data Loading & Feature Engineering ────────────────────────────


def load_and_prepare(csv_path: str):
    """
    Load raw CSV and engineer features.
    Returns: X_train, X_val, y_train, y_val as numpy arrays.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Feature engineering — add derived columns
    # Rush hour flag
    rush = list(range(7, 10)) + list(range(17, 20))
    df["is_rush_hour"] = df["hour_of_day"].isin(rush).astype(float)

    # One-hot encode vehicle type (if column exists in data)
    if "vehicle_type" in df.columns:
        df["vehicle_truck"] = (df["vehicle_type"] == "truck").astype(float)
        df["vehicle_van"] = (df["vehicle_type"] == "van").astype(float)
        df["vehicle_motorcycle"] = (df["vehicle_type"] == "motorcycle").astype(float)
    else:
        # Default: all vehicles are trucks
        df["vehicle_truck"] = 1.0
        df["vehicle_van"] = 0.0
        df["vehicle_motorcycle"] = 0.0

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0  # Default for missing one-hot columns

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples")
    return X_train, X_val, y_train, y_val


# ── Visualisation ─────────────────────────────────────────────────


def plot_feature_importance(model_pipeline, feature_names: list, save_path: str):
    """Create and save a feature importance bar chart."""
    regressor = model_pipeline.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        return None  # Linear models don't have feature_importances_

    importances = regressor.feature_importances_
    # Sort by importance (highest first)
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx], importances[sorted_idx], color="#1565C0"
    )
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("Which features matter most for predicting ETA?", fontsize=12)
    ax.invert_yaxis()  # Most important at top
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_actual_vs_predicted(y_val, y_pred, save_path: str, mae: float):
    """Plot actual vs predicted ETA values."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_val, y_pred, alpha=0.3, s=8, color="#1565C0")
    # Perfect prediction line
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfectprediction"
    )
    ax.set_xlabel("Actual ETA (minutes)", fontsize=11)
    ax.set_ylabel("Predicted ETA (minutes)", fontsize=11)
    ax.set_title(f"Actual vs Predicted ETA  (MAE = {mae:.1f} min)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ── Main Training Function ─────────────────────────────────────────


def train_and_log(
    run_name: str,
    model_type: str,
    hyperparams: dict,
    data_path: str,
    save_model: bool = False,
) -> dict:
    """
        Train a model with given hyperparameters and log everything to MLflow.

        Args:
            run_name:     Name for this run in MLflow UI
            model_type:   'gbr' (Gradient Boosting), 'rf' (Random Forest), 'ridge'
    (Linear)
            hyperparams:  Dictionary of model hyperparameters
            data_path:    Path to the CSV training data
            save_model:   If True, save the model to models/ directory

        Returns:
            Dictionary of metrics for this run
    """
    # Connect to MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and prepare data
    X_train, X_val, y_train, y_val = load_and_prepare(data_path)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n{'=' * 60}")
        print(f"Run: {run_name}")
        print(f"MLflow Run ID: {run.info.run_id}")

        # ── 1. Tag this run ──────────────────────────────────────
        mlflow.set_tags(
            {
                "model_type": model_type,
                "dataset": "logistics-ghana-v1",
                "num_features": len(FEATURE_COLS),
                "features": ",".join(FEATURE_COLS),
            }
        )

        # ── 2. Log all hyperparameters ────────────────────────────
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_params(hyperparams)  # Log all hyperparams at once

        # ── 3. Build model pipeline ───────────────────────────────
        # Pipeline: StandardScaler → Model
        # StandardScaler: normalises features to mean=0, std=1
        # This is important for Ridge (linear) models
        if model_type == "gbr":
            regressor = GradientBoostingRegressor(**hyperparams, random_state=42)
        elif model_type == "rf":
            regressor = RandomForestRegressor(**hyperparams, random_state=42)
        elif model_type == "ridge":
            regressor = Ridge(**hyperparams)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )

        # ── 4. Train ──────────────────────────────────────────────
        print("Training...")
        pipeline.fit(X_train, y_train)

        # ── 5. Cross-validation (5 folds) ─────────────────────────
        # Cross-validation gives a more reliable estimate of model performance
        # than a single train/val split
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,  # Use all CPU cores
        )
        cv_mae = -cv_scores.mean()  # Negate because sklearn uses negative MAE
        cv_std = cv_scores.std()
        mlflow.log_metric("cv_mae_mean", round(cv_mae, 4))
        mlflow.log_metric("cv_mae_std", round(cv_std, 4))

        # ── 6. Validation metrics ──────────────────────────────────
        y_pred = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100

        metrics = {
            "val_mae": round(mae, 4),
            "val_rmse": round(rmse, 4),
            "val_r2": round(r2, 4),
            "val_mape": round(mape, 4),
        }
        mlflow.log_metrics(metrics)

        print(f"  val_mae:  {mae:.2f} minutes")
        print(f"  val_rmse: {rmse:.2f} minutes")
        print(f"  val_r2:   {r2:.4f}")
        print(f"  cv_mae:   {cv_mae:.2f} ± {cv_std:.2f} minutes")

        # ── 7. Save plots as MLflow artifacts ──────────────────────
        Path("mlflow_plots").mkdir(parents=True, exist_ok=True)

        fi_path = plot_feature_importance(
            pipeline, FEATURE_COLS, "mlflow_plots/feature_importance.png"
        )
        if fi_path:
            mlflow.log_artifact(fi_path, "plots")

        avp_path = plot_actual_vs_predicted(
            y_val,
            y_pred,
            "mlflow_plots/actual_vs_predicted.png",
            mae,
        )
        mlflow.log_artifact(avp_path, "plots")

        # ── 8. Log the trained model ───────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="eta-predictor",
            # input_example shows what data looks like (shown in MLflow UI)
            input_example=X_train[:3].tolist(),
        )

        # ── 9. Optionally save model to disk for serving ──────────
        if save_model:
            Path("models").mkdir(exist_ok=True)
            model_path = f"models/eta_model_{run.info.run_id[:8]}.joblib"
            joblib.dump(pipeline, model_path)
            # Also save as 'latest' for the API to load
            joblib.dump(pipeline, "models/eta_model_latest.joblib")
            print(f"  Saved model to {model_path}")
            # Log the path as a param so we can find it later
            mlflow.log_param("saved_model_path", model_path)

            # Save metrics to JSON for DVC tracking
            Path("metrics").mkdir(exist_ok=True)
            with open("metrics/scores.json", "w") as f:
                json.dump(
                    {
                        "val_mae": metrics["val_mae"],
                        "val_rmse": metrics["val_rmse"],
                        "val_r2": metrics["val_r2"],
                    },
                    f,
                    indent=2,
                )
            print("  Metrics saved to metrics/scores.json")

        print("MLflow UI: http://localhost:5000")
        return metrics


# ── Run Multiple Experiments ───────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "data/raw/logistics_eta.csv"

    print("Starting training experiments...")
    print("View live at: http://localhost:5000")

    # Experiment 1: Gradient Boosting — baseline
    train_and_log(
        run_name="gbr-baseline",
        model_type="gbr",
        hyperparams={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        data_path=DATA_PATH,
        save_model=True,
    )

    # Experiment 2: More trees — does it improve?
    train_and_log(
        run_name="gbr-more-trees",
        model_type="gbr",
        hyperparams={"n_estimators": 500, "learning_rate": 0.03, "max_depth": 4},
        data_path=DATA_PATH,
    )

    # Experiment 3: Deeper trees
    train_and_log(
        run_name="gbr-deeper",
        model_type="gbr",
        hyperparams={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
        data_path=DATA_PATH,
    )

    # Experiment 4: Random Forest comparison
    train_and_log(
        run_name="rf-comparison",
        model_type="rf",
        hyperparams={"n_estimators": 200, "max_depth": 10},
        data_path=DATA_PATH,
    )

    # Experiment 5: Linear baseline (should perform worst)
    train_and_log(
        run_name="ridge-baseline",
        model_type="ridge",
        hyperparams={"alpha": 1.0},
        data_path=DATA_PATH,
    )

    print("\n=== All experiments complete ===")
    print("Open http://localhost:5000 to compare results")
    print("Look for the run with the LOWEST val_mae — that is your best model")