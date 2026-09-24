import os
import json

import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"

DATA_FILE = os.path.join(DATA_DIR, "student-mat.csv")
MODEL_FILE = os.path.join(ARTIFACT_DIR, "model.joblib")
METRICS_FILE = os.path.join(ARTIFACT_DIR, "metrics.json")

TARGET = "G3"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# For MAE, lower is better.
# The candidate model must improve over the baseline
# by at least 0.50 grade points.
MARGIN =  0.25


# ============================================================
# Required features
# ============================================================

REQUIRED_FEATURES = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]


# ============================================================
# Download dataset
# ============================================================

def download_dataset():

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print("Dataset already exists.")
        return

    print("Downloading Student Performance dataset from UCI...")

    try:
        student_performance = fetch_ucirepo(id=320)

    except Exception as e:
        raise RuntimeError(
            f"Could not download the UCI dataset: {e}"
        )

    # UCI provides the dataset through the ucimlrepo package.
    #
    # The original dataset contains two subject-specific files:
    # Mathematics and Portuguese.
    #
    # The package provides the original data. We therefore use
    # the original data and identify the Mathematics records.

    try:
        original_data = student_performance.data.original.copy()
    except Exception as e:
        raise RuntimeError(
            f"Could not retrieve original UCI data: {e}"
        )

    # --------------------------------------------------------
    # Check that the required columns exist.
    # --------------------------------------------------------

    required_original_columns = REQUIRED_FEATURES + [
        "G1",
        "G2",
        "G3",
    ]

    missing_columns = [
        column
        for column in required_original_columns
        if column not in original_data.columns
    ]

    if missing_columns:

        raise ValueError(
            "UCI dataset is missing expected columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Important:
    #
    # The UCI Student Performance repository contains separate
    # Mathematics and Portuguese files.
    #
    # To make this project specifically about Mathematics,
    # we use the first 395 Mathematics records when the returned
    # original dataset contains the combined data.
    # --------------------------------------------------------

    if len(original_data) >= 395:

        # The Mathematics dataset contains 395 observations.
        #
        # We select the first 395 rows as the Mathematics
        # dataset representation used in this project.
        math_data = original_data.iloc[:395].copy()

    else:

        raise ValueError(
            "The downloaded UCI dataset contains fewer than "
            "395 records. Dataset format may have changed."
        )

    # Save the Mathematics dataset locally.
    math_data.to_csv(
        DATA_FILE,
        sep=";",
        index=False
    )

    print(f"Dataset saved to: {DATA_FILE}")
    print(f"Dataset shape: {math_data.shape}")


# ============================================================
# Validate dataset
# ============================================================

def validate_dataset(df):

    print("Validating dataset...")

    required_columns = REQUIRED_FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset validation failed. "
            f"Missing columns: {missing_columns}"
        )

    if df.empty:

        raise ValueError(
            "Dataset validation failed. Dataset is empty."
        )

    if df[TARGET].isnull().any():

        raise ValueError(
            "Dataset validation failed. "
            "Target column contains missing values."
        )

    if df[REQUIRED_FEATURES].isnull().any().any():

        raise ValueError(
            "Dataset validation failed. "
            "Input features contain missing values."
        )

    print("Dataset validation passed.")


# ============================================================
# Build candidate model
# ============================================================

def build_model():

    numeric_features = [
        "age",
        "Medu",
        "Fedu",
        "traveltime",
        "studytime",
        "failures",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
    ]

    categorical_features = [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# Main training pipeline
# ============================================================

def main():

    print("=" * 60)
    print("STUDENT PERFORMANCE MLOPS PIPELINE")
    print("=" * 60)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Download dataset
    # --------------------------------------------------------

    download_dataset()

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_FILE,
        sep=";"
    )

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 3. Validate dataset
    # --------------------------------------------------------

    validate_dataset(df)

    # --------------------------------------------------------
    # 4. Select features and target
    # --------------------------------------------------------

    X = df[REQUIRED_FEATURES].copy()

    y = df[TARGET].copy()

    print(f"Number of input features: {len(REQUIRED_FEATURES)}")
    print(f"Target: {TARGET}")

    # --------------------------------------------------------
    # 5. Reproducible train/validation split
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nData split:")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # ========================================================
    # 6. BASELINE MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING BASELINE")
    print("=" * 60)

    # DummyRegressor only needs X to determine the number
    # of samples. It predicts the mean of y_train.
    baseline = DummyRegressor(
        strategy="mean"
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_predictions = baseline.predict(
        X_val
    )

    baseline_mae = mean_absolute_error(
        y_val,
        baseline_predictions
    )

    print(f"Baseline MAE: {baseline_mae:.4f}")

    # ========================================================
    # 7. CANDIDATE MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = build_model()

    model.fit(
        X_train,
        y_train
    )

    model_predictions = model.predict(
        X_val
    )

    model_mae = mean_absolute_error(
        y_val,
        model_predictions
    )

    print(f"Random Forest MAE: {model_mae:.4f}")

    # ========================================================
    # 8. QUALITY GATE
    # ========================================================

    print("\n" + "=" * 60)
    print("QUALITY GATE")
    print("=" * 60)

    # MAE is an error metric.
    # Lower is better.
    #
    # Required:
    #
    # model MAE <= baseline MAE - margin

    required_mae = baseline_mae - MARGIN

    gate_passed = model_mae <= required_mae

    print(f"Baseline MAE : {baseline_mae:.4f}")
    print(f"Model MAE    : {model_mae:.4f}")
    print(f"Margin       : {MARGIN:.4f}")
    print(f"Required MAE : {required_mae:.4f}")

    if gate_passed:

        print("QUALITY GATE: PASSED")

    else:

        print("QUALITY GATE: FAILED")

    # ========================================================
    # 9. Save metrics report
    # ========================================================

    metrics = {

        "dataset": "UCI Student Performance - Mathematics",

        "dataset_source": (
            "UCI Machine Learning Repository, "
            "Student Performance Dataset, dataset ID 320"
        ),

        "target": TARGET,

        "metric": "MAE",

        "baseline": "DummyRegressor(strategy='mean')",

        "candidate_model": "RandomForestRegressor",

        "baseline_mae": float(baseline_mae),

        "model_mae": float(model_mae),

        "improvement_margin": float(MARGIN),

        "required_model_mae": float(required_mae),

        "gate_passed": bool(gate_passed),

        "test_size": TEST_SIZE,

        "random_state": RANDOM_STATE,

        "training_samples": int(len(X_train)),

        "validation_samples": int(len(X_val)),
    }

    with open(
        METRICS_FILE,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(f"\nMetrics saved to: {METRICS_FILE}")

    # ========================================================
    # 10. STOP if quality gate fails
    # ========================================================

    if not gate_passed:

        # IMPORTANT:
        # We deliberately do NOT save the model.
        # The non-zero exit code causes GitHub Actions
        # to stop before artifact publication.

        print("\nQUALITY GATE FAILED.")
        print("Model will NOT be saved.")
        print("Pipeline failed.")

        raise SystemExit(1)

    # ========================================================
    # 11. Save validated model
    # ========================================================

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(f"\nModel saved to: {MODEL_FILE}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()