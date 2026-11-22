"""Compare calibrated shot make-probability baselines on a season holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "PERIOD",
    "SECONDS_REMAINING_IN_PERIOD",
]
CATEGORICAL_FEATURES = [
    "SHOT_TYPE",
    "ACTION_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
]
TARGET = "SHOT_MADE_FLAG"
REQUIRED_COLUMNS = set(
    [
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "SEASON",
        TARGET,
    ]
    + CATEGORICAL_FEATURES
)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/shots.csv"))
    parser.add_argument("--validation-season", type=int, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise SystemExit(f"Input data is missing required columns: {sorted(missing)}")

    # Convert within-period clock fields to a single remaining-time feature.
    data["SECONDS_REMAINING_IN_PERIOD"] = (
        pd.to_numeric(data["MINUTES_REMAINING"], errors="coerce") * 60
        + pd.to_numeric(data["SECONDS_REMAINING"], errors="coerce")
    )
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET, "SEASON"])
    data = data[data[TARGET].isin([0, 1])].copy()

    train = data[data["SEASON"] < args.validation_season]
    valid = data[data["SEASON"] == args.validation_season]
    if train.empty or valid.empty:
        raise SystemExit("Need earlier seasons for training and the requested holdout season")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train[features], train[TARGET].astype(int)
    X_valid, y_valid = valid[features], valid[TARGET].astype(int)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=30,
            max_features="sqrt",
            n_jobs=-1,
            random_state=17,
        ),
    }
    results = {}
    predictions = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocess", make_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        predictions[name] = probabilities
        results[name] = {
            "log_loss": float(log_loss(y_valid, probabilities, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_valid, probabilities)),
        }

    prevalence = float(y_train.mean())
    baseline = [prevalence] * len(y_valid)
    results["training_prevalence_baseline"] = {
        "log_loss": float(log_loss(y_valid, baseline, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_valid, baseline)),
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "validation_season": args.validation_season,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "training_make_rate": prevalence,
        "models": results,
    }
    report_path = args.reports_dir / f"metrics_{args.validation_season}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for name, probabilities in predictions.items():
        observed, predicted = calibration_curve(
            y_valid, probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", label=name.replace("_", " "))
    ax.set(xlabel="Mean predicted make probability", ylabel="Observed make rate")
    ax.set_title(f"Calibration on season {args.validation_season}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.reports_dir / f"calibration_{args.validation_season}.png", dpi=160)
    plt.close(fig)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
