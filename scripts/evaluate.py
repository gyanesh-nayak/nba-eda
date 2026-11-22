"""Compare calibrated shot make-probability baselines on a season holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def plot_rate_table(table: pd.DataFrame, label: str, title: str, path: Path) -> None:
    table = table.sort_values("make_rate")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(table))))
    ax.barh(table[label].astype(str), table["make_rate"], color="#365f91")
    ax.set(xlabel="Observed make rate", title=title, xlim=(0, 1))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def readable_feature_name(name: str) -> str:
    name = name.removeprefix("numeric__").removeprefix("categorical__")
    for prefix, label in [
        ("SHOT_ZONE_BASIC_", "Zone: "),
        ("SHOT_ZONE_AREA_", "Area: "),
        ("SHOT_TYPE_", "Shot type: "),
        ("ACTION_TYPE_", "Action: "),
    ]:
        if name.startswith(prefix):
            return label + name.removeprefix(prefix)
    return {
        "SHOT_DISTANCE": "Shot distance",
        "LOC_X": "Court X location",
        "LOC_Y": "Court Y location",
        "PERIOD": "Quarter",
        "SECONDS_REMAINING_IN_PERIOD": "Seconds left in period",
    }.get(name, name)


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
    data["SHOT_DISTANCE"] = pd.to_numeric(data["SHOT_DISTANCE"], errors="coerce")
    data = data.dropna(subset=[TARGET, "SEASON"])
    data = data[data[TARGET].isin([0, 1])].copy()
    data["DISTANCE_BAND"] = pd.cut(
        data["SHOT_DISTANCE"],
        bins=[0, 5, 10, 15, 20, 25, float("inf")],
        labels=["0-5 ft", "5-10 ft", "10-15 ft", "15-20 ft", "20-25 ft", "25+ ft"],
        include_lowest=True,
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    distance_rates = (
        data.groupby("DISTANCE_BAND", observed=True)[TARGET]
        .agg(shots="size", makes="sum", make_rate="mean")
        .reset_index()
    )
    type_rates = (
        data.groupby("SHOT_TYPE", dropna=False)[TARGET]
        .agg(shots="size", makes="sum", make_rate="mean")
        .reset_index()
    )
    zone_rates = (
        data.groupby("SHOT_ZONE_BASIC", dropna=False)[TARGET]
        .agg(shots="size", makes="sum", make_rate="mean")
        .reset_index()
    )
    action_rates = (
        data.groupby("ACTION_TYPE", dropna=False)[TARGET]
        .agg(shots="size", makes="sum", make_rate="mean")
        .reset_index()
    )
    action_rates = action_rates.sort_values("shots", ascending=False).head(15)

    plot_rate_table(
        distance_rates.rename(columns={"DISTANCE_BAND": "label"}),
        "label",
        "Observed make rate by shot distance",
        args.reports_dir / "make_rate_by_distance.png",
    )
    plot_rate_table(
        type_rates.rename(columns={"SHOT_TYPE": "label"}),
        "label",
        "Observed make rate by shot type",
        args.reports_dir / "make_rate_by_shot_type.png",
    )
    plot_rate_table(
        zone_rates.rename(columns={"SHOT_ZONE_BASIC": "label"}),
        "label",
        "Observed make rate by court zone",
        args.reports_dir / "make_rate_by_zone.png",
    )
    plot_rate_table(
        action_rates.rename(columns={"ACTION_TYPE": "label"}),
        "label",
        "Observed make rate for the 15 most common shot actions",
        args.reports_dir / "make_rate_by_action.png",
    )

    season_table = (
        data.groupby("SEASON")[TARGET]
        .agg(shots="size", makes="sum", make_rate="mean")
        .reset_index()
    )
    missing_summary = (
        data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        .isna()
        .sum()
        .astype(int)
        .to_dict()
    )

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
    fitted_models = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocess", make_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
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

    forest = fitted_models["random_forest"]
    encoded_names = forest.named_steps["preprocess"].get_feature_names_out()
    forest_importance = pd.DataFrame(
        {
            "feature": encoded_names,
            "importance": forest.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    top_importance = forest_importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [readable_feature_name(name) for name in top_importance["feature"]],
        top_importance["importance"],
        color="#7895b2",
    )
    ax.set(xlabel="Random forest impurity importance", title="Top 15 model features")
    fig.tight_layout()
    fig.savefig(args.reports_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    def rate_records(table: pd.DataFrame, label: str) -> list[dict]:
        records = []
        for _, row in table.iterrows():
            records.append(
                {
                    label: "Missing" if pd.isna(row[label]) else str(row[label]),
                    "shots": int(row["shots"]),
                    "makes": int(row["makes"]),
                    "make_rate": float(row["make_rate"]),
                }
            )
        return records

    report = {
        "data_summary": {
            "rows": int(len(data)),
            "overall_make_rate": float(data[TARGET].mean()),
            "season_summary": [
                {
                    "season": int(row["SEASON"]),
                    "shots": int(row["shots"]),
                    "makes": int(row["makes"]),
                    "make_rate": float(row["make_rate"]),
                }
                for _, row in season_table.iterrows()
            ],
            "missing_feature_values": {
                str(key): int(value) for key, value in missing_summary.items()
            },
            "make_rate_by_distance": rate_records(distance_rates, "DISTANCE_BAND"),
            "make_rate_by_shot_type": rate_records(type_rates, "SHOT_TYPE"),
            "make_rate_by_zone": rate_records(zone_rates, "SHOT_ZONE_BASIC"),
            "make_rate_by_common_action": rate_records(action_rates, "ACTION_TYPE"),
        },
        "validation_season": args.validation_season,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "training_make_rate": prevalence,
        "models": results,
        "random_forest_top_features": [
            {"feature": str(row["feature"]), "importance": float(row["importance"])}
            for _, row in forest_importance.head(15).iterrows()
        ],
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
