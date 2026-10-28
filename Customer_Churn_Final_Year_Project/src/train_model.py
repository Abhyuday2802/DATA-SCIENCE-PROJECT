"""Train and evaluate a reproducible customer churn classification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
TARGET = "Churn"
IDENTIFIER = "customerID"


def load_and_clean_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Telco CSV, normalize its target, and handle blank TotalCharges."""
    data = pd.read_csv(csv_path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected a '{TARGET}' target column in {csv_path}.")

    data = data.drop(columns=[IDENTIFIER], errors="ignore").copy()
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        # Blank TotalCharges represent new customers with tenure 0 in this dataset.
        data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

    target = data.pop(TARGET)
    if not pd.api.types.is_numeric_dtype(target):
        target = target.astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    target = pd.to_numeric(target, errors="coerce")
    if target.isna().any():
        raise ValueError("Churn must contain only Yes/No or 1/0 values.")
    target = target.astype("int8")
    if set(target.unique()) != {0, 1}:
        raise ValueError("The training dataset must contain both churn classes.")
    return data, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, Any]:
    """Class weights address imbalance without synthesizing validation samples."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
            solver="liblinear"
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def train(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    cv_folds: int = 5,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Benchmark candidate pipelines, fit the best, and write the project artifacts."""
    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    report_dir = output_dir / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_and_clean_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    comparison: list[dict[str, Any]] = []
    estimators: dict[str, Pipeline] = {}

    for name, classifier in candidate_models().items():
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor(X_train)), ("classifier", classifier)]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"], ddof=1))
        comparison.append(row)
        estimators[name] = pipeline

    comparison.sort(key=lambda row: row["roc_auc_mean"], reverse=True)
    selected_name = comparison[0]["model"]
    selected_model = estimators[selected_name]
    selected_model.fit(X_train, y_train)

    probabilities = selected_model.predict_proba(X_test)[:, 1]
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "threshold": threshold,
        "confusion_matrix_labels": ["No Churn", "Churn"],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    artifact = {
        "model": selected_model,
        "model_name": selected_name,
        "threshold": threshold,
        "feature_names": X.columns.tolist(),
        "target_labels": {0: "No Churn", 1: "Churn"},
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
    }
    joblib.dump(artifact, artifact_dir / "churn_pipeline.joblib")

    metrics = {
        "project": "Customer Churn Prediction",
        "dataset_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "class_counts": {"no_churn": int((y == 0).sum()), "churn": int((y == 1).sum())},
        "test_rows": int(len(X_test)),
        "selected_model": selected_name,
        "selection_metric": "mean 5-fold ROC-AUC on the training split",
        "scikit_learn_version": sklearn.__version__,
        "cross_validation": comparison,
        "holdout_test": test_metrics,
    }
    metrics = _json_safe(metrics)
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["model"] for r in comparison]
    means = [r["roc_auc_mean"] for r in comparison]
    errors = [r["roc_auc_std"] for r in comparison]
    colors = ["#2563eb" if name == selected_name else "#94a3b8" for name in names]
    ax.bar(names, means, yerr=errors, capsize=5, color=colors)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Mean ROC-AUC (5-fold CV)")
    ax.set_title("Candidate Model Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "cv_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Churn", "Churn"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Holdout Test Confusion Matrix\n{selected_name}")
    fig.tight_layout()
    fig.savefig(report_dir / "holdout_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Global feature importance uses only the already selected fitted model.
    classifier = selected_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        values = None
    if values is not None:
        transformed_names = selected_model.named_steps["preprocess"].get_feature_names_out()
        importance = pd.DataFrame({"feature": transformed_names, "importance": values})
        importance = importance.sort_values("importance", ascending=False).head(20)
        importance.to_csv(report_dir / "feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        shown = importance.sort_values("importance", ascending=True)
        ax.barh(shown["feature"], shown["importance"], color="#0f766e")
        ax.set_title("Top Model Features")
        ax.set_xlabel("Importance" if hasattr(classifier, "feature_importances_") else "Absolute coefficient")
        fig.tight_layout()
        fig.savefig(report_dir / "feature_importance.png", dpi=160)
        plt.close(fig)

    print(f"Selected model: {selected_name}")
    print(f"Holdout ROC-AUC: {test_metrics['roc_auc']:.3f}")
    print(f"Holdout F1: {test_metrics['f1']:.3f}")
    print(f"Saved model: {artifact_dir / 'churn_pipeline.joblib'}")
    print(f"Saved metrics: {report_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/telco_customer_churn.csv",
        help="Path to the Telco customer churn CSV file.",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Project directory for artifacts and reports.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    train(args.data, args.output, cv_folds=args.cv_folds)


if __name__ == "__main__":
    main()
