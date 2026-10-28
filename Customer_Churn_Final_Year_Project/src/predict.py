"""Small prediction helpers shared by the notebook and Streamlit application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_artifact(path: str | Path = "artifacts/churn_pipeline.joblib") -> dict[str, Any]:
    artifact = joblib.load(path)
    if "model" not in artifact or "feature_names" not in artifact:
        raise ValueError("The selected file is not a Customer Churn project model.")
    return artifact


def predict_one(
    customer: dict[str, Any], artifact: dict[str, Any]
) -> tuple[str, float]:
    """Return the predicted class label and estimated churn score for one customer."""
    row = pd.DataFrame([customer]).reindex(columns=artifact["feature_names"])
    score = float(artifact["model"].predict_proba(row)[0, 1])
    churn = int(score >= float(artifact.get("threshold", 0.5)))
    label = artifact.get("target_labels", {0: "No Churn", 1: "Churn"})[churn]
    return label, score
