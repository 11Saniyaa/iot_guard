"""SHAP, LIME, and helpers that return display-ready explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from .data import DISPLAY_NAMES, FEATURES, LABEL_MANIPULATED


def build_shap_explainer(rf) -> shap.TreeExplainer:
    return shap.TreeExplainer(rf)


def build_lime_explainer(X_train: pd.DataFrame, seed: int = 42) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=list(FEATURES),
        class_names=["Normal", "Manipulated"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )


def _shap_vector(explainer: shap.TreeExplainer, x: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(np.asarray(x, dtype=float).reshape(1, -1), check_additivity=False)
    if isinstance(values, list):
        idx = 1 if len(values) > 1 else 0
        arr = np.asarray(values[idx])
    else:
        arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[0, :, -1]
    elif arr.ndim == 2:
        arr = arr[0]
    return arr.reshape(-1)


def shap_table(explainer: shap.TreeExplainer, x: np.ndarray) -> pd.DataFrame:
    """Per-sensor SHAP value for the Manipulated class."""
    contrib = _shap_vector(explainer, x)
    rows = []
    for name, value in zip(FEATURES, contrib):
        rows.append(
            {
                "Sensor": DISPLAY_NAMES[name],
                "SHAP value": float(value),
                "Pushes toward": "Manipulated" if value > 0 else "Normal",
            }
        )
    return pd.DataFrame(rows).sort_values("SHAP value", key=np.abs, ascending=False).reset_index(drop=True)


def lime_table(explainer: LimeTabularExplainer, predict_proba, x: np.ndarray) -> pd.DataFrame:
    """Local LIME weights for the Manipulated class."""

    def _proba(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(data, dtype=float), columns=FEATURES)
        return predict_proba(frame)

    explanation = explainer.explain_instance(
        np.asarray(x, dtype=float),
        _proba,
        num_features=min(5, len(FEATURES)),
        num_samples=800,
        labels=(LABEL_MANIPULATED,),
    )
    rows = []
    for rule, weight in explanation.as_list(label=LABEL_MANIPULATED):
        rows.append(
            {
                "Local rule": rule,
                "LIME weight": float(weight),
                "Pushes toward": "Manipulated" if weight > 0 else "Normal",
            }
        )
    return pd.DataFrame(rows)
