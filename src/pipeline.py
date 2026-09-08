"""Train the Random Forest detector and compute evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .data import FEATURES, LABEL_MANIPULATED, generate_dataset
from .xai_methods import build_lime_explainer, build_shap_explainer


@dataclass
class TrainedSystem:
    model: object
    rf: RandomForestClassifier
    shap_explainer: object
    lime_explainer: object
    metrics: dict
    feature_importance: pd.DataFrame
    normal_medians: pd.Series
    normal_low: pd.Series
    normal_high: pd.Series
    dataset: pd.DataFrame
    y_test: np.ndarray
    y_pred: np.ndarray
    confusion: np.ndarray


def train_system(n_samples: int = 5000, seed: int = 42) -> TrainedSystem:
    df = generate_dataset(n_samples=n_samples, seed=seed)
    X = df[FEATURES]
    y = df["Label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        min_samples_leaf=18,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    y_pred = calibrated.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=LABEL_MANIPULATED)),
        "recall": float(recall_score(y_test, y_pred, pos_label=LABEL_MANIPULATED)),
        "f1": float(f1_score(y_test, y_pred, pos_label=LABEL_MANIPULATED)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(df)),
        "n_normal": int((df["Label"] == 0).sum()),
        "n_manipulated": int((df["Label"] == 1).sum()),
    }

    importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    normal_train = X_train.loc[y_train == 0, FEATURES]
    normal_medians = normal_train.median()
    normal_low = normal_train.quantile(0.08)
    normal_high = normal_train.quantile(0.92)
    confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])

    shap_explainer = build_shap_explainer(model)
    lime_explainer = build_lime_explainer(X_train, seed=seed)

    return TrainedSystem(
        model=calibrated,
        rf=model,
        shap_explainer=shap_explainer,
        lime_explainer=lime_explainer,
        metrics=metrics,
        feature_importance=importance,
        normal_medians=normal_medians,
        normal_low=normal_low,
        normal_high=normal_high,
        dataset=df,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        confusion=confusion,
    )
