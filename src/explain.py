"""Counterfactual and occlusion-based explanations for the Random Forest."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .data import DISPLAY_NAMES, FEATURES, LABEL_MANIPULATED, LABEL_NORMAL, UNITS, format_value

MIN_CHANGE = {
    "Temperature": 0.8,
    "Humidity": 1.5,
    "Pressure": 2.5,
    "Vibration": 0.25,
    "Gas_Level": 0.15,
    "Delta_T": 0.8,
    "Delta_V": 0.2,
}

SCALE = np.array([20.0, 15.0, 15.0, 4.0, 1.5, 10.0, 3.0])


def _as_frame(x: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(x, dtype=float).reshape(1, -1)
    return pd.DataFrame(arr, columns=FEATURES)


def predict_label(model: RandomForestClassifier, x: np.ndarray) -> int:
    return int(model.predict(_as_frame(x))[0])


def predict_proba_manipulated(model: RandomForestClassifier, x: np.ndarray) -> float:
    proba = model.predict_proba(_as_frame(x))[0]
    classes = list(model.classes_)
    return float(proba[classes.index(LABEL_MANIPULATED)])


def local_feature_influence(
    model: RandomForestClassifier,
    x: np.ndarray,
    normal_medians: pd.Series,
) -> pd.DataFrame:
    """How much P(manipulated) drops if each feature is set to the normal median."""
    x = np.asarray(x, dtype=float)
    base = predict_proba_manipulated(model, x)
    rows = []
    for i, name in enumerate(FEATURES):
        swapped = x.copy()
        swapped[i] = float(normal_medians[name])
        new_p = predict_proba_manipulated(model, swapped)
        rows.append(
            {
                "feature": name,
                "display": DISPLAY_NAMES[name],
                "influence": base - new_p,
                "value": x[i],
                "normal_median": float(normal_medians[name]),
            }
        )
    return pd.DataFrame(rows).sort_values("influence", ascending=False).reset_index(drop=True)


def _interpolate(x: np.ndarray, indices: tuple[int, ...], targets: np.ndarray, lam: float) -> np.ndarray:
    cand = x.copy()
    for i in indices:
        cand[i] = x[i] + lam * (targets[i] - x[i])
    return cand


def _flip_lambda(
    model: RandomForestClassifier,
    x: np.ndarray,
    indices: tuple[int, ...],
    targets: np.ndarray,
    desired: int,
) -> np.ndarray | None:
    if predict_label(model, _interpolate(x, indices, targets, 1.0)) != desired:
        return None
    lo, hi = 0.0, 1.0
    best = _interpolate(x, indices, targets, 1.0)
    for _ in range(20):
        mid = (lo + hi) / 2.0
        cand = _interpolate(x, indices, targets, mid)
        if predict_label(model, cand) == desired:
            best = cand
            hi = mid
        else:
            lo = mid
    return best


def _material_changes(original: np.ndarray, candidate: np.ndarray) -> list[dict]:
    changed = []
    for i, name in enumerate(FEATURES):
        delta = abs(candidate[i] - original[i])
        if delta < MIN_CHANGE[name]:
            continue
        changed.append(
            {
                "feature": name,
                "display": DISPLAY_NAMES[name],
                "from": float(original[i]),
                "to": float(candidate[i]),
                "unit": UNITS[name],
            }
        )
    preferred = ["Temperature", "Vibration", "Delta_T", "Delta_V", "Gas_Level", "Humidity", "Pressure"]
    changed.sort(key=lambda item: preferred.index(item["feature"]))
    return changed


def nearest_counterfactual(
    model: RandomForestClassifier,
    x: np.ndarray,
    normal_medians: pd.Series,
    normal_low: pd.Series | None = None,
    normal_high: pd.Series | None = None,
    desired: int = LABEL_NORMAL,
) -> dict:
    """Sparse counterfactual: change the fewest sensors into the normal operating band."""
    x = np.asarray(x, dtype=float).copy()
    if predict_label(model, x) == desired:
        return {
            "found": True,
            "original": x,
            "counterfactual": x,
            "changed": [],
            "already_desired": True,
        }

    med = np.array([float(normal_medians[f]) for f in FEATURES])
    if normal_low is None:
        low = med
    else:
        low = np.array([float(normal_low[f]) for f in FEATURES])
    if normal_high is None:
        high = med
    else:
        high = np.array([float(normal_high[f]) for f in FEATURES])

    # Push high outliers down toward the lower normal band, and vice versa.
    targets = np.where(x >= med, low, high)

    influence = local_feature_influence(model, x, normal_medians)
    ranked = [FEATURES.index(name) for name in influence["feature"].tolist()]

    best: np.ndarray | None = None
    best_cost = np.inf
    for k in (1, 2, 3, 4):
        found_k = False
        for combo in combinations(ranked[:6], k):
            cand = _flip_lambda(model, x, combo, targets, desired)
            if cand is None:
                continue
            cost = float(np.sum(np.abs(cand - x) / SCALE) + 0.45 * k)
            if cost < best_cost:
                best, best_cost = cand, cost
                found_k = True
        if found_k:
            break

    if best is None:
        cand = _flip_lambda(model, x, tuple(range(len(FEATURES))), targets, desired)
        if cand is None:
            return {"found": False, "original": x, "counterfactual": None, "changed": []}
        best = cand

    changed = _material_changes(x, best)
    return {
        "found": True,
        "original": x,
        "counterfactual": best,
        "changed": changed,
        "already_desired": False,
    }


def narrative_factors(influence: pd.DataFrame, top_n: int = 3, min_score: float = 0.02) -> list[str]:
    factors = []
    for _, row in influence.iterrows():
        if row["influence"] < min_score:
            continue
        factors.append(_factor_phrase(row))
        if len(factors) >= top_n:
            break
    if factors:
        return factors

    ranked = influence.copy()
    ranked["dev"] = (ranked["value"] - ranked["normal_median"]).abs()
    ranked = ranked.sort_values("dev", ascending=False)
    for _, row in ranked.head(top_n).iterrows():
        if row["dev"] < MIN_CHANGE.get(row["feature"], 0.3):
            continue
        factors.append(_factor_phrase(row))
    return factors


def _factor_phrase(row: pd.Series) -> str:
    if row["feature"] == "Delta_T" and abs(row["value"]) > abs(row["normal_median"]) + 1.5:
        return "sudden temperature change"
    if row["feature"] == "Delta_V" and abs(row["value"]) > abs(row["normal_median"]) + 0.8:
        return "sudden vibration change"
    direction = "high" if row["value"] > row["normal_median"] else "low"
    return f"{direction} {row['display'].lower()}"


def counterfactual_lines(result: dict) -> list[dict]:
    lines = []
    for item in result.get("changed", []):
        left = format_value(item["feature"], item["from"])
        right = format_value(item["feature"], item["to"])
        if item["to"] < item["from"]:
            bound = format_value(item["feature"], item["to"])
            threshold = f"{item['display']} < {bound}"
        else:
            bound = format_value(item["feature"], item["to"])
            threshold = f"{item['display']} > {bound}"
        lines.append(
            {
                "display": item["display"],
                "from_text": left,
                "to_text": right,
                "threshold": threshold,
            }
        )
    return lines


def sample_output_text(
    values: dict[str, float],
    prediction_label: str,
    cf_result: dict,
) -> str:
    shown = ["Temperature", "Humidity", "Vibration", "Pressure"]
    block = []
    width = 12
    for name in shown:
        block.append(f"{name:<{width}}: {format_value(name, values[name])}")
    block.append("")
    block.append(f"Prediction: {prediction_label}")
    block.append("")
    if cf_result.get("already_desired"):
        block.append("Counterfactual:")
        block.append("Already classified as NORMAL — no change required.")
        return "\n".join(block)
    if not cf_result.get("found"):
        block.append("Counterfactual: not found within the normal operating prototype.")
        return "\n".join(block)
    block.append("Counterfactual:")
    for line in counterfactual_lines(cf_result):
        block.append(line["threshold"])
    block.append("")
    block.append("Prediction -> NORMAL")
    return "\n".join(block)
