"""Synthetic IoT sensor dataset with normal and manipulated readings."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "Temperature",
    "Humidity",
    "Pressure",
    "Vibration",
    "Gas_Level",
    "Delta_T",
    "Delta_V",
]

UNITS = {
    "Temperature": "°C",
    "Humidity": "%",
    "Pressure": "hPa",
    "Vibration": "mm/s",
    "Gas_Level": "ppm",
    "Delta_T": "°C/step",
    "Delta_V": "mm/s/step",
}

DISPLAY_NAMES = {
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Pressure": "Pressure",
    "Vibration": "Vibration",
    "Gas_Level": "Gas Level",
    "Delta_T": "Delta T",
    "Delta_V": "Delta V",
}

LABEL_NORMAL = 0
LABEL_MANIPULATED = 1

ATTACK_TYPES = [
    "false_data_injection",
    "data_tampering",
    "replay_attack",
    "sensor_spoofing",
]


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _normal_reading(rng: np.random.Generator) -> dict[str, float]:
    """Typical industrial sensor band: ~60–70°C, modest vibration."""
    return {
        "Temperature": _clip(rng.normal(65.0, 3.8), 55.0, 76.0),
        "Humidity": _clip(rng.normal(61.0, 6.0), 40.0, 82.0),
        "Pressure": _clip(rng.normal(1009.0, 6.0), 990.0, 1030.0),
        "Vibration": _clip(rng.normal(3.6, 1.1), 1.2, 6.2),
        "Gas_Level": _clip(rng.normal(1.25, 0.35), 0.4, 2.6),
        "Delta_T": _clip(rng.normal(0.8, 1.4), -3.0, 5.5),
        "Delta_V": _clip(rng.normal(0.25, 0.45), -1.0, 1.8),
    }


def _false_data_injection(rng: np.random.Generator) -> dict[str, float]:
    """Fabricated spike, e.g. 65°C → 95°C."""
    row = _normal_reading(rng)
    row["Temperature"] = _clip(rng.normal(88.0, 6.0), 74.0, 105.0)
    row["Delta_T"] = _clip(rng.normal(14.0, 6.0), 4.0, 32.0)
    if rng.random() < 0.35:
        row["Vibration"] = _clip(rng.normal(7.8, 1.1), 6.0, 11.0)
        row["Delta_V"] = _clip(rng.normal(4.1, 1.0), 2.0, 7.0)
    return row


def _data_tampering(rng: np.random.Generator) -> dict[str, float]:
    """Existing values altered in transit/storage — often vibration or gas."""
    row = _normal_reading(rng)
    row["Vibration"] = _clip(rng.normal(9.1, 1.3), 6.8, 13.5)
    row["Delta_V"] = _clip(rng.normal(5.4, 1.2), 2.8, 9.0)
    if rng.random() < 0.4:
        row["Gas_Level"] = _clip(rng.normal(4.6, 0.7), 3.2, 6.5)
    if rng.random() < 0.25:
        row["Temperature"] = _clip(rng.normal(82.0, 3.5), 76.0, 92.0)
        row["Delta_T"] = _clip(rng.normal(12.0, 3.0), 6.0, 20.0)
    return row


def _replay_attack(rng: np.random.Generator) -> dict[str, float]:
    """Stale legitimate packet replayed: frozen deltas, often an old high reading."""
    row = _normal_reading(rng)
    if rng.random() < 0.7:
        row["Temperature"] = _clip(rng.normal(91.0, 3.0), 84.0, 99.0)
        row["Vibration"] = _clip(rng.normal(8.2, 0.9), 6.5, 11.0)
    row["Delta_T"] = _clip(rng.normal(0.02, 0.04), -0.08, 0.08)
    row["Delta_V"] = _clip(rng.normal(0.01, 0.03), -0.06, 0.06)
    return row


def _sensor_spoofing(rng: np.random.Generator) -> dict[str, float]:
    """Malicious source mimicking a sensor with physically inconsistent values."""
    return {
        "Temperature": _clip(rng.normal(90.5, 3.8), 82.0, 102.0),
        "Humidity": _clip(rng.normal(88.0, 3.5), 78.0, 98.0),
        "Pressure": _clip(rng.normal(978.0, 6.0), 960.0, 992.0),
        "Vibration": _clip(rng.normal(8.6, 1.2), 6.4, 12.5),
        "Gas_Level": _clip(rng.normal(4.8, 0.8), 3.2, 7.0),
        "Delta_T": _clip(rng.normal(16.5, 4.0), 8.0, 28.0),
        "Delta_V": _clip(rng.normal(3.8, 1.1), 1.5, 7.5),
    }


_ATTACK_FNS = {
    "false_data_injection": _false_data_injection,
    "data_tampering": _data_tampering,
    "replay_attack": _replay_attack,
    "sensor_spoofing": _sensor_spoofing,
}


def generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Build a labelled tabular dataset aligned with the presentation features."""
    rng = np.random.default_rng(seed)
    n_manipulated = int(n_samples * 0.38)
    n_normal = n_samples - n_manipulated

    rows: list[dict] = []
    for _ in range(n_normal):
        row = _normal_reading(rng)
        row["Label"] = LABEL_NORMAL
        row["Attack_Type"] = "none"
        rows.append(row)

    attack_cycle = (
        ["false_data_injection"] * 4
        + ["data_tampering"] * 3
        + ["replay_attack"] * 2
        + ["sensor_spoofing"] * 2
    )
    for i in range(n_manipulated):
        attack = attack_cycle[i % len(attack_cycle)]
        row = _ATTACK_FNS[attack](rng)
        row["Label"] = LABEL_MANIPULATED
        row["Attack_Type"] = attack
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


SLIDE_EXAMPLE = {
    "Temperature": 91.5,
    "Humidity": 64.0,
    "Pressure": 1012.0,
    "Vibration": 8.1,
    "Gas_Level": 1.4,
    "Delta_T": 18.6,
    "Delta_V": 3.8,
}

PRESETS = {
    "Slide 13 sample (91.5°C)": SLIDE_EXAMPLE,
    "Normal operating range": {
        "Temperature": 66.2,
        "Humidity": 61.2,
        "Pressure": 1008.0,
        "Vibration": 3.4,
        "Gas_Level": 1.3,
        "Delta_T": 0.4,
        "Delta_V": 0.1,
    },
    "False data injection": {
        "Temperature": 95.0,
        "Humidity": 60.0,
        "Pressure": 1010.0,
        "Vibration": 4.8,
        "Gas_Level": 1.3,
        "Delta_T": 22.0,
        "Delta_V": 0.3,
    },
    "Data tampering": {
        "Temperature": 68.0,
        "Humidity": 59.0,
        "Pressure": 1007.0,
        "Vibration": 9.6,
        "Gas_Level": 4.4,
        "Delta_T": 0.8,
        "Delta_V": 5.9,
    },
    "Replay attack": {
        "Temperature": 92.0,
        "Humidity": 62.0,
        "Pressure": 1009.0,
        "Vibration": 8.4,
        "Gas_Level": 1.2,
        "Delta_T": 0.02,
        "Delta_V": 0.01,
    },
    "Sensor spoofing": {
        "Temperature": 90.0,
        "Humidity": 89.0,
        "Pressure": 972.0,
        "Vibration": 8.8,
        "Gas_Level": 5.1,
        "Delta_T": 15.0,
        "Delta_V": 4.2,
    },
}


def format_value(feature: str, value: float) -> str:
    if feature == "Temperature":
        return f"{value:.1f}°C"
    if feature == "Humidity":
        return f"{value:.0f}%"
    if feature == "Pressure":
        return f"{value:.0f} hPa"
    if feature == "Vibration":
        return f"{value:.1f} mm/s"
    unit = UNITS[feature]
    return f"{value:.1f} {unit}"
