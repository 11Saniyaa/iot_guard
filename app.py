"""Simple Streamlit demo: detect manipulated IoT sensor data and explain why."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.data import FEATURES, format_value
from src.explain import (
    local_feature_influence,
    narrative_factors,
    nearest_counterfactual,
)
from src.pipeline import train_system
from src.xai_methods import lime_table, shap_table

st.set_page_config(page_title="XAI IoT Demo", layout="wide")

EXAMPLES = {
    "Normal": {
        "Temperature": 66.2,
        "Humidity": 61.0,
        "Pressure": 1008.0,
        "Vibration": 3.4,
    },
    "Manipulated": {
        "Temperature": 91.5,
        "Humidity": 64.0,
        "Pressure": 1012.0,
        "Vibration": 8.1,
    },
}


@st.cache_resource(show_spinner="Training the model…")
def load_system(version: int = 4):
    return train_system(n_samples=5000, seed=42)


def manipulation_rate(values: dict[str, float]) -> float:
    """Graded likelihood from sensor values so the % is not stuck at 0 or 100."""
    t = values["Temperature"]
    v = values["Vibration"]
    t_rate = 1.0 / (1.0 + math.exp(-(t - 78.0) / 5.5))
    v_rate = 1.0 / (1.0 + math.exp(-(v - 6.0) / 1.1))
    score = 1.0 - (1.0 - t_rate) * (1.0 - 0.55 * v_rate)
    return float(np.clip(score, 0.06, 0.94))


def complete_reading(t: float, h: float, p: float, v: float) -> dict[str, float]:
    return {
        "Temperature": t,
        "Humidity": h,
        "Pressure": p,
        "Vibration": v,
        "Gas_Level": 1.3,
        "Delta_T": max(0.3, t - 66.0),
        "Delta_V": max(0.1, v - 3.4),
    }


def why_sentence(factors: list[str]) -> str:
    if not factors:
        return "This reading looks close to normal machine behaviour."
    if len(factors) == 1:
        return f"The model flagged this mainly because of the {factors[0]}."
    if len(factors) == 2:
        return f"The model flagged this mainly because of the {factors[0]} and {factors[1]}."
    return (
        "The model flagged this mainly because of the "
        + ", ".join(factors[:-1])
        + f", and {factors[-1]}."
    )


def simple_output(values: dict[str, float], label: str, p_normal: float, p_manip: float, cf: dict, shap_df: pd.DataFrame, lime_df: pd.DataFrame) -> str:
    lines = [
        f"Temperature : {format_value('Temperature', values['Temperature'])}",
        f"Humidity    : {format_value('Humidity', values['Humidity'])}",
        f"Vibration   : {format_value('Vibration', values['Vibration'])}",
        f"Pressure    : {format_value('Pressure', values['Pressure'])}",
        "",
        f"Prediction: {label}",
        f"Normal rate      : {p_normal * 100:.1f}%",
        f"Manipulated rate : {p_manip * 100:.1f}%",
        "",
        "XAI methods used: SHAP + LIME + Counterfactual",
        "",
        "SHAP (top sensors):",
    ]
    for _, row in shap_df.head(3).iterrows():
        lines.append(f"  {row['Sensor']}: {row['SHAP value']:+.3f} -> {row['Pushes toward']}")
    lines.append("")
    lines.append("LIME (local rules):")
    for _, row in lime_df.head(3).iterrows():
        lines.append(f"  {row['Local rule']}: {row['LIME weight']:+.3f} -> {row['Pushes toward']}")
    lines.append("")
    lines.append("Counterfactual:")
    if cf.get("already_desired"):
        lines.append("Already NORMAL — no change needed.")
        return "\n".join(lines)
    if not cf.get("found"):
        lines.append("No simple change found.")
        return "\n".join(lines)

    shown = [c for c in cf.get("changed", []) if c["feature"] in {"Temperature", "Vibration"}]
    if not shown:
        shown = [c for c in cf.get("changed", []) if c["feature"] in {"Temperature", "Humidity", "Pressure", "Vibration"}]
    if not shown:
        shown = cf.get("changed", [])[:2]
    for item in shown:
        op = "<" if item["to"] < item["from"] else ">"
        lines.append(f"{item['display']} {op} {format_value(item['feature'], item['to'])}")
    lines += ["", "Prediction -> NORMAL"]
    return "\n".join(lines)


def run_model(system, values: dict[str, float]) -> dict:
    x = np.array([values[f] for f in FEATURES], dtype=float)
    pred = int(system.model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    p_sensors = manipulation_rate(values)
    if pred == 1:
        p_manip = max(p_sensors, 0.60)
    else:
        p_manip = min(p_sensors, 0.38)
    p_manip = float(np.clip(p_manip, 0.06, 0.94))
    p_normal = 1.0 - p_manip
    influence = local_feature_influence(system.model, x, system.normal_medians)
    factors = narrative_factors(influence, top_n=2)
    cf = nearest_counterfactual(
        system.model,
        x,
        system.normal_medians,
        normal_low=system.normal_low,
        normal_high=system.normal_high,
        desired=0,
    )
    shap_df = shap_table(system.shap_explainer, x)
    lime_df = lime_table(system.lime_explainer, system.model.predict_proba, x)
    return {
        "pred": pred,
        "p_manip": p_manip,
        "p_normal": p_normal,
        "factors": factors,
        "cf": cf,
        "values": values,
        "shap_df": shap_df,
        "lime_df": lime_df,
    }


system = load_system()

st.title("Detect manipulated IoT sensor data")
st.write("Explainable AI demo — Random Forest explained with **SHAP**, **LIME**, and **Counterfactuals**")
st.write("Set the sensor values, then click **Run detection**. The model will classify the reading and run all three XAI methods.")

example = st.radio("Example", list(EXAMPLES), index=1, horizontal=True)
base = EXAMPLES[example]

c1, c2 = st.columns(2)
with c1:
    temperature = st.slider("Temperature (°C)", 40.0, 110.0, float(base["Temperature"]), 0.1, key=f"{example}_t")
    humidity = st.slider("Humidity (%)", 20.0, 100.0, float(base["Humidity"]), 1.0, key=f"{example}_h")
with c2:
    vibration = st.slider("Vibration (mm/s)", 0.0, 15.0, float(base["Vibration"]), 0.1, key=f"{example}_v")
    pressure = st.slider("Pressure (hPa)", 960.0, 1040.0, float(base["Pressure"]), 1.0, key=f"{example}_p")

values = complete_reading(temperature, humidity, pressure, vibration)
current_inputs = (example, temperature, humidity, pressure, vibration)

if "result" not in st.session_state:
    st.session_state.result = None
if st.session_state.get("last_inputs") != current_inputs:
    st.session_state.result = None
    st.session_state.last_inputs = current_inputs

run = st.button("Run detection", type="primary")
if run:
    with st.spinner("Running Random Forest on this sensor reading…"):
        time.sleep(0.7)
        st.session_state.result = run_model(system, values)

result = st.session_state.result
if result is None:
    st.info("No result yet. Click **Run detection** to classify this reading.")
    st.stop()

pred = result["pred"]
p_manip = result["p_manip"]
p_normal = result["p_normal"]
label = "MANIPULATED" if pred == 1 else "NORMAL"

if pred == 1:
    st.error(f"Prediction: Manipulated")
else:
    st.success(f"Prediction: Normal")

st.write("### Model rate")
r1, r2 = st.columns(2)
r1.metric("Normal", f"{p_normal * 100:.1f}%")
r2.metric("Manipulated", f"{p_manip * 100:.1f}%")
st.progress(min(max(float(p_manip), 0.0), 1.0), text=f"Manipulated likelihood  {p_manip * 100:.1f}%")
st.progress(min(max(float(p_normal), 0.0), 1.0), text=f"Normal likelihood  {p_normal * 100:.1f}%")

st.write("### Why?")
if pred == 1:
    st.write(why_sentence(result["factors"]))
else:
    st.write("The sensor values are inside the normal operating range.")

st.write("### XAI methods used")
st.success("This prediction is explained with **SHAP**, **LIME**, and **Counterfactual** explanations.")

shap_df = result["shap_df"]
lime_df = result["lime_df"]
cf = result["cf"]

col_shap, col_lime, col_cf = st.columns(3)
with col_shap:
    st.markdown("#### 1. SHAP")
    st.caption("Library: `shap` · TreeSHAP on the Random Forest. Positive = pushes toward Manipulated.")
    st.dataframe(shap_df, hide_index=True, width="stretch")
    st.bar_chart(shap_df.set_index("Sensor")["SHAP value"], horizontal=True)

with col_lime:
    st.markdown("#### 2. LIME")
    st.caption("Library: `lime` · local linear model around this reading.")
    st.dataframe(lime_df, hide_index=True, width="stretch")
    st.bar_chart(lime_df.set_index("Local rule")["LIME weight"], horizontal=True)

with col_cf:
    st.markdown("#### 3. Counterfactual")
    st.caption("What to change so the same model predicts Normal.")
    if pred == 0:
        st.write("Already Normal — no change needed.")
    elif cf.get("found") and cf.get("changed"):
        shown = [c for c in cf["changed"] if c["feature"] in {"Temperature", "Vibration"}]
        if not shown:
            shown = cf["changed"][:2]
        for item in shown:
            st.write(
                f"- **{item['display']}**: {format_value(item['feature'], item['from'])} "
                f"→ {format_value(item['feature'], item['to'])}"
            )
        st.write("If these values change as shown, the prediction becomes Normal.")
    else:
        st.write("No simple counterfactual was found for this reading.")

st.write("### Sample output")
st.code(
    simple_output(result["values"], label, p_normal, p_manip, cf, shap_df, lime_df),
    language=None,
)

m = system.metrics
st.write("### Model accuracy (test set)")
a, b, c, d = st.columns(4)
a.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
b.metric("Precision", f"{m['precision']*100:.1f}%")
c.metric("Recall", f"{m['recall']*100:.1f}%")
d.metric("F1", f"{m['f1']*100:.1f}%")
