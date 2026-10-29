"""Streamlit app: run with `streamlit run app.py` (after running src/train.py)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "student_model.joblib"

st.set_page_config(page_title="Student Performance Predictor")
st.title("Student Performance Predictor")
st.caption("Decision-support only. Predictions are uncertain and must not label students permanently.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model, defaults = bundle["pipeline"], bundle["defaults"]

c1, c2 = st.columns(2)
g1 = c1.slider("Grade G1 (0-20)", 0, 20, 10)
g2 = c2.slider("Grade G2 (0-20)", 0, 20, 10)
studytime = c1.selectbox("Study time (1: <2h ... 4: >10h)", [1, 2, 3, 4], index=1)
failures = c2.selectbox("Past failures", [0, 1, 2, 3])
absences = c1.number_input("Absences", 0, 75, 4)
internet = c2.selectbox("Internet at home", ["yes", "no"])

if st.button("Predict"):
    row = dict(defaults)  # all other features fall back to dataset median / most common value
    row.update(studytime=studytime, failures=failures, absences=absences, internet=internet,
               G1=g1, G2=g2, prior_avg=(g1 + g2) / 2,
               absence_band="low" if absences <= 3 else "medium" if absences <= 10 else "high")
    df = pd.DataFrame([row])[bundle["columns"]]
    p = model.predict_proba(df)[0, 1]
    st.metric("Probability of Pass", f"{p:.1%}")
    if p >= 0.5:
        st.success("Likely to Pass")
    else:
        st.warning("May need extra support")
