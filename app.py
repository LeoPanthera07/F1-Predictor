"""
Streamlit app for F1 Result Predictor.

Run:
$ streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle

from src.inference import predict_finish, load_model

st.set_page_config(page_title="F1 Result Predictor", layout="centered")

st.title("🏁 F1 Result Predictor")
st.markdown("Predict a driver’s race-finish position with confidence utilizing full team analysis.")

@st.cache_data
def load_features():
    return pd.read_csv("data/features.csv")

features_df = load_features()

@st.cache_resource
def load_model_cached():
    return load_model("models/f1_predictor.pkl")

model = load_model_cached()

# Inputs
st.subheader("Input")

# Extract Real Current Grid
latest_season = features_df["season"].max()
current_season_df = features_df[features_df["season"] == latest_season]

# Driver formatting
current_grid_drivers = current_season_df["driver_name"].dropna().unique()

# Order circuits purely by their scheduled Round
schedule = current_season_df.drop_duplicates("circuit_name").sort_values("round")
current_grid_circuits = schedule["circuit_name"].tolist()

circuit = st.selectbox(
    "Circuit Scheduled Order",
    options=current_grid_circuits,
    index=0
)

driver = st.selectbox(
    "Driver",
    options=sorted(current_grid_drivers),
    index=0
)

grid = st.number_input(
    "Grid Position",
    min_value=1,
    max_value=24,
    value=10,
    step=1
)

if st.button("Predict"):
    with st.spinner("Analyzing driver & team car telemetry..."):
        result = predict_finish(
            circuit=circuit,
            grid=grid,
            driver_name=driver,
            features_df=features_df,
            model=model,
        )

    st.success("Prediction ready!")
    st.markdown("### Result")
    st.metric("Predicted Finish Position", f"{result['predicted_position']:.2f}")
    
    # Simple formatting hint logic to validate realistic F1 positions visually
    pos = round(result['predicted_position'])
    if pos <= 3:
        st.info("🏆 Podium finish predicted!")
    elif pos <= 10:
        st.info("💰 Strong points finish predicted.")
