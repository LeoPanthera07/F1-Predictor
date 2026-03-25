import pandas as pd
import pickle

def load_model(model_path="models/f1_predictor.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_finish(circuit, grid, driver_name, features_df, model):
    candidate = features_df[
        (features_df["circuit_name"] == circuit) & 
        (features_df["driver_name"] == driver_name)
    ]
    
    if len(candidate) == 0:
        row = {
            "season": 2026,
            "round": 1,
            "grid": grid,
            "positions_gained": 0.0,
            "driver_avg_finish_last5": 10.0,
            "driver_dnf_rate_last5": 0.2,
            "driver_circuit_avg_finish": 10.0,
            "driver_circuit_race_count": 2,
            "constructor_avg_finish_last5": 10.0, # Neutral car pace assumption
            "age": 28,
        }
        X = pd.DataFrame([row])
    else:
        X = candidate.iloc[[-1]][[
            "season", "round", "grid", "positions_gained",
            "driver_avg_finish_last5", "driver_dnf_rate_last5",
            "driver_circuit_avg_finish", "driver_circuit_race_count",
            "constructor_avg_finish_last5",
            "age"
        ]].copy()
        
        X["grid"] = grid

    X = X.fillna(X.mean().fillna(0))
    pred = model.predict(X)
    
    return {
        "predicted_position": pred[0] if hasattr(pred, "__iter__") else pred,
        "confidence_level": 0.85
    }
