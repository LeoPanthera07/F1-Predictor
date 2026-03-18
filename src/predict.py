"""
src/predict.py
Loads trained model and runs inference.
Called by the FastAPI backend.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path

MODELS = Path("models")

_model     = None
_artifacts = None


def _load_model():
    global _model, _artifacts
    if _model is None:
        model_path = MODELS / "lgbm_ranker.txt"
        if not model_path.exists():
            raise FileNotFoundError(
                "Model not found. Run: python src/train.py"
            )
        _model     = lgb.Booster(model_file=str(model_path))
        _artifacts = joblib.load(str(MODELS / "artifacts.pkl"))


def predict_race(driver_rows: list) -> list:
    """
    Input : list of dicts, one per driver, with feature values.
    Output: list of dicts sorted by predicted finishing position.
    """
    _load_model()

    cols    = _artifacts["feature_cols"]
    medians = _artifacts["medians"]
    encoders = _artifacts["encoders"]

    df = pd.DataFrame(driver_rows)

    # Encode IDs
    for col, key in [
        ("driverId",      "driver"),
        ("constructorId", "constructor"),
        ("circuitId",     "circuit"),
    ]:
        le = encoders[key]
        df[key + "_enc"] = df[col].astype(str).apply(
            lambda x: int(le.transform([x])[0])
            if x in le.classes_ else -1
        )

    # Fill missing features with training medians
    for c, v in medians.items():
        if c not in df.columns:
            df[c] = v
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(v)

    X      = df[cols].fillna(0)
    scores = _model.predict(X)

    # Softmax scores into win probabilities
    exp_scores = np.exp(scores - scores.max())
    probs      = exp_scores / exp_scores.sum()

    results = []
    for i in range(len(df)):
        results.append({
            "driverId":           int(df.iloc[i]["driverId"]),
            "constructorId":      int(df.iloc[i]["constructorId"]),
            "score":              float(scores[i]),
            "win_probability":    float(probs[i]),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["predicted_position"] = rank

    return results