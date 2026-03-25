import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def train_model(df, mode="regression", test_size=0.2):
    cols = [
        "season", "round", "grid",
        "positions_gained",
        "driver_avg_finish_last5",
        "driver_dnf_rate_last5",
        "driver_circuit_avg_finish",
        "driver_circuit_race_count",
        "constructor_avg_finish_last5",
        "age"
    ]
    
    X = df[cols].copy()
    y = df["position"].copy()

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", StandardScaler()),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    score = mean_absolute_error(y_test, y_pred)
    print(f"Random Forest MAE: {score:.2f} positions off")

    with open("models/f1_predictor.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline, X_test, y_test, y_pred
