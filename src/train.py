import os
import pandas as pd
from src.features import build_features
from src.model import train_model

def run_pipeline():
    print("Loading raw CSV files from data/...")
    
    try:
        races = pd.read_csv("data/races.csv", low_memory=False)
        results = pd.read_csv("data/results.csv", low_memory=False)
        qualifying = pd.read_csv("data/qualifying.csv", low_memory=False)
        drivers = pd.read_csv("data/drivers.csv", low_memory=False)
        circuits = pd.read_csv("data/circuits.csv", low_memory=False)
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        print("Please ensure your F1 CSVs are extracted properly in the data/ folder.")
        return

    print("Building features... (This might take a minute)")
    features_df = build_features(races, results, qualifying, drivers, circuits)
    
    os.makedirs("models", exist_ok=True)
    features_path = "data/features.csv"
    print(f"Saving generated features to {features_path}...")
    features_df.to_csv(features_path, index=False)

    print("Training the model...")
    # Ensure targets are completely clean
    train_df = features_df.dropna(subset=["position"])
    
    model, X_test, y_test, y_pred = train_model(train_df, mode="regression")
    
    print("Pipeline completed successfully! You can now run Streamlit.")

if __name__ == "__main__":
    run_pipeline()
