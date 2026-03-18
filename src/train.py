"""
src/train.py
Trains LightGBM LambdaRank model on the feature matrix.
Run:
    python src/train.py
Outputs:
    models/lgbm_ranker.txt
    models/artifacts.pkl
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

PROCESSED = Path("data/processed")
MODELS    = Path("models")
MODELS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "grid_pos", "drv_roll5_finish", "drv_roll5_points", "drv_roll5_dnf",
    "drv_roll5_grid", "drv_experience", "drv_circuit_win_rate",
    "drv_circuit_podium_rate", "con_roll5_finish", "con_roll5_points",
    "con_roll5_dnf", "con_avg_pit", "champ_pts", "champ_pos",
    "champ_wins", "con_champ_pts", "con_champ_pos", "season_progress",
    "overtaking_idx", "is_street", "driver_age", "alt",
    "driver_enc", "constructor_enc", "circuit_enc"
]

PARAMS = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "ndcg_eval_at":     [3, 5],
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "n_jobs":           -1,
}


def encode_ids(df: pd.DataFrame):
    """Label-encode driver, constructor, circuit IDs."""
    encoders = {}
    for col, key in [
        ("driverId",      "driver"),
        ("constructorId", "constructor"),
        ("circuitId",     "circuit"),
    ]:
        le = LabelEncoder()
        df[key + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[key] = le
    return df, encoders


def top3_accuracy(df: pd.DataFrame):
    """
    P1 accuracy  = fraction of races where predicted P1 = actual P1
    Podium match = fraction of races where ≥2 of predicted top3 are in actual top3
    """
    p1_correct = podium_match = total = 0

    for _, grp in df.groupby("raceId"):
        pred   = grp.sort_values("pred_score", ascending=False)
        actual = grp.sort_values("finish_pos")

        actual_p1   = actual.iloc[0]["driverId"]
        actual_top3 = set(actual.head(3)["driverId"])
        pred_top3   = set(pred.head(3)["driverId"])

        if pred.iloc[0]["driverId"] == actual_p1:
            p1_correct += 1
        if len(actual_top3 & pred_top3) >= 2:
            podium_match += 1
        total += 1

    return p1_correct / total, podium_match / total


def train():
    print("Loading features...")
    df = pd.read_csv(PROCESSED / "features.csv")
    print(f"  {len(df):,} rows | years {int(df['year'].min())}–{int(df['year'].max())}")

    print("\nEncoding IDs...")
    df, encoders = encode_ids(df)

    # Relevance label: P1 gets highest score, last place gets 0
    df["label"] = (21 - df["finish_pos"]).clip(0, 20).astype(int)

    # ── Split ─────────────────────────────────────────────────────────────────
    # Train: 2000–2021  |  Val: 2022–2023  |  Test: 2024
    train_df = df[df["year"] <= 2021].sort_values(["raceId", "finish_pos"])
    val_df   = df[df["year"].between(2022, 2023)].sort_values(["raceId", "finish_pos"])
    test_df  = df[df["year"] == 2024].sort_values(["raceId", "finish_pos"]).copy()

    print("\nSplit:")
    print(f"  Train: {len(train_df):,} rows ({train_df['raceId'].nunique()} races)")
    print(f"  Val:   {len(val_df):,} rows ({val_df['raceId'].nunique()} races)")
    print(f"  Test:  {len(test_df):,} rows ({test_df['raceId'].nunique()} races)")

    X_train = train_df[FEATURE_COLS].fillna(0)
    X_val   = val_df[FEATURE_COLS].fillna(0)
    X_test  = test_df[FEATURE_COLS].fillna(0)

    q_train = train_df.groupby("raceId").size().values
    q_val   = val_df.groupby("raceId").size().values

    train_data = lgb.Dataset(X_train, label=train_df["label"], group=q_train)
    val_data   = lgb.Dataset(X_val,   label=val_df["label"],   group=q_val,
                             reference=train_data)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining LightGBM LambdaRank...")
    model = lgb.train(
        PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]
    )
    print(f"  Best iteration: {model.best_iteration}")

    # ── Evaluate on 2024 ──────────────────────────────────────────────────────
    test_df["pred_score"] = model.predict(X_test)
    p1_acc, pod_acc = top3_accuracy(test_df)

    print("\n=== Test Results (2024) ===")
    print(f"  Races evaluated : {test_df['raceId'].nunique()}")
    print(f"  P1 accuracy     : {p1_acc*100:.1f}%")
    print(f"  Podium accuracy : {pod_acc*100:.1f}%  (≥2/3 correct)")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\nTop 10 features by importance:")
    for feat, score in fi.head(10).items():
        print(f"  {feat:30s} {score:,.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_model(str(MODELS / "lgbm_ranker.txt"))

    artifacts = {
        "encoders":     encoders,
        "feature_cols": FEATURE_COLS,
        "medians":      df[FEATURE_COLS[:-3]].median().to_dict(),
    }
    joblib.dump(artifacts, str(MODELS / "artifacts.pkl"))

    print("\nSaved:")
    print("  models/lgbm_ranker.txt")
    print("  models/artifacts.pkl")


if __name__ == "__main__":
    train()
