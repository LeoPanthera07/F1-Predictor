"""
src/features.py
Builds the feature matrix for model training.
Run directly to generate data/processed/features.csv:
    python src/features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loader import load

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

STREET_CIRCUITS = [
    "monaco", "baku", "jeddah", "miami", "vegas",
    "singapore", "valencia", "detroit", "adelaide"
]


def build_features(min_year: int = 2000) -> pd.DataFrame:

    dfs = load()
    results      = dfs["results"]
    races        = dfs["races"]
    drivers      = dfs["drivers"]
    constructors = dfs["constructors"]
    circuits     = dfs["circuits"]
    pit_stops    = dfs["pit_stops"]
    driver_standings     = dfs["driver_standings"]
    constructor_standings = dfs["constructor_standings"]

    print("\nBuilding base dataframe...")

    # ── Base merge ────────────────────────────────────────────────────────────
    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "name", "date"]],
        on="raceId"
    )
    df = df.merge(
        circuits[["circuitId", "circuitRef", "lat", "lng", "alt"]],
        on="circuitId"
    )
    df = df.merge(
        drivers[["driverId", "dob", "nationality"]],
        on="driverId"
    )
    df = df.merge(
        constructors[["constructorId", "constructorRef", "nationality"]]
        .rename(columns={"nationality": "con_nationality"}),
        on="constructorId"
    )

    # ── Basic columns ─────────────────────────────────────────────────────────
    df["grid_pos"]   = pd.to_numeric(df["grid"],  errors="coerce").fillna(20)
    df["finish_pos"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["points"]     = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df["dnf"]        = df["position"].isna().astype(int)
    df["won"]        = (df["finish_pos"] == 1).astype(int)
    df["podium"]     = (df["finish_pos"] <= 3).astype(int)
    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25
    df["alt"]        = df["alt"].fillna(0)
    df["is_street"]  = df["circuitRef"].isin(STREET_CIRCUITS).astype(int)

    # ── Filter modern era ─────────────────────────────────────────────────────
    df = df[df["year"] >= min_year].copy()
    df = df.sort_values(["date", "raceId", "finish_pos"]).reset_index(drop=True)

    print(f"  Base shape: {df.shape}")

    # ── 1. Driver rolling features (last 5 races) ─────────────────────────────
    print("  Building driver rolling features...")
    df = df.sort_values(["driverId", "date"])

    for col, out in [
        ("finish_pos", "drv_roll5_finish"),
        ("points",     "drv_roll5_points"),
        ("dnf",        "drv_roll5_dnf"),
        ("grid_pos",   "drv_roll5_grid"),
    ]:
        df[out] = (df.groupby("driverId")[col]
                   .transform(lambda x: x.shift(1)
                               .rolling(5, min_periods=1).mean()))

    # ── 2. Driver career race count ───────────────────────────────────────────
    df["drv_experience"] = df.groupby("driverId").cumcount()

    # ── 3. Driver circuit-specific win & podium rate ──────────────────────────
    print("  Building circuit-specific rates...")
    for col, out in [
        ("won",    "drv_circuit_win_rate"),
        ("podium", "drv_circuit_podium_rate"),
    ]:
        df[out] = (df.groupby(["driverId", "circuitId"])[col]
                   .transform(lambda x: x.shift(1).expanding().mean()))

    # ── 4. Constructor rolling features ──────────────────────────────────────
    print("  Building constructor rolling features...")
    df = df.sort_values(["constructorId", "date"])

    for col, out in [
        ("finish_pos", "con_roll5_finish"),
        ("points",     "con_roll5_points"),
        ("dnf",        "con_roll5_dnf"),
    ]:
        df[out] = (df.groupby("constructorId")[col]
                   .transform(lambda x: x.shift(1)
                               .rolling(5, min_periods=1).mean()))

    # ── 5. Pit stop avg duration per constructor per circuit ──────────────────
    print("  Building pit stop features...")
    pit_agg = (
        pit_stops
        .merge(races[["raceId", "circuitId"]], on="raceId")
        .merge(results[["raceId", "driverId", "constructorId"]],
               on=["raceId", "driverId"])
        .groupby(["constructorId", "circuitId"])["duration"]
        .mean()
        .reset_index()
        .rename(columns={"duration": "con_avg_pit"})
    )
    df = df.merge(pit_agg, on=["constructorId", "circuitId"], how="left")
    df["con_avg_pit"] = df["con_avg_pit"].fillna(df["con_avg_pit"].median())

    # ── 6. Championship standing before this race ─────────────────────────────
    print("  Building championship features...")
    ds = (driver_standings
          .sort_values(["driverId", "raceId"])
          .copy())
    ds["champ_pts"] = ds.groupby("driverId")["points"].shift(1)
    ds["champ_pos"] = ds.groupby("driverId")["position"].shift(1)
    ds["champ_wins"] = ds.groupby("driverId")["wins"].shift(1)

    df = df.merge(
        ds[["raceId", "driverId", "champ_pts", "champ_pos", "champ_wins"]],
        on=["raceId", "driverId"],
        how="left"
    )

    cs = (constructor_standings
          .sort_values(["constructorId", "raceId"])
          .copy())
    cs["con_champ_pts"] = cs.groupby("constructorId")["points"].shift(1)
    cs["con_champ_pos"] = cs.groupby("constructorId")["position"].shift(1)

    df = df.merge(
        cs[["raceId", "constructorId", "con_champ_pts", "con_champ_pos"]],
        on=["raceId", "constructorId"],
        how="left"
    )

    # ── 7. Season progress (0.0 = first race, 1.0 = last race) ───────────────
    max_round = (races.groupby("year")["round"]
                 .max().reset_index(name="max_round"))
    df = df.merge(max_round, on="year")
    df["season_progress"] = df["round"] / df["max_round"]

    # ── 8. Circuit overtaking index ───────────────────────────────────────────
    df["pos_gained"] = df["grid_pos"] - df["finish_pos"]
    ovt = (df.groupby("circuitId")["pos_gained"]
           .mean().reset_index(name="overtaking_idx"))
    df = df.merge(ovt, on="circuitId", how="left")

    # ── 9. Fill all NAs with column median ────────────────────────────────────
    feature_cols = [
        "grid_pos", "drv_roll5_finish", "drv_roll5_points", "drv_roll5_dnf",
        "drv_roll5_grid", "drv_experience", "drv_circuit_win_rate",
        "drv_circuit_podium_rate", "con_roll5_finish", "con_roll5_points",
        "con_roll5_dnf", "con_avg_pit", "champ_pts", "champ_pos",
        "champ_wins", "con_champ_pts", "con_champ_pos", "season_progress",
        "overtaking_idx", "is_street", "driver_age", "alt"
    ]
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = PROCESSED / "features.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Years: {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"Features built: {len(feature_cols)}")
    print("\nFeature list:")
    for i, f in enumerate(feature_cols, 1):
        print(f"  {i:2}. {f}")

    return df


if __name__ == "__main__":
    build_features()