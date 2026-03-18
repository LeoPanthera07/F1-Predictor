"""
src/loader.py
Loads and cleans all 14 raw CSV files.
Run directly to verify all files load correctly:
    python src/loader.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(r"D:\NewMission\Projects\ML\F1-Predictor\data\raw")

FILES = {
    "results":               "results.csv",
    "races":                 "races.csv",
    "drivers":               "drivers.csv",
    "constructors":          "constructors.csv",
    "circuits":              "circuits.csv",
    "qualifying":            "qualifying.csv",
    "pit_stops":             "pit_stops.csv",
    "driver_standings":      "driver_standings.csv",
    "constructor_standings": "constructor_standings.csv",
    "constructor_results":   "constructor_results.csv",
    "status":                "status.csv",
    "sprint_results":        "sprint_results.csv",
    "lap_times":             "lap_times.csv",
    "seasons":               "seasons.csv",
}

def load_raw() -> dict[str, pd.DataFrame]:
    """Load all CSVs, replace \\N with NaN."""
    dfs = {}
    for key, fname in FILES.items():
        path = RAW / fname
        if not path.exists():
            print(f"  MISSING: {fname}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df.replace("\\N", np.nan, inplace=True)
        dfs[key] = df
        print(f"  OK  {key:30s} {df.shape[0]:>7,} rows  {df.shape[1]:>3} cols")
    return dfs


def coerce_types(dfs: dict) -> dict:
    """Convert columns to correct types."""

    # results
    r = dfs["results"]
    for c in ["position", "grid", "milliseconds", "points",
              "fastestLap", "rank", "fastestLapSpeed"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    # races
    races = dfs["races"]
    races["year"]  = pd.to_numeric(races["year"],  errors="coerce")
    races["round"] = pd.to_numeric(races["round"], errors="coerce")
    races["date"]  = pd.to_datetime(races["date"], errors="coerce")

    # drivers
    dfs["drivers"]["dob"] = pd.to_datetime(
        dfs["drivers"]["dob"], errors="coerce")

    # pit stops
    pit = dfs["pit_stops"]
    pit["lap"]          = pd.to_numeric(pit["lap"],          errors="coerce")
    pit["milliseconds"] = pd.to_numeric(pit["milliseconds"], errors="coerce")
    pit["duration"]     = pd.to_numeric(pit["duration"],     errors="coerce")

    # driver standings
    ds = dfs["driver_standings"]
    for c in ["points", "position", "wins"]:
        ds[c] = pd.to_numeric(ds[c], errors="coerce")

    # constructor standings
    cs = dfs["constructor_standings"]
    for c in ["points", "position", "wins"]:
        cs[c] = pd.to_numeric(cs[c], errors="coerce")

    # circuits
    circ = dfs["circuits"]
    for c in ["lat", "lng", "alt"]:
        circ[c] = pd.to_numeric(circ[c], errors="coerce")

    return dfs


def load() -> dict[str, pd.DataFrame]:
    """Full pipeline: load + clean. Use this everywhere."""
    print("Loading CSVs from data/raw/...")
    dfs = load_raw()
    print("\nCoercing types...")
    dfs = coerce_types(dfs)
    print("\nDone.")
    return dfs


if __name__ == "__main__":
    dfs = load()
    print("\n=== SUMMARY ===")
    races = dfs["races"]
    results = dfs["results"]
    drivers = dfs["drivers"]
    print(f"Seasons:  {int(races['year'].min())} – {int(races['year'].max())}")
    print(f"Races:    {races['raceId'].nunique():,}")
    print(f"Results:  {len(results):,} rows")
    print(f"Drivers:  {drivers['driverId'].nunique():,}")
    print(f"Circuits: {dfs['circuits']['circuitId'].nunique():,}")

    wins = (results[results["position"] == 1]
            .groupby("driverId").size()
            .sort_values(ascending=False).head(5))
    name = dfs["drivers"].set_index("driverId")[["forename","surname"]]
    print("\nTop 5 winners:")
    for did, w in wins.items():
        n = name.loc[did]
        print(f"  {n['forename']} {n['surname']}: {w} wins")