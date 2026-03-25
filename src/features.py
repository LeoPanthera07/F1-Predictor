import pandas as pd

def build_features(races, results, qualifying, drivers, circuits):
    circuits = circuits.rename(columns={"name": "circuit_name"})
    
    if "position" in qualifying.columns:
        qualifying = qualifying.rename(columns={"position": "qualifying_position"})
    if "position" in results.columns:
        results = results.drop(columns=["position"])

    df = results.merge(races, on="raceId", how="left")
    df = df.merge(drivers, on="driverId", how="left")
    df = df.merge(circuits, on="circuitId", how="left")
    df = df.merge(qualifying, on=["raceId", "driverId"], how="left")

    df["position"] = pd.to_numeric(df["positionOrder"], errors='coerce').fillna(20)
    df["grid"] = pd.to_numeric(df["grid"], errors='coerce').fillna(20)
    df["positions_gained"] = (df["grid"] - df["position"]).fillna(0.0)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = df["date"].dt.year - df["dob"].dt.year
    df["age"] = df["age"].fillna(28)

    # 1. Driver Names formatting
    df["driver_name"] = df["forename"] + " " + df["surname"]

    # Rolling Driver finishes
    df = df.sort_values(["driverId", "date"])
    df["driver_avg_finish_last5"] = (
        df.groupby("driverId")["position"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    df["dnf"] = (~df["positionText"].astype(str).str.isdigit()).astype(int)
    df["driver_dnf_rate_last5"] = (
        df.groupby("driverId")["dnf"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    ci = df.groupby(["driverId", "circuitId"])["position"].agg(["mean", "count"]).reset_index()
    ci = ci.rename(columns={"mean": "driver_circuit_avg_finish", "count": "driver_circuit_race_count"})
    df = df.merge(ci, on=["driverId", "circuitId"], how="left")

    # 2. Add CRITICAL Team Performance Proxy (Constructor Rolling Average)
    df = df.sort_values(["constructorId", "date"])
    df["constructor_avg_finish_last5"] = (
        df.groupby("constructorId")["position"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    df["season"] = df["year"]
    
    cols_to_keep = [
        "raceId", "driverId", "driver_name", "circuitId", "circuit_name",
        "season", "round", "grid", "position", 
        "positions_gained", "driver_avg_finish_last5",
        "driver_dnf_rate_last5", "driver_circuit_avg_finish",
        "driver_circuit_race_count", "constructor_avg_finish_last5", "age"
    ]
    
    return df[cols_to_keep]
