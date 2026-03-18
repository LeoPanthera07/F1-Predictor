"""
backend/routers/drivers.py
GET /drivers/        — list all drivers
GET /drivers/{id}    — single driver
"""
from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter(prefix="/drivers", tags=["Drivers"])
_df = None


def _load():
    global _df
    if _df is None:
        _df = pd.read_csv("data/raw/drivers.csv").replace("\\N", None)


@router.get("/")
async def get_drivers():
    _load()
    cols = ["driverId", "driverRef", "forename", "surname",
            "code", "number", "nationality"]
    return _df[cols].fillna("").to_dict(orient="records")


@router.get("/{driver_id}")
async def get_driver(driver_id: int):
    _load()
    row = _df[_df["driverId"] == driver_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Driver not found")
    return row.fillna("").to_dict(orient="records")[0]