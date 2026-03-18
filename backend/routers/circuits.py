"""
backend/routers/circuits.py
GET /circuits/       — list all circuits
GET /circuits/{id}   — single circuit
"""
from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter(prefix="/circuits", tags=["Circuits"])
_df = None


def _load():
    global _df
    if _df is None:
        _df = pd.read_csv("data/raw/circuits.csv").replace("\\N", None)


@router.get("/")
async def get_circuits():
    _load()
    return _df.fillna("").to_dict(orient="records")


@router.get("/{circuit_id}")
async def get_circuit(circuit_id: int):
    _load()
    row = _df[_df["circuitId"] == circuit_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Circuit not found")
    return row.fillna("").to_dict(orient="records")[0]
