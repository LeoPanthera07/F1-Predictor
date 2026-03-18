"""
backend/schemas.py
Pydantic models for request and response validation.
"""
from pydantic import BaseModel
from typing import Optional, List


class DriverInput(BaseModel):
    driverId:             int
    constructorId:        int
    circuitId:            int
    grid_pos:             float
    drv_roll5_finish:     Optional[float] = None
    drv_roll5_points:     Optional[float] = None
    drv_roll5_dnf:        Optional[float] = None
    drv_roll5_grid:       Optional[float] = None
    drv_experience:       Optional[float] = None
    drv_circuit_win_rate:    Optional[float] = None
    drv_circuit_podium_rate: Optional[float] = None
    con_roll5_finish:     Optional[float] = None
    con_roll5_points:     Optional[float] = None
    con_roll5_dnf:        Optional[float] = None
    con_avg_pit:          Optional[float] = None
    champ_pts:            Optional[float] = None
    champ_pos:            Optional[float] = None
    champ_wins:           Optional[float] = None
    con_champ_pts:        Optional[float] = None
    con_champ_pos:        Optional[float] = None
    season_progress:      Optional[float] = None
    overtaking_idx:       Optional[float] = None
    is_street:            Optional[int]   = 0
    driver_age:           Optional[float] = None
    alt:                  Optional[float] = 0


class PredictRequest(BaseModel):
    circuitId:  int
    season:     int
    round:      int
    lap_count:  int
    weather:    str = "dry"
    drivers:    List[DriverInput]


class DriverResult(BaseModel):
    driverId:            int
    constructorId:       int
    predicted_position:  int
    win_probability:     float


class PredictResponse(BaseModel):
    race:              str
    season:            int
    circuit:           str
    podium:            List[DriverResult]
    full_grid:         List[DriverResult]
    constructor_winner: int
    model_version:     str = "lgbm_v1"