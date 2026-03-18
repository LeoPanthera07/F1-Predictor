"""
backend/routers/predict.py
POST /predict/race  — returns predicted podium and full grid
"""
from fastapi import APIRouter, HTTPException
from backend.schemas import PredictRequest, PredictResponse, DriverResult

router = APIRouter(prefix="/predict", tags=["Predict"])


def _get_predictor():
    from src.predict import predict_race
    return predict_race


@router.post("/race", response_model=PredictResponse)
async def predict_race_endpoint(req: PredictRequest):
    try:
        predict_race = _get_predictor()
        rows    = [d.model_dump() for d in req.drivers]
        results = predict_race(rows)

        full_grid = [DriverResult(**r) for r in results]
        podium    = full_grid[:3]
        con_winner = podium[0].constructorId

        return PredictResponse(
            race=f"Round {req.round}",
            season=req.season,
            circuit=str(req.circuitId),
            podium=podium,
            full_grid=full_grid,
            constructor_winner=con_winner,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python src/train.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))