import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
backend/main.py
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routers import predict, drivers, circuits

app = FastAPI(
    title="F1 Race Predictor",
    description="Predicts F1 race podium using LightGBM LambdaRank",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(drivers.router)
app.include_router(circuits.router)

frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    index = Path("frontend/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "F1 Predictor API", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "lgbm_v1", "docs": "/docs"}