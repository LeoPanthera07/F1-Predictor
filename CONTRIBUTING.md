# Contributing to F1 Result Predictor

## 1. Branching

- `main` – stable, deployable.
- `dev` – active development.
- Feature branches: `feat/xyz`, `fix/...

## 2. Jupyter workflow

- EDA should live in `notebooks/`.
- Once feature logic is stable, move it to `src/features.py`.
- Export final feature‑set as `data/features.csv` or `data/features.parquet`.
