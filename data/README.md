# Data folder structure

Place CSVs here (or fetch them automatically in `data/update.py`):

- `races.csv` – one row per race (season, round, circuit, date).[1]
- `circuits.csv` – circuit metadata (name, location, length, etc.).[2]
- `drivers.csv` – driver metadata (name, nationality, years active).[2]
- `results.csv` – driver‑race outcomes (raceId, driverId, grid, position).[2]
- `qualifying.csv` – qualifying grid / times for each race.[2]

If needed, you can automatically pull Ergast‑style data via Python script (see `data/update.py`).
