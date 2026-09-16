"""
=============================================================================
backend/main.py  —  Landslide Susceptibility API
=============================================================================
Endpoints
  POST /api/predict-risk      → nearest-neighbor lookup + live rainfall + ML
  POST /api/report-incident   → user incident submission → Neon DB
=============================================================================
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (Environment Variable with fallback)
# ---------------------------------------------------------------------------
DEFAULT_DB_URL = (
    "postgresql+psycopg2://neondb_owner:npg_eu3W1bYFTdqL"
    "@ep-empty-tree-avfhcwdo-pooler.c-11.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require"
)
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "landslide_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.json"

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=rain_sum&timezone=auto"
)

TIER_MAP = [
    (0.25, "Low"),
    (0.50, "Moderate"),
    (0.75, "High"),
    (1.00, "Critical"),
]

app_state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading ML model from %s ...", MODEL_PATH)
    app_state["model"] = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    app_state["feature_names"] = meta["feature_names"]
    log.info("Features loaded: %s", app_state["feature_names"])

    log.info("Creating DB engine ...")
    app_state["engine"] = create_engine(
        DB_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )
    with app_state["engine"].connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Neon PostgreSQL connection verified.")

    yield

    app_state["engine"].dispose()
    log.info("DB engine disposed. Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Landslide Susceptibility API",
    description="ML-powered landslide risk scoring for Sikkim region.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    latitude: float = Field(..., example=27.3139, description="Decimal degrees")
    longitude: float = Field(..., example=88.4441, description="Decimal degrees")


class PredictResponse(BaseModel):
    risk_score: float
    tier: str
    nearest_coordinate: Dict[str, float]
    live_rainfall_mm: Optional[float]
    features: Dict[str, Any]


class ReportRequest(BaseModel):
    latitude: float = Field(..., example=27.31)
    longitude: float = Field(..., example=88.44)
    description: str = Field(..., example="Crack observed on hillside")
    severity: str = Field(..., example="High")


class ReportResponse(BaseModel):
    status: str
    id: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_tier(prob: float) -> str:
    for threshold, label in TIER_MAP:
        if prob <= threshold:
            return label
    return "Critical"


async def fetch_live_rainfall(lat: float, lon: float) -> Optional[float]:
    url = OPEN_METEO_URL.format(lat=lat, lon=lon)
    headers = {"User-Agent": "SikkimLandslideEarlyWarning/1.0"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            rain_list = data.get("daily", {}).get("rain_sum", [])
            if rain_list:
                val = rain_list[0]
                return float(val) if val is not None else None
    except Exception as exc:
        log.warning("Open-Meteo fetch failed (%s). Using DB fallback.", exc)
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/predict-risk", response_model=PredictResponse)
async def predict_risk(payload: PredictRequest):
    lat, lon = payload.latitude, payload.longitude
    engine = app_state["engine"]
    model = app_state["model"]
    feature_names = app_state["feature_names"]

    nearest_sql = text("""
        SELECT *
        FROM location_features
        ORDER BY (("LATITUDE" - :lat)^2 + ("LONGITUDE" - :lon)^2)
        LIMIT 1;
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(nearest_sql, {"lat": lat, "lon": lon}).mappings().fetchone()
    except SQLAlchemyError as exc:
        log.error("DB query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database query failed.")

    if row is None:
        raise HTTPException(status_code=404, detail="No feature data found in DB.")

    row = dict(row)
    nearest_lat = float(row["LATITUDE"])
    nearest_lon = float(row["LONGITUDE"])
    db_rainfall = row.get("RAINFALL1")

    live_rainfall = await fetch_live_rainfall(lat, lon)
    effective_rainfall = live_rainfall if live_rainfall is not None else db_rainfall

    feature_values = {}
    for feat in feature_names:
        val = row.get(feat)
        feature_values[feat] = float(val) if isinstance(val, (int, float, np.number)) else val

    if "RAINFALL1" in feature_values and effective_rainfall is not None:
        feature_values["RAINFALL1"] = float(effective_rainfall)

    X = np.array([[feature_values[f] for f in feature_names]], dtype=float)

    proba = model.predict_proba(X)[0]
    risk_score = float(proba[1])
    tier = classify_tier(risk_score)

    return PredictResponse(
        risk_score=round(risk_score, 4),
        tier=tier,
        nearest_coordinate={"latitude": nearest_lat, "longitude": nearest_lon},
        live_rainfall_mm=round(float(live_rainfall), 2) if live_rainfall is not None else None,
        features={
            k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
            for k, v in feature_values.items()
        },
    )


@app.post("/api/report-incident", response_model=ReportResponse)
async def report_incident(payload: ReportRequest):
    engine = app_state["engine"]

    insert_sql = text("""
        INSERT INTO incident_reports (latitude, longitude, description, severity)
        VALUES (:lat, :lon, :desc, :sev)
        RETURNING id;
    """)
    try:
        with engine.begin() as conn:
            result = conn.execute(insert_sql, {
                "lat": payload.latitude,
                "lon": payload.longitude,
                "desc": payload.description,
                "sev": payload.severity,
            })
            new_id = result.fetchone()[0]
    except SQLAlchemyError as exc:
        log.error("Incident insert failed: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to save incident report.")

    return ReportResponse(status="success", id=new_id)


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "features": app_state.get("feature_names", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)