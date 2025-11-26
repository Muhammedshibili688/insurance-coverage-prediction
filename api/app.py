# api/app.py
import os
import io
import joblib
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# adjust if running from different CWD
from src.config import MODEL_DIR, PROCESSED_DIR, TARGETS, FEATURE_COLS
from src.inference_pipeline import preprocess_new_data

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance-api")

app = FastAPI(title="Insurance Coverage Predictor API", version="1.0")

# cached objects loaded on startup
MODELS = {}
ENCODERS = {}
STATS = {}

class SingleRequest(BaseModel):
    # Accept a single quote (same columns as train raw)
    customer_ID: Optional[str] = None
    shopping_pt: Optional[int] = None
    group_size: Optional[int] = None
    car_age: Optional[int] = None
    risk_factor: Optional[int] = None
    age_oldest: Optional[int] = None
    age_youngest: Optional[int] = None
    cost: Optional[float] = None
    state: Optional[str] = None
    location: Optional[str] = None
    car_value: Optional[str] = None
    G: Optional[str] = None
    C_previous: Optional[float] = 0
    duration_previous: Optional[float] = 0


@app.get("/")
def root():
    return {"message": "Insurance API is running"}

@app.on_event("startup")
def load_artifacts():
    global MODELS, ENCODERS, STATS
    # load models
    logger.info("Loading CatBoost models...")
    for t in TARGETS:
        path = os.path.join(MODEL_DIR, f"catboost_{t}.cbm")
        if not os.path.exists(path):
            raise RuntimeError(f"Model missing: {path}")
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(path)
        MODELS[t] = m
    logger.info(f"Loaded {len(MODELS)} models.")

    # load encoders
    enc_path = os.path.join(PROCESSED_DIR, "encoders.joblib")
    if not os.path.exists(enc_path):
        raise RuntimeError("encoders.joblib not found; run training pipeline first.")
    ENCODERS = joblib.load(enc_path)
    logger.info("Loaded encoders.joblib")

    # load stats
    stats_path = os.path.join(PROCESSED_DIR, "input_stats.joblib")
    if os.path.exists(stats_path):
        STATS = joblib.load(stats_path)
        logger.info("Loaded input_stats.joblib")
    else:
        STATS = {}

@app.get("/health")
def health():
    return {"status": "ok"}

def _predict_df(df_raw: pd.DataFrame):
    """Preprocess, create features, predict, return dataframe with preds + customer_ID."""
    # Preprocess using your inference pipeline (handles encoders)
    X = preprocess_new_data(df_raw, ENCODERS)

    # model predictions (per-row)
    preds = {}
    for t, m in MODELS.items():
        out = m.predict(X)
        # ensure 1D numpy
        if hasattr(out, "reshape"):
            out = out.reshape(-1)
        preds[t] = out.astype(int)

    pred_df = pd.DataFrame(preds)
    # attach customer id if present in original df_raw, else index
    if "customer_ID" in df_raw.columns:
        pred_df["customer_ID"] = df_raw["customer_ID"].values
    else:
        pred_df["row_index"] = df_raw.index.values
    return pred_df

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Upload CSV file of raw quotes (same raw schema). Responds with per-row predictions
    and per-customer aggregated (last shopping_pt).
    """
    if file.content_type not in ("text/csv", "application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    content = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    # If customer repeats appear, keep last shopping_pt row per customer
    if "customer_ID" in df_raw.columns and "shopping_pt" in df_raw.columns:
        df_raw = (
            df_raw.sort_values(["customer_ID", "shopping_pt"])
                  .groupby("customer_ID")
                  .tail(1)
                  .reset_index(drop=True)
        )

    preds = _predict_df(df_raw)

    # per-customer aggregation (take last if duplicates remain)
    if "customer_ID" in preds.columns:
        agg = preds.groupby("customer_ID").last().reset_index()
    else:
        agg = preds.copy()

    return {
        "n_input_rows": len(df_raw),
        "predictions_per_row": preds.to_dict(orient="records"),
        "predictions_per_customer": agg.to_dict(orient="records")
    }

@app.post("/predict_single")
def predict_single(payload: SingleRequest):
    """
    Predict for a single customer given JSON body (fields same as CSV columns).
    Returns predictions for that single row.
    """
    d = payload.dict()
    # minimal validation: cost and shopping_pt required for accurate features (but we allow defaults)
    df = pd.DataFrame([d])
    try:
        preds = _predict_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing/prediction error: {e}")

    out = preds.iloc[0].to_dict()
    return out
