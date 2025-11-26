# pipeline/training_pipeline.py
from pathlib import Path
import joblib
import numpy as np
import os

from src.data_splitter import prepare_train_test_last_step
from src.model_building import train_catboost_models
from src.model_evaluator import evaluate_models

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

def run_training_pipeline(test_size: float = 0.2):
    print(">>> Preparing data...")
    train_last, test_last, feature_cols, target_cols = prepare_train_test_last_step(
        test_size=test_size
    )

    # -----------------------------------------------------------
    # SAVE ENGINEERED DATA
    # -----------------------------------------------------------
    train_last.to_parquet(PROCESSED_DIR / "train_last.parquet", index=False)
    test_last.to_parquet(PROCESSED_DIR / "test_last.parquet", index=False)
    print("✔ Saved train_last & test_last")

    # -----------------------------------------------------------
    # SAVE TARGET-ONLY PARQUET
    # -----------------------------------------------------------
    target_cols = ["A_f","B_f","C_f","D_f","E_f","F_f","G_f"]

    train_targets = train_last[["customer_ID"] + target_cols]
    test_targets  = test_last[["customer_ID"] + target_cols]

    train_targets.to_parquet(PROCESSED_DIR / "train_targets.parquet", index=False)
    test_targets.to_parquet(PROCESSED_DIR / "test_targets.parquet", index=False)

    print("✔ Saved train_targets & test_targets")

    # -----------------------------------------------------------
    # CREATE input_stats.joblib FOR STREAMLIT VALIDATION
    # -----------------------------------------------------------
    input_stats = {
        "states": sorted(train_last["state"].astype(str).unique()),
        "locations": sorted(train_last["location"].astype(str).unique()),
        "car_values": sorted(train_last["car_value"].astype(str).unique()),
        "G_values": sorted(train_last["G"].astype(str).unique()),

        "numeric_ranges": {
            col: {
                "min": float(train_last[col].min()),
                "max": float(train_last[col].max())
            }
            for col in [
                "shopping_pt", "group_size", "car_age", "risk_factor",
                "age_oldest", "age_youngest", "cost"
            ]
        }
    }

    joblib.dump(input_stats, PROCESSED_DIR / "input_stats.joblib")
    print("✔ Saved input_stats.joblib")

    # -----------------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------------
    print("\n>>> Training models...")
    models = train_catboost_models(train_last, feature_cols, target_cols)

    # -----------------------------------------------------------
    # EVALUATE MODELS
    # -----------------------------------------------------------
    print("\n>>> Evaluating models...")
    results = evaluate_models(models, test_last, feature_cols, target_cols)

    print("\nFinal accuracies:")
    for k, v in results.items():
        print(f"  {k}: {v:.5f}")

    # -----------------------------------------------------------
    # SAVE MODELS + META
    # -----------------------------------------------------------
    print("\n>>> Saving models...")
    for targ, model in models.items():
        model.save_model(MODELS_DIR / f"catboost_{targ}.cbm")

    joblib.dump(
        {"feature_cols": feature_cols, "target_cols": target_cols},
        MODELS_DIR / "meta.joblib"
    )

    print("🎉 All models + metadata saved!")
