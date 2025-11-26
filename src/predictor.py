import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier

from src.config import MODEL_DIR, FEATURE_COLS, TARGETS


def load_models() -> dict:
    """
    Load all trained CatBoost models saved in models/
    Returns dictionary: { target_name: CatBoostClassifier }
    """
    models = {}
    for t in TARGETS:
        model_path = os.path.join(MODEL_DIR, f"catboost_{t}.cbm")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file missing: {model_path}")

        model = CatBoostClassifier()
        model.load_model(model_path)
        models[t] = model

    print(f"Loaded {len(models)} models from {MODEL_DIR}")
    return models


def predict(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Predicts A_f … G_f for provided dataframe containing feature columns.
    Returns dataframe with predictions.
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required feature columns: {missing}")

    preds = {}
    for t, model in models.items():
        preds[t] = model.predict(df[FEATURE_COLS]).flatten()

    pred_df = pd.DataFrame(preds)
    return pred_df


if __name__ == "__main__":
    # Example usage — run directly
    print("🚀 Loading models...")
    models = load_models()

    sample_path = "data/raw/comp_test.csv"
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"❌ Test file not found: {sample_path}")

    print(f"📥 Loading sample data from {sample_path}")
    sample = pd.read_csv(sample_path)

    # IMPORTANT — sample must already be feature engineered
    # otherwise predictions will fail
    print("⚠️ Ensure sample file has feature engineered columns before predicting")

    output = predict(sample, models)

    print("\nSample Predictions:")
    print(output.head())
