import argparse
import pandas as pd
import os
from catboost import CatBoostClassifier
import joblib

from src.inference_pipeline import preprocess_new_data
from src.config import MODEL_DIR, TARGETS, FEATURE_COLS


def load_models():
    models = {}
    for t in TARGETS:
        model = CatBoostClassifier()
        model.load_model(os.path.join(MODEL_DIR, f"catboost_{t}.cbm"))
        models[t] = model
    return models


def predict_file(path):
    print(f">>> Reading input CSV: {path}")
    df_raw = pd.read_csv(path)

    print(">>> Applying preprocessing (FE + encoders + last-step)...")
    X = preprocess_new_data(df_raw)

    print(">>> Loading trained models...")
    models = load_models()

    print(">>> Predicting final plan options...")
    preds = {}

    for t, model in models.items():
        preds[t] = model.predict(X).flatten().astype(int)

    out = pd.DataFrame(preds)
    out["customer_ID"] = df_raw.groupby("customer_ID").tail(1)["customer_ID"].values

    out.to_csv("Predicted.csv", index=False)
    print("\nSaved: Predicted.csv")
    print(out.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    predict_file(args.input)
