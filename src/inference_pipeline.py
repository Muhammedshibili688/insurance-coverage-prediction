import pandas as pd
import numpy as np
import joblib
import os
from src.config import FEATURE_COLS, PROCESSED_DIR

# ----------------------------
# Encoders helper
# ----------------------------
def safe_label_encode(series, encoder):
    series = series.astype(str).fillna("nan")

    classes = list(encoder.classes_.astype(str))
    if "unknown" not in classes:
        classes.append("unknown")
        encoder.classes_ = np.array(classes)

    known = set(encoder.classes_.astype(str))
    mapped = series.apply(lambda x: x if x in known else "unknown")
    return encoder.transform(mapped.values)


# --------------------------------------------------------
# Main preprocessing function for new external datasets
# --------------------------------------------------------
def preprocess_new_data(df_raw, encoders):
    """
    Apply same feature engineering used in training.
    Handles: missing values, sequence features, cost features,
    encoding, hashing interactions, Cantor pairs, etc.
    Parameters:
        df_raw   : DataFrame (raw quotes)
        encoders : dict of LabelEncoders loaded from encoders.joblib
    """
    df = df_raw.copy()

    # ========== Missing values ==========
    df["C_previous"] = df["C_previous"].fillna(0) if "C_previous" in df else 0
    df["duration_previous"] = df["duration_previous"].fillna(0) if "duration_previous" in df else 0
    df["location"] = df["location"].fillna(-1) if "location" in df else "missing"
    df["car_value"] = df["car_value"].fillna("missing") if "car_value" in df else "missing"

    if "risk_factor" in df.columns:
        df["risk_factor"] = (
            df.groupby("customer_ID")["risk_factor"]
            .ffill()
            .fillna(-1)
        )
    else:
        df["risk_factor"] = -1

    # ========== Sequence Features ==========
    opt_cols = ["A","B","C","D","E","F","G"]

    if {"customer_ID","shopping_pt"}.issubset(df.columns):
        df = df.sort_values(["customer_ID","shopping_pt"]).reset_index(drop=True)
        for c in opt_cols:
            prev_col = f"{c}_prev"
            if c in df.columns:
                df[prev_col] = df.groupby("customer_ID")[c].shift(1).fillna(df[c]).astype(int)
                df[f"{c}_changed"] = (df[c] != df[prev_col]).astype(int)
            else:
                df[c] = 0
                df[prev_col] = 0
                df[f"{c}_changed"] = 0
        df["num_options_changed"] = df[[f"{c}_changed" for c in opt_cols]].sum(axis=1)
    else:
        for c in opt_cols:
            df[c] = df.get(c, 0)
            df[f"{c}_prev"] = df[c]
            df[f"{c}_changed"] = 0
        df["num_options_changed"] = 0

    # ========== Cost Features ==========
    df["caCost"] = df["cost"] / (df.get("car_age", 0) + 1.0)
    df["ppCost"] = df["cost"] / (df.get("group_size", 1) + 1e-6)

    # Load stCost map from file if available
    stcost_path = os.path.join(PROCESSED_DIR, "stcost.joblib")
    if os.path.exists(stcost_path):
        stcost_map = joblib.load(stcost_path)
        global_cost = stcost_map.get("_global", df["cost"].mean())
        df["stCost"] = df["state"].astype(str).map(stcost_map).fillna(global_cost)
    else:
        df["stCost"] = df.groupby("state")["cost"].transform("mean").fillna(df["cost"].mean())

    # ========== Encoding ==========
    if encoders:
        df["state_enc"] = safe_label_encode(df.get("state", pd.Series("missing")), encoders["state"])
        df["location_enc"] = safe_label_encode(df.get("location", pd.Series("missing")), encoders["location"])
        df["car_value_enc"] = safe_label_encode(df.get("car_value", pd.Series("missing")), encoders["car_value"])
        df["G_enc"] = safe_label_encode(df.get("G", pd.Series("missing")), encoders["G"])
    else:
        df["state_enc"] = pd.factorize(df.get("state", "missing"))[0]
        df["location_enc"] = pd.factorize(df.get("location", "missing"))[0]
        df["car_value_enc"] = pd.factorize(df.get("car_value", "missing"))[0]
        df["G_enc"] = pd.factorize(df.get("G", "missing"))[0]

    # ========== Interactions ==========
    def fast_hash(v):
        return int(abs(hash(v)) % 100000)

    df["hash_state_G"] = (df["state_enc"].astype(str) + "_" + df["G_enc"].astype(str)).apply(fast_hash)
    df["hash_location_G"] = (df["location_enc"].astype(str) + "_" + df["G_enc"].astype(str)).apply(fast_hash)

    if "shopping_pt" in df.columns:
        df["hash_state_sp"] = (df["state_enc"].astype(str) + "_" + df["shopping_pt"].astype(str)).apply(fast_hash)
        df["hash_G_sp"] = (df["G_enc"].astype(str) + "_" + df["shopping_pt"].astype(str)).apply(fast_hash)
    else:
        df["hash_state_sp"] = 0
        df["hash_G_sp"] = 0

    # Cantor
    def cantor(a, b):
        a = a.astype(int)
        b = b.astype(int)
        s = a + b
        return ((s * (s + 1)) // 2 + b).astype(int)

    df["state_G_pair"] = cantor(df["state_enc"], df["G_enc"])
    df["G_sp_pair"] = cantor(df["G_enc"], df.get("shopping_pt", 0))
    df["state_sp_pair"] = cantor(df["state_enc"], df.get("shopping_pt", 0))

    # ========== Ensure all features exist ==========
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLS]
