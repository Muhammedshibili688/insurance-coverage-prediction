# src/feature_engineering.py
# src/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import hashlib
import os
import joblib
from src.config import PROCESSED_DIR

from .handle_missing_values import clean_missing



RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET_COLS = ["A_f","B_f","C_f","D_f","E_f","F_f","G_f"]

def cantor_pair(a, b):
    """Cantor pairing on integer arrays/series."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    s = a + b
    return ((s * (s + 1)) // 2 + b).astype(np.int64)

def fast_hash_str(s, mod=100_000):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

def add_hash_interaction(df, col1, col2, new_col, mod=100_000):
    vals = (df[col1].astype(str) + "_" + df[col2].astype(str)).values
    df[new_col] = [fast_hash_str(v, mod=mod) for v in vals]
    return df

def build_quotes_with_targets(raw_train: pd.DataFrame) -> pd.DataFrame:
    """
    From original train.csv → keep only record_type==0 (quotes),
    merge final purchase options A_f..G_f.
    """
    quotes = raw_train[raw_train["record_type"] == 0].copy()
    finals = raw_train[raw_train["record_type"] == 1].copy()

    finals_lbl = finals[["customer_ID", "A","B","C","D","E","F","G"]].copy()
    finals_lbl.columns = ["customer_ID"] + TARGET_COLS

    data = quotes.merge(finals_lbl, on="customer_ID", how="left")
    return data

def add_sequence_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prev option, changed flags, num_options_changed."""
    data = data.sort_values(["customer_ID", "shopping_pt"]).copy()

    opt_cols = ["A","B","C","D","E","F","G"]
    for c in opt_cols:
        prev_col = f"{c}_prev"
        data[prev_col] = (
            data.groupby("customer_ID")[c].shift(1)
                .fillna(data[c])            # for first quote
                .astype(int)
        )
        data[f"{c}_changed"] = (data[c] != data[prev_col]).astype(int)

    data["num_options_changed"] = data[[f"{c}_changed" for c in opt_cols]].sum(axis=1)
    return data

def add_cost_features(data: pd.DataFrame) -> pd.DataFrame:
    """caCost, ppCost, stCost (by state, using all quotes)."""
    data = data.copy()
    data["caCost"] = data["cost"] / (data["car_age"] + 1.0)
    data["ppCost"] = data["cost"] / (data["group_size"] + 1e-6)

    # mean cost per state (not using targets)
    st_cost = data.groupby("state")["cost"].mean()
    global_cost = data["cost"].mean()
    data["stCost"] = data["state"].map(st_cost).fillna(global_cost)

    return data

def encode_categoricals(train: pd.DataFrame,
                        test: pd.DataFrame):
    """
    Label-encode state, location, car_value, G (quoted).
    Fit on train+test concat to avoid unseen labels error.
    """
    train = train.copy()
    test  = test.copy()

    encoders = {}

    # state
    le_s = LabelEncoder()
    all_s = pd.concat([train["state"], test["state"]], axis=0).astype(str)
    le_s.fit(all_s)
    train["state_enc"] = le_s.transform(train["state"].astype(str))
    test["state_enc"]  = le_s.transform(test["state"].astype(str))
    encoders["state"] = le_s

    # location
    le_l = LabelEncoder()
    all_l = pd.concat([train["location"], test["location"]], axis=0).astype(str)
    le_l.fit(all_l)
    train["location_enc"] = le_l.transform(train["location"].astype(str))
    test["location_enc"]  = le_l.transform(test["location"].astype(str))
    encoders["location"] = le_l

    # car_value
    le_c = LabelEncoder()
    all_c = pd.concat([train["car_value"], test["car_value"]], axis=0).astype(str)
    le_c.fit(all_c)
    train["car_value_enc"] = le_c.transform(train["car_value"].astype(str))
    test["car_value_enc"]  = le_c.transform(test["car_value"].astype(str))
    encoders["car_value"] = le_c

    # G (quoted option G)
    le_g = LabelEncoder()
    all_g = pd.concat([train["G"], test["G"]], axis=0).astype(str)
    le_g.fit(all_g)
    train["G_enc"] = le_g.transform(train["G"].astype(str))
    test["G_enc"]  = le_g.transform(test["G"].astype(str))
    encoders["G"] = le_g

    joblib.dump(encoders, os.path.join(PROCESSED_DIR, "encoders.joblib"))

    return train, test, encoders

def add_interaction_features(train: pd.DataFrame,
                             test: pd.DataFrame):
    """
    Cantor + hash interactions using encoded integers.
    """
    train = train.copy()
    test  = test.copy()

    # Cantor pairs
    train["state_G_pair"] = cantor_pair(train["state_enc"], train["G_enc"])
    test["state_G_pair"]  = cantor_pair(test["state_enc"], test["G_enc"])

    train["G_sp_pair"] = cantor_pair(train["G_enc"], train["shopping_pt"])
    test["G_sp_pair"]  = cantor_pair(test["G_enc"], test["shopping_pt"])

    train["state_sp_pair"] = cantor_pair(train["state_enc"], train["shopping_pt"])
    test["state_sp_pair"]  = cantor_pair(test["state_enc"], test["shopping_pt"])

    # hashed interactions
    add_hash_interaction(train, "state_enc", "G_enc", "hash_state_G")
    add_hash_interaction(test,  "state_enc", "G_enc", "hash_state_G")

    add_hash_interaction(train, "location_enc", "G_enc", "hash_location_G")
    add_hash_interaction(test,  "location_enc", "G_enc", "hash_location_G")

    add_hash_interaction(train, "state_enc", "shopping_pt", "hash_state_sp")
    add_hash_interaction(test,  "state_enc", "shopping_pt", "hash_state_sp")

    add_hash_interaction(train, "G_enc", "shopping_pt", "hash_G_sp")
    add_hash_interaction(test,  "G_enc", "shopping_pt", "hash_G_sp")

    return train, test

def get_feature_cols():
    """
    Final feature list (same as notebook, but without leakage cols).
    """
    opt_cols = ["A","B","C","D","E","F","G"]

    con = [
        "group_size","car_age","risk_factor","age_oldest","age_youngest",
        "C_previous","duration_previous","cost","caCost","ppCost","stCost"
    ]

    cat = [
        "state_enc","location_enc","car_value_enc","homeowner","married_couple"
    ]

    conf = [
        "hash_state_G","hash_location_G","hash_state_sp","hash_G_sp",
        "state_G_pair","G_sp_pair","state_sp_pair"
    ]

    prev_cols = [f"{c}_prev" for c in opt_cols]
    changed_cols = [f"{c}_changed" for c in opt_cols]

    extra = prev_cols + changed_cols + ["num_options_changed"]

    return con + cat + conf + extra

