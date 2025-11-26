# src/data_splitter.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .ingest_data import load_raw_data
from .feature_engineering import (
    build_quotes_with_targets,
    clean_missing,
    add_sequence_features,
    add_cost_features,
    encode_categoricals,
    add_interaction_features,
    get_feature_cols,
)

RANDOM_STATE = 42

def prepare_train_test_last_step(test_size: float = 0.2):
    """
    Full pipeline: raw CSV → cleaned + feature engineered → 
    train/test split → final quote per customer.
    """
    raw_train, _ = load_raw_data()

    # 1) merge quotes + final targets
    data = build_quotes_with_targets(raw_train)

    # 2) missing values
    data = clean_missing(data)

    # 3) sequence + cost features
    data = add_sequence_features(data)
    data = add_cost_features(data)

    # 4) split customers
    unique_customers = data["customer_ID"].unique()
    train_cust, test_cust = train_test_split(
        unique_customers,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train = data[data["customer_ID"].isin(train_cust)].copy()
    test  = data[data["customer_ID"].isin(test_cust)].copy()

    # 5) encode categoricals
    train, test, _ = encode_categoricals(train, test)

    # 6) add interaction features
    train, test = add_interaction_features(train, test)

    # 7) take last quote per customer
    def last_step(df):
        idx = df.groupby("customer_ID")["shopping_pt"].idxmax()
        return df.loc[idx].reset_index(drop=True)

    train_last = last_step(train)
    test_last  = last_step(test)

    feature_cols = get_feature_cols()
    target_cols  = ["A_f","B_f","C_f","D_f","E_f","F_f","G_f"]

    return train_last, test_last, feature_cols, target_cols
