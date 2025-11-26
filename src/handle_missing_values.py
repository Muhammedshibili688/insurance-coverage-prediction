# src/handle_missing_values.py
import pandas as pd

def clean_missing(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Basic NA handling consistent with notebook.
    Works for both train quotes+finals and comp_test.
    """
    df = df.copy()

    # These exist in both train and test
    df["C_previous"] = df["C_previous"].fillna(0)
    df["duration_previous"] = df["duration_previous"].fillna(0)

    if "location" in df.columns:
        df["location"] = df["location"].fillna(-1)

    # risk_factor: ffill per customer then -1 for remaining
    if "risk_factor" in df.columns:
        df["risk_factor"] = (
            df.groupby("customer_ID")["risk_factor"]
              .ffill()
              .fillna(-1)
        )

    if "car_value" in df.columns:
        df["car_value"] = df["car_value"].fillna("missing")

    return df
