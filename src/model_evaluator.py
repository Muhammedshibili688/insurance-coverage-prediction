# src/model_evaluator.py
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

def evaluate_models(
    models: Dict[str, CatBoostClassifier],
    test_last: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
):
    """
    Print accuracy for each target column on hold-out test_last.
    """
    X_test = test_last[feature_cols]

    results = {}
    for targ in target_cols:
        model = models[targ]
        y_true = test_last[targ].astype(int)
        y_pred = model.predict(X_test).reshape(-1)

        acc = accuracy_score(y_true, y_pred)
        results[targ] = acc
        print(f"Accuracy for {targ}: {acc:.5f}")

    return results
