# src/model_building.py
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

RANDOM_STATE = 42

def train_catboost_models(
    train_last: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    verbose: int = 100,
) -> Dict[str, CatBoostClassifier]:
    """
    Train one CatBoostClassifier per target column.
    """
    X = train_last[feature_cols]
    models = {}

    # we treat everything as numerical (we already encoded categoricals)
    cat_features = []   # indices of categorical features (none now)

    for targ in target_cols:
        print(f"\n===== Training CatBoost for {targ} =====")
        y = train_last[targ].astype(int)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            depth=6,
            learning_rate=0.1,
            iterations=300,
            random_seed=RANDOM_STATE,
            eval_metric="Accuracy",
            verbose=verbose,
            task_type="CPU",   # change to "GPU" if you have CatBoost GPU installed
        )

        model.fit(X, y)
        models[targ] = model

    return models
