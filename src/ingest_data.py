# src/ingest_data.py
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

TRAIN_PATH = os.path.join(DATA_DIR, "raw/train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "raw/test.csv")   # your comp_test

def load_raw_data(train_path: str = TRAIN_PATH,
                  test_path: str = TEST_PATH):
    """
    Load original Allstate train.csv + test.csv (comp_test) as provided.
    """
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    return train, test
