import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "raw", "test.csv")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGETS = ["A_f","B_f","C_f","D_f","E_f","F_f","G_f"]

FEATURE_COLS = [
    "group_size","car_age","risk_factor","age_oldest","age_youngest",
    "C_previous","duration_previous","cost","caCost","ppCost","stCost",
    "state_enc","location_enc","car_value_enc","homeowner","married_couple",
    "hash_state_G","hash_location_G","hash_state_sp","hash_G_sp",
    "state_G_pair","G_sp_pair","state_sp_pair",
    "A_prev","B_prev","C_prev","D_prev","E_prev","F_prev","G_prev",
    "A_changed","B_changed","C_changed","D_changed","E_changed","F_changed","G_changed",
    "num_options_changed"
]

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
