# ============================================================
# Streamlit App: Insurance Coverage Predictor (WITH EVAL + SHAP)
# ============================================================

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# ------------------------------------------------------------
# FIX PYTHON PATH
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------
from src.config import FEATURE_COLS, TARGETS, MODEL_DIR, PROCESSED_DIR
from src.inference_pipeline import preprocess_new_data

# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------
@st.cache_resource
def load_encoders():
    enc_path = os.path.join(PROCESSED_DIR, "encoders.joblib")
    return joblib.load(enc_path)

@st.cache_resource
def load_models():
    models = {}
    from catboost import CatBoostClassifier
    for t in TARGETS:
        model = CatBoostClassifier()
        model.load_model(os.path.join(MODEL_DIR, f"catboost_{t}.cbm"))
        models[t] = model
    return models

@st.cache_resource
def load_stats():
    return joblib.load(os.path.join(PROCESSED_DIR, "input_stats.joblib"))


encoders = load_encoders()
models = load_models()
stats = load_stats()

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Insurance Predictor", layout="wide")
st.title("🛡️ Insurance Coverage Predictor — Final Options A–G")

tab1, tab2, tab3 = st.tabs(
    ["📂 Upload CSV", "👤 Single Customer Prediction", "📊 Model Evaluation"]
)

# ============================================================
# TAB 1 — CSV UPLOAD
# ============================================================
with tab1:
    st.header("📤 Upload New Quote CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    df_raw = None
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

    if df_raw is not None:
        # dedupe last step per customer
        if "customer_ID" in df_raw.columns and "shopping_pt" in df_raw.columns:
            df_raw = (
                df_raw.sort_values(["customer_ID", "shopping_pt"])
                      .groupby("customer_ID")
                      .tail(1)
                      .reset_index(drop=True)
            )

        st.write("### 🔍 Preview")
        st.dataframe(df_raw.head())

        X = preprocess_new_data(df_raw, encoders)
        st.write("Features shape:", X.shape)

        if st.button("Predict from CSV"):
            preds = {}
            for t, model in models.items():
                raw_pred = model.predict(X)
                arr = np.array(raw_pred).reshape(-1)          # <- CRASH FIX
                preds[t] = arr.astype(int)

            pred_df = pd.DataFrame(preds)
            if "customer_ID" in df_raw.columns:
                pred_df["customer_ID"] = df_raw["customer_ID"].values

            st.success("✅ Predictions ready")
            st.dataframe(pred_df.head(50))

            st.download_button(
                "Download Predictions",
                pred_df.to_csv(index=False).encode(),
                "predictions.csv",
            )

# ============================================================
# TAB 2 — SINGLE CUSTOMER PREDICTION + SHAP
# ============================================================
with tab2:
    st.header("👤 Single Customer Prediction (with SHAP)")

    state_options = list(encoders["state"].classes_)
    car_values = list(encoders["car_value"].classes_)
    G_options = list(encoders["G"].classes_)
    loc_min, loc_max = 10000, 99999

    with st.form("single_form"):
        c1, c2 = st.columns(2)

        with c1:
            customer_id = st.text_input("Customer ID", "temp_user")
            shopping_pt = st.number_input("Shopping Point", 1, 12, 1)
            group_size = st.number_input(
                "Group Size",
                int(stats["numeric_ranges"]["group_size"]["min"]),
                int(stats["numeric_ranges"]["group_size"]["max"]),
                int(stats["numeric_ranges"]["group_size"]["min"]),
            )
            car_age = st.number_input(
                "Car Age",
                int(stats["numeric_ranges"]["car_age"]["min"]),
                int(stats["numeric_ranges"]["car_age"]["max"]),
                int(stats["numeric_ranges"]["car_age"]["min"]),
            )
            risk_factor = st.selectbox("Risk Factor", [-1, 0, 1, 2, 3, 4, 5, 6])

        with c2:
            age_oldest = st.number_input(
                "Age Oldest",
                int(stats["numeric_ranges"]["age_oldest"]["min"]),
                int(stats["numeric_ranges"]["age_oldest"]["max"]),
                int(stats["numeric_ranges"]["age_oldest"]["min"]),
            )
            age_youngest = st.number_input(
                "Age Youngest",
                int(stats["numeric_ranges"]["age_youngest"]["min"]),
                int(stats["numeric_ranges"]["age_youngest"]["max"]),
                int(stats["numeric_ranges"]["age_youngest"]["min"]),
            )
            cost = st.number_input(
                "Cost",
                float(stats["numeric_ranges"]["cost"]["min"]),
                float(stats["numeric_ranges"]["cost"]["max"]),
                float(stats["numeric_ranges"]["cost"]["min"]),
            )
            state = st.selectbox("State", state_options)
            location = st.number_input("Location", loc_min, loc_max, loc_min)
            car_value = st.selectbox("Car Value", car_values)
            G = st.selectbox("Initial G", G_options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {
            "customer_ID": customer_id,
            "shopping_pt": shopping_pt,
            "group_size": group_size,
            "car_age": car_age,
            "risk_factor": risk_factor,
            "age_oldest": age_oldest,
            "age_youngest": age_youngest,
            "cost": cost,
            "state": state,
            "location": str(location),
            "car_value": car_value,
            "G": G,
            "C_previous": 0,
            "duration_previous": 0,
        }

        df1 = pd.DataFrame([row])
        X1 = preprocess_new_data(df1, encoders)

        # prediction
        pred_result = {}
        for t, model in models.items():
            p = model.predict(X1)
            pred_result[t] = int(np.array(p).reshape(-1)[0])

        st.success("🎯 Prediction Complete")
        st.json(pred_result)

        # ------------ SHAP (only one target at a time) ------------
        st.subheader("🔍 SHAP Explanation (Why this prediction?)")
        shap_target = st.selectbox("Choose target to explain", TARGETS)

        model = models[shap_target]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X1)

        # handle binary vs multiclass
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[0][0])  # class 0, sample 0
        else:
            sv = np.array(shap_vals[0])

        exp = shap.Explanation(
            values=sv,
            base_values=np.array(explainer.expected_value).flatten()[0],
            data=X1.iloc[0],
            feature_names=FEATURE_COLS,
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        shap.waterfall_plot(exp, max_display=15)
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# TAB 3 — MODEL EVALUATION
# ============================================================
with tab3:
    st.header("📊 Model Evaluation Dashboard")

    test_last = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_last.parquet"))
    y_true = test_last[TARGETS]
    X_eval = test_last[FEATURE_COLS]

    # -------- ACCURACY --------
    st.subheader("Accuracy per Target")
    acc = {t: accuracy_score(y_true[t], models[t].predict(X_eval)) for t in TARGETS}
    st.dataframe(pd.DataFrame(acc.items(), columns=["Target", "Accuracy"]))

    # -------- CONFUSION MATRIX --------
    st.subheader("Confusion Matrix")
    choose_target = st.selectbox("Pick Target", TARGETS)

    cm = confusion_matrix(y_true[choose_target], models[choose_target].predict(X_eval))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    ax.set_title(f"Confusion Matrix — {choose_target}")
    st.pyplot(fig)
    plt.close(fig)

    # -------- FEATURE IMPORTANCE PLOT --------
    st.subheader("Feature Importance (Top 15)")
    t_imp = st.selectbox("Target for Importance", TARGETS)
    model_imp = models[t_imp]

    imp = model_imp.get_feature_importance()
    fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
    fi = fi.sort_values("importance", ascending=False).head(15)  # descending

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=fi, y="feature", x="importance", ax=ax)
    ax.set_title(f"Top 15 Features — {t_imp}")

    # numeric labels on bars
    max_val = fi["importance"].max()
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + max_val * 0.01,
            p.get_y() + p.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=8,
        )

    st.pyplot(fig)
    plt.close(fig)

    # -------- GLOBAL SHAP SUMMARY (FIXED FOR CATBOOST) --------
    st.subheader("🌍 Global SHAP Feature Importance")

    # Let user select target for SHAP
    shap_target = st.selectbox("Target for Global SHAP", TARGETS, key="shap_target")

    model_g = models[shap_target]  # selected model

    # sample for performance
    X_sample = X_eval.sample(n=min(300, len(X_eval)), random_state=42)

    # shap explainer
    explainer_g = shap.TreeExplainer(model_g)
    shap_values_raw = explainer_g.shap_values(X_sample)


    # ---- FIX SHAP FOR ALL CATBOOST OUTPUT SHAPES ----
    def fix_catboost_shap(sv):
        import numpy as np

        # Case 1,2: SHAP returns list
        if isinstance(sv, list):
            # binary classifier (2 outputs)
            if len(sv) > 1 and isinstance(sv[1], np.ndarray):
                return sv[1]
            return sv[0]

        # Case 3: already 2D
        if sv.ndim == 2:
            return sv

        # Case 4: 3D bugged output → reduce final axis
        return sv.mean(axis=-1)


    shap_values_fixed = fix_catboost_shap(shap_values_raw)


    # ---- FINAL SUMMARY PLOT ----
    fig = plt.figure(figsize=(6, 4))
    shap.summary_plot(shap_values_fixed, X_sample, show=False, max_display=15)
    st.pyplot(fig)
    plt.clf()


st.caption("Built with ❤️ using CatBoost + SHAP + full feature pipeline")
