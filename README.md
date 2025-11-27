🛡️ Insurance Coverage Recommendation System (A–G Options)

Predicting final insurance coverage selections using end-to-end Machine Learning, feature engineering, explainable AI, and deployable infrastructure.

<p align="left"> <img src="https://img.shields.io/badge/ML-CatBoost-blue?style=flat-square"> <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=flat-square"> <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=flat-square"> <img src="https://img.shields.io/badge/Deployment-Docker-yellow?style=flat-square"> </p>
📌 Project Overview

This project predicts the final insurance coverage options (A_f to G_f) selected by customers during the quoting process.
It is built as a full production-style ML system, not just a notebook model.

Includes

✔ Full ML pipeline (cleaning → feature engineering → training → evaluation)
✔ 7 CatBoost models (multi-target classification)
✔ Streamlit UI for batch & single-customer prediction
✔ SHAP explainability (global + local)
✔ Evaluation dashboards (accuracy, confusion matrix, feature importance)
✔ FastAPI backend for deployment
✔ Docker container for production
✔ Clean modular codebase with reusable components

A perfect portfolio project for Data Science • ML Engineering • MLOps roles.

🧱 Project Architecture
Insurance Coverage Prediction/
│
├── app/
│   └── streamlit_app.py          # Streamlit UI
│
├── api/
│   ├── app.py                    # FastAPI backend
│   └── Dockerfile                # Fully deployable container
│
├── src/
│   ├── config.py
│   ├── inference_pipeline.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluator.py
│   └── data_splitter.py
│
├── models/
│   ├── catboost_A_f.cbm
│   ├── catboost_B_f.cbm
│   └── ... catboost_G_f.cbm
│
├── data/
│   └── processed/
│       ├── train_last.parquet
│       ├── test_last.parquet
│       ├── encoders.joblib
│       ├── input_stats.joblib
│       └── train_targets.parquet
│
├── requirements.txt
└── README.md

💡 Key Features
🔹 1. Full ML Pipeline

Generates engineered features:

Previous selections (A_prev, B_prev, …)

Change indicators (A_changed, …)

Cost ratios (caCost, stCost)

Interaction features

Label encoding + validation

Trains 7 CatBoost models

Stores:

Encoders

Stats for UI validation

Final models

Train/test splits

🔹 2. Streamlit App
🧾 CSV Batch Prediction

Upload → Auto-clean → Predict → Download.

👤 Single-Customer Prediction

Controlled inputs (dropdowns + ranges)

SHAP waterfall plots explaining WHY each option was chosen.

📊 Model Evaluation Dashboard

Accuracy per target

Confusion matrix

Feature importance (sorted)

Global SHAP summary

🔹 3. FastAPI Backend + Docker

Exposes endpoints:

POST /predict_one
POST /predict_batch
GET  /health


Production-ready using Docker:

docker build -t insurance-api -f api/Dockerfile .
docker run -p 8000:8000 insurance-api


Interactive API docs:
👉 http://localhost:8000/docs

🚀 How to Run Locally
1️⃣ Create Environment
python -m venv insurance_venv
insurance_venv\Scripts\activate   # Windows
source insurance_venv/bin/activate  # macOS/Linux

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app/streamlit_app.py


Open in browser:
👉 http://localhost:8501

🧪 Train the ML Models

To train all 7 models and generate artifacts:

python pipeline/training_pipeline.py


Outputs:

CatBoost models

Encoders.joblib

Input_stats.joblib

Train/test parquet files

Metadata

🌐 Deployment (FastAPI + Docker)
Build container
docker build -t insurance-api -f api/Dockerfile .

Run
docker run -p 8000:8000 insurance-api

Open

👉 http://localhost:8000/docs

📊 Example Prediction Output
{
  "A_f": 2,
  "B_f": 3,
  "C_f": 1,
  "D_f": 1,
  "E_f": 4,
  "F_f": 1,
  "G_f": 2
}


SHAP explains the reasoning for each option.

🌱 Future Improvements

Optuna hyperparameter tuning

Transformer-based sequence models

Add authentication

Deploy Streamlit + API on cloud (Railway / Render / HF Spaces)

CI/CD with GitHub Actions

Monitoring (Prometheus + Grafana)

👨‍💻 Author – Muhammed Shibili

Machine Learning Engineer
🔥 Passion for production-grade AI systems
📫 Reach out for collaboration anytime!

If you found this helpful:

⭐ Star the repo
🔗 Share on LinkedIn
🍀 Add to your ML portfolio