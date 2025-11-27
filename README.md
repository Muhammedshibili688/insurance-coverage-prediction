🛡️ Insurance Coverage Recommendation System (A–G Options)

Predicting final insurance coverage selections using advanced ML, engineered features, and explainable AI.

📌 Project Summary

This project builds a production-style ML system that predicts the final insurance coverage options (A_f to G_f) selected by customers during the quoting process.

It includes:

✔ Full ML pipeline (cleaning → feature engineering → training → evaluation)

✔ 7 CatBoost models (one for each target A–G)

✔ A Streamlit app for user interaction

✔ SHAP-based explainability

✔ CSV batch prediction

✔ Single customer prediction

✔ Confusion matrix, accuracy dashboard, feature importance plots

✔ FastAPI backend + Docker setup

Perfect for showcasing end-to-end ML engineering skills.

🧱 Project Architecture
Insurance Coverage Prediction/
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   ├── app.py
│   └── Dockerfile
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
│   └── catboost_A_f.cbm ... catboost_G_f.cbm
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_last.parquet
│       ├── test_last.parquet
│       ├── encoders.joblib
│       ├── input_stats.joblib
│       └── train_targets.parquet
│
├── requirements.txt
└── README.md

💡 Features
🔹 1. Full ML Pipeline

Cleans raw customer quote data

Feature engineering including:

Previous selections (A_prev, …)

Change indicators (A_changed, …)

Cost-based ratios (caCost, ppCost, stCost)

State & location features

Hash interactions

Label encoding + validation

Trains 7 CatBoost models (A_f → G_f)

Saves all artifacts for inference

🔹 2. Streamlit Application
🧾 CSV Batch Prediction

Upload CSV → preprocess → model inference → download predictions.

👤 Single Customer Prediction

Real-time form-based prediction with:

SHAP waterfall explanation

Clean UI

Strict input validation based on training distribution

📊 Evaluation Dashboard

Accuracy scores

Confusion matrices

Feature importance plot (Top 15 features)

Global SHAP summary

🔹 3. FastAPI Backend + Docker

/predict_one

/predict_batch

/health

Dockerized for deployment on:

Render

Railway

AWS EC2

HuggingFace Spaces

🚀 How to Run
1️⃣ Setup Environment
python -m venv insurance_venv
insurance_venv\Scripts\activate   # Windows
source insurance_venv/bin/activate  # Linux/macOS

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Launch Streamlit App
streamlit run app/streamlit_app.py


Open in browser:

👉 http://localhost:8501

🧪 Training Pipeline

Run the full training workflow:

python pipeline/training_pipeline.py


This generates:

Trained CatBoost models

Encoders

Processed parquet files

Input feature stats

Evaluation metrics

🛠 Tech Stack
Machine Learning

CatBoost

Scikit-learn

Pandas / NumPy

SHAP

Application

Streamlit

FastAPI

Uvicorn

DevOps

Docker

Ready for Railway / Render / HuggingFace

🌐 Deployment (Docker + FastAPI)
Build Container
docker build -t insurance-api -f api/Dockerfile .

Run API
docker run -p 8000:8000 insurance-api

Open API Docs

👉 http://localhost:8000/docs

📊 Example Model Output
{
  "A_f": 2,
  "B_f": 3,
  "C_f": 1,
  "D_f": 1,
  "E_f": 4,
  "F_f": 1,
  "G_f": 2
}


Each prediction is followed by SHAP explanation in Streamlit.

🌱 Future Improvements

Optuna hyperparameter tuning

Transformer-based models

Authentication for API

CI/CD pipeline with GitHub Actions

Full cloud deployment

Monitoring (Prometheus + Grafana)

👨‍💻 Author

Muhammed Shibili
💼 Machine Learning Engineer
🔥 Passionate about production-grade AI systems
📫 Open to collaborations

⭐ Support

If this project helped you:

👉 Star the repository
👉 Share it on LinkedIn
👉 Add it to your ML portfolio