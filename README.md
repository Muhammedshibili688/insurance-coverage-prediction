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

Generates powerful engineered features:

Previous options (A_prev, B_prev, …)

Change indicators (A_changed, …)

Pair interactions (G_sp_pair)

Cost-based ratios (caCost, ppCost, stCost)

Hash interactions

Label encoding + data validation

CatBoost models for A_f → G_f

All artifacts saved for inference

🔹 2. Streamlit Application
🧾 CSV Batch Prediction

Upload any CSV → process → get predictions → export results.

👤 Single Customer Prediction

Interactive form → real-time model output

SHAP waterfall plots showing why the model chose an option.

📊 Model Evaluation Dashboard

Accuracy table

Confusion matrix (select target)

Feature importance (top 15)

Global SHAP summary plot

This makes the model fully transparent & business-friendly.

🔹 3. FastAPI Backend + Docker

A lightweight API for:

External tools

Websites

Streamlit production mode

Future cloud deployment

Dockerfile included for easy deployment.

🚀 How to Run
1️⃣ Setup
python -m venv insurance_venv
insurance_venv\Scripts\activate   # Windows
source insurance_venv/bin/activate  # Linux/macOS

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit
streamlit run app/streamlit_app.py


Open in browser:

http://localhost:8501

🧪 Training Pipeline

Train everything end-to-end:

python pipeline/training_pipeline.py


Outputs:

Trained CatBoost models

Encoders

Processed parquet files

Feature stats

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

(Ready for Railway / Render / HuggingFace Spaces)

🌐 Deployment (Docker + FastAPI)

Build container:

docker build -t insurance-api -f api/Dockerfile .


Run:

docker run -p 8000:8000 insurance-api


Open docs:

http://localhost:8000/docs

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


With SHAP explanation provided below each prediction.

🌱 Future Improvements

Add Optuna hyperparameter tuning

Introduce deep learning / transformer-based predictors

Add user authentication

Deploy Streamlit + API on cloud

Build CI/CD pipeline with GitHub Actions

Add monitoring (Prometheus + Grafana)

👨‍💻 Author

Muhammed Shibili
💼 Machine Learning Engineer
🔥 Passionate about production-grade AI systems
📫 Reach me anytime for collaboration!

⭐ If you found this project useful

👉 Star the repository
👉 Share it on LinkedIn
👉 Use it as a reference for your ML portfolio