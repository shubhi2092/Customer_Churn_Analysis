# 📉 ChurnIQ — Customer Churn Prediction Web App

A full-stack machine learning web application built with **Flask** that trains **6 ML models simultaneously** on a telecom customer churn dataset, compares their performance, and predicts churn for individual customers.

---

## 🌐 Live Demo

> Deploy this yourself for free — see the [Deployment](#-deployment) section below.

---

## 🖼️ App Preview

The app has **4 tabs:**

| Tab | What it shows |
|---|---|
| 📊 Overview | Dataset summary + EDA charts |
| 🏆 Model Comparison | All 6 models compared side by side |
| 🔍 Model Details | Feature importance, confusion matrix, ROC curve per model |
| 🔮 Predict | Predict churn for any customer using any model |

---

## 📁 Project Structure

```
churn_flask_app/
│
├── app.py                  ← Flask backend (routes + ML training)
├── requirements.txt        ← Python libraries needed
├── Procfile                ← For deployment (Render / Railway)
└── templates/
    └── index.html          ← Frontend (HTML + CSS + JavaScript)
```

---

## 🤖 Models Used

All 6 models are trained at once when you upload your dataset:

| Model | Type |
|---|---|
| Logistic Regression | Linear classifier |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| XGBoost | Optimized boosting |
| LightGBM | Fast boosting by Microsoft |

The best model is automatically selected based on **ROC-AUC score**.

---

## 📊 Dataset

This app is built for the **IBM Telco Customer Churn** dataset.

- **Rows:** 7,043 customers
- **Columns:** 21 features
- **Target:** `Churn` (Yes / No)
- **Download:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## ⚙️ How It Works

```
User uploads CSV
      ↓
Flask cleans data + creates new features
      ↓
SMOTE balances the imbalanced classes
      ↓
All 6 models train simultaneously
      ↓
App shows EDA charts + model comparison
      ↓
User fills predict form → all 6 models predict instantly
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Main language |
| Flask | Web framework (backend) |
| Pandas & NumPy | Data cleaning & processing |
| Scikit-learn | ML models + evaluation |
| XGBoost | XGBoost model |
| LightGBM | LightGBM model |
| Imbalanced-learn | SMOTE for class balancing |
| Chart.js | Interactive charts in browser |
| HTML + CSS + JS | Frontend UI |

---

## ▶️ How to Run Locally

### Step 1 — Clone or download the project

```bash
git clone https://github.com/yourusername/churn-flask-app.git
cd churn-flask-app
```

Or just unzip the downloaded folder.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the app

```bash
python app.py
```

### Step 4 — Open in browser

```
http://localhost:5000
```

### Step 5 — Upload your CSV

Upload the Telco Customer Churn CSV file and all 6 models will train automatically!

---

## 🚀 Deployment

### Option 1 — Render (Free, Recommended)

1. Push your project to **GitHub**
2. Go to [render.com](https://render.com) and sign up free
3. Click **New Web Service** → connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free
5. Click **Deploy** — you get a live URL like `yourapp.onrender.com`

> ⚠️ Free tier sleeps after 15 mins of inactivity. First load may take ~30 seconds.

### Option 2 — Railway

1. Go to [railway.app](https://railway.app) → Login with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. It auto-detects Flask and deploys using the `Procfile`

### Option 3 — Hugging Face Spaces

Good for showcasing to the data science community.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create a new Space → choose **Docker** SDK
3. Upload your project files

---

## 📦 requirements.txt

```
flask>=2.3.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
gunicorn
```

---

## 📋 Features at a Glance

- ✅ Upload any CSV — model trains instantly
- ✅ All 6 models trained and compared automatically
- ✅ Interactive charts (ROC curves, confusion matrix, feature importance)
- ✅ Predict churn for any customer using any model
- ✅ All 6 models predict side by side for comparison
- ✅ Risk level label — High / Medium / Low
- ✅ Dark themed, responsive UI
- ✅ Free to deploy

---

## 💡 Key ML Concepts Used

| Concept | Why it was used |
|---|---|
| **SMOTE** | Dataset has ~26% churn — SMOTE creates synthetic examples to balance it |
| **StandardScaler** | Brings all numbers to same scale so no column dominates |
| **LabelEncoder** | Converts text columns (Yes/No etc.) to numbers for the model |
| **ROC-AUC** | Main metric used to find the best model |
| **Feature Importance** | Shows which columns matter most in predicting churn |

---

## 📈 Typical Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~79% | ~0.84 |
| Decision Tree | ~78% | ~0.80 |
| Random Forest | ~81% | ~0.87 |
| Gradient Boosting | ~81% | ~0.87 |
| XGBoost | ~82% | ~0.88 |
| LightGBM | ~82% | ~0.88 |

> Note: Exact values may vary slightly each run.

---

## 👤 About

**Experience:** 1.8 years in Data Analytics
**Skills demonstrated:** Python, Flask, Machine Learning, Data Visualization, Web Deployment

---

## 📌 Notes

- This project is built for learning and portfolio purposes
- The dataset is publicly available from IBM / Kaggle
- XGBoost and LightGBM are optional — app works without them if not installed
