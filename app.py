from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

app = Flask(__name__)

store = {
    'trained': False, 'models': {}, 'scaler': None,
    'feature_names': None, 'label_encoders': {},
    'all_metrics': [], 'best_model_name': None,
    'eda': {}, 'dataset_info': {},
}

def build_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                                random_state=42, use_label_encoder=False,
                                                eval_metric='logloss', verbosity=0)
    if LGB_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                                  max_depth=4, random_state=42, verbose=-1)
    return models

def prepare_data(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.dropna(subset=['Churn'], inplace=True)

    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['NumAddOns'] = (
        (df.get('OnlineSecurity',   'No') == 'Yes').astype(int) +
        (df.get('OnlineBackup',     'No') == 'Yes').astype(int) +
        (df.get('DeviceProtection', 'No') == 'Yes').astype(int) +
        (df.get('TechSupport',      'No') == 'Yes').astype(int) +
        (df.get('StreamingTV',      'No') == 'Yes').astype(int) +
        (df.get('StreamingMovies',  'No') == 'Yes').astype(int)
    )

    le_store = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_store[col] = le

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y, le_store

def train_all_models(df):
    X, y, le_store = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc  = scaler.transform(X_test)

    models = build_models()
    all_metrics = []
    fitted_models = {}
    best_auc, best_name = -1, None

    for name, model in models.items():
        model.fit(X_train_sc, y_train_sm)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
        auc  = round(roc_auc_score(y_test, y_proba), 4)
        prec = round(precision_score(y_test, y_pred) * 100, 2)
        rec  = round(recall_score(y_test, y_pred) * 100, 2)
        f1   = round(f1_score(y_test, y_pred) * 100, 2)
        cm   = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        fi = []
        if hasattr(model, 'feature_importances_'):
            fi_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
            fi = fi_df.sort_values('importance', ascending=False).head(10).to_dict(orient='records')
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_[0])
            fi_df = pd.DataFrame({'feature': X.columns, 'importance': coef / coef.sum()})
            fi = fi_df.sort_values('importance', ascending=False).head(10).to_dict(orient='records')

        fitted_models[name] = model
        all_metrics.append({
            'name': name, 'accuracy': acc, 'roc_auc': auc,
            'precision': prec, 'recall': rec, 'f1': f1,
            'confusion_matrix': cm,
            'roc_fpr': [round(float(v), 4) for v in fpr],
            'roc_tpr': [round(float(v), 4) for v in tpr],
            'feature_importance': fi,
        })

        if auc > best_auc:
            best_auc, best_name = auc, name

    all_metrics.sort(key=lambda x: x['roc_auc'], reverse=True)

    dataset_info = {
        'total_customers':   int(len(df)),
        'churned_customers': int(y.sum()),
        'churn_rate':        round(float(y.mean()) * 100, 2),
        'features':          int(X.shape[1]),
    }
    return fitted_models, scaler, X.columns.tolist(), le_store, all_metrics, best_name, dataset_info

def get_eda_stats(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: round((x == 'Yes').mean() * 100, 1)).to_dict()
    internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: round((x == 'Yes').mean() * 100, 1)).to_dict()
    tenure_bins    = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12mo','13-24mo','25-48mo','49-72mo'])
    tenure_churn   = df.groupby(tenure_bins)['Churn'].apply(lambda x: round((x == 'Yes').mean() * 100, 1)).to_dict()
    payment_churn  = df.groupby('PaymentMethod')['Churn'].apply(lambda x: round((x == 'Yes').mean() * 100, 1)).to_dict()

    return {
        'contract_churn': contract_churn,
        'internet_churn': internet_churn,
        'tenure_churn':   {str(k): v for k, v in tenure_churn.items()},
        'payment_churn':  payment_churn,
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    try:
        df = pd.read_csv(file)
        eda = get_eda_stats(df)
        fitted_models, scaler, features, le_store, all_metrics, best_name, dataset_info = train_all_models(df)
        store.update({
            'trained': True, 'models': fitted_models, 'scaler': scaler,
            'feature_names': features, 'label_encoders': le_store,
            'all_metrics': all_metrics, 'best_model_name': best_name,
            'eda': eda, 'dataset_info': dataset_info,
        })
        return jsonify({
            'success': True, 'all_metrics': all_metrics, 'best_model': best_name,
            'eda': eda, 'dataset_info': dataset_info,
            'xgb_available': XGB_AVAILABLE, 'lgb_available': LGB_AVAILABLE,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not store['trained']:
        return jsonify({'error': 'Please upload dataset first'}), 400
    data = request.json
    model_name = data.pop('selected_model', store['best_model_name'])
    try:
        row = pd.DataFrame([data])
        row['TotalCharges'] = pd.to_numeric(row['TotalCharges'], errors='coerce')
        row['TotalCharges'].fillna(row['MonthlyCharges'], inplace=True)
        row['AvgMonthlySpend'] = row['TotalCharges'] / (row['tenure'].astype(float) + 1)
        row['NumAddOns'] = sum([
            (row.get(c, pd.Series(['No'])) == 'Yes').astype(int)
            for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        ])

        for col, le in store['label_encoders'].items():
            if col in row.columns:
                val = str(row[col].iloc[0])
                row[col] = le.transform([val])[0] if val in le.classes_ else 0

        row    = row.reindex(columns=store['feature_names'], fill_value=0)
        row_sc = store['scaler'].transform(row)

        all_preds = []
        for name, model in store['models'].items():
            prob = float(model.predict_proba(row_sc)[0][1])
            all_preds.append({
                'model':       name,
                'probability': round(prob * 100, 1),
                'prediction':  'Churn' if prob >= 0.5 else 'No Churn',
                'risk':        'High' if prob >= 0.7 else ('Medium' if prob >= 0.4 else 'Low'),
                'is_selected': name == model_name,
            })

        all_preds.sort(key=lambda x: x['probability'], reverse=True)
        selected = next((p for p in all_preds if p['is_selected']), all_preds[0])
        return jsonify({'selected': selected, 'all_preds': all_preds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
