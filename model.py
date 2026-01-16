import pandas as pd 
import pickle
import os
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

model:          Optional[xgb.XGBClassifier] = None
scaler:         Optional[RobustScaler] = None
model_results:  Dict[str, float] = {}
medians:        Dict[str, float] = {}

def train_xgb(data_df) -> xgb.XGBClassifier:
    """Обучение модели"""
    global model, scaler, model_results, medians
    
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    X = data_df[features]
    y = data_df["Outcome"]
    
    for feature in features:
        medians[feature] = data_df[feature].median()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.16, random_state=12345, stratify=y
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBClassifier()
    model.fit(X_train_scaled, y_train, verbose=False)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    model_results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    save_model()
    return X_test, y_test

def save_model():
    """Сохранение модели"""
    os.makedirs("models", exist_ok=True)
    
    with open("models/diabetes_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "results": model_results,
            "medians": medians
        }, f)

def load_model() -> bool:
    """Загрузка модели"""
    global model, scaler, model_results, medians
    
    with open("models/diabetes_model.pkl", "rb") as f:
        data = pickle.load(f)
    
    model = data["model"]
    scaler = data["scaler"]
    model_results = data["results"]
    medians = data["medians"]
    
    return True

def predict(features_dict) -> Dict[str, object]:
    """Предсказание по признакам"""
    features_order = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    features_list = []
    for feature_name in features_order:
        value = features_dict.get(feature_name)
        if value == 0 and feature_name in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            if feature_name in medians:
                value = medians[feature_name]
        features_list.append(value)
    
    features_array = pd.DataFrame([features_list], columns=features_order)
    features_scaled = scaler.transform(features_array)
    
    proba = model.predict_proba(features_scaled)[0]
    prob_class_1 = proba[1]
    
    label = ""
    if prob_class_1 > 0.5:
        label = "Диабет"
    else:
        label = "Нет диабета"
    
    return {
        "label": label,
        "prob_formatted": f"{prob_class_1*100:.1f}%",
        "prob": prob_class_1
    }