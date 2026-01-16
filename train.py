"""
Обучение модели XGBoost
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from data_proc import load_data
from model import train_xgb

def create_plots(model, scaler, X_test, y_test, feature_names):
    """Создание графиков метрик"""
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    with open("models/diabetes_model.pkl", "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    
    metrics_names = ["Accuracy", "F1-Score"]
    metrics_values = [results["accuracy"], results["f1_score"]]
    
    bars = axes[0, 0].bar(metrics_names, metrics_values)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Метрики модели")
    for bar, v in zip(bars, metrics_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{v:.3f}")
    
    cm = confusion_matrix(y_test, y_pred).T
    im = axes[0, 1].imshow(cm, interpolation="nearest")
    axes[0, 1].set_title("Матрица ошибок")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(["Нет", "Да"])
    axes[0, 1].set_yticklabels(["Нет", "Да"])
    axes[0, 1].set_xlabel("Предсказание")
    axes[0, 1].set_ylabel("Истина")
    
    plt.colorbar(im, ax=axes[0, 1])
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    axes[1, 0].plot(fpr, tpr, label=f"AUC = {results["roc_auc"]:.3f}")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--")
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].set_title("ROC-кривая")
    axes[1, 0].grid(True, alpha=0.5)
    
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    
    axes[1, 1].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    axes[1, 1].set_yticks(range(len(sorted_idx)))
    axes[1, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
    axes[1, 1].set_xlabel("Важность")
    axes[1, 1].set_title("Важность признаков")
    
    plt.tight_layout()
    return fig

def main():
    data_df = load_data()
    X_test, y_test = train_xgb(data_df)
    
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    with open("models/diabetes_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    scaler = model_data["scaler"]
    
    plot = create_plots(model, scaler, X_test, y_test, features)
    plot.savefig("models/metrics.png")
    
    plt.close("all")

if __name__ == "__main__":
    main()