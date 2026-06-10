"""Clases para entrenamiento y evaluación de modelos de ML."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib


class ModelTrainer:
    """Entrena y evalúa modelos de clasificación para predicción de churn."""

    MODELS = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    def __init__(self, model_name: str = "random_forest"):
        if model_name not in self.MODELS:
            raise ValueError(f"Modelo no válido. Opciones: {list(self.MODELS.keys())}")
        self.model_name = model_name
        self.model = self.MODELS[model_name]
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """Entrena el modelo con los datos de entrenamiento."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def evaluate(self, X_test, y_test) -> dict:
        """Evalúa el modelo y retorna diccionario con métricas."""
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        return {
            "model": self.model_name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }

    def predict(self, X) -> np.ndarray:
        """Genera predicciones para nuevos datos."""
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado.")
        return self.model.predict(X)

    def save(self, path: str):
        """Guarda el modelo entrenado como archivo .pkl."""
        joblib.dump(self.model, path)
        print(f"Modelo guardado en: {path}")

    def load(self, path: str):
        """Carga un modelo previamente guardado."""
        self.model = joblib.load(path)
        self.is_fitted = True
        return self
