"""Clases y funciones para preprocesamiento del dataset de churn."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class ChurnPreprocessor:
    """Encapsula todo el pipeline de preprocesamiento del dataset Telco Churn."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.feature_columns: list = []
        self.target_column = "Churn"

    def load(self, path: str) -> pd.DataFrame:
        """Carga el dataset desde un archivo CSV."""
        df = pd.read_csv(path)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el dataframe: elimina columnas irrelevantes y corrige tipos."""
        df = df.copy()
        df.drop(columns=["customerID"], errors="ignore", inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)
        return df

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica variables categóricas con LabelEncoder."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def split(self, df: pd.DataFrame):
        """Divide en conjuntos de entrenamiento y prueba."""
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        self.feature_columns = X.columns.tolist()
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def scale(self, X_train, X_test):
        """Escala las features numéricas."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def fit_transform(self, path: str):
        """Pipeline completo: carga, limpia, codifica, divide y escala."""
        df = self.load(path)
        df = self.clean(df)
        df = self.encode(df)
        X_train, X_test, y_train, y_test = self.split(df)
        X_train_scaled, X_test_scaled = self.scale(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
