"""
ml_model.py
------------
Funciones auxiliares para entrenar y usar un modelo de Regresión Logística
que prediga la probabilidad de congestión (estado ≥ 2) a +15 minutos.

Se asume que existe un histórico `hist_traffic.csv`
con las columnas: timestamp (ISO-8601) y estado (0-3).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# Número de minutos hacia adelante para predecir congestión
PASOS_ADELANTE = 15


def preparar_features(df: pd.DataFrame, pasos: int = PASOS_ADELANTE):
    """
    A partir de un DataFrame con columnas:
      - timestamp: fechas en ISO-8601
      - estado: entero 0–3
    genera:
      - X: DataFrame con features ['estado', 'hora', 'diasem']
      - y: Serie binaria (1 si el estado a `pasos` minutos ≥ 2, else 0)

    Parámetros:
      df    – DataFrame de histórico
      pasos – desplazamiento en minutos para construir la variable objetivo

    Pasos:
      1. Convertir timestamp a datetime y ordenar cronológicamente.
      2. Extraer la hora del día y el día de la semana.
      3. Crear 'objetivo' = 1 si estado.shift(-pasos) ≥ 2.
      4. Eliminar filas con NaN (al final debido al shift).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # Características temporales
    df["hora"] = df["timestamp"].dt.hour
    df["diasem"] = df["timestamp"].dt.dayofweek

    # Variable objetivo: congestión futura
    df["objetivo"] = (df["estado"].shift(-pasos) >= 2).astype(int)
    df = df.dropna(subset=["objetivo", "estado", "hora", "diasem"])

    X = df[["estado", "hora", "diasem"]]
    y = df["objetivo"].astype(int)
    return X, y


def entrenar_logreg(df_hist: pd.DataFrame):
    """
    Entrena un modelo de Regresión Logística y devuelve:
      • pipe  – Pipeline(StandardScaler, LogisticRegression)
      • acc   – accuracy en test
      • roc   – ROC-AUC en test

    Lanza ValueError si no hay suficientes datos (mínimo 100 filas).
    """
    if df_hist.empty or len(df_hist) < 100:
        raise ValueError("Histórico insuficiente para entrenar (≥ 100 filas).")

    # Preparamos features y target
    X, y = preparar_features(df_hist)

    # Split estratificado
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Pipeline: escalado + regresión logística
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    pipe.fit(X_tr, y_tr)

    # Predicción y métricas
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_prob)

    return pipe, acc, roc
