"""
ml_model.py
------------
Funciones auxiliares para entrenar y usar un modelo de Regresión Logística
que prediga la probabilidad de congestión (estado ≥ 2) a +15 min.

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


PASOS_ADELANTE = 15  # minutos


def preparar_features(df: pd.DataFrame, pasos: int = PASOS_ADELANTE):
    """
    Devuelve X, y usando hora, día de la semana y estado actual como predictores.
    La variable objetivo es 1 si el estado dentro de `pasos` minutos ≥ 2.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    df["hora"]   = df["timestamp"].dt.hour
    df["diasem"] = df["timestamp"].dt.dayofweek

    # Objetivo: congestión futura
    df["objetivo"] = (df["estado"].shift(-pasos) >= 2).astype(int)
    df = df.dropna()

    X = df[["estado", "hora", "diasem"]]
    y = df["objetivo"]
    return X, y


def entrenar_logreg(df_hist: pd.DataFrame):
    """
    Entrena el modelo y devuelve:
      • pipe  – pipeline StandardScaler + LogisticRegression
      • acc   – accuracy en test
      • roc   – ROC-AUC en test
    """
    if df_hist.empty or len(df_hist) < 100:
        raise ValueError("Histórico insuficiente para entrenar (≥ 100 filas).")

    X, y = preparar_features(df_hist)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_prob)

    return pipe, acc, roc
