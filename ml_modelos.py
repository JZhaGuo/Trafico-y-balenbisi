import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

def preparar_features(df: pd.DataFrame, pasos: int = 15):
    """Devuelve X, y usando hora, día y estado actual como predictores."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hora"]   = df["timestamp"].dt.hour
    df["diasem"] = df["timestamp"].dt.dayofweek
    # variable objetivo: 1 si estado futuro (t+15) >= 2
    df["objetivo"] = (df["estado"].shift(-pasos) >= 2).astype(int)
    df = df.dropna()
    X = df[["estado", "hora", "diasem"]]
    y = df["objetivo"]
    return X, y

def entrenar_logreg(df_hist: pd.DataFrame):
    """Entrena y devuelve el modelo + métricas (acc, roc_auc)."""
    X, y = preparar_features(df_hist)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return pipe, acc, roc_auc
