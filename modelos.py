import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def train_logreg(df_hist):
    """Entrena una regresión logística simple y devuelve modelo + métricas."""
    # Creamos variable objetivo: estado en +15 min
    df_hist = df_hist.sort_values("timestamp")
    df_hist["y"] = df_hist["estado"].shift(-15)      # 1 fila = 1 min aprox.
    df_hist = df_hist.dropna(subset=["y"])

    # Features muy básicas: hora y estado actual
    X = pd.DataFrame({
        "hora": pd.to_datetime(df_hist["timestamp"]).dt.hour,
        "estado_actual": df_hist["estado"].astype(int),
    })
    y = (df_hist["y"] >= 2).astype(int)  # congestión = estados 2–3

    # Train/test 80-20
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Métricas
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return model, metrics
