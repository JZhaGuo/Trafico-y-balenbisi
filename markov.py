import numpy as np
import pandas as pd

def build_transition_matrix(hist_df: pd.DataFrame, state_col: str = "estado") -> np.ndarray:
    """
    Construye la matriz de transición 4×4 para estados {0,1,2,3},
    agrupando subterráneos (5-8) en sus equivalentes 0–3.
    """
    df = hist_df.sort_values("timestamp").copy()
    df["s"] = df[state_col].clip(0,3)

    trans = np.zeros((4,4), dtype=int)
    prev, curr = df["s"].values[:-1], df["s"].values[1:]
    for i,j in zip(prev, curr):
        trans[int(i), int(j)] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        P = trans / trans.sum(axis=1, keepdims=True)
        P = np.nan_to_num(P)
    return P

def predict_congestion(current_df: pd.DataFrame,
                       hist_csv: str = "trafico_historico.csv",
                       steps: int = 5) -> float:
    """
    Predice probabilidad de congestión (estado 2) a `steps` pasos (~15min).
    Si no encuentra el histórico, lo crea con la muestra actual y devuelve 0.0.
    """
    # 1) Leer o inicializar histórico
    try:
        hist = pd.read_csv(hist_csv, parse_dates=["timestamp"])
    except FileNotFoundError:
        hist = current_df[["timestamp","estado"]].copy()
        hist.to_csv(hist_csv, index=False)
        return 0.0

    # 2) Calcula matriz y predicción
    P = build_transition_matrix(hist)
    pi0 = current_df["estado"].clip(0,3).value_counts(normalize=True)
    pi0 = pi0.reindex(range(4), fill_value=0).values
    pi_tn = pi0 @ np.linalg.matrix_power(P, steps)

    # 3) Actualiza histórico
    new_entries = current_df[["timestamp","estado"]].copy()
    hist = pd.concat([hist, new_entries], ignore_index=True)
    hist.to_csv(hist_csv, index=False)

    return float(pi_tn[2])
