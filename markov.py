# markov.py

import numpy as np
import pandas as pd

def predict_congestion(current_df: pd.DataFrame, pasos: int = 15) -> float:
    """
    Predice la probabilidad de congestión (estado == 2) a `pasos` minutos
    usando una cadena de Markov entrenada sobre current_df.
    current_df debe tener:
      - columna de tiempo: 'timestamp' o 'fecha'
      - columna de estado: 'estado' (valores 0,1,2,3)
    """

    # 1) Asegurarnos de tener columnas 'timestamp' y 'estado'
    df = current_df.copy()
    if 'fecha' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'fecha': 'timestamp'})

    if 'timestamp' not in df.columns or 'estado' not in df.columns:
        raise KeyError("DataFrame debe contener columnas 'timestamp' y 'estado'")

    # 2) Limpiar y ordenar
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # 3) Construir matriz de transición P 4×4
    #    Estados posibles: 0,1,2,3
    estados = [0,1,2,3]
    P = np.zeros((4,4), dtype=float)
    # Shift para observar transiciones
    df['next'] = df['estado'].shift(-1)
    trans = df.dropna(subset=['next'])  # quitamos el último vacío
    for i in estados:
        row = trans[trans['estado']==i]['next'].value_counts()
        total = row.sum()
        if total>0:
            for j,count in row.items():
                if j in estados:
                    P[i,j] = count/total

    # 4) Distribución inicial π₀ desde la última observación
    pi0 = df['estado'].iloc[-1]
    # Creamos vector one-hot
    v0 = np.zeros(4); v0[int(pi0)] = 1.0

    # 5) Elevamos P a la n-ésima potencia
    Pn = np.linalg.matrix_power(P, pasos)

    # 6) π_n = v0 · P^n
    pi_n = v0.dot(Pn)

    # 7) Devolvemos la probabilidad de estado == 2 (congestión)
    return float(pi_n[2])
