# markov.py

import numpy as np
import pandas as pd

def predict_congestion(current_df: pd.DataFrame, pasos: int = 15) -> float:
    """
    Predice la probabilidad de congestión (estado == 2) a `pasos` minutos
    usando una cadena de Markov basada en los últimos registros de tráfico.
    """

    # Aseguramos que exista la columna timestamp y estado
    df = current_df.copy()
    if 'fecha' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'fecha': 'timestamp'})
    if 'timestamp' not in df.columns or 'estado' not in df.columns:
        raise KeyError("El DataFrame debe contener columnas 'timestamp' y 'estado'.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Definimos los estados válidos
    estados = [0, 1, 2, 3]
    # Matriz de transición inicializada a ceros
    P = np.zeros((4, 4), dtype=float)

    # Creamos columna next con el estado siguiente y descartamos NaN
    df['next'] = df['estado'].shift(-1)
    trans = df.dropna(subset=['next'])

    # Contamos sólo transiciones válidas
    total_por_estado = {i: 0 for i in estados}
    cuentas = { (i, j): 0 for i in estados for j in estados }

    for _, row in trans.iterrows():
        i = int(row['estado'])
        j = int(row['next'])
        if i in estados and j in estados:
            total_por_estado[i] += 1
            cuentas[(i, j)] += 1

    # Llenamos la matriz P con frecuencias relativas
    for i in estados:
        total = total_por_estado[i]
        if total > 0:
            for j in estados:
                P[i, j] = cuentas[(i, j)] / total

    # Vector one-hot para el estado actual
    ultimo = int(df['estado'].iloc[-1])
    v0 = np.zeros(4)
    v0[ultimo] = 1.0

    # Elevamos P a la n-ésima potencia y obtenemos π_n
    Pn = np.linalg.matrix_power(P, pasos)
    pi_n = v0.dot(Pn)

    # Devolvemos probabilidad de congestión (estado == 2)
    return float(pi_n[2])
