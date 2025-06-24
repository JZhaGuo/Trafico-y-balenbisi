import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import streamlit as st
from datetime import datetime, timezone

from markov import predict_congestion          # función Markov existente
from ml_model import entrenar_logreg           # modelo ML

st.set_page_config(page_title="Tráfico y Valenbisi", layout="wide")


# ────────────────────────────────────────────────────────────
# 1 · Carga de datos con caché nueva
# ────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)
def load_valenbisi():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
    params = {"dataset": "valenbisi-estaciones", "rows": 500}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        recs = js.get("records", [])
        rows = [r.get("fields", {}) for r in recs]
        df = pd.DataFrame(rows)
        # Renombrar columna de bicis si hace falta
        if "bikes_available" in df.columns:
            df = df.rename(columns={"bikes_available": "Bicis_disponibles"})
        return df
    except Exception as e:
        print("[Valenbisi] error:", e)
        return pd.DataFrame()


@st.cache_data(ttl=180)
def load_traffic():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
    params = {
        "dataset": "estat-transit-temps-real-estado-trafico-tiempo-real",
        "rows": 1000
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        recs = js.get("records", [])
        rows = [r["fields"] for r in recs if "fields" in r]
        df = pd.DataFrame(rows)
        # Renombrar timestamp
        if "fecha" in df.columns:
            df = df.rename(columns={"fecha": "timestamp"})
        return df
    except Exception as e:
        print("[Tráfico] error:", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 2 · Histórico (para modelo ML)
# ─────────────────────────────────────────────────────────────────
def append_to_history(df_traf: pd.DataFrame):
    if df_traf.empty or "timestamp" not in df_traf.columns:
        return
    hist = df_traf[["timestamp", "estado"]].copy()
    hist.to_csv("hist_traffic.csv", mode="a", header=False, index=False)


# ─────────────────────────────────────────────────────────────────
# 3 · Modelo de regresión logística (se cachea como recurso)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_logreg_model():
    try:
        df_hist = pd.read_csv("hist_traffic.csv", names=["timestamp", "estado"])
    except FileNotFoundError:
        df_hist = pd.DataFrame(columns=["timestamp", "estado"])
    return entrenar_logreg(df_hist)


# ─────────────────────────────────────────────────────────────────
# 4 · Barra lateral: filtros, leyenda y controles
# ─────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici = st.sidebar.checkbox("Mostrar Valenbisi", True)
if st.sidebar.button("🔄  Actualizar datos"):
    load_traffic.clear()
    load_valenbisi.clear()
    st.rerun()

st.sidebar.subheader("Estados de tráfico")
st.sidebar.markdown(
    """
    | Código | Significado |
    |--------|-------------|
    | **0**  | 🟢 Fluido |
    | **1**  | 🟠 Moderado |
    | **2**  | 🔴 Denso |
    | **3**  | ⚫ Cortado |
    """,
    unsafe_allow_html=True,
)

metodo = st.sidebar.radio(
    "Método de predicción",
    ("Cadena de Markov", "Regresión logística"),
)
comparar = st.sidebar.checkbox("Comparar ambos métodos", False)


# ─────────────────────────────────────────────────────────────────
# 5 · Carga de datos
# ─────────────────────────────────────────────────────────────────
df_traf = load_traffic()
df_bici = load_valenbisi()
if df_traf.empty:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if df_bici.empty and show_bici:
    st.warning("⚠️  Sin datos de Valenbisi en este momento.")
append_to_history(df_traf)


# ─────────────────────────────────────────────────────────────────
# 6 · Predicción
# ─────────────────────────────────────────────────────────────────
prob_markov = prob_ml = None
acc = roc = None

if not df_traf.empty:
    estado_actual = int(df_traf["estado"].mode()[0])

    if metodo == "Cadena de Markov":
        with st.spinner("Calculando con cadena de Markov…"):
            prob_markov = predict_congestion(df_traf)

    else:
        with st.spinner("Calculando con regresión logística…"):
            try:
                modelo, acc, roc = get_logreg_model()
                ahora = datetime.now(timezone.utc)
                x_actual = pd.DataFrame({
                    "estado": [estado_actual],
                    "hora": [ahora.hour],
                    "diasem": [ahora.weekday()],
                })
                prob_ml = float(modelo.predict_proba(x_actual)[0, 1])
            except ValueError as e:
                st.warning(f"⚠️ {e}")
else:
    st.info("⏳ Datos de tráfico no disponibles; sin predicción en este momento.")


# ─────────────────────────────────────────────────────────────────
# 7 · Mostrar resultados
# ─────────────────────────────────────────────────────────────────
if prob_markov is not None:
    st.progress(prob_markov)
    st.write(f"🔮 **Probabilidad (Markov): {prob_markov*100:.1f}%**")
if prob_ml is not None:
    st.progress(prob_ml)
    st.write(f"🔮 **Probabilidad (LogReg): {prob_ml*100:.1f}%**")
    st.write(f"**Accuracy:** {acc:.2f} · **ROC-AUC:** {roc:.2f}")
if comparar and prob_markov is not None and prob_ml is not None:
    diff = abs(prob_markov - prob_ml) * 100
    st.write(f"📊 *Diferencia Markov vs LogReg:* **{diff:.1f} puntos**")


# ─────────────────────────────────────────────────────────────────
# 8 · Mapa
# ─────────────────────────────────────────────────────────────────
layers = []
if show_traf and not df_traf.empty:
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=df_traf,
        get_position="[longitud, latitud]",
        get_fill_color="[255, 0, 0, 80]", get_radius=40, pickable=True,
    ))
if show_bici and not df_bici.empty:
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=df_bici,
        get_position="[lon, lat]",
        get_fill_color="[0, 140, 255, 80]", get_radius=40, pickable=True,
    ))
if layers:
    midpoint = [39.47, -0.376]
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0], longitude=midpoint[1], zoom=12
        ), layers=layers,
        tooltip={"text": "{denominacion}"},
    ))
