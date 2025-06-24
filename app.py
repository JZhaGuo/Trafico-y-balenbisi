import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import streamlit as st
from datetime import datetime, timezone

from markov import predict_congestion
from ml_model import entrenar_logreg

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
        rows = []
        for rec in js.get("records", []):
            f = rec.get("fields", {}).copy()
            # bikes_available → Bicis_disponibles
            if "bikes_available" in f:
                f["Bicis_disponibles"] = f.pop("bikes_available")
            # geo_point_2d → lat, lon
            if isinstance(f.get("geo_point_2d"), list):
                f["lat"], f["lon"] = f["geo_point_2d"]
            rows.append(f)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error cargando Valenbisi: {e}")
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
        rows = []
        for rec in js.get("records", []):
            f = rec.get("fields", {}).copy()
            # record_timestamp → timestamp
            ts = rec.get("record_timestamp") or rec.get("recordTimestamp")
            if ts:
                f["timestamp"] = ts
            # geo_point_2d → latitud, longitud
            if isinstance(f.get("geo_point_2d"), list):
                f["latitud"], f["longitud"] = f["geo_point_2d"]
            # fallback keys
            if "latitude" in f and "latitud" not in f:
                f["latitud"] = f["latitude"]
            if "longitude" in f and "longitud" not in f:
                f["longitud"] = f["longitude"]
            rows.append(f)
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"Error cargando tráfico: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 2 · Guardar histórico para ML
# ─────────────────────────────────────────────────────────────────
def append_to_history(df_traf: pd.DataFrame):
    if df_traf.empty or "timestamp" not in df_traf.columns or "estado" not in df_traf.columns:
        return
    hist = df_traf[["timestamp", "estado"]].copy()
    hist.to_csv("hist_traffic.csv", mode="a", header=False, index=False)


# ─────────────────────────────────────────────────────────────────
# 3 · Modelo de Regresión Logística (cache resource)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_logreg_model():
    try:
        df_hist = pd.read_csv("hist_traffic.csv", names=["timestamp", "estado"])
    except FileNotFoundError:
        df_hist = pd.DataFrame(columns=["timestamp", "estado"])
    return entrenar_logreg(df_hist)


# ─────────────────────────────────────────────────────────────────
# 4 · Barra lateral: filtros y leyenda
# ─────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici = st.sidebar.checkbox("Mostrar Valenbisi", True)
if st.sidebar.button("🔄 Actualizar datos"):
    load_traffic.clear()
    load_valenbisi.clear()
    st.rerun()

st.sidebar.subheader("Estados de tráfico")
st.sidebar.markdown(
    """
    | Código | Significado |
    |--------|-------------|
    | **0**  | 🟢 Fluido   |
    | **1**  | 🟠 Denso    |
    | **2**  | 🔴 Congest. |
    | **3**  | ⚫ Cortado  |
    """,
    unsafe_allow_html=True
)

metodo = st.sidebar.radio("Método de predicción", ["Cadena de Markov", "Regresión logística"])
comparar = st.sidebar.checkbox("Comparar ambos métodos")


# ─────────────────────────────────────────────────────────────────
# 5 · Carga de datos
# ─────────────────────────────────────────────────────────────────
df_traf = load_traffic()
df_bici = load_valenbisi()

if df_traf.empty or "estado" not in df_traf.columns:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if df_bici.empty:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")

append_to_history(df_traf)


# ─────────────────────────────────────────────────────────────────
# 6 · Predicción
# ─────────────────────────────────────────────────────────────────
prob_markov = prob_ml = None
acc = roc = None

if (not df_traf.empty) and ("estado" in df_traf.columns):
    estado_actual = int(df_traf["estado"].mode()[0])
    if metodo == "Cadena de Markov":
        with st.spinner("Calculando (Markov)…"):
            prob_markov = predict_congestion(df_traf)
    else:
        with st.spinner("Calculando (LogReg)…"):
            modelo, acc, roc = get_logreg_model()
            ahora = datetime.now(timezone.utc)
            x_act = pd.DataFrame({
                "estado": [estado_actual],
                "hora":   [ahora.hour],
                "diasem": [ahora.weekday()]
            })
            prob_ml = float(modelo.predict_proba(x_act)[0, 1])
else:
    st.info("⏳ Sin datos de tráfico; no se calcula predicción.")


# ─────────────────────────────────────────────────────────────────
# 7 · Mostrar resultados
# ─────────────────────────────────────────────────────────────────
if prob_markov is not None:
    st.progress(prob_markov)
    st.write(f"🔮 Probabilidad (Markov): {prob_markov*100:.1f}%")
if prob_ml is not None:
    st.progress(prob_ml)
    st.write(f"🔮 Probabilidad (LogReg): {prob_ml*100:.1f}%")
    st.write(f"✅ Accuracy: {acc:.2f} · ROC-AUC: {roc:.2f}")
if comparar and (prob_markov is not None) and (prob_ml is not None):
    diff = abs(prob_markov - prob_ml)*100
    st.write(f"⚖️ Diferencia: {diff:.1f} puntos")


# ─────────────────────────────────────────────────────────────────
# 8 · Mapa
# ─────────────────────────────────────────────────────────────────
layers = []

if show_traf and (not df_traf.empty) and {"latitud","longitud"}.issubset(df_traf.columns):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_traf,
        get_position="[longitud, latitud]",
        get_fill_color="[255,0,0,80]",
        get_radius=40,
        pickable=True
    ))

if show_bici and (not df_bici.empty) and {"lat","lon"}.issubset(df_bici.columns):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_bici,
        get_position="[lon, lat]",
        get_fill_color="[0,140,255,80]",
        get_radius=30,
        pickable=True
    ))

if layers:
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=39.47, longitude=-0.376, zoom=12),
        layers=layers,
        tooltip={"text": "{denominacion}"}
    ))
