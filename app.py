import pandas as pd
import pydeck as pdk
import streamlit as st
from datetime import datetime, timezone
from ml_model import entrenar_logreg

st.set_page_config(page_title="Tráfico y Valenbisi", layout="wide")


@st.cache_data(ttl=180)
def load_traffic():
    csv_url = (
        "https://valencia.opendatasoft.com/explore/dataset/"
        "estat-transit-temps-real-estado-trafico-tiempo-real/"
        "download/?format=csv&rows=1000"
    )
    try:
        df = pd.read_csv(csv_url)
        # Asegúrate de tener estas columnas
        df.rename(columns={
            "geo_point_2d_lat": "latitud", 
            "geo_point_2d_lon": "longitud",
            "denominacion": "denominacion",
            "estado": "estado"
        }, inplace=True, errors="ignore")
        df["estado"] = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        st.error(f"Error cargando tráfico CSV: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=180)
def load_valenbisi():
    csv_url = (
        "https://valencia.opendatasoft.com/explore/dataset/"
        "valenbisi-disponibilitat-valenbisi-dsiponibilidad/"
        "download/?format=csv&rows=500"
    )
    try:
        df = pd.read_csv(csv_url)
        # Renombra columnas si el CSV usa otros nombres
        df.rename(columns={
            "geo_point_2d_lat": "lat",
            "geo_point_2d_lon": "lon",
            "slots_disponibles": "Bicis_disponibles",
            "address": "direccion"
        }, inplace=True, errors="ignore")
        return df
    except Exception as e:
        st.error(f"Error cargando Valenbisi CSV: {e}")
        return pd.DataFrame()


# Resto de tu app.py sin cambios:
# ────────────────────────────────
# 2 · Sidebar
st.sidebar.title("Filtros")
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici  = st.sidebar.checkbox("Mostrar Valenbisi", True)
search_street = st.sidebar.text_input("Buscar calle (opcional)", "")
if st.sidebar.button("🔄 Actualizar datos"):
    st.rerun()

# 3 · Carga de datos
df_traf = load_traffic()
df_bici = load_valenbisi()

if show_traf and df_traf.empty:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if show_bici and df_bici.empty:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")

# 4 · Filtrar por calle
if show_traf and search_street and "denominacion" in df_traf.columns:
    df_traf = df_traf[
        df_traf["denominacion"]
        .str.contains(search_street, case=False, na=False)
    ]

# 5 · Colorear y mostrar mapa
color_map = {
    0: [0,255,0,80],
    1: [255,165,0,80],
    2: [255,0,0,80],
    3: [0,0,0,80],
    4: [128,128,128,80],  # Sin datos
}
if show_traf and not df_traf.empty and {"longitud","latitud","estado"}.issubset(df_traf.columns):
    df_traf["fill_color"] = df_traf["estado"].apply(
        lambda s: color_map.get(int(s), [200,200,200,80])
    )
    layer = pdk.Layer(
        "ScatterplotLayer", data=df_traf,
        get_position="[longitud, latitud]",
        get_fill_color="fill_color",
        get_radius=40,
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=39.47, longitude=-0.376, zoom=12),
        layers=[layer],
        tooltip={"text": "{denominacion}"}
    ))
else:
    st.info("No hay capas para mostrar en el mapa.")

# 6 · Predicción ML
@st.cache_resource
def get_logreg_model():
    try:
        df_hist = pd.read_csv(
            "trafico_historico.csv",
            names=["timestamp","estado"],
            header=0
        )
    except FileNotFoundError:
        st.warning("⚠️ No encontré trafico_historico.csv.")
        return None, None, None
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], utc=True)
    if len(df_hist) < 100:
        st.warning("⚠️ Histórico insuficiente para entrenar ML.")
        return None, None, None
    try:
        return entrenar_logreg(df_hist)
    except Exception as e:
        st.warning(f"⚠️ Error entrenando ML: {e}")
        return None, None, None

modelo, acc, roc = get_logreg_model()

if show_traf and modelo and not df_traf.empty:
    estado_act = int(df_traf["estado"].mode()[0])
    ahora = datetime.now(timezone.utc)
    X_act = pd.DataFrame({
        "estado": [estado_act],
        "hora":   [ahora.hour],
        "diasem": [ahora.weekday()]
    })
    prob = modelo.predict_proba(X_act)[0,1]
    st.markdown("---")
    st.subheader("🔮 Predicción ML (15 min adelante)")
    st.write(f"- **Accuracy:** {acc:.2f}")
    st.write(f"- **ROC-AUC:**  {roc:.2f}")
    st.write(f"- **P(congestión ≥ 2):** {prob*100:.1f}%")
elif show_traf:
    st.warning("⚠️ Predicción ML no disponible.")

# 8 · Lista de calles
if show_traf and not df_traf.empty and "denominacion" in df_traf.columns:
    st.subheader("📋 Calles mostradas")
    for calle in sorted(df_traf["denominacion"].dropna().unique()):
        st.markdown(f"- {calle}")
