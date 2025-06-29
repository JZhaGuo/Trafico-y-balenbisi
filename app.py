import pandas as pd
import pydeck as pdk
import streamlit as st
from datetime import datetime, timezone
from ml_model import entrenar_logreg

st.set_page_config(page_title="Tráfico y Valenbisi", layout="wide")


# ────────────────────────────────────────────────────────────
# Helpers para normalizar columnas
# ────────────────────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame):
    # convertir todo a minúsculas y sin espacios
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    return df

def detect_column(df: pd.DataFrame, keywords):
    """Devuelve la primera columna cuyo nombre incluya alguna de las keywords."""
    for kw in keywords:
        for c in df.columns:
            if kw in c:
                return c
    return None


# ────────────────────────────────────────────────────────────
# 1 · Carga de datos con caché: lee CSV remoto y normaliza
# ────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)
def load_traffic():
    csv_url = (
        "https://valencia.opendatasoft.com/api/v2/catalog/datasets/"
        "estat-transit-temps-real-estado-trafico-tiempo-real/exports/csv?"
        "format=csv&rows=1000"
    )
    try:
        df = pd.read_csv(csv_url)
        df = normalize_columns(df)
        
        # Renombrar columnas clave
        state_col = detect_column(df, ["estado"])
        denom_col = detect_column(df, ["denominacion", "denominació", "name"])
        lat_col   = detect_column(df, ["geo_point_2d_lat", "latitud", "latitude", "lat"])
        lon_col   = detect_column(df, ["geo_point_2d_lon", "longitud", "longitude", "lon"])
        
        if not all([state_col, denom_col, lat_col, lon_col]):
            raise KeyError(f"Columnas faltantes: estado={state_col}, denominacion={denom_col}, "
                           f"lat={lat_col}, lon={lon_col}")
        
        df = df.rename(columns={
            state_col: "estado",
            denom_col: "denominacion",
            lat_col:   "latitud",
            lon_col:   "longitud"
        })
        
        df["estado"] = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        st.error(f"Error cargando tráfico CSV: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=180)
def load_valenbisi():
    csv_url = (
        "https://valencia.opendatasoft.com/api/v2/catalog/datasets/"
        "valenbisi-disponibilitat-valenbisi-dsiponibilidad/exports/csv?"
        "format=csv&rows=500"
    )
    try:
        df = pd.read_csv(csv_url)
        df = normalize_columns(df)
        
        # Renombrar columnas clave
        slots_col  = detect_column(df, ["slots_disponibles", "bicis_disponibles"])
        addr_col   = detect_column(df, ["address", "direccion"])
        lat_col    = detect_column(df, ["geo_point_2d_lat", "lat"])
        lon_col    = detect_column(df, ["geo_point_2d_lon", "lon"])
        
        rename_map = {}
        if slots_col: rename_map[slots_col] = "bicis_disponibles"
        if addr_col:  rename_map[addr_col]  = "direccion"
        if lat_col:   rename_map[lat_col]   = "lat"
        if lon_col:   rename_map[lon_col]   = "lon"
        
        if not rename_map:
            raise KeyError(f"No encontré columnas válidas en Valenbisi: {df.columns.tolist()}")
        
        df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        st.error(f"Error cargando Valenbisi CSV: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 2 · Sidebar: filtros y recarga
# ─────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici  = st.sidebar.checkbox("Mostrar Valenbisi", True)
search    = st.sidebar.text_input("Buscar calle (opcional)", "")
if st.sidebar.button("🔄 Actualizar datos"):
    st.experimental_rerun()

st.sidebar.markdown(
    """
    **Estados de tráfico**  
    0: 🟢 Fluido  
    1: 🟠 Moderado  
    2: 🔴 Congestionado  
    3: ⚫ Cortado  
    4–9: Pasos y sin datos  
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# 3 · Carga de datos
# ─────────────────────────────────────────────────────────────────
df_traf = load_traffic()
df_bici  = load_valenbisi()

if show_traf and df_traf.empty:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if show_bici and df_bici.empty:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")


# ─────────────────────────────────────────────────────────────────
# 4 · Filtrar por calle si texto
# ─────────────────────────────────────────────────────────────────
if show_traf and search and "denominacion" in df_traf.columns:
    df_traf = df_traf[
        df_traf["denominacion"]
        .str.contains(search, case=False, na=False)
    ]


# ─────────────────────────────────────────────────────────────────
# 5 · Colorear y mostrar mapa con pydeck
# ─────────────────────────────────────────────────────────────────
color_map = {
    0: [0,255,0,80],
    1: [255,165,0,80],
    2: [255,0,0,80],
    3: [0,0,0,80],
}

if show_traf and not df_traf.empty:
    df_traf["fill_color"] = df_traf["estado"].apply(
        lambda s: color_map.get(int(s), [128,128,128,80])
    )
    layer = pdk.Layer(
        "ScatterplotLayer", data=df_traf,
        get_position="[longitud, latitud]",
        get_fill_color="fill_color",
        get_radius=40,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=39.47, longitude=-0.376, zoom=12),
            layers=[layer],
            tooltip={"text": "{denominacion}"}
        )
    )
else:
    st.info("No hay capas para mostrar en el mapa.")


# ─────────────────────────────────────────────────────────────────
# 6 · Predicción ML (Regresión logística + métricas)
# ─────────────────────────────────────────────────────────────────
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
    ahora      = datetime.now(timezone.utc)
    X_act      = pd.DataFrame({
        "estado":[estado_act],
        "hora":  [ahora.hour],
        "diasem":[ahora.weekday()]
    })
    prob = modelo.predict_proba(X_act)[0,1]
    st.markdown("---")
    st.subheader("🔮 Predicción ML (15 min adelante)")
    st.write(f"- **Accuracy:** {acc:.2f}")
    st.write(f"- **ROC-AUC:**  {roc:.2f}")
    st.write(f"- **P(congestión ≥ 2):** {prob*100:.1f}%")
elif show_traf:
    st.warning("⚠️ Predicción ML no disponible.")


# ─────────────────────────────────────────────────────────────────
# 7 · Lista de calles bajo el mapa
# ─────────────────────────────────────────────────────────────────
if show_traf and not df_traf.empty:
    calles = sorted(df_traf["denominacion"].dropna().unique())
    st.subheader("📋 Calles mostradas")
    for calle in calles:
        st.markdown(f"- {calle}")
