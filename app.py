import pandas as pd
import requests
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Tráfico y Valenbisi", layout="wide")

# ────────────────────────────────────────────────────────────
# 1 · Parámetros dinámicos (puedes cambiar la URL/dataset aquí)
# ────────────────────────────────────────────────────────────
traffic_dataset = st.sidebar.text_input(
    "Dataset tráfico",
    value="estat-transit-temps-real-estado-trafico-tiempo-real"
)
bici_dataset = st.sidebar.text_input(
    "Dataset Valenbisi",
    value="valenbisi-disponibilitat-valenbisi-dsiponibilidad"
)

# ────────────────────────────────────────────────────────────
# 2 · Carga de datos con caché
# ────────────────────────────────────────────────────────────
@st.cache_data(ttl=180, show_spinner=False)
def load_valenbisi(dataset):
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
    params = {"dataset": dataset, "rows": 500}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    recs = r.json().get("records", [])
    rows = []
    for rec in recs:
        f = rec.get("fields", {}).copy()
        # slots_disponibles → Bicis_disponibles
        if "slots_disponibles" in f:
            f["Bicis_disponibles"] = f.pop("slots_disponibles")
        # dirección
        f["direccion"] = f.get("address", "Desconocida")
        # geo_point_2d → lat, lon
        if isinstance(f.get("geo_point_2d"), list):
            f["lat"], f["lon"] = f["geo_point_2d"]
        rows.append(f)
    return pd.DataFrame(rows)

@st.cache_data(ttl=180, show_spinner=False)
def load_traffic(dataset):
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
    params = {"dataset": dataset, "rows": 1000}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    recs = r.json().get("records", [])
    rows = []
    for rec in recs:
        f = rec.get("fields", {}).copy()
        # convertir estado a int si viene como string
        if "estado" in f:
            try: f["estado"] = int(f["estado"])
            except: pass
        # geo_point_2d → latitud, longitud
        if isinstance(f.get("geo_point_2d"), list):
            f["latitud"], f["longitud"] = f["geo_point_2d"]
        # alias falls
        if "latitude" in f and "latitud" not in f:
            f["latitud"] = f["latitude"]
        if "longitude" in f and "longitud" not in f:
            f["longitud"] = f["longitude"]
        rows.append(f)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────
# 3 · Botón para actualizar manualmente
# ─────────────────────────────────────────────────────────────────
if st.sidebar.button("🔄 Actualizar datos"):
    load_valenbisi.clear()
    load_traffic.clear()
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────
# 4 · Filtros de visibilidad
# ─────────────────────────────────────────────────────────────────
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici = st.sidebar.checkbox("Mostrar Valenbisi", True)

# ─────────────────────────────────────────────────────────────────
# 5 · Carga de datos
# ─────────────────────────────────────────────────────────────────
df_traf = load_traffic(traffic_dataset)
df_bici = load_valenbisi(bici_dataset)

if df_traf.empty and show_traf:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if df_bici.empty and show_bici:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")

# ─────────────────────────────────────────────────────────────────
# 6 · Asignar color dinámico según estado de tráfico
# ─────────────────────────────────────────────────────────────────
color_map = {
    0: [0, 255,   0,  80],  # verde
    1: [255,165,   0,  80],  # naranja
    2: [255,  0,   0,  80],  # rojo
    3: [0,    0,   0,  80],  # negro
}
if "estado" in df_traf.columns:
    df_traf["fill_color"] = df_traf["estado"].map(color_map)

# ─────────────────────────────────────────────────────────────────
# 7 · Construcción de capas
# ─────────────────────────────────────────────────────────────────
layers = []

if show_traf and not df_traf.empty and {"latitud","longitud","fill_color"}.issubset(df_traf.columns):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_traf,
        get_position="[longitud, latitud]",
        get_fill_color="fill_color",
        get_radius=40,
        pickable=True,
    ))

if show_bici and not df_bici.empty and {"lat","lon","Bicis_disponibles","direccion"}.issubset(df_bici.columns):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_bici,
        get_position="[lon, lat]",
        get_fill_color="[0, 140, 255, 80]",
        get_radius=30,
        pickable=True,
    ))

# ─────────────────────────────────────────────────────────────────
# 8 · Tooltip & despliegue
# ─────────────────────────────────────────────────────────────────
if layers:
    tooltip = {
        "html": (
            "<b>🚦 Tráfico:</b> {denominacion}<br/>"
            "<b>🚲 Bicis disp.:</b> {Bicis_disponibles}"
        ),
        "style": {"backgroundColor": "white", "color": "black"}
    }
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=39.47, longitude=-0.376, zoom=12),
            layers=layers,
            tooltip=tooltip
        )
    )
else:
    st.info("No hay capas para mostrar en el mapa.")
