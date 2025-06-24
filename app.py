import pandas as pd
import requests
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Tráfico y Valenbisi", layout="wide")


# ────────────────────────────────────────────────────────────
# 1 · Carga de datos con caché
# ────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)
def load_valenbisi():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
    params = {
        "dataset": "valenbisi-disponibilitat-valenbisi-dsiponibilidad",
        "rows": 500
    }
    try:
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
            if "address" in f:
                f["direccion"] = f["address"]
            else:
                f["direccion"] = "Desconocida"
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
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        recs = r.json().get("records", [])
        rows = []
        for rec in recs:
            f = rec.get("fields", {}).copy()
            # geo_point_2d → latitud, longitud
            if isinstance(f.get("geo_point_2d"), list):
                f["latitud"], f["longitud"] = f["geo_point_2d"]
            # fallback keys
            if "latitude" in f and "latitud" not in f:
                f["latitud"] = f["latitude"]
            if "longitude" in f and "longitud" not in f:
                f["longitud"] = f["longitude"]
            rows.append(f)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error cargando tráfico: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 2 · Barra lateral: filtros y leyenda
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
    | Código | Color      |
    |--------|------------|
    | 0      | 🟢 Fluido  |
    | 1      | 🟠 Moderado|
    | 2      | 🔴 Denso   |
    | 3      | ⚫ Cortado |
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# 3 · Carga de datos
# ─────────────────────────────────────────────────────────────────
df_traf = load_traffic()
df_bici = load_valenbisi()

if df_traf.empty:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if df_bici.empty and show_bici:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")


# ─────────────────────────────────────────────────────────────────
# 4 · Definición de capas
# ─────────────────────────────────────────────────────────────────
layers = []

if show_traf and not df_traf.empty and {"latitud", "longitud"}.issubset(df_traf.columns):
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df_traf,
            get_position="[longitud, latitud]",
            get_fill_color="[255, 0, 0, 80]",
            get_radius=40,
            pickable=False,
        )
    )

if show_bici and not df_bici.empty and {"lat", "lon", "Bicis_disponibles", "direccion"}.issubset(df_bici.columns):
    bici_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_bici,
        get_position="[lon, lat]",
        get_fill_color="[0, 140, 255, 80]",
        get_radius=30,
        pickable=True,
    )
    layers.append(bici_layer)

# ─────────────────────────────────────────────────────────────────
# 5 · Chart con tooltip para Valenbisi
# ─────────────────────────────────────────────────────────────────
if layers:
    tooltip = {
        "html": "<b>Dirección:</b> {direccion} <br/>"
                "<b>Bicis disponibles:</b> {Bicis_disponibles}",
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
