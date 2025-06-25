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
            # Renombrar available → Bicis_disponibles si hace falta
            if "bicis_disponibles" not in f and "slots_disponibles" in f:
                f["Bicis_disponibles"] = f.pop("slots_disponibles")
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
            # alias falls
            if "latitude" in f and "latitud" not in f:
                f["latitud"] = f["latitude"]
            if "longitude" in f and "longitud" not in f:
                f["longitud"] = f["longitude"]
            # transformar estado a int si existe
            if "estado" in f:
                try:
                    f["estado"] = int(f["estado"])
                except:
                    f["estado"] = None
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

# ————————————————————————————————————————————————————————
# Opción extra: multiselect de calles (solo si mostramos tráfico)
# ————————————————————————————————————————————————————————
if show_traf:
    df_temp = load_traffic()  # carga sin filtrar
    if "denominacion" in df_temp.columns:
        calles = sorted(df_temp["denominacion"].dropna().unique())
        seleccion = st.sidebar.multiselect(
            "Filtrar por calle (opcional)",
            options=calles,
            default=calles
        )
        # Aplicamos filtro sobre el df_traf definitivo
        df_traf_full = df_temp
        df_traf = df_traf_full[df_traf_full["denominacion"].isin(seleccion)]
    else:
        df_traf = df_temp
else:
    df_traf = load_traffic()

if st.sidebar.button("🔄 Actualizar datos"):
    st.experimental_rerun()

st.sidebar.subheader("Estados de tráfico (colores en mapa)")
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
# 3 · Carga de Valenbisi
# ─────────────────────────────────────────────────────────────────
df_bici = load_valenbisi()

if df_traf.empty and show_traf:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if df_bici.empty and show_bici:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")

# ─────────────────────────────────────────────────────────────────
# 4 · Asignar color dinámico según estado de tráfico
# ─────────────────────────────────────────────────────────────────
color_map = {
    0: [0, 255,   0,  80],   # verde
    1: [255,165,   0,  80],   # naranja
    2: [255,  0,   0,  80],   # rojo
    3: [0,    0,   0,  80],   # negro
}
df_traf["fill_color"] = df_traf["estado"].apply(
    lambda s: color_map.get(s, [200,200,200,80])
)

# ─────────────────────────────────────────────────────────────────
# 5 · Mapa
# ─────────────────────────────────────────────────────────────────
layers = []

if show_traf and not df_traf.empty and {"latitud", "longitud", "fill_color"}.issubset(df_traf.columns):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_traf,
        get_position="[longitud, latitud]",
        get_fill_color="fill_color",
        get_radius=40,
        pickable=True,
    ))

if show_bici and not df_bici.empty and {"lat", "lon"}.issubset(df_bici.columns):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_bici,
        get_position="[lon, lat]",
        get_fill_color="[0, 140, 255, 80]",
        get_radius=30,
        pickable=True,
    ))

if layers:
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=39.47, longitude=-0.376, zoom=12),
        layers=layers,
        tooltip={"text": "{denominacion}"},
    ))
else:
    st.info("No hay capas para mostrar en el mapa.")
