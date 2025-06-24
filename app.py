import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
from markov import predict_congestion

st.set_page_config(page_title="Tráfico + Valenbisi Valencia", layout="wide")

def find_col(df, keywords):
    lc = {c: c.lower() for c in df.columns}
    for orig, low in lc.items():
        for kw in keywords:
            if kw in low:
                return orig
    return None

# ─── Cargar datos de Valenbisi ────────────────────────────────────────────
@st.cache_data(ttl=180)
def load_valenbisi():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
    params = {"dataset": "valenbisi-disponibilitat-valenbisi-dsiponibilidad", "rows": 500}
    try:
        res = requests.get(url, params=params, timeout=30).json()
        recs = res.get("records", [])
        df = pd.DataFrame([r["fields"] for r in recs])
    except Exception:
        df = pd.DataFrame()

    # Detecta dinámicamente columna de dirección y bicis
    col_dir   = find_col(df, ["address", "direcci", "direccion"])
    col_bicis = find_col(df, ["bici", "disponibl"])

    # Extrae lat/lon
    if "geo_point_2d" in df:
        df["lat"] = df["geo_point_2d"].apply(
            lambda x: float(x[0]) if isinstance(x, (list,tuple)) and len(x)==2 else None
        )
        df["lon"] = df["geo_point_2d"].apply(
            lambda x: float(x[1]) if isinstance(x, (list,tuple)) and len(x)==2 else None
        )
    else:
        df["lat"], df["lon"] = None, None

    # Renombra
    ren = {}
    if col_dir:   ren[col_dir]   = "Direccion"
    if col_bicis: ren[col_bicis] = "Bicis_disponibles"
    df = df.rename(columns=ren)

    # Filtra sólo esas columnas + lat/lon
    keep = [c for c in ["Direccion","Bicis_disponibles","lat","lon"] if c in df]
    df = df[keep]

    # Quita filas sin datos críticos
    for c in ["Direccion","lat","lon"]:
        if c in df:
            df = df.dropna(subset=[c])

    # Limpia NaN → None
    df = df.astype(object).where(pd.notnull(df), None)
    return df

# ─── Cargar datos de tráfico ─────────────────────────────────────────────
@st.cache_data(ttl=180)
def load_traffic():
    url = "https://valencia.opendatasoft.com/api/explore/v2.1/catalog/" \
          "datasets/estat-transit-temps-real-estado-trafico-tiempo-real/records"
    res = requests.get(url, params={"limit": -1}, timeout=30).json()
    df = pd.json_normalize(res["results"])

    df = df.rename(columns={
        "geo_point_2d.lat":                  "lat",
        "geo_point_2d.lon":                  "lon",
        "geo_shape.geometry.coordinates":    "path",
        "denominaci_denominaci_":            "denominacion",
        "estat_estado":                      "estado",
        "id_tram_id_tramo":                  "idtramo"
    })

    ESTADOS = {
        0: ([0, 180, 40],    "Fluido"),
        1: ([255,165,   0],  "Denso"),
        2: ([220, 30,  30],  "Congestionado"),
        3: ([0,   0,   0],   "Cortado"),
        4: ([120,120,120],   "Sin datos"),
    }
    df[["color","estado_txt"]] = df["estado"].apply(
        lambda s: pd.Series(ESTADOS.get(s if s in ESTADOS else 4))
    )
    df["timestamp"] = pd.Timestamp.utcnow()

    df = df[df["lat"].notna() & df["lon"].notna()].reset_index(drop=True)
    df = df.astype(object).where(pd.notnull(df), None)
    return df

# ─── Main ────────────────────────────────────────────────────────────────
df_traf = load_traffic()
df_bici = load_valenbisi()

# ─── Barra lateral ───────────────────────────────────────────────────────
st.sidebar.header("Filtros")
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici = st.sidebar.checkbox("Mostrar Valenbisi", True)

# --- Recargar datos manualmente -----------------------------------------
if st.sidebar.button("🔄  Actualizar datos"):
    load_traffic.clear()       # vacía la caché
    load_valenbisi.clear()
    st.experimental_rerun()    # recarga la página completa

# --- Filtro por vía / tramo ---------------------------------------------
vias = sorted(df_traf["denominacion"].dropna().unique())
vias_sel = st.sidebar.multiselect(
    "Filtrar por vía", vias, help="Selecciona una o varias vías")
if vias_sel:
    df_traf = df_traf[df_traf["denominacion"].isin(vias_sel)]

# --- Filtro mínimo de bicis ---------------------------------------------
if show_bici and not df_bici.empty and "Bicis_disponibles" in df_bici:
    max_bicis = int(df_bici["Bicis_disponibles"].max())
    min_bicis = st.sidebar.slider(
        "Mínimo bicis disponibles", 0, max_bicis, 0)
    df_bici = df_bici[df_bici["Bicis_disponibles"] >= min_bicis]

# ─── KPIs tráfico ────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
agg = df_traf["estado_txt"].value_counts(normalize=True).mul(100).round(1)
c1.metric("🚗 Fluido %",    f"{agg.get('Fluido',0)}%")
c2.metric("🚧 Denso %",     f"{agg.get('Denso',0)}%")
c3.metric("⛔ Congest. %",  f"{agg.get('Congestionado',0)}%")
c4.metric("✋ Cortado %",   f"{agg.get('Cortado',0)}%")

# ─── Definir capas PyDeck ────────────────────────────────────────────────
layers = []

if show_traf and not df_traf.empty:
    # líneas de tráfico
    layers.append(pdk.Layer(
        "PathLayer", df_traf,
        get_path="path",
        get_color="color",
        get_width=4
    ))
    # puntos de tráfico
    layers.append(pdk.Layer(
        "ScatterplotLayer", df_traf,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=60,
        pickable=True
    ))

if show_bici and not df_bici.empty:
    # puntos de Valenbisi
    layers.append(pdk.Layer(
        "ScatterplotLayer", df_bici,
        get_position="[lon, lat]",
        get_fill_color="[0,200,255]",
        get_radius=80,
        pickable=True
    ))

# ─── Centrar en Valencia ─────────────────────────────────────────────────
view = pdk.ViewState(
    latitude=39.4699,
    longitude=-0.3763,
    zoom=12
)

# ─── Tooltip combinado ───────────────────────────────────────────────────
tooltip = {
    "html": (
        # si existe 'denominacion' lo muestra, si existe 'Direccion' también
        "<b>{denominacion}</b><br/>{estado_txt}"
        "<br/><b>{Direccion}</b> {Bicis_disponibles} bicis"
    ),
    "style": {"color": "white"}
}

if layers:
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=view,
            layers=layers,
            tooltip=tooltip
        )
    )

# ─── Pronóstico de congestión ────────────────────────────────────────────
prob = predict_congestion(df_traf)
st.subheader("Pronóstico de congestión en 15 minutos")
st.progress(prob)
st.write(f"Probabilidad media de congestión: {prob:.1%}")

# ─── Tabla Valenbisi ──────────────────────────────────────────────────────
if show_bici and not df_bici.empty:
    st.subheader("Disponibilidad Valenbisi")
    cols = [c for c in ["Direccion","Bicis_disponibles"] if c in df_bici]
    st.dataframe(df_bici[cols].reset_index(drop=True))
