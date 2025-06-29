import pandas as pd
import requests
import pydeck as pdk
import streamlit as st
from datetime import datetime, timezone
from ml_model import entrenar_logreg

st.set_page_config(page_title="Tráfico y Valenbisi", layout="wide")


# ────────────────────────────────────────────────────────────
# 1 · Carga de datos con caché
#    Primero intenta CSV local; si no existe o está vacío,
#    intenta la API remota (JSON) como fallback.
# ────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)
def load_valenbisi():
    # 1a) CSV local opcional
    try:
        df = pd.read_csv("valenbisi.csv")
        # Si vienen separados lat y lon en dos columnas distintas,
        # renómbralas aquí (ajusta según tus nombres reales):
        # df.rename(columns={"latitude":"lat","longitude":"lon"}, inplace=True)
        if not df.empty:
            return df
    except FileNotFoundError:
        pass

    # 1b) Fallback API remota (JSON)
    try:
        url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
        params = {
            "dataset": "valenbisi-disponibilitat-valenbisi-dsiponibilidad",
            "rows": 500
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        recs = r.json().get("records", [])
        rows = []
        for rec in recs:
            f = rec.get("fields", {}).copy()
            if "slots_disponibles" in f:
                f["Bicis_disponibles"] = f.pop("slots_disponibles")
            f["direccion"] = f.get("address", "Desconocida")
            if isinstance(f.get("geo_point_2d"), list):
                f["lat"], f["lon"] = f["geo_point_2d"]
            rows.append(f)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error cargando Valenbisi (JSON): {e}")
        return pd.DataFrame()


@st.cache_data(ttl=180)
def load_traffic():
    # 1a) CSV local: usamos tu trafico_historico.csv
    try:
        df = pd.read_csv("trafico_historico.csv")
        if "estado" in df.columns:
            df["estado"] = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
        if not df.empty:
            return df
    except FileNotFoundError:
        pass

    # 1b) Fallback API remota (JSON)
    try:
        url = "https://valencia.opendatasoft.com/api/records/1.0/search/"
        params = {
            "dataset": "estat-transit-temps-real-estado-trafico-tiempo-real",
            "rows": 1000
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        recs = r.json().get("records", [])
        rows = []
        for rec in recs:
            f = rec.get("fields", {}).copy()
            if isinstance(f.get("geo_point_2d"), list):
                f["latitud"], f["longitud"] = f["geo_point_2d"]
            if "latitude" in f and "latitud" not in f:
                f["latitud"] = f["latitude"]
            if "longitude" in f and "longitud" not in f:
                f["longitud"] = f["longitude"]
            if "estado" in f:
                try:
                    f["estado"] = int(f["estado"])
                except:
                    f["estado"] = None
            rows.append(f)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error cargando tráfico (JSON): {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 2 · Sidebar: filtros y recarga
# ─────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
show_traf = st.sidebar.checkbox("Mostrar tráfico", True)
show_bici = st.sidebar.checkbox("Mostrar Valenbisi", True)

search_street = st.sidebar.text_input("Buscar calle (opcional)", "")

if st.sidebar.button("🔄  Actualizar datos"):
    st.rerun()

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
# 3 · Carga de datos
# ─────────────────────────────────────────────────────────────────
df_traf = load_traffic()
df_bici = load_valenbisi()

if show_traf and df_traf.empty:
    st.error("❌ No se pudieron cargar los datos de tráfico.")
if show_bici and df_bici.empty:
    st.warning("⚠️ Sin datos de Valenbisi en este momento.")


# ─────────────────────────────────────────────────────────────────
# 4 · Filtrar por calle si texto
# ─────────────────────────────────────────────────────────────────
if show_traf and search_street and "denominacion" in df_traf.columns:
    df_traf = df_traf[df_traf["denominacion"]
                      .str.contains(search_street, case=False, na=False)]


# ─────────────────────────────────────────────────────────────────
# 5 · Colorear tráfico en tiempo real
# ─────────────────────────────────────────────────────────────────
color_map = {
    0: [0, 255,   0,  80],
    1: [255,165,   0,  80],
    2: [255,  0,   0,  80],
    3: [0,    0,   0,  80],
}
if not df_traf.empty and "estado" in df_traf.columns:
    df_traf["fill_color"] = df_traf["estado"].apply(
        lambda s: color_map.get(int(s), [200, 200, 200, 80])
    )
else:
    st.error("❌ No se pudieron asignar colores: 'estado' ausente o datos vacíos.")


# ─────────────────────────────────────────────────────────────────
# 6 · Construcción de capas y despliegue del mapa
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

if show_bici and not df_bici.empty and {"lat","lon"}.issubset(df_bici.columns):
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


# ─────────────────────────────────────────────────────────────────
# 7 · Predicción ML (Regresión logística + métricas)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_logreg_model():
    try:
        df_hist = pd.read_csv(
            "trafico_historico.csv",      # <— usa el mismo CSV que carga la capa de tráfico
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
        model, acc, roc = entrenar_logreg(df_hist)
    except Exception as e:
        st.warning(f"⚠️ Error entrenando ML: {e}")
        return None, None, None

    return model, acc, roc

modelo, acc, roc = get_logreg_model()

if show_traf and modelo and not df_traf.empty and "estado" in df_traf.columns:
    estado_actual = int(df_traf["estado"].mode()[0])
    ahora = datetime.now(timezone.utc)
    X_act = pd.DataFrame({
        "estado": [estado_actual],
        "hora":   [ahora.hour],
        "diasem": [ahora.weekday()],
    })
    prob_ml = modelo.predict_proba(X_act)[0,1]

    st.markdown("---")
    st.subheader("🔮 Predicción ML (15 min adelante)")
    st.write(f"- **Accuracy:** {acc:.2f}")
    st.write(f"- **ROC-AUC:**  {roc:.2f}")
    st.write(f"- **P(congestión ≥ 2):** {prob_ml*100:.1f}%")
elif show_traf:
    st.markdown("---")
    st.warning("⚠️ Predicción ML no disponible.")


# ─────────────────────────────────────────────────────────────────
# 8 · Lista de calles bajo el mapa
# ─────────────────────────────────────────────────────────────────
if show_traf and not df_traf.empty and "denominacion" in df_traf.columns:
    calles = sorted(df_traf["denominacion"].dropna().unique())
    st.subheader("📋 Calles mostradas")
    for calle in calles:
        st.markdown(f"- {calle}")
