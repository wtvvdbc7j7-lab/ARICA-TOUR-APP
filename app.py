# -*- coding: utf-8 -*-
# ============================================================
# Asistente Turístico Inteligente - Región de Arica y Parinacota
# Streamlit App (versión extendida)
# - Más atractivos (con fotos)
# - Duración sugerida por lugar (min)
# - Horarios de apertura (validación suave)
# - Estimación de tiempo y costo de traslado (editable por el usuario)
# - Itinerario que agrupa por cercanía y calcula horas del día
# - Mapa interactivo por "día"
# - Opción para mostrar un QR con el enlace de la app (si se configura)
# ============================================================

import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------- Configuración general ----------------
LOGO_PATH = "logo.png"  # coloca tu logo en el repo con este nombre (ver sección de logo)
APP_URL = ""            # cuando la app esté desplegada en Streamlit Cloud, pega aquí la URL pública
                        # ejemplo: "https://tu-usuario-tu-repo.streamlit.app"

st.set_page_config(
    page_title="Asistente Turístico Arica y Parinacota",
    page_icon=LOGO_PATH if LOGO_PATH else "🧭",  # si no pones logo.png, usará el emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- Atractivos (dataset base) -----------------
# Campos:
# id, nombre, tipo, lat, lon, desc, link, img, dur_min (tiempo sugerido), horario ("HH:MM-HH:MM")
ATRACTIVOS = pd.DataFrame([
    # --- Arica urbano / costa ---
    {"id": 1, "nombre": "Playa Chinchorro", "tipo": "Playa", "lat": -18.4718, "lon": -70.3039,
     "desc": "Amplia playa urbana ideal para caminar y disfrutar del atardecer.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/5/50/Playa_Chinchorro_Arica.jpg",
     "dur_min": 90, "horario": "00:00-23:59"},
    {"id": 2, "nombre": "Morro de Arica (Mirador/Museo)", "tipo": "Mirador", "lat": -18.4824, "lon": -70.3249,
     "desc": "Ícono histórico con vistas de la ciudad; museo de sitio.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/2/20/Arica_-_Morro.jpg",
     "dur_min": 60, "horario": "10:00-18:00"},
    {"id": 3, "nombre": "Cuevas de Anzota", "tipo": "Formación rocosa", "lat": -18.5495, "lon": -70.3435,
     "desc": "Sendero costero entre acantilados y cuevas con fauna marina.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/b/b3/Cuevas_de_Anzota.jpg",
     "dur_min": 90, "horario": "09:00-18:00"},
    {"id": 4, "nombre": "Humedal del Río Lluta", "tipo": "Humedal", "lat": -18.4389, "lon": -70.3147,
     "desc": "Santuario, hábitat de aves migratorias. Llevar binoculares.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/5/59/Humedal_Rio_Lluta.jpg",
     "dur_min": 60, "horario": "08:00-18:00"},
    {"id": 5, "nombre": "Museo Arqueológico San Miguel de Azapa", "tipo": "Museo", "lat": -18.5212, "lon": -70.1745,
     "desc": "Momias Chinchorro y arqueología regional.",
     "link": "https://www.museoazapa.cl/",
     "img": "https://www.museoazapa.cl/wp-content/uploads/2019/04/IMG_6031.jpg",
     "dur_min": 90, "horario": "10:00-18:00"},
    {"id": 6, "nombre": "Valle de Azapa (Olivares y geoglifos)", "tipo": "Valle", "lat": -18.5000, "lon": -70.2000,
     "desc": "Agricultura, pueblos, geoglifos y gastronomía local.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Valle_de_Azapa.jpg",
     "dur_min": 120, "horario": "08:00-19:00"},
    {"id": 7, "nombre": "Terminal Agro Arica", "tipo": "Mercado", "lat": -18.4827, "lon": -70.2994,
     "desc": "Mercado típico con jugos, frutas, aceitunas y cocina local.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/1/19/Terminal_Agro_Arica.jpg",
     "dur_min": 60, "horario": "08:00-19:00"},
    {"id": 12, "nombre": "Oficina Sobraya (ruinas salitreras)", "tipo": "Sitio histórico", "lat": -18.5895, "lon": -70.1330,
     "desc": "Vestigios de la época del salitre en el desierto costero.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/1/12/Sobraya_ruinas.jpg",
     "dur_min": 60, "horario": "08:00-17:00"},
    # --- Precordillera / Altiplano ---
    {"id": 8, "nombre": "Putre (pueblo altiplánico)", "tipo": "Pueblo", "lat": -18.1987, "lon": -69.5590,
     "desc": "Arquitectura andina; base para explorar el altiplano.",
     "link": "https://www.sernatur.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/7/7e/Putre.jpg",
     "dur_min": 90, "horario": "00:00-23:59"},
    {"id": 9, "nombre": "Parque Nacional Lauca", "tipo": "Parque", "lat": -18.2371, "lon": -69.2960,
     "desc": "Lagunas, bofedales, fauna altoandina y volcanes (CONAF).",
     "link": "https://www.conaf.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/4/4a/Parque_Nacional_Lauca.jpg",
     "dur_min": 180, "horario": "08:00-18:00"},
    {"id": 10, "nombre": "Lago Chungará", "tipo": "Lago", "lat": -18.2485, "lon": -69.1608,
     "desc": "Lago altoandino junto al volcán Parinacota.",
     "link": "https://www.conaf.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/4/4b/Lago_Chungara.jpg",
     "dur_min": 90, "horario": "08:00-18:00"},
    {"id": 11, "nombre": "Salar de Surire (Monumento Natural)", "tipo": "Salar", "lat": -18.9990, "lon": -69.0410,
     "desc": "Flamencos, vicuñas y salares; consultar estado de ruta.",
     "link": "https://www.conaf.cl/",
     "img": "https://upload.wikimedia.org/wikipedia/commons/7/7a/Surire.jpg",
     "dur_min": 120, "horario": "08:00-17:00"},
])

# ---------------- Utilidades geográficas ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def distance_matrix(points):
    n = len(points)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
            M[i, j] = M[j, i] = d
    return M

# ---------------- Clustering (k ≈ días) ----------------
def maxmin_seeds(coords, k):
    if k <= 0: return []
    n = len(coords)
    seeds = [0]
    while len(seeds) < k:
        dmin = []
        for i in range(n):
            if i in seeds:
                dmin.append(-1)
                continue
            mins = [haversine_km(coords[i][0], coords[i][1], coords[s][0], coords[s][1]) for s in seeds]
            dmin.append(min(mins))
        idx = int(np.argmax(dmin))
        seeds.append(idx)
    return seeds

def assign_to_clusters(coords, centers_idx):
    clusters = [[] for _ in centers_idx]
    for i, (lat, lon) in enumerate(coords):
        best_k, best_d = 0, float("inf")
        for k, s in enumerate(centers_idx):
            d = haversine_km(lat, lon, coords[s][0], coords[s][1])
            if d < best_d:
                best_d, best_k = d, k
        clusters[best_k].append(i)
    return clusters

def recompute_centers(coords, clusters):
    new_centers_idx = []
    for cl in clusters:
        if not cl:
            new_centers_idx.append(None)
            continue
        lat_mean = float(np.mean([coords[i][0] for i in cl]))
        lon_mean = float(np.mean([coords[i][1] for i in cl]))
        best_i, best_d = None, float("inf")
        for i in cl:
            d = haversine_km(lat_mean, lon_mean, coords[i][0], coords[i][1])
            if d < best_d:
                best_d, best_i = d, i
        new_centers_idx.append(best_i)
    return new_centers_idx

def simple_kmeans_like(coords, k, iters=5):
    if k <= 0: return [[]]
    k = min(k, len(coords))
    centers_idx = maxmin_seeds(coords, k)
    for _ in range(iters):
        clusters = assign_to_clusters(coords, centers_idx)
        new_centers = recompute_centers(coords, clusters)
        centers_idx = [nc if nc is not None else centers_idx[i] for i, nc in enumerate(new_centers)]
    clusters = assign_to_clusters(coords, centers_idx)
    return clusters

def nearest_neighbor_route(points_idx, distM):
    if len(points_idx) <= 2:
        return points_idx[:]
    unvisited = set(points_idx)
    current = points_idx[0]
    route = [current]
    unvisited.remove(current)
    while unvisited:
        nxt = min(unvisited, key=lambda j: distM[current, j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return route

# ---------------- Helpers de horarios y traslados ----------------
def parse_hhmm(txt):
    return datetime.strptime(txt, "%H:%M").time()

def dentro_horario(hhmm, rango_txt):
    """hhmm: datetime.time; rango_txt: 'HH:MM-HH:MM' """
    try:
        ini_txt, fin_txt = rango_txt.split("-")
        ini, fin = parse_hhmm(ini_txt), parse_hhmm(fin_txt)
        return ini <= hhmm <= fin
    except Exception:
        return True  # si no hay horario bien formado, no bloqueamos

def estimar_tiempo_traslado(dist_km, vel_kmh):
    """Devuelve timedelta estimado dado km y velocidad."""
    horas = dist_km / max(vel_kmh, 1e-6)
    mins = int(round(horas * 60))
    return timedelta(minutes=mins)

# ---------------- Estado de sesión ----------------
if "favoritos" not in st.session_state:
    st.session_state.favoritos = set()
if "seleccion" not in st.session_state:
    st.session_state.seleccion = set([1, 3, 4, 5])  # pre-selección de ejemplo

# ---------------- Sidebar y navegación ----------------
st.sidebar.image(LOGO_PATH, width=140) if LOGO_PATH else None
st.sidebar.title("🧭 Asistente Turístico")
seccion = st.sidebar.radio(
    "Ir a",
    ["🏠 Pantalla de inicio", "📊 Panel de inicio", "🗺️ Explora destinos",
     "ℹ️ Detalles de la atracción", "🗓️ Planificador de viajes", "🧭 Mapa interactivo", "🔗 QR de la app"]
)

# ---------------- Pantalla de inicio ----------------
if seccion == "🏠 Pantalla de inicio":
    st.title("Asistente Turístico • Arica y Parinacota")
    st.markdown("""
    Descubre atractivos, revisa información clave (duración, horarios), **planifica** por cercanía y visualiza el
    itinerario en un **mapa interactivo**. También puedes **editar** manualmente el plan si deseas.
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Arica_-_Morro.jpg/1280px-Arica_-_Morro.jpg",
        caption="Morro de Arica (referencial)",
        use_column_width=True
    )

# ---------------- Panel de inicio ----------------
if seccion == "📊 Panel de inicio":
    st.header("Resumen")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Atractivos disponibles", len(ATRACTIVOS))
    with c2: st.metric("Seleccionados", len(st.session_state.seleccion))
    with c3: st.metric("Favoritos", len(st.session_state.favoritos))
    st.divider()
    st.subheader("Selección actual")
    st.dataframe(ATRACTIVOS[ATRACTIVOS["id"].isin(st.session_state.seleccion)][
        ["id", "nombre", "tipo", "dur_min", "horario"]
    ], use_container_width=True)

# ---------------- Explora destinos ----------------
if seccion == "🗺️ Explora destinos":
    st.header("Explora destinos")
    l, r = st.columns([1, 2])
    with l:
        tipos = ["Todos"] + sorted(ATRACTIVOS["tipo"].unique().tolist())
        tsel = st.selectbox("Filtrar por tipo", tipos)
        q = st.text_input("Buscar por nombre:", "")
        candidatos = ATRACTIVOS.copy()
        if tsel != "Todos":
            candidatos = candidatos[candidatos["tipo"] == tsel]
        if q.strip():
            candidatos = candidatos[candidatos["nombre"].str.contains(q, case=False, regex=False)]

        ids = candidatos["id"].tolist()
        seleccion_local = st.multiselect(
            "Selecciona para tu itinerario",
            options=ids,
            default=sorted(st.session_state.seleccion.intersection(ids)),
            format_func=lambda i: candidatos.loc[candidatos["id"] == i, "nombre"].iloc[0]
        )
        if st.button("Guardar selección"):
            st.session_state.seleccion = set(seleccion_local)
            st.success("Selección actualizada.")
        st.divider()
        st.markdown("**Favoritos**")
        for _, row in candidatos.iterrows():
            label = ("⭐ " if row["id"] in st.session_state.favoritos else "☆ ") + row["nombre"]
            if st.button(label, key=f"fav_{row['id']}"):
                if row["id"] in st.session_state.favoritos:
                    st.session_state.favoritos.remove(row["id"])
                else:
                    st.session_state.favoritos.add(row["id"])
                st.rerun()
    with r:
        st.subheader("Vista")
        st.dataframe(candidatos[["id", "nombre", "tipo", "dur_min", "horario", "desc"]], use_container_width=True, height=480)

# ---------------- Detalles ----------------
if seccion == "ℹ️ Detalles de la atracción":
    st.header("Detalles de la atracción")
    op = ATRACTIVOS.sort_values("nombre")
    elegido = st.selectbox(
        "Elige un atractivo",
        options=op["id"].tolist(),
        format_func=lambda i: op.loc[op["id"] == i, "nombre"].iloc[0]
    )
    data = ATRACTIVOS[ATRACTIVOS["id"] == elegido].iloc[0]
    cA, cB = st.columns([2, 1])
    with cA:
        st.subheader(data["nombre"])
        st.write(f"**Tipo:** {data['tipo']}  |  **Duración sugerida:** {data['dur_min']} min")
        st.write(f"**Horario sugerido:** {data['horario']}")
        st.write(data["desc"])
        st.markdown(f"[Más información]({data['link']})")
        if data["img"]:
            st.image(data["img"], use_column_width=True)
    with cB:
        st.map(pd.DataFrame([{"lat": data["lat"], "lon": data["lon"]}]), zoom=11, use_container_width=True)
        tog = st.toggle("Añadir/Quitar de selección", value=(elegido in st.session_state.seleccion))
        if tog: st.session_state.seleccion.add(elegido)
        else:   st.session_state.seleccion.discard(elegido)

# ---------------- Planificador ----------------
if seccion == "🗓️ Planificador de viajes":
    st.header("Planificador de viajes (proximidad + tiempos y costos)")

    sel_df = ATRACTIVOS[ATRACTIVOS["id"].isin(st.session_state.seleccion)].reset_index(drop=True)
    if sel_df.empty:
        st.warning("Primero elige atractivos en **Explora destinos**.")
        st.stop()

    c0, c1, c2, c3 = st.columns([1, 1, 1, 2])
    with c0:
        dias = st.number_input("Días", min_value=1, max_value=10, value=2, step=1)
    with c1:
        salida_hora = st.time_input("Hora de salida del día", value=datetime.strptime("09:00","%H:%M").time())
    with c2:
        vel_kmh = st.number_input("Velocidad media (km/h)", min_value=10, max_value=120, value=40, step=5,
                                  help="Usa 35–45 en ciudad, 60–80 en rutas.")
    with c3:
        costo_km = st.number_input("Costo aprox. por km (CLP)", min_value=0, max_value=5000, value=500, step=50,
                                   help="Para taxi/traslado. Pon 0 si no te interesa el costo.")

    coords = list(zip(sel_df["lat"].tolist(), sel_df["lon"].tolist()))
    distM = distance_matrix(coords)
    clusters = simple_kmeans_like(coords, k=int(dias), iters=6)

    plan_rows = []
    advertencias = []
    total_costo = 0
    total_km = 0

    for d, cl in enumerate(clusters, start=1):
        if not cl:
            plan_rows.append({"Día": d, "Hora": "—", "Atractivo": "(sin asignar)",
                              "Estancia_min": 0, "Traslado_km": 0.0, "Traslado_min": 0, "Costo_traslado": 0})
            continue

        orden = nearest_neighbor_route(cl, distM)
        hora_actual = datetime.combine(datetime.today(), salida_hora)

        for i, idx in enumerate(orden):
            lugar = sel_df.iloc[idx]
            # traslado desde el anterior
            if i == 0:
                km = 0.0
                t_traslado = timedelta(minutes=0)
            else:
                prev_idx = orden[i-1]
                km = float(distM[prev_idx, idx])
                t_traslado = estimar_tiempo_traslado(km, vel_kmh)
                hora_actual += t_traslado
            # control de horario
            hhmm = hora_actual.time()
            if not dentro_horario(hhmm, lugar["horario"]):
                advertencias.append(f"⚠️ {lugar['nombre']} cae fuera de horario (llegada aprox {hhmm.strftime('%H:%M')}, rango {lugar['horario']}).")

            # estancia
            est_min = int(lugar["dur_min"])
            fin_visita = hora_actual + timedelta(minutes=est_min)

            # costos
            costo = int(round(km * costo_km))
            total_costo += costo
            total_km += km

            plan_rows.append({
                "Día": d,
                "Hora": hora_actual.strftime("%H:%M"),
                "Atractivo": lugar["nombre"],
                "Estancia_min": est_min,
                "Traslado_km": round(km, 1),
                "Traslado_min": int(t_traslado.total_seconds() // 60),
                "Costo_traslado": costo
            })

            # mover reloj al fin de visita
            hora_actual = fin_visita

    st.subheader("Itinerario sugerido")
    df_plan = pd.DataFrame(plan_rows)
    st.dataframe(df_plan, use_container_width=True)

    st.caption("Puedes descargar y editar el itinerario en tu computador.")
    st.download_button("⬇️ Descargar CSV", df_plan.to_csv(index=False).encode("utf-8"),
                       "itinerario_arica.csv", "text/csv")

    # Resumen por día
    st.divider()
    st.subheader("Resumen y avisos")
    colA, colB = st.columns([1,1])
    with colA:
        resumen = df_plan.groupby("Día").agg(
            Paradas=("Atractivo", "count"),
            Km_traslado=("Traslado_km", "sum"),
            Min_traslado=("Traslado_min", "sum"),
            Estancia_total_min=("Estancia_min", "sum"),
            Costo_CLP=("Costo_traslado", "sum")
        ).reset_index()
        st.dataframe(resumen, use_container_width=True)
    with colB:
        st.metric("KM totales aprox", f"{total_km:.1f}")
        st.metric("Costo total aprox (CLP)", f"{int(total_costo):,}".replace(",", "."))

    if advertencias:
        st.warning(" • ".join(advertencias))

# ---------------- Mapa interactivo ----------------
if seccion == "🧭 Mapa interactivo":
    st.header("Mapa interactivo")
    sel_df = ATRACTIVOS[ATRACTIVOS["id"].isin(st.session_state.seleccion)].reset_index(drop=True)
    if sel_df.empty:
        st.warning("No hay selección. Ve a **Explora destinos**.")
        st.stop()
    dias_color = st.number_input("Días (para colorear clusters)", 1, 10, 2)
    coords = list(zip(sel_df["lat"].tolist(), sel_df["lon"].tolist()))
    clusters = simple_kmeans_like(coords, k=int(dias_color), iters=6)

    registros = []
    palette = [
        [31, 119, 180],[255, 127, 14],[44, 160, 44],
        [214, 39, 40],[148, 103, 189],[140, 86, 75],
        [227, 119, 194],[127, 127, 127],[188, 189, 34],[23, 190, 207],
    ]
    for d, cl in enumerate(clusters, start=1):
        for idx in cl:
            r = sel_df.iloc[idx]
            registros.append({
                "nombre": r["nombre"], "tipo": r["tipo"], "lat": r["lat"], "lon": r["lon"],
                "dia": d, "color": palette[(d-1) % len(palette)]
            })
    mdf = pd.DataFrame(registros)
    view_state = pdk.ViewState(latitude=-18.48, longitude=-70.32, zoom=9)
    layer = pdk.Layer("ScatterplotLayer", data=mdf, get_position='[lon, lat]',
                      get_radius=2200, get_fill_color="color", pickable=True)
    tooltip = {"text": "{nombre}\nTipo: {tipo}\nDía: {dia}"}
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer], tooltip=tooltip))

# ---------------- QR de la app ----------------
if seccion == "🔗 QR de la app":
    st.header("Código QR del enlace de la aplicación")
    if APP_URL:
        st.info(f"Enlace actual configurado: {APP_URL}")
        st.markdown("**Genera el QR con una herramienta externa o con un script local (ver pasos más abajo).**")
    else:
        st.warning("Aún no has configurado APP_URL en el código. Cuando publiques la app, pega aquí tu URL pública.")
    st.markdown("""
    **Cómo generar el QR**  
    1) Con una web: prueba `qr-code-generator.com` o `goqr.me`, pega tu URL pública y descarga el PNG.  
    2) Con Python (script local) usando `qrcode` + `Pillow`. Consulta las instrucciones en el README o en esta misma app (más abajo).
    """)
