"""Monitor y Predictor de Amenaza de Asteroides Cercanos a la Tierra (NASA NeoWs).

Aplicación interactiva de Ciencia de Datos, Astrofísica y Machine Learning para la monitorización
en tiempo real de telemetría orbital de la NASA, clasificación de Asteroides Potencialmente Peligrosos (PHA),
cálculo de energía de impacto en megatones de TNT, simulación de deflexión cinética (Misión DART)
y visualización orbital 3D interactiva.
"""

from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api import obtener_telemetria_nasa, cargar_datos_respaldo
from src.impact_physics import (
    calcular_deflexion_dart,
    calcular_energia_cinetica,
    estimar_diametro_crater_km,
    estimar_magnitud_desde_diametro,
    estimar_diametro_desde_magnitud,
)
from src.ml_model import (
    CARACTERISTICAS_POR_DEFECTO,
    entrenar_clasificador_asteroides,
    predecir_riesgo_asteroide,
)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitor y Predicción de Asteroides NASA",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inyección de estilos CSS para paneles de telemetría y tarjetas métricas
st.markdown(
    """
    <style>
    .tarjeta-metrica {
        background-color: #161e31;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #4ea8de;
        margin-bottom: 12px;
    }
    .alerta-peligro {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        color: #fca5a5;
    }
    .alerta-seguro {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px;
        color: #6ee7b7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. CACHÉ Y CARGA DE MODELOS / DATOS HISTÓRICOS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtener_modelo_referencia():
    """Entrena y memoriza en memoria el modelo Random Forest con el conjunto histórico."""
    df_historico = cargar_datos_respaldo()
    modelo, caracteristicas, metricas = entrenar_clasificador_asteroides(df_historico)
    return modelo, caracteristicas, metricas, df_historico


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_telemetria_viva(dias_atras: int, clave_api: str):
    """Consulta la API oficial de NASA NeoWs para el periodo seleccionado."""
    hoy = datetime.today()
    fecha_inicio = (hoy - timedelta(days=dias_atras)).strftime("%Y-%m-%d")
    fecha_fin = hoy.strftime("%Y-%m-%d")
    df_datos, aviso = obtener_telemetria_nasa(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        clave_api=clave_api
    )
    return df_datos, aviso


# -----------------------------------------------------------------------------
# 3. BARRA LATERAL (CONTROLES Y CONFIGURACIÓN)
# -----------------------------------------------------------------------------
st.sidebar.title("☄️ Control de Misión")
st.sidebar.markdown("**Sistema de alerta temprana y telemetría orbital de la NASA.**")

st.sidebar.subheader("🔑 Conexión con la API de NASA")
clave_api_usuario = st.sidebar.text_input(
    "Clave de API NASA (NASA API Key)",
    value="DEMO_KEY",
    type="password",
    help="Usa 'DEMO_KEY' (con límite de peticiones) o tu clave gratuita de https://api.nasa.gov",
)

st.sidebar.subheader("📅 Ventana de Telemetría")
dias_seleccionados = st.sidebar.select_slider(
    "Días de telemetría a monitorizar:",
    options=[1, 3, 5, 7],
    value=7,
    help="La API de NeoWs Feed permite hasta 7 días consecutivos por llamada.",
)

st.sidebar.subheader("⚖️ Calibración del Clasificador ML")
umbral_corte = st.sidebar.slider(
    "Umbral de decisión de amenaza (Sensibilidad):",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="En defensa planetaria, un umbral menor minimiza los falsos negativos para evitar que un impacto pase desapercibido.",
)

entrenar_con_historico = st.sidebar.checkbox(
    "Usar conjunto histórico balanceado de la NASA (Recomendado)",
    value=True,
    help="Garantiza volumen estadístico y balanceo ante la escasez de asteroides peligrosos en periodos cortos.",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Portafolio de Ciencia de Datos & Astrofísica**  
    - Datos: [NASA JPL NeoWs REST API](https://api.nasa.gov/)  
    - Modelo: Random Forest Classifier  
    - Autor: [David Álvarez (Dalvarez78)](https://github.com/Dalvarez78)
    """
)

# -----------------------------------------------------------------------------
# 4. CARGA DE TELEMETRÍA Y EJECUCIÓN DE MODELO
# -----------------------------------------------------------------------------
modelo_base, variables_base, metricas_base, df_historico = obtener_modelo_referencia()

with st.spinner("Descargando telemetría orbital en tiempo real desde la NASA..."):
    df_telemetria, aviso_api = descargar_telemetria_viva(dias_seleccionados, clave_api_usuario)

# Selección de modelo de inferencia
if not entrenar_con_historico and not df_telemetria.empty and len(df_telemetria["Es_Peligroso"].unique()) >= 2:
    modelo_operativo, variables_operativas, metricas_operativas = entrenar_clasificador_asteroides(df_telemetria)
else:
    modelo_operativo, variables_operativas, metricas_operativas = modelo_base, variables_base, metricas_base

# Inferencia probabilística sobre la telemetría viva
if not df_telemetria.empty and modelo_operativo:
    probabilidades, predicciones = predecir_riesgo_asteroide(
        modelo_operativo,
        df_telemetria,
        variables_operativas,
        umbral_decision=umbral_corte
    )
    df_telemetria["Probabilidad_Riesgo_%"] = probabilidades
    df_telemetria["Prediccion_Modelo"] = predicciones

# -----------------------------------------------------------------------------
# 5. ENCABEZADO Y PRESENTACIÓN
# -----------------------------------------------------------------------------
st.title("☄️ Monitor y Predicción de Asteroides Potencialmente Peligrosos (PHA)")
st.markdown(
    """
    Plataforma integral de telemetría espacial con la **API de la NASA**, modelo de **Machine Learning**
    en tiempo real para clasificar el nivel de amenaza de objetos cercanos a la Tierra (NEOs),
    física balística de impacto en megatones de TNT, simulación de deflexión cinética (**Misión DART**)
    y explorador 3D orbital.
    """
)

if aviso_api:
    st.info(f"ℹ️ {aviso_api}")

# -----------------------------------------------------------------------------
# 6. PESTAÑAS PRINCIPALES DEL DASHBOARD
# -----------------------------------------------------------------------------
pestana_monitor, pestana_simulador, pestana_dart, pestana_3d, pestana_ml, pestana_metodologia = st.tabs(
    [
        "🛰️ Monitor en Vivo",
        "🎯 Simulador de Impacto",
        "🛡️ Defensa Planetaria (Misión DART)",
        "🌐 Visualizador 3D Orbital",
        "🧠 Machine Learning & Rendimiento",
        "🔬 Metodología Científica NASA",
    ]
)

# =============================================================================
# PESTAÑA 1: MONITOR EN VIVO
# =============================================================================
with pestana_monitor:
    if df_telemetria.empty:
        st.warning("No se pudo obtener la telemetría. Comprueba tu conexión o la clave de API.")
    else:
        # Fila de métricas clave (KPIs)
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        total_asteroides = len(df_telemetria)
        total_amenazas = int(df_telemetria["Es_Peligroso"].sum())
        velocidad_maxima = df_telemetria["Velocidad_kms"].max()
        paso_minimo_ld = df_telemetria["Distancia_Lunar"].min()
        energia_maxima_mt = df_telemetria["Energia_MT"].max() if "Energia_MT" in df_telemetria.columns else 0.0

        col_kpi1.metric("Objetos Monitorizados", f"{total_asteroides} NEOs", delta=f"{dias_seleccionados} días")
        col_kpi2.metric(
            "Amenazas Reales (PHA)",
            f"{total_amenazas}",
            delta=f"{(total_amenazas/max(1, total_asteroides)*100):.1f}% del total",
            delta_color="inverse" if total_amenazas > 0 else "normal",
        )
        col_kpi3.metric("Asteroide Más Rápido", f"{velocidad_maxima:.1f} km/s", f"{velocidad_maxima * 3600:,.0f} km/h")
        col_kpi4.metric("Aproximación Más Cercana", f"{paso_minimo_ld:.2f} LD", f"{paso_minimo_ld * 384400:,.0f} km")
        col_kpi5.metric("Energía Máxima Cinética", f"{energia_maxima_mt:,.2f} MT", "Megatones TNT")

        st.divider()

        # Gráfico interactivo de proximidad y velocidad
        st.subheader("🌌 Distribución Orbital y Clasificación de Peligrosidad")
        st.caption("Eje X: Distancia mínima de paso (Distancias Lunares). Eje Y: Velocidad relativa (km/s). Tamaño: Diámetro estimado.")

        figura_dispersion = px.scatter(
            df_telemetria,
            x="Distancia_Lunar",
            y="Velocidad_kms",
            size="Diametro_Max_km",
            color="Es_Peligroso",
            hover_name="Nombre",
            hover_data={
                "Distancia_Lunar": ":.2f",
                "Velocidad_kms": ":.2f",
                "Diametro_Max_km": ":.3f",
                "Magnitud_Absoluta_H": ":.1f",
                "Probabilidad_Riesgo_%": ":.1f",
            },
            color_discrete_map={True: "#EF4444", False: "#3B82F6"},
            labels={
                "Distancia_Lunar": "Distancia a la Tierra (Distancias Lunares, 1 LD ≈ 384.400 km)",
                "Velocidad_kms": "Velocidad Relativa (km/s)",
                "Es_Peligroso": "¿Peligro Real?",
                "Diametro_Max_km": "Diámetro Máx (km)",
            },
            template="plotly_dark",
            height=480,
        )
        figura_dispersion.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(figura_dispersion, use_container_width=True)

        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            st.subheader("📊 Distribución por Diámetro Máximo Estimado")
            figura_diametros = px.histogram(
                df_telemetria,
                x="Diametro_Max_km",
                color="Es_Peligroso",
                nbins=25,
                color_discrete_map={True: "#EF4444", False: "#3B82F6"},
                template="plotly_dark",
                labels={"Diametro_Max_km": "Diámetro Máximo Estimado (km)", "count": "Número de Asteroides"},
            )
            figura_diametros.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(figura_diametros, use_container_width=True)

        with col_graf2:
            st.subheader("⚡ Calificación Probabilística del Modelo")
            figura_cajas = px.box(
                df_telemetria,
                x="Es_Peligroso",
                y="Probabilidad_Riesgo_%",
                color="Es_Peligroso",
                color_discrete_map={True: "#EF4444", False: "#3B82F6"},
                points="all",
                template="plotly_dark",
                labels={"Es_Peligroso": "¿Es Asteroide Peligroso (PHA)?", "Probabilidad_Riesgo_%": "Probabilidad Calculada (%)"},
            )
            figura_cajas.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(figura_cajas, use_container_width=True)

        # Registro en formato tabla
        st.subheader("📋 Registro Detallado de Telemetría Orbital")
        columnas_mostrar = [
            "Nombre",
            "Fecha_Aproximacion",
            "Diametro_Medio_km",
            "Magnitud_Absoluta_H",
            "Velocidad_kms",
            "Distancia_Lunar",
            "Es_Peligroso",
            "Probabilidad_Riesgo_%",
            "Energia_MT",
        ]
        df_visualizar = df_telemetria[[c for c in columnas_mostrar if c in df_telemetria.columns]].sort_values(
            by="Probabilidad_Riesgo_%", ascending=False
        )

        st.dataframe(
            df_visualizar.style.format({
                "Diametro_Medio_km": "{:.3f} km",
                "Magnitud_Absoluta_H": "{:.1f}",
                "Velocidad_kms": "{:.1f} km/s",
                "Distancia_Lunar": "{:.2f} LD",
                "Probabilidad_Riesgo_%": "{:.1f}%",
                "Energia_MT": "{:,.2f} MT",
            }),
            use_container_width=True,
            height=300,
        )

        # Botón para descargar en CSV
        csv_bytes = df_telemetria.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Telemetría en Formato CSV",
            data=csv_bytes,
            file_name=f"telemetria_asteroides_nasa_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# =============================================================================
# PESTAÑA 2: SIMULADOR DE IMPACTO BALÍSTICO
# =============================================================================
with pestana_simulador:
    st.subheader("🎯 Simulador Físico de Trayectorias y Balística de Impacto")
    st.markdown(
        """
        Modela las consecuencias físicas y la probabilidad de amenaza de un asteroide ficticio o de prueba.
        Permite calcular en tiempo real la **energía cinética en Megatones de TNT**, la dimensión del cráter
        y su correlación con la clasificación del modelo de Machine Learning.
        """
    )

    col_sim_izq, col_sim_der = st.columns([1, 1])

    with col_sim_izq:
        st.markdown("#### ⚙️ Parámetros Físicos y Cinemáticos")
        diametro_metros = st.slider("Diámetro del asteroide (metros):", min_value=10, max_value=5000, value=350, step=10)
        diametro_kilometros = diametro_metros / 1000.0

        magnitud_estimada = estimar_magnitud_desde_diametro(diametro_kilometros)
        magnitud_h_simulada = st.number_input(
            "Magnitud Absoluta H (brillo intrínseco):",
            min_value=10.0,
            max_value=32.0,
            value=float(magnitud_estimada),
            step=0.1,
            help="Menor número indica asteroide más grande. El límite crítico de la NASA es H <= 22.0.",
        )

        velocidad_simulada_kms = st.slider(
            "Velocidad relativa de aproximación (km/s):",
            min_value=5.0,
            max_value=60.0,
            value=22.5,
            step=0.5,
            help="Velocidad habitual de sobrevuelo: entre 12 y 25 km/s.",
        )

        distancia_simulada_ld = st.slider(
            "Distancia proyectada de sobrevuelo (Distancias Lunares - LD):",
            min_value=0.05,
            max_value=50.0,
            value=3.5,
            step=0.1,
            help="1 Distancia Lunar ≈ 384.400 km. El umbral crítico MOID de la NASA es 19.5 LD (0.05 AU).",
        )
        distancia_simulada_au = (distancia_simulada_ld * 384400.0) / 149597870.7

    with col_sim_der:
        st.markdown("#### 🔮 Evaluación Balística y Predictiva")

        datos_simulacion = pd.DataFrame({
            "Magnitud_Absoluta_H": [magnitud_h_simulada],
            "Diametro_Max_km": [diametro_kilometros],
            "Velocidad_kms": [velocidad_simulada_kms],
            "Distancia_AU": [distancia_simulada_au],
            "Distancia_Lunar": [distancia_simulada_ld],
        })

        prob_sim, pred_sim = predecir_riesgo_asteroide(
            modelo_operativo,
            datos_simulacion,
            variables_operativas,
            umbral_decision=umbral_corte
        )
        prob_porcentaje = float(prob_sim[0])
        es_peligro_detectado = bool(pred_sim[0])

        impacto_fisico = calcular_energia_cinetica(diametro_kilometros, velocidad_simulada_kms)
        diametro_crater = estimar_diametro_crater_km(diametro_kilometros, velocidad_simulada_kms)

        if prob_porcentaje >= (umbral_corte * 100):
            st.error(f"### 🚨 ALERTA: Asteroide Potencialmente Peligroso (PHA)\n**Probabilidad de Amenaza: {prob_porcentaje:.1f}%**")
        else:
            st.success(f"### 🛡️ TRAYECTORIA SEGURA (No Clasificado como PHA)\n**Probabilidad de Amenaza: {prob_porcentaje:.1f}%**")

        st.progress(prob_porcentaje / 100.0)

        col_met1, col_met2, col_met3 = st.columns(3)
        col_met1.metric("Masa Estimada", f"{impacto_fisico['masa_toneladas'] / 1e6:,.2f} M Ton")
        col_met2.metric("Energía Cinética", f"{impacto_fisico['megatones']:,.1f} MT", "Megatones TNT")
        col_met3.metric("Cráter Estimado", f"{diametro_crater:.2f} km" if diametro_crater > 0 else "Sin cráter", "En roca sólida")

        info_comparativa = impacto_fisico["comparativa"]
        st.info(f"**Escala de Consecuencia:** {info_comparativa['nivel']}\n\n{info_comparativa['descripcion']}")

# =============================================================================
# PESTAÑA 3: DEFENSA PLANETARIA (MISIÓN DART)
# =============================================================================
with pestana_dart:
    st.subheader("🛡️ Simulación de Deflexión Cinética (Misión DART de la NASA)")
    st.markdown(
        """
        En septiembre de 2022, la misión **DART (Double Asteroid Redirection Test)** de la NASA demostró
        con éxito que colisionar una nave espacial contra un asteroide puede alterar su trayectoria orbital mediante
        transferencia de momento cinético y retropropulsión por eyección de escombros (*factor $\\beta$*).
        
        Usa este módulo para calcular si sería viable desviar el asteroide simulado para evitar una colisión con la Tierra.
        """
    )

    col_dart_in, col_dart_out = st.columns([1, 1])

    with col_dart_in:
        st.markdown("#### 🚀 Parámetros de la Misión de Deflexión")
        dias_preaviso = st.slider(
            "Tiempo de preaviso antes de la colisión proyectada:",
            min_value=30,
            max_value=3650,
            value=1825,
            step=30,
            help="1825 días equivale a 5 años de anticipación.",
        )
        anos_preaviso = round(dias_preaviso / 365.25, 1)
        st.caption(f"Equivale aproximadamente a **{anos_preaviso} años** de aviso.")

        masa_nave_proyectil = st.number_input(
            "Masa de la nave impactadora (kg):",
            min_value=100.0,
            max_value=5000.0,
            value=610.0,
            step=50.0,
            help="La nave DART tenía una masa de ~610 kg al impactar.",
        )

        velocidad_impacto_nave = st.number_input(
            "Velocidad de colisión de la nave (km/s):",
            min_value=2.0,
            max_value=20.0,
            value=6.1,
            step=0.1,
            help="DART colisionó contra Dimorphos a 6.1 km/s (22.000 km/h).",
        )

        factor_beta_eyecta = st.slider(
            "Factor de mejora de momento β (Eyección de roca):",
            min_value=1.0,
            max_value=4.0,
            value=2.4,
            step=0.1,
            help="β = 1 representa choque perfectamente inelástico sin eyección. El equipo DART midió β ≈ 2.2 a 2.4.",
        )

    with col_dart_out:
        st.markdown("#### 🎯 Resultado del Análisis de Deflexión")
        masa_asteroide_kg = impacto_fisico["masa_kg"]
        analisis_dart = calcular_deflexion_dart(
            masa_asteroide_kg=masa_asteroide_kg,
            dias_aviso=dias_preaviso,
            masa_nave_kg=masa_nave_proyectil,
            velocidad_nave_kms=velocidad_impacto_nave,
            factor_beta=factor_beta_eyecta
        )

        c_d1, c_d2 = st.columns(2)
        c_d1.metric("Variación de Velocidad (Δv)", f"{analisis_dart['delta_v_unitaria_mms']:.4f} mm/s", "Impartida por 1 nave")
        c_d2.metric("Desvío Lineal Logrado", f"{analisis_dart['desvio_un_impactador_km']:,.0f} km", f"Objetivo: {analisis_dart['desvio_objetivo_km']:,.0f} km")

        naves = analisis_dart["naves_necesarias"]

        if analisis_dart["es_viable"]:
            st.success(
                f"### ✅ MISIÓN DE DEFLEXIÓN VIABLE\n"
                f"Con **{anos_preaviso} años** de preaviso, se requeriría el lanzamiento de **{naves} nave(s) impactadora(s)** "
                f"tipo DART para desviar el asteroide más de {analisis_dart['desvio_objetivo_km']:,.0f} km fuera de la trayectoria de colisión."
            )
        else:
            st.warning(
                f"### ⚠️ DEFLEXIÓN CINÉTICA COMPLEJA\n"
                f"Para este tamaño ({diametro_metros} m) y tiempo de aviso, se requerirían **{naves} naves impactadoras coordinadas**, "
                f"o bien un tiempo mínimo de preaviso de **{analisis_dart['anos_minimos_1nave']} años** para lograr el desvío con una sola nave."
            )

        st.markdown(
            f"""
            > **Mecánica Orbital del Desvío:**  
            > Un pequeño cambio de velocidad de apenas **{analisis_dart['delta_v_unitaria_mms']:.3f} mm/s** aplicado años antes de la aproximación
            > se traduce a lo largo de millones de kilómetros de órbita en un desvío lineal de **{analisis_dart['desvio_un_impactador_km']:,.0f} kilómetros**,
            > esquivando la Tierra con amplio margen de seguridad.
            """
        )

# =============================================================================
# PESTAÑA 4: VISUALIZADOR 3D ORBITAL
# =============================================================================
with pestana_3d:
    st.subheader("🌐 Explorador Tridimensional de Aproximación Orbital")
    st.markdown(
        """
        Visualización en 3D interactiva del espacio cercano a la Tierra.
        Representa la **Tierra** en el origen $(0, 0, 0)$, la **órbita de la Luna** ($1\\text{ LD} \\approx 384.400\\text{ km}$),
        el **límite crítico MOID de la NASA** ($19.5\\text{ LD} = 0.05\\text{ AU}$) y las posiciones relativas
        de los asteroides detectados en la telemetría viva.
        """
    )

    if not df_telemetria.empty:
        figura_3d = go.Figure()

        # 1. Esfera de la Tierra en el centro (0, 0, 0)
        u_tierra = np.linspace(0, 2 * np.pi, 25)
        v_tierra = np.linspace(0, np.pi, 25)
        radio_vis_tierra = 0.6  # Escala aumentada para visibilidad en el gráfico
        x_tierra = radio_vis_tierra * np.outer(np.cos(u_tierra), np.sin(v_tierra))
        y_tierra = radio_vis_tierra * np.outer(np.sin(u_tierra), np.sin(v_tierra))
        z_tierra = radio_vis_tierra * np.outer(np.ones(np.size(u_tierra)), np.cos(v_tierra))

        figura_3d.add_trace(
            go.Surface(
                x=x_tierra,
                y=y_tierra,
                z=z_tierra,
                colorscale=[[0, "#1d4ed8"], [1, "#60a5fa"]],
                showscale=False,
                name="Tierra (Origen)",
                hoverinfo="text",
                hovertext="Planeta Tierra (Centro de referencia)",
            )
        )

        # 2. Anillo de la Órbita Lunar (1 Distancia Lunar = 1.0 LD)
        angulos_luna = np.linspace(0, 2 * np.pi, 100)
        radio_lunar = 1.0
        x_luna = radio_lunar * np.cos(angulos_luna)
        y_luna = radio_lunar * np.sin(angulos_luna)
        z_luna = np.zeros_like(x_luna)

        figura_3d.add_trace(
            go.Scatter3d(
                x=x_luna,
                y=y_luna,
                z=z_luna,
                mode="lines",
                line=dict(color="#94a3b8", width=3, dash="dot"),
                name="Órbita Lunar (1 LD ≈ 384.400 km)",
                hoverinfo="name",
            )
        )

        # 3. Límite crítico de la NASA para PHA (0.05 AU ≈ 19.5 LD)
        radio_moid = 19.5
        x_moid = radio_moid * np.cos(angulos_luna)
        y_moid = radio_moid * np.sin(angulos_luna)
        z_moid = np.zeros_like(x_moid)

        figura_3d.add_trace(
            go.Scatter3d(
                x=x_moid,
                y=y_moid,
                z=z_moid,
                mode="lines",
                line=dict(color="#ef4444", width=2, dash="dash"),
                name="Límite Crítico PHA (0.05 AU ≈ 19.5 LD)",
                hoverinfo="name",
            )
        )

        # 4. Posiciones 3D de los asteroides del feed de la NASA
        num_ast = len(df_telemetria)
        np.random.seed(42)
        angulos_theta = np.linspace(0, 2 * np.pi * 1.5, num_ast)
        inclinaciones = np.random.uniform(-0.35, 0.35, num_ast)

        radios_ld = df_telemetria["Distancia_Lunar"].values
        x_ast = radios_ld * np.cos(angulos_theta) * np.cos(inclinaciones)
        y_ast = radios_ld * np.sin(angulos_theta) * np.cos(inclinaciones)
        z_ast = radios_ld * np.sin(inclinaciones)

        colores = ["#ef4444" if p else "#38bdf8" for p in df_telemetria["Es_Peligroso"].values]
        tamanos = np.clip(df_telemetria["Diametro_Max_km"].values * 12.0, 4, 25)

        textos_hover = [
            f"<b>Asteroide:</b> {nom}<br>"
            f"<b>Distancia:</b> {dist:.2f} LD ({dist * 384400:,.0f} km)<br>"
            f"<b>Velocidad:</b> {vel:.1f} km/s<br>"
            f"<b>Diámetro:</b> {diam:.3f} km<br>"
            f"<b>¿Es Peligroso?:</b> {'SÍ (PHA)' if pel else 'NO (Seguro)'}<br>"
            f"<b>Probabilidad ML:</b> {prob:.1f}%"
            for nom, dist, vel, diam, pel, prob in zip(
                df_telemetria["Nombre"],
                df_telemetria["Distancia_Lunar"],
                df_telemetria["Velocidad_kms"],
                df_telemetria["Diametro_Max_km"],
                df_telemetria["Es_Peligroso"],
                df_telemetria["Probabilidad_Riesgo_%"],
            )
        ]

        figura_3d.add_trace(
            go.Scatter3d(
                x=x_ast,
                y=y_ast,
                z=z_ast,
                mode="markers+text",
                marker=dict(
                    size=tamanos,
                    color=colores,
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1),
                ),
                text=df_telemetria["Nombre"],
                textposition="top center",
                textfont=dict(color="#cbd5e1", size=10),
                hoverinfo="text",
                hovertext=textos_hover,
                name="Asteroides Detectados (Rojo = Peligroso, Azul = Seguro)",
            )
        )

        figura_3d.update_layout(
            template="plotly_dark",
            height=650,
            scene=dict(
                xaxis_title="Distancia X (LD)",
                yaxis_title="Distancia Y (LD)",
                zaxis_title="Inclinación Z (LD)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            ),
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=1),
        )

        st.plotly_chart(figura_3d, use_container_width=True)
    else:
        st.info("Carga la telemetría para visualizar las órbitas tridimensionales.")

# =============================================================================
# PESTAÑA 5: MODELO ML & RENDIMIENTO
# =============================================================================
with pestana_ml:
    st.subheader("🧠 Arquitectura del Modelo de Machine Learning y Validación")
    st.markdown(
        """
        Para clasificar con rigor asteroides potencialmente peligrosos, el clasificador se basa en un
        **Random Forest** con balanceo de pesos (`class_weight='balanced'`) y evaluación objetiva
        sobre un **conjunto de prueba independiente retenido (Holdout Test Set 20% estratificado)**.
        """
    )

    if metricas_operativas:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Exactitud (Accuracy)", f"{metricas_operativas.get('exactitud', 0):.1%}")
        m2.metric(
            "Sensibilidad (Recall)",
            f"{metricas_operativas.get('sensibilidad', 0):.1%}",
            help="Porcentaje de asteroides peligrosos reales detectados por el modelo.",
        )
        m3.metric("Precisión (Precision)", f"{metricas_operativas.get('precision', 0):.1%}")
        m4.metric("Puntuación F1", f"{metricas_operativas.get('f1', 0):.3f}")
        m5.metric("Área bajo la curva (ROC-AUC)", f"{metricas_operativas.get('area_roc', 0):.3f}")

        st.divider()

        col_ml_izq, col_ml_der = st.columns(2)

        with col_ml_izq:
            st.subheader("🌟 Importancia de Variables (Feature Importance)")
            st.caption("Ponderación de qué factores orbitales determinan mayor predicción de riesgo.")
            df_importancias = metricas_operativas.get("importancia_variables", pd.DataFrame())
            if not df_importancias.empty:
                fig_importancias = px.bar(
                    df_importancias,
                    x="Importancia_%",
                    y="Caracteristica",
                    orientation="h",
                    text="Importancia_%",
                    color="Importancia_%",
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                )
                fig_importancias.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_importancias, use_container_width=True)

        with col_ml_der:
            st.subheader("📈 Curva ROC (Receiver Operating Characteristic)")
            st.caption("Capacidad de discriminación entre asteroides inocuos y peligrosos.")
            curva_roc = metricas_operativas.get("curva_roc", {})
            if curva_roc:
                fig_curva = go.Figure()
                fig_curva.add_trace(
                    go.Scatter(
                        x=curva_roc["tasa_falsos_positivos"],
                        y=curva_roc["tasa_verdaderos_positivos"],
                        mode="lines",
                        name=f"Random Forest (AUC = {metricas_operativas.get('area_roc', 0):.3f})",
                        line=dict(color="#4EA8DE", width=3),
                    )
                )
                fig_curva.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Línea base aleatoria (AUC = 0.50)",
                        line=dict(color="#888888", dash="dash"),
                    )
                )
                fig_curva.update_layout(
                    xaxis_title="Tasa de Falsos Positivos (1 - Especificidad)",
                    yaxis_title="Tasa de Verdaderos Positivos (Sensibilidad / Recall)",
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_curva, use_container_width=True)

        col_matriz1, col_matriz2 = st.columns([1, 1])
        with col_matriz1:
            st.subheader("🎯 Matriz de Confusión en Conjunto de Prueba")
            matriz = metricas_operativas.get("matriz_confusion")
            if matriz is not None:
                etiquetas_matriz = ["Seguro (Negativo)", "Peligroso (Positivo)"]
                fig_matriz = px.imshow(
                    matriz,
                    text_auto=True,
                    x=etiquetas_matriz,
                    y=etiquetas_matriz,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicción del Modelo", y="Etiqueta Real NASA"),
                    template="plotly_dark",
                )
                fig_matriz.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_matriz, use_container_width=True)

        with col_matriz2:
            st.markdown("#### 💡 Justificación Técnica de Negocio y Misión")
            st.markdown(
                f"""
                - **Muestras de entrenamiento:** {metricas_operativas.get('muestras_entrenamiento', 0)} asteroides.
                - **Muestras de prueba independiente:** {metricas_operativas.get('muestras_prueba', 0)} asteroides.
                - **Compensación de Desbalance:** Los asteroides seguros superan a los peligrosos por amplio margen. Si un modelo predijese siempre "Seguro", obtendría una exactitud engañosa de más del 85%, pero un **Recall del 0%** ante un impacto inminente.
                - **Optimización de Sensibilidad:** En este contexto, el coste de un Falso Negativo (amenaza real no detectada) es incalculable frente al coste de un Falso Positivo (falsa alarma telescópica). Por ello se prioriza maximizar el **Recall**.
                """
            )
    else:
        st.info("Inicializando métricas sobre el conjunto histórico...")

# =============================================================================
# PESTAÑA 6: METODOLOGÍA Y CRITERIOS NASA
# =============================================================================
with pestana_metodologia:
    st.subheader("🔬 ¿Cómo define la NASA un Asteroide Potencialmente Peligroso (PHA)?")
    st.markdown(
        """
        El Centro de Estudios de Objetos Cercanos a la Tierra (**CNEOS**) del *Jet Propulsion Laboratory (JPL / NASA)*
        establece que un objeto cercano a la Tierra se cataloga como **Potencialmente Peligroso (PHA)** si satisface
        simultáneamente dos criterios astronómicos:
        """
    )

    col_crit1, col_crit2 = st.columns(2)
    with col_crit1:
        st.markdown(
            """
            ### 1️⃣ Distancia Mínima de Intersección Orbital (MOID)
            $$\\text{MOID} \\le 0.05 \\text{ AU} \\approx 7.480.000 \\text{ km} \\approx 19.5 \\text{ LD}$$
            - Define la proximidad orbital matemática entre la órbita de la Tierra y la del asteroide.
            - Indica si existe riesgo de colisión en una escala de siglos debido a perturbaciones gravitatorias de Júpiter o la Tierra.
            """
        )
    with col_crit2:
        st.markdown(
            """
            ### 2️⃣ Magnitud Absoluta y Diámetro Mínimo
            $$H \\le 22.0 \\iff \\text{Diámetro} \\ge 140 \\text{ metros}$$
            - La magnitud $H$ cuantifica el brillo intrínseco del asteroide observado a 1 AU del Sol y la Tierra.
            - Un cuerpo de más de 140 metros es capaz de atravesar la atmósfera terrestre y causar devastación a escala continental o tsunamis transoceánicos.
            """
        )

    st.divider()

    st.markdown(
        """
        ### 🤖 ¿Por qué aplicar Machine Learning en Detección Orbital?
        1. **Manejo de Incertidumbre Observacional:** En los primeros días de descubrimiento de un asteroide, el arco de observación es corto y los parámetros orbitales precisos presentan error. El modelo infiere la probabilidad de amenaza relacionando velocidad, magnitud $H$ y distancias relativas.
        2. **Automatización de Alto Rendimiento:** Nuevos telescopios de sondeo rápido como el Observatorio Vera C. Rubin (LSST) procesan millones de alertas por noche. Los clasificadores automatizados filtran los objetos críticos en milisegundos.
        3. **Sistema Continuo de Alerta:** Proporciona un índice continuo de amenaza que permite priorizar horas de observación telescópica en observatorios terrestres e infrarrojos espaciales.
        """
    )
