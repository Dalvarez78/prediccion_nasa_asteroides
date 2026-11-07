"""Cliente y extractor de telemetría de la API REST NASA NeoWs (Near Earth Object Web Service).
Proporciona extracción de telemetría de asteroides cercanos a la Tierra,
gestión de cuotas (HTTP 429) y respaldo automático con datos históricos locales.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import requests
import pandas as pd

from src.impact_physics import calcular_energia_cinetica

DIRECTORIO_DATOS = Path(__file__).resolve().parent.parent / "data"
ARCHIVO_RESPALDO_CSV = DIRECTORIO_DATOS / "sample_asteroids.csv"


def cargar_datos_respaldo() -> pd.DataFrame:
    """Carga el conjunto de datos de referencia local cuando la API no responde o agota su cuota."""
    if ARCHIVO_RESPALDO_CSV.exists():
        df = pd.read_csv(ARCHIVO_RESPALDO_CSV)
        if "Energia_MT" not in df.columns:
            df["Energia_MT"] = df.apply(
                lambda fila: calcular_energia_cinetica(fila["Diametro_Medio_km"], fila["Velocidad_kms"])["megatones"],
                axis=1
            )
        return df
    return pd.DataFrame()


def obtener_telemetria_nasa(
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    clave_api: str = "DEMO_KEY",
    tiempo_espera: int = 12
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Extrae telemetría reciente de objetos cercanos a la Tierra desde la API de la NASA.

    Args:
        fecha_inicio: Fecha inicial (formato AAAA-MM-DD). Por defecto: hoy - 7 días.
        fecha_fin: Fecha final (formato AAAA-MM-DD). Por defecto: hoy.
        clave_api: Clave de la API de NASA (usa DEMO_KEY por defecto).
        tiempo_espera: Segundos máximos antes de considerar tiempo agotado.

    Returns:
        Tupla con (DataFrame de asteroides procesados, mensaje de aviso o None).
    """
    hoy = datetime.today()
    if not fecha_fin:
        fecha_fin = hoy.strftime("%Y-%m-%d")
    if not fecha_inicio:
        fecha_inicio = (hoy - timedelta(days=7)).strftime("%Y-%m-%d")

    # Validación de rango: la API NeoWs feed restringe la consulta a un máximo de 7 días consecutivos
    inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
    fin_dt = datetime.strptime(fecha_fin, "%Y-%m-%d")
    if (fin_dt - inicio_dt).days > 7:
        inicio_dt = fin_dt - timedelta(days=7)
        fecha_inicio = inicio_dt.strftime("%Y-%m-%d")

    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={fecha_inicio}&end_date={fecha_fin}&api_key={clave_api}"

    try:
        respuesta = requests.get(url, timeout=tiempo_espera)

        if respuesta.status_code == 429:
            mensaje = "Aviso: Límite de peticiones de DEMO_KEY alcanzado (HTTP 429). Se ha cargado el conjunto histórico de respaldo de la NASA."
            df_respaldo = cargar_datos_respaldo()
            return df_respaldo, mensaje

        respuesta.raise_for_status()
        datos_json = respuesta.json()

        fechas_dict = datos_json.get("near_earth_objects", {})
        lista_asteroides = []

        for fecha_clave, objetos_fecha in fechas_dict.items():
            for obj in objetos_fecha:
                try:
                    identificador = obj.get("id", "")
                    nombre = obj.get("name", "Desconocido")
                    magnitud_h = float(obj.get("absolute_magnitude_h", 22.0))
                    es_peligroso = bool(obj.get("is_potentially_hazardous_asteroid", False))
                    es_sentry = bool(obj.get("is_sentry_object", False))
                    enlace_jpl = obj.get("nasa_jpl_url", "")

                    diametros = obj.get("estimated_diameter", {}).get("kilometers", {})
                    d_min = float(diametros.get("estimated_diameter_min", 0.0))
                    d_max = float(diametros.get("estimated_diameter_max", 0.0))
                    d_medio = (d_min + d_max) / 2.0 if (d_min and d_max) else d_max

                    aproximaciones = obj.get("close_approach_data", [])
                    if aproximaciones:
                        aprox = aproximaciones[0]
                        vel_kmh = float(aprox.get("relative_velocity", {}).get("kilometers_per_hour", 0.0))
                        vel_kms = float(aprox.get("relative_velocity", {}).get("kilometers_per_second", 0.0))
                        dist_km = float(aprox.get("miss_distance", {}).get("kilometers", 0.0))
                        dist_au = float(aprox.get("miss_distance", {}).get("astronomical", 0.0))
                        dist_lunar = float(aprox.get("miss_distance", {}).get("lunar", 0.0))
                        fecha_aprox = aprox.get("close_approach_date", fecha_clave)
                    else:
                        vel_kmh, vel_kms, dist_km, dist_au, dist_lunar = 0.0, 0.0, 0.0, 0.0, 0.0
                        fecha_aprox = fecha_clave

                    energia = calcular_energia_cinetica(d_medio, vel_kms)

                    lista_asteroides.append({
                        "ID": identificador,
                        "Nombre": nombre,
                        "Fecha_Aproximacion": fecha_aprox,
                        "Magnitud_Absoluta_H": round(magnitud_h, 2),
                        "Diametro_Min_km": round(d_min, 4),
                        "Diametro_Max_km": round(d_max, 4),
                        "Diametro_Medio_km": round(d_medio, 4),
                        "Velocidad_kmh": round(vel_kmh, 2),
                        "Velocidad_kms": round(vel_kms, 2),
                        "Distancia_km": round(dist_km, 2),
                        "Distancia_AU": round(dist_au, 5),
                        "Distancia_Lunar": round(dist_lunar, 2),
                        "Es_Peligroso": es_peligroso,
                        "Es_Sentry": es_sentry,
                        "Energia_MT": round(energia["megatones"], 3),
                        "JPL_URL": enlace_jpl,
                    })
                except (IndexError, KeyError, ValueError):
                    continue

        df_resultado = pd.DataFrame(lista_asteroides)
        if df_resultado.empty:
            return cargar_datos_respaldo(), "Aviso: No se registraron objetos en este periodo. Cargando datos de respaldo."

        return df_resultado, None

    except requests.exceptions.RequestException as error:
        mensaje_error = f"No se pudo conectar a la API de la NASA ({error}). Usando datos locales de respaldo."
        return cargar_datos_respaldo(), mensaje_error
