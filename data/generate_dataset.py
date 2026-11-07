"""Script para generar el conjunto de datos de referencia de asteroides cercanos a la Tierra (NEOs).
Descarga objetos desde el punto de conexión Browse de la API NASA NeoWs,
normaliza las variables y guarda en data/sample_asteroids.csv para entrenamiento y respaldo.
"""

import time
from pathlib import Path
import pandas as pd
import requests

DIRECTORIO_DATOS = Path(__file__).resolve().parent
ARCHIVO_SALIDA_CSV = DIRECTORIO_DATOS / "sample_asteroids.csv"


def construir_registro_asteroide(objeto_json: dict) -> dict:
    """Extrae y normaliza las propiedades físicas y orbitales del JSON de la NASA."""
    try:
        nombre = objeto_json.get("name", "Desconocido")
        identificador = objeto_json.get("id", "")
        magnitud_h = float(objeto_json.get("absolute_magnitude_h", 0.0))
        es_peligroso = bool(objeto_json.get("is_potentially_hazardous_asteroid", False))
        es_sentry = bool(objeto_json.get("is_sentry_object", False))
        enlace_jpl = objeto_json.get("nasa_jpl_url", "")

        diametros = objeto_json.get("estimated_diameter", {}).get("kilometers", {})
        d_min = float(diametros.get("estimated_diameter_min", 0.0))
        d_max = float(diametros.get("estimated_diameter_max", 0.0))
        d_medio = (d_min + d_max) / 2.0 if (d_min and d_max) else d_max

        aproximaciones = objeto_json.get("close_approach_data", [])
        if aproximaciones:
            aprox = aproximaciones[0]
            vel_kmh = float(aprox.get("relative_velocity", {}).get("kilometers_per_hour", 0.0))
            vel_kms = float(aprox.get("relative_velocity", {}).get("kilometers_per_second", 0.0))
            dist_km = float(aprox.get("miss_distance", {}).get("kilometers", 0.0))
            dist_au = float(aprox.get("miss_distance", {}).get("astronomical", 0.0))
            dist_lunar = float(aprox.get("miss_distance", {}).get("lunar", 0.0))
            fecha_aprox = aprox.get("close_approach_date", "")
        else:
            vel_kmh = 45000.0
            vel_kms = 12.5
            dist_km = 15000000.0
            dist_au = 0.1
            dist_lunar = 39.0
            fecha_aprox = "N/D"

        datos_orbitales = objeto_json.get("orbital_data", {})
        moid = float(datos_orbitales.get("minimum_orbit_intersection", 0.0)) if datos_orbitales.get("minimum_orbit_intersection") else None

        return {
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
            "MOID_AU": round(moid, 5) if moid is not None else None,
            "Es_Peligroso": es_peligroso,
            "Es_Sentry": es_sentry,
            "JPL_URL": enlace_jpl,
        }
    except Exception as error:
        print(f"Error procesando objeto: {error}")
        return None


def descargar_desde_browse(paginas_maximas: int = 15, clave_api: str = "DEMO_KEY") -> list:
    """Descarga lotes de asteroides desde el punto de conexión Browse de NASA NeoWs."""
    registros = []
    print(f"Descargando de la API Browse de la NASA (hasta {paginas_maximas} páginas)...")
    for pagina in range(paginas_maximas):
        url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?page={pagina}&size=20&api_key={clave_api}"
        try:
            respuesta = requests.get(url, timeout=10)
            if respuesta.status_code == 429:
                print(f"Límite de cuota alcanzado en la página {pagina}. Deteniendo descarga.")
                break
            respuesta.raise_for_status()
            datos = respuesta.json()
            for obj in datos.get("near_earth_objects", []):
                registro = construir_registro_asteroide(obj)
                if registro:
                    registros.append(registro)
            time.sleep(0.3)
        except Exception as error:
            print(f"Error en la página {pagina}: {error}")
            break
    return registros


def ejecutar():
    registros = descargar_desde_browse(paginas_maximas=15)
    df = pd.DataFrame(registros)
    print(f"Total de registros descargados: {len(df)}")
    if not df.empty:
        print(f"Recuento de asteroides peligrosos: {df['Es_Peligroso'].sum()} / {len(df)}")
        df = df.drop_duplicates(subset=["ID"])
        df.to_csv(ARCHIVO_SALIDA_CSV, index=False)
        print(f"Dataset guardado en {ARCHIVO_SALIDA_CSV}")


if __name__ == "__main__":
    ejecutar()
