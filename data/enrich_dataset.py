"""Enriquece sample_asteroids.csv a ~380 registros combinando objetos reales de la NASA
y distribuciones consistentes con la astrofísica y mecánica orbital de asteroides cercanos a la Tierra (NEOs).
"""

from pathlib import Path
import numpy as np
import pandas as pd

RUTA_CSV = Path(__file__).resolve().parent / "sample_asteroids.csv"
df_real = pd.read_csv(RUTA_CSV)

np.random.seed(42)
num_adicionales = 300

nombres = [
    f"({np.random.randint(1995, 2026)} {chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{np.random.randint(1, 99)})"
    for _ in range(num_adicionales)
]
identificadores = [str(3000000 + i) for i in range(num_adicionales)]

# Magnitud absoluta H: habitualmente entre 14.0 (masivo y raro) y 29.0 (pequeño y frecuente)
magnitud_h = np.random.normal(loc=23.5, scale=2.8, size=num_adicionales).clip(14.0, 29.5)

# Albedo p (reflectividad de superficie) entre 0.05 y 0.25
albedo = np.random.uniform(0.05, 0.25, size=num_adicionales)
# Ecuación estándar IAU: Diámetro (km) = (1329 / sqrt(p)) * 10^(-0.2 * H)
diametro_estimado = (1329.0 / np.sqrt(albedo)) * (10.0 ** (-0.2 * magnitud_h))
diametro_min = diametro_estimado * np.random.uniform(0.70, 0.85, size=num_adicionales)
diametro_max = diametro_estimado * np.random.uniform(1.15, 1.45, size=num_adicionales)

# Velocidades relativas de sobrevuelo: típicamente entre 5 y 40 km/s
velocidad_kms = np.random.gamma(shape=5.0, scale=3.0, size=num_adicionales).clip(4.0, 42.0)
velocidad_kmh = velocidad_kms * 3600.0

# MOID (Distancia Mínima de Intersección Orbital): rango habitual de 0.001 a 0.50 AU
moid_au = np.random.exponential(scale=0.08, size=num_adicionales).clip(0.0005, 0.50)

# Distancia de paso durante el sobrevuelo registrado
distancia_au = np.random.uniform(0.01, 0.35, size=num_adicionales)
distancia_km = distancia_au * 149597870.7
distancia_lunar = distancia_km / 384400.0

# Definición oficial de la NASA: Es PHA si MOID <= 0.05 AU Y Magnitud H <= 22.0 (diámetro >= 140 m)
es_peligroso = (moid_au <= 0.05) & (magnitud_h <= 22.0)

# Objetos monitorizados por el sistema automatizado Sentry
es_sentry = es_peligroso & (np.random.rand(num_adicionales) < 0.15)

df_adicional = pd.DataFrame({
    "ID": identificadores,
    "Nombre": nombres,
    "Fecha_Aproximacion": "2026-11-01",
    "Magnitud_Absoluta_H": np.round(magnitud_h, 2),
    "Diametro_Min_km": np.round(diametro_min, 4),
    "Diametro_Max_km": np.round(diametro_max, 4),
    "Diametro_Medio_km": np.round(diametro_estimado, 4),
    "Velocidad_kmh": np.round(velocidad_kmh, 2),
    "Velocidad_kms": np.round(velocidad_kms, 2),
    "Distancia_km": np.round(distancia_km, 2),
    "Distancia_AU": np.round(distancia_au, 5),
    "Distancia_Lunar": np.round(distancia_lunar, 2),
    "MOID_AU": np.round(moid_au, 5),
    "Es_Peligroso": es_peligroso,
    "Es_Sentry": es_sentry,
    "JPL_URL": [f"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={nid}" for nid in identificadores],
})

df_completo = pd.concat([df_real, df_adicional], ignore_index=True)
df_completo = df_completo.drop_duplicates(subset=["ID"])
df_completo.to_csv(RUTA_CSV, index=False)
print(f"Conjunto de datos enriquecido con éxito a {len(df_completo)} asteroides.")
print(f"Asteroides peligrosos: {df_completo['Es_Peligroso'].sum()} ({df_completo['Es_Peligroso'].mean():.1%})")
