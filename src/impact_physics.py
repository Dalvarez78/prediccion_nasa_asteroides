"""Módulo de cálculos astrofísicos, balística de impacto y defensa planetaria.
Modela la energía cinética del impacto, equivalencia en megatones de TNT,
dimensiones teóricas de cráteres y simulación de deflexión cinética (Misión DART).
"""

import math

# Constantes físicas y astronómicas
JOULES_POR_MEGATON_TNT = 4.184e15      # 1 Megatón de TNT = 4.184 × 10^15 Joules
DENSIDAD_ESTANDAR_ROCA_KG_M3 = 2500.0  # Densidad típica de condrita rocosa (tipo S)
RADIO_TIERRA_KM = 6371.0               # Radio medio terrestre en kilómetros


def calcular_masa_kg(diametro_km: float, densidad_kg_m3: float = DENSIDAD_ESTANDAR_ROCA_KG_M3) -> float:
    """Calcula la masa en kilogramos asumiendo un cuerpo esférico homogéneo.

    Fórmula: V = (4/3) * pi * (R^3), Masa = Densidad * Volumen.
    """
    radio_metros = (diametro_km * 1000.0) / 2.0
    volumen_m3 = (4.0 / 3.0) * math.pi * (radio_metros ** 3)
    return densidad_kg_m3 * volumen_m3


def calcular_energia_cinetica(
    diametro_km: float,
    velocidad_kms: float,
    densidad_kg_m3: float = DENSIDAD_ESTANDAR_ROCA_KG_M3
) -> dict:
    """Calcula la energía cinética del asteroide en Joules y en Megatones de TNT.

    Fórmula: E = 0.5 * m * v^2.
    """
    masa = calcular_masa_kg(diametro_km, densidad_kg_m3)
    velocidad_ms = velocidad_kms * 1000.0
    joules = 0.5 * masa * (velocidad_ms ** 2)
    megatones = joules / JOULES_POR_MEGATON_TNT

    return {
        "masa_kg": masa,
        "masa_toneladas": masa / 1000.0,
        "joules": joules,
        "megatones": megatones,
        "comparativa": obtener_comparativa_impacto(megatones),
    }


def obtener_comparativa_impacto(energia_megatones: float) -> dict:
    """Devuelve la escala de daño histórico y balístico según los megatones liberados."""
    if energia_megatones < 0.01:
        return {
            "nivel": "🟢 Desintegración Atmosférica Inocua",
            "descripcion": "El objeto se fragmenta por fricción en la termosfera/estratosfera. Produce un bólido brillante sin provocar daños en tierra.",
            "evento_referencia": "Bólido atmosférico cotidiano",
        }
    elif energia_megatones < 1.0:
        return {
            "nivel": "🟡 Onda de Choque Aérea (Airburst Local)",
            "descripcion": "Explosión similar al meteorito de Chelyabinsk (Rusia, 2013, ~0.5 MT). Genera onda sónica capaz de quebrar cristales e inducir daños estructurales leves en decenas de kilómetros.",
            "evento_referencia": "Meteorito de Chelyabinsk (2013, ~0.5 MT)",
        }
    elif energia_megatones < 30.0:
        return {
            "nivel": "🟠 Devastación Regional Masiva (Evento Tipo Tunguska)",
            "descripcion": "Comparativo al impacto de Tunguska (Siberia, 1908, ~12 MT). Arrasaría por completo una metrópoli moderna o calcinaría miles de kilómetros cuadrados de bosque.",
            "evento_referencia": "Evento de Tunguska (1908, ~12 MT)",
        }
    elif energia_megatones < 500.0:
        return {
            "nivel": "🔴 Catástrofe Subcontinental",
            "descripcion": "Supera la potencia de 10 bombas termonucleares Zar (50 MT). Provocaría terremotos globales de magnitud superior a 7, incendios masivos y un cráter kilométrico permanente.",
            "evento_referencia": "Múltiples bombas termonucleares Zar",
        }
    else:
        return {
            "nivel": "💀 Extinción Biológica Global (Cataclismo Planetario)",
            "descripcion": "Magnitud similar al impacto de Chicxulub (hace 66 millones de años, ~100.000.000 MT). Conduciría a un invierno de impacto prolongado, colapso de la fotosíntesis y extinción masiva.",
            "evento_referencia": "Impacto de Chicxulub (Extinción Cretácico-Paleógeno)",
        }


def estimar_diametro_crater_km(diametro_km: float, velocidad_kms: float) -> float:
    """Estima el diámetro del cráter transitorio en corteza rocosa mediante leyes de escala."""
    energia = calcular_energia_cinetica(diametro_km, velocidad_kms)["megatones"]
    if energia < 0.05:
        return 0.0
    return max(0.05, 1.35 * (diametro_km ** 0.8) * ((velocidad_kms / 15.0) ** 0.4))


def estimar_magnitud_desde_diametro(diametro_km: float, albedo: float = 0.14) -> float:
    """Aproximación estándar de la Unión Astronómica Internacional (IAU).
    H = 5 * log10(1329 / (D * sqrt(albedo))).
    """
    if diametro_km <= 0:
        return 30.0
    valor = 1329.0 / (diametro_km * math.sqrt(albedo))
    return round(5.0 * math.log10(max(1e-4, valor)), 2)


def estimar_diametro_desde_magnitud(magnitud_h: float, albedo: float = 0.14) -> float:
    """Calcula el diámetro equivalente en kilómetros según su brillo intrínseco (Magnitud H)."""
    return round((1329.0 / math.sqrt(albedo)) * (10.0 ** (-0.2 * magnitud_h)), 4)


def calcular_deflexion_dart(
    masa_asteroide_kg: float,
    dias_aviso: float,
    masa_nave_kg: float = 600.0,
    velocidad_nave_kms: float = 6.1,
    factor_beta: float = 2.4
) -> dict:
    """Calcula la viabilidad de desvío orbital mediante un impactador cinético (Misión DART de la NASA).

    Args:
        masa_asteroide_kg: Masa del asteroide en kilogramos.
        dias_aviso: Días disponibles antes del impacto proyectado con la Tierra.
        masa_nave_kg: Masa de la nave impactadora (DART pesaba ~610 kg al impactar).
        velocidad_nave_kms: Velocidad relativa de colisión frontal (DART colisionó a 6.1 km/s).
        factor_beta: Factor de multiplicación de momento por eyección de escombros (beta medido ≈ 2.2 a 2.4).

    Returns:
        Diccionario con variación de velocidad lograda (delta_v), desvío lineal en km y naves requeridas.
    """
    tiempo_segundos = max(1.0, dias_aviso * 86400.0)
    velocidad_nave_ms = velocidad_nave_kms * 1000.0

    # Momento transferido = beta * m_nave * v_nave
    momento_total_kg_ms = factor_beta * masa_nave_kg * velocidad_nave_ms

    # Delta v impartido por un solo proyectil (m/s)
    delta_v_unitaria_ms = momento_total_kg_ms / max(1.0, masa_asteroide_kg)
    delta_v_unitaria_mms = delta_v_unitaria_ms * 1000.0  # en milímetros por segundo

    # Desvío lineal acumulado con 1 nave: Delta_x = Delta_v * t
    desvio_un_impactador_km = (delta_v_unitaria_ms * tiempo_segundos) / 1000.0

    # Para esquivar la Tierra se requiere superar el radio terrestre + margen de seguridad (1.5 * Radio)
    desvio_objetivo_km = RADIO_TIERRA_KM * 1.5
    delta_v_requerida_ms = (desvio_objetivo_km * 1000.0) / tiempo_segundos

    naves_necesarias = math.ceil(delta_v_requerida_ms / max(1e-12, delta_v_unitaria_ms))

    # Años mínimos con 1 sola nave
    tiempo_segundos_min_1nave = (desvio_objetivo_km * 1000.0) / max(1e-12, delta_v_unitaria_ms)
    anos_minimos_1nave = round(tiempo_segundos_min_1nave / (86400.0 * 365.25), 2)

    es_viable = naves_necesarias <= 5 and dias_aviso >= 365

    return {
        "delta_v_unitaria_mms": round(delta_v_unitaria_mms, 5),
        "desvio_un_impactador_km": round(desvio_un_impactador_km, 2),
        "desvio_objetivo_km": round(desvio_objetivo_km, 2),
        "naves_necesarias": int(naves_necesarias),
        "anos_minimos_1nave": anos_minimos_1nave,
        "es_viable": es_viable,
        "factor_beta": factor_beta,
    }
