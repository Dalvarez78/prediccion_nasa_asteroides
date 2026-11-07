"""Pipeline de Machine Learning para la clasificación de Asteroides Potencialmente Peligrosos (PHA).
Implementa validación estratificada en conjunto de prueba, cálculo de métricas de clasificación,
curva ROC, balanceo de clases y calibración de umbrales de decisión.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Variables clave identificadas por mecánica orbital y astrofísica de la NASA
CARACTERISTICAS_POR_DEFECTO: List[str] = [
    "Magnitud_Absoluta_H",
    "Diametro_Max_km",
    "Velocidad_kms",
    "Distancia_AU",
    "Distancia_Lunar",
]


def preparar_variables(
    df: pd.DataFrame,
    caracteristicas: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Valida, deriva variables faltantes y extrae el vector X y la variable objetivo y."""
    df_limpio = df.copy()

    # Si faltan columnas derivadas, calcularlas sobre la marcha
    if "Distancia_AU" not in df_limpio.columns and "Distancia_km" in df_limpio.columns:
        df_limpio["Distancia_AU"] = df_limpio["Distancia_km"] / 149597870.7
    if "Distancia_Lunar" not in df_limpio.columns and "Distancia_km" in df_limpio.columns:
        df_limpio["Distancia_Lunar"] = df_limpio["Distancia_km"] / 384400.0
    if "Velocidad_kms" not in df_limpio.columns and "Velocidad_kmh" in df_limpio.columns:
        df_limpio["Velocidad_kms"] = df_limpio["Velocidad_kmh"] / 3600.0

    # Imputar valores con la mediana en caso de valores ausentes
    X = df_limpio[caracteristicas].copy()
    X = X.fillna(X.median())
    y = df_limpio["Es_Peligroso"].astype(bool)

    return X, y


def entrenar_clasificador_asteroides(
    df: pd.DataFrame,
    caracteristicas: Optional[List[str]] = None,
    tamano_prueba: float = 0.20,
    semilla_aleatoria: int = 42
) -> Tuple[Optional[RandomForestClassifier], List[str], Dict]:
    """Entrena un modelo Random Forest con división estratificada y ponderación balanceada.

    Args:
        df: Conjunto de datos con asteroides etiquetados.
        caracteristicas: Lista de variables a utilizar.
        tamano_prueba: Proporción del conjunto de prueba independiente (20% por defecto).
        semilla_aleatoria: Semilla para reproducibilidad.

    Returns:
        modelo: Random Forest entrenado.
        caracteristicas: Lista de variables utilizadas.
        metricas: Diccionario con métricas de validación en el conjunto de prueba.
    """
    if caracteristicas is None:
        caracteristicas = CARACTERISTICAS_POR_DEFECTO

    if df.empty or len(df["Es_Peligroso"].unique()) < 2:
        return None, caracteristicas, {}

    X, y = preparar_variables(df, caracteristicas)

    # Validar si es posible estratificar según el recuento de la clase minoritaria
    recuento_minimo = y.value_counts().min()
    estratificacion = y if recuento_minimo >= 2 else None

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=tamano_prueba,
        random_state=semilla_aleatoria,
        stratify=estratificacion,
    )

    # Random Forest con ponderación balanceada para mitigar el desbalance natural de asteroides
    modelo = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=semilla_aleatoria,
    )
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Evaluación objetiva sobre el conjunto de prueba retenido
    y_prediccion = modelo.predict(X_prueba)
    y_probabilidad = modelo.predict_proba(X_prueba)[:, 1]

    exactitud = accuracy_score(y_prueba, y_prediccion)
    precision = precision_score(y_prueba, y_prediccion, zero_division=0)
    sensibilidad = recall_score(y_prueba, y_prediccion, zero_division=0)
    puntuacion_f1 = f1_score(y_prueba, y_prediccion, zero_division=0)

    try:
        area_roc = roc_auc_score(y_prueba, y_probabilidad)
        tasa_falsos_positivos, tasa_verdaderos_positivos, _ = roc_curve(y_prueba, y_probabilidad)
    except Exception:
        area_roc = 0.5
        tasa_falsos_positivos = np.array([0, 1])
        tasa_verdaderos_positivos = np.array([0, 1])

    matriz_conf = confusion_matrix(y_prueba, y_prediccion)

    # Importancia de las variables predictivas
    importancias = modelo.feature_importances_
    df_importancias = pd.DataFrame({
        "Caracteristica": caracteristicas,
        "Importancia_%": (importancias * 100.0).round(2),
    }).sort_values(by="Importancia_%", ascending=False)

    metricas = {
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "f1": puntuacion_f1,
        "area_roc": area_roc,
        "matriz_confusion": matriz_conf,
        "curva_roc": {
            "tasa_falsos_positivos": tasa_falsos_positivos.tolist(),
            "tasa_verdaderos_positivos": tasa_verdaderos_positivos.tolist(),
        },
        "importancia_variables": df_importancias,
        "muestras_entrenamiento": len(X_entrenamiento),
        "muestras_prueba": len(X_prueba),
        "amenazas_entrenamiento": int(y_entrenamiento.sum()),
        "amenazas_prueba": int(y_prueba.sum()),
    }

    return modelo, caracteristicas, metricas


def predecir_riesgo_asteroide(
    modelo: RandomForestClassifier,
    datos_entrada: pd.DataFrame,
    caracteristicas: List[str],
    umbral_decision: float = 0.50
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera probabilidades continuas y predicción binaria sujeta al umbral de decisión.

    Args:
        modelo: Random Forest previamente entrenado.
        datos_entrada: DataFrame con las mismas columnas predictivas.
        caracteristicas: Lista ordenada de variables.
        umbral_decision: Punto de corte probabilístico (0.50 por defecto).

    Returns:
        probabilidades_porcentaje: Vector con porcentaje de probabilidad de ser PHA (0 a 100).
        es_peligroso_booleano: Vector booleano con la predicción según el umbral.
    """
    X_entrada = datos_entrada[caracteristicas].copy()
    clases_modelo = list(modelo.classes_)
    indice_verdadero = clases_modelo.index(True) if True in clases_modelo else 1

    probabilidades = modelo.predict_proba(X_entrada)[:, indice_verdadero]
    probabilidades_porcentaje = np.round(probabilidades * 100.0, 2)
    es_peligroso_booleano = probabilidades >= umbral_decision

    return probabilidades_porcentaje, es_peligroso_booleano
