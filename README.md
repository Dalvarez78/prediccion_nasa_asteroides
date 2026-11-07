# ☄️ Monitor y Predicción de Asteroides Peligrosos (NASA NeoWs)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NASA API](https://img.shields.io/badge/NASA%20API-NeoWs%20REST-0B3D91.svg?logo=nasa&logoColor=white)](https://api.nasa.gov/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualizaci%C3%B3n%203D-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green.svg)](https://opensource.org/licenses/MIT)

Proyecto integral de **Ciencia de Datos, Astrofísica y Machine Learning** que extrae telemetría orbital en tiempo real mediante la API oficial **NASA NeoWs (Near Earth Object Web Service)**, clasifica el nivel de riesgo de objetos cercanos a la Tierra (NEOs) mediante un modelo de aprendizaje automático probabilístico, calcula la balística física de impacto (energía en megatones de TNT y dimensiones del cráter), simula misiones de **deflexión cinética (Misión DART)** y ofrece una **exploración tridimensional interactiva** del espacio circumterrestre.

---

## 🚀 Funcionalidades Principales

- **🛰️ Telemetría Espacial en Vivo:** Conexión con la API REST de NASA NeoWs para monitorizar aproximaciones recientes (ventanas de 1 a 7 días). Incluye control de cuotas HTTP 429 y respaldo automático con datos históricos locales.
- **🧠 Machine Learning con Validación Rigurosa:** Modelo `RandomForestClassifier` optimizado para el desbalance natural de clases (`class_weight='balanced'`), evaluado sobre un **conjunto de prueba retenido independiente (Test Set 20% estratificado)**.
- **📊 Diagnóstico y Métricas de Rendimiento:**
  - Matriz de confusión interactiva.
  - Curva ROC y cálculo de área bajo la curva (**ROC-AUC ≈ 0.92**).
  - Puntuaciones de Sensibilidad (Recall ≈ 92%), Precisión, F1-Score y Exactitud (~90%).
  - Importancia de Variables (*Feature Importance*), donde la **Magnitud Absoluta ($H$)** y el **Diámetro** concentran más del 70% del poder predictivo.
- **🎯 Simulador Físico de Impacto Balístico:**
  - Cálculo de masa a partir de densidad condrítica ($2.500\text{ kg/m}^3$).
  - Energía cinética liberada ($E_k = \frac{1}{2}mv^2$) convertida a **Megatones de TNT** ($1\text{ MT} = 4.184 \times 10^{15}\text{ J}$).
  - Estimación del diámetro de cráter transitorio (fórmulas empíricas de Schmidt-Holsapple).
  - Comparativa con eventos históricos (Chelyabinsk, Tunguska, Bomba Zar, Extinción K-Pg de Chicxulub).
- **🛡️ Módulo de Defensa Planetaria (Misión DART):**
  - Simula la deflexión de la órbita del asteroide mediante impacto cinético frontal (con parámetros reales de la misión DART de la NASA, $\beta = 2.4$, $v = 6.1\text{ km/s}$).
  - Calcula la variación de velocidad necesaria ($\Delta v$), el desvío acumulado en kilómetros y el número de naves o años de anticipación requeridos para esquivar la Tierra.
- **🌐 Visualizador 3D Orbital e Interactivo:**
  - Gráfico tridimensional en Plotly centrado en la Tierra $(0, 0, 0)$, con el anillo de la órbita lunar ($1\text{ LD} \approx 384.400\text{ km}$), el umbral crítico MOID de la NASA ($19.5\text{ LD} = 0.05\text{ AU}$) y las posiciones de los asteroides del feed de la NASA.
- **🌌 Interfaz Moderna en Tema Cósmico:** Desarrollada con **Streamlit** y **Plotly**, con soporte para descarga de telemetría en CSV y enlaces directos a la base de datos orbital **JPL Small-Body Database**.

---

## 🔬 Metodología Científica (Criterios Oficiales NASA JPL)

El Centro de Estudios de Objetos Cercanos a la Tierra (**CNEOS**) de NASA/JPL cataloga formalmente a un asteroide como **Potencialmente Peligroso (PHA)** si cumple dos criterios simultáneos:

1. **Distancia Mínima de Intersección Orbital (MOID):**
   $$\text{MOID} \le 0.05\text{ AU} \approx 7.480.000\text{ km} \approx 19.5\text{ Distancias Lunares (LD)}$$
2. **Magnitud Absoluta y Diámetro Mínimo:**
   $$H \le 22.0 \iff \text{Diámetro estimado } D \ge 140\text{ metros}$$

### ¿Por qué Machine Learning en Defensa Planetaria?
- **Incertidumbre en observaciones tempranas:** Con pocos días de seguimiento telescópico inicial, los parámetros orbitales precisos tienen margen de error. El modelo aprende correlaciones no lineales entre velocidad, luminosidad ($H$) y distancia de paso.
- **Optimización de Sensibilidad (Recall):** En defensa planetaria, clasificar erróneamente un asteroide peligroso como seguro (*falso negativo*) acarrea consecuencias devastadoras. El clasificador permite calibrar dinámicamente el umbral de decisión para maximizar el Recall.

---

## 📁 Estructura del Código

```plaintext
prediccion_nasa_asteroides/
│
├── .streamlit/
│   └── config.toml             # Configuración del tema cósmico oscuro (#0B0F19)
│
├── data/
│   ├── sample_asteroids.csv     # Dataset curado de referencia de la NASA (~380 objetos)
│   ├── generate_dataset.py     # Script de extracción desde el punto de conexión Browse
│   └── enrich_dataset.py       # Script de balanceo y consistencia astrofísica
│
├── src/
│   ├── __init__.py             # Inicializador del paquete
│   ├── api.py                  # Cliente para la API de NASA con gestión de errores y respaldo
│   ├── ml_model.py             # Pipeline Random Forest, Holdout 80/20 y métricas
│   └── impact_physics.py       # Ecuaciones físicas de masa, energía, cráteres y misión DART
│
├── app.py                      # Aplicación interactiva de Streamlit con 6 módulos
├── requirements.txt            # Dependencias del proyecto
├── .gitignore                  # Exclusiones de Git
└── README.md                   # Documentación técnica del proyecto
```

---

## 🛠️ Instalación y Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/Dalvarez78/prediccion_nasa_asteroides.git
cd prediccion_nasa_asteroides
```

### 2. Crear y activar entorno virtual
```bash
# En Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# En Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Lanzar la aplicación
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## 🌐 Despliegue en Streamlit Community Cloud (Gratuito)

La aplicación está lista para publicarse en la nube en 2 minutos:

1. Inicia sesión en [share.streamlit.io](https://share.streamlit.io/) con tu cuenta de GitHub.
2. Haz clic en **"New app"**.
3. Selecciona el repositorio `Dalvarez78/prediccion_nasa_asteroides`, rama `main` y archivo principal `app.py`.
4. *(Opcional)* En **Advanced Settings > Secrets**, puedes configurar tu clave de la NASA:
   ```toml
   NASA_API_KEY = "tu_clave_de_la_nasa"
   ```
5. Haz clic en **"Deploy"**.

---

## 📈 Métricas de Rendimiento del Modelo de ML

Evaluado sobre el conjunto de prueba retenido (*stratified holdout test set*, 20%):

| Métrica | Valor Obtenido | Interpretación Técnica |
| :--- | :---: | :--- |
| **Exactitud (Accuracy)** | **~90.0%** | Proporción global de clasificaciones correctas |
| **Sensibilidad (Recall)** | **~92.0%** | Minimiza los falsos negativos ante asteroides amenazantes |
| **Precisión** | **~85.0%** | Confiabilidad de las alertas emitidas |
| **ROC-AUC** | **~0.92** | Sobresaliente capacidad de discriminación probabilística |

Las variables con mayor peso predictivo son **Magnitud Absoluta ($H$)** (~41%) y **Diámetro Máximo** (~32%), coincidiendo de forma exacta con la física orbital de la NASA.

---

## 👤 Autor

**David Álvarez**
- GitHub: [@Dalvarez78](https://github.com/Dalvarez78)
- Proyecto desarrollado para portafolio profesional en **Ciencia de Datos & Machine Learning**.
