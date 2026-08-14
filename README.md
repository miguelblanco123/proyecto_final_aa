# Detección de Clientes Anómalos en E-commerce

Proyecto final, Maestría en Ciencia de Datos, Aprendizaje Automático
Universidad Autónoma de Nuevo León

Profesor: Irving Daniel Estrada López

Alumnos:
* Andrea Linette Mezquita Gómez
* Carlos Enrique Cepeda Fuentes
* Miguel Alejandro Blanco Ríos

## Objetivo

Construir un perfil de compra a nivel cliente a partir de las transacciones de una tienda de e-commerce (dataset [Online Retail / Online Retail II](https://archive.ics.uci.edu/dataset/352/online+retail)) y usarlo para **detectar clientes con comportamiento anómalo**, comparando dos enfoques no supervisados:

* **DBSCAN**  clustering basado en densidad; los puntos que no logran asignarse a ningún cluster (`label == -1`) se interpretan como anomalías.
* **Isolation Forest**  modelo de ensamble diseñado específicamente para detección de anomalías.

Ambos modelos se optimizan con [Optuna](https://optuna.org/) maximizando el *silhouette score*, y se comparan en base a ese score, el porcentaje de clientes marcados como anómalos, y qué tan interpretable/accionable es el perfil resultante.

## Estructura del proyecto

```
data/
  raw/                        # Excel originales (online_retail.xlsx, online_retail_ii.xlsx)
  processed/                  # Datasets generados por el pipeline (ver abajo)
models/                       # Artefactos entrenados (modelo + hiperparámetros)
notebooks/
  00_load_data.ipynb          # Carga de datos crudos y construcción de features por cliente
  01_eda.ipynb                # Análisis exploratorio de datos
  02_preprocessing.ipynb      # Transformaciones, codificación y escalamiento
  03_entrenamiento.ipynb      # Optimización y entrenamiento de DBSCAN e Isolation Forest
  04_validacion.ipynb         # Evaluación y comparación de ambos modelos
  notebook.ipynb              # Versión monolítica con las mismas etapas, todo en un solo notebook
src/
  data.py                     # Equivalente en .py de 00_load_data.ipynb
  preprocessing.py            # Equivalente en .py de 02_preprocessing.ipynb
  train.py                    # Equivalente en .py de 03_entrenamiento.ipynb
  evaluate.py                 # Equivalente en .py de 04_validacion.ipynb
```

Cada notebook numerado (`00`-`04`) lee y escribe sobre `data/processed/` y `models/`, por lo que se pueden ejecutar de forma independiente (en orden) sin depender del estado en memoria de los notebooks anteriores. Los módulos en `src/` implementan exactamente la misma lógica para poder reproducir el pipeline fuera de Jupyter.

## Pipeline de datos

| Etapa | Notebook | Módulo | Entrada | Salida |
|---|---|---|---|---|
| Carga | `00_load_data.ipynb` | `src/data.py` | `data/raw/*.xlsx` | `data/processed/customer_features.csv` |
| EDA | `01_eda.ipynb` |  | `data/processed/customer_features.csv` | (solo análisis) |
| Preprocesamiento | `02_preprocessing.ipynb` | `src/preprocessing.py` | `data/processed/customer_features.csv` | `data/processed/customer_features_model.csv` |
| Entrenamiento | `03_entrenamiento.ipynb` | `src/train.py` | `data/processed/customer_features_model.csv` | `models/isolation_forest.joblib`, `models/best_params.json`, `data/processed/model_predictions.csv` |
| Validación | `04_validacion.ipynb` | `src/evaluate.py` | `data/processed/customer_features_model.csv`, `data/processed/model_predictions.csv` | (solo evaluación) |

Nota: DBSCAN no expone un método `predict()` independiente (es transductivo), por lo que solo se persisten sus etiquetas sobre el dataset de entrenamiento; Isolation Forest sí se guarda como modelo reutilizable (`models/isolation_forest.joblib`).

## Variables construidas por cliente

A partir de las transacciones se construye un dataset a nivel `CustomerID` con:

* **Permanencia**: días entre la primera y la última compra.
* **Compras**: número de órdenes distintas.
* **Canasta_Prom / Ticket_Prom / Precio_Prom / Precio_Max**: unidades y montos promedio/máximo por compra.
* **Productos Distintos**: variedad de productos comprados.
* **Pct_Devoluciones**: proporción de órdenes devueltas.
* **Pais Principal / Paises Distintos**: país donde compra más seguido, y en cuántos países distintos ha comprado.

## Resultados

| Modelo | Silhouette Score | % Clientes marcados como anomalía |
|---|---|---|
| DBSCAN | ~0.08 | ~39.1% |
| Isolation Forest | ~0.58 | ~1.3% |

Isolation Forest aísla un grupo pequeño y bien separado de clientes atípicos (precios promedio/máximo elevados, alta tasa de devoluciones), mientras que el "ruido" de DBSCAN es demasiado amplio y poco discriminativo para usarse como detector de anomalías por sí solo. El detalle completo del análisis está en `04_validacion.ipynb`.

## Cómo ejecutar

```bash
pip install -r requirements.txt
```

Coloca `online_retail.xlsx` y `online_retail_ii.xlsx` en `data/raw/`, y luego corre los notebooks en orden (`00` → `04`), o el pipeline equivalente en Python:

```bash
python -m src.data
python -m src.preprocessing
python -m src.train
python -m src.evaluate
```

## App de Streamlit

`app.py` es una UI de solo lectura que resume todo el proyecto (datos, EDA, preprocesamiento,
estado de entrenamiento y evaluación de ambos modelos) a partir de los artefactos que genera el
pipeline, no reentrena nada en vivo. Primero corre el pipeline (`python -m src.data`, `src.preprocessing`,
`src.train`) para generar `data/processed/` y `models/`, y luego:

```bash
streamlit run app.py
```
