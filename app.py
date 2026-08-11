"""
App de Streamlit para mostrar el proyecto de deteccion de clientes anomalos.

Es una app de solo lectura: visualiza los artefactos que ya genera el pipeline
(src/data.py -> src/preprocessing.py -> src/train.py -> src/evaluate.py), no
reentrena nada en vivo. Antes de correrla hay que generar los artefactos con:

    python -m src.data
    python -m src.preprocessing
    python -m src.train

y luego:

    streamlit run app.py

Nota de rendimiento: la navegacion usa un selector en el sidebar (no st.tabs)
para que, al cambiar cualquier opcion, solo se vuelva a ejecutar el codigo de
la seccion activa -- con st.tabs, Streamlit re-ejecuta el contenido de TODAS
las pestanas en cada interaccion, aunque esten ocultas. Ademas, todo calculo
puro (metricas de comparacion, perfiles, figuras) esta cacheado con
st.cache_data para que cambiar de modelo/columna reutilice el resultado ya
calculado en vez de recalcularlo.
"""

import json
from pathlib import Path

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA

from src.evaluate import anomaly_profile, compare_models, crosstab_models, load_results
from src.preprocessing import COLS_LOG_NORMAL, COLS_NUMERICAS, COLS_PW_NORMAL, transform_distributions

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

FEATURES_PATH = DATA_DIR / "customer_features.csv"
FEATURES_MODEL_PATH = DATA_DIR / "customer_features_model.csv"
PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
ISOF_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
DBSCAN_TRIALS_PATH = MODELS_DIR / "dbscan_trials.csv"
ISOF_TRIALS_PATH = MODELS_DIR / "isof_trials.csv"

CMAP_CORR = mcolors.LinearSegmentedColormap.from_list("custom_corr", ["#ef796d", "#ffffff", "#1164ad"])
COLOR_NORMAL = "#1164ad"
COLOR_ANOMALIA = "#ef796d"

COLS_PERFIL = [
    "Permanencia", "Compras", "Canasta_Prom", "Ticket_Prom",
    "Precio_Prom", "Precio_Max", "Productos Distintos", "Pct_Devoluciones",
]

st.set_page_config(page_title="Clientes Anomalos - E-commerce", layout="centered")


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------

@st.cache_data
def load_customer_features():
    return pd.read_csv(FEATURES_PATH, index_col="CustomerID")


@st.cache_data
def load_model_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    return load_results(str(FEATURES_MODEL_PATH), str(PREDICTIONS_PATH))


@st.cache_data
def load_features_model():
    return pd.read_csv(FEATURES_MODEL_PATH, index_col="CustomerID")


@st.cache_data
def load_best_params():
    if not BEST_PARAMS_PATH.exists():
        return None
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_trials(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.sort_values(by="value", ascending=False)
    df["best_so_far"] = df.sort_values("number")["value"].cummax()
    return df


@st.cache_resource
def load_isof_model():
    if not ISOF_MODEL_PATH.exists():
        return None
    return joblib.load(ISOF_MODEL_PATH)


@st.cache_data
def compute_pca(df_final: pd.DataFrame):
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(df_final)
    df_plot = pd.DataFrame(componentes, columns=["PC1", "PC2"], index=df_final.index)
    return df_plot, pca.explained_variance_ratio_.sum()


# ---------------------------------------------------------------------------
# Calculos puros cacheados (evita recalcular al cambiar de modelo/columna)
# ---------------------------------------------------------------------------

@st.cache_data
def get_compare_models(df_final: pd.DataFrame, df_resultados: pd.DataFrame):
    return compare_models(df_final, df_resultados)


@st.cache_data
def get_anomaly_profile(df_resultados: pd.DataFrame, anomaly_col: str):
    return anomaly_profile(df_resultados, anomaly_col)


@st.cache_data
def get_crosstab(df_resultados: pd.DataFrame):
    return crosstab_models(df_resultados)


# ---------------------------------------------------------------------------
# Figuras cacheadas (matplotlib es lo mas lento de regenerar en cada rerun)
# ---------------------------------------------------------------------------

@st.cache_resource
def fig_correlacion(df_customer: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_customer[COLS_NUMERICAS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=CMAP_CORR, center=0, ax=ax)
    return fig


@st.cache_resource
def fig_pais_principal(df_customer: pd.DataFrame):
    cat_counts = df_customer["Pais Principal"].value_counts().sort_values(ascending=False)
    cat_counts = pd.concat([
        cat_counts.iloc[:10],
        pd.Series([cat_counts.iloc[10:].sum()], index=[f"Otros ({cat_counts.iloc[10:].shape[0]})"]),
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=cat_counts.values, y=cat_counts.index, color=COLOR_NORMAL, ax=ax)
    ax.set_title("Pais Principal", fontweight="bold")
    ax.set_xlabel("Cantidad de clientes")
    return fig


@st.cache_resource
def fig_paises_distintos(df_customer: pd.DataFrame):
    market_counts = df_customer["Paises Distintos"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        market_counts,
        labels=market_counts.index.astype(str),
        autopct="%1.1f%%",
        colors=[COLOR_NORMAL, COLOR_ANOMALIA] + list(sns.color_palette("Set2", max(0, len(market_counts) - 2))),
    )
    ax.set_title("Paises Distintos por cliente", fontweight="bold")
    return fig


@st.cache_resource
def fig_boxplots_numericas(df_customer: pd.DataFrame):
    n_cols = 4
    n_rows = int(np.ceil(len(COLS_NUMERICAS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    for ax, col in zip(axes, COLS_NUMERICAS):
        sns.boxplot(y=df_customer[col], ax=ax, color=COLOR_NORMAL)
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("")
    for ax in axes[len(COLS_NUMERICAS):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_resource
def fig_transformacion(df_customer: pd.DataFrame, col_elegida: str):
    df_transformed, _ = transform_distributions(df_customer)

    fig_orig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_customer[col_elegida], kde=True, color=COLOR_ANOMALIA, ax=ax)
    ax.set_title(f"{col_elegida} - original", fontweight="bold")

    fig_trans, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_transformed[col_elegida], kde=True, color=COLOR_NORMAL, ax=ax)
    ax.set_title(f"{col_elegida} - transformada", fontweight="bold")

    return fig_orig, fig_trans


@st.cache_resource
def fig_comparacion_barras(resumen: pd.DataFrame):
    fig1, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=resumen.index, y=resumen["Silhouette Score"], hue=resumen.index,
                palette={"DBSCAN": COLOR_ANOMALIA, "Isolation Forest": COLOR_NORMAL}, legend=False, ax=ax)
    ax.set_title("Silhouette Score", fontweight="bold")

    fig2, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=resumen.index, y=resumen["Porcentaje Anomalias"], hue=resumen.index,
                palette={"DBSCAN": COLOR_ANOMALIA, "Isolation Forest": COLOR_NORMAL}, legend=False, ax=ax)
    ax.set_title("% Clientes marcados como anomalia", fontweight="bold")

    return fig1, fig2


@st.cache_resource
def fig_boxplots_por_anomalia(df_resultados: pd.DataFrame, anomaly_col: str):
    n_cols = 4
    n_rows = int(np.ceil(len(COLS_PERFIL) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    for ax, col in zip(axes, COLS_PERFIL):
        sns.boxplot(data=df_resultados, x=anomaly_col, y=col, ax=ax,
                    hue=anomaly_col, palette={0: COLOR_NORMAL, 1: COLOR_ANOMALIA}, legend=False)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomalia"])
    for ax in axes[len(COLS_PERFIL):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_resource
def fig_pca_scatter(df_plot: pd.DataFrame, etiquetas: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 6))
    normales = df_plot[etiquetas != -1]
    anomalos = df_plot[etiquetas == -1]
    ax.scatter(normales["PC1"], normales["PC2"], color=COLOR_NORMAL, alpha=0.4, s=15, label="Normal")
    ax.scatter(anomalos["PC1"], anomalos["PC2"], color=COLOR_ANOMALIA, alpha=0.7, s=25, marker="x", label="Anomalia")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Datos compartidos + navegacion
# ---------------------------------------------------------------------------

df_customer = load_customer_features()
df_resultados = load_model_predictions()
best_params = load_best_params()

PAGINAS = [
    "Resumen del Proyecto",
    "Datos y EDA",
    "Preprocesamiento",
    "Entrenamiento",
    "Evaluacion",
    "Conclusion",
]
pagina = st.sidebar.radio("Seccion", PAGINAS)


# ---------------------------------------------------------------------------
# 1. Resumen del proyecto
# ---------------------------------------------------------------------------
if pagina == "Resumen del Proyecto":
    st.title("Deteccion de Clientes Anomalos en E-commerce")
    st.caption("Proyecto final - Maestria en Ciencia de Datos, Aprendizaje Automatico - UANL")

    st.markdown(
        """
Construimos un perfil de compra a nivel cliente a partir de las transacciones de una tienda
de e-commerce (dataset [Online Retail / Online Retail II](https://archive.ics.uci.edu/dataset/352/online+retail))
y lo usamos para **detectar clientes con comportamiento anomalo**, comparando dos enfoques no supervisados:

* **DBSCAN** - clustering basado en densidad (visto en clase); los clientes que no se asignan a ningun
  cluster (`label == -1`) se interpretan como anomalias.
* **Isolation Forest** - modelo de ensamble disenado especificamente para deteccion de anomalias, que
  ademas expone un `predict()` reutilizable sobre clientes nuevos (a diferencia de DBSCAN, que es transductivo).

Ambos modelos se optimizan con [Optuna](https://optuna.org/) maximizando el *silhouette score*.
        """
    )

    n_categoricas = 2
    n_continuas = len(COLS_NUMERICAS)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clientes", f"{df_customer.shape[0]:,}")
    col2.metric("Variables continuas", n_continuas)
    col3.metric("Variables categoricas", n_categoricas)

    if df_resultados is not None:
        resumen = get_compare_models(load_features_model(), df_resultados)
        mejor_modelo = resumen["Silhouette Score"].idxmax()
        col4.metric("Mejor modelo", mejor_modelo, f"silhouette {resumen.loc[mejor_modelo, 'Silhouette Score']:.2f}")
    else:
        col4.metric("Mejor modelo", "N/D")
        st.info("Corre `python -m src.train` para generar las predicciones y ver las metricas del modelo.")

    st.markdown("**Equipo:** Andrea Linette Mezquita Gomez, Carlos Enrique Cepeda Fuentes, Miguel Alejandro Blanco Rios")


# ---------------------------------------------------------------------------
# 2. Datos y EDA
# ---------------------------------------------------------------------------
elif pagina == "Datos y EDA":
    st.header("Datos y Analisis Exploratorio")
    st.markdown(f"Dataset a nivel cliente: **{df_customer.shape[0]:,} filas x {df_customer.shape[1]} columnas**")

    st.dataframe(df_customer.head(20), width="stretch")

    st.subheader("Estadisticas descriptivas")
    st.dataframe(df_customer.describe().round(2), width="stretch")

    st.subheader("Correlacion entre variables continuas")
    st.pyplot(fig_correlacion(df_customer))

    st.subheader("Variables categoricas")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_pais_principal(df_customer))
    with col2:
        st.pyplot(fig_paises_distintos(df_customer))

    st.subheader("Distribucion de variables numericas")
    st.pyplot(fig_boxplots_numericas(df_customer))


# ---------------------------------------------------------------------------
# 3. Preprocesamiento
# ---------------------------------------------------------------------------
elif pagina == "Preprocesamiento":
    st.header("Preprocesamiento de datos")
    st.markdown(
        """
El preprocesamiento aplicado antes de entrenar los modelos consiste en 3 pasos:

1. **Transformacion de distribuciones** - log-normal para las variables con sesgo a la derecha
   (`{log_cols}`), potencia (Yeo-Johnson) para `{pw_cols}`.
2. **Codificacion de variables categoricas** - en vez de un one-hot completo de `Pais Principal`
   (mas de 40 categorias), se resume junto con `Paises Distintos` en dos banderas binarias:
   *Comprador Local* (compra principalmente en UK) y *Nomada* (compra en mas de un pais).
3. **Escalamiento** - se estandarizan unicamente las variables numericas continuas; las banderas
   binarias se dejan sin escalar.
        """.format(
            log_cols=", ".join(COLS_LOG_NORMAL),
            pw_cols=", ".join(COLS_PW_NORMAL),
        )
    )

    st.subheader("Efecto de la transformacion (antes / despues)")
    col_elegida = st.selectbox("Variable", COLS_LOG_NORMAL + COLS_PW_NORMAL, index=1)

    fig_orig, fig_trans = fig_transformacion(df_customer, col_elegida)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_orig)
    with col2:
        st.pyplot(fig_trans)

    if FEATURES_MODEL_PATH.exists():
        st.subheader("Dataset final para los modelos")
        st.dataframe(load_features_model().head(20), width="stretch")
    else:
        st.info("Corre `python -m src.preprocessing` para generar el dataset preprocesado.")


# ---------------------------------------------------------------------------
# 4. Entrenamiento - estado del modelo
# ---------------------------------------------------------------------------
elif pagina == "Entrenamiento":
    st.header("Estado de entrenamiento")

    if best_params is None:
        st.warning("No se encontro `models/best_params.json`. Corre `python -m src.train` para entrenar los modelos.")
    else:
        model_choice = st.radio("Modelo", ["DBSCAN", "Isolation Forest"], horizontal=True)

        if model_choice == "DBSCAN":
            params = best_params["dbscan"]
            trials = load_trials(DBSCAN_TRIALS_PATH)
        else:
            params = best_params["isolation_forest"]
            trials = load_trials(ISOF_TRIALS_PATH)

        st.subheader("Mejores hiperparametros")
        cols = st.columns(len(params))
        for col, (k, v) in zip(cols, params.items()):
            col.metric(k, f"{v:.4g}" if isinstance(v, float) else v)

        if trials is None:
            st.info(
                "No se encontro el historial de trials para este modelo "
                "(vuelve a correr `python -m src.train` para regenerarlo)."
            )
        else:
            st.subheader("Progreso de la optimizacion (Optuna)")
            st.caption(f"{len(trials)} trials evaluados, maximizando silhouette score")

            trials_ordenados = trials.sort_values("number")
            st.line_chart(trials_ordenados.set_index("number")[["value", "best_so_far"]])

            st.subheader("Mejores trials")
            cols_mostrar = [c for c in trials.columns if c in ("number", "value") or c.startswith("params_")]
            st.dataframe(
                trials.sort_values("value", ascending=False)[cols_mostrar].head(10).round(4),
                width="stretch",
            )

    st.subheader("Modelo persistido")
    isof_model = load_isof_model()
    if isof_model is not None:
        st.success(f"Isolation Forest cargado desde `{ISOF_MODEL_PATH}`: {isof_model}")
    else:
        st.info("Aun no se ha entrenado/guardado el modelo de Isolation Forest.")

    st.caption(
        "DBSCAN es transductivo (no expone `predict()` independiente), por lo que solo se "
        "persisten sus etiquetas sobre el dataset de entrenamiento, no un artefacto reutilizable."
    )


# ---------------------------------------------------------------------------
# 5. Evaluacion / Validacion
# ---------------------------------------------------------------------------
elif pagina == "Evaluacion":
    st.header("Evaluacion y validacion de modelos")

    if df_resultados is None:
        st.warning("No se encontraron predicciones. Corre `python -m src.train` para generarlas.")
    else:
        df_final = load_features_model()
        resumen = get_compare_models(df_final, df_resultados)

        st.subheader("Comparacion DBSCAN vs. Isolation Forest")
        st.dataframe(resumen, width="stretch")

        fig1, fig2 = fig_comparacion_barras(resumen)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig1)
        with col2:
            st.pyplot(fig2)

        st.subheader("Coincidencia entre modelos")
        st.dataframe(get_crosstab(df_resultados), width="stretch")

        st.subheader("Detalle por modelo")
        modelo_eval = st.radio("Modelo a inspeccionar", ["DBSCAN", "Isolation Forest"], horizontal=True, key="modelo_eval")
        anomaly_col = "anomalia_dbscan" if modelo_eval == "DBSCAN" else "anomalia_isof"
        label_col = "label_dbscan" if modelo_eval == "DBSCAN" else "label_isof"

        st.markdown("**Perfil promedio de variables (escaladas) por grupo**")
        st.dataframe(get_anomaly_profile(df_resultados, anomaly_col), width="stretch")

        st.pyplot(fig_boxplots_por_anomalia(df_resultados, anomaly_col))

        st.markdown("**Proyeccion PCA (2 componentes)**")
        df_plot, var_explicada = compute_pca(df_final)
        st.caption(f"Varianza explicada por las 2 componentes: {var_explicada:.2%}")

        st.pyplot(fig_pca_scatter(df_plot, df_resultados[label_col]))


# ---------------------------------------------------------------------------
# 6. Conclusion
# ---------------------------------------------------------------------------
elif pagina == "Conclusion":
    st.header("Conclusion")
    st.markdown(
        """
Para esta tarea de deteccion de anomalias en clientes de e-commerce, **Isolation Forest es el
modelo mas adecuado**: obtiene un silhouette score mucho mayor (~0.58 vs. ~0.08), identifica un
porcentaje de anomalias mas razonable y accionable (~1.3% vs. ~39%), y su perfil de anomalias
(precios promedio y maximos elevados, alta tasa de devoluciones) es mucho mas interpretable desde
el punto de vista de negocio.

DBSCAN resulta mas util como herramienta de segmentacion general (encontro un cluster mayoritario
coherente), pero no como detector de anomalias por si solo, dado que su nocion de "ruido" es
demasiado amplia y poco discriminativa para este dataset, en el que el comportamiento de compra
se comporta mas como un continuo que como grupos de densidad bien separados.

**Siguientes pasos:**

* Validar el perfil de anomalias con el equipo de negocio (¿son clientes VIP de alto valor, o
  casos de fraude/abuso de devoluciones?).
* Monitorear el modelo en produccion (drift de las variables de entrada conforme cambian los
  patrones de compra).
* Reentrenar periodicamente el Isolation Forest conforme se acumulen nuevas transacciones,
  aprovechando que expone un `predict()` reutilizable sobre clientes nuevos.
        """
    )
