"""Laboratorio interactivo: reentrenar Isolation Forest y probarlo en vivo.

A diferencia del resto de la app (que solo lee artefactos ya generados),
esta vista sí ejecuta modelos en el momento: deja que el usuario mueva los
hiperparámetros de Isolation Forest, lo entrene sobre el mismo dataset
preprocesado del proyecto, y lo pruebe sobre una muestra de clientes
reales. Nada de esto sobreescribe los artefactos oficiales en
`models/`: todo vive en `st.session_state`, solo para esta sesión.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score

from src.story import style as sty
from src.train import train_isolation_forest

COLS_MOSTRAR = [
    "Permanencia", "Compras", "Ticket_Prom", "Precio_Prom", "Precio_Max", "Pct_Devoluciones",
]


def render(df_customer, df_final, isof_oficial, best_params, resumen_oficial):
    sty.hero(
        kicker="El laboratorio",
        title="Pruébalo tú mismo",
        subtitle=(
            "Aquí puedes mover los controles de Isolation Forest, entrenar tu propia versión "
            "sobre el mismo dataset del proyecto, y ver a quién marca como anómalo en una muestra "
            "de clientes reales."
        ),
    )
    sty.nav("laboratorio")

    _seccion_entrenamiento(df_final, best_params, resumen_oficial)
    sty.divider()
    _seccion_prediccion(df_customer, df_final, isof_oficial)


def _seccion_entrenamiento(df_final, best_params, resumen_oficial):
    sty.section(
        eyebrow="Paso 1",
        title="Entrena tu propia versión del modelo",
        body_html="""
        <p>Mueve los mismos cuatro controles que Optuna ajustó automáticamente en el proyecto
        original y entrena tu propio Isolation Forest sobre los 5,876 clientes. Los valores
        iniciales son los que resultaron óptimos en la búsqueda original.</p>
        """,
    )

    defaults = best_params["isolation_forest"] if best_params else {
        "n_estimators": 236, "max_samples": 0.57, "contamination": 0.013, "max_features": 0.98,
    }

    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("n_estimators (número de árboles)", 50, 300, int(defaults["n_estimators"]), step=10)
        max_samples = st.slider("max_samples (fracción de clientes por árbol)", 0.1, 1.0, float(defaults["max_samples"]), step=0.05)
    with col2:
        contamination = st.slider("contamination (proporción esperada de anomalías)", 0.01, 0.4, float(defaults["contamination"]), step=0.01)
        max_features = st.slider("max_features (fracción de variables por corte)", 0.5, 1.0, float(defaults["max_features"]), step=0.05)

    if st.button("Entrenar Isolation Forest", icon="⚙️"):
        params = dict(
            n_estimators=n_estimators, max_samples=max_samples,
            contamination=contamination, max_features=max_features,
        )
        with st.spinner("Entrenando sobre 5,876 clientes..."):
            modelo, labels = train_isolation_forest(df_final, params)
            n_clusters = len(set(labels))
            sil = silhouette_score(df_final, labels) if n_clusters >= 2 else None

        st.session_state.lab_model = modelo
        st.session_state.lab_params = params
        st.session_state.lab_labels = labels
        st.session_state.lab_silhouette = sil

    if "lab_model" in st.session_state:
        labels = st.session_state.lab_labels
        pct = (labels == -1).mean() * 100
        n_anom = int((labels == -1).sum())
        sil = st.session_state.lab_silhouette

        st.markdown("**Tu modelo, comparado con el oficial del proyecto**")
        tabla = pd.DataFrame({
            "Silhouette Score": [resumen_oficial.loc["Isolation Forest", "Silhouette Score"], sil if sil is not None else float("nan")],
            "Anomalías detectadas": [int(resumen_oficial.loc["Isolation Forest", "Anomalias Detectadas"]), n_anom],
            "% del total": [resumen_oficial.loc["Isolation Forest", "Porcentaje Anomalias"], round(pct, 2)],
        }, index=["Modelo oficial", "Tu modelo"])
        st.dataframe(tabla, use_container_width=True)

        if sil is None:
            sty.caption("Con esta combinación de parámetros el modelo no formó al menos 2 grupos distintos, así que el Silhouette Score no se pudo calcular. Prueba con otra contaminación o max_samples.")
    else:
        sty.caption("Todavía no has entrenado ningún modelo en esta sesión. Ajusta los controles y presiona \"Entrenar Isolation Forest\".")


def _seccion_prediccion(df_customer, df_final, isof_oficial):
    sty.section(
        eyebrow="Paso 2",
        title="Ponlo a prueba con clientes reales",
        body_html="""
        <p>Toma una muestra al azar de clientes del dataset y pide una predicción: ¿cuáles marca
        el modelo como anómalos? Puedes elegir entre el modelo oficial del proyecto o el que
        acabas de entrenar arriba.</p>
        """,
    )

    opciones_modelo = ["Modelo oficial del proyecto"]
    if "lab_model" in st.session_state:
        opciones_modelo.append("Tu modelo (el que entrenaste arriba)")
    modelo_elegido = st.radio("¿Con qué modelo quieres predecir?", opciones_modelo, horizontal=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        n_muestra = st.slider("Tamaño de la muestra", 5, 50, 15)
    with col2:
        st.write("")
        st.write("")
        tomar_muestra = st.button("Tomar muestra nueva", icon="🎲")

    if tomar_muestra or "lab_sample_ids" not in st.session_state or len(st.session_state.get("lab_sample_ids", [])) != n_muestra:
        st.session_state.lab_sample_ids = df_final.sample(n=n_muestra).index

    sample_ids = st.session_state.lab_sample_ids
    X_muestra = df_final.loc[sample_ids]

    modelo = st.session_state.lab_model if modelo_elegido.startswith("Tu modelo") else isof_oficial
    if modelo is None:
        st.warning("No hay un modelo oficial persistido en `models/isolation_forest.joblib`. Entrena uno arriba para poder predecir.")
        return

    labels_pred = modelo.predict(X_muestra)
    scores = modelo.decision_function(X_muestra)

    tabla = df_customer.loc[sample_ids, COLS_MOSTRAR].copy()
    tabla.insert(0, "Predicción", np.where(labels_pred == -1, "🔴 Anómalo", "🔵 Normal"))
    tabla["Puntaje"] = scores.round(3)
    tabla = tabla.sort_values("Puntaje")

    n_anomalos = int((labels_pred == -1).sum())
    sty.cards([
        (f"{n_anomalos}", "de la muestra marcados como anómalos"),
        (f"{n_muestra}", "clientes en la muestra"),
    ], accent_indices={0})

    st.dataframe(tabla, use_container_width=True)
    sty.caption("El puntaje viene de decision_function: entre más negativo, más atípico le parece el cliente al modelo. Los valores por encima de 0 se consideran normales.")
