"""Scrollytelling del proyecto "Deteccion de Clientes Anomalos en E-commerce".

Cuenta, en tramos cortos, la historia del reporte (docs/reporte.tex): el
problema, el metodo, el proceso de ajuste y los hallazgos. Antes de
correrla:

    python -m src.data
    python -m src.preprocessing
    python -m src.train

y luego:

    streamlit run scrollytelling.py

La app tiene tres vistas controladas por `st.session_state.story_view`:

    historia      -- la historia corta (por defecto), de solo lectura.
    detalle       -- recorrido completo notebook por notebook (src/story/deep_dive.py).
    laboratorio   -- unica vista que SI ejecuta modelos en vivo: reentrenar
                      Isolation Forest y probarlo sobre una muestra de
                      clientes reales (src/story/lab.py). No sobreescribe
                      los artefactos oficiales en models/.

La navegacion entre vistas usa botones normales (st.session_state +
st.rerun), no un enlace de multipagina, para no depender del
descubrimiento automatico de la carpeta `pages/` (su comportamiento
varia entre versiones de Streamlit).
"""

import streamlit as st

from src.story import data as sd
from src.story import deep_dive
from src.story import lab
from src.story import sections as sec
from src.story import style as sty

st.set_page_config(page_title="La historia de los clientes anomalos", page_icon="🔎", layout="centered", initial_sidebar_state="collapsed")
sty.inject_css()

df_customer = sd.load_customer_features()
df_final = sd.load_features_model()
df_resultados = sd.load_predictions()
best_params = sd.load_best_params()
isof_oficial = sd.load_isof_model()

if df_resultados is None:
    st.error(
        "No se encontraron predicciones en `data/processed/model_predictions.csv`. "
        "Corre `python -m src.train` para generar los artefactos antes de abrir esta historia."
    )
    st.stop()

resumen = sd.get_compare_models(df_final, df_resultados)
crosstab = sd.get_crosstab(df_resultados)

if "story_view" not in st.session_state:
    st.session_state.story_view = "historia"

if st.session_state.story_view == "detalle":
    deep_dive.render(df_customer, df_final, df_resultados, best_params, resumen, crosstab)
    st.stop()

if st.session_state.story_view == "laboratorio":
    lab.render(df_customer, df_final, isof_oficial, best_params, resumen)
    st.stop()

trials_dbscan = sd.load_trials("dbscan")
trials_isof = sd.load_trials("isof")
perfil_isof = sd.get_anomaly_profile(df_resultados, "anomalia_isof")

df_plot, var_explicada = sd.compute_pca(df_final)

sec.section_1_hook(df_customer)
sty.nav("historia", key_suffix="top")
sty.divider()

sec.section_2_contexto(df_customer)
sty.divider()

sec.section_3_dos_sospechosos()
sty.divider()

sec.section_4_proceso(trials_dbscan, trials_isof, best_params)
sty.divider()

sec.section_5_resultados(resumen)
sty.divider()

sec.section_6_perfil(
    perfil_isof,
    df_plot, df_resultados["label_isof"], var_explicada,
    df_plot_dbscan=df_plot, etiquetas_dbscan=df_resultados["label_dbscan"], var_explicada_dbscan=var_explicada,
)
sty.divider()

sec.section_7_implicaciones(
    n_isof=int(resumen.loc["Isolation Forest", "Anomalias Detectadas"]),
    pct_isof=resumen.loc["Isolation Forest", "Porcentaje Anomalias"],
)
sty.divider()

sec.section_8_cierre()

sty.nav("historia", key_suffix="bottom")
