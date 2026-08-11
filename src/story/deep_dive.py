"""Version extendida del scrollytelling: un tramo por notebook (00 a 04).

Se muestra dentro de `scrollytelling.py` cuando el usuario presiona el
boton "Explorar el proceso completo". No es una pagina de Streamlit
aparte (evita depender del descubrimiento automatico de `pages/`, que se
comporta distinto entre versiones de Streamlit): es una funcion mas que
se llama condicionalmente segun `st.session_state`.
"""

import numpy as np
import streamlit as st

from src.preprocessing import COLS_LOG_NORMAL, COLS_NUMERICAS, COLS_PW_NORMAL, transform_distributions
from src.story import charts as ch
from src.story import style as sty


def render(df_customer, df_final, df_resultados, best_params, resumen, crosstab):
    sty.hero(
        kicker="El recorrido completo",
        title="Notebook por notebook, del Excel crudo a la conclusión",
        subtitle=(
            "Los mismos hallazgos de la historia corta, mostrando ahora cada paso del pipeline "
            "tal como está organizado en los notebooks del proyecto."
        ),
    )
    sty.nav("detalle", key_suffix="top")

    # -----------------------------------------------------------------
    # 00 - Carga de datos
    # -----------------------------------------------------------------
    sty.section(
        eyebrow="Notebook 00. Carga de datos",
        title="De dos archivos de Excel a un solo registro transaccional",
        body_html="""
        <p>Todo empieza con dos archivos públicos, Online Retail y su extensión Online Retail II,
        ambos con el historial de compras (a nivel línea de factura) de una tienda con sede en
        Reino Unido, entre 2009 y 2011. Se cargan por separado y se combinan en un único registro
        transaccional, cuidando no duplicar facturas que ya aparecían en ambos archivos.</p>
        <p>Sobre ese registro combinado se hace la primera limpieza: se descartan las
        transacciones sin un CustomerID asociado (no hay a quién atribuirlas) y se calcula
        Sales = Quantity x UnitPrice, la base para todas las métricas monetarias que vienen
        después.</p>
        """,
    )

    sty.section(
        eyebrow="Notebook 00. Carga de datos",
        title="De línea de factura a perfil de cliente",
        body_html="""
        <p>La unidad de análisis del proyecto no es la transacción, es el cliente. Por eso,
        después de separar compras de devoluciones (para poder calcular una tasa de devolución)
        y de excluir productos de prueba interna, todas las transacciones de cada cliente se
        agregan en un solo renglón por CustomerID.</p>
        """,
    )
    st.dataframe(
        {
            "Variable": ["Permanencia", "Compras", "Canasta_Prom", "Ticket_Prom", "Precio_Prom", "Precio_Max", "Productos Distintos", "Pct_Devoluciones", "Pais Principal", "Paises Distintos"],
            "Qué mide": [
                "Días entre la primera y la última compra",
                "Número de órdenes distintas",
                "Unidades promedio por compra",
                "Monto promedio gastado por compra",
                "Precio promedio ponderado por unidades",
                "Precio unitario más alto comprado",
                "Variedad de productos distintos",
                "Proporción de órdenes devueltas",
                "País donde compra más seguido",
                "Cantidad de países desde los que ha comprado",
            ],
        },
        use_container_width=True, hide_index=True,
    )
    sty.cards([
        (f"{df_customer.shape[0]:,}", "clientes resultantes"),
        (f"{df_customer.shape[1]}", "variables por cliente"),
    ])
    sty.divider()

    # -----------------------------------------------------------------
    # 01 - EDA
    # -----------------------------------------------------------------
    sty.section(
        eyebrow="Notebook 01. Análisis exploratorio",
        title="Una dispersión enorme, incluso antes de buscar anomalías",
        body_html=f"""
        <p>Las estadísticas descriptivas ya adelantan el fenómeno que el proyecto busca capturar.
        El cliente mediano hace 3 compras y permanece activo poco más de 7 meses, pero existen
        clientes con cientos de compras. El precio promedio por producto tiene una mediana de
        apenas $1.83, contra un máximo de más de $10,000: varios órdenes de magnitud de
        diferencia.</p>
        <p>El {(df_customer["Pais Principal"] == "United Kingdom").mean():.0%} de los clientes
        concentra la mayoría de sus compras en el Reino Unido, y el resto se reparte entre otros
        {df_customer["Pais Principal"].nunique() - 1} países. Esta alta concentración es la razón
        por la que, más adelante, esta variable no se codifica de forma tradicional.</p>
        """,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribución por país**")
        st.plotly_chart(ch.fig_categoricas_bar(df_customer), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown("**Variables numéricas (escala log)**")
        st.plotly_chart(ch.fig_boxplots_log(df_customer, COLS_NUMERICAS), use_container_width=True, config={"displayModeBar": False})

    sty.paragraph("""
        <p>Para no dejar que un puñado de valores extremos domine el análisis, las variables de
        conteo y monto se transforman con logaritmo antes de calcular la correlación de Pearson.
        Los patrones más claros: a mayor permanencia, más compras (correlación de 0.81); a mayor
        número de compras, mayor variedad de productos (0.69); y a menor precio unitario, más
        unidades por compra (-0.60).</p>
        """)
    st.markdown("**Correlación entre variables (transformación log1p aplicada)**")
    df_log = df_customer.copy()
    for col in COLS_NUMERICAS:
        df_log[col] = np.log1p(df_log[col].clip(lower=0))
    st.plotly_chart(ch.fig_correlation_heatmap(df_log, COLS_NUMERICAS), use_container_width=True, config={"displayModeBar": False})
    sty.caption("Estos boxplots y esta correlación son, precisamente, lo que justifica no recortar ni eliminar los valores extremos: son el fenómeno que se quiere detectar, no ruido a limpiar.")
    sty.divider()

    # -----------------------------------------------------------------
    # 02 - Preprocesamiento
    # -----------------------------------------------------------------
    sty.section(
        eyebrow="Notebook 02. Preprocesamiento",
        title="Poner a todas las variables en el mismo idioma",
        body_html="""
        <p>DBSCAN mide distancias entre clientes. Isolation Forest hace cortes aleatorios sobre
        cada variable. Ninguno de los dos funciona bien si una variable domina solo porque su
        escala numérica es más grande, como montos en cientos contra conteos de 1 a 20. El
        preprocesamiento resuelve eso en tres pasos.</p>
        """,
    )
    col_elegida = st.selectbox("Ver el efecto de la transformación logarítmica en:", COLS_LOG_NORMAL + COLS_PW_NORMAL, index=2)
    df_transformed, _ = transform_distributions(df_customer)
    st.plotly_chart(
        ch.fig_transform_hist(df_customer[col_elegida], df_transformed[col_elegida], col_elegida),
        use_container_width=True, config={"displayModeBar": False},
    )
    sty.caption("Comprimir la cola derecha ayuda a que los modelos basados en distancia no queden dominados por un puñado de clientes extremos.")

    sty.paragraph("""
        <p>Para la variable de país, un one-hot encoding directo generaría más de 40 columnas
        dispersas que, combinadas, pesarían más que las variables continuas al calcular
        distancias. En su lugar, se resume en dos banderas con significado de negocio:
        Comprador Local (compra sobre todo en Reino Unido) y Nómada (ha comprado desde más de un
        país). Por último, las variables continuas se estandarizan con media 0 y desviación 1
        usando StandardScaler, mientras que las dos banderas binarias se dejan en su escala
        natural.</p>
        """)
    sty.divider()

    # -----------------------------------------------------------------
    # 03 - Entrenamiento
    # -----------------------------------------------------------------
    sty.section(
        eyebrow="Notebook 03. Entrenamiento",
        title="Cómo decide DBSCAN quién es ruido",
        body_html="""
        <p>DBSCAN clasifica a cada cliente en una de tres categorías: punto núcleo (tiene
        suficientes vecinos cerca), punto frontera (está cerca de un núcleo pero no lo es) o
        punto de ruido (no es ninguna de las dos). Los puntos de ruido, los que quedan en zonas
        de baja densidad del espacio de variables, son los que se reinterpretan como
        anomalías.</p>
        """,
    )
    sty.section(
        eyebrow="Notebook 03. Entrenamiento",
        title="Cómo decide Isolation Forest quién es fácil de aislar",
        body_html="""
        <p>Isolation Forest construye cientos de árboles que parten los datos al azar, una
        variable y un punto de corte a la vez, hasta que cada cliente queda solo en su propia
        rama. A los clientes que se aíslan en muy pocos cortes se les asigna un puntaje de
        anomalía cercano a 1. A los que necesitan muchos cortes para aislarse, un puntaje bajo.</p>
        """,
    )

    if best_params is not None:
        st.markdown("**Hiperparámetros óptimos encontrados por Optuna**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*DBSCAN*")
            for k, v in best_params["dbscan"].items():
                st.metric(k, f"{v:.4g}" if isinstance(v, float) else v)
        with col2:
            st.markdown("*Isolation Forest*")
            for k, v in best_params["isolation_forest"].items():
                st.metric(k, f"{v:.4g}" if isinstance(v, float) else v)
        sty.caption(
            "Ambos se optimizaron maximizando el Silhouette Score con el muestreador TPE de "
            "Optuna, ya que no existe una variable objetivo contra la cual validar directamente."
        )
    sty.divider()

    # -----------------------------------------------------------------
    # 04 - Validacion
    # -----------------------------------------------------------------
    sty.section(
        eyebrow="Notebook 04. Validación",
        title="¿Las anomalías de un modelo son las mismas que las del otro?",
        body_html="""
        <p>Cruzando las etiquetas de ambos modelos aparece una relación clara: todas las
        anomalías que encuentra Isolation Forest ya estaban dentro del grupo de ruido que marcaba
        DBSCAN. Ningún cliente que Isolation Forest señala como raro había sido considerado
        normal por DBSCAN.</p>
        """,
    )
    st.dataframe(crosstab, use_container_width=True)
    sty.paragraph("""
        <p>Esto sugiere que Isolation Forest no está encontrando un grupo distinto, sino aislando
        (dentro del amplio y poco discriminativo grupo de ruido de DBSCAN) a los casos
        verdaderamente extremos. Es la evidencia final que respalda la conclusión del proyecto:
        para esta base de clientes, un modelo diseñado específicamente para aislamiento
        estructural encuentra un grupo de anomalías más chico, más separado y más accionable que
        reciclar el ruido de un algoritmo de clustering.</p>
        """)
    st.plotly_chart(ch.fig_model_comparison(resumen), use_container_width=True, config={"displayModeBar": False})
    sty.divider()

    sty.nav("detalle", key_suffix="bottom")
