"""Contenido y orden de las 8 secciones del scrollytelling.

Cada funcion recibe los datos ya cargados (ver `scrollytelling.py`) y se
encarga de renderizar su propio texto y su visualizacion. Mantenerlas
separadas hace que agregar, quitar o reordenar una seccion sea un cambio
de una sola linea en el archivo principal.
"""

import streamlit as st

from src.story import charts as ch
from src.story import style as sty


def section_1_hook(df_customer):
    sty.hero(
        kicker="Una historia de datos",
        title="¿Quién es el cliente que no encaja?",
        subtitle=(
            "En cualquier tienda hay compradores que se salen del patrón: gastan mucho más, "
            "compran distinto o devuelven casi todo. La pregunta es cómo encontrarlos sin que "
            "nadie te diga de antemano quiénes son."
        ),
    )
    sty.section(
        eyebrow="El problema",
        title="Miles de clientes, ni una sola pista",
        body_html="""
        <p>Imagina que administras una tienda en línea y quieres saber qué clientes merecen
        atención especial. ¿Son compradores VIP que hay que cuidar? ¿Cuentas con actividad
        sospechosa? ¿Errores de captura que están inflando tus reportes?</p>
        <p>El problema es que nadie te entrega una lista con la respuesta correcta. No hay una
        columna en la base de datos que diga "anómalo: sí o no". Hay que enseñarle a una máquina
        cómo luce el comportamiento normal para que, por sí sola, note cuándo algo se sale de ese
        molde.</p>
        """,
    )
    st.plotly_chart(ch.fig_hook_scatter(df_customer), use_container_width=True, config={"displayModeBar": False})
    sty.caption("Cada punto es un cliente. Los marcados en coral son solo un adelanto visual, todavía no es el resultado de ningún modelo.")


def section_2_contexto(df_customer):
    n_clientes = df_customer.shape[0]
    n_paises = df_customer["Pais Principal"].nunique()

    sty.section(
        eyebrow="El terreno de juego",
        title=f"{n_clientes:,} clientes, cero etiquetas",
        body_html=f"""
        <p>Los datos vienen de una tienda de e-commerce real con sede en el Reino Unido, con
        transacciones registradas entre 2009 y 2011. Cada renglón original es una línea de una
        factura: qué producto, cuántas unidades, a qué precio, en qué fecha.</p>
        <p>Eso no sirve tal cual para buscar clientes raros. Primero hay que resumir a cada
        cliente en un perfil: cuánto compra, con qué frecuencia, cuánto gasta en promedio, qué tan
        caro es lo que compra, qué tanto devuelve. Después de esa limpieza y ese resumen quedaron
        {n_clientes:,} clientes, cada uno descrito por su propio comportamiento de compra a lo
        largo de su relación con la tienda.</p>
        """,
    )
    sty.cards([
        (f"{n_clientes:,}", "clientes analizados"),
        (f"{n_paises}", "países de origen"),
        ("2009-2011", "periodo de compras"),
        ("0", "etiquetas de \"anómalo\" conocidas"),
    ], accent_indices={3})


def section_3_dos_sospechosos():
    sty.section(
        eyebrow="El giro",
        title="Dos formas distintas de sospechar",
        body_html="""
        <p>Para este proyecto se probaron dos maneras de detectar comportamiento raro. Piensan el
        problema de forma opuesta.</p>
        """,
    )
    col1, col2 = st.columns(2)
    with col1:
        sty.callout(
            """
            <strong>DBSCAN</strong><br>
            Junta a los clientes que se parecen entre sí en grupos. Al que no logra encajar en
            ningún grupo lo marca como "ruido".<br><br>
            <em>Es como agrupar a la gente de una fiesta según en qué rincón se paran: el que
            queda solo, deambulando, es el que llama la atención.</em>
            """,
        )
    with col2:
        sty.callout(
            """
            <strong>Isolation Forest</strong><br>
            No agrupa a nadie. Mide qué tan fácil es aislar a un cliente del resto haciendo
            preguntas al azar sobre sus datos.<br><br>
            <em>Es como jugar "20 preguntas": si a alguien lo identificas en 3 preguntas es porque
            tiene rasgos que lo distinguen muy rápido del resto.</em>
            """,
            warn=True,
        )
    sty.paragraph("""
        <p>Ninguno de los dos necesita ver ejemplos previos de "cliente anómalo". Ambos aprenden
        la estructura general del grupo y, a partir de ahí, señalan a quien se aleja de ella,
        solo que llegan a esa conclusión por caminos distintos.</p>
        """)


def section_4_proceso(trials_dbscan, trials_isof, best_params):
    sty.section(
        eyebrow="El proceso",
        title="Afinando el instinto de la máquina",
        body_html="""
        <p>Antes de comparar resultados, cada modelo necesita que se ajusten sus "perillas". Para
        DBSCAN, qué tan cerca deben estar dos clientes para considerarse similares. Para Isolation
        Forest, cuántos árboles de decisión usar y qué tan agresivo ser al marcar anomalías.</p>
        <p>Como no hay una respuesta correcta contra la cual comparar, se usó un puntaje interno
        llamado Silhouette Score (entre más alto, más nítida es la separación entre el grupo
        "normal" y el resto) junto con una búsqueda automática (Optuna) que prueba decenas de
        combinaciones y aprende de cada intento hacia dónde conviene buscar después.</p>
        """,
    )

    col1, col2 = st.columns(2)
    if trials_dbscan is not None:
        with col1:
            st.markdown("**DBSCAN**: búsqueda de `eps` y `min_samples`")
            st.plotly_chart(ch.fig_optuna_progress(trials_dbscan, sty.COLOR_ANOMALIA), use_container_width=True, config={"displayModeBar": False})
    if trials_isof is not None:
        with col2:
            st.markdown("**Isolation Forest**: búsqueda de sus 4 parámetros")
            st.plotly_chart(ch.fig_optuna_progress(trials_isof, sty.COLOR_NORMAL), use_container_width=True, config={"displayModeBar": False})

    if best_params is not None:
        sty.caption(
            f"DBSCAN terminó con eps de {best_params['dbscan']['eps']:.2f} y "
            f"min_samples de {best_params['dbscan']['min_samples']}. Isolation Forest terminó con "
            f"{best_params['isolation_forest']['n_estimators']} árboles, usando un "
            f"{best_params['isolation_forest']['max_samples']:.0%} de los clientes por árbol."
        )


def section_5_resultados(resumen):
    pct_dbscan = resumen.loc["DBSCAN", "Porcentaje Anomalias"]
    pct_isof = resumen.loc["Isolation Forest", "Porcentaje Anomalias"]
    n_isof = int(resumen.loc["Isolation Forest", "Anomalias Detectadas"])
    n_dbscan = int(resumen.loc["DBSCAN", "Anomalias Detectadas"])

    sty.section(
        eyebrow="La revelación",
        title="Un detector alarmista contra uno preciso",
        body_html=f"""
        <p>Con los mejores ajustes de cada modelo, los resultados se ven muy distintos entre sí.
        DBSCAN termina marcando como "ruido" (es decir, como posible anomalía) a
        <strong>{n_dbscan:,} clientes, un {pct_dbscan:.0f}% del total</strong>. Eso es casi 4 de
        cada 10 clientes, demasiados como para que un equipo de negocio pueda revisarlos uno por
        uno.</p>
        <p>Isolation Forest, en cambio, señala apenas a <strong>{n_isof} clientes
        ({pct_isof:.1f}%)</strong>. Un grupo mucho más chico y, como se ve abajo, también mucho
        más nítido en cuánto se aleja del resto.</p>
        """,
    )
    st.plotly_chart(ch.fig_model_comparison(resumen), use_container_width=True, config={"displayModeBar": False})
    sty.caption("El Silhouette Score mide qué tan bien separado queda el grupo de anomalías del resto. Más alto es mejor.")


def section_6_perfil(
    perfil_isof, df_plot_isof, etiquetas_isof, var_explicada_isof,
    df_plot_dbscan=None, etiquetas_dbscan=None, var_explicada_dbscan=None,
):
    sty.section(
        eyebrow="El perfil del sospechoso",
        title="¿Qué tienen en común los atípicos?",
        body_html="""
        <p>No basta con saber cuántos clientes raros hay, también hay que entender en qué son
        raros. Comparando el perfil promedio de los clientes marcados por Isolation Forest contra
        el resto aparece un patrón consistente: compran productos bastante más caros de lo usual
        y devuelven una fracción alta de sus pedidos.</p>
        """,
    )
    st.plotly_chart(ch.fig_profile_diff(perfil_isof), use_container_width=True, config={"displayModeBar": False})
    sty.caption("Cada barra compara el promedio del grupo anómalo contra el grupo normal, en unidades estandarizadas (0 = promedio general).")

    sty.paragraph("""
        <p>Proyectando a los clientes en un mapa de dos dimensiones, que conserva la mayor parte
        de la variación original, el grupo de anomalías de Isolation Forest no se dispersa por
        todos lados. Se concentra en zonas puntuales, bien separadas del resto.</p>
        """)
    st.plotly_chart(ch.fig_pca_scatter(df_plot_isof, etiquetas_isof), use_container_width=True, config={"displayModeBar": False})
    sty.caption(f"Varianza explicada por ambas componentes: {var_explicada_isof:.0%}")

    if df_plot_dbscan is not None:
        with st.expander("¿Y cómo se ve el \"ruido\" de DBSCAN en el mismo mapa?"):
            st.plotly_chart(
                ch.fig_pca_scatter(df_plot_dbscan, etiquetas_dbscan),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.caption(
                f"Varianza explicada: {var_explicada_dbscan:.0%}. A diferencia de Isolation "
                "Forest, el ruido de DBSCAN forma una nube extensa que rodea casi todo el grupo "
                "central, en vez de concentrarse en un subconjunto claro."
            )


def section_7_implicaciones(n_isof, pct_isof):
    sty.section(
        eyebrow="Por qué importa",
        title="De una lista de sospechosos a una acción",
        body_html=f"""
        <p>Un detector que señala al {pct_isof:.1f}% de los clientes (en este caso,
        {n_isof} personas) es algo con lo que un equipo de negocio puede trabajar de verdad.
        Se pueden revisar uno por uno y decidir si son clientes VIP que conviene cuidar o cuentas
        con abuso de devoluciones que ameritan una revisión.</p>
        <p>Isolation Forest tiene además una ventaja práctica: una vez entrenado, puede evaluar
        clientes nuevos sin volver a procesar a toda la base. DBSCAN solo sabe hablar del grupo
        de clientes con el que fue entrenado, así que para evaluar a alguien nuevo hay que repetir
        el proceso desde cero.</p>
        """,
    )
    sty.cards([
        ("Retención", "clientes de alto valor a cuidar"),
        ("Fraude", "cuentas con actividad sospechosa"),
        ("Calidad de datos", "errores de captura a corregir"),
    ])


def section_8_cierre():
    sty.section(
        eyebrow="El cierre",
        title="La forma de tus datos importa",
        body_html="""
        <p>La lección central de este proyecto no es que un modelo sea mejor que otro en
        abstracto. Es que la elección depende de cómo se comportan tus datos. Cuando los clientes
        no forman grupos densos y bien separados sino un continuo de comportamientos, reciclar el
        "ruido" de un algoritmo de agrupamiento como si fuera la respuesta puede sobreestimar por
        mucho el tamaño del problema. Un modelo diseñado específicamente para aislar casos
        extremos, como Isolation Forest, tiende a dar una respuesta más chica y más útil.</p>
        <p>Este recorrido resumió las ideas centrales del proyecto. El reporte completo entra en
        el detalle técnico, las fórmulas y la metodología completa de limpieza y validación.</p>
        """,
    )
    sty.caption("Equipo: Andrea Linette Mezquita Gómez, Carlos Enrique Cepeda Fuentes y Miguel Alejandro Blanco Ríos. Maestría en Ciencia de Datos, UANL.")
