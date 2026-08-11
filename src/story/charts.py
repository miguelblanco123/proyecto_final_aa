"""Figuras Plotly usadas en el scrollytelling.

Todas devuelven un `go.Figure` ya estilizado (fondo transparente, tipografia
y colores de marca), listo para `st.plotly_chart`.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_NORMAL = "#1164AD"
COLOR_ANOMALIA = "#EF796D"
COLOR_GRID = "rgba(120, 120, 120, 0.15)"
FONT_FAMILY = "'Source Sans Pro', 'Segoe UI', system-ui, sans-serif"

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_FAMILY, size=14, color="#31333f"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family=FONT_FAMILY),
)


def _apply_base(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**_BASE_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False)
    return fig


def fig_hook_scatter(df_customer: pd.DataFrame) -> go.Figure:
    """Scatter 'teaser' con un puñado de clientes que llaman la atencion.

    No usa las etiquetas de ningun modelo (esos se presentan mas adelante):
    solo resalta, de forma simple e ilustrativa, a los clientes con el gasto
    promedio por compra mas extremo, para plantar la pregunta del misterio.
    """
    x = np.log1p(df_customer["Compras"])
    y = np.log1p(df_customer["Ticket_Prom"].clip(lower=0))

    umbral = y.quantile(0.985)
    es_raro = y >= umbral

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[~es_raro], y=y[~es_raro], mode="markers",
        marker=dict(color=COLOR_NORMAL, size=6, opacity=0.35),
        name="Clientes",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x[es_raro], y=y[es_raro], mode="markers",
        marker=dict(color=COLOR_ANOMALIA, size=10, symbol="x", line=dict(width=1, color=COLOR_ANOMALIA)),
        name="¿Casos raros?",
        hoverinfo="skip",
    ))
    fig.update_xaxes(title="Frecuencia de compra (escala log)", showticklabels=False)
    fig.update_yaxes(title="Gasto promedio por compra (escala log)", showticklabels=False)
    return _apply_base(fig, showlegend=True, legend=dict(orientation="h", y=1.15, x=0), height=380)


def fig_optuna_progress(trials: pd.DataFrame, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trials["number"], y=trials["value"], mode="markers",
        marker=dict(color=color, size=6, opacity=0.4),
        name="Intento",
    ))
    fig.add_trace(go.Scatter(
        x=trials["number"], y=trials["best_so_far"], mode="lines",
        line=dict(color=color, width=3),
        name="Mejor hasta el momento",
    ))
    fig.update_xaxes(title="Numero de intento")
    fig.update_yaxes(title="Silhouette Score")
    return _apply_base(fig, showlegend=True, legend=dict(orientation="h", y=1.15, x=0), height=360)


def fig_model_comparison(resumen: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Silhouette Score", "% de clientes marcados"))
    colores = [COLOR_ANOMALIA, COLOR_NORMAL]

    fig.add_trace(go.Bar(
        x=resumen.index, y=resumen["Silhouette Score"],
        marker_color=colores, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=resumen.index, y=resumen["Porcentaje Anomalias"],
        marker_color=colores, showlegend=False,
        text=[f"{v:.1f}%" for v in resumen["Porcentaje Anomalias"]],
        textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(title="Score", row=1, col=1)
    fig.update_yaxes(title="% de clientes", row=1, col=2)
    return _apply_base(fig, height=380)


def fig_pca_scatter(df_plot: pd.DataFrame, etiquetas: pd.Series) -> go.Figure:
    normales = df_plot[etiquetas != -1]
    anomalos = df_plot[etiquetas == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normales["PC1"], y=normales["PC2"], mode="markers",
        marker=dict(color=COLOR_NORMAL, size=6, opacity=0.35),
        name="Normal",
    ))
    fig.add_trace(go.Scatter(
        x=anomalos["PC1"], y=anomalos["PC2"], mode="markers",
        marker=dict(color=COLOR_ANOMALIA, size=8, symbol="x"),
        name="Anomalia",
    ))
    fig.update_xaxes(title="Componente principal 1")
    fig.update_yaxes(title="Componente principal 2")
    return _apply_base(
        fig, showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=440,
    )


def fig_profile_diff(perfil: pd.DataFrame) -> go.Figure:
    """Barras horizontales con la diferencia (anomalia - normal) por variable."""
    diff = (perfil.loc["Anomalia (1)"] - perfil.loc["Normal (0)"]).sort_values()
    colores = [COLOR_ANOMALIA if v >= 0 else COLOR_NORMAL for v in diff.values]

    fig = go.Figure(go.Bar(
        x=diff.values, y=diff.index, orientation="h",
        marker_color=colores,
        text=[f"{v:+.2f}" for v in diff.values],
        textposition="outside",
    ))
    fig.update_xaxes(title="Diferencia (desviaciones estandar) vs. cliente normal")
    fig.update_yaxes(title="")
    return _apply_base(fig, height=360)


def fig_correlation_heatmap(df: pd.DataFrame, cols: list) -> go.Figure:
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, COLOR_ANOMALIA], [0.5, "#ffffff"], [1, COLOR_NORMAL]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _apply_base(fig, height=460, margin=dict(l=10, r=10, t=20, b=10))


def fig_categoricas_bar(df_customer: pd.DataFrame) -> go.Figure:
    cat_counts = df_customer["Pais Principal"].value_counts().sort_values(ascending=False)
    top = cat_counts.iloc[:8]
    otros = cat_counts.iloc[8:].sum()
    conteo = pd.concat([top, pd.Series([otros], index=[f"Otros ({cat_counts.iloc[8:].shape[0]} paises)"])])

    fig = go.Figure(go.Bar(
        x=conteo.values, y=conteo.index, orientation="h",
        marker_color=COLOR_NORMAL,
    ))
    fig.update_xaxes(title="Cantidad de clientes")
    fig.update_yaxes(title="", autorange="reversed")
    return _apply_base(fig, height=380)


def fig_boxplots_log(df_customer: pd.DataFrame, cols: list) -> go.Figure:
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Box(
            y=np.log1p(df_customer[col].clip(lower=0)), name=col,
            marker_color=COLOR_NORMAL, boxpoints="outliers",
            marker=dict(color=COLOR_ANOMALIA, size=3, opacity=0.4),
        ))
    fig.update_yaxes(title="Valor (escala log1p)")
    return _apply_base(fig, showlegend=False, height=420)


def fig_transform_hist(serie_original: pd.Series, serie_transformada: pd.Series, nombre: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{nombre} (original)", f"{nombre} (transformada)"))
    fig.add_trace(go.Histogram(x=serie_original, marker_color=COLOR_ANOMALIA, showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=serie_transformada, marker_color=COLOR_NORMAL, showlegend=False), row=1, col=2)
    return _apply_base(fig, height=340)
