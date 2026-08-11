"""CSS y helpers de layout para el efecto de scrollytelling.

No usa componentes HTML/JS embebidos (nada de iframes ni
`components.html`): el "reveal" al hacer scroll se logra con CSS puro
(`animation-timeline: view()`), soportado de forma nativa en navegadores
Chromium/Edge recientes. En navegadores que no lo soportan (Firefox/Safari
en versiones mas viejas) el `@supports` simplemente no aplica y las
secciones se muestran normales, sin animacion: la app nunca se rompe ni
queda en blanco, solo pierde el efecto cosmetico.
"""

import streamlit as st

COLOR_NORMAL = "#1164AD"
COLOR_ANOMALIA = "#EF796D"

_CSS = f"""
<style>
.block-container {{
    max-width: 780px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}}

.story-hero {{
    text-align: center;
    padding: 1.5rem 0 2.5rem;
    border-bottom: 1px solid rgba(120,120,120,0.15);
    margin-bottom: 2rem;
}}
.story-hero .story-kicker {{
    color: {COLOR_ANOMALIA};
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.85rem;
}}
.story-hero h1 {{
    font-size: 2.4rem;
    font-weight: 800;
    color: {COLOR_NORMAL};
    line-height: 1.2;
    margin: 0.5rem 0 0.8rem;
}}
.story-hero p {{
    font-size: 1.15rem;
    color: #444;
    max-width: 560px;
    margin: 0 auto;
}}

.story-section {{
    padding: 2.4rem 0 1.2rem;
    opacity: 1;
}}
.story-eyebrow {{
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    font-size: 0.78rem;
    color: {COLOR_ANOMALIA};
    margin-bottom: 0.4rem;
}}
.story-title {{
    font-size: 1.7rem;
    font-weight: 800;
    color: {COLOR_NORMAL};
    margin: 0 0 0.9rem;
    line-height: 1.3;
}}
.story-body p {{
    font-size: 1.05rem;
    line-height: 1.7;
    color: #31333f;
    margin: 0 0 0.9rem;
}}
.story-body strong {{ color: {COLOR_NORMAL}; }}

.story-callout {{
    background: rgba(17, 100, 173, 0.06);
    border-left: 4px solid {COLOR_NORMAL};
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.98rem;
}}
.story-callout.warn {{
    background: rgba(239, 121, 109, 0.08);
    border-left-color: {COLOR_ANOMALIA};
}}

.story-cards {{
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}}
.story-card {{
    flex: 1 1 140px;
    background: rgba(120,120,120,0.06);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}}
.story-card .value {{
    font-size: 1.5rem;
    font-weight: 800;
    color: {COLOR_NORMAL};
}}
.story-card.accent .value {{ color: {COLOR_ANOMALIA}; }}
.story-card .label {{
    font-size: 0.82rem;
    color: #555;
    margin-top: 0.2rem;
}}

.story-divider {{
    text-align: center;
    color: rgba(120,120,120,0.5);
    font-size: 1.4rem;
    letter-spacing: 0.6em;
    margin: 0.4rem 0 0.6rem;
}}

.story-caption {{
    font-size: 0.85rem;
    color: #666;
    text-align: center;
    margin-top: -0.6rem;
}}

div[data-testid="stButton"] button {{
    background: {COLOR_NORMAL};
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
    border: none;
    width: 100%;
    transition: opacity 0.15s ease;
}}
div[data-testid="stButton"] button:hover {{
    opacity: 0.88;
    color: white;
}}
div[data-testid="stButton"] button p {{
    color: white;
    font-size: 1.02rem;
}}
div[data-testid="stButton"] button:disabled {{
    background: rgba(17, 100, 173, 0.35);
    opacity: 1;
}}
div[data-testid="stButton"] button:disabled p {{
    color: white;
}}

section[data-testid="stSidebar"] {{
    display: none;
}}

@supports (animation-timeline: view()) {{
    .story-section, .story-hero {{
        animation: story-fade-in linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 25%;
    }}
    @keyframes story-fade-in {{
        from {{ opacity: 0; transform: translateY(28px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


_NAV_ITEMS = [
    ("historia", "🔎", "La historia"),
    ("detalle", "🔬", "El proceso completo"),
    ("laboratorio", "🧪", "Pruébalo tú mismo"),
]


def nav(current: str, key_suffix: str = ""):
    cols = st.columns(len(_NAV_ITEMS))
    for col, (key, icon, label) in zip(cols, _NAV_ITEMS):
        with col:
            widget_key = f"nav_{key}_{key_suffix}" if key_suffix else f"nav_{key}"
            if key == current:
                st.button(label, icon=icon, disabled=True, key=widget_key, use_container_width=True)
            elif st.button(label, icon=icon, key=widget_key, use_container_width=True):
                st.session_state.story_view = key
                st.rerun()


def hero(kicker: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="story-hero">
            <div class="story-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(eyebrow: str, title: str, body_html: str):
    st.markdown(
        f"""
        <div class="story-section">
            <div class="story-eyebrow">{eyebrow}</div>
            <div class="story-title">{title}</div>
            <div class="story-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def paragraph(body_html: str):
    st.markdown(f'<div class="story-section story-body">{body_html}</div>', unsafe_allow_html=True)


def callout(text_html: str, warn: bool = False):
    cls = "story-callout warn" if warn else "story-callout"
    st.markdown(f'<div class="{cls}">{text_html}</div>', unsafe_allow_html=True)


def cards(items: list[tuple[str, str]], accent_indices: set[int] = frozenset()):
    """items: lista de (valor, etiqueta)."""
    html = ['<div class="story-cards">']
    for i, (value, label) in enumerate(items):
        cls = "story-card accent" if i in accent_indices else "story-card"
        html.append(f'<div class="{cls}"><div class="value">{value}</div><div class="label">{label}</div></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def divider():
    st.markdown('<div class="story-divider">• • •</div>', unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<div class="story-caption">{text}</div>', unsafe_allow_html=True)
