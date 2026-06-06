"""Shared visual theme for the Streamlit app and all Plotly charts.

Keeping color, typography, and layout defaults in one place makes every page
look like part of the same product instead of a collection of separate scripts.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Brand palette (kept in sync with .streamlit/config.toml).
PRIMARY = "#00e88f"      # poisonous / accent
ACCENT = "#2d8cff"       # edible / secondary
WARNING = "#ffb454"
BACKGROUND = "#071421"
PANEL = "#102033"
TEXT = "#f4f7fb"
MUTED = "#9fb3c8"
GRID = "rgba(159, 179, 200, 0.14)"

COLOR_SEQUENCE = [PRIMARY, ACCENT, WARNING, "#ff6b6b", "#b794f6"]
TEMPLATE_NAME = "mushroom"


def _build_template() -> go.layout.Template:
    return go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT),
            title=dict(font=dict(size=16, color=TEXT), x=0.01, xanchor="left"),
            colorway=COLOR_SEQUENCE,
            margin=dict(l=48, r=24, t=56, b=44),
            xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside", tickcolor=GRID),
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside", tickcolor=GRID),
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
            hoverlabel=dict(bgcolor=PANEL, bordercolor=PRIMARY, font=dict(color=TEXT)),
            colorscale=dict(sequential=[[0, PANEL], [1, PRIMARY]]),
        )
    )


def register_plotly_theme() -> None:
    """Register and activate the shared Plotly template (idempotent)."""
    if TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = f"plotly_dark+{TEMPLATE_NAME}"


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the shared template and a consistent height to a figure."""
    fig.update_layout(template=f"plotly_dark+{TEMPLATE_NAME}", height=height)
    return fig


_GLOBAL_CSS = """
<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #102033 0%, #0a1a2b 60%, #071421 100%);
        border: 1px solid rgba(0, 232, 143, 0.22);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.35);
    }
    .hero h1 {margin: 0; font-size: 1.85rem; font-weight: 700; color: #f4f7fb; letter-spacing: -0.01em;}
    .hero p {margin: 8px 0 0; color: #9fb3c8; font-size: 1rem; max-width: 760px;}
    .hero .pill {
        display: inline-block; margin-top: 14px; margin-right: 8px;
        padding: 4px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        background: rgba(0, 232, 143, 0.12); color: #00e88f; border: 1px solid rgba(0, 232, 143, 0.3);
    }
    .hero .pill.alt {background: rgba(45, 140, 255, 0.12); color: #2d8cff; border-color: rgba(45, 140, 255, 0.3);}

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #102033;
        border: 1px solid rgba(159, 179, 200, 0.16);
        border-radius: 14px;
        padding: 16px 18px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    [data-testid="stMetric"]:hover {border-color: rgba(0, 232, 143, 0.45); transform: translateY(-2px);}
    [data-testid="stMetricLabel"] p {color: #9fb3c8; font-weight: 600; font-size: 0.82rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}

    /* Headings + dividers */
    h2, h3 {color: #f4f7fb; letter-spacing: -0.01em;}
    hr {border-color: rgba(159, 179, 200, 0.16);}

    /* Sidebar */
    [data-testid="stSidebar"] {background: #0a1a2b; border-right: 1px solid rgba(159, 179, 200, 0.12);}

    /* Buttons */
    .stButton > button {border-radius: 10px; font-weight: 600;}

    /* Footer caption */
    .app-footer {color: #6f8298; font-size: 0.8rem; margin-top: 30px; border-top: 1px solid rgba(159,179,200,0.14); padding-top: 14px;}
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet. Call once near the top of each page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    """Render the branded hero header.

    pills: list of (label, variant) where variant is "" (green) or "alt" (blue).
    """
    pill_html = ""
    for label, variant in pills or []:
        css_class = "pill alt" if variant == "alt" else "pill"
        pill_html += f'<span class="{css_class}">{label}</span>'
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_setup(page_title: str, title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    """One-call setup for a page: theme + CSS + hero header.

    Note: st.set_page_config must already have been called by the page.
    """
    register_plotly_theme()
    inject_css()
    hero(title, subtitle, pills)


def footer() -> None:
    st.markdown(
        '<div class="app-footer">Educational project only — do not use for real mushroom '
        "consumption or safety decisions. Built with Streamlit, scikit-learn, and Plotly.</div>",
        unsafe_allow_html=True,
    )
