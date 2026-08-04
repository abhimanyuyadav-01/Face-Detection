"""
Small shared helpers for the Streamlit layer only.

Deliberately kept separate from `backend/`, which has no UI dependency at
all. Nothing here does any ML work — it only renders things.

Styling philosophy: lean on Streamlit's native, version-stable building
blocks (theme colors from `.streamlit/config.toml`, `st.container(border=True)`
for cards, `st.tabs`, `st.metric`) rather than deep CSS overrides of
Streamlit's internal (and frequently-changing) class names. The one bit of
custom CSS here only targets plain semantic HTML tags (`h1`/`h2`/`h3`),
which is stable across Streamlit versions.
"""

from __future__ import annotations

import streamlit as st

ACCENT = "#0D9488"
ACCENT_SOFT = "#CCFBF1"


def inject_base_styles() -> None:
    """Injects a small, restrained amount of shared CSS. Call once per page."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&display=swap');

        h1, h2, h3 {{
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.02em;
        }}

        .ap-eyebrow {{
            display: inline-block;
            background: {ACCENT_SOFT};
            color: {ACCENT};
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.6rem;
        }}

        .ap-subtitle {{
            color: #475569;
            font-size: 1.05rem;
            margin-top: -0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(icon: str, title: str, eyebrow: str, subtitle: str) -> None:
    """Renders a consistent page header: eyebrow label, icon + title, subtitle."""
    st.markdown(f'<span class="ap-eyebrow">{eyebrow}</span>', unsafe_allow_html=True)
    st.markdown(f"# {icon} {title}")
    st.markdown(f'<p class="ap-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.divider()


def render_footer(prev_label: str | None = None, prev_page: str | None = None) -> None:
    """Renders a consistent footer with a link back to the home page."""
    st.divider()
    cols = st.columns([1, 1, 2])
    with cols[0]:
        st.page_link("app.py", label="Home", icon="🏠")
    if prev_label and prev_page:
        with cols[1]:
            st.page_link(prev_page, label=prev_label, icon="↩️")
