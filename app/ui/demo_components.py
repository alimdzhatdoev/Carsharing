"""
Стили демо и заголовки страниц. Тексты для устного доклада — в docs/defense_narrative.md.
"""

from __future__ import annotations

import streamlit as st


def inject_demo_styles() -> None:
    """Лёгкие стили: схема пайплайна и отступы."""
    st.markdown(
        """
<style>
  div.block-container { padding-top: 1.25rem; }
  .diploma-pipe-panel {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.95rem;
    padding: 0.75rem 0.85rem;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    background: #f1f5f9 !important;
    color: #111827 !important;
  }
  .diploma-pipe-panel * {
    color: #111827 !important;
  }
  .diploma-pipe-chip {
    display: inline-block;
    padding: 0.35rem 0.6rem;
    border-radius: 8px;
    font-weight: 500;
    color: #111827 !important;
  }
  .diploma-pipe-chip--blue {
    background: #e8f4fd !important;
    border: 1px solid #b3d9f7;
  }
  .diploma-pipe-chip--amber {
    background: #fff8e6 !important;
    border: 1px solid #f5d76e;
  }
  .diploma-pipe-chip--green {
    background: #eef9e8 !important;
    border: 1px solid #9dcf8a;
  }
  h1 { letter-spacing: -0.02em; }
  div[data-testid="stDataFrame"] {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 10px;
    overflow: hidden;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str) -> None:
    """Заголовок экрана (тексты для защиты — в docs/defense_narrative.md, без подписи в UI)."""
    st.title(title)


def defense_scenario_sidebar() -> None:
    with st.expander("Документация проекта", expanded=False):
        st.markdown(
            "В каталоге `docs/` — в том числе `defense_demo_guide.md` и `defense_narrative.md`."
        )


def pipeline_step_card(step: str, title: str, detail: str) -> None:
    st.markdown(f"**{step}. {title}**  \n{detail}")
