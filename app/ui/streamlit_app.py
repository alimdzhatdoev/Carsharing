"""
Точка входа дипломного интерфейса (Streamlit).

Запуск из корня репозитория:
  streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.ui.demo_components import defense_scenario_sidebar, inject_demo_styles
from app.ui.views import (
    data_view,
    evaluation_view,
    help_view,
    practical_value_view,
    prediction_demo_view,
    preprocessing_view,
    problem_statement_view,
    project_about_view,
    training_view,
)

st.set_page_config(
    page_title="Каршеринг: нейросетевая классификация риска поездки",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_demo_styles()

PAGES: list[tuple[str, object]] = [
    ("1. О проекте", project_about_view.render),
    ("2. Постановка задачи", problem_statement_view.render),
    ("3. Данные", data_view.render),
    ("4. Предобработка данных", preprocessing_view.render),
    ("5. Обучение модели", training_view.render),
    ("6. Результаты и метрики", evaluation_view.render),
    ("7. Демонстрация прогнозирования", prediction_demo_view.render),
    ("8. Практическая ценность", practical_value_view.render),
    ("9. Справка и инструкция", help_view.render),
]

labels = [p[0] for p in PAGES]
with st.sidebar:
    st.markdown("### Навигация по этапам исследования")
    page = st.radio("Раздел", labels, label_visibility="collapsed")
    st.divider()
    defense_scenario_sidebar()
    st.divider()
    st.caption("Запуск из корня проекта:")
    st.code("streamlit run app/ui/streamlit_app.py", language="bash")

page_index = labels.index(page)
PAGES[page_index][1]()
