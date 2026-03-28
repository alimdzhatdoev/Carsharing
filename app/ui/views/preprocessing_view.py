"""Наглядное объяснение предобработки данных (без дублирования fit — только образовательный слой)."""

from __future__ import annotations

import streamlit as st

from app.features.build_features import CATEGORICAL_COLUMNS, IDENTIFIER_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from app.ui.demo_components import page_title, pipeline_step_card
from app.ui.russian_ui import feature_label_ru


def render() -> None:
    page_title("Предобработка данных")

    st.info(
        "В программе препроцессор **обучается только на train** и сохраняется в `artifacts/encoders/`. "
        "При прогнозе применяется тот же препроцессор — иначе распределение признаков на входе сети не совпадало бы с обучением."
    )

    st.subheader("Схема пайплайна")
    st.markdown(
        """
<div class="diploma-pipe-panel">
<span class="diploma-pipe-chip diploma-pipe-chip--blue">Сырой CSV</span>
<span>→</span>
<span class="diploma-pipe-chip diploma-pipe-chip--blue">Проверка схемы</span>
<span>→</span>
<span class="diploma-pipe-chip diploma-pipe-chip--blue">Разбиение train / val / test</span>
<span>→</span>
<span class="diploma-pipe-chip diploma-pipe-chip--amber">Препроцессор (fit на train)</span>
<span>→</span>
<span class="diploma-pipe-chip diploma-pipe-chip--green">Матрица признаков → нейросеть</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Шаги предобработки признаков")
    pipeline_step_card(
        "1",
        "Исключение неинформативных полей",
        f"Колонки `{', '.join(IDENTIFIER_COLUMNS)}` и `{TARGET_COLUMN}` не подаются в модель как признаки: "
        "идентификаторы не несут обобщающей информации, таргет — то, что мы предсказываем.",
    )
    pipeline_step_card(
        "2",
        "Обработка пропусков",
        "Для **числовых** признаков: заполнение **медианой** (robust к выбросам). "
        "Для **категориальных**: заполнение **наиболее частым** значением. Параметры имputation оцениваются на train.",
    )
    pipeline_step_card(
        "3",
        "Разделение типов",
        f"**Числовых** признаков: {len(NUMERIC_COLUMNS)}; **категориальных**: {len(CATEGORICAL_COLUMNS)}. "
        "Для каждого типа — свой подпайплайн sklearn.",
    )
    pipeline_step_card(
        "4",
        "Кодирование категорий",
        "**One-Hot Encoding** с `handle_unknown='ignore'`: на инференсе новые редкие категории не ломают систему.",
    )
    pipeline_step_card(
        "5",
        "Масштабирование чисел",
        "**StandardScaler** (среднее и масштаб по обучающей выборке): ускоряет и стабилизирует обучение MLP.",
    )
    pipeline_step_card(
        "6",
        "Объединение и обучение",
        "`ColumnTransformer` объединяет ветки; на выходе — **вещественная матрица**, которую получает **TabularMLP** в PyTorch.",
    )

    with st.expander("Состав признаков по типам (русские названия)", expanded=False):
        st.markdown("**Числовые**")
        for c in NUMERIC_COLUMNS:
            st.markdown(f"- **{feature_label_ru(c)}** (`{c}`)")
        st.markdown("**Категориальные**")
        for c in CATEGORICAL_COLUMNS:
            st.markdown(f"- **{feature_label_ru(c)}** (`{c}`)")

    st.subheader("До и после (логика)")
    st.markdown(
        """
| Этап | «До» | «После» |
|------|------|---------|
| Числа | Разные масштабы (км, часы, рубли) | Приведены к сопоставимому масштабу |
| Категории | Строки `economy`, `rain`, … | Разреженные бинарные столбцы one-hot |
| Пропуски | NaN в таблице | Заполнены по правилам train |
        """
    )
