"""Данные: источник, описание признаков, полная таблица, графики распределений, баланс классов."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.features.build_features import (
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from app.services.data_service import (
    dataset_full,
    generate_demo_dataset,
    list_available_datasets,
    raw_data_dir,
    save_uploaded_csv,
    summarize_dataset,
    validate_training_csv,
)
from app.ui.demo_components import page_title
from app.ui.russian_ui import FEATURE_DESCRIPTIONS_RU, feature_label_ru
from app.ui.utils import get_resolved_config


def _rename_preview_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: feature_label_ru(c) if c in FEATURE_DESCRIPTIONS_RU else c)


def _chart_column_title(col: str, *, use_ru: bool) -> str:
    if use_ru and col in FEATURE_DESCRIPTIONS_RU:
        return FEATURE_DESCRIPTIONS_RU[col][0]
    return col


def _render_column_charts(df: pd.DataFrame, *, use_ru: bool) -> None:
    """Гистограммы / столбчатые диаграммы для всех столбцов, кроме идентификаторов."""
    st.subheader("Графики по показателям")
    st.caption(
        "Для каждого столбца датасета (кроме уникальных идентификаторов) показано распределение значений. "
        "Для редких категорий в текстовых полях выводятся топ-25 значений по частоте."
    )
    skip = set(IDENTIFIER_COLUMNS)
    numeric_cols = [
        c
        for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [
        c
        for c in df.columns
        if c not in skip and not pd.api.types.is_numeric_dtype(df[c])
    ]

    if numeric_cols:
        st.markdown("**Числовые поля**")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            title = _chart_column_title(col, use_ru=use_ru)
            fig, ax = plt.subplots(figsize=(7.5, 2.9))
            nuniq = int(series.nunique())
            if nuniq <= 12:
                vc = series.value_counts().sort_index()
                x = [str(v) for v in vc.index]
                ax.bar(x, vc.values, color="#4C78A8", edgecolor="white", linewidth=0.4)
                ax.tick_params(axis="x", rotation=45)
                ax.set_ylabel("Записей")
            else:
                bins = min(40, max(12, nuniq // 4))
                ax.hist(series.astype(float), bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
                ax.set_ylabel("Частота")
            ax.set_title(title, fontsize=11)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if other_cols:
        st.markdown("**Категориальные и текстовые поля**")
        for col in other_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            title = _chart_column_title(col, use_ru=use_ru)
            vc = series.astype(str).value_counts()
            top_n = 25
            extra_caption = None
            if len(vc) > top_n:
                extra_caption = f"Показаны топ-{top_n} категорий; всего уникальных значений: {series.nunique():,}."
                vc = vc.head(top_n)
            fig_h = max(2.8, 0.32 * len(vc))
            fig, ax = plt.subplots(figsize=(7.5, min(fig_h, 14)))
            y = list(vc.index)[::-1]
            w = list(vc.values)[::-1]
            ax.barh(y, w, color="#F58518", edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Записей")
            ax.set_title(title, fontsize=11)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if extra_caption:
                st.caption(extra_caption)


def render() -> None:
    page_title("Данные исследования")

    cfg = get_resolved_config()
    raw_dir = raw_data_dir()
    config_path = Path(cfg.paths.raw_data)

    st.subheader("Источник текущих данных")
    st.write(
        "Путь к файлу, заданный в конфигурации обучения: при выборе другого файла в разделе «Обучение модели» "
        "будет использоваться выбранный CSV. Базовый путь в `configs/config.yaml` может указывать на демо-файл."
    )
    st.code(str(config_path), language="text")
    if config_path.exists():
        st.success("Файл по этому пути на диске **найден**.")
    else:
        st.warning("Файл по этому пути **не найден** — сгенерируйте демо-набор или загрузите CSV ниже.")

    st.info(
        "**Режим демонстрации:** по умолчанию используются **синтетические** данные, сгенерированные скриптом. "
        "Они сохраняют осмысленную структуру признаков каршеринга, но **не являются** реальной выгрузкой оператора."
    )

    st.subheader("Сгенерировать демонстрационный датасет")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_rows = st.number_input("Число строк", min_value=500, max_value=100_000, value=5000, step=500)
    with c2:
        demo_name = st.text_input("Имя файла", value="trips_demo.csv")
    with c3:
        seed = st.number_input("Seed воспроизводимости", value=42, step=1)
    if st.button("Сгенерировать и сохранить в data/raw"):
        with st.spinner("Генерация..."):
            path = generate_demo_dataset(int(n_rows), filename=demo_name, seed=int(seed))
        st.success(f"Файл сохранён: `{path}`")

    st.subheader("Загрузить свой CSV")
    up = st.file_uploader(
        "Файл в кодировке UTF-8 с колонками по контракту проекта (для обучения нужен столбец target_class)",
        type=["csv"],
    )
    if up is not None:
        if st.button("Сохранить в каталог data/raw"):
            p = save_uploaded_csv(up.getvalue(), up.name)
            st.success(f"Сохранено: `{p}`")

    st.subheader("Файлы в каталоге data/raw")
    names = list_available_datasets()
    if not names:
        st.warning("CSV в data/raw не найдены. Сгенерируйте демо-набор или загрузите файл.")
        return

    choice = st.selectbox("Выберите файл для анализа", names)
    path = raw_dir / choice

    ok, errors, warnings = validate_training_csv(path, strict_categories=False)
    for w in warnings:
        st.warning(w)
    if not ok:
        st.error("Проверка структуры не пройдена: " + "; ".join(errors))
    else:
        st.success("Базовая проверка структуры пройдена: ключевые колонки на месте.")

    st.subheader("Описание признаков, участвующих в модели")
    model_cols = list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)
    desc_rows = []
    for col in model_cols:
        short, long_txt = FEATURE_DESCRIPTIONS_RU.get(col, (col, ""))
        kind = "числовой" if col in NUMERIC_COLUMNS else "категориальный"
        desc_rows.append({"Признак (техн. имя)": col, "Название": short, "Тип": kind, "Смысл": long_txt})
    id_rows = [
        {"Признак (техн. имя)": c, "Название": feature_label_ru(c), "Тип": "идентификатор", "Смысл": FEATURE_DESCRIPTIONS_RU[c][1]}
        for c in IDENTIFIER_COLUMNS
    ]
    st.caption("Идентификаторы в модель **не подаются**.")
    st.dataframe(pd.DataFrame(id_rows), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

    summ = summarize_dataset(path)
    st.subheader("Объём и типы столбцов")
    st.metric("Число записей", f"{summ['n_rows']:,}")
    st.metric("Число столбцов в файле", summ["n_columns"])
    dtypes_df = pd.DataFrame(
        {"Столбец": list(summ["dtypes"].keys()), "Тип pandas": [str(v) for v in summ["dtypes"].values()]}
    )
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    df_full = dataset_full(path)
    num_df = df_full.select_dtypes(include=["number"])
    if not num_df.empty:
        st.subheader("Базовая статистика (числовые поля)")
        st.dataframe(num_df.describe().T, use_container_width=True)

    show_ru = st.checkbox("Русские заголовки столбцов в таблице и на графиках", value=True)
    st.subheader("Полный набор данных")
    st.caption(
        f"Отображаются **все** {len(df_full):,} строк выбранного файла. Таблицу можно прокручивать и сортировать."
    )
    display_df = _rename_preview_columns(df_full.copy()) if show_ru else df_full
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=520)
    st.download_button(
        label="Скачать полный CSV",
        data=df_full.to_csv(index=False).encode("utf-8-sig"),
        file_name=choice,
        mime="text/csv",
        key="download_full_dataset_csv",
    )

    _render_column_charts(df_full, use_ru=show_ru)

    if summ.get("target_counts") and TARGET_COLUMN in df_full.columns:
        st.subheader("Целевая переменная и баланс классов")
        st.write(
            f"Столбец **`{TARGET_COLUMN}`**: **0** — поездка в категории «норма», **1** — «повышенный риск» "
            "(в демо задаётся генератором)."
        )
        vc = df_full[TARGET_COLUMN].value_counts().sort_index()
        labels = {0: "0 — норма", 1: "1 — повышенный риск"}
        chart_df = pd.DataFrame(
            {"Класс": [labels.get(int(i), str(i)) for i in vc.index], "Число записей": vc.values}
        )
        st.bar_chart(chart_df.set_index("Класс"))
        st.caption(
            "Сильный дисбаланс классов типичен для задач риска; в обучении используется взвешивание положительного класса (см. раздел «Обучение»)."
        )
