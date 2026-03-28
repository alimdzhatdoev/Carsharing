"""Обучение MLP: пояснения, параметры, графики на русском."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.config_helpers import apply_training_ui_overrides, parse_hidden_layers, set_raw_data_relative
from app.services.data_service import list_available_datasets
from app.services.status_service import load_training_meta_dict, model_is_trained
from app.services.training_service import run_full_training
from app.ui.demo_components import page_title
from app.ui.russian_ui import TRAINING_HISTORY_COLUMNS_RU
from app.ui.utils import get_resolved_config
from app.utils.common import get_project_root


def render() -> None:
    page_title("Обучение модели")

    cfg_base = get_resolved_config()
    root = get_project_root()

    st.subheader("Какая модель используется")
    st.markdown(
        """
- **Архитектура:** многослойный перцептрон (**MLP**, полносвязные слои) для табличных данных.  
- **Выход:** один логит → **сигмоида** даёт вероятность класса «риск» (1).  
- **Функция потерь:** `BCEWithLogitsLoss` с **весом положительного класса** (`pos_weight`), чтобы лучше учитывать дисбаланс.  
- **Оптимизатор:** **AdamW** (адаптивная скорость обучения + L2-регуляризация через weight decay из конфига).  
- **Ранняя остановка:** обучение прерывается, если качество на **валидации** перестаёт улучшаться заданное число эпох.  
        """
    )

    st.subheader("Почему нейросеть для этой задачи")
    st.write(
        "Табличные данные с **нелинейными** зависимостями и **смешанными** типами признаков (после one-hot) хорошо подходят для **универсального аппроксиматора** в виде MLP. "
        "Альтернативы (логистическая регрессия, градиентный бустинг) в проекте могут сравниваться в офлайн-экспериментах; для диплома акцент — на **нейросетевом** решении."
    )

    names = list_available_datasets()
    if not names:
        st.warning("Нет CSV в data/raw. Перейдите в раздел **«Данные»** и сгенерируйте или загрузите файл.")
        return

    if model_is_trained(cfg_base):
        meta = load_training_meta_dict(cfg_base)
        with st.expander("Сохранённая модель (последнее обучение на диске)", expanded=False):
            if meta:
                st.write(f"Размерность входа после препроцессинга: **{meta.get('input_dim')}**")
                st.write(f"Порог классификации: **{meta.get('classification_threshold')}**")
                m = meta.get("model") or {}
                if isinstance(m, dict):
                    st.json(m)
            st.caption("После нового обучения эти значения обновятся.")

    st.subheader("Датасет и конфигурация")
    data_choice = st.selectbox("Файл для обучения (каталог data/raw)", names)
    cfg_path = root / "configs" / "config.yaml"
    if cfg_path.exists():
        with st.expander("Фрагмент configs/config.yaml (для справки)", expanded=False):
            st.code(cfg_path.read_text(encoding="utf-8")[:4000], language="yaml")

    st.subheader("Параметры обучения")
    with st.form("train_form"):
        c1, c2 = st.columns(2)
        epochs = c1.number_input("Максимум эпох", min_value=1, max_value=500, value=100)
        batch_size = c2.number_input("Размер мини-батча", min_value=8, max_value=512, value=64)
        lr = st.number_input("Скорость обучения (learning rate)", min_value=1e-6, max_value=0.5, value=0.001, format="%.6f")
        hl_text = st.text_input("Скрытые слои: размеры через запятую", value="128, 64, 32")
        dropout = st.slider("Вероятность dropout", 0.0, 0.7, 0.2)
        threshold = st.slider("Порог отнесения к классу «риск» (1)", 0.05, 0.95, 0.5)
        seed = st.number_input("Seed случайности", value=42, step=1)
        use_bn = st.checkbox("Пакетная нормализация (BatchNorm)", value=True)
        activation = st.selectbox("Функция активации", ["relu", "gelu"], format_func=lambda x: "ReLU" if x == "relu" else "GELU")
        submitted = st.form_submit_button("Запустить обучение")

    if not submitted:
        st.info(
            "После запуска отобразятся **графики потерь** и **F1 на валидации**. "
            "Низкие потери на обучении при росте потерь на валидации — признак переобучения."
        )
        _render_architecture_diagram(hl_text, hidden_layers=None)
        return

    try:
        hidden = parse_hidden_layers(hl_text)
    except ValueError as e:
        st.error(f"Неверный формат списка слоёв: {e}")
        return

    cfg = set_raw_data_relative(cfg_base, data_choice)
    cfg = apply_training_ui_overrides(
        cfg,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(lr),
        hidden_layers=hidden,
        dropout=float(dropout),
        use_batch_norm=use_bn,
        activation=activation,
        classification_threshold=float(threshold),
        random_seed=int(seed),
    )

    st.subheader("Описание выбранной архитектуры")
    _render_architecture_diagram(hl_text, hidden_layers=hidden)

    progress = st.progress(0)
    status = st.empty()

    def on_epoch(row: dict) -> None:
        ep = row.get("epoch", 0)
        progress.progress(min(1.0, ep / max(cfg.training.epochs, 1)))
        line = (
            f"Эпоха {ep}: потери train={row.get('train_loss', 0):.4f}, "
            f"потери val={row.get('val_loss', 0):.4f}, F1 val={row.get('val_f1', 0):.4f}"
        )
        status.text(line)

    with st.spinner("Выполняется обучение (сплит, препроцессинг, эпохи)..."):
        result = run_full_training(cfg, epoch_callback=on_epoch, capture_logs=True)

    progress.progress(1.0)

    if result.success:
        st.success(result.message)
        st.caption(f"Путь к весам: `{result.model_path}`")
        if result.training_summary:
            s = result.training_summary
            st.metric("Фактически пройдено эпох", s.get("epochs_ran", "—"))
            st.metric("Лучшая потеря на валидации", f"{s.get('best_val_loss', 0):.4f}")
        if result.test_metrics:
            with st.expander("Метрики на отложенном тесте (кратко)", expanded=False):
                st.json(result.test_metrics)
        if result.history:
            st.subheader("Динамика обучения")
            st.caption(
                "**Потери** — насколько средний прогноз модели расходится с меткой; "
                "**F1 на валидации** — баланс точности и полноты на проверочной части данных."
            )
            hdf = pd.DataFrame(result.history).set_index("epoch")
            rename = {k: v for k, v in TRAINING_HISTORY_COLUMNS_RU.items() if k in hdf.columns}
            st.line_chart(hdf[list(rename.keys())].rename(columns=rename))
        if result.log_lines:
            with st.expander("Журнал сообщений (фрагмент)"):
                st.text("\n".join(result.log_lines[-200:]))
    else:
        st.error("Ошибка обучения: " + result.message)
        if result.log_lines:
            st.text("\n".join(result.log_lines[-200:]))


def _render_architecture_diagram(hl_text: str, hidden_layers: list[int] | None = None) -> None:
    try:
        hidden = list(hidden_layers) if hidden_layers is not None else parse_hidden_layers(hl_text)
    except ValueError:
        hidden = []
    lines = [
        "1. **Вход:** вектор признаков фиксированной размерности (после предобработки sklearn).",
    ]
    step = 2
    for i, h in enumerate(hidden, start=1):
        lines.append(
            f"{step}. **Скрытый слой {i}:** {h} нейронов, активация и dropout по настройкам, "
            "при включённой опции — нормализация по батчу (BatchNorm)."
        )
        step += 1
    lines.append(
        f"{step}. **Выход:** один логит; при прогнозе применяется сигмоида → вероятность класса «риск» (1)."
    )
    st.markdown("\n".join(lines))
