"""Демонстрация прогнозирования: сценарии (только просмотр + авто-результат), ручной ввод, справка по весам."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.payload_examples import TRIP_FEATURES_EXAMPLE
from app.data.schema import CATEGORICAL_ALLOWED
from app.features.build_features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from app.services.prediction_service import build_inference_service
from app.services.status_service import model_is_trained
from app.ui.demo_components import page_title
from app.ui.prediction_demo_config import (
    DEMO_SCENARIOS,
    feature_keys_by_influence_desc,
    influence_label,
    influence_stars,
    scenario_feature_dict,
)
from app.ui.russian_ui import feature_label_ru
from app.ui.utils import get_resolved_config

_MANUAL_FORM_VER = "manual_form_version"


def _field_label(name: str) -> str:
    return f"{feature_label_ru(name)} — {influence_label(name)}"


def _format_feature_value_display(name: str, v) -> str:
    if name in ("late_finish_flag", "night_trip_flag", "weekend_flag"):
        return "да" if int(v) == 1 else "нет"
    if name in ("start_hour", "day_of_week", "payment_delay_count", "previous_incidents_count"):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}" if abs(v) >= 0.01 or v == 0 else f"{v:.4f}"
    return str(v)


def _render_feature_table_readonly(merged: dict) -> None:
    """Таблица признаков для демонстрации сценария: название, звёзды влияния, значение."""
    ordered = feature_keys_by_influence_desc()
    rows = [
        {
            "Признак": feature_label_ru(name),
            "Влияние": influence_stars(name),
            "Значение": _format_feature_value_display(name, merged[name]),
        }
        for name in ordered
    ]
    df = pd.DataFrame(rows)
    st.caption("Строки отсортированы по убыванию условного влияния признака.")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _scenario_explanation_with_result(scenario: dict, cls: int, prob: float, thr: float) -> str:
    base = (scenario.get("rationale_ru") or "").strip()
    pct = prob * 100
    if cls == 1:
        tail = (
            f" Итог модели: класс «риск»; вероятность проблемной поездки {pct:.1f} %, "
            f"не ниже порога {thr:.2f}. Нейросеть относит такой набор признаков к положительному классу: "
            "совокупность сигналов для неё сильнее выбранного порога."
        )
    else:
        tail = (
            f" Итог модели: класс «норма»; вероятность риска {pct:.1f} % ниже порога {thr:.2f}. "
            "Сигналов проблемности в векторе недостаточно, чтобы перейти через порог классификации."
        )
    return base + tail


def _manual_explanation_ru(features: dict, cls: int, prob: float, thr: float) -> str:
    ordered = feature_keys_by_influence_desc()
    top = ordered[:6]
    bits = [f"{feature_label_ru(k)} = {features[k]}" for k in top]
    head = (
        "Заданы значения всех признаков. Для устного комментария удобно опереться на поля с наибольшим условным весом: "
        + "; ".join(bits)
        + ". "
    )
    pct = prob * 100
    if cls == 1:
        tail = (
            f"Модель отнесла поездку к классу «риск»: вероятность {pct:.1f} % при пороге {thr:.2f}, "
            "то есть нейросеть считает совокупность признаков достаточно «тяжёлой» относительно обучения."
        )
    else:
        tail = (
            f"Модель отнесла поездку к классу «норма»: {pct:.1f} % вероятности риска ниже порога {thr:.2f}."
        )
    return head + tail


def render() -> None:
    page_title("Демонстрация прогнозирования")

    cfg = get_resolved_config()
    if not model_is_trained(cfg):
        st.error(
            "Модель ещё не обучена или не найдены артефакты в `artifacts/`. "
            "Сначала выполните обучение в разделе «Обучение модели»."
        )
        return

    svc = build_inference_service(cfg)
    if not svc.ready:
        st.error("Не удалось загрузить модель или препроцессор.")
        return

    tab_scen, tab_manual, tab_weights = st.tabs(
        [
            "Готовые сценарии",
            "Ручной ввод",
            "Условные веса признаков",
        ]
    )

    with tab_scen:
        _tab_scenarios(svc)

    with tab_manual:
        _tab_manual(svc)

    with tab_weights:
        _tab_weights_reference()


def _tab_scenarios(svc) -> None:
    titles = [s["title"] for s in DEMO_SCENARIOS]
    choice = st.selectbox("Выберите сценарий", titles, key="demo_scen_pick")
    scenario = next(s for s in DEMO_SCENARIOS if s["title"] == choice)

    st.markdown(f"**Кратко:** {scenario['short']}")
    merged = scenario_feature_dict(scenario)

    st.subheader("Параметры сценария")
    _render_feature_table_readonly(merged)

    st.divider()
    st.subheader("Результат")
    try:
        out = svc.predict_one(merged)
    except Exception as e:
        st.error(str(e))
        return

    _show_prediction_result(out)
    st.subheader("Почему такой результат")
    st.markdown(_scenario_explanation_with_result(scenario, int(out["predicted_class"]), float(out["probability_positive"]), float(out["threshold_used"])))


def _tab_manual(svc) -> None:
    if _MANUAL_FORM_VER not in st.session_state:
        st.session_state[_MANUAL_FORM_VER] = 0

    st.caption(
        "Поля отсортированы от большего условного влияния к меньшему. "
        "«Сброс» возвращает значения демо-примера `TRIP_FEATURES_EXAMPLE`."
    )

    if st.button("Сброс к стандартному примеру", key="btn_reset_manual"):
        st.session_state[_MANUAL_FORM_VER] = int(st.session_state[_MANUAL_FORM_VER]) + 1
        st.rerun()

    ver = int(st.session_state[_MANUAL_FORM_VER])
    defaults = TRIP_FEATURES_EXAMPLE
    values: dict = {}

    ordered = feature_keys_by_influence_desc()
    num_set = set(NUMERIC_COLUMNS)
    cat_set = set(CATEGORICAL_COLUMNS)

    st.subheader("Признаки (по убыванию условного влияния)")
    cols = st.columns(3)
    idx = 0
    for name in ordered:
        lab = _field_label(name)
        d = defaults.get(name, 0)
        col = cols[idx % 3]
        idx += 1
        suf = f"v{ver}_{name}"
        if name in num_set:
            if name in ("late_finish_flag", "night_trip_flag", "weekend_flag"):
                values[name] = col.selectbox(lab, [0, 1], index=int(d) if d in (0, 1) else 0, key=f"m_n_{suf}")
            elif name in ("start_hour", "day_of_week", "payment_delay_count", "previous_incidents_count"):
                values[name] = col.number_input(lab, value=int(d), step=1, key=f"m_n_{suf}")
            else:
                values[name] = col.number_input(lab, value=float(d), step=0.1, format="%.4f", key=f"m_n_{suf}")
        elif name in cat_set:
            allowed = sorted(CATEGORICAL_ALLOWED.get(name, frozenset()))
            default_val = defaults.get(name, allowed[0] if allowed else "")
            if allowed:
                i = allowed.index(default_val) if default_val in allowed else 0
                values[name] = col.selectbox(lab, allowed, index=i, key=f"m_c_{suf}")
            else:
                values[name] = col.text_input(lab, value=str(default_val), key=f"m_c_{suf}")

    if st.button("Выполнить прогноз", key="btn_manual_pred"):
        try:
            out = svc.predict_one(values)
        except Exception as e:
            st.error(str(e))
            return
        _show_prediction_result(out)
        st.subheader("Почему такой результат")
        st.markdown(
            _manual_explanation_ru(
                values,
                int(out["predicted_class"]),
                float(out["probability_positive"]),
                float(out["threshold_used"]),
            )
        )


def _tab_weights_reference() -> None:
    st.markdown("Признаки отсортированы по убыванию условного веса (как в формах выше).")
    rows = []
    for key in feature_keys_by_influence_desc():
        rows.append(
            {
                "Признак (код)": key,
                "Название": feature_label_ru(key),
                "Условный вес": influence_label(key),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _show_prediction_result(out: dict) -> None:
    cls = int(out["predicted_class"])
    p = float(out["probability_positive"])
    thr = float(out["threshold_used"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Предсказанный класс", "Риск" if cls == 1 else "Норма")
    c2.metric("Вероятность класса «риск»", f"{p:.2%}")
    c3.metric("Порог", f"{thr:.2f}")
    if cls == 1:
        st.warning(
            f"По оценке модели — **повышенный риск**; вероятность **{p:.2%}** (порог {thr:.2f}). "
            "В продукте это сигнал для проверки, не автоматическое решение."
        )
    else:
        st.success(
            f"По порогу — класс **без повышенного риска**; P(риск) **{p:.2%}**.",
        )
