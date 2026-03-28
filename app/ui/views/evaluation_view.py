"""Результаты и метрики: подробный разбор, графики и интерпретация для защиты."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.models.predict import load_training_meta
from app.services.evaluation_service import (
    confusion_matrix_image_path,
    read_classification_report_text,
    read_model_report_markdown,
    read_test_metrics,
    read_training_history,
    run_full_evaluation,
)
from app.ui.demo_components import page_title
from app.ui.russian_ui import METRIC_LABELS_RU, TRAINING_HISTORY_COLUMNS_RU
from app.ui.utils import get_resolved_config


def _parse_confusion_counts(metrics: dict[str, Any]) -> tuple[int, int, int, int] | None:
    cm = metrics.get("confusion_matrix")
    if not isinstance(cm, list) or len(cm) != 2:
        return None
    try:
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        return tn, fp, fn, tp
    except (TypeError, ValueError, IndexError):
        return None


def _test_split_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Сводка по тесту из JSON или из матрицы ошибок (старые артефакты)."""
    n = metrics.get("n_test")
    n_pos = metrics.get("n_positive")
    n_neg = metrics.get("n_negative")
    prev = metrics.get("prevalence_positive")
    if isinstance(n, int) and isinstance(n_pos, int) and isinstance(n_neg, int):
        return {
            "n_test": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence_positive": float(prev) if prev is not None else (n_pos / n if n else 0.0),
        }
    parsed = _parse_confusion_counts(metrics)
    if not parsed:
        return {"n_test": None, "n_positive": None, "n_negative": None, "prevalence_positive": None}
    tn, fp, fn, tp = parsed
    n_tot = tn + fp + fn + tp
    n_p = tp + fn
    return {
        "n_test": n_tot,
        "n_positive": n_p,
        "n_negative": tn + fp,
        "prevalence_positive": n_p / n_tot if n_tot else 0.0,
    }


def _training_meta_block(cfg) -> None:
    p = Path(cfg.paths.training_config_dump)
    if not p.exists():
        st.info(
            "Файл `training_config.json` не найден — раздел «Что обучалось» будет доступен после обучения модели."
        )
        return
    try:
        meta = load_training_meta(p)
    except OSError:
        st.warning("Не удалось прочитать конфигурацию обучения.")
        return

    st.subheader("Что именно обучалось и как оценивалось")
    st.markdown(
        """
Ниже — связка **данные → модель → порог → метрики** (как в типовом ML-пайплайне):

1. **Данные** разбиты на обучающую, валидационную и **отложенную тестовую** часть (тест не участвует в подборе весов).
2. **Модель** — полносвязная нейросеть (MLP) над вектором признаков после препроцессинга (масштабирование, имputation, one-hot).
3. На выходе сети — **логит**; вероятность риска считается как сигмоида от логита.
4. **Порог** превращает вероятность в класс 0/1; все метрики ниже (кроме ROC-AUC) считаются **при этом пороге**.
5. **ROC-AUC** характеризует качество **ранжирования** по вероятности и **не зависит** от одного конкретного порога.
        """
    )

    mc = meta.get("model") or {}
    thr = meta.get("classification_threshold", cfg.inference.classification_threshold)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Параметры модели (из `training_config.json`)**")
        st.json(
            {
                "input_dim": meta.get("input_dim"),
                "hidden_layers": mc.get("hidden_layers"),
                "dropout": mc.get("dropout"),
                "activation": mc.get("activation"),
                "use_batch_norm": mc.get("use_batch_norm"),
            }
        )
    with c2:
        st.markdown("**Гиперпараметры обучения (текущий `config.yaml`)**")
        st.caption("Число эпох и шаг обучения в дампе весов не хранятся — ниже значения из активной конфигурации UI.")
        st.json(
            {
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "classification_threshold": float(thr) if thr is not None else None,
                "random_seed": meta.get("random_seed"),
                "trained_at_utc": meta.get("trained_at_utc"),
            }
        )
    st.caption(f"Полный дамп: `{p}`")


def _render_metric_cards(metrics: dict[str, Any]) -> None:
    row1 = st.columns(5)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for col, key in zip(row1, keys):
        title, help_txt = METRIC_LABELS_RU.get(key, (key, ""))
        val = metrics.get(key)
        if val is None:
            col.metric(title, "—", help=help_txt)
        else:
            col.metric(title, f"{float(val):.4f}", help=help_txt)


def _render_metrics_bar_chart(metrics: dict[str, Any]) -> None:
    rows = []
    for key in ("accuracy", "precision", "recall", "f1"):
        v = metrics.get(key)
        if v is not None:
            label = METRIC_LABELS_RU.get(key, (key, ""))[0]
            rows.append({"Метрика": label, "Значение": float(v)})
    if not rows:
        return
    st.markdown("**Сводка метрик при выбранном пороге (столбчатая диаграмма)**")
    bdf = pd.DataFrame(rows).set_index("Метрика")
    st.bar_chart(bdf)


def _render_metric_expanders() -> None:
    st.subheader("Пояснение метрик простыми словами")
    items = [
        (
            "Accuracy (доля верных)",
            "Доля всех поездок, где предсказанный класс совпал с разметкой. "
            "Если класс «риск» редкий, модель может набрать высокую точность, почти всегда предсказывая «норма» — "
            "поэтому одной accuracy недостаточно.",
        ),
        (
            "Precision (точность для класса «риск»)",
            "Из всех поездок, которые модель назвала рискованными, какая доля действительно была рискованными. "
            "Низкая точность → много **ложных тревог**: придётся проверять лишние нормальные поездки.",
        ),
        (
            "Recall / TPR (полнота, чувствительность)",
            "Из всех **реально** рискованных поездок, какую долю модель нашла. "
            "Низкая полнота → много **пропусков** реальных проблем (опасно для бизнеса, если важно не упустить риск).",
        ),
        (
            "F1",
            "Баланс между точностью и полнотой. Удобно, когда нужен один числовой компромисс.",
        ),
        (
            "ROC-AUC",
            "Насколько хорошо модель **отделяет** рисковые поездки от нормальных по величине предсказанной вероятности, "
            "без привязки к одному порогу. 0.5 ≈ случайное угадывание, 1.0 — идеальное ранжирование.",
        ),
    ]
    for title, body in items:
        with st.expander(title):
            st.write(body)


def _render_roc_pr(metrics: dict[str, Any], prevalence: float | None) -> None:
    roc = metrics.get("roc_curve")
    pr = metrics.get("pr_curve")
    if not roc or not isinstance(roc, dict) or not roc.get("fpr"):
        st.subheader("Кривые ROC и Precision–Recall")
        st.info(
            "В сохранённых метриках нет точек кривых (старый файл после обучения). "
            "**Запустите обучение заново** или нажмите **«Пересчитать метрики на отложенном тесте»** — "
            "тогда в `test_metrics.json` появятся данные для графиков."
        )
        return

    st.subheader("Кривые ROC и Precision–Recall")
    st.markdown(
        "**ROC:** по оси X — доля нормальных поездок, ошибочно отмеченных как риск (FPR); "
        "по оси Y — доля найденных среди всех реально рискованных (TPR = recall). "
        "**PR-кривая:** компромисс точности и полноты при переборе порогов; на несбалансированных данных часто информативнее."
    )

    fpr = np.asarray(roc["fpr"], dtype=float)
    tpr = np.asarray(roc["tpr"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color="#4C78A8", linewidth=2, label="Обученная модель")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1, label="Случайный классификатор")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR (ложные тревоги / все нормальные)")
    ax.set_ylabel("TPR = Recall (найденные риски / все риски)")
    ax.set_title("ROC-кривая (тест)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if pr and isinstance(pr, dict) and pr.get("recall"):
        prec = np.asarray(pr["precision"], dtype=float)
        rec = np.asarray(pr["recall"], dtype=float)
        fig2, ax2 = plt.subplots(figsize=(5.2, 5))
        ax2.plot(rec, prec, color="#F58518", linewidth=2, label="Модель")
        if prevalence is not None and 0 < prevalence < 1:
            ax2.axhline(y=prevalence, color="k", linestyle="--", alpha=0.4, label=f"Базовая линия ≈ {prevalence:.3f}")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("Recall (полнота)")
        ax2.set_ylabel("Precision (точность)")
        ax2.set_title("Precision–Recall (тест)")
        ax2.legend(loc="upper right", fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


def _render_confusion_deep_dive(metrics: dict[str, Any]) -> None:
    st.subheader("Матрица ошибок и доли по ячейкам")
    parsed = _parse_confusion_counts(metrics)
    if not parsed:
        st.warning("Матрица ошибок в метриках отсутствует или имеет неожиданный формат.")
        return
    tn, fp, fn, tp = parsed
    n = tn + fp + fn + tp
    if n == 0:
        return

    st.markdown(
        f"| | **Предсказано: норма** | **Предсказано: риск** |\n"
        f"|---|---:|---:|\n"
        f"| **Истина: норма** | TN = {tn} | FP = {fp} |\n"
        f"| **Истина: риск** | FN = {fn} | TP = {tp} |"
    )

    spec = tn / (tn + fp) if (tn + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    st.markdown(
        f"- **Специфичность** (нормальные не отмечены как риск): TN / (TN+FP) = **{spec:.4f}**.\n"
        f"- **Чувствительность** (= recall для класса риск): TP / (TP+FN) = **{sens:.4f}**."
    )

    breakdown = pd.DataFrame(
        {
            "Исход": ["Верно: норма (TN)", "Верно: риск (TP)", "Ложная тревога (FP)", "Пропуск риска (FN)"],
            "Количество": [tn, tp, fp, fn],
        }
    )
    breakdown["Доля от теста, %"] = (100.0 * breakdown["Количество"] / n).round(2)
    st.dataframe(breakdown, use_container_width=True, hide_index=True)
    st.bar_chart(breakdown.set_index("Исход")[["Количество"]])

    st.markdown(
        """
**Как читать для каршеринга**

- **TN** — модель справедливо отнесла поездку к «норме»; оператору не нужно тратить ресурс.
- **TP** — модель отметила реальный риск; можно инициировать проверку или эскалацию.
- **FP (ложная тревога)** — пользователь и поездка в целом «нормальные», но сработал фильтр: лишняя нагрузка на поддержку.
- **FN (пропуск)** — реальная проблема не была отмечена: с точки зрения безопасности и убытков это обычно самый нежелательный тип ошибки, если важно не пропускать риски.
        """
    )


def _render_verdict(metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    st.subheader("Краткий вывод по проделанной работе (для защиты)")
    lines: list[str] = []
    if summary.get("n_test"):
        lines.append(
            f"На **отложенном тесте** проверено **{summary['n_test']:,}** поездок; "
            f"доля реальных рисков (класс 1) — **{summary.get('prevalence_positive', 0):.1%}**."
        )
    ra = metrics.get("roc_auc")
    if ra is not None:
        lines.append(
            f"**ROC-AUC = {float(ra):.3f}** показывает, насколько уверенно модель отделяет классы по вероятности "
            f"(чем выше 0.5, тем лучше случайного угадывания)."
        )
    f1 = metrics.get("f1")
    pr = metrics.get("precision")
    rc = metrics.get("recall")
    if f1 is not None and pr is not None and rc is not None:
        lines.append(
            f"При зафиксированном пороге: **precision {float(pr):.3f}**, **recall {float(rc):.3f}**, **F1 {float(f1):.3f}** — "
            "это конкретный компромисс «ложные тревоги vs пропуски», его можно сдвигать изменением порога."
        )
    thr = metrics.get("threshold_used")
    if thr is not None:
        lines.append(
            f"Использованный **порог вероятности** для класса «риск»: **{float(thr):.3f}** "
            "(его можно настроить под политику сервиса, не переобучая сеть)."
        )
    lines.append(
        "Пайплайн воспроизводим: данные → препроцессинг → обучение MLP → сохранение весов и метрик → "
        "демонстрация прогноза в отдельном разделе приложения."
    )
    for line in lines:
        st.markdown(f"- {line}")


def render() -> None:
    page_title("Результаты и метрики")

    cfg = get_resolved_config()

    st.markdown(
        """
В этом разделе собран **итог эксперимента**: как обучалась модель, как она ведёт себя на **тесте**,
который не использовался при подборе весов, и как интерпретировать цифры с точки зрения каршеринга.
        """
    )

    _training_meta_block(cfg)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Пересчитать метрики на отложенном тесте"):
            with st.spinner("Пересчёт..."):
                res = run_full_evaluation(cfg)
            if res.success:
                st.success(res.message)
            else:
                st.error(res.message)
    with c2:
        st.caption(
            "Кнопка заново прогоняет сохранённую модель по `test_split.csv` и обновляет `test_metrics.json`, "
            "графики ROC/PR и отчёты — удобно после замены весов вручную."
        )

    history = read_training_history(cfg)
    if history:
        with st.expander("Кривые обучения (потери и F1 по эпохам)", expanded=True):
            st.caption(
                "По горизонтали — номер эпохи. Падение **потерь на валидации** и рост **F1 на валидации** "
                "обычно означают, что модель учится обобщать, а не только запоминать обучающую выборку."
            )
            hdf = pd.DataFrame(history).set_index("epoch")
            cols = [c for c in ("train_loss", "val_loss", "val_f1") if c in hdf.columns]
            if cols:
                rename = {k: TRAINING_HISTORY_COLUMNS_RU[k] for k in cols if k in TRAINING_HISTORY_COLUMNS_RU}
                st.line_chart(hdf[cols].rename(columns=rename))

    metrics = read_test_metrics(cfg)
    if not metrics:
        st.warning(
            "Метрики на тесте ещё не найдены. Выполните **обучение модели** в соответствующем разделе "
            "или убедитесь, что файл `artifacts/metrics/test_metrics.json` существует."
        )
        return

    summary = _test_split_summary(metrics)
    thr_used = metrics.get("threshold_used")

    st.subheader("Тестовая выборка")
    if summary["n_test"] is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Записей на тесте", f"{summary['n_test']:,}")
        m2.metric("Класс «норма» (0)", f"{summary['n_negative']:,}")
        m3.metric("Класс «риск» (1)", f"{summary['n_positive']:,}")
        prev = summary.get("prevalence_positive") or 0.0
        m4.metric("Доля риска на тесте", f"{prev:.1%}")
        bal = pd.DataFrame(
            {
                "Класс": ["Норма (0)", "Риск (1)"],
                "Количество": [summary["n_negative"] or 0, summary["n_positive"] or 0],
            }
        )
        st.bar_chart(bal.set_index("Класс"))
    else:
        st.info("Не удалось восстановить размер теста — проверьте файл метрик.")

    if thr_used is not None:
        st.caption(f"Все метрики ниже (кроме ROC-AUC) посчитаны при пороге вероятности **{float(thr_used):.4f}**.")

    st.subheader("Метрики на тестовой выборке")
    st.markdown(
        "**Почему недостаточно одной точности (Accuracy):** при дисбалансе классов модель может чаще предсказывать "
        "частый класс и получать высокую долю верных ответов при плохом качестве на редком классе «риск». "
        "Поэтому смотрят **precision**, **recall**, **F1** и **ROC-AUC**."
    )
    _render_metric_cards(metrics)
    _render_metrics_bar_chart(metrics)
    _render_metric_expanders()

    _render_roc_pr(metrics, summary.get("prevalence_positive"))

    st.subheader("Ошибки первого и второго рода в терминах каршеринга")
    st.markdown(
        """
- **Ложноположительный результат (false positive):** модель предсказала класс «риск», а по факту поездка относится к «норме».  
  *Последствие:* лишняя нагрузка на модерацию, возможный негатив у добросовестного пользователя.  
- **Ложноотрицательный результат (false negative):** модель предсказала «норма», а поездка была проблемной.  
  *Последствие:* пропущенный инцидент, мошенничество или ущерб для сервиса.  

Какой тип ошибки «хуже», решает **бизнес**: часто для риска важнее не пропускать проблемные кейсы → повышают **recall** (ценой ложных тревог), снижая порог.
        """
    )

    st.subheader("Визуализация матрицы ошибок")
    img_path = confusion_matrix_image_path(cfg)
    if img_path and Path(img_path).exists():
        st.image(str(img_path), use_container_width=True)
        st.caption(
            "Строки — **истинный** класс, столбцы — **предсказанный**. Диагональ — верные ответы; вне диагонали — ошибки."
        )
    else:
        st.info("Изображение матрицы не найдено. Выполните обучение или пересчёт метрик.")

    _render_confusion_deep_dive(metrics)

    rep = read_classification_report_text(cfg)
    if rep:
        with st.expander("Технический отчёт sklearn (классы 0 и 1)", expanded=False):
            st.text(rep)
            st.caption(
                "Стандартный вывод sklearn; основные метрики дублируются и поясняются выше по-русски."
            )

    md = read_model_report_markdown(cfg)
    if md:
        with st.expander("Сохранённый отчёт model_report.md", expanded=False):
            st.markdown(md)

    _render_verdict(metrics, summary)
