#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grading-preview page.

The teacher registers questions and reference answers on the main page.  This
page lets the teacher try a student answer against any registered question and
see exactly how the machine reached its score: similarity, threshold, matched
parts and the final points.

It also offers two bank-wide self-checks:

* every reference answer should receive full points;
* every labelled example should receive the score the teacher gave it.

Open it from the Streamlit sidebar after starting ``streamlit run
streamlit_app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs page scripts with the pages/ directory as the working
# directory in some versions, so make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grading_engine as ge
from app_shared import (
    METHOD_NAMES,
    create_scorer,
    grading_sidebar,
    load_bank,
)

st.set_page_config(
    page_title="Дүгнэлт шалгах",
    page_icon="🧪",
    layout="wide",
)

threshold, method, api_key, gemini_model = grading_sidebar()

st.title("🧪 Машины дүгнэлтийг шалгах")
st.caption(
    "Асуултын санд бүртгэсэн асуулт, жишиг хариултаа сонгоод туршилтын "
    "хариулт бичнэ. Машин ямар ойролцоо байдал тооцож, яагаад тухайн "
    "оноог өгснийг эндээс шалгана."
)

entries = [
    e for e in load_bank() if (e.get("question") or "").strip()
]

if not entries:
    st.warning(
        "Асуултын сан хоосон байна. Эхлээд үндсэн хуудасны **1️⃣ Асуултын сан** "
        "хэсэгт асуулт, жишиг хариултаа бүртгээд **Хадгалах** дарна уу."
    )
    st.stop()


def question_label(index: int, entry: dict) -> str:
    question = " ".join((entry.get("question") or "").split())
    if len(question) > 95:
        question = question[:92] + "…"
    max_score = entry.get("max_score", 1)
    return f"{index}. [{max_score} оноо] {question}"


def _example_score(example: dict) -> float:
    try:
        return float(example.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def render_result(result: ge.WrittenResult, entry: dict, answer: str) -> None:
    """Show the score and a transparent explanation of the decision."""
    max_score = result.max_score
    similarity_pct = result.similarity * 100
    threshold_pct = threshold * 100

    if result.score >= max_score:
        st.success(f"✅ Бүтэн оноо: **{ge.fmt_number(result.score)} / {max_score}**")
    elif result.score > 0:
        st.info(
            f"🟡 Хэсэгчилсэн оноо: **{ge.fmt_number(result.score)} / {max_score}**"
        )
    else:
        st.error(f"❌ Оноо өгөхгүй: **0 / {max_score}**")

    col_score, col_sim, col_threshold, col_parts = st.columns(4)
    col_score.metric("Оноо", f"{ge.fmt_number(result.score)} / {max_score}")
    col_sim.metric("Ойролцоо байдал", f"{similarity_pct:.0f}%")
    col_threshold.metric("Босго", f"{threshold_pct:.0f}%")
    col_parts.metric(
        "Таарсан хэсэг",
        f"{result.matched_parts}/{result.total_parts}"
        if result.total_parts > 1 else "—",
    )

    st.markdown(f"**Машины тайлбар:** {result.detail}")
    st.caption(
        f"Харьцуулах арга: **{METHOD_NAMES.get(method, method)}** · "
        f"Асуултын төрөл: `{entry.get('type', 'generic')}`"
    )

    if result.total_parts > 1:
        parts = ge.split_parts(entry.get("reference_answer", ""))
        rows = []
        for i, (part, sim) in enumerate(zip(parts, result.part_scores), start=1):
            passed = ge.passes(sim, threshold)
            rows.append({
                "№": i,
                "Жишиг хэсэг": part,
                "Ойролцоо байдал": f"{sim * 100:.0f}%",
                "Босго давсан": "✅ тийм" if passed else "❌ үгүй",
            })
        st.markdown("**Хэсэг тус бүрийн дүгнэлт**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Оноо = дээд оноо × босго давсан хэсгийн тоо ÷ нийт хэсгийн тоо "
            "(хамгийн ойр бүхэл тоо руу дугуйруулна)."
        )
    else:
        passed = ge.passes(result.similarity, threshold)
        comparison = "дээш" if passed else "доош"
        st.caption(
            f"Ойролцоо байдал {similarity_pct:.0f}% нь босго "
            f"{threshold_pct:.0f}%-аас {comparison} байгаа тул "
            f"**{ge.fmt_number(result.score)} оноо** өглөө."
        )

    alternatives = ge.reference_alternatives(entry)
    if result.total_parts <= 1 and len(alternatives) > 1:
        with st.expander(f"✅ Хүлээн авах боломжтой {len(alternatives)} хувилбар"):
            for alt in alternatives:
                st.markdown(f"- {alt}")

    if not answer.strip():
        st.caption("Хоосон хариулт тул 0 оноо өгсөн.")


# ---------------------------------------------------------------------------
# Selected question
# ---------------------------------------------------------------------------

st.subheader("1️⃣ Асуулт сонгох")

selected = st.selectbox(
    "Бүртгэсэн асуулт",
    options=list(range(len(entries))),
    format_func=lambda i: question_label(i + 1, entries[i]),
    key="preview_question",
)
entry = entries[selected]
try:
    max_score = max(1, int(entry.get("max_score", 1) or 1))
except (TypeError, ValueError):
    max_score = 1
reference = (entry.get("reference_answer") or "").strip()

info_col, ref_col = st.columns([3, 2])
with info_col:
    st.markdown(f"**Асуулт:** {entry.get('question', '')}")
    st.caption(
        f"Дээд оноо: **{max_score}** · Төрөл: `{entry.get('type', 'generic')}`"
    )
with ref_col:
    st.markdown("**Жишиг хариулт**")
    st.info(reference or "— (жишиг хариулт бүртгээгүй)")

examples = entry.get("examples") or []
if examples:
    with st.expander(f"📚 Сан дахь {len(examples)} жишээ хариулт"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Оноо": ge.fmt_number(_example_score(ex)),
                    "Хариулт": ex.get("answer", ""),
                }
                for ex in examples
            ]),
            width="stretch",
            hide_index=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Try an answer
# ---------------------------------------------------------------------------

st.subheader("2️⃣ Хариулт туршиж үзэх")

answer_key = f"preview_answer_{entry.get('id') or selected}"
answer = st.text_area(
    "Оюутны хариулт (туршилтын)",
    key=answer_key,
    height=150,
    placeholder="Оюутны бичсэн хариултыг энд оруулна уу…",
)

col_check, col_reference = st.columns([1, 1])
with col_check:
    check_clicked = st.button(
        "🔍 Дүгнэх",
        type="primary",
        width="stretch",
        disabled=not answer.strip(),
    )
with col_reference:
    reference_clicked = st.button(
        "🧾 Жишиг хариултаар шалгах",
        width="stretch",
        disabled=not reference,
    )

if check_clicked or reference_clicked:
    test_answer = reference if reference_clicked else answer
    if not test_answer.strip():
        st.warning("Эхлээд туршилтын хариулт оруулна уу.")
    else:
        scorer = create_scorer(method, api_key, gemini_model)
        with st.spinner("Машин дүгнэж байна…"):
            result = ge.grade_written(
                entry,
                test_answer,
                threshold,
                scorer,
                question=entry.get("question", ""),
            )
        if reference_clicked:
            st.caption("Туршсан хариулт: **жишиг хариулт**")
        render_result(result, entry, test_answer)
        if scorer.last_error and method != "local":
            st.warning(
                f"Gemini-ээс хариу авч чадсангүй ({scorer.last_error}). "
                "Локал аргаар дүгнэлээ."
            )

st.divider()

# ---------------------------------------------------------------------------
# Bank-wide self checks
# ---------------------------------------------------------------------------

st.subheader("3️⃣ Сангаа бүхэлд нь шалгах")
st.caption(
    "Эдгээр шалгалт нь жишиг хариулт, жишээ хариултууд болон босго "
    "тохиргоо хэр зөв ажиллаж байгааг илрүүлнэ."
)

with st.expander("🧾 Бүх асуултын жишиг хариулт бүтэн оноо авч байгаа эсэх"):
    st.caption(
        "Жишиг хариулт нь өөрийнхөө босгыг давж, бүтэн оноо авах ёстой. "
        "Авахгүй бол тухайн асуултын жишиг хариулт хэт богино эсвэл босго "
        "хэт өндөр байж болно."
    )
    if st.button("Жишиг хариултуудыг шалгах", key="check_all_references"):
        scorer = create_scorer(method, api_key, gemini_model)
        rows = []
        with st.spinner("Бүх жишиг хариултыг шалгаж байна…"):
            for i, item in enumerate(entries, start=1):
                ref = (item.get("reference_answer") or "").strip()
                result = ge.grade_written(
                    item,
                    ref,
                    threshold,
                    scorer,
                    question=item.get("question", ""),
                )
                full = result.score >= result.max_score
                rows.append({
                    "№": i,
                    "Асуулт": question_label(i, item),
                    "Оноо": f"{ge.fmt_number(result.score)}/{result.max_score}",
                    "Ойролцоо байдал": f"{result.similarity * 100:.0f}%",
                    "Дүгнэлт": "✅ бүтэн" if full else "⚠️ дутуу",
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        failed = [row for row in rows if row["Дүгнэлт"] != "✅ бүтэн"]
        if failed:
            st.warning(
                f"{len(failed)} асуултын жишиг хариулт бүтэн оноо авсангүй. "
                "Жишиг хариултаа тодорхой болгох эсвэл босгыг бууруулна уу."
            )
        else:
            st.success("Бүх жишиг хариулт бүтэн оноо авлаа.")
        if scorer.last_error and method != "local":
            st.warning(
                f"Gemini-ээс хариу авч чадсангүй ({scorer.last_error}). "
                "Локал аргаар дүгнэлээ."
            )

with st.expander("📋 Сан дахь жишээ хариултууд багшийн оноотой таарч байгаа эсэх"):
    st.caption(
        "Сан дахь жишээ бүрийг дахин дүгнэж, багшийн өмнө өгсөн оноотой "
        "харьцуулна. Зөрүү гарвал тухайн жишээ эсвэл жишиг хариултыг "
        "сайжруулах шаардлагатай."
    )
    if st.button("Жишээ хариултуудыг шалгах", key="check_all_examples"):
        scorer = create_scorer(method, api_key, gemini_model)
        rows = []
        with st.spinner("Жишээ хариултуудыг шалгаж байна…"):
            for i, item in enumerate(entries, start=1):
                for j, example in enumerate(item.get("examples") or [], start=1):
                    sample = (example.get("answer") or "").strip()
                    expected = _example_score(example)
                    result = ge.grade_written(
                        item,
                        sample,
                        threshold,
                        scorer,
                        question=item.get("question", ""),
                    )
                    matches = abs(result.score - expected) < 1e-9
                    rows.append({
                        "Асуулт": i,
                        "Жишээ": j,
                        "Багшийн оноо": ge.fmt_number(expected),
                        "Машины оноо": f"{ge.fmt_number(result.score)}/{result.max_score}",
                        "Ойролцоо байдал": f"{result.similarity * 100:.0f}%",
                        "Тохирсон": "✅" if matches else "⚠️ зөрүү",
                        "Хариулт": sample,
                    })
        if not rows:
            st.info("Санд жишээ хариулт бүртгэгдээгүй байна.")
        else:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            mismatches = [row for row in rows if row["Тохирсон"] != "✅"]
            if mismatches:
                st.warning(
                    f"{len(mismatches)} / {len(rows)} жишээ багшийн оноотой "
                    "таарсангүй."
                )
            else:
                st.success(f"Бүх {len(rows)} жишээ багшийн оноотой таарлаа.")
        if scorer.last_error and method != "local":
            st.warning(
                f"Gemini-ээс хариу авч чадсангүй ({scorer.last_error}). "
                "Локал аргаар дүгнэлээ."
            )
