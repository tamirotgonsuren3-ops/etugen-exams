#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit app for grading Mongolian written-answer exams.

Run with:

    cd /home/bumblebee/etugen
    streamlit run streamlit_app.py

The teacher can:
  * add / remove written questions and give a reference answer,
  * set the similarity threshold with a slider,
  * upload student PDFs and press Check,
  * download the result as a CSV file.

The separate ``pages/1_Дүгнэлт_шалгах.py`` page lets the teacher try an
answer against any registered question and see how the machine scored it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import grade_exams as gx
import grading_engine as ge
from app_shared import (
    BANK_PATH,
    create_scorer,
    grading_sidebar,
    load_bank,
    merge_edited,
    save_bank,
)

st.set_page_config(
    page_title="Монгол хэл — шалгалт засах",
    page_icon="📝",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_one_exam(exam, entries, matcher, scorer, threshold, details):
    for num in sorted(exam.questions):
        q = exam.questions[num]
        if not q.is_written:
            continue

        entry, question_sim = matcher.best(q.prompt)
        if entry is None or question_sim < 0.30:
            q.score = 0.0
            q.detail = f"Асуулт санд танигдсангүй (qsim={question_sim:.2f})"
            details.append({
                "Оюутан": exam.student_id,
                "Асуулт": f"Q{num}",
                "Оноо": f"0/{q.max_score}",
                "Ойролцоо байдал": "—",
                "Тайлбар": q.detail,
                "Оюутны хариулт": q.written_answer,
            })
            continue

        q.max_score = int(entry.get("max_score", q.max_score) or q.max_score)
        result = ge.grade_written(
            entry, q.written_answer, threshold, scorer, question=q.prompt
        )
        q.score = float(result.score)
        q.detail = result.detail
        details.append({
            "Оюутан": exam.student_id,
            "Асуулт": f"Q{num}",
            "Оноо": f"{ge.fmt_number(result.score)}/{result.max_score}",
            "Ойролцоо байдал": f"{result.similarity * 100:.0f}%",
            "Хэсэг": (
                f"{result.matched_parts}/{result.total_parts}"
                if result.total_parts > 1 else "—"
            ),
            "Тайлбар": result.detail,
            "Оюутны хариулт": q.written_answer,
            "Жишиг хариулт": entry.get("reference_answer", ""),
        })


def render_results(exams, details):
    all_nums = sorted({n for exam in exams for n in exam.questions}, reverse=True)
    rows = []
    for i, exam in enumerate(exams, start=1):
        row = {
            "num": i,
            "Оюутны код": exam.student_id,
            "Нийт оноо": ge.fmt_number(exam.total),
        }
        for n in all_nums:
            q = exam.questions.get(n)
            row[str(n)] = ge.fmt_number(q.score) if q else ""
        rows.append(row)
    df = pd.DataFrame(rows)

    st.subheader("📊 Дүн")
    st.dataframe(df, width="stretch", hide_index=True)

    csv_bytes = ge.build_csv(exams)
    st.download_button(
        "⬇️ CSV татах",
        data=csv_bytes,
        file_name="mongol-hel.csv",
        mime="text/csv",
        type="primary",
    )

    st.subheader("🔎 Оюутан тус бүрийн дэлгэрэнгүй")
    for exam in exams:
        st.markdown(
            f"**{exam.student_id}** — авто {ge.fmt_number(exam.objective_score)} + "
            f"бичгийн {ge.fmt_number(exam.written_score)} = "
            f"**{ge.fmt_number(exam.total)}**"
        )
        if exam.header_total is not None:
            st.caption(
                f"PDF-д бүртгэсэн авто оноо: {ge.fmt_number(exam.header_total)}"
            )
        student_details = [d for d in details if d["Оюутан"] == exam.student_id]
        if student_details:
            st.dataframe(
                pd.DataFrame(student_details),
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

threshold, method, api_key, gemini_model = grading_sidebar()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📝 Монгол хэл — шалгалт засах")
st.caption(
    "Багш асуулт, жишиг хариултаа оруулаад оюутны PDF-үүдийг upload хийж "
    "Check дарна. Систем бүх шалгалтыг шалгаж CSV үүсгэнэ."
)

# Load bank once into session state.
if "bank" not in st.session_state:
    st.session_state.bank = load_bank()
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0

# ---------------------------------------------------------------------------
# 1. Question bank
# ---------------------------------------------------------------------------

st.header("1️⃣ Асуултын сан")

flash = st.session_state.pop("bank_flash", None)
if flash:
    st.success(flash)

st.info(
    "Асуулт нэмэх: хүснэгтийн доод мөрөнд шинэ асуулт бичнэ. "
    "Устгах: мөрийн зүүн талын ✓-г авч, Delete дарна. "
    "Олон зөв хувилбар байвал **|** тэмдгээр тусгаарлана. "
    "2+ оноотой асуултын хэсгүүдийг **1. ... 2. ...** эсвэл **|**-ээр тусгаарлана."
)

bank_df = pd.DataFrame(
    [
        {
            "Асуулт": e.get("question", ""),
            "Жишиг хариулт": e.get("reference_answer", ""),
            "Оноо": int(e.get("max_score", 1) or 1),
        }
        for e in st.session_state.bank
    ],
    columns=["Асуулт", "Жишиг хариулт", "Оноо"],
)

edited_df = st.data_editor(
    bank_df,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    key=f"question_editor_{st.session_state.editor_version}",
    column_config={
        "Асуулт": st.column_config.TextColumn(
            "Асуулт", width="large", required=True
        ),
        "Жишиг хариулт": st.column_config.TextColumn(
            "Жишиг хариулт (| -ээр олон хувилбар)",
            width="large",
        ),
        "Оноо": st.column_config.NumberColumn(
            "Оноо", min_value=1, max_value=10, step=1, default=1
        ),
    },
)

col_save, col_reload, col_info = st.columns([1, 1, 3])
with col_save:
    if st.button("💾 Хадгалах", width="stretch"):
        entries = merge_edited(edited_df, st.session_state.bank)
        save_bank(entries)
        st.session_state.bank = entries
        st.success(f"{len(entries)} асуулт хадгагдлаа.")
with col_reload:
    if st.button("↩️ Файлаас сэргээх", width="stretch"):
        st.session_state.bank = load_bank()
        st.success("`reference_answers.json`-оос дахин ачааллаа.")
with col_info:
    st.caption(
        f"Одоогийн сан: **{len(st.session_state.bank)}** асуулт · "
        f"Файл: `{BANK_PATH.name}`"
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Upload and check
# ---------------------------------------------------------------------------

st.header("2️⃣ Оюутны шалгалт шалгах")

uploaded = st.file_uploader(
    "Оюутны PDF файлуудыг сонгоно уу (нэг эсвэл олон файл)",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

col_check, col_note = st.columns([1, 3])
with col_check:
    check_clicked = st.button(
        "✅ Check — бүгдийг шалгах",
        type="primary",
        disabled=not uploaded,
        width="stretch",
    )
with col_note:
    if uploaded:
        st.caption(f"{len(uploaded)} файл сонгогдсон.")
    else:
        st.caption("Эхлээд PDF файл сонгоно уу.")

if check_clicked and uploaded:
    # Use the current editor content, even if the Save button was not pressed.
    entries = merge_edited(edited_df, st.session_state.bank)
    if not entries:
        st.error("Асуултын сан хоосон байна. Эхлээд асуулт, жишиг хариулт оруулна уу.")
    else:
        save_bank(entries)
        st.session_state.bank = entries

        scorer = create_scorer(method, api_key, gemini_model)
        matcher = ge.QuestionMatcher(entries)

        exams = []
        details: list[dict] = []
        errors: list[str] = []
        progress = st.progress(0.0, text="Эхэлж байна...")

        tmpdir = Path(tempfile.mkdtemp(prefix="exam_upload_"))
        try:
            for i, file in enumerate(uploaded):
                progress.progress(
                    (i) / max(1, len(uploaded)),
                    text=f"Уншиж байна: {file.name}",
                )
                pdf_path = tmpdir / file.name
                pdf_path.write_bytes(file.getbuffer())
                try:
                    exam = gx.parse_exam(pdf_path, with_objective=True)
                except Exception as exc:
                    errors.append(f"{file.name}: {exc}")
                    continue
                grade_one_exam(
                    exam, entries, matcher, scorer, threshold, details
                )
                exams.append(exam)
                progress.progress(
                    (i + 1) / max(1, len(uploaded)),
                    text=f"{file.name} — дууслаа",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        progress.progress(1.0, text="Дууслаа.")

        if scorer.last_error and method != "local":
            st.warning(
                f"Gemini-ээс хариу авч чадсангүй ({scorer.last_error}). "
                "Хариултуудыг локал аргаар дүгнэлээ."
            )
        for err in errors:
            st.error(f"Уншиж чадсангүй — {err}")

        if not exams:
            st.error("Ямар ч шалгалт уншигдсангүй.")
        else:
            st.success(f"{len(exams)} шалгалт амжилттай шалгагдлаа.")
            unmatched = [
                d for d in details if "танигдсангүй" in d.get("Тайлбар", "")
            ]
            if unmatched:
                with st.expander(
                    f"⚠️ {len(unmatched)} бичгийн асуулт асуултын санд танигдсангүй"
                ):
                    st.dataframe(
                        pd.DataFrame(unmatched)[
                            ["Оюутан", "Асуулт", "Тайлбар", "Оюутны хариулт"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Тухайн асуултыг дээрх санд нэмээд дахин Check дарна уу."
                    )

            st.session_state["last_exams"] = exams
            st.session_state["last_details"] = details
            render_results(exams, details)

# Re-show previous results after a normal Streamlit rerun.
elif "last_exams" in st.session_state:
    st.divider()
    st.caption("Өмнөх шалгалтын үр дүн (дахин шалгах бол Check дарна уу).")
    render_results(
        st.session_state["last_exams"], st.session_state.get("last_details", [])
    )

# ---------------------------------------------------------------------------
# Footer help
# ---------------------------------------------------------------------------

with st.expander("ℹ️ Хэрхэн ашиглах вэ?"):
    st.markdown(
        """
1. **Асуултын сан** хэсэгт багш өөрөө бичиж хариулах асуултуудыг оруулна.
   - `Асуулт`: оюутны PDF доторх асуултын тексттэй ижил байх тусам сайн танигдана.
   - `Жишиг хариулт`: багшийн зөв гэж үзсэн хариулт.
     Олон зөв хувилбар байвал `|` тэмдгээр тусгаарлана.
   - `Оноо`: тухайн асуултын дээд оноо.
2. **Босго** slider-ээр хариулт хэдэн хувиар ойролцоо байвал зөв болохыг тохируулна.
3. Оюутны **PDF** файлуудыг upload хийж **Check** дарна.
4. Үр дүн болон **CSV татах** товч гарч ирнэ.
5. Асуулт, жишиг хариултаа бүртгэсний дараа sidebar-аас
   **🧪 Дүгнэлт шалгах** хуудсыг нээж, туршилтын хариултаар машин хэрхэн
   дүгнэж байгааг шалгана.
6. Хэрэв зарим асуулт танигдахгүй бол тухайн асуултын текстийг санд байгаа
   тексттэй ойролцоо болгож засаад дахин Check дарна.

**2+ оноотой асуулт:** жишиг хариултыг хэсэг болгон бичнэ, жишээ нь:

```
1. Хөтөвч – бие засахад зориулсан сав.
2. Хатавч – гэрийн хаалганы хоёр хажуугийн хана.
3. Хонгиносон – нарийн, тод, цуурайтсан дуу авиа.
```

Систем хэсэг тус бүрийг тусад нь харьцуулж, таарсан хэсгийн хувьд оноо өгнө.
        """
    )
