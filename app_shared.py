#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the Streamlit pages.

The question bank is stored in ``reference_answers.json`` next to this file.
Both the main grading page and the grading-preview page use the helpers here,
so the threshold, comparison method and Gemini settings stay consistent.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import grading_engine as ge

BASE = Path(__file__).resolve().parent
BANK_PATH = BASE / "reference_answers.json"
BACKUP_PATH = BASE / "reference_answers.backup.json"
CACHE_PATH = BASE / ".semantic_cache.json"

METHOD_LABELS = {
    "Локал (TF-IDF)": "local",
    "Gemini (утгын)": "gemini",
    "Хосолмол (дундаж)": "combined",
}
METHOD_NAMES = {value: key for key, value in METHOD_LABELS.items()}


# ---------------------------------------------------------------------------
# Question bank persistence
# ---------------------------------------------------------------------------

def load_bank() -> list[dict]:
    if not BANK_PATH.exists():
        return []
    try:
        data = json.loads(BANK_PATH.read_text(encoding="utf-8"))
        return list(data.get("questions", []))
    except Exception as exc:
        st.error(f"`reference_answers.json` уншиж чадсангүй: {exc}")
        return []


def save_bank(entries: list[dict]) -> None:
    if BANK_PATH.exists():
        try:
            shutil.copy2(BANK_PATH, BACKUP_PATH)
        except Exception:
            pass
    payload = {
        "_comment": (
            "Багшийн бэлдсэн асуулт + жишиг хариултын сан. "
            "Streamlit app автоматаар удирдана."
        ),
        "questions": entries,
    }
    tmp = BANK_PATH.with_name(BANK_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(BANK_PATH)


def merge_edited(df: pd.DataFrame, old_entries: list[dict]) -> list[dict]:
    """Turn the data-editor rows back into question-bank entries."""
    old_by_question = {
        ge.normalize(e.get("question", "")): e for e in old_entries
    }
    out: list[dict] = []
    for _, row in df.iterrows():
        question = str(row.get("Асуулт", "") or "").strip()
        if not question:
            continue
        reference = str(row.get("Жишиг хариулт", "") or "").strip()
        try:
            max_score = max(1, int(row.get("Оноо", 1)))
        except Exception:
            max_score = 1
        entry = dict(old_by_question.get(ge.normalize(question), {}))
        entry.setdefault("id", f"q-{len(out) + 1}")
        entry.setdefault("type", "generic")
        entry.setdefault("examples", [])
        entry["question"] = question
        entry["reference_answer"] = reference
        entry["max_score"] = max_score
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Sidebar + scorer
# ---------------------------------------------------------------------------

def grading_sidebar() -> tuple[float, str, Optional[str], str]:
    """Render the shared grading settings and return the effective values.

    Returns ``(threshold, method, api_key, gemini_model)`` where ``threshold``
    is in the 0..1 range used by :mod:`grading_engine`.
    """
    st.sidebar.header("⚙️ Тохиргоо")

    threshold_pct = st.sidebar.slider(
        "Зөв гэж тооцох ойролцоо байдлын босго (%)",
        min_value=0,
        max_value=100,
        value=65,
        step=5,
        key="threshold_pct",
        help=(
            "Оюутны хариулт багшийн жишиг хариулттай энэ хувьцаас илүү "
            "ойролцоо байвал зөв гэж тооцно. 2+ оноотой асуултад хэсэг "
            "тус бүрээр энэ босгыг хэрэглэнэ."
        ),
    )

    method_label = st.sidebar.selectbox(
        "Харьцуулах арга",
        list(METHOD_LABELS),
        index=0,
        key="method_label",
        help=(
            "Локал: интернэтгүй ажиллана. Gemini: утгын ойролцоо байдлыг "
            "AI-аар үнэлнэ (API key шаардлагатай). Хосолмол: хоёрын дундаж."
        ),
    )
    method = METHOD_LABELS[method_label]

    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    )

    gemini_model = "gemini-2.5-flash"
    if method in {"gemini", "combined"}:
        if not api_key:
            api_key = st.sidebar.text_input(
                "Gemini API key", type="password", key="gemini_key_input"
            )
        gemini_model = st.sidebar.text_input(
            "Gemini model", value="gemini-2.5-flash", key="gemini_model_input"
        )
        if not api_key:
            st.sidebar.warning(
                "API key олдсонгүй тул **Локал** арга ашиглана."
            )
            method = "local"
        else:
            st.sidebar.caption(
                "Gemini үнэгүй хязгаарт хүрвэл автоматаар локал арга руу шилжинэ."
            )

    st.sidebar.divider()
    st.sidebar.caption(
        "Босго бага → илүү эвлэрүүлж дүгнэнэ. "
        "Босго өндөр → зөвхөн маш ойролцоо хариултыг зөв гэнэ."
    )
    return threshold_pct / 100.0, method, api_key or None, gemini_model


def create_scorer(method: str, api_key: Optional[str], model: str) -> ge.SemanticScorer:
    """Build a :class:`grading_engine.SemanticScorer` with the shared cache."""
    return ge.SemanticScorer(
        mode=method,
        api_key=api_key,
        model=model,
        cache_path=CACHE_PATH,
    )
