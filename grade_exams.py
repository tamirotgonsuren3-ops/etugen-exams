#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mongolian written-answer exam grader
====================================

This script reads exam PDFs exported by the "Багшийн вэб" system, extracts

  * the automatically graded multiple-choice score of every question,
  * the student's own written answer, and
  * the teacher's score, when the PDF has already been graded,

then grades the written questions and writes one row per student to a CSV file.

The written-answer grader is a hybrid:

  1. A transparent rubric (keywords / required ideas) is used as the main
     judge.  This is what makes partial credit for the 2- and 3-point
     questions reliable.
  2. A small machine-learning model is trained on the teacher's question +
     reference-answer pairs (`reference_answers.json`) and, optionally, on
     the already teacher-graded answers found in `sain-4-training/`.
     The model is a character n-gram TF-IDF k-nearest-neighbour model.  It is
     used as a second opinion and as the fallback when the rubric cannot
     decide.

How the teacher can improve the model
-------------------------------------
Edit `reference_answers.json` and add more `{ "answer": "...", "score": N }`
examples under the matching question.  The next run retrains automatically.
No code changes are needed.  This is the "teacher writes the question and the
answer, then the machine learns from that pair" workflow.

Examples
--------
    python3 grade_exams.py
    python3 grade_exams.py --test-dir muu-4-test --csv mongol-hel.csv
    python3 grade_exams.py --learn-from-training
    python3 grade_exams.py --ml-only --debug

Requirements
------------
    poppler-utils  (pdftotext)
    pypdf          (pip install pypdf)
    scikit-learn   (optional, for the ML part; a pure-Python fallback exists)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    """Lower-case and collapse whitespace, keeping Cyrillic letters."""
    s = (s or "").replace("\u00a0", " ").replace("\uf0c7", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def has_any(text: str, needles: Iterable[str]) -> bool:
    t = norm(text)
    return any(n in t for n in needles)


def run_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, errors="replace")


def pdf_page_count(path: Path) -> int:
    if PdfReader is not None:
        try:
            return len(PdfReader(str(path)).pages)
        except Exception:
            pass
    out = run_text(["pdfinfo", str(path)])
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


def layout_text(path: Path) -> str:
    return run_text(["pdftotext", "-layout", str(path), "-"])


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Option:
    text: str
    score: int
    selected: bool = False


@dataclass
class Question:
    num: int
    prompt: str
    options: list[Option] = field(default_factory=list)
    written_answer: str = ""
    qtype: str = "objective"
    max_score: int = 1
    score: float = 0.0
    teacher_score: Optional[float] = None
    detail: str = ""

    @property
    def is_written(self) -> bool:
        return self.qtype != "objective"


@dataclass
class Exam:
    student_id: str
    source: Path
    header_total: Optional[float]
    questions: dict[int, Question] = field(default_factory=dict)

    @property
    def objective_score(self) -> float:
        return sum(q.score for q in self.questions.values() if not q.is_written)

    @property
    def written_score(self) -> float:
        return sum(q.score for q in self.questions.values() if q.is_written)

    @property
    def total(self) -> float:
        return self.objective_score + self.written_score

    @property
    def max_total(self) -> float:
        return sum(q.max_score for q in self.questions.values())


# ---------------------------------------------------------------------------
# PDF text parsing
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r"^\s*(\d{1,2})\s{2,}\S")
SOLO_NUMBER_RE = re.compile(r"^\s{0,3}(\d{1,2})\s*$")
OPTION_RE = re.compile(r"^(.*?)\s{2,}(\d+)\s+оноотой\s*$")
END_MARKERS = ("Оноог хадгалах", "Таны өгсөн оноо")

# Lines that end the question prompt and are followed by the answer.
ANSWER_START_RE = re.compile(
    r"(Оноог хадгалах|Таны өгсөн оноо"
    r"|\?|бичнэ үү|бичээрэй|тодорхойл|зохионо уу"
    r"|харгалзуулаарай|сонгоорой|нөхөөрэй"
    r"|\(Д\.\s*Нацагдорж|асуултад дэлгэрэнгүй бичиж хариулна уу"
    r"|Д\.\s*Батбаяр|Эхийг дараах даалгаврын)"
)
DOTS_ONLY_RE = re.compile(r"^[\s.·…_\-–—]*$")
VOCAB_TEMPLATE_RE = re.compile(r"^\s*Хонгиносон\s*[-–—]?\s*[.·…]*\s*$")


def split_questions(text: str) -> list[tuple[int, list[str]]]:
    """Return [(question_number, segment_lines), ...] in document order."""
    lines = text.replace("\f", "\n").split("\n")
    heads: list[tuple[int, int]] = []
    expected = 1
    for i, line in enumerate(lines):
        name: Optional[int] = None
        m = HEADER_RE.match(line)
        if m:
            name = int(m.group(1))
        else:
            m = SOLO_NUMBER_RE.match(line)
            if m:
                name = int(m.group(1))
        if name is None or not (1 <= name <= 99):
            continue
        if not heads:
            if name == 1:
                heads.append((name, i))
                expected = 2
        elif name == expected:
            heads.append((name, i))
            expected += 1

    out: list[tuple[int, list[str]]] = []
    for idx, (num, start) in enumerate(heads):
        end = heads[idx + 1][1] if idx + 1 < len(heads) else len(lines)
        out.append((num, lines[start:end]))
    return out


def extract_written_answer(segment: list[str]) -> str:
    """Cut the student's answer out of a question segment."""
    end = len(segment)
    for i, line in enumerate(segment):
        if any(m in line for m in END_MARKERS):
            end = i
            break

    start = 0

    # Vocabulary questions have a fixed three-line template.  The last
    # template line is the safest place to start.
    for i in range(end):
        if VOCAB_TEMPLATE_RE.match(segment[i]):
            start = i + 1

    if start == 0:
        for i in range(end):
            line = segment[i]
            if ANSWER_START_RE.search(line) and not DOTS_ONLY_RE.match(line):
                start = i + 1

    answer: list[str] = []
    for line in segment[start:end]:
        t = line.strip()
        if not t or DOTS_ONLY_RE.match(t):
            continue
        if any(m in t for m in END_MARKERS):
            continue
        # Garbled template rows produced by the PDF export contain many
        # consecutive spaces.  They are not the student's answer.
        if re.search(r"\s{4,}", t):
            continue
        if re.match(r"^\d{1,2}\s*[“\"(]", t) and len(t) < 60:
            continue
        answer.append(t)
    return " ".join(answer).strip()


def infer_qtype(prompt: str) -> str:
    p = norm(prompt)
    if "ойролцоо утгатай" in p:
        return "synonym"
    if "хэв маягийг тодорхойл" in p:
        return "style"
    if "өгүүлбэр зохионо уу" in p:
        return "idiom_sentence"
    if "утгыг тайлбарлаж бичнэ үү" in p:
        return "idiom_meaning"
    if "батбаяр" in p and "алтай" in p:
        return "essay"
    if "тайлбарлан бичээрэй" in p or "хөтөвч" in p:
        return "vocab"
    return "generic"


def infer_max_score(prompt: str, qtype: str) -> int:
    m = re.search(r"[\[(](\d+)\s*оноо[^\])]*[\])]", prompt)
    if m:
        return int(m.group(1))
    return {"essay": 3, "vocab": 2}.get(qtype, 1)


def parse_questions(text: str) -> dict[int, Question]:
    questions: dict[int, Question] = {}
    for num, segment in split_questions(text):
        options: list[Option] = []
        option_line_indices: set[int] = set()
        for i, line in enumerate(segment):
            m = OPTION_RE.match(line)
            if m:
                options.append(Option(text=m.group(1).strip(), score=int(m.group(2))))
                option_line_indices.add(i)

        prompt_lines = [
            line.strip()
            for i, line in enumerate(segment)
            if i not in option_line_indices
        ]
        prompt = " ".join(prompt_lines)
        # Drop the leading question number from the prompt.
        prompt = re.sub(rf"^\s*{num}\s+", "", prompt).strip()

        qtype = infer_qtype(prompt)
        max_score = infer_max_score(prompt, qtype) if qtype != "objective" else (
            max((o.score for o in options), default=1) or 1
        )

        if options:
            qtype = "objective"
            max_score = max((o.score for o in options), default=1) or 1
            written_answer = ""
            detail = ""
        else:
            written_answer = extract_written_answer(segment)
            max_score = infer_max_score(prompt, qtype)
            detail = ""

        teacher_score = None
        tm = re.search(r"Таны өгсөн оноо:\s*(\d+(?:\.\d+)?)", "\n".join(segment))
        if tm:
            teacher_score = float(tm.group(1))

        questions[num] = Question(
            num=num,
            prompt=prompt,
            options=options,
            written_answer=written_answer,
            qtype=qtype,
            max_score=max_score,
            teacher_score=teacher_score,
            detail=detail,
        )
    return questions


# ---------------------------------------------------------------------------
# Detect which multiple-choice option the student selected.
#
# The web app draws a small blue filled circle inside the selected radio
# button.  The circle is a normal PDF path, so it can be found in the page
# content stream even though pdftotext does not see it.
# ---------------------------------------------------------------------------

BLUE_FILL_RE = re.compile(r"0 \.4588 1 rg")
ANY_FILL_RE = re.compile(r"([\d.]+) ([\d.]+) ([\d.]+) rg")


def _path_center(chunk: str) -> Optional[tuple[float, float, float, float]]:
    """Return (cx, cy, w, h) of the first filled path in `chunk`."""
    pm = re.search(r"\n(?:h\s*\n)?(f|f\*|F)\s", chunk)
    path = chunk[: pm.start()] if pm else chunk
    pts = [
        (float(a), float(b))
        for a, b in re.findall(r"([-\d.]+) ([-\d.]+) (?:m|l|c)", path)
    ]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2,
            max(xs) - min(xs), max(ys) - min(ys))


def _dot_top_y(data: str, cy: float) -> float:
    """Convert a radio-dot centre from content space to page-top space.

    The exported PDFs use:
        .24 0 0 -.24 0 841.92 cm
        3.125 0 0 3.125 59.375 OFFSET cm
    so the page-top coordinate of a point is ``0.24 * (3.125 * cy + OFFSET)``.
    """
    m = re.search(r"3\.125 0 0 3\.125 ([-.\d]+) ([-.\d]+) cm", data)
    offset = float(m.group(2)) if m else 59.375
    return 0.24 * (3.125 * cy + offset)


def selected_dot_ys(path: Path, page_index: int) -> tuple[list[float], float]:
    """Return (top-y coordinates of selected radio dots, page height)."""
    if PdfReader is None:
        return [], 841.92
    try:
        data = PdfReader(str(path)).pages[page_index].get_contents().get_data().decode(
            "latin-1", "replace"
        )
    except Exception:
        return [], 841.92

    page_height = 841.92

    ys: list[float] = []
    for m in BLUE_FILL_RE.finditer(data):
        chunk = data[m.end(): m.end() + 20000]
        c = _path_center(chunk)
        if not c:
            continue
        cx, cy, w, h = c
        if w > 30 or h > 30:
            continue
        ys.append(_dot_top_y(data, cy))
    return ys, page_height


def bbox_lines(path: Path, page_index: int) -> list[tuple[float, float, str]]:
    html = run_text([
        "pdftotext", "-f", str(page_index + 1), "-l", str(page_index + 1),
        "-bbox-layout", str(path), "-",
    ])
    out: list[tuple[float, float, str]] = []
    for xmin, ymin, xmax, ymax, body in re.findall(
        r'<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>',
        html, re.S,
    ):
        text = " ".join(re.findall(r">([^<]*)</word>", body)).strip()
        out.append((float(ymin), float(xmin), text))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def detect_selections(path: Path) -> dict[int, list[bool]]:
    """Return {question_number: [selected per option]} from the PDF dots."""
    n_pages = pdf_page_count(path)
    events: list[tuple[int, float, str, int]] = []
    dots: list[tuple[int, float]] = []

    for p in range(n_pages):
        for y, x, text in bbox_lines(path, p):
            if 30 <= x <= 37 and text.isdigit() and 1 <= int(text) <= 99:
                events.append((p, y, "header", int(text)))
            elif re.match(r"^\d+ оноотой$", text):
                events.append((p, y, "option", int(text.split()[0])))
        page_dots, _ = selected_dot_ys(path, p)
        dots.extend((p, y) for y in page_dots)

    events.sort(key=lambda e: (e[0], e[1]))

    selected: set[tuple[int, float]] = set()
    for p, top in dots:
        candidates = [
            e for e in events
            if e[0] == p and e[2] == "option" and abs(e[1] - top) < 5.0
        ]
        if candidates:
            best = min(candidates, key=lambda e: abs(e[1] - top))
            selected.add((best[0], best[1]))

    result: dict[int, list[bool]] = {}
    current: Optional[int] = None
    for page, y, kind, value in events:
        if kind == "header":
            current = value
            result.setdefault(current, [])
        elif kind == "option" and current is not None:
            result[current].append((page, y) in selected)
    return result


# ---------------------------------------------------------------------------
# Rubric grading
# ---------------------------------------------------------------------------

class RubricGrader:
    """Keyword / idea based grader calibrated on the teacher's examples."""

    def grade(self, q: Question) -> tuple[float, str]:
        text = q.written_answer
        if not text.strip():
            return 0.0, "хоосон хариулт"
        fn = getattr(self, f"_grade_{q.qtype}", self._grade_generic)
        return fn(text, q.max_score)

    # -- 1 point questions -------------------------------------------------

    def _grade_synonym(self, text: str, max_score: int) -> tuple[float, str]:
        hits = [w for w in ("өндий", "бос", "сүмб", "сүнд", "сүрлэг",
                            "өргөгд", "өндөр", "орших") if w in norm(text)]
        if hits:
            return float(max_score), "ойролцоо утга: " + ", ".join(hits)
        return 0.0, "ойролцоо утга танигдсангүй"

    def _grade_style(self, text: str, max_score: int) -> tuple[float, str]:
        t = norm(text)
        if any(w in t for w in ("дүрслэх", "дүрслэмж", "дүрсэл", "уран зохиол")):
            return float(max_score), "дүрслэх / уран зохиолын хэв маяг"
        return 0.0, "хэв маяг танигдсангүй"

    def _grade_idiom_meaning(self, text: str, max_score: int) -> tuple[float, str]:
        t = norm(text)
        hits = [w for w in ("дуулгавар", "захирагд", "үгэнд", "тушаал",
                            "хэлсэн", "биелүүл", "эсэргүүц") if w in t]
        if hits:
            return float(max_score), "утга: " + ", ".join(hits)
        return 0.0, "хэлц үгийн утга танигдсангүй"

    def _grade_idiom_sentence(self, text: str, max_score: int) -> tuple[float, str]:
        t = norm(text)
        if "морь нохой мэт зүтгэ" not in t:
            return 0.0, "хэлц үг өгүүлбэрт ороогүй"
        # A real sentence, not just a definition.
        if len(t.split()) >= 5 and any(
            w in t for w in ("зүтгэ", "ажилла", "хөдөлмөрлө", "тэжээ", "хий")
        ):
            return float(max_score), "хэлц үг зөв өгүүлбэрт орсон"
        return 0.0, "өгүүлбэр хэт богино / тодорхойгүй"

    # -- multi-point questions --------------------------------------------

    def _grade_essay(self, text: str, max_score: int) -> tuple[float, str]:
        t = norm(text)
        parts: list[str] = []
        score = 0

        # (1) Why the mother stopped looking after the household.
        mother = (
            ("аав" in t and any(w in t for w in ("морь", "уна", "ойч", "гэмт", "хэвтэр", "өвчт")))
            or ("ээж" in t and "мал" in t and any(w in t for w in ("гэрийн бараа", "гэрийн ажил", "орхи")))
        )
        if mother:
            score += 1
            parts.append("1✓")

        # (2) Why the boy hesitated to go to the stone toy.
        toy = (
            "тоглоом" in t
            and any(w in t for w in ("аав", "дууд", "хүлээ", "чагна", "тээнэгэлз", "санагд"))
        )
        if toy:
            score += 1
            parts.append("2✓")

        # (3) Why he was not named Baatar / why he was named Altai.
        name = (
            "нэр" in t
            and any(w in t for w in ("алтай", "уул", "нутаг", "алт"))
        )
        if name:
            score += 1
            parts.append("3✓")

        score = min(score, max_score)
        if score == 0:
            return 0.0, "гурван учирт хариу танигдсангүй"
        return float(score), "зөв дэд хариулт: " + ", ".join(parts)

    def _grade_vocab(self, text: str, max_score: int) -> tuple[float, str]:
        t = norm(text)
        # Split the answer into the three definitions.  Students use either
        # "1. ... 2. ... 3. ..." or "Хөтөвч- ... Хатавч- ...".
        segments: dict[str, str] = {}
        patterns = {
            "khotovch": r"(?:хөтөвч|хотовч)\s*[-–—:.]?\s*(.*?)(?=(?:хатавч|хонгиносон|\d\s*[.)]|$))",
            "khatavch": r"хатавч\s*[-–—:.]?\s*(.*?)(?=(?:хөтөвч|хотовч|хонгиносон|\d\s*[.)]|$))",
            "khonginoson": r"хонгиносон\s*[-–—:.]?\s*(.*)$",
        }
        # Simpler: split on numbering first.
        num_parts = re.split(r"(?=\b\d\s*[.)]\s*)", text)
        if len([p for p in num_parts if p.strip()]) >= 2:
            joined = " ".join(num_parts)
            for key, pat in patterns.items():
                m = re.search(pat, norm(joined), re.S)
                if m:
                    segments[key] = m.group(1)
        else:
            for key, pat in patterns.items():
                m = re.search(pat, t, re.S)
                if m:
                    segments[key] = m.group(1)

        def ok_khotovch(s: str) -> bool:
            return ("сав" in s or "суулга" in s) and any(
                w in s for w in ("бие засах", "шээс", "баах", "хэвтэр", "хүүхэд", "өвчт")
            )

        def ok_khatavch(s: str) -> bool:
            return "хаалга" in s and any(
                w in s for w in ("хана", "хажуу", "мод", "уулзвар", "хүрээ", "тал")
            )

        def ok_khonginoson(s: str) -> bool:
            return any(w in s for w in ("дуу", "чимээ", "авиа", "аялгуу")) and any(
                w in s for w in ("нарийн", "жингэн", "цуурай", "чанга", "тод", "хөндий", "сул")
            )

        checks = {
            "Хөтөвч": ok_khotovch(segments.get("khotovch", "")),
            "Хатавч": ok_khatavch(segments.get("khatavch", "")),
            "Хонгиносон": ok_khonginoson(segments.get("khonginoson", "")),
        }
        correct = sum(checks.values())
        if correct == 0:
            return 0.0, "гурван үгийн утга танигдсангүй"
        if correct == 3:
            score = float(max_score)
        else:
            score = 1.0 if max_score >= 2 else 0.0
        detail = ", ".join(f"{k}{'✓' if v else '✗'}" for k, v in checks.items())
        return score, detail

    def _grade_generic(self, text: str, max_score: int) -> tuple[float, str]:
        # An unknown question cannot be graded by keywords.  The
        # question+answer model must decide; if it cannot, stay conservative.
        if len(norm(text).split()) < 3:
            return 0.0, "хариулт хэт богино"
        return 0.0, "жишиг хариулттай тулгах шаардлагатай"


# ---------------------------------------------------------------------------
# Reference-answer ML model (char n-gram TF-IDF k-NN)
# ---------------------------------------------------------------------------

class ReferenceModel:
    """k-NN over teacher-written question + answer pairs.

    The teacher bank (`reference_answers.json`) stores, for every written
    question, the question text, a reference answer and labelled student
    examples.  Two models are combined:

    * a question matcher (char n-gram TF-IDF) finds which teacher question a
      student's question belongs to, so brand-new teacher questions are
      recognised without code changes;
    * an answer grader (k-NN over the teacher's labelled answers) predicts
      the score for the student's answer.

    Character n-grams are used instead of words because Mongolian is
    agglutinative; this makes the model tolerant of suffixes and small
    spelling differences.
    """

    def __init__(self) -> None:
        # qtype -> [(normalised answer, score, max_score)]
        self.examples: dict[str, list[tuple[str, float, int]]] = {}
        # Teacher questions, each with its own labelled answers.
        self.questions: list[dict] = []
        self._vectorizers: dict[str, object] = {}
        self._matrices: dict[str, object] = {}
        self._fitted: set[str] = set()
        self._q_vec = None
        self._q_mat = None

    def add(self, qtype: str, answer: str, score: float, max_score: int) -> None:
        answer = norm(answer)
        if not answer:
            return
        self.examples.setdefault(qtype, []).append((answer, float(score), max_score))
        self._fitted.discard(qtype)

    def load_json(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            qtype = q.get("type", "generic")
            max_score = int(q.get("max_score", 1))
            prompt = q.get("question", "")
            entry_examples: list[tuple[str, float, int]] = []
            for ex in q.get("examples", []):
                answer = norm(ex.get("answer", ""))
                score = float(ex.get("score", 0))
                if answer:
                    entry_examples.append((answer, score, max_score))
                    self.add(qtype, answer, score, max_score)
            ref = q.get("reference_answer")
            if ref:
                r = norm(ref)
                if r:
                    entry_examples.append((r, float(max_score), max_score))
                    self.add(qtype, ref, float(max_score), max_score)
            if prompt:
                self.questions.append({
                    "prompt": norm(prompt),
                    "qtype": qtype,
                    "max_score": max_score,
                    "examples": entry_examples,
                })
        self._q_mat = None

    def _fit(self, qtype: str) -> None:
        if qtype in self._fitted:
            return
        self._fitted.add(qtype)
        examples = self.examples.get(qtype, [])
        if len(examples) < 2:
            return
        texts = [e[0] for e in examples]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            mat = vec.fit_transform(texts)
        except Exception:
            # Pure-Python fallback: character n-gram Jaccard similarity.
            vec = None
            mat = None
        self._vectorizers[qtype] = vec
        self._matrices[qtype] = mat

    def _fit_questions(self) -> None:
        if self._q_mat is not None or len(self.questions) < 2:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            self._q_vec = vec
            self._q_mat = vec.fit_transform([q["prompt"] for q in self.questions])
        except Exception:
            self._q_vec = None
            self._q_mat = None

    @staticmethod
    def _char_ngrams(s: str, n: int = 3) -> set[str]:
        s = norm(s)
        return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ga, gb = ReferenceModel._char_ngrams(a), ReferenceModel._char_ngrams(b)
        return len(ga & gb) / max(1, len(ga | gb))

    def match_question(self, prompt: str) -> Optional[tuple[dict, float]]:
        """Return (teacher question entry, similarity) for a question prompt."""
        if not prompt or not self.questions:
            return None
        self._fit_questions()
        prompt = norm(prompt)
        if self._q_vec is not None and self._q_mat is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                sims = cosine_similarity(self._q_vec.transform([prompt]), self._q_mat)[0]
                best = int(max(range(len(sims)), key=lambda i: sims[i]))
                return self.questions[best], float(sims[best])
            except Exception:
                pass
        sims = [self._jaccard(prompt, q["prompt"]) for q in self.questions]
        best = int(max(range(len(sims)), key=lambda i: sims[i]))
        return self.questions[best], float(sims[best])

    def _predict_from(self, examples: list[tuple[str, float, int]],
                      answer: str) -> Optional[tuple[float, float]]:
        if not examples:
            return None
        texts = [e[0] for e in examples]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            mat = vec.fit_transform(texts)
            sims = cosine_similarity(vec.transform([norm(answer)]), mat)[0]
        except Exception:
            sims = [self._jaccard(answer, t) for t in texts]

        order = sorted(range(len(examples)), key=lambda i: sims[i], reverse=True)[:3]
        weights = [max(0.0, float(sims[i])) for i in order]
        if not weights or max(weights) == 0:
            return None
        total_w = sum(weights)
        pred = sum(weights[j] * examples[i][1] for j, i in enumerate(order)) / total_w
        return pred, float(max(weights))

    def predict(self, qtype: str, answer: str, max_score: int) -> Optional[tuple[float, float]]:
        """Return (predicted_score, confidence) from examples of one type."""
        self._fit(qtype)
        return self._predict_from(self.examples.get(qtype, []), answer)

    def predict_for_question(self, prompt: str, qtype: str, answer: str,
                             max_score: int) -> Optional[tuple[float, float]]:
        """Question-aware prediction.

        First find the teacher question with the closest wording.  If the
        match is good and that question has labelled answers, grade with
        them; otherwise fall back to all examples of the same question type.
        The reported confidence combines question-match and answer-match so
        that a confident answer to the wrong question is not trusted.
        """
        match = self.match_question(prompt)
        if match:
            entry, qsim = match
            if qsim >= 0.35 and entry.get("examples"):
                res = self._predict_from(entry["examples"], answer)
                if res is not None:
                    score, confidence = res
                    # Keep the strongest of the two signals.
                    return score, confidence * max(0.5, qsim)
        return self.predict(qtype, answer, max_score)


# ---------------------------------------------------------------------------
# Exam parsing + grading pipeline
# ---------------------------------------------------------------------------

def parse_exam(path: Path, with_objective: bool = True) -> Exam:
    text = layout_text(path)
    student_id = path.stem.strip()
    m = re.search(r"Оюутны код\s+(\S+)", text)
    if m:
        student_id = m.group(1).strip()
    hm = re.search(r"Авсан оноо\s+(\d+(?:\.\d+)?)", text)
    header_total = float(hm.group(1)) if hm else None

    questions = parse_questions(text)
    if with_objective:
        selections = detect_selections(path)
        for num, q in questions.items():
            sel = selections.get(num, [])
            for i, opt in enumerate(q.options):
                if i < len(sel):
                    opt.selected = sel[i]
            if q.options:
                selected_scores = [o.score for o in q.options if o.selected]
                q.score = float(sum(selected_scores))
                q.detail = "сонгосон: " + ", ".join(
                    o.text for o in q.options if o.selected
                )
    return Exam(student_id=student_id, source=path, header_total=header_total, questions=questions)


def grade_exam(exam: Exam, rubric: RubricGrader, model: ReferenceModel,
               ml_only: bool = False, debug: bool = False) -> None:
    for num in sorted(exam.questions):
        q = exam.questions[num]
        if not q.is_written:
            continue
        rule_score, rule_detail = rubric.grade(q)
        ml = model.predict_for_question(q.prompt, q.qtype, q.written_answer, q.max_score)
        ml_score = None
        ml_conf = 0.0
        if ml is not None:
            ml_score, ml_conf = ml
            ml_score = max(0.0, min(float(q.max_score), round(ml_score * 2) / 2))

        if ml_only and ml_score is not None:
            q.score = ml_score
            q.detail = f"ML={ml_score:g} (итгэл {ml_conf:.2f})"
        elif q.qtype == "generic" and ml_score is not None and ml_conf >= 0.5:
            # A brand-new teacher question with no rubric handler: trust the
            # question+answer model, which found the matching example bank.
            q.score = ml_score
            q.detail = f"жишиг хариултаар ML={ml_score:g} (итгэл {ml_conf:.2f})"
        else:
            q.score = rule_score
            if ml_score is not None and rule_score == 0 and ml_conf >= 0.55 and ml_score > 0:
                # The rubric found nothing but the teacher-answer model is
                # quite confident: accept the model's positive answer.
                q.score = ml_score
                q.detail = f"ML fallback={ml_score:g} (итгэл {ml_conf:.2f}); {rule_detail}"
            else:
                q.detail = rule_detail
                if ml_score is not None:
                    q.detail += f" [ML={ml_score:g}, итгэл {ml_conf:.2f}]"

        if debug:
            print(
                f"  Q{num:<2} {q.qtype:<14} score={q.score:g}/{q.max_score} "
                f"answer={q.written_answer[:100]!r}"
            )
            print(f"       {q.detail}")


def load_training_examples(model: ReferenceModel, training_dir: Path) -> int:
    """Add teacher-graded written answers from the training folder."""
    added = 0
    if not training_dir.exists():
        return added
    for path in sorted(training_dir.glob("*.pdf")):
        try:
            exam = parse_exam(path, with_objective=False)
        except Exception as exc:
            print(f"  ! training file skipped: {path.name}: {exc}", file=sys.stderr)
            continue
        for q in exam.questions.values():
            if q.is_written and q.teacher_score is not None and q.written_answer:
                model.add(q.qtype, q.written_answer, q.teacher_score, q.max_score)
                added += 1
    return added


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(exams: list[Exam], csv_path: Path) -> None:
    all_nums = sorted({n for e in exams for n in e.questions}, reverse=True)
    header = ["num", "Оюутны код", "Нийт оноо"] + [str(n) for n in all_nums]
    rows = []
    for idx, exam in enumerate(exams, start=1):
        row = [idx, exam.student_id, fmt(exam.total)]
        for n in all_nums:
            q = exam.questions.get(n)
            row.append(fmt(q.score) if q else "")
        rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", default="muu-4-test",
                    help="folder with the exam PDFs to grade (default: muu-4-test)")
    ap.add_argument("--training-dir", default="sain-4-training",
                    help="folder with teacher-graded PDFs (default: sain-4-training)")
    ap.add_argument("--csv", default="mongol-hel.csv",
                    help="output CSV (default: mongol-hel.csv)")
    ap.add_argument("--reference", default="reference_answers.json",
                    help="teacher question + answer bank (default: reference_answers.json)")
    ap.add_argument("--learn-from-training", action="store_true",
                    help="also train the ML model on teacher-graded answers in the training folder")
    ap.add_argument("--ml-only", action="store_true",
                    help="use only the ML model for written questions (experimental)")
    ap.add_argument("--debug", action="store_true", help="print each answer and its decision")
    args = ap.parse_args(argv)

    base = Path(__file__).resolve().parent
    test_dir = (base / args.test_dir) if not os.path.isabs(args.test_dir) else Path(args.test_dir)
    training_dir = (base / args.training_dir) if not os.path.isabs(args.training_dir) else Path(args.training_dir)
    csv_path = (base / args.csv) if not os.path.isabs(args.csv) else Path(args.csv)
    ref_path = (base / args.reference) if not os.path.isabs(args.reference) else Path(args.reference)

    if not test_dir.exists():
        print(f"Тестийн хавтас олдсонгүй: {test_dir}", file=sys.stderr)
        return 2

    model = ReferenceModel()
    model.load_json(ref_path)
    n_ref = sum(len(v) for v in model.examples.values())

    n_train = 0
    if args.learn_from_training:
        n_train = load_training_examples(model, training_dir)

    rubric = RubricGrader()
    exams: list[Exam] = []
    for pdf in sorted(test_dir.glob("*.pdf")):
        print(f"Уншиж байна: {pdf.name}")
        exam = parse_exam(pdf, with_objective=True)
        grade_exam(exam, rubric, model, ml_only=args.ml_only, debug=args.debug)
        exams.append(exam)

    if not exams:
        print(f"PDF олдсонгүй: {test_dir}", file=sys.stderr)
        return 3

    print("\n" + "=" * 72)
    for exam in exams:
        print(f"{exam.student_id}: авто {fmt(exam.objective_score)} + бичгийн "
              f"{fmt(exam.written_score)} = {fmt(exam.total)} / {fmt(exam.max_total)}")
        for num in sorted(exam.questions, reverse=True):
            q = exam.questions[num]
            if q.is_written:
                print(f"  Q{num}: {fmt(q.score)}/{q.max_score}  {q.detail}")
        if exam.header_total is not None:
            print(f"  (PDF-д бүртгэсэн авто оноо: {fmt(exam.header_total)})")
    print("=" * 72)
    print(f"ML жишиг: {n_ref} бэлдсэн жишээ" + (f" + {n_train} сургалтын жишээ" if n_train else ""))

    write_csv(exams, csv_path)
    print(f"Бичигдлээ: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
