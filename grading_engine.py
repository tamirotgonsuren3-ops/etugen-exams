#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Similarity-based written-answer grading engine.

The teacher writes, for every written question, a reference answer (and
optionally several accepted variants).  A student's answer is compared with
that reference and the teacher chooses, with a slider, how similar an answer
must be to count as correct.

Two comparison methods are available:

* ``local``    – character n-gram TF-IDF cosine + fuzzy string matching.
                 Works offline, no API key needed.
* ``gemini``   – asks Gemini to judge *meaning* (0-100).  Handles paraphrases
                 that share few words with the teacher's answer.
* ``combined`` – the average of the two (default in the Streamlit app).

For questions worth more than one point the reference can be split into
parts.  Each part is compared separately, so partial credit is possible:

    Хөтөвч – ... | Хатавч – ... | Хонгиносон – ...

or numbered:

    1. ... 2. ... 3. ...

This module has no Streamlit dependency; it is used by ``streamlit_app.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from rapidfuzz import fuzz

try:  # scikit-learn is optional; a Jaccard fallback is used if missing.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_CYR = "0-9a-zа-яөү"


def normalize(text: str) -> str:
    """Lower-case, remove punctuation and collapse whitespace."""
    s = (text or "").lower().replace("\u00a0", " ").replace("\uf0c7", " ")
    s = s.replace("ё", "е")
    s = re.sub(rf"[^{_CYR}\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _char_ngrams(s: str, n: int = 3) -> set[str]:
    s = normalize(s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    return len(ga & gb) / max(1, len(ga | gb))


def char_cosine(a: str, b: str) -> float:
    """Character n-gram TF-IDF cosine similarity in [0, 1]."""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if _HAS_SKLEARN:
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            mat = vec.fit_transform([a, b])
            return float(cosine_similarity(mat[0], mat[1])[0][0])
        except Exception:
            pass
    return _jaccard(a, b, 3)


def lexical_similarity(a: str, b: str) -> float:
    """Robust offline similarity: TF-IDF, fuzzy ratio and token-set ratio."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return max(
        char_cosine(na, nb),
        fuzz.WRatio(na, nb) / 100.0,
        fuzz.token_set_ratio(na, nb) / 100.0,
    )


def clean_question_prompt(prompt: str) -> str:
    """Remove the answer / form junk from a parsed exam prompt."""
    p = prompt or ""
    for marker in (
        "Оноог хадгалах",
        "Оноо хадгалах",
        "Таны өгсөн оноо",
        "Оноо хадгалах",
    ):
        p = p.split(marker)[0]
    # Most written questions end with '?'.  The student's answer follows it.
    if "?" in p:
        p = p[: p.index("?") + 1]
    return p.strip()


def reference_alternatives(entry: dict) -> list[str]:
    """All acceptable answer variants for a one-point question.

    The teacher can separate variants with ``|``, a newline or ``;``.
    Full-credit examples from the answer bank are also accepted.
    """
    ref = (entry.get("reference_answer") or "").strip()
    max_score = _int(entry.get("max_score", 1), 1)
    texts: list[str] = [ref]
    for ex in entry.get("examples", []) or []:
        try:
            score = float(ex.get("score", 0))
        except Exception:
            score = 0.0
        answer = (ex.get("answer") or "").strip()
        if answer and score >= max_score:
            texts.append(answer)

    out: list[str] = []
    for text in texts:
        if not text:
            continue
        out.append(text)
        for part in re.split(r"\s*[|\n;]\s*", text):
            part = part.strip()
            if part:
                out.append(part)
        # Short lists such as "Өндийх, босох, сүмбэрлэх" are alternatives.
        if len(text) < 100:
            out.extend(p.strip() for p in re.split(r",\s*", text) if p.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for text in out:
        key = normalize(text)
        if key and key not in seen:
            seen.add(key)
            unique.append(text)
    return unique


def split_parts(text: str) -> list[str]:
    """Split a multi-point reference / answer into its parts.

    Supports:
      * numbered parts      ``1. ... 2. ...``
      * pipe / newline / ;  ``... | ...``
      * vocabulary headwords ``Хөтөвч- ... Хатавч- ... Хонгиносон- ...``
    If no marker is found the whole text is returned as one part.
    """
    t = (text or "").strip()
    if not t:
        return []
    low = t.lower()

    if "хөтөвч" in low and "хатавч" in low:
        parts = re.split(r"(?=хөтөвч|хатавч|хонгиносон)", t, flags=re.I)
    else:
        parts = re.split(r"\s*\d{1,2}[.)]\s+", t)
        if len([p for p in parts if p.strip()]) < 2:
            parts = re.split(r"\s*[|\n;]\s*", t)

    cleaned = [p.strip(" \t\r\n.;|") for p in parts if p.strip(" \t\r\n.;|")]
    return cleaned or [t.strip()]


def _int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def passes(similarity: float, threshold: float) -> bool:
    """Compare on the whole-percent scale the teacher actually sees."""
    return round(similarity * 100) >= round(threshold * 100)


# ---------------------------------------------------------------------------
# Question matching
# ---------------------------------------------------------------------------

class QuestionMatcher:
    """Find the teacher question that best matches a student's question text."""

    def __init__(self, entries: Sequence[dict]) -> None:
        self.entries = [e for e in entries if (e.get("question") or "").strip()]
        self._vec = None
        self._mat = None
        if len(self.entries) >= 2 and _HAS_SKLEARN:
            try:
                self._vec = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 4), min_df=1
                )
                self._mat = self._vec.fit_transform(
                    [normalize(e["question"]) for e in self.entries]
                )
            except Exception:
                self._vec = None
                self._mat = None

    def best(self, prompt: str) -> tuple[Optional[dict], float]:
        if not self.entries:
            return None, 0.0
        p = normalize(clean_question_prompt(prompt))
        if not p:
            return None, 0.0

        if self._vec is not None and self._mat is not None:
            try:
                sims = list(
                    cosine_similarity(self._vec.transform([p]), self._mat)[0]
                )
                best_i = int(max(range(len(sims)), key=lambda i: sims[i]))
                return self.entries[best_i], float(sims[best_i])
            except Exception:
                pass

        # Fallback when scikit-learn is unavailable: fuzzy token matching.
        best_i, best_s = 0, 0.0
        for i, entry in enumerate(self.entries):
            q = normalize(entry.get("question", ""))
            s = fuzz.token_set_ratio(p, q) / 100.0 if q else 0.0
            if s > best_s:
                best_i, best_s = i, s
        return self.entries[best_i], best_s


# ---------------------------------------------------------------------------
# Gemini semantic similarity
# ---------------------------------------------------------------------------

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

_SINGLE_PROMPT = """Чи монгол хэлний шалгалтын багш. Дараах асуултад оюутны бичсэн хариулт нь багшийн жишиг хариулттай утгын хувьд хэр ойролцоо, зөв эсэхийг 0-100 оноогоор үнэл.

Асуулт: {question}
Багшийн жишиг хариулт (аль нэг хувилбар нь зөв):
{reference}
Оюутны хариулт: {answer}

Дүгнэх дүрэм:
- Оюутны хариулт нь асуултын утгыг тайлбарлаж, багшийн хариултын гол санааг агуулж байвал өндөр оноо (80-100).
- Зөвхөн асуултын эх эсвэл хэлц үгийг давтсан боловч тайлбарлаагүй бол бага оноо (0-20).
- Хэсэгчилсэн санаа байвал 20-80.
- Үсгийн алдаа, найруулгыг бүү шийт.
Зөвхөн JSON-оор хариул: {{"similarity": <0-100 бүхэл тоо>, "reason": "<монголоор товч>"}}"""

_PARTS_PROMPT = """Чи монгол хэлний шалгалтын багш. Оюутны хариулт нь багшийн жишиг хариултын хэсэг тус бүрийг хэр зэрэг тусгасныг 0-100 оноогоор үнэл.

Асуулт: {question}
Багшийн жишиг хариултын хэсгүүд:
{parts}

Оюутны хариулт: {answer}

Дүрэм:
- Хэсэг бүрийн гол санаа оюутны хариултад байвал өндөр оноо (80-100).
- Хэсэгчилсэн эсвэл ойролцоо бол 20-80.
- Огт байхгүй бол 0-20.
- Үсгийн алдаа, найруулгыг бүү шийт.
Зөвхөн JSON-оор хариул: {{"parts": [{{"part": 1, "similarity": <0-100>}}, ...], "reason": "<товч>"}}"""


class SemanticScorer:
    """Compare a student answer with a teacher reference answer."""

    def __init__(
        self,
        mode: str = "combined",
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        cache_path: Optional[Path] = None,
    ) -> None:
        self.mode = mode if mode in {"local", "gemini", "combined"} else "combined"
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, Any] = self._load_cache()
        self.last_error: Optional[str] = None

    # -- cache -------------------------------------------------------------

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _cache_key(self, *parts: str) -> str:
        raw = "||".join(["gemini", self.model, *[normalize(p) for p in parts]])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -- Gemini call -------------------------------------------------------

    def _gemini_json(self, prompt: str, retries: int = 3) -> Optional[dict]:
        if not self.api_key:
            self.last_error = "GEMINI_API_KEY / GOOGLE_API_KEY тохируулаагүй."
            return None
        url = GEMINI_ENDPOINT.format(model=self.model, key=self.api_key)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0,
            },
        }
        data = json.dumps(body).encode("utf-8")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    payload = json.load(resp)
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except urllib.error.HTTPError as exc:
                self.last_error = f"Gemini HTTP {exc.code}"
                if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None
            except Exception as exc:  # network / JSON errors
                self.last_error = f"Gemini error: {exc}"
                if attempt + 1 < retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None
        return None

    # -- public API --------------------------------------------------------

    def single_similarity(
        self, question: str, answer: str, alternatives: Sequence[str]
    ) -> float:
        """Similarity of one answer to the best acceptable variant."""
        alternatives = [a for a in alternatives if (a or "").strip()]
        if not answer.strip() or not alternatives:
            return 0.0

        local = max(lexical_similarity(answer, alt) for alt in alternatives)
        if self.mode == "local":
            return local

        key = self._cache_key("single", question, answer, "\n".join(alternatives))
        if key in self.cache:
            gem = float(self.cache[key])
        else:
            ref_text = "\n".join(f"- {a}" for a in alternatives)
            prompt = _SINGLE_PROMPT.format(
                question=question, reference=ref_text, answer=answer
            )
            result = self._gemini_json(prompt)
            if result is None or "similarity" not in result:
                # Fall back to the offline score.
                return local
            try:
                gem = max(0.0, min(1.0, float(result["similarity"]) / 100.0))
            except Exception:
                return local
            self.cache[key] = gem
            self._save_cache()

        if self.mode == "gemini":
            return gem
        return (local + gem) / 2.0

    def part_similarities(
        self, question: str, answer: str, parts: Sequence[str]
    ) -> list[float]:
        """Similarity for every part of a multi-point reference."""
        parts = [p for p in parts if (p or "").strip()]
        if not parts or not answer.strip():
            return [0.0 for _ in parts]

        answer_parts = split_parts(answer) or [answer]
        local = [
            max(lexical_similarity(ap, part) for ap in answer_parts)
            for part in parts
        ]
        if self.mode == "local":
            return local

        key = self._cache_key("parts", question, answer, "|".join(parts))
        if key in self.cache and isinstance(self.cache[key], list):
            gem = [float(x) for x in self.cache[key]]
        else:
            plist = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))
            prompt = _PARTS_PROMPT.format(
                question=question, parts=plist, answer=answer
            )
            result = self._gemini_json(prompt)
            gem = self._parse_parts(result, len(parts))
            if gem is None:
                return local
            self.cache[key] = gem
            self._save_cache()

        if self.mode == "gemini":
            return gem
        return [(a + b) / 2.0 for a, b in zip(local, gem)]

    @staticmethod
    def _parse_parts(result: Optional[dict], n: int) -> Optional[list[float]]:
        if not result:
            return None
        raw = result.get("parts")
        if not isinstance(raw, list):
            return None
        scores = [0.0] * n
        for item in raw:
            try:
                idx = int(item.get("part", 0)) - 1
                sim = float(item.get("similarity", 0)) / 100.0
            except Exception:
                continue
            if 0 <= idx < n:
                scores[idx] = max(0.0, min(1.0, sim))
        return scores


# ---------------------------------------------------------------------------
# Written grading
# ---------------------------------------------------------------------------

@dataclass
class WrittenResult:
    score: float
    max_score: int
    similarity: float
    matched_parts: int
    total_parts: int
    detail: str
    part_scores: list[float] = field(default_factory=list)


def grade_written(
    entry: dict,
    student_answer: str,
    threshold: float,
    scorer: SemanticScorer,
    question: str = "",
) -> WrittenResult:
    """Grade one written answer with the teacher's reference answer."""
    max_score = max(1, _int(entry.get("max_score", 1), 1))
    reference = (entry.get("reference_answer") or "").strip()
    question = question or (entry.get("question") or "")

    if not (student_answer or "").strip():
        return WrittenResult(0, max_score, 0.0, 0, 1, "Хоосон хариулт", [])

    if not reference:
        return WrittenResult(0, max_score, 0.0, 0, 1, "Жишиг хариулт байхгүй", [])

    parts = split_parts(reference)
    if max_score > 1 and len(parts) > 1:
        sims = scorer.part_similarities(question, student_answer, parts)
        matched = sum(1 for s in sims if passes(s, threshold))
        raw = max_score * matched / len(parts)
        score = min(max_score, max(0, int(raw + 0.5)))
        similarity = sum(sims) / len(sims)
        detail = f"{matched}/{len(parts)} хэсэг таарсан"
        return WrittenResult(
            float(score), max_score, similarity, matched, len(parts), detail, sims
        )

    alternatives = reference_alternatives(entry)
    sim = scorer.single_similarity(question, student_answer, alternatives)
    correct = passes(sim, threshold)
    return WrittenResult(
        float(max_score if correct else 0),
        max_score,
        sim,
        1 if correct else 0,
        1,
        "Зөв" if correct else "Жишиг хариулттай хангалттай ойр биш",
        [sim],
    )


# ---------------------------------------------------------------------------
# CSV construction
# ---------------------------------------------------------------------------

def fmt_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def build_csv(exams: Sequence[Any]) -> bytes:
    """Build the ``num,Оюутны код,Нийт оноо,30,...,1`` CSV as UTF-8 bytes."""
    import csv
    import io

    all_nums = sorted({n for exam in exams for n in exam.questions}, reverse=True)
    header = ["num", "Оюутны код", "Нийт оноо"] + [str(n) for n in all_nums]
    rows = []
    for i, exam in enumerate(exams, start=1):
        row = [i, exam.student_id, fmt_number(exam.total)]
        for n in all_nums:
            q = exam.questions.get(n)
            row.append(fmt_number(q.score) if q else "")
        rows.append(row)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")
