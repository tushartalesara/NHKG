#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect and normalize Hindi temporal expressions as a post-extraction layer."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .enrichment_common import (
        collect_sentence_keys,
        iter_input_events,
        load_json,
        load_sentence_text_map,
        normalize_char_span,
        sentence_key,
        stable_event_id,
        spans_overlap,
    )
    from .internal_quality_common import DEFAULT_STAGE_VERSION, build_stage_metadata
    from .temporal_reasoning import infer_document_temporal_relations
except ImportError:  # pragma: no cover
    from enrichment_common import (
        collect_sentence_keys,
        iter_input_events,
        load_json,
        load_sentence_text_map,
        normalize_char_span,
        sentence_key,
        stable_event_id,
        spans_overlap,
    )
    from internal_quality_common import DEFAULT_STAGE_VERSION, build_stage_metadata
    from temporal_reasoning import infer_document_temporal_relations


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


MONTHS_HI = {
    "जनवरी": 1,
    "फ़रवरी": 2,
    "फरवरी": 2,
    "मार्च": 3,
    "अप्रैल": 4,
    "मई": 5,
    "जून": 6,
    "जुलाई": 7,
    "अगस्त": 8,
    "सितंबर": 9,
    "अक्टूबर": 10,
    "नवंबर": 11,
    "दिसंबर": 12,
}

WEEKDAY_TIMEX = {
    "सोमवार": "XXXX-WXX-1",
    "मंगलवार": "XXXX-WXX-2",
    "बुधवार": "XXXX-WXX-3",
    "गुरुवार": "XXXX-WXX-4",
    "शुक्रवार": "XXXX-WXX-5",
    "शनिवार": "XXXX-WXX-6",
    "रविवार": "XXXX-WXX-7",
}

NUMBER_WORDS = {
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पाँच": 5,
    "पांच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
}

DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})\b"),
    re.compile(
        r"\b\d{1,2}\s+(?:जनवरी|फ़रवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)(?:\s+\d{4})?\b"
    ),
    re.compile(r"\b(?:आज|कल|परसों|नरसों|बीते कल|आने वाले कल)\b"),
    re.compile(r"\b(?:पिछले|अगले|इस)\s+(?:सप्ताह|हफ्ते|महीने|माह|साल|वर्ष|दिन)\b"),
]

TIME_PATTERNS = [
    re.compile(r"\b(?:सुबह|शाम|रात|दोपहर)?\s*\d{1,2}(?::\d{2})?\s*बजे\b"),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
]

DURATION_PATTERNS = [
    re.compile(
        r"\b(?:[0-9०-९]+|एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह|सोलह|सत्रह|अठारह|उन्नीस|बीस)\s+(?:दिन|दिवस|हफ्ते|सप्ताह|महीने|माह|साल|वर्ष|घंटे|घंटा|मिनट)\b(?:\s+तक)?"
    )
]

SET_PATTERNS = [
    re.compile(r"\b(?:हर|प्रति)\s+(?:दिन|सप्ताह|हफ्ता|महीना|माह|साल|वर्ष)\b"),
    re.compile(r"\b(?:हर|प्रति)\s+(?:सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)\b"),
    re.compile(r"\b(?:रोज|प्रतिदिन|दैनिक)\b"),
]

TIMEX3_RE = re.compile(
    r"<TIMEX3[^>]*type=\"([^\"]+)\"[^>]*value=\"([^\"]*)\"[^>]*>(.*?)</TIMEX3>",
    flags=re.IGNORECASE | re.DOTALL,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect Hindi temporal expressions and write align/time_cache.json.",
    )
    parser.add_argument("--input", default="", help="Extraction JSONL/JSON for event-time linking")
    parser.add_argument("--sentences", default="", help="One sentence per line text file")
    parser.add_argument("--syntax-json", default="", help="Optional POS/syntax cache")
    parser.add_argument("--ner-json", default="", help="Optional NER cache")
    parser.add_argument("--doc-dates", default="", help='Optional JSON mapping doc_id -> "YYYY-MM-DD"')
    parser.add_argument(
        "--engine",
        choices=["rules", "dateparser", "heideltime", "hybrid"],
        default="hybrid",
        help="Temporal detection backend",
    )
    parser.add_argument(
        "--heideltime-path",
        default="",
        help="Optional HeidelTime executable or jar path; used on a best-effort basis",
    )
    parser.add_argument("--context-window", type=int, default=6, help="Neighborhood size for fallback proximity")
    parser.add_argument("--out", default="align/time_cache.json", help="Output JSON cache path")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of sample timexes to print")
    return parser


def load_sentence_entries(input_path: Optional[Path], sentence_path: Optional[Path]) -> List[Tuple[str, str]]:
    if sentence_path is None or not sentence_path.exists():
        raise SystemExit("Temporal enrichment needs sentence text. Pass --sentences with one sentence per line.")
    sentence_keys = collect_sentence_keys(input_path) if input_path else []
    sentence_map = load_sentence_text_map(sentence_path, sentence_keys=sentence_keys)
    if not sentence_map:
        raise SystemExit("No sentence text lines found in the supplied --sentences file.")
    if sentence_keys:
        ordered = [(key, sentence_map[key]) for key in sentence_keys if key in sentence_map]
        if ordered:
            return ordered
    return list(sentence_map.items())


def load_doc_dates(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("--doc-dates must be a JSON object of doc_id -> YYYY-MM-DD")
    return {str(key): str(value) for key, value in payload.items() if str(value).strip()}


def load_syntax_cache(path: Optional[Path]) -> Dict[str, List[dict]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[dict]] = {}
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            tokens = item.get("tokens", [])
            if isinstance(tokens, list):
                out[f"{doc_id}::{sent_id}"] = tokens
    return out


def load_ner_cache(path: Optional[Path]) -> Dict[str, List[dict]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[dict]] = {}
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            entities = item.get("entities", [])
            if isinstance(entities, list):
                out[f"{doc_id}::{sent_id}"] = entities
    return out


def load_events_by_sentence(path: Optional[Path]) -> Dict[str, List[dict]]:
    if path is None or not path.exists():
        return {}
    grouped: Dict[str, List[dict]] = {}
    for event in iter_input_events(path):
        key = sentence_key(event.get("doc_id", "batch_run"), event.get("sent_id", 0))
        grouped.setdefault(key, []).append(event)
    return grouped


def parse_doc_date(doc_date: Optional[str]) -> Optional[date]:
    if not doc_date:
        return None
    try:
        return datetime.strptime(doc_date, "%Y-%m-%d").date()
    except ValueError:
        return None


def devanagari_number_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    normalized = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    if normalized.isdigit():
        return int(normalized)
    return NUMBER_WORDS.get(normalized.strip())


def match_candidates(patterns: Sequence[re.Pattern[str]], text: str, timex_type: str) -> List[dict]:
    out = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            out.append(
                {
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "type": timex_type,
                    "notes": [],
                }
            )
    return out


def best_effort_heideltime(text: str, doc_date_value: Optional[str], path: Optional[Path]) -> List[dict]:
    if path is None or not path.exists():
        return []

    command = ["java", "-jar", str(path)] if path.suffix.lower() == ".jar" else [str(path)]
    if doc_date_value:
        command.extend(["--dct", doc_date_value])

    try:
        completed = subprocess.run(
            command,
            input=text,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )
    except Exception as exc:
        print(f"[WARN] HeidelTime invocation failed, falling back to Python rules: {exc}", file=sys.stderr)
        return []

    if completed.returncode != 0:
        print(
            f"[WARN] HeidelTime returned {completed.returncode}: {completed.stderr.strip() or 'no stderr'}",
            file=sys.stderr,
        )
        return []

    results = []
    for match in TIMEX3_RE.finditer(completed.stdout or ""):
        timex_type, value, surface = match.groups()
        start = text.find(surface)
        if start < 0:
            continue
        results.append(
            {
                "text": surface,
                "start": start,
                "end": start + len(surface),
                "type": timex_type.upper(),
                "value": value or None,
                "resolution_status": "resolved" if value else "unresolved",
                "engine": "heideltime",
                "confidence": 0.90 if value else 0.50,
                "notes": [],
                "unresolved_reasons": [] if value else ["parser_failed"],
            }
        )
    return results


def try_dateparser_parse(text: str, doc_date_value: Optional[str]) -> Optional[datetime]:
    try:
        import dateparser
    except ImportError:
        return None

    settings = {
        "PREFER_DATES_FROM": "future",
    }
    base_date = parse_doc_date(doc_date_value)
    if base_date is not None:
        settings["RELATIVE_BASE"] = datetime.combine(base_date, datetime.min.time())
    try:
        return dateparser.parse(text, languages=["hi"], settings=settings)
    except Exception:
        return None


def normalize_relative_date(text: str, doc_date_value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    notes: List[str] = []
    base = parse_doc_date(doc_date_value)
    if base is None:
        return None, ["Missing document date for relative normalization"]

    stripped = text.strip()
    if stripped == "आज":
        return base.isoformat(), notes
    if stripped in {"कल", "आने वाले कल"}:
        if stripped == "कल":
            notes.append("Hindi 'कल' can mean yesterday or tomorrow; defaulted to next day.")
        return (base + timedelta(days=1)).isoformat(), notes
    if stripped == "बीते कल":
        return (base - timedelta(days=1)).isoformat(), notes
    if stripped in {"परसों", "नरसों"}:
        notes.append("Hindi 'परसों' can be ambiguous; defaulted to two days ahead.")
        return (base + timedelta(days=2)).isoformat(), notes
    if "पिछले" in stripped and ("सप्ताह" in stripped or "हफ्ते" in stripped):
        prev = base - timedelta(days=7)
        iso_year, iso_week, _ = prev.isocalendar()
        return f"{iso_year}-W{iso_week:02d}", notes
    if "अगले" in stripped and ("सप्ताह" in stripped or "हफ्ते" in stripped):
        nxt = base + timedelta(days=7)
        iso_year, iso_week, _ = nxt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}", notes
    if "इस" in stripped and ("सप्ताह" in stripped or "हफ्ते" in stripped):
        iso_year, iso_week, _ = base.isocalendar()
        return f"{iso_year}-W{iso_week:02d}", notes
    if "पिछले" in stripped and ("महीने" in stripped or "माह" in stripped):
        month = base.month - 1 or 12
        year = base.year - 1 if base.month == 1 else base.year
        return f"{year:04d}-{month:02d}", notes
    if "अगले" in stripped and ("महीने" in stripped or "माह" in stripped):
        month = 1 if base.month == 12 else base.month + 1
        year = base.year + 1 if base.month == 12 else base.year
        return f"{year:04d}-{month:02d}", notes
    if "इस" in stripped and ("महीने" in stripped or "माह" in stripped):
        return f"{base.year:04d}-{base.month:02d}", notes
    if "पिछले" in stripped and ("साल" in stripped or "वर्ष" in stripped):
        return f"{base.year - 1:04d}", notes
    if "अगले" in stripped and ("साल" in stripped or "वर्ष" in stripped):
        return f"{base.year + 1:04d}", notes
    if "इस" in stripped and ("साल" in stripped or "वर्ष" in stripped):
        return f"{base.year:04d}", notes
    if "अगले" in stripped and "दिन" in stripped:
        return (base + timedelta(days=1)).isoformat(), notes
    if "पिछले" in stripped and "दिन" in stripped:
        return (base - timedelta(days=1)).isoformat(), notes
    return None, notes


def normalize_date_candidate(text: str, doc_date_value: Optional[str], engine: str) -> Tuple[Optional[str], str, List[str], float]:
    notes: List[str] = []
    stripped = text.strip()
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", stripped):
        return stripped, "resolved", notes, 0.98

    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", stripped):
        day, month, year = re.split(r"[/-]", stripped)
        year_value = int(year)
        if year_value < 100:
            year_value += 2000
        try:
            normalized = date(year_value, int(month), int(day)).isoformat()
            return normalized, "resolved", notes, 0.96
        except ValueError:
            pass

    month_match = re.fullmatch(
        r"(\d{1,2})\s+(जनवरी|फ़रवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)(?:\s+(\d{4}))?",
        stripped,
    )
    if month_match:
        day, month_name, year = month_match.groups()
        base_date = parse_doc_date(doc_date_value)
        year_value = int(year) if year else base_date.year if base_date else None
        if year_value is not None:
            try:
                normalized = date(year_value, MONTHS_HI[month_name], int(day)).isoformat()
                if year is None:
                    notes.append("Year inferred from document date.")
                return normalized, "resolved", notes, 0.92
            except ValueError:
                pass

    relative_value, relative_notes = normalize_relative_date(stripped, doc_date_value)
    notes.extend(relative_notes)
    if relative_value:
        return relative_value, "resolved", notes, 0.85

    parsed = try_dateparser_parse(stripped, doc_date_value) if engine in {"dateparser", "hybrid", "rules", "heideltime"} else None
    if parsed is not None:
        return parsed.date().isoformat(), "resolved", notes, 0.80

    return None, "unresolved", notes, 0.40


def normalize_time_candidate(text: str, doc_date_value: Optional[str]) -> Tuple[Optional[str], str, List[str], float]:
    notes: List[str] = []
    stripped = text.strip()
    part_of_day_only = {
        "सुबह": "TMO",
        "दोपहर": "TAF",
        "शाम": "TEV",
        "रात": "TNI",
    }
    if stripped in part_of_day_only:
        return part_of_day_only[stripped], "resolved", notes, 0.76
    match = re.search(r"(?:(सुबह|शाम|रात|दोपहर)\s*)?(\d{1,2})(?::(\d{2}))?\s*बजे", text)
    if not match:
        match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"T{hour:02d}:{minute:02d}", "resolved", notes, 0.90
        return None, "unresolved", notes, 0.35

    part_of_day, hour_str, minute_str = match.groups()
    hour = int(hour_str)
    minute = int(minute_str or 0)
    if part_of_day in {"शाम", "रात"} and hour < 12:
        hour += 12
    if part_of_day == "दोपहर" and 1 <= hour <= 6:
        hour += 12
    return f"T{hour:02d}:{minute:02d}", "resolved", notes, 0.88


def normalize_duration_candidate(text: str) -> Tuple[Optional[str], str, List[str], float]:
    match = re.search(
        r"([0-9०-९]+|एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह|सोलह|सत्रह|अठारह|उन्नीस|बीस)\s+(दिन|दिवस|हफ्ते|सप्ताह|महीने|माह|साल|वर्ष|घंटे|घंटा|मिनट)",
        text,
    )
    if not match:
        return None, "unresolved", [], 0.35
    amount = devanagari_number_to_int(match.group(1))
    unit = match.group(2)
    if amount is None:
        return None, "unresolved", [], 0.35
    unit_map = {
        "दिन": f"P{amount}D",
        "दिवस": f"P{amount}D",
        "हफ्ते": f"P{amount}W",
        "सप्ताह": f"P{amount}W",
        "महीने": f"P{amount}M",
        "माह": f"P{amount}M",
        "साल": f"P{amount}Y",
        "वर्ष": f"P{amount}Y",
        "घंटे": f"PT{amount}H",
        "घंटा": f"PT{amount}H",
        "मिनट": f"PT{amount}M",
    }
    value = unit_map.get(unit)
    if not value:
        return None, "unresolved", [], 0.35
    return value, "resolved", [], 0.90


def normalize_set_candidate(text: str) -> Tuple[Optional[str], str, List[str], float]:
    stripped = text.strip()
    if stripped in {"रोज", "प्रतिदिन", "दैनिक"} or "हर दिन" in stripped or "प्रति दिन" in stripped:
        return "P1D", "resolved", [], 0.82
    if "सप्ताह" in stripped or "हफ्ता" in stripped:
        return "P1W", "resolved", [], 0.82
    if "महीना" in stripped or "माह" in stripped:
        return "P1M", "resolved", [], 0.82
    if "साल" in stripped or "वर्ष" in stripped:
        return "P1Y", "resolved", [], 0.82
    for day_name, timex_value in WEEKDAY_TIMEX.items():
        if day_name in stripped:
            return timex_value, "resolved", [], 0.80
    return None, "unresolved", [], 0.35


def classify_unresolved_timex_reasons(
    text: str,
    timex_type: str,
    doc_date_value: Optional[str],
    notes: Sequence[str],
    engine: str,
) -> List[str]:
    note_blob = " ".join(str(note) for note in notes).lower()
    stripped = str(text or "").strip()
    reasons: List[str] = []
    relative_markers = ("आज", "कल", "परसों", "नरसों", "पिछले", "अगले", "इस")
    if ("missing document date" in note_blob) or (any(marker in stripped for marker in relative_markers) and not doc_date_value):
        reasons.append("no_doc_date_for_relative_expression")
    if "ambiguous" in note_blob:
        reasons.append("ambiguous_relative_reference")
    if not reasons and re.search(r"[0-9०-९]", stripped) and timex_type in {"DATE", "TIME", "DURATION"}:
        reasons.append("malformed_numeric_expression")
    if not reasons and engine in {"dateparser", "hybrid", "heideltime"}:
        reasons.append("parser_failed")
    if not reasons:
        reasons.append("unsupported_expression_pattern")
    return reasons


def normalize_timex(candidate: dict, doc_date_value: Optional[str], engine: str) -> dict:
    timex_type = str(candidate.get("type", "DATE")).upper()
    text = str(candidate.get("text", "")).strip()
    value = None
    status = "unresolved"
    notes = list(candidate.get("notes", []))
    confidence = 0.40

    if timex_type == "DATE":
        value, status, extra_notes, confidence = normalize_date_candidate(text, doc_date_value, engine)
        notes.extend(extra_notes)
    elif timex_type == "TIME":
        value, status, extra_notes, confidence = normalize_time_candidate(text, doc_date_value)
        notes.extend(extra_notes)
    elif timex_type == "DURATION":
        value, status, extra_notes, confidence = normalize_duration_candidate(text)
        notes.extend(extra_notes)
    elif timex_type == "SET":
        value, status, extra_notes, confidence = normalize_set_candidate(text)
        notes.extend(extra_notes)

    item = dict(candidate)
    item["type"] = timex_type
    item["value"] = value
    item["resolution_status"] = status
    item["confidence"] = round(confidence, 6)
    item["notes"] = notes
    item["unresolved_reasons"] = classify_unresolved_timex_reasons(text, timex_type, doc_date_value, notes, engine) if status != "resolved" else []
    return item


def dedupe_timexes(candidates: Iterable[dict]) -> List[dict]:
    best: Dict[Tuple[Optional[int], Optional[int], str, str], dict] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        key = (
            item.get("start"),
            item.get("end"),
            str(item.get("type", "")).upper(),
            str(item.get("text", "")).strip(),
        )
        if key[3] == "":
            continue
        current = best.get(key)
        score = float(item.get("confidence", 0.0) or 0.0)
        if current is None or score >= float(current.get("confidence", 0.0) or 0.0):
            best[key] = dict(item)
    return [
        best[key]
        for key in sorted(
            best.keys(),
            key=lambda value: ((value[0] if value[0] is not None else -1), (value[1] if value[1] is not None else -1), value[2], value[3]),
        )
    ]


def detect_python_timexes(
    sent_key_value: str,
    text: str,
    doc_date_value: Optional[str],
    *,
    engine: str,
    ner_mentions: Optional[Sequence[dict]] = None,
) -> List[dict]:
    candidates = []
    candidates.extend(match_candidates(DATE_PATTERNS, text, "DATE"))
    candidates.extend(match_candidates(TIME_PATTERNS, text, "TIME"))
    candidates.extend(match_candidates(DURATION_PATTERNS, text, "DURATION"))
    candidates.extend(match_candidates(SET_PATTERNS, text, "SET"))

    for ner_item in ner_mentions or []:
        if not isinstance(ner_item, dict):
            continue
        if str(ner_item.get("label", "")).upper() != "TIME":
            continue
        try:
            start = int(ner_item.get("start"))
            end = int(ner_item.get("end"))
        except (TypeError, ValueError):
            continue
        surface = str(ner_item.get("text", "")).strip()
        if not surface:
            continue
        candidates.append(
            {
                "text": surface,
                "start": start,
                "end": end,
                "type": "TIME" if ("बजे" in surface or ":" in surface or surface.strip() in {"सुबह", "शाम", "रात", "दोपहर"}) else "DATE",
                "notes": ["Seeded from NER TIME label"],
            }
        )

    normalized = []
    for index, candidate in enumerate(candidates):
        item = normalize_timex(candidate, doc_date_value, engine)
        item["timex_id"] = f"{sent_key_value}::timex::{index}"
        item["engine"] = "dateparser" if engine in {"dateparser", "hybrid"} else "rules"
        item["mod"] = None
        item["linked_event_ids"] = []
        item["linked_events"] = []
        normalized.append(item)
    return dedupe_timexes(normalized)


def token_node_id(token: dict, fallback_index: int) -> Tuple[int, int]:
    sent_index = int(token.get("sent_index", 0) or 0)
    word_index = int(token.get("word_index", fallback_index + 1) or (fallback_index + 1))
    return sent_index, word_index


def token_span(token: dict) -> Optional[Tuple[int, int]]:
    try:
        start = int(token.get("start"))
        end = int(token.get("end"))
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def tokens_for_span(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if span is None:
        return []
    overlaps = []
    for index, token in enumerate(tokens):
        offsets = token_span(token)
        if offsets is None or not spans_overlap(offsets, span):
            continue
        overlaps.append(token_node_id(token, index))
    return overlaps


def dependency_distance(tokens: Sequence[dict], left: Optional[Tuple[int, int]], right: Optional[Tuple[int, int]]) -> Optional[int]:
    left_nodes = tokens_for_span(tokens, left)
    right_nodes = tokens_for_span(tokens, right)
    if not left_nodes or not right_nodes:
        return None

    adjacency: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for index, token in enumerate(tokens):
        node = token_node_id(token, index)
        adjacency.setdefault(node, set())
        head = token.get("head")
        if not isinstance(head, int) or head <= 0:
            continue
        head_node = (int(token.get("sent_index", 0) or 0), int(head))
        adjacency.setdefault(head_node, set())
        adjacency[node].add(head_node)
        adjacency[head_node].add(node)

    frontier = list(left_nodes)
    seen = {node: 0 for node in frontier}
    targets = set(right_nodes)
    while frontier:
        current = frontier.pop(0)
        if current in targets:
            return seen[current]
        for neighbor in adjacency.get(current, set()):
            if neighbor in seen:
                continue
            seen[neighbor] = seen[current] + 1
            frontier.append(neighbor)
    return None


def span_distance(left: Optional[Tuple[int, int]], right: Optional[Tuple[int, int]]) -> int:
    if left is None or right is None:
        return 10**6
    if spans_overlap(left, right):
        return 0
    if left[1] <= right[0]:
        return right[0] - left[1]
    return left[0] - right[1]


def ensure_document_timex(sent_key_value: str, doc_date_value: str) -> dict:
    return {
        "timex_id": f"{sent_key_value}::docdate",
        "text": doc_date_value,
        "start": None,
        "end": None,
        "type": "DATE",
        "value": doc_date_value,
        "mod": None,
        "resolution_status": "resolved",
        "engine": "docdate",
        "confidence": 1.0,
        "linked_event_ids": [],
        "linked_events": [],
        "notes": ["Document creation time sidecar"],
        "unresolved_reasons": [],
        "is_document_time": True,
    }


def link_events_to_timexes(
    sent_key_value: str,
    timexes: List[dict],
    events: Sequence[dict],
    tokens: Sequence[dict],
    doc_date_value: Optional[str],
) -> Tuple[List[dict], Counter]:
    stats = Counter()
    event_links: List[dict] = []
    doc_timex: Optional[dict] = None

    for event in events:
        event_id = stable_event_id(event)
        doc_id = str(event.get("doc_id", "batch_run"))
        sent_id = str(event.get("sent_id", 0))
        trigger_span = normalize_char_span((event.get("trigger", {}) or {}).get("char_span"))

        argument_spans = []
        arguments = event.get("arguments", {}) or {}
        if isinstance(arguments, dict):
            for arg in arguments.values():
                if not isinstance(arg, dict):
                    continue
                arg_span = normalize_char_span(arg.get("char_span"))
                if arg_span is not None:
                    argument_spans.append(arg_span)

        chosen = None
        strategy = None

        overlap_candidates = []
        for timex in timexes:
            timex_span = normalize_char_span([timex.get("start"), timex.get("end")])
            if timex_span is None:
                continue
            if any(spans_overlap(arg_span, timex_span) for arg_span in argument_spans):
                overlap_candidates.append((span_distance(trigger_span, timex_span), timex))
        if overlap_candidates:
            overlap_candidates.sort(key=lambda item: item[0])
            chosen = overlap_candidates[0][1]
            strategy = "overlap"

        if chosen is None and tokens and trigger_span is not None:
            syntax_candidates = []
            for timex in timexes:
                timex_span = normalize_char_span([timex.get("start"), timex.get("end")])
                if timex_span is None:
                    continue
                dep_distance = dependency_distance(tokens, trigger_span, timex_span)
                if dep_distance is None:
                    continue
                syntax_candidates.append((dep_distance, span_distance(trigger_span, timex_span), timex))
            if syntax_candidates:
                syntax_candidates.sort(key=lambda item: (item[0], item[1]))
                chosen = syntax_candidates[0][2]
                strategy = "syntax"

        if chosen is None and trigger_span is not None:
            proximity_candidates = []
            for timex in timexes:
                timex_span = normalize_char_span([timex.get("start"), timex.get("end")])
                if timex_span is None:
                    continue
                proximity_candidates.append((span_distance(trigger_span, timex_span), timex))
            if proximity_candidates:
                proximity_candidates.sort(key=lambda item: item[0])
                chosen = proximity_candidates[0][1]
                strategy = "proximity"

        predicate = "hasTimeExpression"
        if chosen is None and doc_date_value:
            if doc_timex is None:
                doc_timex = ensure_document_timex(sent_key_value, doc_date_value)
            chosen = doc_timex
            strategy = "docdate"
            predicate = "hasDocumentTime"

        if chosen is None or strategy is None:
            continue

        if event_id not in chosen["linked_event_ids"]:
            chosen["linked_event_ids"].append(event_id)
        chosen["linked_events"].append(
            {
                "event_id": event_id,
                "doc_id": doc_id,
                "sent_id": sent_id,
                "strategy": strategy,
                "predicate": predicate,
            }
        )
        event_links.append(
            {
                "event_id": event_id,
                "doc_id": doc_id,
                "sent_id": sent_id,
                "timex_id": chosen["timex_id"],
                "strategy": strategy,
                "predicate": predicate,
                "confidence": {
                    "overlap": 0.88,
                    "syntax": 0.84,
                    "proximity": 0.65,
                    "docdate": 0.75,
                }.get(strategy, 0.60),
                "evidence": chosen.get("value") or chosen.get("text") or strategy,
            }
        )
        stats[strategy] += 1

    if doc_timex is not None:
        timexes.append(doc_timex)
    return event_links, stats



def attach_event_event_relations(payload: Dict[str, object], events_by_sentence: Dict[str, List[dict]]) -> None:
    documents: Dict[str, List[dict]] = {}
    for sent_key_value, events in events_by_sentence.items():
        doc_id, sent_id = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
        item = ((payload.get("sentences", {}) or {}).get(doc_id, {}) or {}).get(sent_id, {})
        if not isinstance(item, dict):
            continue
        timex_lookup = {}
        for timex in item.get("timexes", []) if isinstance(item.get("timexes", []), list) else []:
            if isinstance(timex, dict):
                timex_lookup[str(timex.get("timex_id", ""))] = timex
        event_to_timex = {}
        for link in item.get("event_time_links", []) if isinstance(item.get("event_time_links", []), list) else []:
            if not isinstance(link, dict):
                continue
            event_to_timex[str(link.get("event_id", ""))] = timex_lookup.get(str(link.get("timex_id", "")), {})
        event_rows = []
        for event in events:
            row = dict(event)
            row["event_id"] = stable_event_id(event)
            event_rows.append(row)
        documents.setdefault(doc_id, []).append(
            {
                "doc_id": doc_id,
                "sent_id": sent_id,
                "sent_key": sent_key_value,
                "text": item.get("text", ""),
                "events": event_rows,
                "event_to_timex": event_to_timex,
            }
        )

    for doc_id, rows in documents.items():
        relation_map: Dict[str, List[dict]] = {}
        for relation in infer_document_temporal_relations(rows):
            if not isinstance(relation, dict):
                continue
            source_event_id = str(relation.get("source_event_id", ""))
            sent_key_value = ""
            for row in rows:
                if any(str(event.get("event_id", stable_event_id(event))) == source_event_id for event in row.get("events", [])):
                    sent_key_value = str(row.get("sent_key", ""))
                    break
            if sent_key_value:
                relation_map.setdefault(sent_key_value, []).append(relation)
        for sent_key_value, relations in relation_map.items():
            doc_part, sent_part = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
            target = ((payload.get("sentences", {}) or {}).get(doc_part, {}) or {}).get(sent_part)
            if isinstance(target, dict):
                target["event_event_relations"] = relations

def print_summary(payload: Dict[str, object], debug_samples: int) -> None:
    total_sentences = 0
    total_timexes = 0
    per_type = Counter()
    resolved = 0
    unresolved = 0
    unresolved_reasons = Counter()
    event_links = 0
    relation_count = 0
    strategy_counts = Counter()
    relation_type_counts = Counter()
    confidences = []
    samples = []

    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            total_sentences += 1
            timexes = item.get("timexes", [])
            links = item.get("event_time_links", [])
            relations = item.get("event_event_relations", [])
            if isinstance(links, list):
                event_links += len(links)
                for link in links:
                    if isinstance(link, dict):
                        strategy_counts[str(link.get("strategy", "unknown"))] += 1
            if isinstance(relations, list):
                relation_count += len(relations)
                for relation in relations:
                    if isinstance(relation, dict):
                        relation_type_counts[str(relation.get("relation", "UNKNOWN"))] += 1
            if not isinstance(timexes, list):
                continue
            total_timexes += len(timexes)
            for timex in timexes:
                if not isinstance(timex, dict):
                    continue
                per_type[str(timex.get("type", "DATE"))] += 1
                if timex.get("resolution_status") == "resolved":
                    resolved += 1
                else:
                    unresolved += 1
                    for reason in timex.get("unresolved_reasons", []) if isinstance(timex.get("unresolved_reasons", []), list) else []:
                        unresolved_reasons[str(reason)] += 1
                try:
                    confidences.append(float(timex.get("confidence", 0.0)))
                except (TypeError, ValueError):
                    pass
            if len(samples) < max(0, debug_samples) and timexes:
                preview = []
                for timex in timexes[:4]:
                    preview.append(
                        f"{timex.get('text', '')}/{timex.get('type', '')}->{timex.get('value', '')}"
                    )
                samples.append(f"{sentence_key(doc_id, sent_id)} :: {' | '.join(preview)}")

    print(f"[OK] total_sentences={total_sentences}")
    print(f"[OK] total_timexes={total_timexes}")
    print(f"[OK] timex_count_by_type={dict(sorted(per_type.items()))}")
    print(f"[OK] resolved={resolved} unresolved={unresolved}")
    if unresolved_reasons:
        print(f"[OK] unresolved_reasons={dict(sorted(unresolved_reasons.items()))}")
    print(f"[OK] event_time_links_created={event_links}")
    print(f"[OK] links_by_strategy={dict(sorted(strategy_counts.items()))}")
    print(f"[OK] event_event_relations={relation_count}")
    print(f"[OK] event_event_relations_by_type={dict(sorted(relation_type_counts.items()))}")
    if confidences:
        print(f"[OK] average_confidence={statistics.mean(confidences):.4f}")
    for sample in samples:
        print(f"[DBG] {sample}")


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input) if args.input else None
    if input_path is not None and not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    sentence_path = Path(args.sentences) if args.sentences else None
    entries = load_sentence_entries(input_path, sentence_path)
    syntax_data = load_syntax_cache(Path(args.syntax_json) if args.syntax_json else None)
    ner_data = load_ner_cache(Path(args.ner_json) if args.ner_json else None)
    doc_dates = load_doc_dates(Path(args.doc_dates) if args.doc_dates else None)
    events_by_sentence = load_events_by_sentence(input_path)
    heideltime_path = Path(args.heideltime_path) if args.heideltime_path else None

    payload: Dict[str, object] = {
        "meta": build_stage_metadata(
            stage_name="temporal_enrich",
            stage_version=DEFAULT_STAGE_VERSION,
            engine=args.engine,
            source_paths={
                "input": input_path,
                "sentences": sentence_path,
                "syntax_json": Path(args.syntax_json) if args.syntax_json else None,
                "ner_json": Path(args.ner_json) if args.ner_json else None,
                "doc_dates": Path(args.doc_dates) if args.doc_dates else None,
            },
            input_counts={"sentences": len(entries)},
            warnings=[],
            extra={"context_window": args.context_window},
        ),
        "sentences": {},
    }

    for sent_key_value, text in entries:
        doc_id, sent_id = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
        doc_date_value = doc_dates.get(doc_id)

        timexes: List[dict] = []
        if args.engine in {"heideltime", "hybrid"}:
            timexes.extend(best_effort_heideltime(text, doc_date_value, heideltime_path))
        if args.engine in {"rules", "dateparser", "hybrid"} or not timexes:
            timexes.extend(
                detect_python_timexes(
                    sent_key_value,
                    text,
                    doc_date_value,
                    engine=args.engine,
                    ner_mentions=ner_data.get(sent_key_value),
                )
            )

        timexes = dedupe_timexes(timexes)
        for index, timex in enumerate(timexes):
            timex["timex_id"] = timex.get("timex_id") or f"{sent_key_value}::timex::{index}"
            timex.setdefault("mod", None)
            timex.setdefault("engine", "rules")
            timex.setdefault("linked_event_ids", [])
            timex.setdefault("linked_events", [])

        event_links, _ = link_events_to_timexes(
            sent_key_value,
            timexes,
            events_by_sentence.get(sent_key_value, []),
            syntax_data.get(sent_key_value, []),
            doc_date_value,
        )

        payload["sentences"].setdefault(doc_id, {})
        payload["sentences"][doc_id][sent_id] = {
            "text": text,
            "doc_date": doc_date_value,
            "timexes": timexes,
            "event_time_links": event_links,
            "event_event_relations": [],
        }

    attach_event_event_relations(payload, events_by_sentence)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"[OK] wrote_time_cache={out_path}")
    print_summary(payload, args.debug_samples)


if __name__ == "__main__":
    main()
