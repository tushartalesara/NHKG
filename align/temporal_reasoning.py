#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conservative event-event temporal reasoning for NHKG."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .enrichment_common import normalize_char_span
except ImportError:  # pragma: no cover
    from enrichment_common import normalize_char_span


FORWARD_SEQUENCE_MARKERS = ("उसके बाद", "फिर", "बाद में", "तत्पश्चात", "अगले दिन")
SIMULTANEOUS_MARKERS = ("एक साथ", "उसी समय", "साथ ही", "एकसाथ")


def _parse_date_interval(value: str) -> Optional[Tuple[date, date]]:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        point = datetime.strptime(value, "%Y-%m-%d").date()
        return point, point
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        return start, end
    if re.fullmatch(r"\d{4}", value):
        year = int(value)
        return date(year, 1, 1), date(year, 12, 31)
    week_match = re.fullmatch(r"(\d{4})-W(\d{2})", value)
    if week_match:
        year, week = int(week_match.group(1)), int(week_match.group(2))
        start = datetime.fromisocalendar(year, week, 1).date()
        end = datetime.fromisocalendar(year, week, 7).date()
        return start, end
    return None


def compare_timex_values(left_value: Optional[str], right_value: Optional[str]) -> Optional[str]:
    if not left_value or not right_value:
        return None
    if left_value == right_value:
        return "SHARED_TIME_ANCHOR"
    if left_value.startswith("T") and right_value.startswith("T"):
        return "BEFORE" if left_value < right_value else "AFTER"
    left_interval = _parse_date_interval(left_value)
    right_interval = _parse_date_interval(right_value)
    if left_interval is None or right_interval is None:
        return None
    if left_interval[1] < right_interval[0]:
        return "BEFORE"
    if left_interval[0] > right_interval[1]:
        return "AFTER"
    if left_interval == right_interval:
        return "SHARED_TIME_ANCHOR"
    if left_interval[0] <= right_interval[0] and left_interval[1] >= right_interval[1]:
        return "INCLUDES"
    if right_interval[0] <= left_interval[0] and right_interval[1] >= left_interval[1]:
        return "OVERLAPS"
    return "OVERLAPS"


def _trigger_start(event: dict) -> int:
    span = normalize_char_span((event.get("trigger", {}) or {}).get("char_span"))
    return span[0] if span is not None else 10**6


def infer_same_sentence_relations(*, sent_key: str, text: str, events: Sequence[dict], event_to_timex: Dict[str, dict]) -> List[dict]:
    relations: List[dict] = []
    if len(events) < 2:
        return relations
    ordered_events = sorted(events, key=_trigger_start)
    lowered_text = str(text or "")
    for index, left in enumerate(ordered_events):
        left_id = str(left.get("event_id", ""))
        for right in ordered_events[index + 1 :]:
            right_id = str(right.get("event_id", ""))
            if not left_id or not right_id:
                continue
            left_timex = event_to_timex.get(left_id, {})
            right_timex = event_to_timex.get(right_id, {})
            relation = compare_timex_values(left_timex.get("value"), right_timex.get("value"))
            strategy = ""
            evidence = ""
            reason = ""
            evidence_strength = "weak"
            confidence = 0.0
            if relation:
                strategy = "timex_value_order"
                evidence = f"{left_timex.get('value')} vs {right_timex.get('value')}"
                if relation == "SHARED_TIME_ANCHOR":
                    confidence = 0.68
                    evidence_strength = "weak"
                    reason = "events share a normalized timex value but no explicit simultaneity cue was found"
                else:
                    confidence = 0.88
                    evidence_strength = "strong"
                    reason = "normalized timex values provide a direct temporal ordering signal"
            else:
                left_span = normalize_char_span((left.get("trigger", {}) or {}).get("char_span"))
                right_span = normalize_char_span((right.get("trigger", {}) or {}).get("char_span"))
                between = lowered_text[left_span[1] : right_span[0]] if left_span and right_span and left_span[1] <= right_span[0] else ""
                if any(marker in between for marker in FORWARD_SEQUENCE_MARKERS):
                    relation = "BEFORE"
                    strategy = "discourse_marker"
                    evidence = between.strip() or "forward_sequence_marker"
                    confidence = 0.77
                    evidence_strength = "strong"
                    reason = "explicit forward-sequence discourse marker links the events"
                elif any(marker in between for marker in SIMULTANEOUS_MARKERS):
                    relation = "SIMULTANEOUS"
                    strategy = "discourse_marker"
                    evidence = between.strip() or "simultaneous_marker"
                    confidence = 0.8
                    evidence_strength = "strong"
                    reason = "explicit simultaneity discourse marker links the events"
            if not relation:
                continue
            relation_id = f"{sent_key}::{left_id}::{right_id}::{relation}"
            relations.append(
                {
                    "relation_id": relation_id,
                    "source_event_id": left_id,
                    "target_event_id": right_id,
                    "relation": relation,
                    "strategy": strategy,
                    "confidence": round(confidence, 4),
                    "evidence": evidence,
                    "evidence_strength": evidence_strength,
                    "reason": reason,
                }
            )
    return relations


def infer_cross_sentence_relations(document_rows: Sequence[dict]) -> List[dict]:
    relations: List[dict] = []
    ordered_rows = sorted(document_rows, key=lambda item: (int(item.get("sent_id", 0)) if str(item.get("sent_id", "")).isdigit() else 10**6, item.get("sent_key", "")))
    for previous, current in zip(ordered_rows, ordered_rows[1:]):
        previous_events = previous.get("events", [])
        current_events = current.get("events", [])
        if not previous_events or not current_events:
            continue
        current_text = str(current.get("text", "")).strip()
        if not current_text:
            continue
        first_event = sorted(previous_events, key=_trigger_start)[-1]
        second_event = sorted(current_events, key=_trigger_start)[0]
        prefix = current_text[:25]
        relation = None
        strategy = ""
        evidence = ""
        reason = ""
        evidence_strength = "weak"
        confidence = 0.0
        if any(prefix.startswith(marker) for marker in FORWARD_SEQUENCE_MARKERS):
            relation = "BEFORE"
            strategy = "cross_sentence_marker"
            evidence = prefix
            confidence = 0.74
            evidence_strength = "strong"
            reason = "sentence-initial forward marker links the previous event to the current event"
        else:
            prev_map = previous.get("event_to_timex", {})
            curr_map = current.get("event_to_timex", {})
            prev_timex = prev_map.get(str(first_event.get("event_id", "")), {})
            curr_timex = curr_map.get(str(second_event.get("event_id", "")), {})
            relation = compare_timex_values(prev_timex.get("value"), curr_timex.get("value"))
            if relation:
                strategy = "cross_sentence_timex"
                evidence = f"{prev_timex.get('value')} vs {curr_timex.get('value')}"
                if relation == "SHARED_TIME_ANCHOR":
                    confidence = 0.64
                    evidence_strength = "weak"
                    reason = "events share a normalized timex value across adjacent sentences"
                else:
                    confidence = 0.82
                    evidence_strength = "strong"
                    reason = "adjacent sentences provide comparable normalized timex values"
        if not relation:
            continue
        sent_key = str(current.get("sent_key", ""))
        relation_id = f"{sent_key}::{first_event.get('event_id', '')}::{second_event.get('event_id', '')}::{relation}"
        relations.append(
            {
                "relation_id": relation_id,
                "source_event_id": str(first_event.get("event_id", "")),
                "target_event_id": str(second_event.get("event_id", "")),
                "relation": relation,
                "strategy": strategy,
                "confidence": round(confidence, 4),
                "evidence": evidence,
                "evidence_strength": evidence_strength,
                "reason": reason,
            }
        )
    return relations


def infer_document_temporal_relations(document_rows: Sequence[dict]) -> List[dict]:
    relations: List[dict] = []
    for row in document_rows:
        relations.extend(infer_same_sentence_relations(sent_key=str(row.get("sent_key", "")), text=str(row.get("text", "")), events=row.get("events", []), event_to_timex=row.get("event_to_timex", {})))
    relations.extend(infer_cross_sentence_relations(document_rows))
    deduped: Dict[Tuple[str, str, str], dict] = {}
    for relation in relations:
        key = (str(relation.get("source_event_id", "")), str(relation.get("target_event_id", "")), str(relation.get("relation", "")))
        current = deduped.get(key)
        if current is None or float(relation.get("confidence", 0.0) or 0.0) >= float(current.get("confidence", 0.0) or 0.0):
            deduped[key] = relation
    return [deduped[key] for key in sorted(deduped.keys())]
