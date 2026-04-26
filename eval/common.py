#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for thesis evaluation scripts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def load_jsonl_lines(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = []
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{idx}") from exc
    return rows


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _iter_events_from_obj(obj: dict) -> Iterable[dict]:
    if not isinstance(obj, dict):
        return

    sentence_meta = {
        "doc_id": obj.get("doc_id", "batch_run"),
        "sent_id": obj.get("sent_id", 0),
    }

    if isinstance(obj.get("events"), list):
        for event in obj["events"]:
            if not isinstance(event, dict):
                continue
            row = dict(event)
            for k, v in sentence_meta.items():
                row.setdefault(k, v)
            yield row
        return

    if "frame" in obj:
        row = dict(obj)
        for k, v in sentence_meta.items():
            row.setdefault(k, v)
        yield row


def load_event_records(path: Path) -> List[dict]:
    records = load_jsonl_lines(path)
    events = []
    for rec in records:
        events.extend(_iter_events_from_obj(rec))
    return events


def extract_span(obj: dict, prefer: str = "char") -> Optional[Tuple[int, int]]:
    if not isinstance(obj, dict):
        return None

    if prefer == "char":
        for key in ("char_span", "span"):
            span = obj.get(key)
            if _is_span_pair(span):
                return tuple(span)
        return None

    span = obj.get("span")
    if _is_span_pair(span):
        return tuple(span)
    span = obj.get("char_span")
    if _is_span_pair(span):
        return tuple(span)
    return None


def extract_span_candidates(obj: dict) -> List[Tuple[str, Tuple[int, int]]]:
    if not isinstance(obj, dict):
        return []

    spans: List[Tuple[str, Tuple[int, int]]] = []
    for key in ("char_span", "span"):
        span = obj.get(key)
        if _is_span_pair(span):
            spans.append((key, tuple(span)))
    return spans


def _is_span_pair(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
        and value[0] >= 0
        and value[1] >= 0
        and value[1] >= value[0]
    )


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def extract_arg_items(event: dict, include_text: bool = True) -> List[Tuple[str, Optional[Tuple[int, int]], str]]:
    args = event.get("arguments", {})
    if not isinstance(args, dict):
        return []

    items = []
    for role, payload in args.items():
        if not isinstance(payload, dict):
            continue
        span = extract_span(payload, prefer="char")
        text = normalize_text(payload.get("text", ""))
        if include_text:
            items.append((role, span, text))
        else:
            items.append((role, span, ""))
    return items


def match_events(
    gold: List[dict], pred: List[dict], mode: str = "strict"
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if mode not in {"strict", "lenient", "token_relaxed"}:
        raise ValueError(f"Unknown matching mode: {mode}")

    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, ev in enumerate(gold):
        key = (str(ev.get("doc_id", "batch_run")), str(ev.get("sent_id", 0)))
        buckets[key].append(i)

    pred_buckets = defaultdict(list)
    for j, ev in enumerate(pred):
        key = (str(ev.get("doc_id", "batch_run")), str(ev.get("sent_id", 0)))
        pred_buckets[key].append(j)

    matched = []
    used_gold = set()
    used_pred = set()

    sentence_keys = set(buckets.keys()) | set(pred_buckets.keys())
    for key in sentence_keys:
        g_idx = [i for i in buckets.get(key, [])]
        p_idx = [j for j in pred_buckets.get(key, [])]
        if not g_idx or not p_idx:
            continue

        candidates = []
        for gi in g_idx:
            g_ev = gold[gi]
            g_frame = str(g_ev.get("frame", ""))
            g_trigger = g_ev.get("trigger", {})
            g_span = extract_span(g_trigger, prefer="char")
            g_span_candidates = {t: s for t, s in extract_span_candidates(g_trigger)}
            for pi in p_idx:
                p_ev = pred[pi]
                p_frame = str(p_ev.get("frame", ""))
                p_span = extract_span(p_ev.get("trigger", {}), prefer="char")
                p_trigger = p_ev.get("trigger", {})
                p_span_candidates = {t: s for t, s in extract_span_candidates(p_trigger)}

                score = 0
                if g_frame and p_frame and g_frame == p_frame:
                    score += 4

                # Strict mode: keep current char-centric behavior.
                if mode == "strict":
                    if g_span and p_span:
                        if g_span == p_span:
                            score += 6
                        elif spans_overlap(g_span, p_span):
                            score += 2
                    elif g_span is None and p_span is None:
                        score += 1
                else:
                    # lenient/token_relaxed mode: tolerate mixed span conventions.
                    if g_span == p_span and g_span is not None:
                        score += 6
                    else:
                        # Match by token or char alternatives.
                        if any(gs == ps for gs in g_span_candidates.values() for ps in p_span_candidates.values()):
                            score += 6
                        elif any(
                            spans_overlap(gs, ps)
                            for gs in g_span_candidates.values()
                            for ps in p_span_candidates.values()
                        ):
                            score += 3

                    # trigger text fallback for noisy span annotation
                    g_text = normalize_text(str(g_trigger.get("text", "")))
                    p_text = normalize_text(str(p_trigger.get("text", "")))
                    if g_text and p_text and g_text == p_text:
                        score += 2

                candidates.append((score, gi, pi))

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        for score, gi, pi in candidates:
            if score <= 0 or gi in used_gold or pi in used_pred:
                continue
            used_gold.add(gi)
            used_pred.add(pi)
            matched.append((gi, pi))

    unmatched_gold = [i for i in range(len(gold)) if i not in used_gold]
    unmatched_pred = [i for i in range(len(pred)) if i not in used_pred]
    return matched, unmatched_gold, unmatched_pred


def count_frames_with_schema(schemas_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    if not schemas_dir.exists():
        return out

    for path in sorted(schemas_dir.glob("*.schema.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        frame_id = path.stem.replace(".schema", "")
        if "properties" in data and "frame" in data["properties"]:
            frame_id = str(data["properties"]["frame"].get("const", frame_id))
        arg_names = []
        args = data.get("properties", {}).get("arguments", {}).get("properties", {})
        if isinstance(args, dict):
            arg_names = sorted(args.keys())
        out[frame_id] = {"roles": arg_names}
    return out


def read_sameas_links(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []

    links = []
    pattern = re.compile(r"^\s*<([^>]+)>\s+<[^>]+>\s+<([^>]+)>\s+<[^>]+>\s*\.$")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue
            subj, obj = m.group(1), m.group(2)
            links.append((subj, obj))
    return links


def read_rdf_labels(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    out: Dict[str, str] = {}
    pattern = re.compile(r"^\s*<([^>]+)>\s+<[^>]+(?:#label|/label)>\s+\"([^\"]*)\"(?:@[^\s>]+)?\s+<[^>]+>\s*\.$")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            subj, label = m.group(1), m.group(2)
            if subj not in out:
                out[subj] = label
    return out
