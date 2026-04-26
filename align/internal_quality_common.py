#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for NHKG internal quality-improvement stages."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

try:
    from .enrichment_common import NS, clean_uri, normalize_char_span, spans_overlap
except ImportError:  # pragma: no cover
    from enrichment_common import NS, clean_uri, normalize_char_span, spans_overlap


STAGE_CACHE_SCHEMA_VERSION = "nhkg.stage-cache.v1"
DEFAULT_STAGE_VERSION = "1.0.0"

OUTER_PUNCTUATION = " \t\r\n,;:!?।॥()[]{}\"'`“”‘’<>|/"
MULTIWORD_POSTPOSITIONS = (
    "के लिए",
    "के पास",
    "के साथ",
    "की तरफ",
    "की ओर",
    "के बाद",
    "के पहले",
)
SINGLE_POSTPOSITIONS = (
    "में",
    "से",
    "को",
    "पर",
    "तक",
    "ने",
    "का",
    "की",
    "के",
)
HONORIFIC_PREFIXES = (
    "श्री",
    "सुश्री",
    "श्रीमती",
    "डॉ",
    "डा",
    "प्रो",
    "प्रो.",
    "प्रोफेसर",
    "स्वर्गीय",
)

PRONOUN_PROFILES: Dict[str, Dict[str, object]] = {
    "वह": {"number": "singular", "personish": True},
    "उसने": {"number": "singular", "personish": True},
    "उसे": {"number": "singular", "personish": True},
    "उसको": {"number": "singular", "personish": True},
    "उसका": {"number": "singular", "personish": True},
    "उसकी": {"number": "singular", "personish": True},
    "उसके": {"number": "singular", "personish": True},
    "वे": {"number": "plural", "personish": True},
    "उन्होंने": {"number": "plural", "personish": True},
    "उन्हें": {"number": "plural", "personish": True},
    "उनको": {"number": "plural", "personish": True},
    "उनका": {"number": "plural", "personish": True},
    "उनकी": {"number": "plural", "personish": True},
    "उनके": {"number": "plural", "personish": True},
    "यह": {"number": "singular", "personish": False},
    "इसने": {"number": "singular", "personish": True},
    "इसे": {"number": "singular", "personish": True},
    "इसको": {"number": "singular", "personish": True},
    "इसका": {"number": "singular", "personish": True},
    "इसकी": {"number": "singular", "personish": True},
    "इसके": {"number": "singular", "personish": True},
    "ये": {"number": "plural", "personish": True},
    "इन्होंने": {"number": "plural", "personish": True},
    "इन्हें": {"number": "plural", "personish": True},
    "इनको": {"number": "plural", "personish": True},
    "इनका": {"number": "plural", "personish": True},
    "इनकी": {"number": "plural", "personish": True},
    "इनके": {"number": "plural", "personish": True},
}

ENTITY_TYPE_TO_SCHEMA = {
    "PER": f"{NS['schema']}Person",
    "LOC": f"{NS['schema']}Place",
    "ORG": f"{NS['schema']}Organization",
}

ROLE_PERSONISH = {
    "agent",
    "actor",
    "experiencer",
    "subject",
    "owner",
    "speaker",
    "leader",
}
ROLE_LOCATIVE = {
    "destination",
    "location",
    "source",
    "origin",
    "place",
    "venue",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Optional[Path], default=None):
    if path is None or not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml_config(path: Optional[Path], default: Optional[dict] = None) -> dict:
    if path is None or not path.exists():
        return dict(default or {})
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else dict(default or {})


def snapshot_paths(paths: Dict[str, object]) -> Dict[str, dict]:
    snapshot: Dict[str, dict] = {}
    for key, raw_path in paths.items():
        if not raw_path:
            continue
        path = Path(str(raw_path))
        item = {"path": str(path)}
        if path.exists():
            stat = path.stat()
            item.update(
                {
                    "exists": True,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        else:
            item["exists"] = False
        snapshot[str(key)] = item
    return snapshot


def build_stage_metadata(
    *,
    stage_name: str,
    stage_version: str,
    engine: str,
    source_paths: Dict[str, object],
    input_counts: Optional[Dict[str, object]] = None,
    warnings: Optional[Sequence[str]] = None,
    config_path: Optional[str] = None,
    config_snapshot: Optional[dict] = None,
    extra: Optional[Dict[str, object]] = None,
) -> dict:
    meta = {
        "schema_version": STAGE_CACHE_SCHEMA_VERSION,
        "stage_name": stage_name,
        "stage_version": stage_version,
        "engine": engine,
        "created_at": utc_now_iso(),
        "source_paths": snapshot_paths(source_paths),
        "input_counts": dict(input_counts or {}),
        "warnings": list(warnings or []),
    }
    if config_path:
        meta["config_path"] = str(config_path)
    if config_snapshot is not None:
        meta["config_snapshot"] = config_snapshot
    if extra:
        meta.update(extra)
    return meta


def normalize_text(text: object) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    value = value.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def trim_outer_punctuation(text: object) -> str:
    return normalize_text(text).strip(OUTER_PUNCTUATION)


def strip_honorific_prefixes(text: object) -> Tuple[str, List[str]]:
    value = trim_outer_punctuation(text)
    actions: List[str] = []
    changed = True
    while changed and value:
        changed = False
        for honorific in HONORIFIC_PREFIXES:
            if value == honorific:
                continue
            if value.startswith(honorific + " "):
                value = value[len(honorific) :].strip()
                actions.append(f"removed_honorific:{honorific}")
                changed = True
                break
    return value, actions


def strip_postposition_suffixes(text: object) -> Tuple[str, List[str]]:
    value = trim_outer_punctuation(text)
    actions: List[str] = []
    changed = True
    while changed and value:
        changed = False
        for postposition in MULTIWORD_POSTPOSITIONS:
            if value.endswith(" " + postposition):
                value = value[: -(len(postposition) + 1)].strip()
                actions.append(f"removed_postposition:{postposition}")
                changed = True
                break
        if changed:
            continue
        parts = value.split()
        if parts and parts[-1] in SINGLE_POSTPOSITIONS:
            actions.append(f"removed_postposition:{parts[-1]}")
            value = " ".join(parts[:-1]).strip()
            changed = True
    return value, actions


def build_mention_forms(text: object) -> dict:
    raw = normalize_text(text)
    normalized = trim_outer_punctuation(raw)
    honorific_stripped, honorific_actions = strip_honorific_prefixes(normalized)
    postposition_stripped, postposition_actions = strip_postposition_suffixes(normalized)
    cleaned = honorific_stripped
    if postposition_stripped and len(postposition_stripped) >= len(cleaned):
        cleaned = postposition_stripped
    if honorific_stripped:
        stripped_again, extra_actions = strip_postposition_suffixes(honorific_stripped)
        if stripped_again:
            cleaned = stripped_again
            postposition_actions.extend(extra_actions)
    alternate_forms = []
    for candidate in (normalized, honorific_stripped, postposition_stripped, cleaned):
        candidate = trim_outer_punctuation(candidate)
        if candidate:
            alternate_forms.append(candidate)
    deduped_alternates = []
    seen = set()
    for candidate in alternate_forms:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped_alternates.append(candidate)
    return {
        "raw_text": raw,
        "normalized_text": normalized,
        "cleaned_text": trim_outer_punctuation(cleaned or normalized),
        "alternate_forms": deduped_alternates,
        "normalization_actions": honorific_actions + postposition_actions,
    }


def token_span(token: dict) -> Optional[Tuple[int, int]]:
    return normalize_char_span([token.get("start"), token.get("end")]) if isinstance(token, dict) else None


def overlapping_tokens(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> List[dict]:
    if span is None:
        return []
    out = []
    for token in tokens:
        offsets = token_span(token)
        if offsets is None:
            continue
        if spans_overlap(offsets, span):
            out.append(token)
    return out


def select_head_token(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> Optional[dict]:
    overlaps = overlapping_tokens(tokens, span)
    if not overlaps:
        return None
    priority = {
        "PROPN": 0,
        "NOUN": 1,
        "VERB": 2,
        "ADJ": 3,
        "PRON": 4,
        "ADV": 5,
        "NUM": 6,
        "AUX": 7,
    }
    best = sorted(
        overlaps,
        key=lambda token: (
            priority.get(str(token.get("upos", "")).upper(), 99),
            0 if str(token.get("deprel", "")).lower() in {"root", "nsubj", "obj", "obl", "compound", "appos"} else 1,
            -(min(token_span(token)[1], span[1]) - max(token_span(token)[0], span[0])) if token_span(token) and span else 0,
            int(token.get("word_index", 0) or 0),
        ),
    )
    return best[0] if best else None


def head_summary(token: Optional[dict]) -> dict:
    if not isinstance(token, dict):
        return {
            "text": "",
            "lemma": "",
            "upos": "",
            "xpos": "",
            "deprel": "",
            "head": None,
            "start": None,
            "end": None,
            "word_index": None,
        }
    return {
        "text": normalize_text(token.get("text", "")),
        "lemma": normalize_text(token.get("lemma", token.get("text", ""))),
        "upos": str(token.get("upos", "")).upper(),
        "xpos": str(token.get("xpos", "")),
        "deprel": str(token.get("deprel", "")),
        "head": token.get("head"),
        "start": token.get("start"),
        "end": token.get("end"),
        "word_index": token.get("word_index"),
    }


def cluster_uri(doc_id: object, cluster_id: str) -> str:
    return f"{NS['nhkg']}CanonicalEntity_{clean_uri(str(doc_id))}_{clean_uri(cluster_id)}"


def temporal_relation_uri(doc_id: object, relation_id: str) -> str:
    return f"{NS['nhkg']}TemporalRelation_{clean_uri(str(doc_id))}_{clean_uri(relation_id)}"


def type_to_schema_uri(entity_type: str) -> str:
    return ENTITY_TYPE_TO_SCHEMA.get(str(entity_type or "").upper(), "")


def get_pronoun_profile(text: object) -> Optional[dict]:
    return PRONOUN_PROFILES.get(trim_outer_punctuation(text))


def is_numeric_text(text: object) -> bool:
    value = trim_outer_punctuation(text)
    return bool(value) and bool(re.fullmatch(r"[0-9०-९]+(?:[.,][0-9०-९]+)?", value))


def role_to_type_hint(role: object, config: Optional[dict] = None) -> str:
    role_name = str(role or "").strip().lower()
    configured = ((config or {}).get("role_type_hints", {}) or {}).get(role_name)
    if configured:
        return str(configured).upper()
    if role_name in ROLE_PERSONISH:
        return "PER"
    if role_name in ROLE_LOCATIVE:
        return "LOC"
    return "MISC"


def sentence_distance(left_sent_id: object, right_sent_id: object) -> int:
    try:
        return abs(int(left_sent_id) - int(right_sent_id))
    except (TypeError, ValueError):
        return 9999


def context_excerpt(sentence: str, start: Optional[int], end: Optional[int], radius: int = 30) -> str:
    if start is None or end is None:
        return normalize_text(sentence)
    sentence = sentence or ""
    left = max(0, int(start) - radius)
    right = min(len(sentence), int(end) + radius)
    return normalize_text(sentence[left:right])


def write_flat_csv(path: Optional[Path], rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
