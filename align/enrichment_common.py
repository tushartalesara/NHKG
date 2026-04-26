#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for additive enrichment layers."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import unicodedata
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


NS = {
    "nhkg": "http://ns.nhkg.org/resource/",
    "uhvn": "http://ns.nhkg.org/uhvn/",
    "schema": "https://schema.org/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": "http://www.w3.org/ns/prov#",
    "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "wn": "http://ns.nhkg.org/wordnet/",
}


NQ_RE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.*)\s+<([^>]+)>\s+\.$")
OBJ_URI_RE = re.compile(r"^<([^>]+)>$")
OBJ_LIT_RE = re.compile(r'^("(?:\\.|[^"\\])*")(?:@([A-Za-z][A-Za-z0-9-]*)|\^\^<([^>]+)>)?$')
SURROUNDING_PUNCT_RE = re.compile(r'^[\s"\'“”‘’`.,;:!?()\[\]{}<>|/\\\-]+|[\s"\'“”‘’`.,;:!?()\[\]{}<>|/\\\-]+$')

HINDI_ENTITY_POSTPOSITIONS = (
    "के लिए",
    "के पास",
    "के साथ",
    "के ऊपर",
    "के नीचे",
    "के भीतर",
    "की ओर",
    "में",
    "से",
    "को",
    "पर",
    "तक",
    "ने",
)

HINDI_HONORIFICS = (
    "श्री",
    "सुश्री",
    "श्रीमती",
    "डॉ",
    "डा",
    "प्रोफेसर",
    "प्रो.",
    "स्वर्गीय",
)


@dataclass(frozen=True)
class ParsedObject:
    kind: str
    value: Any
    lang: Optional[str] = None
    datatype: Optional[str] = None


@dataclass(frozen=True)
class ParsedQuad:
    subject: str
    predicate: str
    obj: ParsedObject
    graph: str


def clean_uri(text: object) -> str:
    return urllib.parse.quote(str(text).strip().replace(" ", "_"))


def json_literal(text: object) -> str:
    return json.dumps(str(text), ensure_ascii=False)


def serialize_nquad_uri(value: object) -> str:
    uri = str(value or "").strip()
    if not uri:
        raise ValueError("URI term cannot be empty")
    return f"<{uri}>"


def serialize_nquad_literal(
    literal: object,
    *,
    lang: Optional[str] = None,
    datatype: Optional[str] = None,
) -> str:
    if lang and datatype:
        raise ValueError("Use either lang or datatype, not both")

    if datatype == f"{NS['xsd']}boolean":
        lexical = "true" if bool(literal) else "false"
        return f'{json.dumps(lexical, ensure_ascii=False)}^^<{datatype}>'
    if datatype == f"{NS['xsd']}integer":
        lexical = str(int(literal))
        return f'{json.dumps(lexical, ensure_ascii=False)}^^<{datatype}>'
    if datatype in {f"{NS['xsd']}decimal", f"{NS['xsd']}double"}:
        try:
            number = Decimal(str(literal))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric literal for {datatype}: {literal!r}") from exc
        lexical = format(number, "f")
        if "." not in lexical and datatype == f"{NS['xsd']}decimal":
            lexical = f"{lexical}.0"
        return f'{json.dumps(lexical, ensure_ascii=False)}^^<{datatype}>'

    payload = json.dumps(str(literal), ensure_ascii=False)
    if lang:
        return f"{payload}@{lang}"
    if datatype:
        return f"{payload}^^<{datatype}>"
    return payload


def to_nquad_uri(subject: str, predicate: str, obj_uri: str, graph: str) -> str:
    return f"{serialize_nquad_uri(subject)} {serialize_nquad_uri(predicate)} {serialize_nquad_uri(obj_uri)} {serialize_nquad_uri(graph)} ."


def to_nquad_literal(
    subject: str,
    predicate: str,
    literal: object,
    graph: str,
    *,
    lang: Optional[str] = None,
    datatype: Optional[str] = None,
) -> str:
    obj = serialize_nquad_literal(literal, lang=lang, datatype=datatype)
    return f"{serialize_nquad_uri(subject)} {serialize_nquad_uri(predicate)} {obj} {serialize_nquad_uri(graph)} ."


def to_nquad_int(subject: str, predicate: str, value: int, graph: str) -> str:
    return to_nquad_literal(subject, predicate, int(value), graph, datatype=f"{NS['xsd']}integer")


def to_nquad_decimal(subject: str, predicate: str, value: float, graph: str) -> str:
    return to_nquad_literal(subject, predicate, f"{float(value):.6f}", graph, datatype=f"{NS['xsd']}decimal")


def to_nquad_bool(subject: str, predicate: str, value: bool, graph: str) -> str:
    datatype = f"{NS['xsd']}boolean"
    return f"{serialize_nquad_uri(subject)} {serialize_nquad_uri(predicate)} {serialize_nquad_literal(value, datatype=datatype)} {serialize_nquad_uri(graph)} ."


def parse_nquad_line(line: str) -> Optional[ParsedQuad]:
    raw = line.strip()
    if not raw:
        return None
    match = NQ_RE.match(raw)
    if not match:
        return None
    subject, predicate, obj_raw, graph = match.groups()
    obj = parse_nquad_object(obj_raw)
    if obj is None:
        return None
    return ParsedQuad(subject=subject, predicate=predicate, obj=obj, graph=graph)


def parse_nquad_object(raw: str) -> Optional[ParsedObject]:
    uri_match = OBJ_URI_RE.match(raw)
    if uri_match:
        return ParsedObject(kind="uri", value=uri_match.group(1))

    lit_match = OBJ_LIT_RE.match(raw)
    if not lit_match:
        return None

    quoted, lang, datatype = lit_match.groups()
    try:
        value = json.loads(quoted)
    except json.JSONDecodeError:
        return None
    return ParsedObject(kind="literal", value=value, lang=lang, datatype=datatype)


def sentence_key(doc_id: object, sent_id: object) -> str:
    return f"{doc_id or 'batch_run'}::{sent_id}"


def split_sentence_key(key: str) -> Tuple[str, str]:
    if "::" not in key:
        return str(key), "0"
    doc_id, sent_id = key.split("::", 1)
    return doc_id, sent_id


def sentence_uri(doc_id: object, sent_id: object) -> str:
    return f"{NS['nhkg']}Sentence_{clean_uri(doc_id)}_{sent_id}"


def sentence_uri_for_key(key: str) -> str:
    doc_id, sent_id = split_sentence_key(key)
    return sentence_uri(doc_id, sent_id)


def parse_sentence_key_from_uri(uri: str) -> Optional[str]:
    local = uri.rsplit("/", 1)[-1]
    if not local.startswith("Sentence_"):
        return None
    payload = local[len("Sentence_") :]
    if "_" not in payload:
        return None
    doc_part, sent_part = payload.rsplit("_", 1)
    if not sent_part:
        return None
    return urllib.parse.unquote(f"{doc_part}::{sent_part}")


def event_uri(doc_id: object, sent_id: object, event_id_value: object) -> str:
    return f"{NS['nhkg']}Event_{clean_uri(doc_id)}_{sent_id}_{clean_uri(event_id_value)}"


def trigger_uri(event_id_value: object) -> str:
    return f"{NS['nhkg']}Trigger_{clean_uri(event_id_value)}"


def mention_uri(event_id_value: object, role: object) -> str:
    return f"{NS['nhkg']}Mention_{clean_uri(event_id_value)}_{clean_uri(role)}"


def word_uri(sentence_key_value: str, token_index: int, start: int, end: int) -> str:
    return f"{NS['nhkg']}Word_{clean_uri(sentence_key_value)}_{int(token_index)}_{int(start)}_{int(end)}"


def entity_mention_uri(sentence_key_value: str, start: int, end: int, label: str) -> str:
    return f"{NS['nhkg']}EntityMention_{clean_uri(sentence_key_value)}_{clean_uri(label)}_{int(start)}_{int(end)}"


def timex_uri(sentence_key_value: str, start: Optional[int], end: Optional[int], timex_id: str = "") -> str:
    if timex_id:
        return f"{NS['nhkg']}Timex_{clean_uri(timex_id)}"
    if start is None or end is None:
        return f"{NS['nhkg']}Timex_{clean_uri(sentence_key_value)}_docdate"
    return f"{NS['nhkg']}Timex_{clean_uri(sentence_key_value)}_{int(start)}_{int(end)}"


def normalize_char_span(value: object) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < start:
        return None
    return start, end


def normalize_entity_text(text: object) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    raw = raw.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    raw = raw.replace("।", " ").replace("॥", " ")
    raw = SURROUNDING_PUNCT_RE.sub("", raw)
    return " ".join(raw.split()).strip()


def strip_entity_postpositions(text: object) -> str:
    value = normalize_entity_text(text)
    if not value:
        return ""
    lowered = value
    for suffix in sorted(HINDI_ENTITY_POSTPOSITIONS, key=len, reverse=True):
        marker = f" {suffix}"
        if lowered.endswith(marker):
            trimmed = lowered[: -len(marker)].strip()
            if trimmed:
                return trimmed
    words = lowered.split()
    while len(words) > 1 and words[-1] in HINDI_ENTITY_POSTPOSITIONS:
        words = words[:-1]
    return " ".join(words).strip()


def strip_entity_honorifics(text: object) -> str:
    value = normalize_entity_text(text)
    if not value:
        return ""
    words = value.split()
    while words and words[0] in HINDI_HONORIFICS:
        words = words[1:]
    return " ".join(words).strip()


def build_entity_text_forms(text: object) -> Dict[str, object]:
    raw_text = normalize_entity_text(text)
    normalized_text = raw_text
    no_honorific = strip_entity_honorifics(raw_text)
    cleaned_text = strip_entity_postpositions(raw_text)
    cleaned_no_honorific = strip_entity_postpositions(no_honorific)

    alternate_forms: List[str] = []
    for candidate in (no_honorific, cleaned_text, cleaned_no_honorific):
        candidate = normalize_entity_text(candidate)
        if candidate and candidate != raw_text and candidate not in alternate_forms:
            alternate_forms.append(candidate)

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "cleaned_text": cleaned_text or raw_text,
        "alternate_forms": alternate_forms,
    }


def spans_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return not (left[1] <= right[0] or right[1] <= left[0])


def span_distance(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    if spans_overlap(left, right):
        return 0
    if left[1] <= right[0]:
        return right[0] - left[1]
    return left[0] - right[1]


def iter_events(item: object) -> Iterator[dict]:
    if isinstance(item, dict) and isinstance(item.get("events"), list):
        doc_id = item.get("doc_id", "batch_run")
        sent_id = item.get("sent_id", 0)
        for event in item["events"]:
            if not isinstance(event, dict):
                continue
            row = dict(event)
            row.setdefault("doc_id", doc_id)
            row.setdefault("sent_id", sent_id)
            yield row
        return

    if isinstance(item, dict):
        yield item


def iter_input_events(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if isinstance(payload, list):
                for item in payload:
                    yield from iter_events(item)
            return

        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            yield from iter_events(row)


def stable_event_id(event: Dict[str, Any]) -> str:
    existing = event.get("event_id")
    if existing:
        return str(existing)
    seed = "|".join(
        [
            str(event.get("doc_id", "")),
            str(event.get("sent_id", "")),
            str(event.get("frame", "")),
            str((event.get("trigger", {}) or {}).get("text", "")),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def collect_sentence_keys(input_path: Optional[Path]) -> List[str]:
    if input_path is None:
        return []
    keys: List[str] = []
    seen = set()
    for event in iter_input_events(input_path):
        key = sentence_key(event.get("doc_id", "batch_run"), event.get("sent_id", 0))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def load_sentence_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def load_sentence_text_map(
    sentence_file: Path,
    *,
    sentence_keys: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    lines = load_sentence_lines(sentence_file)
    if not lines:
        return {}

    ordered_keys = list(sentence_keys or [])
    if not ordered_keys:
        return {sentence_key("batch_run", idx): text for idx, text in enumerate(lines)}

    if len(lines) == len(ordered_keys):
        return {key: lines[idx] for idx, key in enumerate(ordered_keys)}

    by_key: Dict[str, str] = {}
    for key in ordered_keys:
        _, sent_id = split_sentence_key(key)
        try:
            idx = int(sent_id)
        except ValueError:
            idx = 0
        if 0 <= idx < len(lines):
            by_key[key] = lines[idx]
        elif lines:
            by_key[key] = lines[-1]
    return by_key


def sentence_texts_from_lines(path: Path) -> Dict[str, str]:
    return load_sentence_text_map(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
