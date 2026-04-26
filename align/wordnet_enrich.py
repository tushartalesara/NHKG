#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hindi WordNet enrichment for mention/trigger nodes.

The script augments NHKG exports with lexical signals from Hindi WordNet:
- lemma normalization and stop-word filtering,
- multi-POS + multi-synset lookup with graceful API fallbacks,
- deduped synonyms and hypernyms,
- optional shared lexeme nodes so identical surface forms across sentences connect.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .enrichment_common import (
        to_nquad_decimal as shared_to_nquad_decimal,
        to_nquad_literal as shared_to_nquad_literal,
        to_nquad_uri as shared_to_nquad_uri,
    )
except ImportError:  # pragma: no cover
    from enrichment_common import (
        to_nquad_decimal as shared_to_nquad_decimal,
        to_nquad_literal as shared_to_nquad_literal,
        to_nquad_uri as shared_to_nquad_uri,
    )

try:
    import pyiwn
except ImportError:  # pragma: no cover
    pyiwn = None

try:
    import stanza
except ImportError:  # pragma: no cover
    stanza = None

try:
    from gold.span import Tokenizer as SpanTokenizer
except Exception:  # pragma: no cover
    SpanTokenizer = None


@dataclass(frozen=True)
class PosToken:
    text: str
    start: int
    end: int
    upos: str
    xpos: str = ""
    feats: str = ""
    lemma: str = ""

NS = {
    "nhkg": "http://ns.nhkg.org/resource/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "wn": "http://ns.nhkg.org/wordnet/",
}

DEFAULT_GRAPH = "http://ns.nhkg.org/graph/wordnet"
DEFAULT_MAX_SYNSETS = 2
DEFAULT_MAX_SYNONYMS = 8
DEFAULT_MAX_HYPERNYMS = 6
DEFAULT_MAX_CONTEXT_TOKENS = 24
DEFAULT_MIN_TERM_LENGTH = 2
SYNSCOPE_LITERAL = "subject"
SYNSCOPE_SYNSET = "synset"
DEFAULT_SYNONYM_SCOPE = SYNSCOPE_SYNSET

SENTENCE_TOKENIZER = SpanTokenizer() if SpanTokenizer is not None else None
SENTENCE_TOKEN_RE = re.compile(r"(\s+|[.,;!?।\"'`|()\[\]{}<>\\u0964\\u0965])")
STRIP_CHARS = " \t\r\n.,;:!?()[]{}\"'`|/\\-\u2013\u2014"

CASE_MARKER_TOKENS = {
    "है",
    "हैं",
    "है",
    "क्या",
    "तो",
    "ही",
    "थी",
    "थे",
}
POSTPOSITION_TOKENS = {
    "में",
    "से",
    "पर",
    "को",
    "तक",
    "की",
    "का",
    "ने",
    "का पास",
    "के साथ",
    "की तरफ",
    "के बाद",
    "के पहले",
}
STOPWORD_TOKENS = {
    "न",
    "नहीं",
    "और",
    "या",
    "यह",
    "उस",
    "जो",
    "इस",
    "जब",
    "वह",
    "वा",
    "जो",
    "वहाँ",
}


WSD_CONTEXT_WINDOW = 18
BANNED_TRIGGER_POS = {"PUNCT", "SYM"}
BANNED_ARGUMENT_POS = {
    "PRON",
    "ADP",
    "SCONJ",
    "CCONJ",
    "DET",
    "PART",
    "NUM",
    "SYM",
    "PUNCT",
    "INTJ",
}
DEFAULT_SKIP_PROPN = True


def clean_uri(text: str) -> str:
    return urllib.parse.quote(str(text).strip().replace(" ", "_"))


def json_str(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def to_nquad_uri(s: str, p: str, o: str, g: str) -> str:
    return shared_to_nquad_uri(s, p, o, g)


def to_nquad_lit(s: str, p: str, literal: str, g: str, lang: str = "hi") -> str:
    return shared_to_nquad_literal(s, p, literal, g, lang=lang)


def to_nquad_decimal(s: str, p: str, value: float, g: str) -> str:
    return shared_to_nquad_decimal(s, p, value, g)


def normalize_surface(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200d", "").replace("\u200c", "").replace("\ufeff", "")
    return " ".join(str(text).strip().split())


def normalize_token(text: str) -> str:
    if not text:
        return ""
    cleaned = normalize_surface(text)
    if not cleaned:
        return ""
    return cleaned.strip("".join(sorted(" \t\r\n.,;:!?()[]{}\"'`|/\\-\u2013\u2014")))


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenize text and return token offsets. Uses project tokenizer when available."""
    if SENTENCE_TOKENIZER is not None:
        tokens, offsets = SENTENCE_TOKENIZER.tokenize(text or "")
        return list(tokens), list(offsets)

    tokens = []
    offsets = []
    current_pos = 0
    for part in SENTENCE_TOKEN_RE.split(text or ""):
        if not part:
            continue
        if part.strip() == "":
            current_pos += len(part)
            continue
        start = current_pos
        end = current_pos + len(part)
        tokens.append(part)
        offsets.append((start, end))
        current_pos = end
    return tokens, offsets


def sentence_tokens_for_context(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    raw_tokens, offsets = tokenize_with_offsets(normalize_surface(text))
    filtered_tokens: List[str] = []
    filtered_offsets: List[Tuple[int, int]] = []
    for token, (start, end) in zip(raw_tokens, offsets):
        token_norm = normalize_token(token)
        if not token_norm:
            continue
        if token_norm in {",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'", "`", "|", "|", "।", "।"}:
            continue
        filtered_tokens.append(token_norm)
        filtered_offsets.append((start, end))
    return filtered_tokens, filtered_offsets


def context_window_indices(
    offsets: List[Tuple[int, int]],
    char_span: Optional[Tuple[int, int]],
    window: int,
) -> Tuple[int, int]:
    if not offsets or not char_span:
        return 0, len(offsets)

    start, end = char_span
    start_idx = None
    end_idx = None
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if not (tok_end <= start or tok_start >= end):
            if start_idx is None:
                start_idx = idx
            end_idx = idx + 1

    if start_idx is None:
        # nearest token fallback
        midpoint = (start + end) // 2
        nearest = min(
            range(len(offsets)),
            key=lambda i: min(abs(offsets[i][0] - midpoint), abs(offsets[i][1] - midpoint)),
        )
        start_idx = nearest
        end_idx = nearest + 1

    if window <= 0:
        return max(0, start_idx), min(len(offsets), end_idx)

    radius = max(1, window // 2)
    if len(offsets) <= window:
        return 0, len(offsets)

    start_idx = max(0, start_idx - radius)
    end_idx = min(len(offsets), end_idx + radius)
    if end_idx - start_idx < window:
        if start_idx == 0:
            end_idx = min(len(offsets), start_idx + window)
        else:
            start_idx = max(0, end_idx - window)
        return start_idx, end_idx

    if end_idx - start_idx > window:
        end_idx = start_idx + window

    return start_idx, end_idx


def tokenize_surface(text: str) -> List[str]:
    return [tok.strip(STRIP_CHARS) for tok in normalize_surface(text).split() if tok.strip(STRIP_CHARS)]


def canonical_token(text: str) -> str:
    return normalize_surface(text).casefold().strip()


def normalize_char_span(char_span: object) -> Optional[Tuple[int, int]]:
    if not isinstance(char_span, (list, tuple)) or len(char_span) != 2:
        return None
    start, end = char_span
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < 0 or end < start:
        return None
    return start, end


def spans_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return not (left[1] <= right[0] or right[1] <= left[0])


def _sentence_key(doc_id: object, sent_id: object) -> str:
    return f"{doc_id or 'batch_run'}::{sent_id}"


def load_sentence_texts(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Sentence text file not found: {path}")
    sentences: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sentences[_sentence_key("batch_run", idx)] = line.rstrip("\r\n")
    return sentences


def tokenise_for_wsd(text: str) -> List[str]:
    return [t for t in re.split(r"[\s" + re.escape(STRIP_CHARS) + "]+", normalize_surface(text)) if t]


def sentence_context_terms(text: str, char_span: Optional[Tuple[int, int]] = None, window: int = DEFAULT_MAX_CONTEXT_TOKENS) -> List[str]:
    tokens = tokenise_for_wsd(text)
    if not tokens:
        return []
    if char_span is None:
        return tokens if len(tokens) <= window else tokens[:window]

    token_context, offset_context = sentence_tokens_for_context(text)
    if not token_context:
        return tokens if len(tokens) <= window else tokens[:window]

    start, end = context_window_indices(offset_context, char_span, max(4, window))
    slice_tokens = token_context[start:end]
    if slice_tokens:
        return slice_tokens
    return tokens[: min(window, len(tokens))]


def sentence_key_lookup(
    event: dict,
    sentence_texts: Dict[str, str],
) -> Optional[str]:
    doc_id = event.get("doc_id", "batch_run")
    sent_id = event.get("sent_id", "")
    key = _sentence_key(doc_id, sent_id)
    sentence = sentence_texts.get(key)
    if sentence is not None:
        return sentence
    # Some files may not include doc id in a stable way.
    return sentence_texts.get(_sentence_key("batch_run", sent_id))


def load_pos_cache(path: Path) -> Dict[str, List[PosToken]]:
    if not path.exists():
        raise FileNotFoundError(f"POS cache not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, List[PosToken]] = {}
    if not isinstance(payload, dict):
        raise ValueError("POS cache must be JSON object")

    # Supported formats:
    # 1) {"batch_run::0": [{...}], ...}
    # 2) {"sentences": {"batch_run": {"0": [...]}}}
    # 3) {"batch_run": {"0": {"tokens": [...], "text": "..."}}}
    if "sentences" in payload and isinstance(payload["sentences"], dict):
        for doc, sent_map in payload["sentences"].items():
            if not isinstance(sent_map, dict):
                continue
            for sid, item in sent_map.items():
                tokens = item["tokens"] if isinstance(item, dict) else item
                if isinstance(tokens, list):
                    out[_sentence_key(doc, sid)] = _normalize_pos_token_list(tokens)
        return out

    for key, value in payload.items():
        if key == "meta":
            continue
        if isinstance(value, list):
            out[key] = _normalize_pos_token_list(value)
            continue
        if not isinstance(value, dict):
            continue
        for sid, item in value.items():
            tokens = item["tokens"] if isinstance(item, dict) else item
            if not isinstance(tokens, list):
                continue
            out[_sentence_key(key, sid)] = _normalize_pos_token_list(tokens)
    return out


def load_ner_cache(path: Path) -> Dict[str, List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"NER cache not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, List[dict]] = {}
    if not isinstance(payload, dict):
        return out

    for doc, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sid, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            entities = item.get("entities", [])
            if isinstance(entities, list):
                out[_sentence_key(doc, sid)] = [entity for entity in entities if isinstance(entity, dict)]
    return out


def load_time_cache(path: Path) -> Dict[str, List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Time cache not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, List[dict]] = {}
    if not isinstance(payload, dict):
        return out

    for doc, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sid, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            timexes = item.get("timexes", [])
            if isinstance(timexes, list):
                out[_sentence_key(doc, sid)] = [timex for timex in timexes if isinstance(timex, dict)]
    return out


def load_dbpedia_cache(path: Path) -> Dict[str, List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"DBpedia link cache not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, List[dict]] = {}
    if not isinstance(payload, dict):
        return out

    for doc, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sid, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            links = item.get("links", [])
            if isinstance(links, list):
                out[_sentence_key(doc, sid)] = [link for link in links if isinstance(link, dict)]
    return out


def lookup_sentence_annotations(annotation_map: Dict[str, List[dict]], doc_id: object, sent_id: object) -> List[dict]:
    candidates = [
        _sentence_key(doc_id, sent_id),
        _sentence_key("batch_run", sent_id),
    ]
    for key in candidates:
        values = annotation_map.get(key)
        if values is not None:
            return values
    return []


def span_overlaps_entity(
    char_span: Optional[Tuple[int, int]],
    entities: Sequence[dict],
    allowed_labels: Optional[Set[str]] = None,
) -> bool:
    if char_span is None:
        return False
    labels = allowed_labels or {"PER", "LOC", "ORG", "TIME", "NUM"}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        label = str(entity.get("label", "")).upper()
        if label not in labels:
            continue
        entity_span = normalize_char_span([entity.get("start"), entity.get("end")])
        if entity_span is None:
            continue
        if spans_overlap(char_span, entity_span):
            return True
    return False


def span_overlaps_timex(char_span: Optional[Tuple[int, int]], timexes: Sequence[dict]) -> bool:
    if char_span is None:
        return False
    for timex in timexes:
        if not isinstance(timex, dict):
            continue
        timex_span = normalize_char_span([timex.get("start"), timex.get("end")])
        if timex_span is None:
            continue
        if spans_overlap(char_span, timex_span):
            return True
    return False


def span_overlaps_dbpedia_link(char_span: Optional[Tuple[int, int]], links: Sequence[dict]) -> bool:
    if char_span is None:
        return False
    for link in links:
        if not isinstance(link, dict):
            continue
        link_span = normalize_char_span([link.get("start"), link.get("end")])
        if link_span is not None and spans_overlap(char_span, link_span):
            return True
    return False


def dbpedia_skip_reason(char_span: Optional[Tuple[int, int]], links: Sequence[dict]) -> str:
    if char_span is None:
        return ""
    for link in links:
        if not isinstance(link, dict):
            continue
        link_span = normalize_char_span([link.get("start"), link.get("end")])
        if link_span is None or not spans_overlap(char_span, link_span):
            continue
        ner_label = str(link.get("ner_label", "")).upper()
        if ner_label in {"PER", "LOC", "ORG"}:
            return "link"
        predicted_types = {str(item).rsplit("/", 1)[-1] for item in (link.get("predicted_dbo_types", []) or []) if str(item).strip()}
        if predicted_types & {"Person", "Place", "Organisation", "Organization", "Company"}:
            return "type"
    return ""


def _normalize_pos_token_list(tokens: Sequence[dict]) -> List[PosToken]:
    normalized: List[PosToken] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = normalize_surface(token.get("text", ""))
        if not text:
            continue
        start = token.get("start")
        end = token.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        feats = token.get("ufeats", token.get("feats", ""))
        normalized.append(
            PosToken(
                text=text,
                start=start,
                end=end,
                upos=str(token.get("upos", "")).upper(),
                xpos=str(token.get("xpos", "")),
                feats=str(feats or ""),
                lemma=normalize_surface(token.get("lemma", text)),
            )
        )
    return normalized


def get_token_for_span(tokens: List[PosToken], char_span: Optional[Tuple[int, int]]) -> Optional[PosToken]:
    if not tokens or not char_span:
        return None
    start, end = char_span
    overlapping: List[PosToken] = [
        t for t in tokens if not (t.end <= start or t.start >= end) and t.upos
    ]
    if overlapping:
        overlapping.sort(key=lambda t: (min(t.end, end) - max(t.start, start)), reverse=True)
        return overlapping[0]

    # Fallback to nearest token by start index.
    if not tokens:
        return None
    nearest = min(tokens, key=lambda t: abs(t.start - start))
    return nearest


def get_pos_from_span(tokens: List[PosToken], char_span: Optional[Tuple[int, int]]) -> Optional[str]:
    token = get_token_for_span(tokens, char_span)
    if token is None:
        return None
    return token.upos or None


def is_pos_allowed_for_wordnet(
    upos: Optional[str],
    kind: str,
    skip_propn: bool = DEFAULT_SKIP_PROPN,
    allowed_pos: Optional[Set[str]] = None,
) -> bool:
    if not upos:
        return True
    pos = upos.upper()
    if allowed_pos is not None:
        return pos in allowed_pos

    if kind == "trigger":
        return pos not in BANNED_TRIGGER_POS
    if kind == "argument":
        return pos not in BANNED_ARGUMENT_POS
    return True


def parse_pos_filter_list(raw: str) -> Optional[Set[str]]:
    if not raw:
        return None
    values: List[str] = []
    for token in raw.split(","):
        token = token.strip().upper()
        if token:
            values.append(token)
    if not values:
        return set()
    return set(values)


def wsd_context_signature(context_terms: List[str]) -> str:
    if not context_terms:
        return ""
    key = " ".join(term.casefold() for term in context_terms[:WSD_CONTEXT_WINDOW]).strip()
    if not key:
        return ""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def is_function_word(term: str) -> bool:
    term = canonical_token(term)
    if not term:
        return True
    return term in CASE_MARKER_TOKENS or term in POSTPOSITION_TOKENS or term in STOPWORD_TOKENS


def is_function_only(term: str) -> bool:
    tokens = tokenize_surface(term)
    if not tokens:
        return True
    return all(is_function_word(t) for t in tokens)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = normalize_surface(str(value))
        if not text:
            continue
        normalized = canonical_token(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(text)
    return out


def candidate_terms(surface: str, include_stop_terms: bool = False) -> List[str]:
    surface = normalize_surface(surface)
    if not surface:
        return []

    tokens = tokenize_surface(surface)
    if not tokens:
        return []

    trimmed = list(tokens)
    while trimmed and is_function_word(trimmed[-1]):
        trimmed.pop()
    if not trimmed:
        trimmed = list(tokens)

    variants: List[str] = []
    variants.append(" ".join(trimmed))
    variants.append(trimmed[-1])
    variants.append(" ".join(tokens))

    content_tokens = [tok for tok in trimmed if not is_function_word(tok)]
    if content_tokens and content_tokens != trimmed:
        variants.append(" ".join(content_tokens))
    if len(trimmed) > 1:
        variants.append(" ".join(trimmed[:-1]))

    for token in tokens:
        variants.append(token)
    for token in trimmed:
        if not is_function_word(token):
            variants.append(token)

    # Keep last variant for API compatibility with existing scripts.
    variants.append(" ".join(reversed(trimmed)))

    out = dedupe_preserve_order(variants)
    if include_stop_terms:
        return out
    return [term for term in out if not is_function_only(term)]


def resolve_pos_tags(kind: str) -> List[tuple]:
    if pyiwn is None:
        return []

    preferred = {
        "trigger": ("VERB", "NOUN"),
        "argument": ("NOUN", "ADJECTIVE", "VERB"),
    }.get(kind, ("NOUN", "VERB"))

    tag_enum = getattr(pyiwn, "PosTag", None)
    if not tag_enum:
        return []

    alias_map = {
        "NOUN": ("NOUN",),
        "VERB": ("VERB",),
        "ADJECTIVE": ("ADJECTIVE", "ADJ"),
        "ADVERB": ("ADVERB", "ADV"),
    }

    out = []
    seen = set()
    for name in preferred:
        aliases = alias_map.get(name, (name,))
        for alias in aliases:
            value = getattr(tag_enum, alias, None)
            if value is None:
                continue
            if value in seen:
                continue
            seen.add(value)
            out.append((name, value))
    return out


def _safe_method(owner: Any, method_name: str, fallback: Optional[str] = None) -> Optional[Any]:
    method = getattr(owner, method_name, None)
    if method is None or not callable(method):
        return None
    try:
        value = method()
        return value
    except Exception:
        if fallback is None:
            return None
        method = getattr(owner, fallback, None)
        if method is None or not callable(method):
            return None
        try:
            return method()
        except Exception:
            return None


def synset_id_str(synset) -> str:
    for attr in ("synset_id", "id"):
        value = getattr(synset, attr, None)
        if value is None:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        if value:
            return str(value)
    return ""


def lemma_names(synset) -> List[str]:
    method = getattr(synset, "lemma_names", None)
    if not callable(method):
        return []
    try:
        names = method() or []
    except Exception:
        return []
    return dedupe_preserve_order([str(name).strip() for name in names if str(name).strip()])


def synset_text_fields(synset) -> List[str]:
    fields: List[str] = []
    for meth in (
        "definition",
        "definitions",
        "gloss",
        "definition_and_examples",
        "get_gloss",
        "get_definition",
    ):
        value = _safe_method(synset, meth)
        if value is None:
            continue
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, (list, tuple)):
            value = [str(value)]
        for item in value:
            text = str(item).strip()
            if text:
                fields.append(text)
    return dedupe_preserve_order(fields)


def score_synset_for_context(synset, term: str, context_terms: List[str]) -> float:
    if not context_terms:
        return 0.0
    context = {canonical_token(t) for t in context_terms if canonical_token(t)}
    if not context:
        return 0.0

    score = 0.0
    text_features = [
        *lemma_names(synset),
        *synset_text_fields(synset),
    ]
    for feat in dedupe_preserve_order(text_features):
        norm = canonical_token(feat)
        if norm in context:
            score += 1.0
        else:
            # Partial match fallback for inflectional variants.
            score += sum(1.0 for token in tokenize_surface(feat) if canonical_token(token) in context) * 0.15

    # Slightly favor shorter term match to reduce noisy rare forms.
    lemma_overlap = 0
    for lemma in lemma_names(synset):
        if canonical_token(lemma) == canonical_token(term):
            lemma_overlap += 1.0
    score += 0.25 * lemma_overlap
    return score


def _safe_invoke(owner: Any, method_name: str, *args) -> List[Any]:
    method = getattr(owner, method_name, None)
    if method is None or not callable(method):
        return []
    try:
        result = method(*args)
    except Exception:
        return []
    if result is None:
        return []
    if isinstance(result, (str, bytes)):
        return [result]
    if isinstance(result, (list, tuple, set)):
        return list(result)
    return [result]


def _looks_like_hyper_relation(candidate: Any) -> bool:
    text = str(candidate).lower()
    return any(key in text for key in ("hyper", "super"))


def _is_synset_like(value: Any) -> bool:
    return hasattr(value, "lemma_names") or hasattr(value, "synset_id")


def _extract_relation_target(item: Any) -> Optional[Any]:
    if item is None:
        return None
    if _is_synset_like(item):
        return item

    if isinstance(item, dict):
        for key in ("target_synset", "synset", "target", "value"):
            candidate = item.get(key)
            if _is_synset_like(candidate):
                return candidate
            nested = _extract_relation_target(candidate)
            if nested is not None:
                return nested
        for key in ("relation", "name"):
            text = item.get(key)
            if isinstance(text, str) and _looks_like_hyper_relation(text):
                for alt_key in ("target_synset", "synset", "target", "value"):
                    candidate = item.get(alt_key)
                    nested = _extract_relation_target(candidate)
                    if nested is not None:
                        return nested
        return None

    if isinstance(item, (tuple, list)):
        if len(item) >= 2:
            left = item[0]
            right = item[1]
            if _looks_like_hyper_relation(left) and _is_synset_like(right):
                return right
            if _looks_like_hyper_relation(right) and _is_synset_like(left):
                return left
            nested = _extract_relation_target(left)
            if nested is not None:
                return nested
            return _extract_relation_target(right)
        return _extract_relation_target(item[0]) if item else None

    for attr in ("target_synset", "synset", "target", "lemma_names"):
        maybe = getattr(item, attr, None)
        if _is_synset_like(maybe):
            return maybe
        if isinstance(maybe, (tuple, list, set, dict)):
            nested = _extract_relation_target(maybe)
            if nested is not None:
                return nested
    return None


def _dedupe_targets(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for item in items:
        target = _extract_relation_target(item)
        if target is None:
            continue
        sid = synset_id_str(target) or repr(target)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(target)
    return out


def related_synsets(iwn_obj, synset, relation_name: str = "HYPER") -> List[Any]:
    # relation_name retained for API compatibility. We attempt hyper-like relations by probing
    # multiple variants because pyiwn versions differ.
    if pyiwn is None or iwn_obj is None or synset is None:
        return []

    raw: List[Any] = []
    # Primary API in newer/older pyiwn versions is IndoWordNet.synset_relation(synset, rel)
    # Older docs sometimes call this method on the synset object itself (get_relation),
    # so we keep both paths for compatibility.
    rel_enum = getattr(pyiwn, "SynsetRelations", None)

    relation_candidates = []
    if rel_enum is not None:
        for name in dir(rel_enum):
            if name.startswith("_"):
                continue
            if _looks_like_hyper_relation(name):
                relation_candidates.append(getattr(rel_enum, name))

    # Also include explicit aliases if present.
    explicit_candidates = [getattr(rel_enum, "HYPERNYMY", None), getattr(rel_enum, "HYPERNYM", None), getattr(rel_enum, "HYPERNYMY_OF", None)]
    if rel_enum is not None:
        for rel in explicit_candidates:
            if rel is not None and rel not in relation_candidates:
                relation_candidates.append(rel)

    # Query through IndoWordNet API first.
    for relation in [r for r in relation_candidates if r is not None]:
        raw.extend(_safe_invoke(iwn_obj, "synset_relation", synset, relation))

    # Some versions expose instance-level relation methods.
    for relation in [r for r in relation_candidates if r is not None]:
        raw.extend(_safe_invoke(synset, "get_relation", relation))
    raw.extend(_safe_invoke(synset, "relations"))

    # API alternatives (direct methods).
    for method_name in ("get_hypernyms", "hypernyms", "hypernym", "get_hypernym"):
        raw.extend(_safe_invoke(synset, method_name))

    hyper_targets = _dedupe_targets(raw)
    if hyper_targets:
        return hyper_targets

    # Fallback: explicit relation tuples with relation label.
    relation_text_items = []
    for maybe in raw:
        if isinstance(maybe, (tuple, list)) and len(maybe) == 2:
            relation_text_items.append(_extract_relation_target(maybe))
    return _dedupe_targets(relation_text_items)


def lookup_confidence(pos_rank: int, synset_count: int, term_len: int) -> float:
    if term_len <= 0:
        return 0.0
    base = 1.0
    base -= 0.1 * pos_rank
    base += 0.02 * min(1.0, term_len / 10.0)
    base -= 0.015 * max(0, synset_count - 2)
    return max(0.0, min(0.99, base))


def lookup_term(
    iwn_obj,
    term: str,
    kind: str,
    max_synsets: int = DEFAULT_MAX_SYNSETS,
    context_terms: Optional[List[str]] = None,
    enable_wsd: bool = False,
) -> Optional[dict]:
    term_text = normalize_surface(term)
    if not term_text:
        return None

    max_synsets = max(1, int(max_synsets))

    context_terms = context_terms or []

    for pos_rank, (pos_name, pos_value) in enumerate(resolve_pos_tags(kind)):
        try:
            synsets = iwn_obj.synsets(term_text, pos=pos_value) or []
        except TypeError:
            try:
                synsets = iwn_obj.synsets(term_text) or []
            except Exception:
                synsets = []
        except Exception:
            continue

        if not synsets:
            continue

        selected_synsets = list(synsets[:max_synsets])
        if enable_wsd and len(selected_synsets) > 1 and context_terms:
            ranked = []
            for order_index, candidate in enumerate(selected_synsets):
                score = score_synset_for_context(candidate, term_text, context_terms)
                ranked.append((score, -order_index, candidate))
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            selected_synsets = [s for _, _, s in ranked]
        primary_synset = selected_synsets[0]
        primary_id = synset_id_str(primary_synset)
        if not primary_id:
            continue

        synonyms: List[str] = []
        hypernyms: List[str] = []
        synset_ids: List[str] = []

        for selected in selected_synsets:
            sid = synset_id_str(selected)
            if sid:
                synset_ids.append(sid)
            synonyms.extend(lemma_names(selected))
            for hyper in related_synsets(iwn_obj, selected):
                if hyper:
                    hypernyms.extend(lemma_names(hyper))

        synonyms = dedupe_preserve_order(synonyms)
        if term_text and canonical_token(term_text) in {canonical_token(s) for s in synonyms}:
            synonyms = [s for s in synonyms if canonical_token(s) != canonical_token(term_text)]
        hypernyms = dedupe_preserve_order(hypernyms)

        return {
            "lookup_term": term_text,
            "candidate_term": term_text,
            "pos": pos_name,
            "pos_rank": pos_rank,
            "synset_id": primary_id,
            "synset_ids": dedupe_preserve_order(synset_ids),
            "synset_count": len(synsets),
            "synonyms": synonyms,
            "hypernyms": hypernyms,
            "confidence": round(lookup_confidence(pos_rank, len(synsets), len(tokenize_surface(term_text))), 6),
        }

    return None


def lookup_cache_key(
    term: str,
    kind: str,
    max_synsets: int,
    enable_wsd: bool = False,
    context_signature: str = "",
) -> str:
    parts = [kind, canonical_token(term), str(max(1, int(max_synsets)))]
    if enable_wsd:
        parts.append("wsd")
        parts.append(context_signature or "nocontext")
    return "\t".join(parts)


def load_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_cache(path: Path, data: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def canonical_lexeme_uri(term: str, pos: str = "") -> str:
    token = canonical_token(term)
    pos_token = normalize_surface(pos).upper().strip()
    if not token:
        return ""
    if pos_token:
        return f"{NS['wn']}lemma/{clean_uri(token)}__{clean_uri(pos_token)}"
    return f"{NS['wn']}lemma/{clean_uri(token)}"


def canonical_synonym_uri(term: str) -> str:
    token = canonical_token(term)
    if not token:
        return ""
    return f"{NS['wn']}synonym/{clean_uri(token)}"


def canonical_concept_uri(term: str) -> str:
    token = canonical_token(term)
    if not token:
        return ""
    return f"{NS['wn']}concept/{clean_uri(token)}"


def enrich_surface(
    iwn_obj,
    text: str,
    kind: str,
    cache: Dict[str, dict],
    context_terms: Optional[List[str]],
    context_signature: str,
    skip_stop_terms: bool,
    min_confidence: float,
    enable_wsd: bool,
    max_synsets: int,
    max_synonyms: int,
    max_hypernyms: int,
    min_term_length: int,
    upos_hint: Optional[str] = None,
    skip_propn: bool = DEFAULT_SKIP_PROPN,
    lemma_hint: Optional[str] = None,
    use_lemma_first: bool = True,
    allowed_pos: Optional[Set[str]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Optional[dict]:
    if not is_pos_allowed_for_wordnet(
        upos_hint,
        kind=kind,
        skip_propn=skip_propn,
        allowed_pos=allowed_pos,
    ):
        if stats is not None:
            stats["miss_reason_pos_filtered"] = stats.get("miss_reason_pos_filtered", 0) + 1
        return None

    lemma_key = canonical_token(normalize_surface(lemma_hint or ""))
    strict_propn = bool(skip_propn and (upos_hint or "").upper() == "PROPN")
    term_candidates: List[Tuple[str, str]] = []
    seen = set()

    if use_lemma_first:
        lemma = normalize_surface(lemma_hint or "")
        if lemma:
            lemma_key = canonical_token(lemma)
            if lemma_key:
                seen.add(lemma_key)
                term_candidates.append((lemma, "lemma"))

    for term in candidate_terms(text, include_stop_terms=not skip_stop_terms):
        term_norm = canonical_token(term)
        if not term_norm:
            continue
        if term_norm in seen:
            continue
        seen.add(term_norm)
        term_candidates.append((term, "surface"))

    if not term_candidates:
        if stats is not None:
            stats["miss_reason_no_candidates"] = stats.get("miss_reason_no_candidates", 0) + 1
        return None

    min_term_length = max(1, int(min_term_length))
    max_synsets = max(1, int(max_synsets))
    context_terms = dedupe_preserve_order(context_terms or [])
    last_reason = "lookup_failed"

    for term, source in term_candidates:
        term_key = canonical_token(term)
        if len(term_key) < min_term_length:
            continue
        if strict_propn:
            if not lemma_key:
                last_reason = "pos_filtered"
                continue
            if term_key != lemma_key:
                continue
        key = lookup_cache_key(term, kind, max_synsets, enable_wsd=enable_wsd, context_signature=context_signature)
        cached = cache.get(key)
        if isinstance(cached, dict):
            if cached.get("_miss"):
                last_reason = "cache_miss"
                continue
            if cached.get("confidence", 0.0) < min_confidence:
                last_reason = "low_confidence"
                continue
            result = dict(cached)
            result["synonyms"] = dedupe_preserve_order(result.get("synonyms", []))[:max_synonyms]
            result["hypernyms"] = dedupe_preserve_order(result.get("hypernyms", []))[:max_hypernyms]
            result["lexeme_lemma"] = normalize_surface(lemma_hint or result.get("lookup_term", ""))
            result["lexeme_pos"] = normalize_surface(result.get("pos", "")).upper()
            result["_lookup_source"] = source
            if stats is not None:
                if source == "lemma":
                    stats["enriched_by_lemma"] = stats.get("enriched_by_lemma", 0) + 1
                else:
                    stats["enriched_by_surface"] = stats.get("enriched_by_surface", 0) + 1
            return result

        result = lookup_term(
            iwn_obj,
            term,
            kind,
            max_synsets=max_synsets,
            context_terms=context_terms,
            enable_wsd=enable_wsd,
        )
        if result is None:
            cache[key] = {"_miss": True}
            last_reason = "lookup_failed"
            continue
        if result.get("confidence", 0.0) < min_confidence:
            last_reason = "low_confidence"
            cache[key] = {"_miss": True, "low_confidence": result.get("confidence", 0.0)}
            continue

        result["synonyms"] = dedupe_preserve_order(result.get("synonyms", []))[:max_synonyms]
        result["hypernyms"] = dedupe_preserve_order(result.get("hypernyms", []))[:max_hypernyms]
        result["lexeme_lemma"] = normalize_surface(lemma_hint or result.get("lookup_term", ""))
        result["lexeme_pos"] = normalize_surface(result.get("pos", "")).upper()
        cache[key] = result
        result = dict(result)
        result["_lookup_source"] = source
        if stats is not None:
            if source == "lemma":
                stats["enriched_by_lemma"] = stats.get("enriched_by_lemma", 0) + 1
            else:
                stats["enriched_by_surface"] = stats.get("enriched_by_surface", 0) + 1
        return result

    if stats is not None:
        stats[f"miss_reason_{last_reason}"] = stats.get(f"miss_reason_{last_reason}", 0) + 1
    return None


def iter_events(item: object) -> Iterable[dict]:
    if isinstance(item, dict) and isinstance(item.get("events"), list):
        for event in item["events"]:
            if isinstance(event, dict):
                yield event
        return
    if isinstance(item, dict):
        yield item


def iter_input_events(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            rows = json.load(f)
            if isinstance(rows, list):
                for row in rows:
                    yield from iter_events(row)
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield from iter_events(row)


def stable_event_id(event: dict) -> str:
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


def add_quad(quads: List[str], seen: set, quad: str) -> None:
    if quad in seen:
        return
    quads.append(quad)
    seen.add(quad)


def emit_wordnet_enrichment(
    subject_uri: str,
    enrich: dict,
    graph_uri: str,
    quads: List[str],
    seen: set,
    seen_synsets: Set[str],
    seen_concepts: Set[str],
    seen_synonyms: Set[str],
    seen_synset_synonyms: Set[str],
    synonym_scope: str = DEFAULT_SYNONYM_SCOPE,
    emit_synonym_nodes: bool = True,
) -> None:
    if not enrich:
        return

    add_quad(
        quads,
        seen,
        to_nquad_lit(subject_uri, f"{NS['wn']}lookupForm", enrich.get("lookup_term", ""), graph_uri, lang="hi"),
    )
    add_quad(
        quads,
        seen,
        to_nquad_lit(
            subject_uri, f"{NS['wn']}candidateTerm", enrich.get("candidate_term", ""), graph_uri, lang="hi"
        ),
    )

    confidence = enrich.get("confidence")
    if confidence is not None:
        add_quad(quads, seen, to_nquad_decimal(subject_uri, f"{NS['wn']}lookupConfidence", float(confidence), graph_uri))

    if enrich.get("pos"):
        add_quad(
            quads,
            seen,
            to_nquad_lit(subject_uri, f"{NS['wn']}posTag", enrich.get("pos", ""), graph_uri, lang="en"),
        )

    if enrich.get("pos_rank") is not None:
        add_quad(
            quads,
            seen,
            to_nquad_decimal(subject_uri, f"{NS['wn']}posRank", float(enrich.get("pos_rank", 0.0)), graph_uri),
        )

    if enrich.get("synset_count") is not None:
        add_quad(
            quads,
            seen,
            to_nquad_decimal(subject_uri, f"{NS['wn']}synsetCount", float(enrich.get("synset_count", 0.0)), graph_uri),
        )

    synset_ids = dedupe_preserve_order(enrich.get("synset_ids", []))
    if not synset_ids and enrich.get("synset_id"):
        synset_ids = [enrich.get("synset_id")]

    for index, synset_id in enumerate(synset_ids):
        if not synset_id:
            continue
        synset_uri = f"{NS['wn']}synset/{clean_uri(synset_id)}"
        if synset_uri not in seen_synsets:
            seen_synsets.add(synset_uri)
            add_quad(quads, seen, to_nquad_uri(synset_uri, f"{NS['rdf']}type", f"{NS['wn']}Synset", graph_uri))
        predicate = f"{NS['wn']}inSynset" if index == 0 else f"{NS['wn']}alternativeSynset"
        add_quad(quads, seen, to_nquad_uri(subject_uri, predicate, synset_uri, graph_uri))

    for index, synset_id in enumerate(synset_ids):
        if not synset_id:
            continue
        synset_uri = f"{NS['wn']}synset/{clean_uri(synset_id)}"
        for synonym in dedupe_preserve_order(enrich.get("synonyms", [])):
            if not synonym:
                continue
            if synonym_scope == SYNSCOPE_SYNSET:
                key = f"{synset_uri}\t{canonical_token(synonym)}"
                if key in seen_synset_synonyms:
                    continue
                seen_synset_synonyms.add(key)
                add_quad(
                    quads,
                    seen,
                    to_nquad_lit(synset_uri, f"{NS['wn']}synonym", synonym, graph_uri, lang="hi"),
                )
                if emit_synonym_nodes:
                    synonym_uri = canonical_synonym_uri(synonym)
                    if not synonym_uri:
                        continue
                    if synonym_uri not in seen_synonyms:
                        seen_synonyms.add(synonym_uri)
                        add_quad(
                            quads,
                            seen,
                            to_nquad_uri(synonym_uri, f"{NS['rdf']}type", f"{NS['wn']}Synonym", graph_uri),
                        )
                        add_quad(
                            quads,
                            seen,
                            to_nquad_lit(
                                synonym_uri, f"{NS['rdfs']}label", synonym, graph_uri, lang="hi"
                            ),
                        )
                    add_quad(
                        quads,
                        seen,
                        to_nquad_uri(synonym_uri, f"{NS['rdfs']}seeAlso", synset_uri, graph_uri),
                    )
            else:
                add_quad(
                    quads,
                    seen,
                    to_nquad_lit(subject_uri, f"{NS['wn']}synonym", synonym, graph_uri, lang="hi"),
                )
                if not emit_synonym_nodes:
                    continue
                synonym_uri = canonical_synonym_uri(synonym)
                if not synonym_uri:
                    continue
                if synonym_uri not in seen_synonyms:
                    seen_synonyms.add(synonym_uri)
                    add_quad(quads, seen, to_nquad_uri(synonym_uri, f"{NS['rdf']}type", f"{NS['wn']}Synonym", graph_uri))
                    add_quad(
                        quads,
                        seen,
                        to_nquad_lit(synonym_uri, f"{NS['rdfs']}label", synonym, graph_uri, lang="hi"),
                    )
                add_quad(quads, seen, to_nquad_uri(subject_uri, f"{NS['wn']}hasSynonym", synonym_uri, graph_uri))

    for hypernym in dedupe_preserve_order(enrich.get("hypernyms", [])):
        if not hypernym:
            continue
        concept_uri = canonical_concept_uri(hypernym)
        if concept_uri in seen_concepts:
            add_quad(quads, seen, to_nquad_uri(subject_uri, f"{NS['wn']}hypernym", concept_uri, graph_uri))
            continue
        seen_concepts.add(concept_uri)
        add_quad(quads, seen, to_nquad_uri(concept_uri, f"{NS['rdf']}type", f"{NS['wn']}Concept", graph_uri))
        add_quad(quads, seen, to_nquad_lit(concept_uri, f"{NS['rdfs']}label", hypernym, graph_uri, lang="hi"))
        add_quad(quads, seen, to_nquad_uri(subject_uri, f"{NS['wn']}hypernym", concept_uri, graph_uri))


def emit_lexeme_node(
    subject_uri: str,
    enrich: dict,
    graph_uri: str,
    quads: List[str],
    seen: set,
    created_lexemes: Set[str],
) -> str:
    lookup_term = normalize_surface(enrich.get("lookup_term", ""))
    lexeme_lemma = normalize_surface(enrich.get("lexeme_lemma", "")) or lookup_term
    lexeme_pos = normalize_surface(enrich.get("lexeme_pos", enrich.get("pos", ""))).upper()
    lexeme_uri = canonical_lexeme_uri(lexeme_lemma, lexeme_pos)
    if not lexeme_uri:
        return subject_uri

    add_quad(
        quads,
        seen,
        to_nquad_uri(subject_uri, f"{NS['wn']}lexemeForm", lexeme_uri, graph_uri),
    )

    if lexeme_uri in created_lexemes:
        return lexeme_uri

    created_lexemes.add(lexeme_uri)
    add_quad(quads, seen, to_nquad_uri(lexeme_uri, f"{NS['rdf']}type", f"{NS['wn']}Lexeme", graph_uri))
    add_quad(quads, seen, to_nquad_lit(lexeme_uri, f"{NS['rdfs']}label", lexeme_lemma, graph_uri, lang="hi"))
    if lexeme_pos:
        add_quad(
            quads,
            seen,
            to_nquad_lit(lexeme_uri, f"{NS['wn']}posTag", lexeme_pos, graph_uri, lang="en"),
        )
    return lexeme_uri


def build_wordnet() -> Any:
    if pyiwn is None:
        raise RuntimeError("pyiwn not available")

    lang_enum = getattr(pyiwn, "Language", None)
    hindi = getattr(lang_enum, "HINDI", None) if lang_enum else None

    try:
        if hindi is not None:
            return pyiwn.IndoWordNet(lang=hindi)
        return pyiwn.IndoWordNet()
    except Exception as exc:
        if isinstance(exc, UnicodeDecodeError) or "charmap" in str(exc).lower():
            raise RuntimeError(
                "Failed to decode IndoWordNet data (Windows locale encoding issue). "
                "Run with UTF-8 mode: `py -3 -X utf8 align/wordnet_enrich.py ...` "
                "or set `PYTHONUTF8=1`."
            ) from exc
        raise RuntimeError("Failed to initialize IndoWordNet. Install pyiwn and ensure Hindi resources are available.") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL/JSON from extraction pipeline")
    ap.add_argument("--out", required=True, help="Output N-Quads for WordNet enrichment")
    ap.add_argument("--cache", default="align/wordnet_cache.json", help="WordNet lookup cache")
    ap.add_argument("--graph", default=DEFAULT_GRAPH, help="Named graph URI for emitted quads")
    ap.add_argument(
        "--sentence-texts",
        default="",
        help="Optional one-line-per-sentence text file to enable sentence-aware WSD",
    )
    ap.add_argument(
        "--pos-json",
        default="",
        help="Optional JSON POS cache from align/annotate_hindi_tokens.py",
    )
    ap.add_argument(
        "--ner-json",
        default="",
        help="Optional JSON NER cache from align/run_hindi_ner.py",
    )
    ap.add_argument(
        "--time-json",
        default="",
        help="Optional JSON temporal cache from align/temporal_enrich.py",
    )
    ap.add_argument(
        "--dbpedia-json",
        default="",
        help="Optional DBpedia link cache from align/link_dbpedia.py",
    )
    ap.add_argument(
        "--wsd",
        action="store_true",
        help="Enable lightweight WordNet WSD using context overlap scoring",
    )
    ap.add_argument(
        "--no-pos-filter",
        action="store_true",
        help="Do not skip tokens based on UPOS when POS cache is available",
    )
    ap.add_argument(
        "--lemma-first",
        dest="lemma_first",
        action="store_true",
        help="Prefer lemma lookup before surface forms when POS cache has lemma",
    )
    ap.add_argument(
        "--no-lemma-first",
        dest="lemma_first",
        action="store_false",
        help="Use surface forms only for lookup",
    )
    ap.add_argument(
        "--pos-filter-list",
        default="",
        help="Explicit comma-separated UPOS allow-list (e.g. NOUN,VERB,ADJ,ADV)",
    )
    ap.add_argument("--max-synonyms", type=int, default=DEFAULT_MAX_SYNONYMS, help="Max synonyms per node")
    ap.add_argument("--max-hypernyms", type=int, default=DEFAULT_MAX_HYPERNYMS, help="Max hypernyms per node")
    ap.add_argument("--max-synsets", type=int, default=DEFAULT_MAX_SYNSETS, help="Synsets scanned per lookup")
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="Maximum context tokens used for optional WSD",
    )
    ap.add_argument(
        "--min-term-length",
        type=int,
        default=DEFAULT_MIN_TERM_LENGTH,
        help="Skip lookup candidates shorter than this many characters",
    )
    ap.add_argument("--min-confidence", type=float, default=0.0, help="Skip enrichments below this confidence")
    ap.add_argument(
        "--synonym-scope",
        default=DEFAULT_SYNONYM_SCOPE,
        choices=[SYNSCOPE_SYNSET, SYNSCOPE_LITERAL],
        help="Where to attach synonyms. synset(default)=one node per synset; subject=on mention/lexeme.",
    )
    ap.add_argument("--no-triggers", action="store_true", help="Skip trigger enrichment")
    ap.add_argument(
        "--skip-stop-terms",
        dest="skip_stop_terms",
        action="store_true",
        help="Skip pure function words and postpositions from candidates",
    )
    ap.add_argument(
        "--include-stop-terms",
        dest="skip_stop_terms",
        action="store_false",
        help="Allow function words and postpositions in candidates",
    )
    ap.add_argument(
        "--skip-propn",
        dest="skip_propn",
        action="store_true",
        help="Do not enrich mentions tagged as proper nouns (default)",
    )
    ap.add_argument(
        "--keep-propn",
        dest="skip_propn",
        action="store_false",
        help="Allow enriching mentions tagged as proper nouns",
    )
    ap.add_argument(
        "--merge-surface-nodes",
        dest="merge_surface_nodes",
        action="store_true",
        help="Share WordNet enrichment using a single node per unique surface form",
    )
    ap.add_argument(
        "--no-surface-nodes",
        dest="merge_surface_nodes",
        action="store_false",
        help="Attach WordNet enrichment directly on event-local subject nodes",
    )
    ap.add_argument(
        "--emit-synonym-nodes",
        dest="emit_synonym_nodes",
        action="store_true",
        help="Emit each synonym as a dedicated URI node (better for large graph reasoning)",
    )
    ap.add_argument(
        "--no-synonym-nodes",
        dest="emit_synonym_nodes",
        action="store_false",
        help="Emit synonyms only as literals",
    )
    ap.add_argument(
        "--skip-entities",
        dest="skip_entities",
        action="store_true",
        help="Skip WordNet enrichment when a span overlaps PER/LOC/ORG/TIME/NUM NER mentions (default)",
    )
    ap.add_argument(
        "--no-skip-entities",
        dest="skip_entities",
        action="store_false",
        help="Allow WordNet enrichment even when spans overlap named entities",
    )
    ap.add_argument(
        "--skip-timex",
        dest="skip_timex",
        action="store_true",
        help="Skip WordNet enrichment when a span overlaps a temporal expression (default)",
    )
    ap.add_argument(
        "--no-skip-timex",
        dest="skip_timex",
        action="store_false",
        help="Allow WordNet enrichment even when spans overlap temporal expressions",
    )
    ap.add_argument(
        "--stats-out",
        default="",
        help="Optional JSON file for enrichment counters and gating statistics",
    )
    ap.add_argument(
        "--skip-dbpedia-linked",
        dest="skip_dbpedia_linked",
        action="store_true",
        help="Skip WordNet enrichment when a span already links to a typed DBpedia entity (default when --dbpedia-json is provided)",
    )
    ap.add_argument(
        "--no-skip-dbpedia-linked",
        dest="skip_dbpedia_linked",
        action="store_false",
        help="Allow WordNet enrichment even when a span overlaps a DBpedia-linked entity",
    )
    ap.set_defaults(
        skip_stop_terms=True,
        merge_surface_nodes=True,
        emit_synonym_nodes=True,
        skip_propn=True,
        lemma_first=True,
        skip_entities=True,
        skip_timex=True,
        skip_dbpedia_linked=True,
    )
    args = ap.parse_args()

    if pyiwn is None:
        sys.exit("Error: pyiwn is not installed. Run: pip install pyiwn")

    input_path = Path(args.input)
    out_path = Path(args.out)
    cache_path = Path(args.cache)
    graph_uri = args.graph

    sentence_texts: Dict[str, str] = {}
    pos_annotations: Dict[str, List[PosToken]] = {}
    ner_annotations: Dict[str, List[dict]] = {}
    time_annotations: Dict[str, List[dict]] = {}
    dbpedia_annotations: Dict[str, List[dict]] = {}
    if args.sentence_texts:
        sentence_texts = load_sentence_texts(Path(args.sentence_texts))
    if args.pos_json:
        pos_annotations = load_pos_cache(Path(args.pos_json))
    if args.ner_json:
        ner_annotations = load_ner_cache(Path(args.ner_json))
    if args.time_json:
        time_annotations = load_time_cache(Path(args.time_json))
    if args.dbpedia_json:
        dbpedia_annotations = load_dbpedia_cache(Path(args.dbpedia_json))

    cache = load_cache(cache_path)
    try:
        iwn = build_wordnet()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")

    quads: List[str] = []
    seen_quads = set()
    events = 0
    enriched_instances = 0
    misses = 0
    enriched_subjects = 0
    lexeme_nodes = 0
    wsd_enabled_count = 0
    enrichment_stats: Dict[str, int] = {}

    created_lexemes: Set[str] = set()
    seen_synsets: Set[str] = set()
    seen_concepts: Set[str] = set()
    seen_synonyms: Set[str] = set()
    seen_synset_synonyms: Set[str] = set()
    allowed_pos_filter = parse_pos_filter_list(args.pos_filter_list) if (not args.no_pos_filter and args.pos_filter_list) else None
    effective_skip_propn = args.skip_propn and not args.no_pos_filter

    for event in iter_input_events(input_path):
        events += 1
        event_id = stable_event_id(event)
        event_id_clean = clean_uri(event_id)
        sentence_text = sentence_key_lookup(event, sentence_texts) if sentence_texts else None
        pos_tokens = []
        if pos_annotations:
            doc_id = event.get("doc_id", "batch_run")
            sent_id = event.get("sent_id", "")
            pos_tokens = pos_annotations.get(_sentence_key(doc_id, sent_id), [])
            if not pos_tokens:
                pos_tokens = pos_annotations.get(_sentence_key("batch_run", sent_id), [])
        doc_id = event.get("doc_id", "batch_run")
        sent_id = event.get("sent_id", "")
        sentence_entities = lookup_sentence_annotations(ner_annotations, doc_id, sent_id) if ner_annotations else []
        sentence_timexes = lookup_sentence_annotations(time_annotations, doc_id, sent_id) if time_annotations else []
        sentence_dbpedia_links = lookup_sentence_annotations(dbpedia_annotations, doc_id, sent_id) if dbpedia_annotations else []
        if args.wsd and sentence_text:
            wsd_enabled_count += 1

        trigger = event.get("trigger", {}) or {}
        if not args.no_triggers and isinstance(trigger, dict):
            trigger_text = normalize_surface(str(trigger.get("text", "")))
            if trigger_text:
                trigger_uri = f"{NS['nhkg']}Trigger_{event_id_clean}"
                trigger_span = normalize_char_span(trigger.get("char_span"))
                skip_trigger = False
                if args.skip_entities and span_overlaps_entity(trigger_span, sentence_entities):
                    enrichment_stats["skipped_due_to_named_entity"] = enrichment_stats.get("skipped_due_to_named_entity", 0) + 1
                    skip_trigger = True
                if not skip_trigger and args.skip_timex and span_overlaps_timex(trigger_span, sentence_timexes):
                    enrichment_stats["skipped_due_to_temporal_expression"] = enrichment_stats.get("skipped_due_to_temporal_expression", 0) + 1
                    skip_trigger = True
                if not skip_trigger and args.skip_dbpedia_linked and sentence_dbpedia_links:
                    dbpedia_reason = dbpedia_skip_reason(trigger_span, sentence_dbpedia_links)
                    if dbpedia_reason == "link":
                        enrichment_stats["skipped_due_to_dbpedia_link"] = enrichment_stats.get("skipped_due_to_dbpedia_link", 0) + 1
                        skip_trigger = True
                    elif dbpedia_reason == "type":
                        enrichment_stats["skipped_due_to_dbpedia_type"] = enrichment_stats.get("skipped_due_to_dbpedia_type", 0) + 1
                        skip_trigger = True
                    elif span_overlaps_dbpedia_link(trigger_span, sentence_dbpedia_links):
                        enrichment_stats["wordnet_kept_after_dbpedia_check"] = enrichment_stats.get("wordnet_kept_after_dbpedia_check", 0) + 1
                        enrichment_stats["wordnet_allowed_after_dbpedia_check"] = enrichment_stats.get("wordnet_allowed_after_dbpedia_check", 0) + 1
                if not skip_trigger:
                    trigger_context_terms = (
                        sentence_context_terms(sentence_text, trigger_span, window=args.context_tokens)
                        if sentence_text
                        else tokenise_for_wsd(trigger_text)
                    )
                    trigger_context_signature = wsd_context_signature(trigger_context_terms)
                    trigger_upos = None
                    trigger_lemma = None
                    if not args.no_pos_filter and pos_tokens:
                        token = get_token_for_span(pos_tokens, trigger_span)
                        if token is not None:
                            trigger_upos = token.upos
                            trigger_lemma = token.lemma
                    enrich = enrich_surface(
                        iwn,
                        trigger_text,
                        "trigger",
                        cache,
                        trigger_context_terms,
                        trigger_context_signature,
                        skip_stop_terms=args.skip_stop_terms,
                        min_confidence=args.min_confidence,
                        enable_wsd=args.wsd,
                        max_synsets=args.max_synsets,
                        max_synonyms=args.max_synonyms,
                        max_hypernyms=args.max_hypernyms,
                        min_term_length=args.min_term_length,
                        upos_hint=trigger_upos,
                        skip_propn=effective_skip_propn,
                        lemma_hint=trigger_lemma,
                        use_lemma_first=args.lemma_first,
                        allowed_pos=allowed_pos_filter,
                        stats=enrichment_stats,
                    )
                    if enrich:
                        enriched_subjects += 1
                        enriched_instances += 1
                        target_uri = trigger_uri
                        if args.merge_surface_nodes:
                            target_uri = emit_lexeme_node(
                                trigger_uri,
                                enrich,
                                graph_uri,
                                quads,
                                seen_quads,
                                created_lexemes,
                            )
                            lexeme_nodes = len(created_lexemes)
                        emit_wordnet_enrichment(
                            target_uri,
                            enrich,
                            graph_uri,
                            quads,
                            seen_quads,
                            seen_synsets,
                            seen_concepts,
                            seen_synonyms,
                            seen_synset_synonyms,
                            synonym_scope=args.synonym_scope,
                            emit_synonym_nodes=args.emit_synonym_nodes,
                        )
                    else:
                        misses += 1

        args_map = event.get("arguments", {}) or {}
        if not isinstance(args_map, dict):
            continue

        for role, arg_data in args_map.items():
            if not isinstance(arg_data, dict):
                continue
            arg_text = normalize_surface(str(arg_data.get("text", "")))
            if not arg_text:
                continue

            mention_uri = f"{NS['nhkg']}Mention_{event_id_clean}_{clean_uri(role)}"
            arg_span = normalize_char_span(arg_data.get("char_span"))
            if args.skip_entities and span_overlaps_entity(arg_span, sentence_entities):
                enrichment_stats["skipped_due_to_named_entity"] = enrichment_stats.get("skipped_due_to_named_entity", 0) + 1
                continue
            if args.skip_timex and span_overlaps_timex(arg_span, sentence_timexes):
                enrichment_stats["skipped_due_to_temporal_expression"] = enrichment_stats.get("skipped_due_to_temporal_expression", 0) + 1
                continue
            if args.skip_dbpedia_linked and sentence_dbpedia_links:
                dbpedia_reason = dbpedia_skip_reason(arg_span, sentence_dbpedia_links)
                if dbpedia_reason == "link":
                    enrichment_stats["skipped_due_to_dbpedia_link"] = enrichment_stats.get("skipped_due_to_dbpedia_link", 0) + 1
                    continue
                if dbpedia_reason == "type":
                    enrichment_stats["skipped_due_to_dbpedia_type"] = enrichment_stats.get("skipped_due_to_dbpedia_type", 0) + 1
                    continue
                if span_overlaps_dbpedia_link(arg_span, sentence_dbpedia_links):
                    enrichment_stats["wordnet_kept_after_dbpedia_check"] = enrichment_stats.get("wordnet_kept_after_dbpedia_check", 0) + 1
                    enrichment_stats["wordnet_allowed_after_dbpedia_check"] = enrichment_stats.get("wordnet_allowed_after_dbpedia_check", 0) + 1
            arg_context_terms = (
                sentence_context_terms(sentence_text, arg_span, window=args.context_tokens) if sentence_text else tokenise_for_wsd(arg_text)
            )
            arg_context_signature = wsd_context_signature(arg_context_terms)
            arg_upos = None
            arg_lemma = None
            if not args.no_pos_filter and pos_tokens:
                token = get_token_for_span(pos_tokens, arg_span)
                if token is not None:
                    arg_upos = token.upos
                    arg_lemma = token.lemma
            enrich = enrich_surface(
                iwn,
                arg_text,
                "argument",
                cache,
                arg_context_terms,
                arg_context_signature,
                skip_stop_terms=args.skip_stop_terms,
                min_confidence=args.min_confidence,
                enable_wsd=args.wsd,
                max_synsets=args.max_synsets,
                max_synonyms=args.max_synonyms,
                max_hypernyms=args.max_hypernyms,
                min_term_length=args.min_term_length,
                    upos_hint=arg_upos,
                    skip_propn=effective_skip_propn,
                    lemma_hint=arg_lemma,
                    use_lemma_first=args.lemma_first,
                    allowed_pos=allowed_pos_filter,
                    stats=enrichment_stats,
                )
            if enrich:
                enriched_subjects += 1
                enriched_instances += 1
                target_uri = mention_uri
                if args.merge_surface_nodes:
                    target_uri = emit_lexeme_node(
                        mention_uri,
                        enrich,
                        graph_uri,
                        quads,
                        seen_quads,
                        created_lexemes,
                    )
                    lexeme_nodes = len(created_lexemes)
                emit_wordnet_enrichment(
                    target_uri,
                    enrich,
                    graph_uri,
                    quads,
                    seen_quads,
                    seen_synsets,
                    seen_concepts,
                    seen_synonyms,
                    seen_synset_synonyms,
                    synonym_scope=args.synonym_scope,
                    emit_synonym_nodes=args.emit_synonym_nodes,
                )
            else:
                misses += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for quad in quads:
            f.write(quad + "\n")

    save_cache(cache_path, cache)
    unresolved_lookups = (
        enrichment_stats.get("miss_reason_no_candidates", 0)
        + enrichment_stats.get("miss_reason_cache_miss", 0)
        + enrichment_stats.get("miss_reason_lookup_failed", 0)
        + enrichment_stats.get("miss_reason_low_confidence", 0)
    )
    print(
        "[OK] WordNet enrichment complete: "
        f"events={events}, enriched_instances={enriched_instances}, enriched_subjects={enriched_subjects}, "
        f"misses={misses}, wsd_enabled_sentences={wsd_enabled_count}, "
        f"min_term_length={args.min_term_length}, max_synsets={args.max_synsets}, "
        f"max_synonyms={args.max_synonyms}, max_hypernyms={args.max_hypernyms}, "
        f"lexeme_nodes={lexeme_nodes}, "
        f"synset_nodes={len(seen_synsets)}, concept_nodes={len(seen_concepts)}, synonym_nodes={len(seen_synonyms)}, "
        f"quads={len(quads)}"
    )
    print(
        "[OK] WordNet enrichment metrics: "
        f"skipped_due_to_named_entity={enrichment_stats.get('skipped_due_to_named_entity', 0)}, "
        f"skipped_due_to_temporal_expression={enrichment_stats.get('skipped_due_to_temporal_expression', 0)}, "
        f"skipped_due_to_dbpedia_link={enrichment_stats.get('skipped_due_to_dbpedia_link', 0)}, "
        f"skipped_due_to_dbpedia_type={enrichment_stats.get('skipped_due_to_dbpedia_type', 0)}, "
        f"wordnet_kept_after_dbpedia_check={enrichment_stats.get('wordnet_kept_after_dbpedia_check', 0)}, "
        f"wordnet_allowed_after_dbpedia_check={enrichment_stats.get('wordnet_allowed_after_dbpedia_check', 0)}, "
        f"skipped_due_to_pos={enrichment_stats.get('miss_reason_pos_filtered', 0)}, "
        f"lemma_hits={enrichment_stats.get('enriched_by_lemma', 0)}, "
        f"surface_hits={enrichment_stats.get('enriched_by_surface', 0)}, "
        f"unresolved_lookups={unresolved_lookups}, "
        f"miss_pos_filtered={enrichment_stats.get('miss_reason_pos_filtered', 0)}, "
        f"miss_no_candidates={enrichment_stats.get('miss_reason_no_candidates', 0)}, "
        f"miss_cache={enrichment_stats.get('miss_reason_cache_miss', 0)}, "
        f"miss_lookup={enrichment_stats.get('miss_reason_lookup_failed', 0)}, "
        f"miss_low_confidence={enrichment_stats.get('miss_reason_low_confidence', 0)}, "
        f"enriched_instances={enriched_instances}"
    )
    if args.stats_out:
        stats_payload = {
            "events": events,
            "enriched_instances": enriched_instances,
            "enriched_subjects": enriched_subjects,
            "misses": misses,
            "wsd_enabled_sentences": wsd_enabled_count,
            "skipped_due_to_named_entity": enrichment_stats.get("skipped_due_to_named_entity", 0),
            "skipped_due_to_temporal_expression": enrichment_stats.get("skipped_due_to_temporal_expression", 0),
            "skipped_due_to_dbpedia_link": enrichment_stats.get("skipped_due_to_dbpedia_link", 0),
            "skipped_due_to_dbpedia_type": enrichment_stats.get("skipped_due_to_dbpedia_type", 0),
            "wordnet_kept_after_dbpedia_check": enrichment_stats.get("wordnet_kept_after_dbpedia_check", 0),
            "wordnet_allowed_after_dbpedia_check": enrichment_stats.get("wordnet_allowed_after_dbpedia_check", 0),
            "skipped_due_to_pos": enrichment_stats.get("miss_reason_pos_filtered", 0),
            "lemma_hits": enrichment_stats.get("enriched_by_lemma", 0),
            "surface_hits": enrichment_stats.get("enriched_by_surface", 0),
            "unresolved_lookups": unresolved_lookups,
            "miss_no_candidates": enrichment_stats.get("miss_reason_no_candidates", 0),
            "miss_cache": enrichment_stats.get("miss_reason_cache_miss", 0),
            "miss_lookup": enrichment_stats.get("miss_reason_lookup_failed", 0),
            "miss_low_confidence": enrichment_stats.get("miss_reason_low_confidence", 0),
            "lexeme_nodes": lexeme_nodes,
            "synset_nodes": len(seen_synsets),
            "concept_nodes": len(seen_concepts),
            "synonym_nodes": len(seen_synonyms),
            "quads": len(quads),
        }
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Stats: {stats_path}")
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Cache: {cache_path}")


if __name__ == "__main__":
    main()
