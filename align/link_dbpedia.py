#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Link likely Hindi entity mentions to canonical English DBpedia resource URIs."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import math
import re
import sys
from urllib.parse import unquote
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .enrichment_common import (
        build_entity_text_forms,
        entity_mention_uri,
        iter_input_events,
        mention_uri,
        normalize_char_span,
        sentence_key,
        stable_event_id,
    )
except ImportError:  # pragma: no cover
    from enrichment_common import (
        build_entity_text_forms,
        entity_mention_uri,
        iter_input_events,
        mention_uri,
        normalize_char_span,
        sentence_key,
        stable_event_id,
    )

try:
    from fusion.dbpedia_common import (
        DBPEDIA_ONTOLOGY_PREFIX,
        connect_db,
        db_fetch_candidates,
        db_fetch_exact_candidates,
        load_yaml_or_default,
        normalize_lookup_text,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from fusion.dbpedia_common import (
        DBPEDIA_ONTOLOGY_PREFIX,
        connect_db,
        db_fetch_candidates,
        db_fetch_exact_candidates,
        load_yaml_or_default,
        normalize_lookup_text,
    )


DEFAULT_NER_MAP = Path("lexicons/ner2dbo_class_map.yaml")
DEFAULT_ROLE_MAP = Path("lexicons/role2dbo_type_priors.yaml")
DEFAULT_WEIGHT_CONFIG = Path("lexicons/dbpedia_linking_config.yaml")

COMMON_POSTPOSITIONS = {
    "में",
    "पर",
    "से",
    "को",
    "का",
    "की",
    "के",
    "तक",
    "लिए",
    "द्वारा",
    "ने",
}
SKIP_NER_LABELS = {"TIME", "NUM"}
PRONOUN_WORDS = {
    "मैं",
    "हम",
    "तुम",
    "आप",
    "वह",
    "वे",
    "यह",
    "ये",
    "जो",
    "जिस",
    "उस",
    "इस",
    "उन",
    "इन्हें",
    "उन्हें",
    "मुझे",
    "हमें",
}
NON_ENTITY_TITLE_HINTS = (
    "relations",
    "episodes",
    "episode",
    "season",
    "discography",
    "filmography",
    "list_of",
    "award",
    "awards",
    "league",
    "broadcast",
    "radio",
)
INDEPENDENT_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
DEPENDENT_VOWELS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ृ": "ri",
}
CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ळ": "l",
    "क़": "q",
    "ख़": "kh",
    "ग़": "g",
    "ज़": "z",
    "ड़": "r",
    "ढ़": "rh",
    "फ़": "f",
}
DEFAULT_WEIGHTS = {
    "exact_hi_label": 4.8,
    "normalized_hi_label": 4.1,
    "exact_hi_alias": 3.9,
    "normalized_hi_alias": 3.1,
    "redirect_match": 2.6,
    "cleaned_form_bonus": 1.1,
    "alternate_form_bonus": 0.8,
    "head_form_bonus": 1.0,
    "lemma_form_bonus": 0.6,
    "ner_type_match": 2.5,
    "role_type_match": 1.8,
    "type_conflict_penalty": -1.5,
    "context_overlap": 0.3,
    "abstract_overlap": 0.2,
    "hindi_bonus": 0.8,
    "english_fallback_penalty": -0.35,
    "popularity_scale": 0.001,
    "no_type_penalty": -0.15,
    "non_entity_title_penalty": -4.0,
    "transliteration_match_bonus": 1.2,
    "transliteration_mismatch_penalty": -3.5,
    "acronym_title_penalty": -3.0,
    "ambiguous_alias_penalty": -2.2,
    "missing_expected_type_penalty": -1.1,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Link likely Hindi entity mentions to canonical English DBpedia resources.")
    parser.add_argument("--input", required=True, help="Extraction JSONL/JSON path")
    parser.add_argument("--sentences", required=True, help="One-sentence-per-line source text file")
    parser.add_argument("--pos-json", default="", help="Optional POS/syntax cache for entity-likelihood heuristics")
    parser.add_argument("--syntax-json", default="", help="Optional syntax cache; defaults to --pos-json if omitted")
    parser.add_argument("--ner-json", default="", help="NER cache from align/run_hindi_ner.py")
    parser.add_argument("--time-json", default="", help="Temporal cache from align/temporal_enrich.py")
    parser.add_argument("--dbpedia-index", required=True, help="SQLite DB produced by fusion/prepare_dbpedia_index.py")
    parser.add_argument("--engine", default="local", choices=["local", "lookup", "spotlight", "hybrid"])
    parser.add_argument("--lookup-url", default="", help="Optional DBpedia Lookup endpoint")
    parser.add_argument("--spotlight-url", default="", help="Optional DBpedia Spotlight endpoint")
    parser.add_argument("--top-k", type=int, default=5, help="Max candidate links preserved per mention")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence needed to accept a link")
    parser.add_argument("--out", required=True, help="Output JSON link cache")
    parser.add_argument("--ner-type-map", default=str(DEFAULT_NER_MAP), help="NER label -> dbo type map YAML")
    parser.add_argument("--role-type-map", default=str(DEFAULT_ROLE_MAP), help="Role -> dbo type prior map YAML")
    parser.add_argument("--linking-config", default=str(DEFAULT_WEIGHT_CONFIG), help="Optional YAML with ranking weights")
    parser.add_argument("--link-only-ner", action="store_true", help="Restrict DBpedia linking to NER spans only")
    parser.add_argument("--allow-propn-head-fallback", dest="allow_propn_head_fallback", action="store_true", help="Allow argument spans whose syntactic head is PROPN (default)")
    parser.add_argument("--no-propn-head-fallback", dest="allow_propn_head_fallback", action="store_false", help="Disable PROPN-headed argument fallback")
    parser.add_argument("--allow-misc", action="store_true", help="Allow MISC NER spans as DBpedia candidates")
    parser.add_argument("--skip-pronouns", dest="skip_pronouns", action="store_true", help="Skip pronouns as DBpedia candidates (default)")
    parser.add_argument("--no-skip-pronouns", dest="skip_pronouns", action="store_false")
    parser.add_argument("--skip-common-nouns", dest="skip_common_nouns", action="store_true", help="Skip generic common nouns as DBpedia candidates (default)")
    parser.add_argument("--no-skip-common-nouns", dest="skip_common_nouns", action="store_false")
    parser.add_argument("--debug-samples", type=int, default=3, help="Print a few linked sample rows")
    parser.set_defaults(allow_propn_head_fallback=True, skip_pronouns=True, skip_common_nouns=True)
    return parser


def load_sentence_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def load_annotation_map(path: Optional[Path], field_name: str) -> Dict[str, List[dict]]:
    payload = load_json(path)
    if not payload:
        return {}
    out: Dict[str, List[dict]] = {}
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            values = item.get(field_name, [])
            if isinstance(values, list):
                out[sentence_key(doc_id, sent_id)] = [row for row in values if isinstance(row, dict)]
    return out


def load_pos_map(path: Optional[Path]) -> Dict[str, List[dict]]:
    return load_annotation_map(path, "tokens")


def load_type_map(path: Path) -> Dict[str, List[str]]:
    payload = load_yaml_or_default(path, {})
    out: Dict[str, List[str]] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key).strip().upper()] = [str(item).strip() for item in value if str(item).strip()]
        elif value:
            out[str(key).strip().upper()] = [str(value).strip()]
    return out


def load_weight_config(path: Path) -> Dict[str, float]:
    payload = load_yaml_or_default(path, {})
    if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
        payload = payload["weights"]
    weights = dict(DEFAULT_WEIGHTS)
    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return weights


def safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def spans_overlap(left: Optional[Tuple[int, int]], right: Optional[Tuple[int, int]]) -> bool:
    if left is None or right is None:
        return False
    return not (left[1] <= right[0] or right[1] <= left[0])


def normalize_types(values: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text.startswith(DBPEDIA_ONTOLOGY_PREFIX):
            out.add(text)
            out.add(text.rsplit("/", 1)[-1])
        else:
            out.add(text)
    return out


def role_lookup_key(role: object) -> str:
    return str(role or "").strip().replace(" ", "_").upper()


def sentence_text_for_sent_id(sentences: List[str], sent_id_value: object) -> str:
    sent_index = safe_int(sent_id_value, -1)
    if 0 <= sent_index < len(sentences):
        return sentences[sent_index]
    return ""


def tokenize_text(text: str) -> List[str]:
    cleaned = text.replace("।", " ").replace(",", " ").replace(";", " ").replace(":", " ")
    return [token for token in cleaned.split() if token.strip()]


def context_terms(text: str) -> Set[str]:
    return {normalize_lookup_text(token) for token in tokenize_text(text) if normalize_lookup_text(token)}


def transliterate_hindi_to_ascii(text: str) -> str:
    independent_vowels = {
        "\u0905": "a",
        "\u0906": "aa",
        "\u0907": "i",
        "\u0908": "ii",
        "\u0909": "u",
        "\u090a": "uu",
        "\u090f": "e",
        "\u0910": "ai",
        "\u0913": "o",
        "\u0914": "au",
    }
    dependent_vowels = {
        "\u093e": "aa",
        "\u093f": "i",
        "\u0940": "ii",
        "\u0941": "u",
        "\u0942": "uu",
        "\u0947": "e",
        "\u0948": "ai",
        "\u094b": "o",
        "\u094c": "au",
        "\u0943": "ri",
    }
    consonants = {
        "\u0915": "k",
        "\u0916": "kh",
        "\u0917": "g",
        "\u0918": "gh",
        "\u0919": "n",
        "\u091a": "ch",
        "\u091b": "chh",
        "\u091c": "j",
        "\u091d": "jh",
        "\u091e": "n",
        "\u091f": "t",
        "\u0920": "th",
        "\u0921": "d",
        "\u0922": "dh",
        "\u0923": "n",
        "\u0924": "t",
        "\u0925": "th",
        "\u0926": "d",
        "\u0927": "dh",
        "\u0928": "n",
        "\u092a": "p",
        "\u092b": "ph",
        "\u092c": "b",
        "\u092d": "bh",
        "\u092e": "m",
        "\u092f": "y",
        "\u0930": "r",
        "\u0932": "l",
        "\u0935": "v",
        "\u0936": "sh",
        "\u0937": "sh",
        "\u0938": "s",
        "\u0939": "h",
        "\u0933": "l",
        "\u0958": "q",
        "\u0959": "kh",
        "\u095a": "g",
        "\u095b": "z",
        "\u095c": "r",
        "\u095d": "rh",
        "\u095e": "f",
    }
    virama = "\u094d"

    out: List[str] = []
    chars = list(str(text or ""))
    idx = 0
    while idx < len(chars):
        ch = chars[idx]
        if ch in independent_vowels:
            out.append(independent_vowels[ch])
        elif ch in consonants:
            roman = consonants[ch]
            if idx + 1 < len(chars) and chars[idx + 1] == virama:
                out.append(roman)
                idx += 1
            elif idx + 1 < len(chars) and chars[idx + 1] in dependent_vowels:
                out.append(roman + dependent_vowels[chars[idx + 1]])
                idx += 1
            else:
                out.append(roman + "a")
        elif ch in dependent_vowels:
            out.append(dependent_vowels[ch])
        idx += 1
    return re.sub(r"[^a-z]", "", "".join(out).lower())


def english_title_candidates(candidate: dict) -> List[str]:
    values: List[str] = []
    label_en = str(candidate.get("label_en", "")).strip()
    if label_en:
        values.append(label_en)
    canonical_uri = str(candidate.get("canonical_uri", "")).strip()
    if canonical_uri:
        local_title = unquote(canonical_uri.rsplit("/", 1)[-1]).replace("_", " ")
        if local_title:
            values.append(local_title)
    alias = str(candidate.get("alias", "")).strip()
    lang = str(candidate.get("lang", "")).strip().lower()
    if alias and lang == "en":
        values.append(alias)
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def normalize_english_surface(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text or "").lower())


def best_transliteration_similarity(mention: dict, candidate: dict) -> float:
    forms = [
        str(mention.get("cleaned_text", "")).strip(),
        str(mention.get("raw_text", "")).strip(),
        *[str(item).strip() for item in (mention.get("alternate_forms_used") or []) if str(item).strip()],
    ]
    transliterated_forms = [transliterate_hindi_to_ascii(item) for item in forms if item]
    transliterated_forms = [item for item in transliterated_forms if item]
    if not transliterated_forms:
        return 0.0

    title_forms = [normalize_english_surface(item) for item in english_title_candidates(candidate)]
    title_forms = [item for item in title_forms if item]
    if not title_forms:
        return 0.0

    best = 0.0
    for left in transliterated_forms:
        for right in title_forms:
            best = max(best, SequenceMatcher(None, left, right).ratio())
    return round(best, 4)


def looks_like_acronym_title(candidate: dict) -> bool:
    canonical_uri = str(candidate.get("canonical_uri", "")).strip()
    title = unquote(canonical_uri.rsplit("/", 1)[-1]) if canonical_uri else ""
    if not title:
        return False
    if re.match(r"^[A-Z0-9]{2,}(?:[_-][A-Z0-9]{2,})+$", title):
        return True
    head = title.split("_", 1)[0]
    return bool(re.fullmatch(r"[A-Z0-9]{3,}", head))


def span_tokens(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> List[dict]:
    if span is None:
        return []
    out: List[dict] = []
    for token in tokens:
        token_span = (safe_int(token.get("start")), safe_int(token.get("end")))
        if token_span[0] < 0 or token_span[1] < token_span[0]:
            continue
        if spans_overlap(span, token_span):
            out.append(token)
    return out


def span_lemma(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> str:
    lemmas: List[str] = []
    for token in span_tokens(tokens, span):
        lemma = str(token.get("lemma", "")).strip()
        if lemma:
            lemmas.append(lemma)
    return " ".join(lemmas).strip()


def span_upos_set(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> Set[str]:
    out: Set[str] = set()
    for token in span_tokens(tokens, span):
        upos = str(token.get("upos", "")).strip().upper()
        if upos:
            out.add(upos)
    return out


def head_token_for_span(tokens: Sequence[dict], span: Optional[Tuple[int, int]]) -> Optional[dict]:
    overlapping = span_tokens(tokens, span)
    if not overlapping:
        return None
    index_set = {safe_int(token.get("word_index", token.get("token_id", 0))) for token in overlapping}
    for token in overlapping:
        head = safe_int(token.get("head"), 0)
        idx = safe_int(token.get("word_index", token.get("token_id", 0)), 0)
        if idx <= 0:
            continue
        if head == 0 or head not in index_set:
            return token
    return overlapping[0]


def is_numeric_expression(text: str, tokens: Sequence[dict]) -> bool:
    if re.search(r"\d", text):
        return True
    return bool(tokens) and all(str(token.get("upos", "")).upper() == "NUM" for token in tokens)


def is_pronoun_candidate(text: str, tokens: Sequence[dict]) -> bool:
    if normalize_lookup_text(text) in {normalize_lookup_text(item) for item in PRONOUN_WORDS}:
        return True
    return any(str(token.get("upos", "")).upper() == "PRON" for token in tokens)


def is_functional_fragment(cleaned_text: str, tokens: Sequence[dict]) -> bool:
    if not cleaned_text:
        return True
    if len(tokens) == 1:
        upos = str(tokens[0].get("upos", "")).upper()
        if upos in {"ADP", "PART", "CCONJ", "SCONJ", "DET", "AUX", "PRON"}:
            return True
    return normalize_lookup_text(cleaned_text) in {normalize_lookup_text(item) for item in COMMON_POSTPOSITIONS}


def expected_types_for_mention(ner_label: str, role: str, ner_type_map: Dict[str, List[str]], role_type_map: Dict[str, List[str]]) -> Set[str]:
    expected: Set[str] = set()
    if ner_label:
        expected.update(normalize_types(ner_type_map.get(ner_label.upper(), [])))
    if role:
        expected.update(normalize_types(role_type_map.get(role_lookup_key(role), [])))
    return expected


def determine_entity_likelihood(
    mention: dict,
    tokens: Sequence[dict],
    time_map: Dict[str, List[dict]],
    *,
    link_only_ner: bool,
    allow_propn_head_fallback: bool,
    allow_misc: bool,
    skip_pronouns: bool,
    skip_common_nouns: bool,
) -> Tuple[bool, str, bool]:
    sent_key_value = str(mention["sent_key"])
    span = (int(mention["start"]), int(mention["end"]))
    ner_label = str(mention.get("ner_label", "")).upper()
    source = str(mention.get("mention_source", ""))
    cleaned_text = str(mention.get("cleaned_text", "")).strip()
    overlapping_tokens = span_tokens(tokens, span)
    head_token = head_token_for_span(tokens, span)
    head_upos = str((head_token or {}).get("upos", "")).upper()
    has_propn = "PROPN" in span_upos_set(tokens, span)

    for timex in time_map.get(sent_key_value, []):
        timex_span = normalize_char_span([timex.get("start"), timex.get("end")])
        if spans_overlap(span, timex_span):
            return False, "skipped_due_to_temporal", False

    if is_numeric_expression(cleaned_text, overlapping_tokens):
        return False, "skipped_due_to_numeric", False
    if skip_pronouns and is_pronoun_candidate(cleaned_text, overlapping_tokens):
        return False, "skipped_due_to_pronoun", False
    if is_functional_fragment(cleaned_text, overlapping_tokens):
        return False, "skipped_due_to_low_entity_likelihood", False

    if source == "ner":
        if ner_label in {"PER", "LOC", "ORG"}:
            return True, "", True
        if ner_label == "MISC" and allow_misc:
            return True, "", True
        return False, "skipped_due_to_low_entity_likelihood", False

    if link_only_ner:
        return False, "skipped_due_to_low_entity_likelihood", False

    if allow_propn_head_fallback and (head_upos == "PROPN" or has_propn):
        return True, "", True

    if skip_common_nouns:
        return False, "skipped_due_to_common_noun", False

    return True, "", False


def base_mention_record(
    *,
    sent_key_value: str,
    mention_uri_value: str,
    mention_source: str,
    text: str,
    start: int,
    end: int,
    ner_label: str = "",
    role: str = "",
) -> dict:
    forms = build_entity_text_forms(text)
    return {
        "sent_key": sent_key_value,
        "mention_uri": mention_uri_value,
        "mention_id": mention_uri_value.rsplit("/", 1)[-1],
        "mention_source": mention_source,
        "text": forms["raw_text"],
        "raw_text": forms["raw_text"],
        "normalized_text": forms["normalized_text"],
        "cleaned_text": forms["cleaned_text"],
        "alternate_forms_used": forms["alternate_forms"],
        "start": start,
        "end": end,
        "ner_label": ner_label,
        "role": role,
    }


def generate_surface_steps(mention: dict, tokens: Sequence[dict]) -> List[dict]:
    steps: List[dict] = []
    seen: Set[Tuple[str, str]] = set()

    def add(step: str, query: str, lookup: str) -> None:
        value = str(query or "").strip()
        marker = (step, value)
        if not value or marker in seen:
            return
        seen.add(marker)
        steps.append({"step": step, "query": value, "lookup": lookup})

    raw_text = str(mention.get("raw_text", "")).strip()
    cleaned_text = str(mention.get("cleaned_text", "")).strip()
    normalized_text = str(mention.get("normalized_text", "")).strip()

    add("exact_cleaned_hi", cleaned_text, "exact")
    add("normalized_cleaned_hi", normalize_lookup_text(cleaned_text), "normalized")
    add("exact_raw_hi", raw_text, "exact")
    add("normalized_raw_hi", normalize_lookup_text(raw_text), "normalized")
    add("normalized_text", normalize_lookup_text(normalized_text), "normalized")
    for alternate in mention.get("alternate_forms_used", []) or []:
        add("alternate_hi", alternate, "exact")
        add("alternate_hi_normalized", normalize_lookup_text(alternate), "normalized")

    head_token = head_token_for_span(tokens, (int(mention["start"]), int(mention["end"])))
    if head_token is not None:
        add("syntax_head_form", str(head_token.get("text", "")).strip(), "exact")
        add("syntax_head_form_normalized", normalize_lookup_text(head_token.get("text", "")), "normalized")
        add("syntax_head_lemma", str(head_token.get("lemma", "")).strip(), "exact")

    lemma = span_lemma(tokens, (int(mention["start"]), int(mention["end"])))
    add("lemma_form", lemma, "exact")
    add("lemma_form_normalized", normalize_lookup_text(lemma), "normalized")
    return steps


def types_match_expected(candidate_types: Iterable[str], expected: Set[str]) -> bool:
    if not expected:
        return False
    candidate_norm = normalize_types(candidate_types)
    return any(item in candidate_norm for item in expected)


def build_candidate_features(
    mention: dict,
    sentence_text: str,
    candidate: dict,
    *,
    query_step: str,
    expected_types: Set[str],
    weights: Dict[str, float],
    head_text: str,
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}
    match_type = str(candidate.get("match_type", "")).strip()
    lang = str(candidate.get("lang", "")).strip().lower()
    alias_value = str(candidate.get("alias", "")).strip()
    canonical_uri = str(candidate.get("canonical_uri", "")).strip()
    raw_text = str(mention.get("raw_text", "")).strip()
    cleaned_text = str(mention.get("cleaned_text", "")).strip()
    normalized_text = normalize_lookup_text(raw_text)
    alias_norm = normalize_lookup_text(alias_value)

    if lang == "hi":
        breakdown["hindi_bonus"] = weights["hindi_bonus"]
    elif lang == "en":
        breakdown["english_fallback_penalty"] = weights["english_fallback_penalty"]

    if alias_value and alias_value == cleaned_text and lang == "hi" and match_type.startswith("label_hi"):
        breakdown["exact_hi_label"] = weights["exact_hi_label"]
    elif alias_norm and alias_norm == normalize_lookup_text(cleaned_text) and lang == "hi" and match_type.startswith("label_hi"):
        breakdown["normalized_hi_label"] = weights["normalized_hi_label"]

    if alias_value and alias_value == raw_text and lang == "hi" and "alias" in match_type:
        breakdown["exact_hi_alias"] = weights["exact_hi_alias"]
    elif alias_norm and alias_norm == normalized_text and lang == "hi" and ("alias" in match_type or match_type.startswith("label_hi")):
        breakdown["normalized_hi_alias"] = weights["normalized_hi_alias"]

    if match_type == "redirect":
        breakdown["redirect_match"] = weights["redirect_match"]

    if "cleaned" in query_step or "postposition" in query_step:
        breakdown["cleaned_form_bonus"] = weights["cleaned_form_bonus"]
    if "alternate" in query_step:
        breakdown["alternate_form_bonus"] = weights["alternate_form_bonus"]
    if "head" in query_step and head_text:
        breakdown["head_form_bonus"] = weights["head_form_bonus"]
    if "lemma" in query_step:
        breakdown["lemma_form_bonus"] = weights["lemma_form_bonus"]

    candidate_types = candidate.get("types_dbo", []) or []
    if expected_types:
        if types_match_expected(candidate_types, expected_types):
            if mention.get("ner_label"):
                breakdown["ner_type_match"] = weights["ner_type_match"]
            if mention.get("role"):
                breakdown["role_type_match"] = weights["role_type_match"]
        elif candidate_types:
            breakdown["type_conflict_penalty"] = weights["type_conflict_penalty"]
        else:
            breakdown["missing_expected_type_penalty"] = weights["missing_expected_type_penalty"]
    elif not candidate_types:
        if mention.get("role") or mention.get("ner_label"):
            breakdown["missing_expected_type_penalty"] = weights["missing_expected_type_penalty"]
        else:
            breakdown["no_type_penalty"] = weights["no_type_penalty"]

    sentence_terms = context_terms(sentence_text)
    label_terms = context_terms(" ".join([str(candidate.get("label_hi", "")), str(candidate.get("label_en", "")), alias_value]))
    abstract_terms = context_terms(" ".join([str(candidate.get("abstract_hi", ""))[:160], str(candidate.get("abstract_en", ""))[:160]]))
    label_overlap = len(sentence_terms & label_terms)
    abstract_overlap = len(sentence_terms & abstract_terms)
    if label_overlap:
        breakdown["context_overlap"] = min(label_overlap, 4) * weights["context_overlap"]
    if abstract_overlap:
        breakdown["abstract_overlap"] = min(abstract_overlap, 4) * weights["abstract_overlap"]

    try:
        popularity = float(candidate.get("popularity_score") or 0.0)
    except (TypeError, ValueError):
        popularity = 0.0
    if popularity > 0:
        breakdown["popularity"] = min(popularity, 1000.0) * weights["popularity_scale"]

    title_slug = canonical_uri.rsplit("/", 1)[-1].lower()
    if any(hint in title_slug for hint in NON_ENTITY_TITLE_HINTS):
        breakdown["non_entity_title_penalty"] = weights["non_entity_title_penalty"]
    if looks_like_acronym_title(candidate):
        breakdown["acronym_title_penalty"] = weights["acronym_title_penalty"]

    retrieval_pool_size = int(candidate.get("retrieval_pool_size") or 0)
    if retrieval_pool_size >= 4 and not candidate_types:
        breakdown["ambiguous_alias_penalty"] = weights["ambiguous_alias_penalty"]

    translit_similarity = best_transliteration_similarity(mention, candidate)
    title_forms_available = bool(english_title_candidates(candidate))
    if translit_similarity >= 0.75:
        breakdown["transliteration_match_bonus"] = round(weights["transliteration_match_bonus"] * translit_similarity, 4)
    elif title_forms_available and not candidate_types and translit_similarity <= 0.6:
        breakdown["transliteration_mismatch_penalty"] = weights["transliteration_mismatch_penalty"]
    candidate["transliteration_similarity"] = translit_similarity

    score = round(sum(breakdown.values()), 6)
    return score, breakdown


def confidence_from_score(score: float) -> float:
    return round(1.0 / (1.0 + math.exp(-(score - 2.8))), 6)


def choose_best_candidates(
    conn,
    mention: dict,
    sentence_text: str,
    tokens: Sequence[dict],
    *,
    expected_types: Set[str],
    weights: Dict[str, float],
    top_k: int,
) -> Tuple[List[dict], List[dict]]:
    generation_steps = generate_surface_steps(mention, tokens)
    head_token = head_token_for_span(tokens, (int(mention["start"]), int(mention["end"])))
    head_text = str((head_token or {}).get("text", "")).strip()
    by_uri: Dict[str, dict] = {}

    for step in generation_steps:
        query = str(step["query"]).strip()
        if not query:
            continue
        if step["lookup"] == "exact":
            candidate_rows = db_fetch_exact_candidates(conn, query, limit=max(top_k * 6, 30))
        else:
            candidate_rows = db_fetch_candidates(conn, query, limit=max(top_k * 6, 30))

        for candidate in candidate_rows:
            canonical_uri = str(candidate.get("canonical_uri", "")).strip()
            if not canonical_uri:
                continue
            candidate = dict(candidate)
            candidate["retrieval_pool_size"] = len(candidate_rows)
            score, feature_breakdown = build_candidate_features(
                mention,
                sentence_text,
                candidate,
                query_step=str(step["step"]),
                expected_types=expected_types,
                weights=weights,
                head_text=head_text,
            )
            enriched = dict(candidate)
            enriched["score"] = score
            enriched["confidence"] = confidence_from_score(score)
            enriched["matched_via"] = f"{candidate.get('match_type', 'alias')}:{step['step']}"
            enriched["matched_lang"] = str(candidate.get("lang", "")).strip() or "unknown"
            enriched["candidate_source"] = str(candidate.get("match_type", "alias"))
            enriched["feature_breakdown"] = feature_breakdown
            enriched["transliteration_similarity"] = float(candidate.get("transliteration_similarity", 0.0) or 0.0)
            enriched["why_selected"] = ", ".join(f"{k}={v:.2f}" for k, v in sorted(feature_breakdown.items()))
            current = by_uri.get(canonical_uri)
            if current is None or float(enriched["score"]) > float(current["score"]):
                by_uri[canonical_uri] = enriched

    ranked = list(by_uri.values())
    ranked.sort(key=lambda item: (-float(item.get("score", 0.0)), -float(item.get("popularity_score", 0.0) or 0.0), str(item.get("canonical_uri", ""))))
    return ranked[:top_k], generation_steps


def collect_mentions(
    input_path: Path,
    ner_map: Dict[str, List[dict]],
    time_map: Dict[str, List[dict]],
    pos_map: Dict[str, List[dict]],
) -> Dict[str, List[dict]]:
    mentions_by_sent: Dict[str, List[dict]] = defaultdict(list)
    seen_keys: Set[Tuple[str, str, int, int]] = set()

    for event in iter_input_events(input_path):
        event_id = stable_event_id(event)
        sent_key_value = sentence_key(event.get("doc_id", "batch_run"), event.get("sent_id", 0))
        for role, payload in (event.get("arguments", {}) or {}).items():
            if not isinstance(payload, dict):
                continue
            span = normalize_char_span(payload.get("char_span"))
            if span is None:
                continue
            text = " ".join(str(payload.get("text", "")).split()).strip()
            if not text:
                continue
            marker = (sent_key_value, text, span[0], span[1])
            if marker in seen_keys:
                continue
            mentions_by_sent[sent_key_value].append(
                base_mention_record(
                    sent_key_value=sent_key_value,
                    mention_uri_value=mention_uri(event_id, role),
                    mention_source="argument",
                    text=text,
                    start=span[0],
                    end=span[1],
                    role=str(role),
                )
            )
            seen_keys.add(marker)

    for sent_key_value, entities in ner_map.items():
        for entity in entities:
            label = str(entity.get("label", entity.get("raw_label", "MISC"))).strip().upper() or "MISC"
            if label in SKIP_NER_LABELS:
                continue
            start = safe_int(entity.get("start"))
            end = safe_int(entity.get("end"))
            if start < 0 or end < start:
                continue
            text = " ".join(str(entity.get("text", "")).split()).strip()
            if not text:
                continue
            marker = (sent_key_value, text, start, end)
            if marker in seen_keys:
                continue
            mentions_by_sent[sent_key_value].append(
                base_mention_record(
                    sent_key_value=sent_key_value,
                    mention_uri_value=entity_mention_uri(sent_key_value, start, end, label),
                    mention_source="ner",
                    text=text,
                    start=start,
                    end=end,
                    ner_label=label,
                )
            )
            seen_keys.add(marker)

    return mentions_by_sent


def record_by_sentence(accepted_links: List[dict], mention_evaluations: List[dict]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for row in accepted_links:
        sent_key_value = str(row.get("sent_key", "batch_run::0"))
        doc_id, sent_id = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
        out.setdefault(doc_id, {}).setdefault(sent_id, {"links": [], "mentions": []})["links"].append(row)
    for row in mention_evaluations:
        sent_key_value = str(row.get("sent_key", "batch_run::0"))
        doc_id, sent_id = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
        out.setdefault(doc_id, {}).setdefault(sent_id, {"links": [], "mentions": []})["mentions"].append(row)
    return out


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    sentences_path = Path(args.sentences)
    db_path = Path(args.dbpedia_index)
    if not input_path.exists():
        raise SystemExit(f"Extraction input not found: {input_path}")
    if not sentences_path.exists():
        raise SystemExit(f"Sentence file not found: {sentences_path}")
    if not db_path.exists():
        raise SystemExit(f"DBpedia index not found: {db_path}")

    sentences = load_sentence_lines(sentences_path)
    pos_path = Path(args.syntax_json) if args.syntax_json else (Path(args.pos_json) if args.pos_json else None)
    pos_map = load_pos_map(pos_path)
    ner_map = load_annotation_map(Path(args.ner_json), "entities") if args.ner_json else {}
    time_map = load_annotation_map(Path(args.time_json), "timexes") if args.time_json else {}
    ner_type_map = load_type_map(Path(args.ner_type_map))
    role_type_map = load_type_map(Path(args.role_type_map))
    weights = load_weight_config(Path(args.linking_config))
    conn = connect_db(db_path)

    mentions_by_sent = collect_mentions(input_path, ner_map, time_map, pos_map)
    accepted_links: List[dict] = []
    mention_evaluations: List[dict] = []
    warnings: List[str] = []
    match_counter = Counter()
    entity_type_counter = Counter()
    source_counter = Counter()
    counters = Counter()
    debug_rows: List[dict] = []

    if args.engine in {"lookup", "spotlight", "hybrid"} and not (args.lookup_url or args.spotlight_url):
        warnings.append("online_backends_requested_without_urls_falling_back_to_local")

    for sent_key_value, mentions in mentions_by_sent.items():
        sent_id_value = sent_key_value.split("::", 1)[1] if "::" in sent_key_value else "0"
        sentence_text = sentence_text_for_sent_id(sentences, sent_id_value)
        sentence_tokens = pos_map.get(sent_key_value, [])
        for mention in mentions:
            counters["total_raw_candidate_mentions"] += 1
            accepted, skip_reason, likely_linkable = determine_entity_likelihood(
                mention,
                sentence_tokens,
                time_map,
                link_only_ner=args.link_only_ner,
                allow_propn_head_fallback=args.allow_propn_head_fallback,
                allow_misc=args.allow_misc,
                skip_pronouns=args.skip_pronouns,
                skip_common_nouns=args.skip_common_nouns,
            )
            evaluation = dict(mention)
            evaluation["likely_linkable"] = likely_linkable
            evaluation["status"] = "skipped"
            evaluation["candidate_generation_steps"] = []
            evaluation["candidate_sources"] = []
            evaluation["top_candidates"] = []
            evaluation["no_link_reason"] = skip_reason
            if likely_linkable:
                counters["likely_linkable_mentions"] += 1
            if not accepted:
                counters[skip_reason] += 1
                mention_evaluations.append(evaluation)
                continue

            counters["total_filtered_candidate_mentions"] += 1
            expected_types = expected_types_for_mention(
                str(mention.get("ner_label", "")),
                str(mention.get("role", "")),
                ner_type_map,
                role_type_map,
            )
            top_candidates, generation_steps = choose_best_candidates(
                conn,
                mention,
                sentence_text,
                sentence_tokens,
                expected_types=expected_types,
                weights=weights,
                top_k=args.top_k,
            )
            evaluation["candidate_generation_steps"] = generation_steps
            evaluation["candidate_sources"] = sorted({str(item.get("candidate_source", "")) for item in top_candidates if str(item.get("candidate_source", "")).strip()})
            evaluation["top_candidates"] = [
                {
                    "canonical_uri": item.get("canonical_uri", ""),
                    "score": float(item.get("score", 0.0)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "matched_label": item.get("alias", ""),
                    "matched_lang": item.get("matched_lang", ""),
                    "matched_via": item.get("matched_via", ""),
                    "types_dbo": item.get("types_dbo", []),
                    "feature_breakdown": item.get("feature_breakdown", {}),
                }
                for item in top_candidates
            ]

            if not top_candidates:
                counters["mentions_with_no_candidates"] += 1
                evaluation["no_link_reason"] = "no_candidates"
                mention_evaluations.append(evaluation)
                continue

            counters["candidate_recall_at_k"] += 1
            best = top_candidates[0]
            confidence = float(best.get("confidence", 0.0))
            if confidence < args.min_confidence:
                counters["mentions_below_confidence_threshold"] += 1
                evaluation["status"] = "rejected"
                evaluation["no_link_reason"] = "low_confidence"
                mention_evaluations.append(evaluation)
                continue

            link = dict(mention)
            link.update(
                {
                    "canonical_uri": best.get("canonical_uri", ""),
                    "score": float(best.get("score", 0.0)),
                    "confidence": confidence,
                    "link_engine": "local",
                    "model_or_backend": "sqlite_local_index",
                    "candidate_count": len(top_candidates),
                    "candidate_generation_steps": generation_steps,
                    "candidate_sources": evaluation["candidate_sources"],
                    "top_candidates": evaluation["top_candidates"],
                    "matched_label": best.get("alias", ""),
                    "matched_lang": best.get("matched_lang", ""),
                    "matched_via": best.get("matched_via", ""),
                    "predicted_dbo_types": best.get("types_dbo", []),
                    "wikidata_uri": best.get("wikidata_uri", ""),
                    "sameas_language_uris": best.get("sameas_language_uris", []),
                    "feature_breakdown": best.get("feature_breakdown", {}),
                    "why_selected": best.get("why_selected", ""),
                    "notes": [],
                    "warnings": [],
                    "no_link_reason": "",
                }
            )
            accepted_links.append(link)
            evaluation["status"] = "linked"
            evaluation["no_link_reason"] = ""
            evaluation["matched_via"] = link["matched_via"]
            evaluation["predicted_uri"] = link["canonical_uri"]
            evaluation["confidence"] = link["confidence"]
            evaluation["why_selected"] = link["why_selected"]
            entity_type_counter[str(mention.get("ner_label", "UNLABELED")) or "UNLABELED"] += 1
            source_counter[str(mention.get("mention_source", "unknown")) or "unknown"] += 1
            match_counter[str(link.get("matched_via", "unknown"))] += 1
            if len(debug_rows) < max(args.debug_samples, 0):
                debug_rows.append(link)
            mention_evaluations.append(evaluation)

    payload = {
        "meta": {
            "engine": args.engine,
            "effective_engine": "local",
            "dbpedia_index": str(db_path.resolve()),
            "input": str(input_path.resolve()),
            "sentences": str(sentences_path.resolve()),
            "total_raw_candidate_mentions": counters["total_raw_candidate_mentions"],
            "total_filtered_candidate_mentions": counters["total_filtered_candidate_mentions"],
            "likely_linkable_mentions": counters["likely_linkable_mentions"],
            "linked_mentions": len(accepted_links),
            "skipped_mentions": len([row for row in mention_evaluations if row.get("status") == "skipped"]),
            "rejected_mentions": len([row for row in mention_evaluations if row.get("status") == "rejected"]),
            "link_coverage_on_likely_linkable_mentions": round(len(accepted_links) / counters["likely_linkable_mentions"], 4) if counters["likely_linkable_mentions"] else 0.0,
            "candidate_recall_at_k": round(counters["candidate_recall_at_k"] / counters["total_filtered_candidate_mentions"], 4) if counters["total_filtered_candidate_mentions"] else 0.0,
            "links_per_entity_type": dict(sorted(entity_type_counter.items())),
            "links_per_source": dict(sorted(source_counter.items())),
            "match_via_counts": dict(sorted(match_counter.items())),
            "filter_counters": {
                "skipped_due_to_pronoun": counters["skipped_due_to_pronoun"],
                "skipped_due_to_common_noun": counters["skipped_due_to_common_noun"],
                "skipped_due_to_temporal": counters["skipped_due_to_temporal"],
                "skipped_due_to_numeric": counters["skipped_due_to_numeric"],
                "skipped_due_to_low_entity_likelihood": counters["skipped_due_to_low_entity_likelihood"],
            },
            "warnings": warnings,
        },
        "sentences": record_by_sentence(accepted_links, mention_evaluations),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    conn.close()
    print(f"[OK] wrote_dbpedia_links={out_path}")
    print(
        f"[OK] raw_candidates={payload['meta']['total_raw_candidate_mentions']} "
        f"filtered_candidates={payload['meta']['total_filtered_candidate_mentions']} "
        f"likely_linkable={payload['meta']['likely_linkable_mentions']} "
        f"linked_mentions={payload['meta']['linked_mentions']}"
    )
    if debug_rows:
        for idx, row in enumerate(debug_rows, start=1):
            print(
                f"[DBG] sample_{idx}: mention={row.get('text', '')} "
                f"uri={row.get('canonical_uri', '')} via={row.get('matched_via', '')} "
                f"confidence={row.get('confidence', 0.0):.3f}"
            )


if __name__ == "__main__":
    main()
