#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create additive syntax-guided refinement and validation metadata for extracted events."""

from __future__ import annotations

import argparse
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .enrichment_common import (
        collect_sentence_keys,
        iter_input_events,
        load_sentence_text_map,
        normalize_char_span,
        sentence_key,
        spans_overlap,
        stable_event_id,
    )
    from .internal_quality_common import (
        DEFAULT_STAGE_VERSION,
        build_mention_forms,
        build_stage_metadata,
        context_excerpt,
        head_summary,
        load_json,
        load_yaml_config,
        normalize_text,
        role_to_type_hint,
        save_json,
        select_head_token,
        token_span,
    )
except ImportError:  # pragma: no cover
    from enrichment_common import (
        collect_sentence_keys,
        iter_input_events,
        load_sentence_text_map,
        normalize_char_span,
        sentence_key,
        spans_overlap,
        stable_event_id,
    )
    from internal_quality_common import (
        DEFAULT_STAGE_VERSION,
        build_mention_forms,
        build_stage_metadata,
        context_excerpt,
        head_summary,
        load_json,
        load_yaml_config,
        normalize_text,
        role_to_type_hint,
        save_json,
        select_head_token,
        token_span,
    )


DEFAULT_CONFIG = {
    "plausible_trigger_upos": ["VERB", "ADJ", "NOUN", "PROPN"],
    "plausible_trigger_categories": ["verb", "lexical", "light_verb", "adjective", "noun", "proper_noun", "number"],
    "locative_roles": ["destination", "location", "source", "origin", "place", "venue"],
    "source_roles": ["source", "origin"],
    "destination_roles": ["destination"],
    "manual_review_threshold": 0.65,
    "hard_review_threshold": 0.45,
    "role_type_hints": {
        "agent": "PER",
        "actor": "PER",
        "speaker": "PER",
        "leader": "PER",
        "destination": "LOC",
        "location": "LOC",
        "source": "LOC",
        "origin": "LOC",
        "place": "LOC",
    },
}


def deep_merge_config(base: dict, override: object) -> dict:
    merged = dict(base or {})
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[str(key)] = deep_merge_config(merged.get(key, {}), value)
        else:
            merged[str(key)] = value
    return merged


def normalize_category(value: object) -> str:
    return str(value or "").strip().lower()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add syntax-guided normalization and validation metadata to extracted events.")
    parser.add_argument("--input", required=True, help="Extraction JSONL/JSON")
    parser.add_argument("--sentences", required=True, help="One sentence per line text file")
    parser.add_argument("--syntax-json", default="", help="POS/syntax cache")
    parser.add_argument("--ner-json", default="", help="NER cache")
    parser.add_argument("--time-json", default="", help="Temporal cache")
    parser.add_argument("--coref-json", default="", help="Canonical entity cluster cache")
    parser.add_argument("--config", default="lexicons/refinement_config.yaml", help="YAML refinement config")
    parser.add_argument("--out", required=True, help="Refinement cache output path")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of sample events to print")
    return parser


def load_sentence_entries(input_path: Path, sentence_path: Path) -> Dict[str, str]:
    keys = collect_sentence_keys(input_path)
    return load_sentence_text_map(sentence_path, sentence_keys=keys)


def load_pos_cache(path: Optional[Path]) -> Dict[str, List[dict]]:
    payload = load_json(path, {})
    out: Dict[str, List[dict]] = {}
    if not isinstance(payload, dict):
        return out
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            tokens = item.get("tokens", []) if isinstance(item, dict) else item
            if isinstance(tokens, list):
                out[f"{doc_id}::{sent_id}"] = [token for token in tokens if isinstance(token, dict)]
    return out


def load_ner_cache(path: Optional[Path]) -> Dict[str, List[dict]]:
    payload = load_json(path, {})
    out: Dict[str, List[dict]] = {}
    if not isinstance(payload, dict):
        return out
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            entities = item.get("entities", []) if isinstance(item, dict) else []
            if isinstance(entities, list):
                out[f"{doc_id}::{sent_id}"] = [entity for entity in entities if isinstance(entity, dict)]
    return out


def load_time_cache(path: Optional[Path]) -> Dict[str, dict]:
    payload = load_json(path, {})
    out: Dict[str, dict] = {}
    if not isinstance(payload, dict):
        return out
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if isinstance(item, dict):
                out[f"{doc_id}::{sent_id}"] = item
    return out


def load_coref_mentions(path: Optional[Path]) -> Dict[str, List[dict]]:
    payload = load_json(path, {})
    out: Dict[str, List[dict]] = defaultdict(list)
    if not isinstance(payload, dict):
        return out
    for _, item in (payload.get("documents", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        for cluster in item.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            for mention in cluster.get("mentions", []):
                if not isinstance(mention, dict):
                    continue
                sent_key_value = str(mention.get("sent_key", "")).strip()
                if not sent_key_value:
                    continue
                out[sent_key_value].append(
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "cluster_uri": cluster.get("cluster_uri"),
                        "entity_type": cluster.get("predicted_entity_type", "MISC"),
                        "canonical_text": cluster.get("canonical_text", ""),
                        "start": mention.get("start"),
                        "end": mention.get("end"),
                    }
                )
    return out


def overlap_item(span: Optional[Tuple[int, int]], items: Sequence[dict], *, start_key: str = "start", end_key: str = "end") -> Optional[dict]:
    if span is None:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate_span = normalize_char_span([item.get(start_key), item.get(end_key)])
        if candidate_span is not None and spans_overlap(span, candidate_span):
            return item
    return None


def role_postposition_warning(role: str, original_text: str, config: dict) -> Optional[str]:
    role_name = str(role or "").lower()
    text = normalize_text(original_text)
    if role_name in {name.lower() for name in config.get("source_roles", [])} and "से" not in text:
        return "source_without_se_marker"
    if role_name in {name.lower() for name in config.get("destination_roles", [])} and all(marker not in text for marker in ("को", "तक", "में", "पर")):
        return "destination_without_locative_marker"
    if role_name in {name.lower() for name in config.get("locative_roles", [])} and all(marker not in text for marker in ("में", "पर", "तक", "से", "को")):
        return "locative_role_without_case_marker"
    return None


def token_span_to_char_span(tokens: Sequence[dict], span: object) -> Optional[Tuple[int, int]]:
    if not (isinstance(span, list) and len(span) == 2):
        return None
    try:
        start, end = map(int, span)
    except Exception:
        return None
    if end <= start:
        return None
    matched = []
    for token in tokens:
        try:
            word_index = int(token.get("word_index", 0) or 0) - 1
        except Exception:
            word_index = -1
        if start <= word_index < end:
            offsets = token_span(token)
            if offsets is not None:
                matched.append(offsets)
    if not matched:
        return None
    return min(item[0] for item in matched), max(item[1] for item in matched)


def resolve_trigger_signal(event: dict, tokens: Sequence[dict]) -> dict:
    trigger = event.get("trigger", {}) or {}
    meta = event.get("meta", {}) or {}
    candidate_acceptance = meta.get("candidate_acceptance", {}) or {}
    trigger_reanchoring = meta.get("trigger_reanchoring", {}) or {}

    raw_char_span = (
        normalize_char_span(trigger_reanchoring.get("from_char_span"))
        or token_span_to_char_span(tokens, trigger_reanchoring.get("from_span"))
        or normalize_char_span(trigger.get("char_span"))
    )
    raw_text = normalize_text(trigger_reanchoring.get("from_text", "")) or normalize_text(trigger.get("text", ""))
    raw_head = head_summary(select_head_token(tokens, raw_char_span))

    final_char_span = (
        normalize_char_span(candidate_acceptance.get("final_trigger_char_span"))
        or normalize_char_span(trigger.get("char_span"))
        or token_span_to_char_span(tokens, candidate_acceptance.get("final_trigger_preferred_span"))
        or token_span_to_char_span(tokens, candidate_acceptance.get("final_trigger_span"))
    )
    final_text = normalize_text(candidate_acceptance.get("final_trigger_text", "")) or normalize_text(trigger.get("text", ""))
    final_head = head_summary(select_head_token(tokens, final_char_span))
    final_category = str(
        candidate_acceptance.get("final_trigger_category", "")
        or trigger.get("category", "")
        or candidate_acceptance.get("effective_trigger_category", "")
    ).lower()

    reanchored_char_span = (
        normalize_char_span(trigger_reanchoring.get("to_char_span"))
        or token_span_to_char_span(tokens, trigger_reanchoring.get("to_span"))
    )
    reanchored_text = normalize_text(trigger_reanchoring.get("to_text", ""))
    reanchored_head = head_summary(select_head_token(tokens, reanchored_char_span))
    reanchored_category = str(trigger_reanchoring.get("to_category", "")).lower()

    anchor_char_span = (
        token_span_to_char_span(tokens, candidate_acceptance.get("predicate_center_span"))
        or token_span_to_char_span(tokens, candidate_acceptance.get("anchor_span"))
    )
    anchor_text = normalize_text(candidate_acceptance.get("predicate_center_text", "")) or normalize_text(candidate_acceptance.get("anchor_text", ""))
    anchor_head = head_summary(select_head_token(tokens, anchor_char_span))
    anchor_category = str(
        candidate_acceptance.get("predicate_center_category", "")
        or candidate_acceptance.get("anchor_category", "")
    ).lower()

    source_used = "final_extraction_trigger"
    plausibility_head = final_head
    plausibility_text = final_text
    plausibility_category = final_category
    plausibility_char_span = final_char_span

    if not (plausibility_head.get("text") or plausibility_text or plausibility_category):
        if reanchored_head.get("text") or reanchored_text or reanchored_category:
            source_used = "trigger_reanchoring_target"
            plausibility_head = reanchored_head
            plausibility_text = reanchored_text
            plausibility_category = reanchored_category
            plausibility_char_span = reanchored_char_span
        else:
            source_used = "candidate_acceptance_anchor_fallback"
            plausibility_head = anchor_head
            plausibility_text = anchor_text
            plausibility_category = anchor_category
            plausibility_char_span = anchor_char_span

    return {
        "raw_trigger_text": raw_text,
        "raw_trigger_char_span": list(raw_char_span) if raw_char_span is not None else None,
        "raw_trigger_head": raw_head,
        "final_trigger_text": final_text,
        "final_trigger_char_span": list(final_char_span) if final_char_span is not None else None,
        "final_trigger_head": final_head,
        "trigger_source_used_for_plausibility": source_used,
        "plausibility_head": plausibility_head,
        "plausibility_text": plausibility_text,
        "plausibility_char_span": list(plausibility_char_span) if plausibility_char_span is not None else None,
        "plausibility_category": plausibility_category,
    }


def score_trigger(
    trigger_head: dict,
    warnings: List[str],
    plausible_upos: Sequence[str],
    *,
    fallback_category: str = "",
    plausible_categories: Sequence[str] = (),
) -> Tuple[float, str]:
    score = 0.35
    if trigger_head.get("text"):
        score += 0.2
    head_upos = str(trigger_head.get("upos", "") or "").strip().upper()
    category = normalize_category(fallback_category)
    normalized_plausible_categories = {normalize_category(item) for item in plausible_categories if normalize_category(item)}
    upos_is_plausible = bool(head_upos and head_upos in plausible_upos)
    category_is_plausible = bool(category and category in normalized_plausible_categories)
    if upos_is_plausible:
        score += 0.2
        plausibility_reason = f"head_upos:{head_upos}"
    elif category_is_plausible:
        score += 0.18
        plausibility_reason = f"fallback_category:{category}"
    else:
        warnings.append("trigger_head_not_plausible")
        plausibility_reason = f"no_plausible_upos_or_category:upos={head_upos or '<empty>'},category={category or '<empty>'}"
    if trigger_head.get("deprel") == "root":
        score += 0.1
    if head_upos == "AUX" or category in {"auxiliary", "copula"}:
        warnings.append("trigger_auxiliary_like")
        score -= 0.15
    return max(0.05, min(1.0, score - (0.06 * len(warnings)))), plausibility_reason


def score_argument(aligned_cluster: Optional[dict], aligned_timex: Optional[dict], aligned_ner: Optional[dict], warnings: List[str], head: dict) -> float:
    score = 0.34
    if head.get("text"):
        score += 0.18
    if head.get("upos") in {"PROPN", "NOUN", "PRON", "ADJ"}:
        score += 0.12
    if aligned_cluster is not None:
        score += 0.16
    if aligned_timex is not None:
        score += 0.12
    if aligned_ner is not None:
        score += 0.12
    score -= 0.06 * len(warnings)
    return max(0.05, min(1.0, score))


def warning_severity(label: str) -> str:
    text = str(label or "")
    if text.startswith("overlapping_arguments:"):
        return "hard_review"
    if text.startswith("overlaps_with:"):
        return "caution"
    if text in {"multi_event_sentence", "trigger_auxiliary_like", "trigger_head_not_plausible", "argument_head_function_word"}:
        return "caution"
    if text.startswith("duplicate_role_filler:"):
        return "caution"
    if text in {"source_without_se_marker", "destination_without_locative_marker", "locative_role_without_case_marker", "locative_role_without_locative_alignment"}:
        return "caution"
    return "info"


def warning_details(labels: Sequence[str]) -> List[dict]:
    return [{"label": str(label), "severity": warning_severity(str(label))} for label in sorted({str(label) for label in labels if str(label)})]


def review_priority(event_confidence: float, trigger_confidence: float, warning_rows: Sequence[dict], config: dict) -> str:
    severities = {str(item.get("severity", "")) for item in warning_rows if isinstance(item, dict)}
    hard_threshold = float(config.get("hard_review_threshold", 0.45))
    caution_threshold = float(config.get("manual_review_threshold", 0.65))
    if "hard_review" in severities or event_confidence < hard_threshold:
        return "hard_review"
    if "caution" in severities or trigger_confidence < hard_threshold or event_confidence < caution_threshold:
        return "caution"
    return "info"


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    sentence_path = Path(args.sentences)
    syntax_path = Path(args.syntax_json) if args.syntax_json else None
    ner_path = Path(args.ner_json) if args.ner_json else None
    time_path = Path(args.time_json) if args.time_json else None
    coref_path = Path(args.coref_json) if args.coref_json else None
    config_path = Path(args.config) if args.config else None

    sentence_texts = load_sentence_entries(input_path, sentence_path)
    syntax_by_sentence = load_pos_cache(syntax_path)
    ner_by_sentence = load_ner_cache(ner_path)
    time_by_sentence = load_time_cache(time_path)
    coref_by_sentence = load_coref_mentions(coref_path)
    loaded_config = load_yaml_config(config_path, {})
    config = deep_merge_config(DEFAULT_CONFIG, loaded_config)

    warnings: List[str] = []
    if not syntax_by_sentence:
        warnings.append("Syntax cache unavailable or empty; refinement confidence will be conservative.")

    events_by_sentence: Dict[str, List[dict]] = defaultdict(list)
    event_rows: List[dict] = []
    event_failures: List[dict] = []
    plausible_trigger_upos = [str(value).strip().upper() for value in config.get("plausible_trigger_upos", []) if str(value).strip()]
    plausible_trigger_categories = [normalize_category(value) for value in config.get("plausible_trigger_categories", []) if normalize_category(value)]

    for event in iter_input_events(input_path):
        doc_id = str(event.get("doc_id", "batch_run"))
        sent_id = str(event.get("sent_id", 0))
        events_by_sentence[sentence_key(doc_id, sent_id)].append(event)

    for event in iter_input_events(input_path):
        doc_id = str(event.get("doc_id", "batch_run"))
        sent_id = str(event.get("sent_id", 0))
        sent_key_value = sentence_key(doc_id, sent_id)
        event_id = stable_event_id(event)
        try:
            sentence_text = sentence_texts.get(sent_key_value, "")
            tokens = syntax_by_sentence.get(sent_key_value, [])

            trigger = event.get("trigger", {}) or {}
            trigger_signal = resolve_trigger_signal(event, tokens)
            trigger_span = normalize_char_span(trigger.get("char_span"))
            trigger_head = trigger_signal.get("plausibility_head", {}) or {}
            trigger_warnings: List[str] = []
            trigger_confidence, trigger_plausibility_reason = score_trigger(
                trigger_head,
                trigger_warnings,
                plausible_trigger_upos,
                fallback_category=str(trigger_signal.get("plausibility_category", "")),
                plausible_categories=plausible_trigger_categories,
            )

            arg_rows: Dict[str, dict] = {}
            event_warnings: List[str] = list(trigger_warnings)
            argument_spans: Dict[str, Tuple[int, int]] = {}
            arguments = event.get("arguments", {}) or {}

            for role, arg in arguments.items():
                if not isinstance(arg, dict):
                    continue
                original_text = normalize_text(arg.get("text", ""))
                original_span = normalize_char_span(arg.get("char_span"))
                if not original_text or original_span is None:
                    continue
                forms = build_mention_forms(original_text)
                head = head_summary(select_head_token(tokens, original_span))
                aligned_cluster = overlap_item(original_span, coref_by_sentence.get(sent_key_value, []))
                aligned_ner = overlap_item(original_span, ner_by_sentence.get(sent_key_value, []))
                aligned_timex = overlap_item(original_span, time_by_sentence.get(sent_key_value, {}).get("timexes", []))
                arg_warnings: List[str] = []
                postposition_warning = role_postposition_warning(str(role), original_text, config)
                if postposition_warning:
                    arg_warnings.append(postposition_warning)
                role_hint_type = role_to_type_hint(role, config)
                if str(role_hint_type).upper() == "LOC":
                    cluster_type = str((aligned_cluster or {}).get("entity_type", "")).upper()
                    ner_type = str((aligned_ner or {}).get("label", "")).upper()
                    if cluster_type and cluster_type != "LOC" and ner_type and ner_type != "LOC":
                        arg_warnings.append("locative_role_without_locative_alignment")
                if head.get("upos") in {"ADP", "PART"}:
                    arg_warnings.append("argument_head_function_word")

                arg_confidence = score_argument(aligned_cluster, aligned_timex, aligned_ner, arg_warnings, head)
                arg_rows[str(role)] = {
                    "role": str(role),
                    "original_text": original_text,
                    "original_span": list(original_span),
                    "normalized_text": forms.get("normalized_text", ""),
                    "cleaned_text": forms.get("cleaned_text", ""),
                    "alternate_forms": forms.get("alternate_forms", []),
                    "normalization_actions": forms.get("normalization_actions", []),
                    "head": head,
                    "aligned_entity_cluster_id": (aligned_cluster or {}).get("cluster_id"),
                    "aligned_entity_cluster_uri": (aligned_cluster or {}).get("cluster_uri"),
                    "aligned_entity_type": (aligned_cluster or {}).get("entity_type"),
                    "aligned_ner_label": (aligned_ner or {}).get("label"),
                    "aligned_timex_id": (aligned_timex or {}).get("timex_id"),
                    "aligned_timex_value": (aligned_timex or {}).get("value"),
                    "confidence": round(arg_confidence, 4),
                    "warnings": sorted(set(arg_warnings)),
                    "warning_details": warning_details(arg_warnings),
                    "context": context_excerpt(sentence_text, original_span[0], original_span[1]),
                }
                argument_spans[str(role)] = original_span

            role_names = list(arg_rows.keys())
            for index, left_role in enumerate(role_names):
                for right_role in role_names[index + 1 :]:
                    left_span = argument_spans.get(left_role)
                    right_span = argument_spans.get(right_role)
                    if left_span is not None and right_span is not None and spans_overlap(left_span, right_span):
                        event_warnings.append(f"overlapping_arguments:{left_role}:{right_role}")
                        arg_rows[left_role]["warnings"].append(f"overlaps_with:{right_role}")
                        arg_rows[right_role]["warnings"].append(f"overlaps_with:{left_role}")
                    if arg_rows[left_role].get("cleaned_text") and arg_rows[left_role].get("cleaned_text") == arg_rows[right_role].get("cleaned_text"):
                        event_warnings.append(f"duplicate_role_filler:{left_role}:{right_role}")

            if len(events_by_sentence.get(sent_key_value, [])) > 1:
                event_warnings.append("multi_event_sentence")

            argument_confidences = [float(item.get("confidence", 0.0) or 0.0) for item in arg_rows.values()]
            event_confidence = round(max(0.05, min(1.0, (trigger_confidence + (sum(argument_confidences) / len(argument_confidences) if argument_confidences else trigger_confidence)) / 2.0 - (0.04 * len(event_warnings)))), 4)
            event_warning_details = warning_details(event_warnings)
            priority = review_priority(event_confidence, trigger_confidence, event_warning_details, config)

            event_rows.append(
                {
                    "event_id": event_id,
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "sent_key": sent_key_value,
                    "frame": event.get("frame", ""),
                    "sentence_text": sentence_text,
                    "trigger": {
                        "original_text": normalize_text(trigger.get("text", "")),
                        "original_span": list(trigger_span) if trigger_span is not None else None,
                        "normalized_head_text": trigger_head.get("text", ""),
                        "head": trigger_head,
                        "raw_trigger_text": trigger_signal.get("raw_trigger_text", ""),
                        "raw_trigger_head": trigger_signal.get("raw_trigger_head", {}),
                        "final_trigger_text": trigger_signal.get("final_trigger_text", ""),
                        "final_trigger_head": trigger_signal.get("final_trigger_head", {}),
                        "trigger_source_used_for_plausibility": trigger_signal.get("trigger_source_used_for_plausibility", ""),
                        "trigger_head_upos": str(trigger_head.get("upos", "") or "").strip().upper(),
                        "trigger_head_category": normalize_category(trigger_signal.get("plausibility_category", "")),
                        "trigger_head_plausibility_reason": trigger_plausibility_reason,
                        "plausible_trigger_categories_used": plausible_trigger_categories,
                        "confidence": round(trigger_confidence, 4),
                        "warnings": sorted(set(trigger_warnings)),
                        "warning_details": warning_details(trigger_warnings),
                    },
                    "arguments": arg_rows,
                    "event_confidence": event_confidence,
                    "warnings": sorted(set(event_warnings)),
                    "warning_details": event_warning_details,
                    "warning_severity_counts": {
                        "info": sum(1 for item in event_warning_details if item.get("severity") == "info"),
                        "caution": sum(1 for item in event_warning_details if item.get("severity") == "caution"),
                        "hard_review": sum(1 for item in event_warning_details if item.get("severity") == "hard_review"),
                    },
                    "review_priority": priority,
                    "manual_review": priority == "hard_review",
                    "syntax_evidence": {
                        "token_count": len(tokens),
                        "argument_count": len(arg_rows),
                    },
                }
            )
        except Exception as exc:
            failure = {
                "event_id": event_id,
                "doc_id": doc_id,
                "sent_id": sent_id,
                "sent_key": sent_key_value,
                "frame": event.get("frame", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=8).splitlines()[-8:],
            }
            event_failures.append(failure)
            warnings.append(
                f"Skipped refinement for event {event_id} ({doc_id}::{sent_id}, frame={event.get('frame', '')}) due to {type(exc).__name__}"
            )
            print(
                f"[WARN] skipped_event={event_id} sent={doc_id}::{sent_id} frame={event.get('frame', '')} error={type(exc).__name__}: {exc}",
                flush=True,
            )

    payload = {
        "meta": build_stage_metadata(
            stage_name="refine_arguments_with_syntax",
            stage_version=DEFAULT_STAGE_VERSION,
            engine="rule_syntax_refinement",
            source_paths={
                "input": input_path,
                "sentences": sentence_path,
                "syntax_json": syntax_path,
                "ner_json": ner_path,
                "time_json": time_path,
                "coref_json": coref_path,
            },
            input_counts={"events": len(event_rows), "sentences": len(sentence_texts), "skipped_events": len(event_failures)},
            warnings=warnings,
            config_path=str(config_path) if config_path else "",
            config_snapshot=config,
            extra={"skipped_event_count": len(event_failures)},
        ),
        "events": event_rows,
        "event_failures": event_failures,
    }

    out_path = Path(args.out)
    save_json(out_path, payload)
    priority_counts = Counter(str(row.get("review_priority", "info")) for row in event_rows if isinstance(row, dict))
    print(f"[OK] wrote_refinement_cache={out_path}")
    print(f"[OK] refined_events={len(event_rows)} hard_review_events={sum(1 for row in event_rows if row.get('review_priority') == 'hard_review')} review_priority_counts={dict(sorted(priority_counts.items()))}")
    for sample in event_rows[: max(0, args.debug_samples)]:
        print(f"[DBG] {sample['event_id']} :: trigger={sample['trigger']['normalized_head_text']} event_confidence={sample['event_confidence']:.4f} review_priority={sample.get('review_priority')} warnings={len(sample['warnings'])}")


if __name__ == "__main__":
    main()
