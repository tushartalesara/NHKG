#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build document-level canonical entity clusters and conservative pronoun links."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .enrichment_common import (
        collect_sentence_keys,
        entity_mention_uri,
        iter_input_events,
        load_sentence_text_map,
        mention_uri,
        normalize_char_span,
        sentence_key,
        spans_overlap,
        stable_event_id,
    )
    from .internal_quality_common import (
        DEFAULT_STAGE_VERSION,
        build_mention_forms,
        build_stage_metadata,
        cluster_uri,
        context_excerpt,
        get_pronoun_profile,
        head_summary,
        is_numeric_text,
        load_json,
        load_yaml_config,
        normalize_text,
        role_to_type_hint,
        save_json,
        select_head_token,
        sentence_distance,
    )
except ImportError:  # pragma: no cover
    from enrichment_common import (
        collect_sentence_keys,
        entity_mention_uri,
        iter_input_events,
        load_sentence_text_map,
        mention_uri,
        normalize_char_span,
        sentence_key,
        spans_overlap,
        stable_event_id,
    )
    from internal_quality_common import (
        DEFAULT_STAGE_VERSION,
        build_mention_forms,
        build_stage_metadata,
        cluster_uri,
        context_excerpt,
        get_pronoun_profile,
        head_summary,
        is_numeric_text,
        load_json,
        load_yaml_config,
        normalize_text,
        role_to_type_hint,
        save_json,
        select_head_token,
        sentence_distance,
    )


DEFAULT_CONFIG = {
    "allow_misc": False,
    "cluster_threshold": 0.72,
    "pronoun_threshold": 0.62,
    "pronoun_max_sentence_distance": 2,
    "low_confidence_pronoun_margin": 0.08,
    "repeat_nominal_min_mentions": 2,
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
ALLOWED_NER_TYPES = {"PER", "LOC", "ORG"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create canonical entity clusters and conservative coreference links.")
    parser.add_argument("--input", required=True, help="Extraction JSONL/JSON")
    parser.add_argument("--sentences", required=True, help="One sentence per line text file")
    parser.add_argument("--syntax-json", default="", help="POS/syntax cache")
    parser.add_argument("--ner-json", default="", help="NER cache")
    parser.add_argument("--time-json", default="", help="Temporal cache used to exclude timex spans")
    parser.add_argument("--config", default="lexicons/entity_clustering_config.yaml", help="Optional YAML config")
    parser.add_argument("--out", required=True, help="Output entity cluster cache JSON")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of sample clusters to print")
    return parser


def load_sentence_entries(input_path: Path, sentence_path: Path) -> List[Tuple[str, str]]:
    keys = collect_sentence_keys(input_path)
    sentence_map = load_sentence_text_map(sentence_path, sentence_keys=keys)
    if keys:
        ordered = [(key, sentence_map[key]) for key in keys if key in sentence_map]
        if ordered:
            return ordered
    return list(sentence_map.items())


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


def load_time_cache(path: Optional[Path]) -> Dict[str, List[Tuple[int, int]]]:
    payload = load_json(path, {})
    out: Dict[str, List[Tuple[int, int]]] = {}
    if not isinstance(payload, dict):
        return out
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            spans: List[Tuple[int, int]] = []
            if not isinstance(item, dict):
                continue
            for timex in item.get("timexes", []):
                span = normalize_char_span([timex.get("start"), timex.get("end")]) if isinstance(timex, dict) else None
                if span is not None:
                    spans.append(span)
            out[f"{doc_id}::{sent_id}"] = spans
    return out


def overlaps_any(span: Optional[Tuple[int, int]], spans: Sequence[Tuple[int, int]]) -> bool:
    if span is None:
        return False
    return any(spans_overlap(span, other) for other in spans)


def overlaps_ner(span: Optional[Tuple[int, int]], ner_mentions: Sequence[dict]) -> Optional[dict]:
    if span is None:
        return None
    best = None
    for entity in ner_mentions:
        entity_span = normalize_char_span([entity.get("start"), entity.get("end")]) if isinstance(entity, dict) else None
        if entity_span is None or not spans_overlap(span, entity_span):
            continue
        label = str(entity.get("label", "")).upper()
        if label in {"TIME", "NUM"}:
            continue
        best = entity
        break
    return best


def build_ner_mentions(ner_by_sentence: Dict[str, List[dict]]) -> List[dict]:
    mentions: List[dict] = []
    for sent_key_value, entities in sorted(ner_by_sentence.items()):
        doc_id, sent_id = sent_key_value.split("::", 1)
        for entity in entities:
            label = str(entity.get("label", "")).upper()
            if label not in ALLOWED_NER_TYPES:
                continue
            span = normalize_char_span([entity.get("start"), entity.get("end")])
            if span is None:
                continue
            forms = build_mention_forms(entity.get("text", ""))
            mention = {
                "doc_id": doc_id,
                "sent_id": sent_id,
                "sent_key": sent_key_value,
                "text": normalize_text(entity.get("text", "")),
                "start": span[0],
                "end": span[1],
                "span": span,
                "source": "ner",
                "source_uri": entity_mention_uri(sent_key_value, span[0], span[1], label),
                "predicted_entity_type": label,
                "score": float(entity.get("confidence", entity.get("score", 0.7)) or 0.7),
                "forms": forms,
                "role": "",
                "role_hint_type": label,
                "role_in_cluster": "head",
                "pronoun": False,
                "head": {},
                "evidence": ["ner_label"],
            }
            mentions.append(mention)
    return mentions


def build_argument_mentions(
    input_path: Path,
    sentence_texts: Dict[str, str],
    pos_by_sentence: Dict[str, List[dict]],
    ner_by_sentence: Dict[str, List[dict]],
    timex_by_sentence: Dict[str, List[Tuple[int, int]]],
    config: dict,
) -> Tuple[List[dict], List[dict]]:
    raw_mentions: List[dict] = []
    unresolved: List[dict] = []

    for event in iter_input_events(input_path):
        doc_id = str(event.get("doc_id", "batch_run"))
        sent_id = str(event.get("sent_id", 0))
        sent_key_value = sentence_key(doc_id, sent_id)
        tokens = pos_by_sentence.get(sent_key_value, [])
        ner_mentions = ner_by_sentence.get(sent_key_value, [])
        timex_spans = timex_by_sentence.get(sent_key_value, [])
        event_id = stable_event_id(event)
        arguments = event.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            continue
        for role, arg in arguments.items():
            if not isinstance(arg, dict):
                continue
            text = normalize_text(arg.get("text", ""))
            span = normalize_char_span(arg.get("char_span"))
            if not text or span is None:
                continue
            forms = build_mention_forms(text)
            head_token = select_head_token(tokens, span)
            head = head_summary(head_token)
            ner_match = overlaps_ner(span, ner_mentions)
            pronoun_profile = get_pronoun_profile(forms["cleaned_text"] or forms["normalized_text"])
            is_pronoun = pronoun_profile is not None or head.get("upos") == "PRON"
            if overlaps_any(span, timex_spans):
                unresolved.append(
                    {
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "sent_key": sent_key_value,
                        "text": text,
                        "start": span[0],
                        "end": span[1],
                        "reason": "temporal_overlap",
                    }
                )
                continue
            if is_numeric_text(forms["cleaned_text"] or forms["normalized_text"]):
                unresolved.append(
                    {
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "sent_key": sent_key_value,
                        "text": text,
                        "start": span[0],
                        "end": span[1],
                        "reason": "numeric_expression",
                    }
                )
                continue

            propn_evidence = head.get("upos") == "PROPN" or any(
                str(token.get("upos", "")).upper() == "PROPN" for token in tokens if spans_overlap(span, normalize_char_span([token.get("start"), token.get("end")]) or (-1, -1))
            )
            entity_like = bool(ner_match) or propn_evidence or is_pronoun
            predicted_type = str(ner_match.get("label", "")).upper() if isinstance(ner_match, dict) else role_to_type_hint(role, config)
            raw_mentions.append(
                {
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "sent_key": sent_key_value,
                    "text": text,
                    "start": span[0],
                    "end": span[1],
                    "span": span,
                    "source": "argument",
                    "source_uri": mention_uri(event_id, role),
                    "event_id": event_id,
                    "role": str(role),
                    "predicted_entity_type": predicted_type,
                    "role_hint_type": role_to_type_hint(role, config),
                    "score": 0.78 if entity_like and not is_pronoun else (0.62 if is_pronoun else 0.45),
                    "forms": forms,
                    "pronoun": is_pronoun,
                    "pronoun_profile": pronoun_profile or {},
                    "head": head,
                    "propn_evidence": propn_evidence,
                    "role_in_cluster": "pronoun" if is_pronoun else "head",
                    "evidence": [
                        signal
                        for signal, keep in (
                            ("role_argument", True),
                            ("ner_overlap", bool(ner_match)),
                            ("propn_head", propn_evidence),
                            ("pronoun", is_pronoun),
                        )
                        if keep
                    ],
                    "sentence_context": context_excerpt(sentence_texts.get(sent_key_value, ""), span[0], span[1]),
                }
            )
    return raw_mentions, unresolved


def filter_argument_mentions(raw_mentions: Sequence[dict], config: dict) -> List[dict]:
    counts = Counter()
    for mention in raw_mentions:
        forms = mention.get("forms", {})
        key = str(forms.get("cleaned_text", "") or forms.get("normalized_text", "")).casefold()
        if key:
            counts[key] += 1

    filtered: List[dict] = []
    for mention in raw_mentions:
        forms = mention.get("forms", {})
        cleaned = str(forms.get("cleaned_text", "") or forms.get("normalized_text", ""))
        if mention.get("pronoun"):
            filtered.append(dict(mention))
            continue
        if mention.get("predicted_entity_type", "").upper() in ALLOWED_NER_TYPES:
            filtered.append(dict(mention))
            continue
        if mention.get("propn_evidence"):
            filtered.append(dict(mention))
            continue
        if counts[cleaned.casefold()] >= int(config.get("repeat_nominal_min_mentions", 2)) and str(mention.get("head", {}).get("upos", "")).upper() in {"NOUN", "PROPN"}:
            filtered.append(dict(mention))
    return filtered


def cluster_score(mention: dict, cluster: dict) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    score = 0.0
    mention_type = str(mention.get("predicted_entity_type", "MISC")).upper()
    cluster_type = str(cluster.get("predicted_entity_type", "MISC")).upper()
    if mention_type in ALLOWED_NER_TYPES and cluster_type in ALLOWED_NER_TYPES and mention_type != cluster_type:
        return -1.0, ["type_conflict"]

    mention_forms = {form.casefold() for form in mention.get("forms", {}).get("alternate_forms", [])}
    cluster_forms = set(cluster.get("form_index", set()))
    if mention_forms & cluster_forms:
        score += 0.48
        evidence.append("alternate_form_match")
    if str(mention.get("forms", {}).get("cleaned_text", "")).casefold() == str(cluster.get("canonical_normalized_text", "")).casefold():
        score += 0.22
        evidence.append("cleaned_match")
    if str(mention.get("forms", {}).get("normalized_text", "")).casefold() == str(cluster.get("canonical_normalized_text", "")).casefold():
        score += 0.15
        evidence.append("normalized_match")
    mention_head = str(mention.get("head", {}).get("text", "")).casefold()
    cluster_head = str(cluster.get("head_text", "")).casefold()
    if mention_head and cluster_head and mention_head == cluster_head:
        score += 0.10
        evidence.append("head_surface_match")
    mention_lemma = str(mention.get("head", {}).get("lemma", "")).casefold()
    cluster_lemma = str(cluster.get("head_lemma", "")).casefold()
    if mention_lemma and cluster_lemma and mention_lemma == cluster_lemma:
        score += 0.08
        evidence.append("head_lemma_match")
    if mention_type and mention_type == cluster_type:
        score += 0.08
        evidence.append("entity_type_match")
    if mention.get("propn_evidence"):
        score += 0.04
        evidence.append("propn_evidence")
    sent_gap = sentence_distance(mention.get("sent_id"), cluster.get("last_sent_id"))
    if sent_gap <= 1:
        score += 0.05
        evidence.append("nearby_sentence")
    elif sent_gap <= 2:
        score += 0.02
        evidence.append("document_locality")
    return score, evidence


def build_cluster_record(doc_id: str, mention: dict, index: int) -> dict:
    forms = mention.get("forms", {})
    canonical_text = str(forms.get("cleaned_text", "") or mention.get("text", ""))
    cluster_id = f"c{index:04d}"
    return {
        "cluster_id": cluster_id,
        "cluster_uri": cluster_uri(doc_id, cluster_id),
        "doc_id": doc_id,
        "canonical_text": canonical_text,
        "canonical_normalized_text": str(forms.get("cleaned_text", "") or forms.get("normalized_text", "")),
        "predicted_entity_type": str(mention.get("predicted_entity_type", "MISC")).upper(),
        "mentions": [],
        "representative_mention": None,
        "confidence": 0.0,
        "evidence_summary": [],
        "conflicts": [],
        "form_index": {form.casefold() for form in forms.get("alternate_forms", [])},
        "head_text": str(mention.get("head", {}).get("text", "")),
        "head_lemma": str(mention.get("head", {}).get("lemma", "")),
        "last_sent_id": str(mention.get("sent_id", "0")),
        "type_votes": Counter(),
    }


def attach_mention(cluster: dict, mention: dict, score: float, evidence: Sequence[str]) -> None:
    item = {
        "text": mention.get("text", ""),
        "start": mention.get("start"),
        "end": mention.get("end"),
        "sent_key": mention.get("sent_key"),
        "sent_id": mention.get("sent_id"),
        "mention_uri": mention.get("source_uri"),
        "role_in_cluster": mention.get("role_in_cluster", "alias"),
        "confidence": round(score if score >= 0 else float(mention.get("score", 0.5)), 4),
        "evidence": list(evidence) or list(mention.get("evidence", [])),
        "source": mention.get("source"),
        "role": mention.get("role", ""),
        "predicted_entity_type": mention.get("predicted_entity_type", "MISC"),
        "normalized_text": mention.get("forms", {}).get("normalized_text", ""),
        "cleaned_text": mention.get("forms", {}).get("cleaned_text", ""),
        "head": mention.get("head", {}),
    }
    cluster["mentions"].append(item)
    cluster["form_index"].update(form.casefold() for form in mention.get("forms", {}).get("alternate_forms", []))
    cluster["last_sent_id"] = str(mention.get("sent_id", cluster.get("last_sent_id", "0")))
    predicted_type = str(mention.get("predicted_entity_type", "MISC")).upper()
    cluster["type_votes"][predicted_type] += 1
    if cluster["representative_mention"] is None or (
        item["role_in_cluster"] != "pronoun"
        and len(str(item.get("cleaned_text", ""))) >= len(str(cluster["representative_mention"].get("cleaned_text", "")))
    ):
        cluster["representative_mention"] = item
        cluster["canonical_text"] = item["cleaned_text"] or item["text"]
        cluster["canonical_normalized_text"] = item["cleaned_text"] or item["normalized_text"]
        cluster["head_text"] = str(item.get("head", {}).get("text", ""))
        cluster["head_lemma"] = str(item.get("head", {}).get("lemma", ""))


def finalize_cluster(cluster: dict) -> dict:
    mentions = cluster.get("mentions", [])
    confidences = [float(item.get("confidence", 0.0) or 0.0) for item in mentions if isinstance(item, dict)]
    votes = cluster.get("type_votes", Counter())
    if votes:
        cluster["predicted_entity_type"] = votes.most_common(1)[0][0]
    cluster["confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    evidence_counter = Counter()
    for item in mentions:
        for evidence in item.get("evidence", []):
            evidence_counter[str(evidence)] += 1
    cluster["evidence_summary"] = [f"{label}:{count}" for label, count in evidence_counter.most_common()]
    if len({str(item.get("predicted_entity_type", "MISC")).upper() for item in mentions}) > 2:
        cluster["conflicts"].append("mixed_entity_types")
    cluster.pop("form_index", None)
    cluster.pop("type_votes", None)
    cluster.pop("last_sent_id", None)
    cluster.pop("head_text", None)
    cluster.pop("head_lemma", None)
    return cluster


def resolve_pronouns(doc_id: str, clusters: List[dict], mentions: Sequence[dict], config: dict) -> tuple[List[dict], dict]:
    unresolved: List[dict] = []
    max_distance = int(config.get("pronoun_max_sentence_distance", 2))
    threshold = float(config.get("pronoun_threshold", 0.62))
    low_margin = float(config.get("low_confidence_pronoun_margin", 0.08))
    diagnostics = {
        "pronoun_candidates": len([mention for mention in mentions if isinstance(mention, dict)]),
        "pronoun_links_made": 0,
        "pronoun_unresolved": 0,
        "low_confidence_pronouns": 0,
        "overmerge_prevented_count": 0,
        "pronoun_unresolved_reasons": {},
    }
    for mention in mentions:
        best_cluster = None
        best_score = -1.0
        best_evidence: List[str] = []
        pronoun_profile = mention.get("pronoun_profile", {}) or {}
        role_hint = str(mention.get("role_hint_type", "MISC")).upper()
        considered_candidate = False
        rejection_counts = Counter()
        candidate_summaries: List[dict] = []
        for cluster in clusters:
            representative = cluster.get("representative_mention") or {}
            sent_gap = sentence_distance(mention.get("sent_id"), representative.get("sent_id"))
            if sent_gap > max_distance:
                rejection_counts["distance_too_large"] += 1
                continue
            mention_sent_id = int(mention.get("sent_id", 0) or 0)
            representative_sent_id = int(representative.get("sent_id", 0) or 0)
            representative_start = int(representative.get("start", 0) or 0)
            mention_start = int(mention.get("start", 0) or 0)
            if representative_sent_id > mention_sent_id:
                continue
            if representative_sent_id == mention_sent_id and representative_start >= mention_start:
                continue
            considered_candidate = True
            score = 0.0
            evidence: List[str] = []
            if sent_gap == 0:
                score += 0.32
                evidence.append("same_sentence")
            elif sent_gap == 1:
                score += 0.26
                evidence.append("previous_sentence")
            else:
                score += 0.18
                evidence.append("document_locality")
            cluster_type = str(cluster.get("predicted_entity_type", "MISC")).upper()
            if role_hint in {"PER", "ORG"} and cluster_type in {"PER", "ORG"}:
                score += 0.18
                evidence.append("role_type_compatibility")
            elif role_hint == "LOC" and cluster_type == "LOC":
                score += 0.18
                evidence.append("role_type_compatibility")
            elif role_hint in {"PER", "ORG"} and cluster_type == "LOC":
                rejection_counts["type_conflict"] += 1
                continue
            if bool(pronoun_profile.get("personish")) and cluster_type == "LOC":
                rejection_counts["type_conflict"] += 1
                continue
            if pronoun_profile.get("number") == "plural":
                if len(cluster.get("mentions", [])) > 1 or cluster_type == "ORG":
                    score += 0.08
                    evidence.append("plural_compatible")
                else:
                    rejection_counts["number_gender_conflict"] += 1
            elif len(cluster.get("mentions", [])) >= 1:
                score += 0.05
                evidence.append("singular_compatible")
            candidate_summaries.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "canonical_text": cluster.get("canonical_text", ""),
                    "predicted_entity_type": cluster.get("predicted_entity_type", "MISC"),
                    "score": round(score, 4),
                    "evidence": list(evidence),
                }
            )
            if score > best_score:
                best_cluster = cluster
                best_score = score
                best_evidence = evidence
        if best_cluster is None or best_score < threshold:
            diagnostics["pronoun_unresolved"] += 1
            if considered_candidate:
                diagnostics["overmerge_prevented_count"] += 1
            if best_score >= max(0.0, threshold - low_margin):
                diagnostics["low_confidence_pronouns"] += 1
            near_threshold_candidates = [
                item
                for item in candidate_summaries
                if float(item.get("score", 0.0) or 0.0) >= max(0.0, max(best_score, threshold - low_margin) - 0.03)
            ]
            if considered_candidate and len(near_threshold_candidates) > 1:
                reason = "ambiguity_multiple_candidates"
            elif best_score >= max(0.0, threshold - low_margin):
                reason = "low_confidence"
            elif rejection_counts.get("type_conflict"):
                reason = "type_conflict"
            elif rejection_counts.get("number_gender_conflict"):
                reason = "number_gender_conflict"
            elif rejection_counts.get("distance_too_large"):
                reason = "distance_too_large"
            else:
                reason = "no_compatible_antecedent"
            diagnostics["pronoun_unresolved_reasons"][reason] = int(diagnostics["pronoun_unresolved_reasons"].get(reason, 0) or 0) + 1
            unresolved.append(
                {
                    "doc_id": doc_id,
                    "sent_id": mention.get("sent_id"),
                    "sent_key": mention.get("sent_key"),
                    "text": mention.get("text"),
                    "start": mention.get("start"),
                    "end": mention.get("end"),
                    "sentence_context": mention.get("sentence_context", ""),
                    "reason": reason,
                    "best_score": round(best_score, 4) if best_score >= 0.0 else None,
                    "threshold": threshold,
                    "candidate_antecedents": sorted(candidate_summaries, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)[:3],
                }
            )
            continue
        attach_mention(best_cluster, mention, best_score, list(mention.get("evidence", [])) + best_evidence)
        diagnostics["pronoun_links_made"] += 1
    return unresolved, diagnostics


def print_summary(payload: dict, debug_samples: int) -> None:
    documents = payload.get("documents", {}) or {}
    diagnostics = payload.get("diagnostics", {}) or {}
    cluster_count = 0
    mention_count = 0
    pronoun_links = 0
    type_counter = Counter()
    samples: List[str] = []
    for doc_id, item in documents.items():
        for cluster in item.get("clusters", []):
            cluster_count += 1
            mentions = cluster.get("mentions", [])
            mention_count += len(mentions)
            type_counter[str(cluster.get("predicted_entity_type", "MISC"))] += 1
            pronoun_links += sum(1 for mention in mentions if str(mention.get("role_in_cluster", "")) == "pronoun")
            if len(samples) < max(0, debug_samples):
                members = ", ".join(str(mention.get("text", "")) for mention in mentions[:4])
                samples.append(f"{doc_id}::{cluster.get('cluster_id')} :: {cluster.get('canonical_text')} <= {members}")
    print(f"[OK] canonical_entity_clusters={cluster_count}")
    print(f"[OK] clustered_mentions={mention_count}")
    print(f"[OK] pronoun_linked_mentions={pronoun_links}")
    if diagnostics:
        print(f"[OK] pronoun_diagnostics={diagnostics}")
    print(f"[OK] cluster_types={dict(sorted(type_counter.items()))}")
    for sample in samples:
        print(f"[DBG] {sample}")


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    sentence_path = Path(args.sentences)
    pos_path = Path(args.syntax_json) if args.syntax_json else None
    ner_path = Path(args.ner_json) if args.ner_json else None
    time_path = Path(args.time_json) if args.time_json else None
    config_path = Path(args.config) if args.config else None

    entries = load_sentence_entries(input_path, sentence_path)
    sentence_texts = {key: text for key, text in entries}
    config = load_yaml_config(config_path, DEFAULT_CONFIG)
    pos_by_sentence = load_pos_cache(pos_path)
    ner_by_sentence = load_ner_cache(ner_path)
    timex_by_sentence = load_time_cache(time_path)

    warnings: List[str] = []
    if not ner_by_sentence:
        warnings.append("NER cache unavailable or empty; clustering will rely on syntax and argument spans.")

    ner_mentions = build_ner_mentions(ner_by_sentence)
    raw_argument_mentions, unresolved = build_argument_mentions(
        input_path,
        sentence_texts,
        pos_by_sentence,
        ner_by_sentence,
        timex_by_sentence,
        config,
    )
    argument_mentions = filter_argument_mentions(raw_argument_mentions, config)

    mentions_by_doc: Dict[str, List[dict]] = defaultdict(list)
    pronouns_by_doc: Dict[str, List[dict]] = defaultdict(list)
    for mention in ner_mentions + argument_mentions:
        if mention.get("pronoun"):
            pronouns_by_doc[str(mention.get("doc_id", "batch_run"))].append(mention)
        else:
            mentions_by_doc[str(mention.get("doc_id", "batch_run"))].append(mention)

    documents: Dict[str, dict] = {}
    cluster_threshold = float(config.get("cluster_threshold", 0.72))
    total_clusters = 0

    for doc_id in sorted(set(list(mentions_by_doc.keys()) + list(pronouns_by_doc.keys()))):
        clusters: List[dict] = []
        sorted_mentions = sorted(
            mentions_by_doc.get(doc_id, []),
            key=lambda item: (int(item.get("sent_id", 0)), int(item.get("start", 0)), str(item.get("text", ""))),
        )
        for mention in sorted_mentions:
            best_cluster = None
            best_score = -1.0
            best_evidence: List[str] = []
            for cluster in clusters:
                score, evidence = cluster_score(mention, cluster)
                if score > best_score:
                    best_cluster = cluster
                    best_score = score
                    best_evidence = evidence
            if best_cluster is not None and best_score >= cluster_threshold:
                attach_mention(best_cluster, mention, best_score, list(mention.get("evidence", [])) + best_evidence)
            else:
                cluster = build_cluster_record(doc_id, mention, total_clusters + 1)
                attach_mention(cluster, mention, float(mention.get("score", 0.7)), list(mention.get("evidence", [])) or ["seed"])
                clusters.append(cluster)
                total_clusters += 1

        doc_unresolved, doc_diagnostics = resolve_pronouns(
            doc_id,
            clusters,
            sorted(pronouns_by_doc.get(doc_id, []), key=lambda item: (int(item.get("sent_id", 0)), int(item.get("start", 0)))),
            config,
        )
        unresolved.extend(doc_unresolved)
        documents[doc_id] = {
            "clusters": [finalize_cluster(cluster) for cluster in clusters],
            "unresolved_mentions": sorted(
                [item for item in unresolved if str(item.get("doc_id", "")) == doc_id],
                key=lambda item: (int(item.get("sent_id", 0)), int(item.get("start", 0))),
            ),
            "diagnostics": doc_diagnostics,
        }

    payload_diagnostics = {
        "pronoun_candidates": sum(int((item.get("diagnostics", {}) or {}).get("pronoun_candidates", 0) or 0) for item in documents.values() if isinstance(item, dict)),
        "pronoun_links_made": sum(int((item.get("diagnostics", {}) or {}).get("pronoun_links_made", 0) or 0) for item in documents.values() if isinstance(item, dict)),
        "pronoun_unresolved": sum(int((item.get("diagnostics", {}) or {}).get("pronoun_unresolved", 0) or 0) for item in documents.values() if isinstance(item, dict)),
        "low_confidence_pronouns": sum(int((item.get("diagnostics", {}) or {}).get("low_confidence_pronouns", 0) or 0) for item in documents.values() if isinstance(item, dict)),
        "overmerge_prevented_count": sum(int((item.get("diagnostics", {}) or {}).get("overmerge_prevented_count", 0) or 0) for item in documents.values() if isinstance(item, dict)),
        "pronoun_unresolved_reasons": dict(
            sorted(
                Counter(
                    reason
                    for item in documents.values()
                    if isinstance(item, dict)
                    for reason, count in ((item.get("diagnostics", {}) or {}).get("pronoun_unresolved_reasons", {}) or {}).items()
                    for _ in range(int(count or 0))
                ).items()
            )
        ),
    }

    payload = {
        "meta": build_stage_metadata(
            stage_name="cluster_entities",
            stage_version=DEFAULT_STAGE_VERSION,
            engine="rule_entity_clustering",
            source_paths={
                "input": input_path,
                "sentences": sentence_path,
                "syntax_json": pos_path,
                "ner_json": ner_path,
                "time_json": time_path,
            },
            input_counts={
                "sentences": len(entries),
                "ner_seed_mentions": len(ner_mentions),
                "argument_seed_mentions": len(argument_mentions),
            },
            warnings=warnings,
            config_path=str(config_path) if config_path else "",
            config_snapshot=config,
        ),
        "documents": documents,
        "diagnostics": payload_diagnostics,
    }

    out_path = Path(args.out)
    save_json(out_path, payload)
    print(f"[OK] wrote_entity_cluster_cache={out_path}")
    print_summary(payload, args.debug_samples)


if __name__ == "__main__":
    main()
