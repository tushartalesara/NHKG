#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute thesis-friendly coverage and quality stats for NHKG enrichment layers."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from common import extract_arg_items, load_event_records, spans_overlap
except ImportError:  # pragma: no cover
    from eval.common import extract_arg_items, load_event_records, spans_overlap

try:
    from align.enrichment_common import NS, parse_nquad_line
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.enrichment_common import NS, parse_nquad_line


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute syntax/coref/refinement/time enrichment statistics.")
    parser.add_argument("--input-jsonl", default="", help="Extraction JSONL/JSON used for event and overlap stats")
    parser.add_argument("--graph", default="", help="Final RDF graph for graph-level counts")
    parser.add_argument("--pos-json", default="", help="POS/syntax cache path")
    parser.add_argument("--ner-json", default="", help="NER cache path")
    parser.add_argument("--time-json", default="", help="Temporal cache path")
    parser.add_argument("--coref-json", default="", help="Entity cluster cache path")
    parser.add_argument("--refinement-json", default="", help="Syntax refinement cache path")
    parser.add_argument("--event-density-json", default="", help="Optional JSON emitted by eval/analyze_event_density.py")
    parser.add_argument("--event-audit-json", default="", help="Optional JSON emitted by eval/audit_event_count_consistency.py")
    parser.add_argument("--validation-json", default="", help="Optional JSON emitted by eval/validate_graph.py")
    parser.add_argument("--candidate-decision-jsonl", default="", help="Optional JSONL emitted by gold/pipeline.py candidate acceptance logging")
    parser.add_argument("--dbpedia-json", default="", help="Optional DBpedia link cache path")
    parser.add_argument("--dbpedia-stats", default="", help="Optional JSON stats emitted by eval/compute_dbpedia_stats.py")
    parser.add_argument("--wordnet-stats", default="", help="Optional JSON stats emitted by align/wordnet_enrich.py")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    return parser


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def load_jsonl(path: Optional[Path]) -> List[dict]:
    if path is None or not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def assess_cache_status(path_value: str, payload: Optional[dict], *, required_keys: Iterable[str] = ()) -> dict:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {"available": False, "reason": "missing_cache", "source_path": ""}
    path = Path(path_text)
    source_path = str(path.resolve())
    if not path.exists():
        return {"available": False, "reason": "missing_cache", "source_path": source_path}
    if payload is None:
        return {"available": False, "reason": "malformed_cache", "source_path": source_path}
    for key in required_keys:
        if key not in payload:
            return {"available": False, "reason": "schema_mismatch", "source_path": source_path}
    return {"available": True, "reason": "", "source_path": source_path}


def iter_sentence_items(payload: Optional[dict]) -> Iterable[Tuple[str, dict]]:
    if not payload:
        return []
    items = []
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if isinstance(item, dict):
                items.append((f"{doc_id}::{sent_id}", item))
    return items


def compute_syntax_stats(payload: Optional[dict]) -> dict:
    token_count = 0
    upos_count = 0
    lemma_count = 0
    syntax_count = 0
    deprel_counts = Counter()
    meta = (payload or {}).get("meta", {}) if isinstance(payload, dict) else {}
    for _, item in iter_sentence_items(payload):
        tokens = item.get("tokens", [])
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            if not isinstance(token, dict):
                continue
            token_count += 1
            if str(token.get("upos", "")).strip():
                upos_count += 1
            if str(token.get("lemma", "")).strip():
                lemma_count += 1
            head = token.get("head")
            deprel = str(token.get("deprel", "")).strip()
            if isinstance(head, int) and head >= 0 and deprel:
                syntax_count += 1
                deprel_counts[deprel] += 1
    return {
        "token_count": token_count,
        "tokens_with_upos": upos_count,
        "tokens_with_lemma": lemma_count,
        "tokens_with_head_deprel": syntax_count,
        "pct_tokens_with_upos": round(upos_count / token_count, 4) if token_count else 0.0,
        "pct_tokens_with_lemma": round(lemma_count / token_count, 4) if token_count else 0.0,
        "pct_tokens_with_head_deprel": round(syntax_count / token_count, 4) if token_count else 0.0,
        "dependency_relation_distribution": dict(sorted(deprel_counts.items())),
        "stage_metadata": meta,
    }


def compute_ner_stats(payload: Optional[dict], events: List[dict]) -> dict:
    total_mentions = 0
    per_type = Counter()
    confidence_values: List[float] = []
    source_counts = Counter()
    backend_counts = Counter()
    argument_spans: Dict[str, List[Tuple[int, int]]] = {}
    meta = (payload or {}).get("meta", {}) if isinstance(payload, dict) else {}
    for event in events:
        key = f"{event.get('doc_id', 'batch_run')}::{event.get('sent_id', 0)}"
        for _, span, _ in extract_arg_items(event):
            if span is not None:
                argument_spans.setdefault(key, []).append(span)
    overlap_mentions = 0
    for sent_key, item in iter_sentence_items(payload):
        mentions = item.get("entities", [])
        if not isinstance(mentions, list):
            continue
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            total_mentions += 1
            per_type[str(mention.get("canonical_label", mention.get("label", "MISC")))] += 1
            source_counts[str(mention.get("source", "unknown"))] += 1
            backend_counts[str(mention.get("backend", mention.get("engine", "unknown")))] += 1
            try:
                score = mention.get("confidence", mention.get("score"))
                if score is not None:
                    confidence_values.append(float(score))
            except (TypeError, ValueError):
                pass
            try:
                start = int(mention.get("start"))
                end = int(mention.get("end"))
            except (TypeError, ValueError):
                continue
            entity_span = (start, end)
            if any(spans_overlap(entity_span, arg_span) for arg_span in argument_spans.get(sent_key, [])):
                overlap_mentions += 1
    backend_used = str(meta.get("engine", "")) or ("local_hf" if backend_counts.get("local_hf") else ("rules" if backend_counts.get("rules") else "unknown"))
    model_name = str(meta.get("model_name", "") or "")
    return {
        "total_entity_mentions": total_mentions,
        "mentions_by_type": dict(sorted(per_type.items())),
        "mentions_per_type": dict(sorted(per_type.items())),
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "overlap_with_argument_mentions": overlap_mentions,
        "overlap_rate_with_argument_mentions": round(overlap_mentions / total_mentions, 4) if total_mentions else 0.0,
        "backend_used": backend_used,
        "model_name": model_name,
        "mentions_per_backend": dict(sorted(backend_counts.items())),
        "mentions_per_source": dict(sorted(source_counts.items())),
        "rule_backed_mentions": int(source_counts.get("rules", 0) or 0),
        "rules_backed_mentions": int(source_counts.get("rules", 0) or 0),
        "model_backed_mentions": int(source_counts.get("model", 0) or 0),
        "percent_from_rules": round(source_counts.get("rules", 0) / total_mentions, 4) if total_mentions else 0.0,
        "percent_from_model": round(source_counts.get("model", 0) / total_mentions, 4) if total_mentions else 0.0,
        "rules_only_mode": backend_used == "rules",
        "stage_metadata": meta,
    }


def compute_temporal_stats(payload: Optional[dict]) -> dict:
    total_timexes = 0
    per_type = Counter()
    resolved = 0
    unresolved = 0
    link_count = 0
    strategy_counts = Counter()
    relation_count = 0
    relation_type_counts = Counter()
    evidence_strength_counts = Counter()
    unresolved_reason_counts = Counter()
    relation_confidences: List[float] = []
    confidence_by_type: Dict[str, List[float]] = {}
    for _, item in iter_sentence_items(payload):
        timexes = item.get("timexes", [])
        if isinstance(timexes, list):
            for timex in timexes:
                if not isinstance(timex, dict):
                    continue
                total_timexes += 1
                per_type[str(timex.get("type", "DATE"))] += 1
                if timex.get("resolution_status") == "resolved":
                    resolved += 1
                else:
                    unresolved += 1
                    for reason in timex.get("unresolved_reasons", []) if isinstance(timex.get("unresolved_reasons", []), list) else []:
                        unresolved_reason_counts[str(reason)] += 1
        links = item.get("event_time_links", [])
        if isinstance(links, list):
            link_count += len([link for link in links if isinstance(link, dict)])
            for link in links:
                if isinstance(link, dict):
                    strategy_counts[str(link.get("strategy", "unknown"))] += 1
        relations = item.get("event_event_relations", [])
        if isinstance(relations, list):
            relation_count += len([relation for relation in relations if isinstance(relation, dict)])
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                relation_type = str(relation.get("relation", "UNKNOWN"))
                relation_type_counts[relation_type] += 1
                evidence_strength = str(relation.get("evidence_strength", "unknown"))
                evidence_strength_counts[evidence_strength] += 1
                try:
                    confidence_value = float(relation.get("confidence", 0.0) or 0.0)
                    relation_confidences.append(confidence_value)
                    confidence_by_type.setdefault(relation_type, []).append(confidence_value)
                except (TypeError, ValueError):
                    pass
    return {
        "total_timexes": total_timexes,
        "timexes_per_type": dict(sorted(per_type.items())),
        "resolved": resolved,
        "unresolved": unresolved,
        "unresolved_reason_distribution": dict(sorted(unresolved_reason_counts.items())),
        "event_time_link_count": link_count,
        "links_by_strategy": dict(sorted(strategy_counts.items())),
        "event_event_relation_count": relation_count,
        "event_event_relations_by_type": dict(sorted(relation_type_counts.items())),
        "event_event_average_confidence": round(sum(relation_confidences) / len(relation_confidences), 4) if relation_confidences else None,
        "evidence_strength_distribution": dict(sorted(evidence_strength_counts.items())),
        "average_confidence_by_relation_type": {key: round(sum(values) / len(values), 4) for key, values in sorted(confidence_by_type.items()) if values},
        "strong_cotemporal_count": relation_type_counts.get("SIMULTANEOUS", 0) + relation_type_counts.get("SIMULTANEOUS_STRONG", 0),
        "weak_cotemporal_count": relation_type_counts.get("SHARED_TIME_ANCHOR", 0),
        "stage_metadata": (payload or {}).get("meta", {}) if isinstance(payload, dict) else {},
    }


def compute_coref_stats(payload: Optional[dict], availability: Optional[dict] = None) -> dict:
    availability = availability or {"available": payload is not None, "reason": "", "source_path": ""}
    if not payload or not availability.get("available"):
        return {
            "available": bool(availability.get("available", False)),
            "reason": str(availability.get("reason", "")),
            "source_path": str(availability.get("source_path", "")),
            "total_entity_mentions": 0,
            "total_entity_mentions_clustered": 0,
            "total_canonical_entity_clusters": 0,
            "canonical_entity_clusters": 0,
            "average_mentions_per_cluster": 0.0,
            "singleton_clusters": 0,
            "multi_mention_clusters": 0,
            "pronoun_linked_mentions": 0,
            "pronoun_candidates": 0,
            "pronoun_links_made": 0,
            "pronoun_unresolved": 0,
            "low_confidence_pronouns": 0,
            "overmerge_prevented_count": 0,
            "cluster_type_distribution": {},
            "merge_evidence_distribution": {},
            "uncertain_clusters": 0,
            "stage_metadata": (payload or {}).get("meta", {}) if isinstance(payload, dict) else {},
        }
    cluster_count = 0
    mention_count = 0
    singleton = 0
    multi = 0
    pronouns = 0
    type_counts = Counter()
    uncertain = 0
    evidence_counts = Counter()
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
    for _, item in (payload.get("documents", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        for cluster in item.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            cluster_count += 1
            mentions = cluster.get("mentions", [])
            mention_count += len(mentions)
            if len(mentions) <= 1:
                singleton += 1
            else:
                multi += 1
            type_counts[str(cluster.get("predicted_entity_type", "MISC"))] += 1
            pronouns += sum(1 for mention in mentions if str(mention.get("role_in_cluster", "")) == "pronoun")
            if float(cluster.get("confidence", 0.0) or 0.0) < 0.7 or cluster.get("conflicts"):
                uncertain += 1
            for label in cluster.get("evidence_summary", []) or []:
                evidence_counts[str(label)] += 1
    return {
        "available": True,
        "reason": "",
        "source_path": str(availability.get("source_path", "")),
        "total_entity_mentions": mention_count,
        "total_entity_mentions_clustered": mention_count,
        "total_canonical_entity_clusters": cluster_count,
        "canonical_entity_clusters": cluster_count,
        "average_mentions_per_cluster": round(mention_count / cluster_count, 4) if cluster_count else 0.0,
        "singleton_clusters": singleton,
        "multi_mention_clusters": multi,
        "pronoun_linked_mentions": pronouns,
        "pronoun_candidates": int(diagnostics.get("pronoun_candidates", 0) or 0),
        "pronoun_links_made": int(diagnostics.get("pronoun_links_made", 0) or 0),
        "pronoun_unresolved": int(diagnostics.get("pronoun_unresolved", 0) or 0),
        "low_confidence_pronouns": int(diagnostics.get("low_confidence_pronouns", 0) or 0),
        "overmerge_prevented_count": int(diagnostics.get("overmerge_prevented_count", 0) or 0),
        "pronoun_unresolved_reasons": dict(sorted(((diagnostics.get("pronoun_unresolved_reasons", {}) or {})).items())),
        "cluster_type_distribution": dict(sorted(type_counts.items())),
        "merge_evidence_distribution": dict(sorted(evidence_counts.items())),
        "uncertain_clusters": uncertain,
        "stage_metadata": (payload or {}).get("meta", {}) if isinstance(payload, dict) else {},
    }


def compute_refinement_stats(payload: Optional[dict], availability: Optional[dict] = None) -> dict:
    availability = availability or {"available": payload is not None, "reason": "", "source_path": ""}
    if not payload or not availability.get("available"):
        return {
            "available": bool(availability.get("available", False)),
            "reason": str(availability.get("reason", "")),
            "source_path": str(availability.get("source_path", "")),
            "events": 0,
            "refinement_events": 0,
            "arguments_normalized": 0,
            "spans_cleaned": 0,
            "arguments_aligned_to_canonical_entity": 0,
            "arguments_aligned_to_timex": 0,
            "warning_counts_by_category": {},
            "warnings_by_category": {},
            "warning_counts_by_severity": {},
            "average_event_confidence": None,
            "average_argument_confidence": None,
            "manual_review_events": 0,
            "hard_review_events": 0,
            "caution_only_events": 0,
            "info_only_events": 0,
            "hard_review_rate": 0.0,
            "caution_rate": 0.0,
            "info_only_rate": 0.0,
            "stage_metadata": (payload or {}).get("meta", {}) if isinstance(payload, dict) else {},
        }
    events = payload.get("events", []) if isinstance(payload, dict) else []
    event_confidences: List[float] = []
    arg_confidences: List[float] = []
    warnings = Counter()
    severity_counts = Counter()
    arguments_normalized = 0
    spans_cleaned = 0
    aligned_clusters = 0
    aligned_timex = 0
    manual_review = 0
    hard_review = 0
    caution_only = 0
    info_only = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            event_confidences.append(float(event.get("event_confidence", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        priority = str(event.get("review_priority", "info"))
        if event.get("manual_review"):
            manual_review += 1
        if priority == "hard_review":
            hard_review += 1
        elif priority == "caution":
            caution_only += 1
        else:
            info_only += 1
        for warning in event.get("warnings", []):
            warnings[str(warning)] += 1
        for item in event.get("warning_details", []) or []:
            if isinstance(item, dict):
                severity_counts[str(item.get("severity", "info"))] += 1
        for item in (event.get("arguments", {}) or {}).values():
            if not isinstance(item, dict):
                continue
            arguments_normalized += 1
            if str(item.get("cleaned_text", "")) and str(item.get("cleaned_text", "")) != str(item.get("original_text", "")):
                spans_cleaned += 1
            if item.get("aligned_entity_cluster_id"):
                aligned_clusters += 1
            if item.get("aligned_timex_id"):
                aligned_timex += 1
            try:
                arg_confidences.append(float(item.get("confidence", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
            for warning in item.get("warnings", []):
                warnings[str(warning)] += 1
            for warning_row in item.get("warning_details", []) or []:
                if isinstance(warning_row, dict):
                    severity_counts[str(warning_row.get("severity", "info"))] += 1
    event_count = len([event for event in events if isinstance(event, dict)])
    return {
        "available": True,
        "reason": "",
        "source_path": str(availability.get("source_path", "")),
        "events": event_count,
        "refinement_events": event_count,
        "arguments_normalized": arguments_normalized,
        "spans_cleaned": spans_cleaned,
        "arguments_aligned_to_canonical_entity": aligned_clusters,
        "arguments_aligned_to_timex": aligned_timex,
        "warning_counts_by_category": dict(sorted(warnings.items())),
        "warnings_by_category": dict(sorted(warnings.items())),
        "warning_counts_by_severity": dict(sorted(severity_counts.items())),
        "average_event_confidence": round(sum(event_confidences) / len(event_confidences), 4) if event_confidences else None,
        "average_argument_confidence": round(sum(arg_confidences) / len(arg_confidences), 4) if arg_confidences else None,
        "manual_review_events": manual_review,
        "hard_review_events": hard_review,
        "caution_only_events": caution_only,
        "info_only_events": info_only,
        "hard_review_rate": round(hard_review / event_count, 4) if event_count else 0.0,
        "caution_rate": round(caution_only / event_count, 4) if event_count else 0.0,
        "info_only_rate": round(info_only / event_count, 4) if event_count else 0.0,
        "stage_metadata": (payload or {}).get("meta", {}) if isinstance(payload, dict) else {},
    }


def compute_dbpedia_stats(payload: Optional[dict], dbpedia_stats_payload: Optional[dict]) -> dict:
    if dbpedia_stats_payload:
        return dbpedia_stats_payload
    if not payload:
        return {"candidate_mentions_seen": 0, "linked_mentions": 0, "coverage": 0.0, "links_per_entity_type": {}}
    total_links = 0
    per_type = Counter()
    candidate_mentions_seen = int((payload.get("meta", {}) or {}).get("candidate_mentions_seen", 0))
    for _, item in iter_sentence_items(payload):
        links = item.get("links", [])
        if not isinstance(links, list):
            continue
        for link in links:
            if not isinstance(link, dict):
                continue
            total_links += 1
            per_type[str(link.get("ner_label", "UNLABELED")) or "UNLABELED"] += 1
    if not candidate_mentions_seen:
        candidate_mentions_seen = total_links
    return {
        "candidate_mentions_seen": candidate_mentions_seen,
        "linked_mentions": total_links,
        "coverage": round(total_links / candidate_mentions_seen, 4) if candidate_mentions_seen else 0.0,
        "links_per_entity_type": dict(sorted(per_type.items())),
    }


def compute_graph_stats(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {
            "total_triples": 0,
            "total_event_nodes": 0,
            "total_token_nodes": 0,
            "total_entity_mention_nodes": 0,
            "total_timex_nodes": 0,
            "total_canonical_entity_nodes": 0,
            "total_temporal_relation_nodes": 0,
            "total_dbpedia_resource_nodes": 0,
        }
    total_triples = 0
    events = set()
    words = set()
    entities = set()
    timexes = set()
    canonical_entities = set()
    temporal_relations = set()
    dbpedia_resources = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            total_triples += 1
            if parsed.subject.startswith("http://dbpedia.org/resource/"):
                dbpedia_resources.add(parsed.subject)
            if parsed.obj.kind == "uri" and str(parsed.obj.value).startswith("http://dbpedia.org/resource/"):
                dbpedia_resources.add(str(parsed.obj.value))
            local = parsed.subject.rsplit("/", 1)[-1]
            if parsed.predicate == f"{NS['rdf']}type" and parsed.obj.kind == "uri":
                if parsed.obj.value == f"{NS['schema']}Event" or local.startswith("Event_"):
                    events.add(parsed.subject)
                elif parsed.obj.value == f"{NS['nif']}Word" or local.startswith("Word_"):
                    words.add(parsed.subject)
                elif parsed.obj.value == f"{NS['nhkg']}EntityMention" or local.startswith("EntityMention_"):
                    entities.add(parsed.subject)
                elif parsed.obj.value == f"{NS['nhkg']}Timex" or local.startswith("Timex_"):
                    timexes.add(parsed.subject)
                elif parsed.obj.value == f"{NS['nhkg']}CanonicalEntity" or local.startswith("CanonicalEntity_"):
                    canonical_entities.add(parsed.subject)
                elif parsed.obj.value == f"{NS['nhkg']}TemporalRelation" or local.startswith("TemporalRelation_"):
                    temporal_relations.add(parsed.subject)
    return {
        "total_triples": total_triples,
        "total_event_nodes": len(events),
        "total_token_nodes": len(words),
        "total_entity_mention_nodes": len(entities),
        "total_timex_nodes": len(timexes),
        "total_canonical_entity_nodes": len(canonical_entities),
        "total_temporal_relation_nodes": len(temporal_relations),
        "total_dbpedia_resource_nodes": len(dbpedia_resources),
    }


def compute_event_density_summary(payload: Optional[dict], events: List[dict]) -> dict:
    if payload:
        totals = payload.get("totals", {}) or {}
        histogram = payload.get("histogram", {}) or {}
        return {
            "events": int(totals.get("events", 0) or 0),
            "sentences": int(totals.get("sentences", 0) or 0),
            "mean_per_sentence": float(totals.get("mean_per_sentence", 0.0) or 0.0),
            "median_per_sentence": float(totals.get("median_per_sentence", 0.0) or 0.0),
            "p95_per_sentence": float(totals.get("p95_per_sentence", 0.0) or 0.0),
            "cap_hits": int(totals.get("cap_hits", 0) or 0),
            "cap_hit_rate": float(totals.get("cap_hit_rate", 0.0) or 0.0),
            "multi_event_warning_sentences": int(totals.get("multi_event_warning_sentences", 0) or 0),
            "multi_event_warning_rate": float(totals.get("multi_event_warning_rate", 0.0) or 0.0),
            "missing_sentence_text": int(totals.get("missing_sentence_text", 0) or 0),
            "histogram": histogram,
        }
    per_sentence = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        sent_key = f"{event.get('doc_id', 'batch_run')}::{event.get('sent_id', 0)}"
        per_sentence[sent_key] += 1
    counts = list(per_sentence.values())
    sorted_counts = sorted(counts)
    def simple_percentile(values: List[int], pct: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        position = pct * (len(values) - 1)
        lower = int(position)
        upper = min(len(values) - 1, lower + 1)
        fraction = position - lower
        return float(values[lower] + ((values[upper] - values[lower]) * fraction))

    histogram = Counter(counts)
    return {
        "events": sum(counts),
        "sentences": len(counts),
        "mean_per_sentence": round(statistics.mean(counts), 4) if counts else 0.0,
        "median_per_sentence": float(statistics.median(counts)) if counts else 0.0,
        "p95_per_sentence": round(simple_percentile(sorted_counts, 0.95), 4) if sorted_counts else 0.0,
        "cap_hits": 0,
        "cap_hit_rate": 0.0,
        "multi_event_warning_sentences": 0,
        "multi_event_warning_rate": 0.0,
        "histogram": {str(key): histogram[key] for key in sorted(histogram.keys())},
    }


def compute_validation_status(payload: Optional[dict]) -> dict:
    if not payload:
        return {
            "custom_errors": 0,
            "custom_warnings": 0,
            "parse_ok": None,
            "parse_reason": "",
            "shacl_requested": False,
            "shacl_ran": False,
            "shacl_available": False,
            "shacl_conforms": None,
            "overall_ok": None,
        }
    shacl = payload.get("shacl_validation", {}) or {}
    metadata = payload.get("metadata", {}) or {}
    custom = payload.get("custom_validation", {}) or {}
    parse_validation = payload.get("parse_validation", {}) or {}
    return {
        "custom_errors": len(custom.get("errors", []) or []),
        "custom_warnings": len(custom.get("warnings", []) or []),
        "parse_ok": parse_validation.get("parse_ok"),
        "parse_reason": parse_validation.get("reason", ""),
        "parse_error": parse_validation.get("error", ""),
        "parse_line_number": parse_validation.get("line_number"),
        "shacl_requested": bool(metadata.get("shacl_requested", False)),
        "shacl_ran": bool(shacl.get("available", False)),
        "shacl_available": bool(shacl.get("available", False)),
        "shacl_conforms": shacl.get("conforms"),
        "overall_ok": payload.get("overall_ok", payload.get("ok")),
        "shacl_warning": shacl.get("warning", ""),
        "shacl_reason": shacl.get("reason", ""),
    }


def make_quality_sections(report: dict) -> dict:
    ner = report["ner"]
    coref = report["coreference"]
    refinement = report["argument_refinement"]
    temporal = report["temporal"]
    validation = report["validation_status"]
    event_density = report["event_density"]
    extraction_precision = report["extraction_precision"]
    temporal_types = temporal.get("event_event_relations_by_type", {}) or {}
    return {
        "event_density": {
            "mean_per_sentence": event_density.get("mean_per_sentence", 0.0),
            "cap_hit_rate": event_density.get("cap_hit_rate", 0.0),
            "multi_event_warning_rate": event_density.get("multi_event_warning_rate", 0.0),
        },
        "ner_quality": {
            "backend": ner.get("backend_used", "unknown"),
            "rules_only_mode": ner.get("rules_only_mode", False),
            "type_distribution": ner.get("mentions_per_type", {}),
            "average_confidence": ner.get("average_confidence"),
        },
        "clustering_quality": {
            "average_mentions_per_cluster": coref.get("average_mentions_per_cluster", 0.0),
            "pronoun_candidates": coref.get("pronoun_candidates", 0),
            "pronoun_links_made": coref.get("pronoun_links_made", 0),
            "pronoun_unresolved": coref.get("pronoun_unresolved", 0),
            "uncertain_clusters": coref.get("uncertain_clusters", 0),
        },
        "refinement_quality": {
            "average_event_confidence": refinement.get("average_event_confidence"),
            "hard_review_rate": refinement.get("hard_review_rate", 0.0),
            "caution_rate": refinement.get("caution_rate", 0.0),
            "info_only_rate": refinement.get("info_only_rate", 0.0),
        },
        "temporal_relation_quality": {
            "relation_distribution": temporal_types,
            "weak_cotemporal_count": temporal.get("weak_cotemporal_count", 0),
            "strong_cotemporal_count": temporal.get("strong_cotemporal_count", 0),
            "average_confidence": temporal.get("event_event_average_confidence"),
        },
        "extraction_precision": extraction_precision,
        "validation_status": validation,
    }


def compute_event_audit_summary(payload: Optional[dict]) -> dict:
    if not payload:
        return {
            "extracted_event_count": 0,
            "graph_event_node_count": 0,
            "event_density_total": None,
            "mismatch_count": 0,
            "graph_unmapped_event_nodes": 0,
            "mismatch_reasons": {},
        }
    counts = payload.get("counts", {}) or {}
    return {
        "extracted_event_count": int(counts.get("extracted_event_count", 0) or 0),
        "graph_event_node_count": int(counts.get("graph_event_node_count", 0) or 0),
        "event_density_total": counts.get("event_density_total"),
        "mismatch_count": int(counts.get("mismatch_count", 0) or 0),
        "graph_unmapped_event_nodes": int(counts.get("graph_unmapped_event_nodes", 0) or 0),
        "mismatch_reasons": dict(sorted((payload.get("mismatch_reasons", {}) or {}).items())),
    }


def compute_extraction_precision_stats(path: Optional[Path]) -> dict:
    rows = load_jsonl(path)
    if not rows:
        return {
            "available": False,
            "source_path": str(path.resolve()) if path else "",
            "sentences_logged": 0,
            "candidates_retrieved": 0,
            "candidates_rejected_by_trigger_filter": 0,
            "candidates_rejected_as_copular_overgeneration": 0,
            "candidates_collapsed_as_duplicates": 0,
            "candidates_grouped_by_clause_center": 0,
            "candidates_dropped_as_same_center_duplicates": 0,
            "candidates_downranked_for_mismatch": 0,
            "candidates_downranked_as_helper_like": 0,
            "candidates_kept_as_lexical_predicate_centers": 0,
            "clause_compression_applied_count": 0,
            "final_candidates_considered": 0,
            "final_candidates_emitted": 0,
            "rejection_reasons_distribution": {},
            "clause_type_distribution": {},
            "accepted_candidate_family_distribution": {},
        }
    totals = Counter()
    reasons = Counter()
    clause_types = Counter()
    accepted_families = Counter()
    for row in rows:
        summary = row.get("summary", {}) or {}
        clause_profile = row.get("clause_profile", {}) or {}
        clause_types[str(clause_profile.get("clause_type", "unknown"))] += 1
        for key in (
            "candidates_retrieved",
            "candidates_rejected_by_trigger_filter",
            "candidates_rejected_as_copular_overgeneration",
            "candidates_collapsed_as_duplicates",
            "candidates_grouped_by_clause_center",
            "candidates_dropped_as_same_center_duplicates",
            "candidates_downranked_for_mismatch",
            "candidates_downranked_as_helper_like",
            "candidates_kept_as_lexical_predicate_centers",
            "clause_compression_applied_count",
            "final_candidates_considered",
            "final_candidates_emitted",
        ):
            totals[key] += int(summary.get(key, 0) or 0)
        for reason, value in (summary.get("rejection_reasons_distribution", {}) or {}).items():
            reasons[str(reason)] += int(value or 0)
        for family, value in (summary.get("accepted_candidate_family_distribution", {}) or {}).items():
            accepted_families[str(family)] += int(value or 0)
    return {
        "available": True,
        "source_path": str(path.resolve()) if path else "",
        "sentences_logged": len(rows),
        "candidates_retrieved": int(totals.get("candidates_retrieved", 0) or 0),
        "candidates_rejected_by_trigger_filter": int(totals.get("candidates_rejected_by_trigger_filter", 0) or 0),
        "candidates_rejected_as_copular_overgeneration": int(totals.get("candidates_rejected_as_copular_overgeneration", 0) or 0),
        "candidates_collapsed_as_duplicates": int(totals.get("candidates_collapsed_as_duplicates", 0) or 0),
        "candidates_grouped_by_clause_center": int(totals.get("candidates_grouped_by_clause_center", 0) or 0),
        "candidates_dropped_as_same_center_duplicates": int(totals.get("candidates_dropped_as_same_center_duplicates", 0) or 0),
        "candidates_downranked_for_mismatch": int(totals.get("candidates_downranked_for_mismatch", 0) or 0),
        "candidates_downranked_as_helper_like": int(totals.get("candidates_downranked_as_helper_like", 0) or 0),
        "candidates_kept_as_lexical_predicate_centers": int(totals.get("candidates_kept_as_lexical_predicate_centers", 0) or 0),
        "clause_compression_applied_count": int(totals.get("clause_compression_applied_count", 0) or 0),
        "final_candidates_considered": int(totals.get("final_candidates_considered", 0) or 0),
        "final_candidates_emitted": int(totals.get("final_candidates_emitted", 0) or 0),
        "rejection_reasons_distribution": dict(sorted(reasons.items())),
        "clause_type_distribution": dict(sorted(clause_types.items())),
        "accepted_candidate_family_distribution": dict(sorted(accepted_families.items())),
    }


def make_summary(report: dict) -> dict:
    ner = report["ner"]
    coref = report["coreference"]
    refinement = report["argument_refinement"]
    temporal = report["temporal"]
    validation = report["validation_status"]
    extraction_precision = report["extraction_precision"]
    temporal_type_count = len([key for key, value in (temporal.get("event_event_relations_by_type", {}) or {}).items() if value])
    return {
        "shacl_ran": validation.get("shacl_ran", False),
        "parse_ok": validation.get("parse_ok"),
        "ner_backend": ner.get("backend_used", "unknown"),
        "pronoun_coref_contributed": bool(coref.get("pronoun_links_made", 0)),
        "refinement_ready_for_auto_cleanup": bool(refinement.get("hard_review_rate", 1.0) <= 0.1),
        "temporal_relation_diversity_score": temporal_type_count,
        "overall_validation_ok": validation.get("overall_ok"),
        "precision_filter_active": bool(extraction_precision.get("available")),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    input_jsonl = Path(args.input_jsonl) if args.input_jsonl else None
    graph_path = Path(args.graph) if args.graph else None
    pos_payload = load_json(Path(args.pos_json)) if args.pos_json else None
    ner_payload = load_json(Path(args.ner_json)) if args.ner_json else None
    time_payload = load_json(Path(args.time_json)) if args.time_json else None
    coref_payload = load_json(Path(args.coref_json)) if args.coref_json else None
    refinement_payload = load_json(Path(args.refinement_json)) if args.refinement_json else None
    event_density_payload = load_json(Path(args.event_density_json)) if args.event_density_json else None
    event_audit_payload = load_json(Path(args.event_audit_json)) if args.event_audit_json else None
    validation_payload = load_json(Path(args.validation_json)) if args.validation_json else None
    candidate_decision_path = Path(args.candidate_decision_jsonl) if args.candidate_decision_jsonl else None
    dbpedia_payload = load_json(Path(args.dbpedia_json)) if args.dbpedia_json else None
    dbpedia_stats_payload = load_json(Path(args.dbpedia_stats)) if args.dbpedia_stats else None
    wordnet_stats = load_json(Path(args.wordnet_stats)) if args.wordnet_stats else None
    events = load_event_records(input_jsonl) if input_jsonl and input_jsonl.exists() else []
    coref_status = assess_cache_status(args.coref_json, coref_payload, required_keys=("documents",))
    refinement_status = assess_cache_status(args.refinement_json, refinement_payload, required_keys=("events",))

    report = {
        "syntax": compute_syntax_stats(pos_payload),
        "ner": compute_ner_stats(ner_payload, events),
        "coreference": compute_coref_stats(coref_payload, availability=coref_status),
        "argument_refinement": compute_refinement_stats(refinement_payload, availability=refinement_status),
        "temporal": compute_temporal_stats(time_payload),
        "event_density": compute_event_density_summary(event_density_payload, events),
        "event_count_audit": compute_event_audit_summary(event_audit_payload),
        "validation_status": compute_validation_status(validation_payload),
        "extraction_precision": compute_extraction_precision_stats(candidate_decision_path),
        "dbpedia": compute_dbpedia_stats(dbpedia_payload, dbpedia_stats_payload),
        "wordnet_gating": wordnet_stats or {},
        "graph": compute_graph_stats(graph_path),
    }
    report["coref"] = report["coreference"]
    report["refinement"] = report["argument_refinement"]
    report.update(make_quality_sections(report))
    report["summary"] = make_summary(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"[OK] wrote_report={out_path}")
    print(
        f"[OK] syntax_tokens={report['syntax']['token_count']} coref_clusters={report['coreference']['total_canonical_entity_clusters']} "
        f"refined_events={report['argument_refinement']['events']} timexes={report['temporal']['total_timexes']} "
        f"event_density_mean={report['event_density']['mean_per_sentence']:.4f}"
    )
    print(
        f"[OK] graph_events={report['graph']['total_event_nodes']} graph_tokens={report['graph']['total_token_nodes']} "
        f"graph_entities={report['graph']['total_entity_mention_nodes']} graph_canonical_entities={report['graph']['total_canonical_entity_nodes']} "
        f"graph_timexes={report['graph']['total_timex_nodes']} graph_temporal_relations={report['graph']['total_temporal_relation_nodes']} "
        f"shacl_ran={report['summary']['shacl_ran']}"
    )


if __name__ == "__main__":
    main()
