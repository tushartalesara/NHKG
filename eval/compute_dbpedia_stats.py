#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute coverage and graph-impact statistics for DBpedia entity linking."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

try:
    from common import load_event_records
except ImportError:  # pragma: no cover
    from eval.common import load_event_records

try:
    from align.enrichment_common import NS, parse_nquad_line
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.enrichment_common import NS, parse_nquad_line

try:
    from fusion.dbpedia_common import DBPEDIA_RESOURCE_PREFIX, load_yaml_or_default
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from fusion.dbpedia_common import DBPEDIA_RESOURCE_PREFIX, load_yaml_or_default


DEFAULT_NER_MAP = Path("lexicons/ner2dbo_class_map.yaml")
DEFAULT_ROLE_MAP = Path("lexicons/role2dbo_type_priors.yaml")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute DBpedia linking coverage and graph impact statistics.")
    parser.add_argument("--dbpedia-json", required=True, help="Link cache from align/link_dbpedia.py")
    parser.add_argument("--graph", default="", help="Optional final RDF graph for graph impact counts")
    parser.add_argument("--wordnet-stats", default="", help="Optional WordNet stats JSON to report DBpedia gating effect")
    parser.add_argument("--input-jsonl", default="", help="Optional extraction JSONL for sentence/event counts")
    parser.add_argument("--ner-type-map", default=str(DEFAULT_NER_MAP))
    parser.add_argument("--role-type-map", default=str(DEFAULT_ROLE_MAP))
    parser.add_argument("--out", required=True, help="Output JSON stats path")
    parser.add_argument("--csv-out", default="", help="Optional compact CSV summary")
    return parser


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


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


def normalize_types(values: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        out.add(text)
        out.add(text.rsplit("/", 1)[-1])
    return out


def collect_rows(payload: dict) -> tuple[List[dict], List[dict]]:
    links: List[dict] = []
    mentions: List[dict] = []
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            sentence_key = f"{doc_id}::{sent_id}"
            for link in item.get("links", []):
                if isinstance(link, dict):
                    row = dict(link)
                    row.setdefault("sent_key", sentence_key)
                    links.append(row)
            for mention in item.get("mentions", []):
                if isinstance(mention, dict):
                    row = dict(mention)
                    row.setdefault("sent_key", sentence_key)
                    mentions.append(row)
    return links, mentions


def compute_linking_stats(links: List[dict], mentions: List[dict], meta: dict, ner_type_map: Dict[str, List[str]], role_type_map: Dict[str, List[str]]) -> dict:
    entity_counter = Counter()
    source_counter = Counter()
    matched_via_counter = Counter()
    no_link_reasons = Counter()
    confidence_values: List[float] = []
    candidate_sizes: List[int] = []
    ner_type_consistent = 0
    ner_type_conflicts = 0
    role_type_warnings = 0

    for row in links:
        entity_counter[str(row.get("ner_label", "UNLABELED")) or "UNLABELED"] += 1
        source_counter[str(row.get("mention_source", "unknown")) or "unknown"] += 1
        matched_via_counter[str(row.get("matched_via", "unknown")) or "unknown"] += 1
        try:
            confidence_values.append(float(row.get("confidence", row.get("score", 0.0))))
        except (TypeError, ValueError):
            pass
        try:
            candidate_sizes.append(int(row.get("candidate_count", len(row.get("top_candidates", [])))))
        except (TypeError, ValueError):
            pass

        predicted_types = normalize_types(row.get("predicted_dbo_types", []))
        ner_label = str(row.get("ner_label", "")).upper()
        if ner_label and ner_label in ner_type_map:
            expected = normalize_types(ner_type_map.get(ner_label, []))
            if predicted_types and expected:
                if any(item in predicted_types for item in expected):
                    ner_type_consistent += 1
                else:
                    ner_type_conflicts += 1

        role = str(row.get("role", "")).strip().upper().replace(" ", "_")
        if role and role in role_type_map:
            role_expected = normalize_types(role_type_map.get(role, []))
            if predicted_types and role_expected and not any(item in predicted_types for item in role_expected):
                role_type_warnings += 1

    raw_candidate_mentions = int(meta.get("total_raw_candidate_mentions", len(mentions)))
    filtered_candidate_mentions = int(meta.get("total_filtered_candidate_mentions", len([row for row in mentions if row.get("status") != "skipped"])))
    likely_linkable_mentions = int(meta.get("likely_linkable_mentions", len([row for row in mentions if row.get("likely_linkable")])))
    linked_mentions = len(links)

    for row in mentions:
        if row.get("status") != "linked":
            no_link_reasons[str(row.get("no_link_reason", "unknown")) or "unknown"] += 1

    return {
        "raw_candidate_mentions": raw_candidate_mentions,
        "filtered_candidate_mentions": filtered_candidate_mentions,
        "likely_linkable_mentions": likely_linkable_mentions,
        "linked_mentions": linked_mentions,
        "coverage_on_raw_candidates": round(linked_mentions / raw_candidate_mentions, 4) if raw_candidate_mentions else 0.0,
        "coverage_on_filtered_candidates": round(linked_mentions / filtered_candidate_mentions, 4) if filtered_candidate_mentions else 0.0,
        "coverage_on_likely_linkable_mentions": round(linked_mentions / likely_linkable_mentions, 4) if likely_linkable_mentions else 0.0,
        "candidate_recall_at_k": round(len([row for row in mentions if row.get("top_candidates")]) / likely_linkable_mentions, 4) if likely_linkable_mentions else 0.0,
        "links_per_entity_type": dict(sorted(entity_counter.items())),
        "links_per_source": dict(sorted(source_counter.items())),
        "match_via_counts": dict(sorted(matched_via_counter.items())),
        "no_link_reason_counts": dict(sorted(no_link_reasons.items())),
        "filter_counters": dict(sorted((meta.get("filter_counters") or {}).items())),
        "confidence": {
            "count": len(confidence_values),
            "avg": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
            "min": round(min(confidence_values), 4) if confidence_values else None,
            "max": round(max(confidence_values), 4) if confidence_values else None,
        },
        "candidate_set_size": {
            "avg": round(sum(candidate_sizes) / len(candidate_sizes), 4) if candidate_sizes else 0.0,
            "max": max(candidate_sizes) if candidate_sizes else 0,
        },
        "type_consistency": {
            "ner_type_consistent": ner_type_consistent,
            "ner_type_conflicts": ner_type_conflicts,
            "role_type_warnings": role_type_warnings,
        },
    }


def compute_graph_impact(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {
            "dbpedia_resource_nodes": 0,
            "mention_to_dbpedia_links": 0,
            "sameas_links": 0,
            "labels_added": 0,
            "types_added": 0,
            "abstracts_added": 0,
            "fused_fact_predicates": {},
        }

    resources: Set[str] = set()
    mention_links = 0
    sameas_links = 0
    labels_added = 0
    types_added = 0
    abstracts_added = 0
    predicate_counter = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            subject_is_dbpedia = parsed.subject.startswith(DBPEDIA_RESOURCE_PREFIX)
            object_is_dbpedia = parsed.obj.kind == "uri" and str(parsed.obj.value).startswith(DBPEDIA_RESOURCE_PREFIX)
            if subject_is_dbpedia:
                resources.add(parsed.subject)
            if object_is_dbpedia:
                resources.add(str(parsed.obj.value))

            if parsed.predicate == f"{NS['nhkg']}refersToDbpediaResource" and object_is_dbpedia:
                mention_links += 1
            elif parsed.predicate == f"{NS['owl']}sameAs" and (subject_is_dbpedia or object_is_dbpedia):
                sameas_links += 1
            elif parsed.predicate == f"{NS['rdfs']}label" and subject_is_dbpedia:
                labels_added += 1
            elif subject_is_dbpedia and parsed.predicate == f"{NS['rdf']}type" and parsed.obj.kind == "uri" and str(parsed.obj.value).startswith("http://dbpedia.org/ontology/"):
                types_added += 1
            elif subject_is_dbpedia and parsed.predicate == "http://dbpedia.org/ontology/abstract":
                abstracts_added += 1

            if subject_is_dbpedia:
                predicate_counter[parsed.predicate] += 1

    return {
        "dbpedia_resource_nodes": len(resources),
        "mention_to_dbpedia_links": mention_links,
        "sameas_links": sameas_links,
        "labels_added": labels_added,
        "types_added": types_added,
        "abstracts_added": abstracts_added,
        "fused_fact_predicates": dict(sorted(predicate_counter.items())),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_arg_parser().parse_args()

    cache_path = Path(args.dbpedia_json)
    if not cache_path.exists():
        raise SystemExit(f"DBpedia link cache not found: {cache_path}")

    cache_payload = load_json(cache_path) or {}
    links, mentions = collect_rows(cache_payload)
    meta = cache_payload.get("meta", {}) if isinstance(cache_payload.get("meta"), dict) else {}
    ner_type_map = load_type_map(Path(args.ner_type_map))
    role_type_map = load_type_map(Path(args.role_type_map))
    graph_path = Path(args.graph) if args.graph else None
    wordnet_stats = load_json(Path(args.wordnet_stats)) if args.wordnet_stats else None
    events = load_event_records(Path(args.input_jsonl)) if args.input_jsonl else []

    report = {
        "linking": compute_linking_stats(links, mentions, meta, ner_type_map, role_type_map),
        "graph_impact": compute_graph_impact(graph_path),
        "wordnet_effect": wordnet_stats or {},
        "context": {
            "event_count": len(events),
            "dbpedia_cache": str(cache_path.resolve()),
            "graph": str(graph_path.resolve()) if graph_path and graph_path.exists() else "",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if args.csv_out:
        rows = [
            {"metric": "raw_candidate_mentions", "value": report["linking"]["raw_candidate_mentions"]},
            {"metric": "filtered_candidate_mentions", "value": report["linking"]["filtered_candidate_mentions"]},
            {"metric": "likely_linkable_mentions", "value": report["linking"]["likely_linkable_mentions"]},
            {"metric": "linked_mentions", "value": report["linking"]["linked_mentions"]},
            {"metric": "coverage_on_likely_linkable_mentions", "value": report["linking"]["coverage_on_likely_linkable_mentions"]},
            {"metric": "candidate_recall_at_k", "value": report["linking"]["candidate_recall_at_k"]},
            {"metric": "dbpedia_resource_nodes", "value": report["graph_impact"]["dbpedia_resource_nodes"]},
        ]
        write_csv(Path(args.csv_out), rows)

    print(f"[OK] wrote_dbpedia_stats={out_path}")
    print(
        f"[OK] raw_candidates={report['linking']['raw_candidate_mentions']} "
        f"filtered_candidates={report['linking']['filtered_candidate_mentions']} "
        f"likely_linkable={report['linking']['likely_linkable_mentions']} "
        f"linked_mentions={report['linking']['linked_mentions']}"
    )
    print(
        f"[OK] dbpedia_resources={report['graph_impact']['dbpedia_resource_nodes']} "
        f"mention_links={report['graph_impact']['mention_to_dbpedia_links']} "
        f"sameas={report['graph_impact']['sameas_links']}"
    )


if __name__ == "__main__":
    main()
