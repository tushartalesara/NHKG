#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Materialize temporal expressions and event-time links into RDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from .enrichment_common import (
        NS,
        clean_uri,
        event_uri,
        parse_nquad_line,
        parse_sentence_key_from_uri,
        timex_uri,
        to_nquad_decimal,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
    )
    from .internal_quality_common import temporal_relation_uri
except ImportError:  # pragma: no cover
    from enrichment_common import (
        NS,
        clean_uri,
        event_uri,
        parse_nquad_line,
        parse_sentence_key_from_uri,
        timex_uri,
        to_nquad_decimal,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
    )
    from internal_quality_common import temporal_relation_uri


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append temporal expressions to an NHKG RDF graph.")
    parser.add_argument("--input", required=True, help="Input graph, optionally already NER-augmented")
    parser.add_argument("--time-json", required=True, help="Temporal cache from align/temporal_enrich.py")
    parser.add_argument("--out", required=True, help="Output N-Quads path")
    return parser


def load_graph_context(path: Path) -> Tuple[Dict[str, str], Set[str], Set[str], Set[str]]:
    sentence_map: Dict[str, str] = {}
    event_nodes: Set[str] = set()
    graphs: Set[str] = set()
    existing_timexes: Set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            graphs.add(parsed.graph)
            if parsed.subject.rsplit("/", 1)[-1].startswith("Event_"):
                event_nodes.add(parsed.subject)
            if parsed.predicate == f"{NS['rdf']}type" and parsed.obj.kind == "uri":
                if parsed.obj.value == f"{NS['nif']}Sentence":
                    sent_key = parse_sentence_key_from_uri(parsed.subject)
                    if sent_key:
                        sentence_map[sent_key] = parsed.subject
                elif parsed.obj.value == f"{NS['schema']}Event":
                    event_nodes.add(parsed.subject)
                elif parsed.obj.value == f"{NS['nhkg']}Timex":
                    existing_timexes.add(parsed.subject)
    return sentence_map, event_nodes, graphs, existing_timexes


def load_time_cache(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Temporal cache must be a JSON object")

    out: Dict[str, dict] = {}
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if isinstance(item, dict):
                out[f"{doc_id}::{sent_id}"] = item
    return out


def link_records_from_sentence(item: dict) -> List[dict]:
    links = item.get("event_time_links", [])
    if isinstance(links, list) and links:
        return [link for link in links if isinstance(link, dict)]

    derived_links = []
    for timex in item.get("timexes", []) if isinstance(item.get("timexes", []), list) else []:
        if not isinstance(timex, dict):
            continue
        for link in timex.get("linked_events", []) if isinstance(timex.get("linked_events", []), list) else []:
            if not isinstance(link, dict):
                continue
            row = dict(link)
            row.setdefault("timex_id", timex.get("timex_id"))
            derived_links.append(row)
    return derived_links


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    time_path = Path(args.time_json)
    if not input_path.exists():
        raise SystemExit(f"Input graph not found: {input_path}")
    if not time_path.exists():
        raise SystemExit(f"Temporal cache not found: {time_path}")

    sentence_map, event_nodes, graphs, existing_timexes = load_graph_context(input_path)
    time_data = load_time_cache(time_path)
    if not sentence_map:
        raise SystemExit("No sentence nodes found in the input graph.")

    graph_uri = sorted(graphs)[0] if graphs else "http://ns.nhkg.org/graph/gold"
    with input_path.open("r", encoding="utf-8") as handle:
        quads = [line.rstrip("\n") for line in handle]
    seen_quads = set(quads)

    emitted_timexes = 0
    event_time_edges = 0
    temporal_relation_nodes = 0
    missing_events = 0

    for sent_key, sent_uri in sorted(sentence_map.items()):
        sent_payload = time_data.get(sent_key)
        if not isinstance(sent_payload, dict):
            continue

        timex_uri_map: Dict[str, str] = {}
        for timex in sent_payload.get("timexes", []) if isinstance(sent_payload.get("timexes", []), list) else []:
            if not isinstance(timex, dict):
                continue
            start = timex.get("start")
            end = timex.get("end")
            if not isinstance(start, int):
                start = None
            if not isinstance(end, int):
                end = None
            timex_id = str(timex.get("timex_id", "")).strip()
            node_uri = timex_uri(sent_key, start, end, timex_id=timex_id)
            timex_uri_map[timex_id] = node_uri
            if node_uri not in existing_timexes:
                base_quads = [
                    to_nquad_uri(node_uri, f"{NS['rdf']}type", f"{NS['nhkg']}Timex", graph_uri),
                    to_nquad_uri(node_uri, f"{NS['rdf']}type", f"{NS['nif']}Phrase", graph_uri),
                    to_nquad_literal(node_uri, f"{NS['nif']}anchorOf", str(timex.get("text", "")), graph_uri, lang="hi"),
                    to_nquad_uri(node_uri, f"{NS['nif']}referenceContext", sent_uri, graph_uri),
                    to_nquad_uri(node_uri, f"{NS['prov']}wasDerivedFrom", sent_uri, graph_uri),
                    to_nquad_literal(
                        node_uri,
                        f"{NS['nhkg']}timexType",
                        str(timex.get("type", "DATE")).upper(),
                        graph_uri,
                        lang="en",
                    ),
                    to_nquad_literal(
                        node_uri,
                        f"{NS['nhkg']}resolutionStatus",
                        str(timex.get("resolution_status", "unresolved")),
                        graph_uri,
                        lang="en",
                    ),
                    to_nquad_literal(
                        node_uri,
                        f"{NS['nhkg']}temporalEngine",
                        str(timex.get("engine", "rules")),
                        graph_uri,
                        lang="en",
                    ),
                ]
                if isinstance(start, int) and isinstance(end, int):
                    base_quads.append(to_nquad_int(node_uri, f"{NS['nif']}beginIndex", start, graph_uri))
                    base_quads.append(to_nquad_int(node_uri, f"{NS['nif']}endIndex", end, graph_uri))
                value = timex.get("value")
                if value is not None:
                    base_quads.append(to_nquad_literal(node_uri, f"{NS['nhkg']}timexValue", value, graph_uri))
                mod = timex.get("mod")
                if mod:
                    base_quads.append(to_nquad_literal(node_uri, f"{NS['nhkg']}timexMod", mod, graph_uri, lang="en"))
                confidence = timex.get("confidence")
                try:
                    if confidence is not None:
                        base_quads.append(to_nquad_decimal(node_uri, f"{NS['nhkg']}confidence", float(confidence), graph_uri))
                except (TypeError, ValueError):
                    pass

                for quad in base_quads:
                    if quad not in seen_quads:
                        quads.append(quad)
                        seen_quads.add(quad)
                existing_timexes.add(node_uri)
                emitted_timexes += 1

        for link in link_records_from_sentence(sent_payload):
            event_id_value = str(link.get("event_id", "")).strip()
            doc_id = str(link.get("doc_id", sent_key.split("::", 1)[0]))
            sent_id = str(link.get("sent_id", sent_key.split("::", 1)[1] if "::" in sent_key else "0"))
            if not event_id_value:
                continue
            event_node = event_uri(doc_id, sent_id, event_id_value)
            if event_node not in event_nodes:
                missing_events += 1
                continue

            timex_id_value = str(link.get("timex_id", "")).strip()
            node_uri = timex_uri_map.get(timex_id_value)
            if not node_uri:
                node_uri = timex_uri(sent_key, None, None, timex_id=timex_id_value)

            predicate_name = str(link.get("predicate", "hasTimeExpression"))
            predicate_uri = f"{NS['nhkg']}{clean_uri(predicate_name)}"
            edge_quad = to_nquad_uri(event_node, predicate_uri, node_uri, graph_uri)
            if edge_quad not in seen_quads:
                quads.append(edge_quad)
                seen_quads.add(edge_quad)
                event_time_edges += 1

            strategy = str(link.get("strategy", "")).strip()
            if strategy:
                strategy_quad = to_nquad_literal(
                    event_node,
                    f"{NS['nhkg']}timeLinkStrategy",
                    strategy,
                    graph_uri,
                    lang="en",
                )
                if strategy_quad not in seen_quads:
                    quads.append(strategy_quad)
                    seen_quads.add(strategy_quad)
            confidence = link.get("confidence")
            try:
                if confidence is not None:
                    confidence_quad = to_nquad_decimal(event_node, f"{NS['nhkg']}temporalConfidence", float(confidence), graph_uri)
                    if confidence_quad not in seen_quads:
                        quads.append(confidence_quad)
                        seen_quads.add(confidence_quad)
            except (TypeError, ValueError):
                pass
            evidence = str(link.get("evidence", "")).strip()
            if evidence:
                evidence_quad = to_nquad_literal(event_node, f"{NS['nhkg']}temporalEvidence", evidence, graph_uri, lang="hi")
                if evidence_quad not in seen_quads:
                    quads.append(evidence_quad)
                    seen_quads.add(evidence_quad)

    for sent_key, sent_payload in sorted(time_data.items()):
        if not isinstance(sent_payload, dict):
            continue
        doc_id, sent_id = sent_key.split("::", 1) if "::" in sent_key else (sent_key, "0")
        for relation in sent_payload.get("event_event_relations", []) if isinstance(sent_payload.get("event_event_relations", []), list) else []:
            if not isinstance(relation, dict):
                continue
            source_event = event_uri(doc_id, sent_id, str(relation.get("source_event_id", "")))
            target_event = event_uri(doc_id, sent_id, str(relation.get("target_event_id", "")))
            if source_event not in event_nodes or target_event not in event_nodes:
                continue
            relation_node = temporal_relation_uri(doc_id, str(relation.get("relation_id", "")) or f"{relation.get('source_event_id', '')}_{relation.get('target_event_id', '')}")
            relation_quads = [
                to_nquad_uri(relation_node, f"{NS['rdf']}type", f"{NS['nhkg']}TemporalRelation", graph_uri),
                to_nquad_uri(relation_node, f"{NS['nhkg']}sourceEvent", source_event, graph_uri),
                to_nquad_uri(relation_node, f"{NS['nhkg']}targetEvent", target_event, graph_uri),
                to_nquad_literal(relation_node, f"{NS['nhkg']}temporalRelation", str(relation.get("relation", "")), graph_uri, lang="en"),
            ]
            strategy = str(relation.get("strategy", "")).strip()
            if strategy:
                relation_quads.append(to_nquad_literal(relation_node, f"{NS['nhkg']}timeLinkStrategy", strategy, graph_uri, lang="en"))
            evidence = str(relation.get("evidence", "")).strip()
            if evidence:
                relation_quads.append(to_nquad_literal(relation_node, f"{NS['nhkg']}temporalEvidence", evidence, graph_uri, lang="hi"))
            evidence_strength = str(relation.get("evidence_strength", "")).strip()
            if evidence_strength:
                relation_quads.append(to_nquad_literal(relation_node, f"{NS['nhkg']}temporalEvidenceStrength", evidence_strength, graph_uri, lang="en"))
            try:
                confidence = relation.get("confidence")
                if confidence is not None:
                    relation_quads.append(to_nquad_decimal(relation_node, f"{NS['nhkg']}temporalConfidence", float(confidence), graph_uri))
            except (TypeError, ValueError):
                pass
            for quad in relation_quads:
                if quad not in seen_quads:
                    quads.append(quad)
                    seen_quads.add(quad)
            temporal_relation_nodes += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for quad in quads:
            handle.write(quad + "\n")

    print(f"[OK] wrote_graph={out_path}")
    print(f"[OK] emitted_timexes={emitted_timexes} event_time_edges={event_time_edges} temporal_relation_nodes={temporal_relation_nodes} missing_events={missing_events}")


if __name__ == "__main__":
    main()
