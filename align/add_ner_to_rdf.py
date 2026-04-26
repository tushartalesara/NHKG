#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Materialize sentence-scoped NER mentions into RDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from .enrichment_common import (
        NS,
        entity_mention_uri,
        parse_nquad_line,
        parse_sentence_key_from_uri,
        spans_overlap,
        to_nquad_decimal,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
    )
except ImportError:  # pragma: no cover
    from enrichment_common import (
        NS,
        entity_mention_uri,
        parse_nquad_line,
        parse_sentence_key_from_uri,
        spans_overlap,
        to_nquad_decimal,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
    )


SCHEMA_TYPE_MAP = {
    "PER": f"{NS['schema']}Person",
    "LOC": f"{NS['schema']}Place",
    "ORG": f"{NS['schema']}Organization",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append NER mentions to an NHKG N-Quads graph.")
    parser.add_argument("--input", required=True, help="Input graph, optionally already POS-augmented")
    parser.add_argument("--ner-json", required=True, help="NER cache from align/run_hindi_ner.py")
    parser.add_argument("--out", required=True, help="Output N-Quads path")
    return parser


def overlap_ratio(left: Tuple[int, int], right: Tuple[int, int]) -> float:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return 0.0
    shorter = min(left[1] - left[0], right[1] - right[0])
    return overlap / max(1, shorter)


def load_graph_context(path: Path) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, Tuple[int, int]]]], Set[str], Set[str]]:
    sentence_map: Dict[str, str] = {}
    mention_spans: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {}
    graphs: Set[str] = set()
    existing_entities: Set[str] = set()

    subject_types: Dict[str, Set[str]] = {}
    begin_index: Dict[str, int] = {}
    end_index: Dict[str, int] = {}
    sentence_ref: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            graphs.add(parsed.graph)
            if parsed.predicate == f"{NS['rdf']}type" and parsed.obj.kind == "uri":
                subject_types.setdefault(parsed.subject, set()).add(parsed.obj.value)
                if parsed.obj.value == f"{NS['nhkg']}EntityMention":
                    existing_entities.add(parsed.subject)
                if parsed.obj.value == f"{NS['nif']}Sentence":
                    sent_key = parse_sentence_key_from_uri(parsed.subject)
                    if sent_key:
                        sentence_map[sent_key] = parsed.subject
            elif parsed.predicate == f"{NS['nif']}beginIndex" and parsed.obj.kind == "literal":
                try:
                    begin_index[parsed.subject] = int(parsed.obj.value)
                except (TypeError, ValueError):
                    pass
            elif parsed.predicate == f"{NS['nif']}endIndex" and parsed.obj.kind == "literal":
                try:
                    end_index[parsed.subject] = int(parsed.obj.value)
                except (TypeError, ValueError):
                    pass
            elif parsed.predicate in {f"{NS['prov']}wasDerivedFrom", f"{NS['nif']}referenceContext"} and parsed.obj.kind == "uri":
                sentence_ref[parsed.subject] = parsed.obj.value

    for subject, start in begin_index.items():
        end = end_index.get(subject)
        sent_uri = sentence_ref.get(subject)
        if end is None or sent_uri is None:
            continue
        sent_key = parse_sentence_key_from_uri(sent_uri)
        if not sent_key:
            continue
        local = subject.rsplit("/", 1)[-1]
        if not local.startswith("Mention_"):
            continue
        mention_spans.setdefault(sent_key, []).append((subject, (start, end)))

    return sentence_map, mention_spans, graphs, existing_entities


def load_ner_cache(path: Path) -> Dict[str, List[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("NER cache must be a JSON object")

    out: Dict[str, List[dict]] = {}
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            entities = item.get("entities", [])
            if isinstance(entities, list):
                out[f"{doc_id}::{sent_id}"] = entities
    return out


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    ner_path = Path(args.ner_json)
    if not input_path.exists():
        raise SystemExit(f"Input graph not found: {input_path}")
    if not ner_path.exists():
        raise SystemExit(f"NER cache not found: {ner_path}")

    sentence_map, mention_spans, graphs, existing_entities = load_graph_context(input_path)
    ner_data = load_ner_cache(ner_path)
    if not sentence_map:
        raise SystemExit("No sentence nodes found in the input graph.")

    graph_uri = sorted(graphs)[0] if graphs else "http://ns.nhkg.org/graph/gold"
    with input_path.open("r", encoding="utf-8") as handle:
        quads = [line.rstrip("\n") for line in handle]
    seen_quads = set(quads)

    emitted_mentions = 0
    alignment_links = 0

    for sent_key, sent_uri in sorted(sentence_map.items()):
        for entity in ner_data.get(sent_key, []):
            if not isinstance(entity, dict):
                continue
            try:
                start = int(entity.get("start"))
                end = int(entity.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            label = str(entity.get("label", "MISC") or "MISC").strip().upper()
            text = str(entity.get("text", "")).strip()
            if not text:
                continue

            node_uri = entity_mention_uri(sent_key, start, end, label)
            if node_uri not in existing_entities:
                base_quads = [
                    to_nquad_uri(node_uri, f"{NS['rdf']}type", f"{NS['nhkg']}EntityMention", graph_uri),
                    to_nquad_uri(node_uri, f"{NS['rdf']}type", f"{NS['nif']}Phrase", graph_uri),
                    to_nquad_literal(node_uri, f"{NS['nif']}anchorOf", text, graph_uri, lang="hi"),
                    to_nquad_int(node_uri, f"{NS['nif']}beginIndex", start, graph_uri),
                    to_nquad_int(node_uri, f"{NS['nif']}endIndex", end, graph_uri),
                    to_nquad_uri(node_uri, f"{NS['nif']}referenceContext", sent_uri, graph_uri),
                    to_nquad_uri(node_uri, f"{NS['prov']}wasDerivedFrom", sent_uri, graph_uri),
                    to_nquad_literal(node_uri, f"{NS['nhkg']}entityType", label, graph_uri, lang="en"),
                    to_nquad_literal(
                        node_uri,
                        f"{NS['nhkg']}detectedBy",
                        str(entity.get("engine", "")) or "unknown",
                        graph_uri,
                        lang="en",
                    ),
                ]
                raw_label = str(entity.get("raw_label", "")).strip()
                if raw_label:
                    base_quads.append(
                        to_nquad_literal(node_uri, f"{NS['nhkg']}rawEntityLabel", raw_label, graph_uri, lang="en")
                    )
                confidence = entity.get("confidence", entity.get("score"))
                try:
                    if confidence is not None:
                        base_quads.append(
                            to_nquad_decimal(node_uri, f"{NS['nhkg']}entityConfidence", float(confidence), graph_uri)
                        )
                except (TypeError, ValueError):
                    pass

                broader_type = SCHEMA_TYPE_MAP.get(label)
                if broader_type:
                    base_quads.append(to_nquad_uri(node_uri, f"{NS['rdf']}type", broader_type, graph_uri))

                for quad in base_quads:
                    if quad not in seen_quads:
                        quads.append(quad)
                        seen_quads.add(quad)
                existing_entities.add(node_uri)
                emitted_mentions += 1

            entity_span = (start, end)
            for mention_uri_value, mention_span in mention_spans.get(sent_key, []):
                if not spans_overlap(entity_span, mention_span):
                    continue
                if overlap_ratio(entity_span, mention_span) < 0.5:
                    continue
                align_quad = to_nquad_uri(
                    node_uri,
                    f"{NS['nhkg']}alignsWithMention",
                    mention_uri_value,
                    graph_uri,
                )
                if align_quad not in seen_quads:
                    quads.append(align_quad)
                    seen_quads.add(align_quad)
                    alignment_links += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for quad in quads:
            handle.write(quad + "\n")

    print(f"[OK] wrote_graph={out_path}")
    print(f"[OK] emitted_entity_mentions={emitted_mentions} alignment_links={alignment_links}")


if __name__ == "__main__":
    main()
