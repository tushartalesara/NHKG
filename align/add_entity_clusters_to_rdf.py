#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Materialize canonical entity clusters into RDF without collapsing mention nodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set, Tuple

try:
    from .enrichment_common import (
        NS,
        parse_nquad_line,
        to_nquad_decimal,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
    )
    from .internal_quality_common import load_json, type_to_schema_uri
except ImportError:  # pragma: no cover
    from enrichment_common import (
        NS,
        parse_nquad_line,
        to_nquad_decimal,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
    )
    from internal_quality_common import load_json, type_to_schema_uri


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append canonical entity cluster nodes to an NHKG graph.")
    parser.add_argument("--input", required=True, help="Input N-Quads graph")
    parser.add_argument("--coref-json", required=True, help="Entity cluster cache from align/cluster_entities.py")
    parser.add_argument("--out", required=True, help="Output N-Quads path")
    return parser


def load_graph_context(path: Path) -> Tuple[Set[str], str]:
    subjects: Set[str] = set()
    graphs: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            subjects.add(parsed.subject)
            graphs.add(parsed.graph)
    return subjects, sorted(graphs)[0] if graphs else "http://ns.nhkg.org/graph/gold"


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    coref_path = Path(args.coref_json)
    payload = load_json(coref_path, {})
    documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
    subjects, graph_uri = load_graph_context(input_path)

    with input_path.open("r", encoding="utf-8") as handle:
        quads = [line.rstrip("\n") for line in handle]
    seen_quads = set(quads)

    emitted_clusters = 0
    emitted_links = 0

    for _, item in (documents or {}).items():
        if not isinstance(item, dict):
            continue
        for cluster in item.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            cluster_node = str(cluster.get("cluster_uri", "")).strip()
            if not cluster_node:
                continue
            if cluster_node not in subjects:
                base_quads = [
                    to_nquad_uri(cluster_node, f"{NS['rdf']}type", f"{NS['nhkg']}CanonicalEntity", graph_uri),
                    to_nquad_literal(cluster_node, f"{NS['rdfs']}label", str(cluster.get("canonical_text", "")), graph_uri, lang="hi"),
                    to_nquad_literal(cluster_node, f"{NS['nhkg']}canonicalNormalizedText", str(cluster.get("canonical_normalized_text", "")), graph_uri, lang="hi"),
                    to_nquad_literal(cluster_node, f"{NS['nhkg']}entityType", str(cluster.get("predicted_entity_type", "MISC")), graph_uri, lang="en"),
                    to_nquad_literal(cluster_node, f"{NS['nhkg']}clusterEngine", "rule_coref", graph_uri, lang="en"),
                    to_nquad_int(cluster_node, f"{NS['nhkg']}clusterMentionCount", len(cluster.get("mentions", [])), graph_uri),
                ]
                try:
                    base_quads.append(to_nquad_decimal(cluster_node, f"{NS['nhkg']}clusterConfidence", float(cluster.get("confidence", 0.0) or 0.0), graph_uri))
                except (TypeError, ValueError):
                    pass
                schema_uri = type_to_schema_uri(str(cluster.get("predicted_entity_type", "")))
                if schema_uri:
                    base_quads.append(to_nquad_uri(cluster_node, f"{NS['rdf']}type", schema_uri, graph_uri))
                for quad in base_quads:
                    if quad not in seen_quads:
                        quads.append(quad)
                        seen_quads.add(quad)
                emitted_clusters += 1
                subjects.add(cluster_node)

            for mention in cluster.get("mentions", []):
                if not isinstance(mention, dict):
                    continue
                mention_uri_value = str(mention.get("mention_uri", "")).strip()
                if not mention_uri_value or mention_uri_value not in subjects:
                    continue
                align_quad = to_nquad_uri(mention_uri_value, f"{NS['nhkg']}refersToCanonicalEntity", cluster_node, graph_uri)
                if align_quad not in seen_quads:
                    quads.append(align_quad)
                    seen_quads.add(align_quad)
                    emitted_links += 1
                role = str(mention.get("role_in_cluster", "")).strip()
                if role:
                    role_quad = to_nquad_literal(mention_uri_value, f"{NS['nhkg']}clusterRole", role, graph_uri, lang="en")
                    if role_quad not in seen_quads:
                        quads.append(role_quad)
                        seen_quads.add(role_quad)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for quad in quads:
            handle.write(quad + "\n")

    print(f"[OK] wrote_graph={out_path}")
    print(f"[OK] emitted_canonical_entities={emitted_clusters} emitted_cluster_links={emitted_links}")


if __name__ == "__main__":
    main()
