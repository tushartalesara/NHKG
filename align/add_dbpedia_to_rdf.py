#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Materialize DBpedia links and compact resource annotations into the NHKG RDF graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set

try:
    from .enrichment_common import NS, parse_nquad_line, to_nquad_decimal, to_nquad_literal, to_nquad_uri
except ImportError:  # pragma: no cover
    from enrichment_common import NS, parse_nquad_line, to_nquad_decimal, to_nquad_literal, to_nquad_uri

try:
    from fusion.dbpedia_common import connect_db, db_fetch_resource
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from fusion.dbpedia_common import connect_db, db_fetch_resource


DBPEDIA_GRAPH = "http://ns.nhkg.org/graph/dbpedia"
REFERS_TO_DBPEDIA = f"{NS['nhkg']}refersToDbpediaResource"
LINKED_EXTERNAL_ENTITY = f"{NS['nhkg']}LinkedExternalEntity"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append DBpedia links/resources to an NHKG N-Quads graph.")
    parser.add_argument("--input", required=True, help="Input graph, typically after NER + temporal materialization")
    parser.add_argument("--dbpedia-json", required=True, help="Link cache from align/link_dbpedia.py")
    parser.add_argument("--dbpedia-index", required=True, help="SQLite DB produced by fusion/prepare_dbpedia_index.py")
    parser.add_argument("--out", required=True, help="Output N-Quads path")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum link confidence to materialize")
    parser.add_argument("--include-abstracts", action="store_true", help="Materialize compact @hi/@en abstracts when available")
    return parser


def load_link_cache(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    links: List[dict] = []
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            for link in item.get("links", []):
                if isinstance(link, dict):
                    row = dict(link)
                    row.setdefault("sent_key", f"{doc_id}::{sent_id}")
                    links.append(row)
    return links


def load_graph_lines(path: Path) -> tuple[List[str], str]:
    with path.open("r", encoding="utf-8") as handle:
        quads = [line.rstrip("\n") for line in handle]
    graphs: Set[str] = set()
    for line in quads:
        parsed = parse_nquad_line(line)
        if parsed is None:
            continue
        graphs.add(parsed.graph)
    graph_uri = sorted(graphs)[0] if graphs else DBPEDIA_GRAPH
    return quads, graph_uri


def add_once(quads: List[str], seen: Set[str], quad: str) -> None:
    if quad not in seen:
        quads.append(quad)
        seen.add(quad)


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    link_path = Path(args.dbpedia_json)
    db_path = Path(args.dbpedia_index)
    if not input_path.exists():
        raise SystemExit(f"Input graph not found: {input_path}")
    if not link_path.exists():
        raise SystemExit(f"DBpedia link cache not found: {link_path}")
    if not db_path.exists():
        raise SystemExit(f"DBpedia SQLite index not found: {db_path}")

    quads, _ = load_graph_lines(input_path)
    seen_quads = set(quads)
    links = load_link_cache(link_path)
    conn = connect_db(db_path)

    emitted_links = 0
    emitted_resources = 0
    emitted_sameas = 0
    materialized_resources: Set[str] = set()

    for link in links:
        try:
            confidence = float(link.get("confidence", link.get("score", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < args.min_confidence:
            continue

        mention_uri = str(link.get("mention_uri") or "").strip()
        canonical_uri = str(link.get("canonical_uri") or "").strip()
        if not mention_uri or not canonical_uri:
            continue

        add_once(quads, seen_quads, to_nquad_uri(mention_uri, REFERS_TO_DBPEDIA, canonical_uri, DBPEDIA_GRAPH))
        add_once(
            quads,
            seen_quads,
            to_nquad_literal(
                mention_uri,
                f"{NS['nhkg']}dbpediaMatchedVia",
                str(link.get("matched_via", "")) or "unknown",
                DBPEDIA_GRAPH,
                lang="en",
            ),
        )
        matched_lang = str(link.get("matched_lang", "")).strip()
        if matched_lang:
            add_once(
                quads,
                seen_quads,
                to_nquad_literal(mention_uri, f"{NS['nhkg']}dbpediaMatchedLang", matched_lang, DBPEDIA_GRAPH, lang="en"),
            )
        engine = str(link.get("link_engine", "") or link.get("engine", "") or "local")
        add_once(
            quads,
            seen_quads,
            to_nquad_literal(mention_uri, f"{NS['nhkg']}dbpediaLinkEngine", engine, DBPEDIA_GRAPH, lang="en"),
        )
        add_once(
            quads,
            seen_quads,
            to_nquad_decimal(mention_uri, f"{NS['nhkg']}dbpediaLinkConfidence", confidence, DBPEDIA_GRAPH),
        )
        emitted_links += 1

        if canonical_uri in materialized_resources:
            continue

        row = db_fetch_resource(conn, canonical_uri)
        add_once(quads, seen_quads, to_nquad_uri(canonical_uri, f"{NS['rdf']}type", LINKED_EXTERNAL_ENTITY, DBPEDIA_GRAPH))
        add_once(
            quads,
            seen_quads,
            to_nquad_literal(canonical_uri, f"{NS['nhkg']}sourceSystem", "DBpedia", DBPEDIA_GRAPH, lang="en"),
        )
        if row:
            if row.get("label_hi"):
                add_once(quads, seen_quads, to_nquad_literal(canonical_uri, f"{NS['rdfs']}label", row["label_hi"], DBPEDIA_GRAPH, lang="hi"))
            if row.get("label_en"):
                add_once(quads, seen_quads, to_nquad_literal(canonical_uri, f"{NS['rdfs']}label", row["label_en"], DBPEDIA_GRAPH, lang="en"))
            for dbo_type in row.get("types_dbo", []):
                add_once(quads, seen_quads, to_nquad_uri(canonical_uri, f"{NS['rdf']}type", dbo_type, DBPEDIA_GRAPH))
            if args.include_abstracts:
                if row.get("abstract_hi"):
                    add_once(
                        quads,
                        seen_quads,
                        to_nquad_literal(canonical_uri, "http://dbpedia.org/ontology/abstract", row["abstract_hi"], DBPEDIA_GRAPH, lang="hi"),
                    )
                if row.get("abstract_en"):
                    add_once(
                        quads,
                        seen_quads,
                        to_nquad_literal(canonical_uri, "http://dbpedia.org/ontology/abstract", row["abstract_en"], DBPEDIA_GRAPH, lang="en"),
                    )
            wikidata_uri = str(row.get("wikidata_uri", "")).strip()
            if wikidata_uri:
                add_once(quads, seen_quads, to_nquad_uri(canonical_uri, f"{NS['owl']}sameAs", wikidata_uri, DBPEDIA_GRAPH))
                emitted_sameas += 1
            for sameas_uri in row.get("sameas_language_uris", []):
                add_once(quads, seen_quads, to_nquad_uri(canonical_uri, f"{NS['owl']}sameAs", sameas_uri, DBPEDIA_GRAPH))
                emitted_sameas += 1
        else:
            for dbo_type in link.get("predicted_dbo_types", []) or []:
                add_once(quads, seen_quads, to_nquad_uri(canonical_uri, f"{NS['rdf']}type", dbo_type, DBPEDIA_GRAPH))
            wikidata_uri = str(link.get("wikidata_uri", "")).strip()
            if wikidata_uri:
                add_once(quads, seen_quads, to_nquad_uri(canonical_uri, f"{NS['owl']}sameAs", wikidata_uri, DBPEDIA_GRAPH))
                emitted_sameas += 1
        materialized_resources.add(canonical_uri)
        emitted_resources += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for quad in quads:
            handle.write(quad + "\n")

    conn.close()
    print(f"[OK] wrote_graph={out_path}")
    print(
        f"[OK] emitted_dbpedia_links={emitted_links} "
        f"dbpedia_resources={emitted_resources} sameas_links={emitted_sameas}"
    )


if __name__ == "__main__":
    main()
