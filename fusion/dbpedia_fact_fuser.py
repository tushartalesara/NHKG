#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fuse a compact whitelist of DBpedia facts for already-linked resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from align.enrichment_common import NS, parse_nquad_line, to_nquad_decimal, to_nquad_literal, to_nquad_uri
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.enrichment_common import NS, parse_nquad_line, to_nquad_decimal, to_nquad_literal, to_nquad_uri

try:
    from .dbpedia_common import connect_db, db_fetch_facts, db_fetch_resource
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from fusion.dbpedia_common import connect_db, db_fetch_facts, db_fetch_resource


FACT_GRAPH = "http://ns.nhkg.org/graph/dbpedia/facts"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse a small whitelist of DBpedia facts into a separate RDF layer.")
    parser.add_argument("--input", required=True, help="Current RDF graph used only for duplicate suppression")
    parser.add_argument("--dbpedia-json", required=True, help="DBpedia link cache from align/link_dbpedia.py")
    parser.add_argument("--dbpedia-index", required=True, help="SQLite DB built by fusion/prepare_dbpedia_index.py")
    parser.add_argument("--fact-whitelist", required=True, help="YAML whitelist of DBpedia predicates to materialize")
    parser.add_argument("--out", required=True, help="Output N-Quads layer with fused DBpedia facts")
    return parser


def load_yaml_list(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Fact whitelist not found: {path}")
    if yaml is None:
        raise SystemExit("PyYAML is required to read the DBpedia fact whitelist.")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    values = payload.get("predicates", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise SystemExit(f"Whitelist format must be a list or a dict with 'predicates': {path}")
    return [str(item).strip() for item in values if str(item).strip()]


def load_existing_quads(path: Path) -> Set[str]:
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if raw:
                seen.add(raw)
    return seen


def load_linked_resources(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    resources: Set[str] = set()
    for sent_map in (payload.get("sentences", {}) or {}).values():
        if not isinstance(sent_map, dict):
            continue
        for item in sent_map.values():
            if not isinstance(item, dict):
                continue
            for link in item.get("links", []):
                if isinstance(link, dict):
                    uri = str(link.get("canonical_uri", "")).strip()
                    if uri:
                        resources.add(uri)
    return resources


def emit_fact_quad(subject: str, predicate: str, fact: dict) -> Optional[str]:
    obj_kind = str(fact.get("obj_kind", "")).strip()
    obj_value = fact.get("obj_value", "")
    if obj_kind == "uri":
        return to_nquad_uri(subject, predicate, str(obj_value), FACT_GRAPH)
    if obj_kind == "literal":
        lang = str(fact.get("lang", "") or "").strip() or None
        datatype = str(fact.get("datatype", "") or "").strip() or None
        return to_nquad_literal(subject, predicate, obj_value, FACT_GRAPH, lang=lang, datatype=datatype)
    if obj_kind == "decimal":
        try:
            return to_nquad_decimal(subject, predicate, float(obj_value), FACT_GRAPH)
        except (TypeError, ValueError):
            return None
    return None


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    link_path = Path(args.dbpedia_json)
    db_path = Path(args.dbpedia_index)
    whitelist_path = Path(args.fact_whitelist)
    if not input_path.exists():
        raise SystemExit(f"Input graph not found: {input_path}")
    if not link_path.exists():
        raise SystemExit(f"DBpedia link cache not found: {link_path}")
    if not db_path.exists():
        raise SystemExit(f"DBpedia index not found: {db_path}")

    whitelist = load_yaml_list(whitelist_path)
    existing_quads = load_existing_quads(input_path)
    linked_resources = load_linked_resources(link_path)
    conn = connect_db(db_path)

    emitted: List[str] = []
    added_labels = 0
    added_types = 0
    added_sameas = 0
    added_facts = 0

    for canonical_uri in sorted(linked_resources):
        row = db_fetch_resource(conn, canonical_uri)
        if row is None:
            continue

        for label_value, lang in ((row.get("label_hi", ""), "hi"), (row.get("label_en", ""), "en")):
            if label_value and f"{NS['rdfs']}label" in whitelist:
                quad = to_nquad_literal(canonical_uri, f"{NS['rdfs']}label", label_value, FACT_GRAPH, lang=lang)
                if quad not in existing_quads:
                    emitted.append(quad)
                    existing_quads.add(quad)
                    added_labels += 1

        if f"{NS['rdf']}type" in whitelist:
            for dbo_type in row.get("types_dbo", []):
                quad = to_nquad_uri(canonical_uri, f"{NS['rdf']}type", dbo_type, FACT_GRAPH)
                if quad not in existing_quads:
                    emitted.append(quad)
                    existing_quads.add(quad)
                    added_types += 1

        for sameas_uri in ([row.get("wikidata_uri", "")] + list(row.get("sameas_language_uris", []))):
            sameas_value = str(sameas_uri or "").strip()
            if sameas_value and f"{NS['owl']}sameAs" in whitelist:
                quad = to_nquad_uri(canonical_uri, f"{NS['owl']}sameAs", sameas_value, FACT_GRAPH)
                if quad not in existing_quads:
                    emitted.append(quad)
                    existing_quads.add(quad)
                    added_sameas += 1

        for fact in db_fetch_facts(conn, canonical_uri, predicates=whitelist):
            predicate = str(fact.get("predicate", "")).strip()
            if predicate in {f"{NS['rdf']}type", f"{NS['rdfs']}label", f"{NS['owl']}sameAs"}:
                continue
            quad = emit_fact_quad(canonical_uri, predicate, fact)
            if quad and quad not in existing_quads:
                emitted.append(quad)
                existing_quads.add(quad)
                added_facts += 1

        provenance_quad = to_nquad_literal(canonical_uri, f"{NS['nhkg']}sourceSystem", "DBpedia", FACT_GRAPH, lang="en")
        if provenance_quad not in existing_quads:
            emitted.append(provenance_quad)
            existing_quads.add(provenance_quad)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for quad in emitted:
            handle.write(quad + "\n")

    conn.close()
    print(f"[OK] wrote_dbpedia_facts={out_path}")
    print(
        f"[OK] linked_resources={len(linked_resources)} emitted_quads={len(emitted)} "
        f"labels={added_labels} types={added_types} sameas={added_sameas} facts={added_facts}"
    )


if __name__ == "__main__":
    main()
