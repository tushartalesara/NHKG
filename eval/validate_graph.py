#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structural + SHACL validation for NHKG RDF graphs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from pyshacl import validate as pyshacl_validate
except ImportError:  # pragma: no cover
    pyshacl_validate = None

try:
    from rdflib import ConjunctiveGraph, Graph, Namespace
except ImportError:  # pragma: no cover
    ConjunctiveGraph = None
    Graph = None
    Namespace = None


NS = {
    "nhkg": "http://ns.nhkg.org/resource/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "prov": "http://www.w3.org/ns/prov#",
    "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "schema": "https://schema.org/",
    "wn": "http://ns.nhkg.org/wordnet/",
}

NQ_RE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.*)\s+<([^>]+)>\s+\.$")
URI_RE = re.compile(r"^<([^>]+)>$")
INT_RE = re.compile(r'^"(-?\d+)"\^\^<[^>]+>$')
LIT_RE = re.compile(r'^"(.*)"(?:@([A-Za-z0-9-]+)|\^\^<([^>]+)>)?$')
PARSE_LINE_RE = re.compile(r"line\s+(\d+)", flags=re.IGNORECASE)
DBR_PREFIX = "http://dbpedia.org/resource/"
ROLE_PERSONISH = {"agent", "actor", "experiencer", "subject", "owner", "leader", "speaker"}
ROLE_PLACEISH = {"destination", "location", "source", "origin", "place", "venue"}


def parse_term(raw: str) -> Tuple[str, str]:
    raw = raw.strip()
    uri_match = URI_RE.match(raw)
    if uri_match:
        return "uri", uri_match.group(1)
    int_match = INT_RE.match(raw)
    if int_match:
        return "int", int_match.group(1)
    lit_match = LIT_RE.match(raw)
    if lit_match:
        return "literal", lit_match.group(1)
    return "raw", raw


def local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def is_local(uri: str) -> bool:
    return uri.startswith(NS["nhkg"])


def run_custom_checks(input_path: Path) -> dict:
    events: Set[str] = set()
    mentions: Set[str] = set()
    words: Set[str] = set()
    entity_mentions: Set[str] = set()
    timexes: Set[str] = set()
    canonical_entities: Set[str] = set()
    temporal_relations: Set[str] = set()
    has_trigger: Set[str] = set()
    begin_index: Dict[str, int] = {}
    end_index: Dict[str, int] = {}
    labels: Dict[str, str] = {}
    derived_from: Dict[str, str] = {}
    mention_dbpedia_links: List[Tuple[str, str]] = []
    mention_cluster_links: List[Tuple[str, str]] = []
    entity_type_labels: Dict[str, str] = {}
    dbpedia_resource_types: Dict[str, Set[str]] = {}
    event_role_edges: List[Tuple[str, str, str]] = []
    relation_sources: Dict[str, str] = {}
    relation_targets: Dict[str, str] = {}

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            match = NQ_RE.match(line)
            if not match:
                continue
            subject, predicate, obj_raw, _ = match.groups()
            obj_kind, obj_value = parse_term(obj_raw)
            local = local_name(subject)
            if predicate == f"{NS['rdf']}type" and obj_kind == "uri":
                if obj_value == f"{NS['schema']}Event" or local.startswith("Event_"):
                    events.add(subject)
                elif obj_value == f"{NS['nif']}Phrase" and local.startswith("Mention_"):
                    mentions.add(subject)
                elif obj_value == f"{NS['nif']}Word" or local.startswith("Word_"):
                    words.add(subject)
                elif obj_value == f"{NS['nhkg']}EntityMention" or local.startswith("EntityMention_"):
                    entity_mentions.add(subject)
                elif obj_value == f"{NS['nhkg']}Timex" or local.startswith("Timex_"):
                    timexes.add(subject)
                elif obj_value == f"{NS['nhkg']}CanonicalEntity" or local.startswith("CanonicalEntity_"):
                    canonical_entities.add(subject)
                elif obj_value == f"{NS['nhkg']}TemporalRelation" or local.startswith("TemporalRelation_"):
                    temporal_relations.add(subject)
                elif obj_value.startswith("http://dbpedia.org/ontology/"):
                    dbpedia_resource_types.setdefault(subject, set()).add(obj_value.rsplit("/", 1)[-1])
            if predicate == f"{NS['nhkg']}hasTrigger" and obj_kind == "uri":
                has_trigger.add(subject)
            elif predicate == f"{NS['nif']}beginIndex" and obj_kind == "int":
                begin_index[subject] = int(obj_value)
            elif predicate == f"{NS['nif']}endIndex" and obj_kind == "int":
                end_index[subject] = int(obj_value)
            elif predicate == f"{NS['rdfs']}label" and obj_kind == "literal":
                labels[subject] = obj_value
            elif predicate in {f"{NS['prov']}wasDerivedFrom", f"{NS['nif']}referenceContext"} and obj_kind == "uri":
                derived_from[subject] = obj_value
            elif predicate == f"{NS['nhkg']}entityType" and obj_kind == "literal":
                entity_type_labels[subject] = str(obj_value).upper()
            elif predicate == f"{NS['nhkg']}refersToDbpediaResource" and obj_kind == "uri":
                mention_dbpedia_links.append((subject, obj_value))
            elif predicate == f"{NS['nhkg']}refersToCanonicalEntity" and obj_kind == "uri":
                mention_cluster_links.append((subject, obj_value))
            elif predicate == f"{NS['nhkg']}sourceEvent" and obj_kind == "uri":
                relation_sources[subject] = obj_value
            elif predicate == f"{NS['nhkg']}targetEvent" and obj_kind == "uri":
                relation_targets[subject] = obj_value
            elif subject in events and obj_kind == "uri" and predicate not in {f"{NS['rdf']}type", f"{NS['rdfs']}label", f"{NS['prov']}wasGeneratedBy", f"{NS['prov']}wasDerivedFrom", f"{NS['nhkg']}hasTrigger"}:
                event_role_edges.append((subject, predicate, obj_value))

    errors: List[str] = []
    warnings: List[str] = []

    for event in sorted(events):
        if event not in has_trigger:
            errors.append(f"Event missing trigger: {event}")
    for word in sorted(words):
        start = begin_index.get(word)
        end = end_index.get(word)
        if start is None or end is None:
            errors.append(f"Word missing offsets: {word}")
            continue
        if end <= start:
            errors.append(f"Word has invalid offset range: {word} [{start}, {end}]")
        sentence_uri = derived_from.get(word)
        sentence_text = labels.get(sentence_uri or "", "")
        if sentence_uri and sentence_text and (start < 0 or end > len(sentence_text)):
            errors.append(f"Word offsets outside sentence bounds: {word} [{start}, {end}] > len={len(sentence_text)}")
    for relation_node in sorted(temporal_relations):
        if relation_node not in relation_sources or relation_node not in relation_targets:
            errors.append(f"Temporal relation missing source/target event: {relation_node}")
    for mention, cluster in mention_cluster_links:
        if cluster not in canonical_entities:
            warnings.append(f"Mention linked to missing canonical entity node: {mention} -> {cluster}")
    for mention, resource in mention_dbpedia_links:
        if not resource.startswith(DBR_PREFIX):
            warnings.append(f"Mention linked to non-canonical DBpedia resource URI: {mention} -> {resource}")
    for event, predicate, obj in event_role_edges:
        if is_local(obj) and obj not in mentions and obj not in entity_mentions and obj not in timexes and obj not in canonical_entities:
            warnings.append(f"Event role edge points to unexpected local node: {event} {predicate} {obj}")
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            match = NQ_RE.match(line)
            if not match:
                continue
            subject, predicate, obj_raw, _ = match.groups()
            obj_kind, obj_value = parse_term(obj_raw)
            if predicate == f"{NS['owl']}sameAs" and obj_kind == "uri":
                if local_name(subject).startswith(("Mention_", "EntityMention_")):
                    warnings.append(f"Unsafe mention/entity owl:sameAs usage: {subject} -> {obj_value}")
    mention_to_dbpedia = {mention: resource for mention, resource in mention_dbpedia_links}
    for mention, resource in mention_dbpedia_links:
        expected_label = entity_type_labels.get(mention, "")
        resource_types = dbpedia_resource_types.get(resource, set())
        if not expected_label or not resource_types:
            continue
        if expected_label == "PER" and "Person" not in resource_types:
            warnings.append(f"NER/DBpedia type mismatch: {mention} PER -> {resource_types}")
        elif expected_label == "LOC" and "Place" not in resource_types:
            warnings.append(f"NER/DBpedia type mismatch: {mention} LOC -> {resource_types}")
        elif expected_label == "ORG" and not ({"Organisation", "Organization", "Company"} & resource_types):
            warnings.append(f"NER/DBpedia type mismatch: {mention} ORG -> {resource_types}")
    for event, predicate, obj in event_role_edges:
        resource = mention_to_dbpedia.get(obj, "")
        if not resource:
            continue
        resource_types = dbpedia_resource_types.get(resource, set())
        if not resource_types:
            continue
        role_name = local_name(predicate).lower()
        if role_name in ROLE_PLACEISH and "Person" in resource_types and "Place" not in resource_types:
            warnings.append(f"Role/type mismatch: {event} {role_name} -> Person-linked resource {resource}")
        if role_name in ROLE_PERSONISH and "Place" in resource_types and not ({"Person", "Organisation", "Organization", "Company"} & resource_types):
            warnings.append(f"Role/type mismatch: {event} {role_name} -> Place-linked resource {resource}")

    return {
        "counts": {
            "events": len(events),
            "mentions": len(mentions),
            "words": len(words),
            "entity_mentions": len(entity_mentions),
            "timexes": len(timexes),
            "canonical_entities": len(canonical_entities),
            "temporal_relations": len(temporal_relations),
            "dbpedia_links": len(mention_dbpedia_links),
        },
        "errors": errors,
        "warnings": warnings,
    }


def line_number_from_error(exc: Exception) -> int | None:
    match = PARSE_LINE_RE.search(str(exc))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def run_parse_validation(input_path: Path) -> dict:
    if ConjunctiveGraph is None:
        return {
            "available": False,
            "parse_ok": None,
            "reason": "missing_dependency",
            "error": "rdflib is required for RDF parse validation.",
            "install_hint": "Install RDF dependencies with: pip install rdflib",
            "offending_file": str(input_path.resolve()),
            "line_number": None,
        }
    try:
        graph = ConjunctiveGraph()
        graph.parse(str(input_path), format="nquads")
        return {
            "available": True,
            "parse_ok": True,
            "reason": "",
            "error": "",
            "install_hint": "",
            "offending_file": str(input_path.resolve()),
            "line_number": None,
            "quad_count": len(graph),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "available": True,
            "parse_ok": False,
            "reason": "parse_failure",
            "error": str(exc),
            "install_hint": "",
            "offending_file": str(input_path.resolve()),
            "line_number": line_number_from_error(exc),
        }


def run_shacl_validation(input_path: Path, shacl_dir: Path, *, enabled: bool, parse_validation: dict) -> dict:
    if not enabled:
        return {
            "available": False,
            "enabled": False,
            "conforms": None,
            "reason": "SHACL validation disabled by CLI flag.",
            "install_hint": "",
            "warning": "SHACL validation disabled by CLI flag.",
            "results": [],
            "sample_violations": [],
            "counts_by_severity": {},
            "counts_by_shape": {},
            "shapes_loaded": 0,
            "report_text_summary": "",
        }
    if pyshacl_validate is None or ConjunctiveGraph is None or Graph is None or Namespace is None:
        return {
            "available": False,
            "enabled": True,
            "conforms": None,
            "reason": "missing_dependency",
            "install_hint": "Install SHACL dependencies with: pip install pyshacl rdflib",
            "warning": "pyshacl/rdflib not installed",
            "results": [],
            "sample_violations": [],
            "counts_by_severity": {},
            "counts_by_shape": {},
            "shapes_loaded": 0,
            "report_text_summary": "",
        }
    if parse_validation.get("parse_ok") is False:
        return {
            "available": False,
            "enabled": True,
            "conforms": None,
            "reason": "parse_failure",
            "install_hint": "",
            "warning": "SHACL validation skipped because RDF parsing failed before shape validation.",
            "parse_error": parse_validation.get("error", ""),
            "offending_file": parse_validation.get("offending_file", str(input_path.resolve())),
            "line_number": parse_validation.get("line_number"),
            "results": [],
            "sample_violations": [],
            "counts_by_severity": {},
            "counts_by_shape": {},
            "shapes_loaded": 0,
            "report_text_summary": "",
        }
    if not shacl_dir.exists():
        return {
            "available": False,
            "enabled": True,
            "conforms": None,
            "reason": f"SHACL directory not found: {shacl_dir}",
            "install_hint": "",
            "warning": f"SHACL directory not found: {shacl_dir}",
            "results": [],
            "sample_violations": [],
            "counts_by_severity": {},
            "counts_by_shape": {},
            "shapes_loaded": 0,
            "report_text_summary": "",
        }
    ttl_paths = sorted(shacl_dir.glob("*.ttl"))
    if not ttl_paths:
        return {
            "available": False,
            "enabled": True,
            "conforms": None,
            "reason": f"No SHACL shape files found in: {shacl_dir}",
            "install_hint": "",
            "warning": f"No SHACL shape files found in: {shacl_dir}",
            "results": [],
            "sample_violations": [],
            "counts_by_severity": {},
            "counts_by_shape": {},
            "shapes_loaded": 0,
            "report_text_summary": "",
        }
    try:
        data_graph = ConjunctiveGraph()
        data_graph.parse(str(input_path), format="nquads")
        shapes_graph = Graph()
        for ttl_path in ttl_paths:
            shapes_graph.parse(str(ttl_path), format="turtle")
        conforms, result_graph, report_text = pyshacl_validate(
            data_graph,
            shacl_graph=shapes_graph,
            inference="rdfs",
            abort_on_first=False,
            meta_shacl=False,
            advanced=True,
        )
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "enabled": True,
            "conforms": None,
            "reason": "shacl_runtime_failure",
            "install_hint": "Verify pyshacl/rdflib installation and the SHACL shape files, then rerun validation.",
            "warning": f"SHACL validation failed: {exc}",
            "results": [],
            "sample_violations": [],
            "counts_by_severity": {},
            "counts_by_shape": {},
            "shapes_loaded": len(ttl_paths),
            "report_text_summary": "",
        }
    SH = Namespace("http://www.w3.org/ns/shacl#")
    counts_by_severity: Dict[str, int] = {}
    counts_by_shape: Dict[str, int] = {}
    results: List[dict] = []
    for result in result_graph.subjects(predicate=None, object=SH.ValidationResult):
        node = result_graph.value(result, SH.focusNode)
        path = result_graph.value(result, SH.resultPath)
        message = result_graph.value(result, SH.resultMessage)
        severity = result_graph.value(result, SH.resultSeverity)
        source_shape = result_graph.value(result, SH.sourceShape)
        severity_text = str(severity).rsplit("#", 1)[-1] if severity else "Unknown"
        shape_text = str(source_shape).rsplit("#", 1)[-1] if source_shape else ""
        counts_by_severity[severity_text] = counts_by_severity.get(severity_text, 0) + 1
        if shape_text:
            counts_by_shape[shape_text] = counts_by_shape.get(shape_text, 0) + 1
        results.append(
            {
                "focus_node": str(node) if node else "",
                "path": str(path) if path else "",
                "message": str(message) if message else "",
                "severity": severity_text,
                "source_shape": str(source_shape) if source_shape else "",
                "source_shape_name": shape_text,
            }
        )
    return {
        "available": True,
        "enabled": True,
        "conforms": bool(conforms),
        "reason": "",
        "install_hint": "",
        "warning": "",
        "results": results,
        "sample_violations": results[:25],
        "counts_by_severity": counts_by_severity,
        "counts_by_shape": counts_by_shape,
        "shapes_loaded": len(ttl_paths),
        "report_text_summary": str(report_text).strip()[:4000],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run custom structural validation and optional SHACL validation over an NHKG graph.")
    ap.add_argument("--input", required=True, help="N-Quads graph to validate")
    ap.add_argument("--json", default="", help="Optional JSON report output")
    ap.add_argument("--shacl-dir", default=str((Path(__file__).resolve().parent.parent / "schemas" / "shacl")), help="Directory containing SHACL TTL files")
    ap.set_defaults(with_shacl=True)
    ap.add_argument("--with-shacl", dest="with_shacl", action="store_true", help="Run SHACL validation in addition to custom structural checks (default).")
    ap.add_argument("--no-shacl", dest="with_shacl", action="store_false", help="Skip SHACL validation and only run the custom structural checks.")
    args = ap.parse_args()

    input_path = Path(args.input)
    custom = run_custom_checks(input_path)
    parse_validation = run_parse_validation(input_path)
    shacl = run_shacl_validation(input_path, Path(args.shacl_dir), enabled=bool(args.with_shacl), parse_validation=parse_validation)

    custom_errors = list(custom["errors"])
    custom_warnings = list(custom["warnings"])
    parse_errors = []
    parse_warnings = []
    if parse_validation.get("parse_ok") is False:
        parse_errors.append(str(parse_validation.get("error") or "RDF parse validation failed"))
    elif parse_validation.get("available") is False and parse_validation.get("error"):
        parse_warnings.append(str(parse_validation.get("error")))
    shacl_warnings = [shacl["warning"]] if shacl.get("warning") else []
    shacl_errors = []
    if args.with_shacl and not shacl.get("available"):
        shacl_errors.append(str(shacl.get("reason") or "SHACL validation unavailable"))

    overall_ok = (
        (not custom_errors)
        and (not parse_errors)
        and ((not args.with_shacl) or (shacl.get("available") and shacl.get("conforms") is True))
    )
    report = {
        "graph": str(input_path.resolve()),
        "metadata": {
            "validator_version": "1.2.0",
            "shacl_requested": bool(args.with_shacl),
            "shacl_dir": str(Path(args.shacl_dir).resolve()),
        },
        "counts": custom["counts"],
        "custom_validation": {
            "counts": custom["counts"],
            "errors": custom_errors,
            "warnings": custom_warnings,
            "ok": not custom_errors,
        },
        "parse_validation": parse_validation,
        "shacl_validation": shacl,
        "errors": custom_errors + parse_errors + shacl_errors,
        "warnings": custom_warnings + parse_warnings + shacl_warnings,
        "overall_ok": overall_ok,
        "ok": overall_ok,
    }

    print(f"[OK] graph={report['graph']}")
    print(f"[OK] events={report['counts']['events']} mentions={report['counts']['mentions']} words={report['counts']['words']} entity_mentions={report['counts']['entity_mentions']} timexes={report['counts']['timexes']} canonical_entities={report['counts']['canonical_entities']} temporal_relations={report['counts']['temporal_relations']}")
    print(
        f"[OK] custom_errors={len(custom_errors)} custom_warnings={len(custom_warnings)} "
        f"parse_ok={parse_validation.get('parse_ok')} shacl_requested={bool(args.with_shacl)} "
        f"shacl_available={shacl.get('available')} shacl_conforms={shacl.get('conforms')}"
    )
    if parse_validation.get("parse_ok") is False:
        print(
            f"[WARN] parse_failure file={parse_validation.get('offending_file')} "
            f"line={parse_validation.get('line_number')} error={parse_validation.get('error')}"
        )
    if shacl.get("counts_by_severity"):
        print(f"[OK] shacl_counts_by_severity={shacl['counts_by_severity']}")
    if shacl.get("counts_by_shape"):
        print(f"[OK] shacl_counts_by_shape={shacl['counts_by_shape']}")
    if shacl.get("reason") and not shacl.get("available"):
        print(f"[WARN] shacl_reason={shacl['reason']}")
    for item in custom["errors"][:20]:
        print(f"[ERR] {item}")
    for item in custom["warnings"][:20]:
        print(f"[WARN] {item}")
    for item in shacl.get("sample_violations", [])[:10]:
        print(f"[SHACL] {item.get('severity')} :: {item.get('focus_node')} :: {item.get('message')}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"[OK] wrote_json={out_path}")


if __name__ == "__main__":
    main()
