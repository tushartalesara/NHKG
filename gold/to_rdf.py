#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 3.6: Gold JSON -> RDF N-Quads Materializer.
Converts extracted Frames into a Provenance-Aware Knowledge Graph.
"""

import argparse
import json
import uuid
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
import yaml

# Namespaces
NS = {
    "nhkg": "http://ns.nhkg.org/resource/",
    "uhvn": "http://ns.nhkg.org/uhvn/",
    "dbo": "http://dbpedia.org/ontology/",
    "dbr": "http://dbpedia.org/resource/",
    "schema": "https://schema.org/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": "http://www.w3.org/ns/prov#",
    "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "owl": "http://www.w3.org/2002/07/owl#",
}

CURIE_PREFIXES = {
    "dbo": NS["dbo"],
    "dbr": NS["dbr"],
    "schema": NS["schema"],
    "ns": NS["nhkg"],
    "uhvn": NS["uhvn"],
    "rdf": NS["rdf"],
    "rdfs": NS["rdfs"],
    "xsd": NS["xsd"],
    "prov": NS["prov"],
    "nif": NS["nif"],
    "skos": NS["skos"],
    "owl": NS["owl"],
}

GRAPH_URI = "http://ns.nhkg.org/graph/gold"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_uri(text):
    return urllib.parse.quote(str(text).strip().replace(" ", "_"))


def expand_curie(value: str) -> str:
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if ":" in value:
        prefix, suffix = value.split(":", 1)
        base = CURIE_PREFIXES.get(prefix)
        if base:
            return f"{base}{suffix}"
    return value


def format_node(value: str, as_uri: bool = True) -> str:
    if value is None:
        raise ValueError("Cannot format empty RDF term")

    if as_uri:
        uri = expand_curie(value)
        if not (uri.startswith("http://") or uri.startswith("https://")):
            raise ValueError(f"Expected URI term, got: {value}")
        return f"<{uri}>"

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f'"{value}"^^<{NS["xsd"]}integer>'
    if isinstance(value, str):
        if value.startswith("<") and value.endswith(">"):
            return value
        if value.count("@") == 1 and value.rsplit("@", 1)[1].isalpha():
            text_value, lang = value.rsplit("@", 1)
            return f'"{text_value}"@{lang}'
        if value.endswith("^^") or "^^<" in value:
            return value
        if value.startswith('"') and value.endswith('"'):
            return value
        return json.dumps(value, ensure_ascii=False)

    return f'"{str(value)}"'


def to_nquad(s, p, o, g=GRAPH_URI):
    subj = format_node(s, as_uri=True)
    pred = format_node(p, as_uri=True)
    if isinstance(o, str):
        expanded_o = expand_curie(o)
        if expanded_o.startswith("http://") or expanded_o.startswith("https://"):
            obj = format_node(expanded_o, as_uri=True)
        else:
            obj = format_node(o, as_uri=False)
    else:
        obj = format_node(o, as_uri=False)
    return f"{subj} {pred} {obj} <{g}> ."


def emit_span_metadata(subject_uri: str, meta: Dict, sentence_uri: str) -> List[str]:
    quads = []
    char_span = meta.get("char_span") if isinstance(meta, dict) else None
    if isinstance(char_span, list) and len(char_span) == 2:
        start, end = char_span
        if isinstance(start, int) and isinstance(end, int):
            quads.append(to_nquad(subject_uri, f"{NS['nif']}beginIndex", start))
            quads.append(to_nquad(subject_uri, f"{NS['nif']}endIndex", end))
    token_span = meta.get("span") if isinstance(meta, dict) else None
    if isinstance(token_span, list) and len(token_span) == 2:
        anchor = meta.get("text", "")
        if anchor:
            quads.append(to_nquad(subject_uri, f"{NS['nif']}anchorOf", f"{anchor}@hi"))
    if sentence_uri:
        quads.append(to_nquad(subject_uri, f"{NS['prov']}wasDerivedFrom", sentence_uri))
    return quads


def materialize_frame(event: dict, mapping: Dict, run_uri: str, sentence_texts: dict) -> List[str]:
    quads: List[str] = []

    meta = event.get("meta", {}) if isinstance(event.get("meta"), dict) else {}
    run_model = meta.get("model", "unknown-model")
    event_id = event.get("event_id") or f"{uuid.uuid4().hex[:12]}"

    doc_id = event.get("doc_id", "doc0")
    sent_id = event.get("sent_id", 0)
    frame = event.get("frame")

    event_uri = f"{NS['nhkg']}Event_{clean_uri(doc_id)}_{sent_id}_{clean_uri(event_id)}"
    sentence_uri = f"{NS['nhkg']}Sentence_{clean_uri(doc_id)}_{sent_id}"

    frame_map = mapping.get(frame, {}) if isinstance(mapping, dict) else {}
    if not frame_map:
        frame_map = mapping.get("DEFAULT", {}) if isinstance(mapping, dict) else {}
    rdf_type = expand_curie(frame_map.get("rdf_type", f"{NS['uhvn']}{frame}"))

    quads.append(to_nquad(event_uri, f"{NS['rdf']}type", rdf_type))
    trigger_text = (event.get("trigger", {}) or {}).get("text", "")
    if trigger_text:
        quads.append(to_nquad(event_uri, f"{NS['rdfs']}label", f"{trigger_text}@hi"))
    quads.append(to_nquad(event_uri, f"{NS['prov']}wasGeneratedBy", run_uri))
    quads.append(to_nquad(event_uri, f"{NS['prov']}wasDerivedFrom", sentence_uri))

    # Sentence node (minimal provenance record)
    sentence_text = sentence_texts.get((str(doc_id), str(sent_id)))
    quads.append(to_nquad(sentence_uri, f"{NS['rdf']}type", f"{NS['nif']}Sentence"))
    if sentence_text:
        quads.append(to_nquad(sentence_uri, f"{NS['rdfs']}label", sentence_text + "@hi"))

    # Trigger annotation as a mention node
    trigger = event.get("trigger", {}) or {}
    trigger_text = trigger.get("text", "")
    if trigger_text:
        trigger_uri = f"{NS['nhkg']}Trigger_{clean_uri(event_id)}"
        quads.append(to_nquad(event_uri, f"{NS['nhkg']}hasTrigger", trigger_uri))
        quads.append(to_nquad(trigger_uri, f"{NS['rdf']}type", f"{NS['nif']}Phrase"))
        quads.append(to_nquad(trigger_uri, f"{NS['rdfs']}label", trigger_text + "@hi"))
        quads.extend(emit_span_metadata(trigger_uri, trigger, sentence_uri))

    # Role arguments
    args = event.get("arguments", {}) or {}
    role_map = frame_map.get("roles", {}) if isinstance(frame_map, dict) else {}

    for role, arg_data in args.items():
        if not isinstance(arg_data, dict):
            continue
        arg_text = arg_data.get("text")
        if not arg_text:
            continue

        arg_uri = f"{NS['nhkg']}Mention_{clean_uri(event_id)}_{clean_uri(role)}"
        quads.append(to_nquad(arg_uri, f"{NS['rdf']}type", f"{NS['nif']}Phrase"))
        quads.append(to_nquad(arg_uri, f"{NS['rdfs']}label", f"{arg_text}@hi"))
        quads.extend(emit_span_metadata(arg_uri, arg_data, sentence_uri))

        mapping_conf = role_map.get(role, {})
        if isinstance(mapping_conf, str):
            pred = mapping_conf
        else:
            pred = mapping_conf.get("uri", f"{NS['uhvn']}{role}") if isinstance(mapping_conf, dict) else f"{NS['uhvn']}{role}"

        pred_uri = expand_curie(pred)
        quads.append(to_nquad(event_uri, pred_uri, arg_uri))

        # Keep mention + entity distinction for downstream linking.
        entity_uri = arg_data.get("entity_uri")
        if entity_uri:
            quads.append(to_nquad(arg_uri, f"{NS['skos']}exactMatch", entity_uri))
            quads.append(to_nquad(event_uri, pred_uri, entity_uri))

    model_uri = f"{NS['nhkg']}Model_{clean_uri(run_model)}"
    quads.append(to_nquad(model_uri, f"{NS['rdf']}type", f"{NS['nhkg']}Model"))
    quads.append(to_nquad(model_uri, f"{NS['rdfs']}label", run_model + "@en"))
    quads.append(to_nquad(run_uri, f"{NS['prov']}used", model_uri))

    run_time = meta.get("generated_at") or datetime.now(timezone.utc).isoformat()
    quads.append(to_nquad(run_uri, f"{NS['prov']}startedAtTime", f"\"{run_time}\"^^<{NS['xsd']}dateTime>"))
    return quads


def iter_events(item: object) -> Iterable[dict]:
    if isinstance(item, dict) and isinstance(item.get("events"), list):
        for event in item["events"]:
            if isinstance(event, dict):
                yield event
        return
    if isinstance(item, dict):
        yield item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file (output of pipeline.py)")
    ap.add_argument("--map", required=True, help="Path to uhvn2dbo_map.yaml")
    ap.add_argument("--out", required=True, help="Output .nq file")
    ap.add_argument("--sentence-texts", default="", help="Optional sidecar with one sentence per line")
    args = ap.parse_args()

    mapping = load_yaml(args.map)

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_uri = f"{NS['nhkg']}ExtractionRun_{run_id}"

    sentence_texts = {}
    if args.sentence_texts:
        with open(args.sentence_texts, "r", encoding="utf-8") as sf:
            for idx, line in enumerate(sf):
                sentence_texts[("batch_run", str(idx))] = line.strip()

    print(f"Materializing from {args.input}...")

    all_quads = [
        to_nquad(run_uri, f"{NS['rdf']}type", f"{NS['prov']}Activity"),
        to_nquad(run_uri, f"{NS['rdfs']}label", f'"Extraction run {run_id}"'),
    ]

    with open(args.input, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        rows: List[dict] = []
        if first_char == "[":
            rows = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        print(f"Materializing {len(rows)} row(s)...")

        for row in rows:
            for event in iter_events(row):
                for q in materialize_frame(event, mapping, run_uri, sentence_texts):
                    all_quads.append(q)

    with open(args.out, "w", encoding="utf-8") as f:
        for q in all_quads:
            f.write(q + "\n")

    print(f"[OK] Success. Wrote {len(all_quads)} quads to {args.out}")


if __name__ == "__main__":
    main()
