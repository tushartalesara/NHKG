#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a compact local DBpedia index for Hindi entity linking and fact fusion."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    import rdflib
except ImportError:  # pragma: no cover
    rdflib = None

from fusion.dbpedia_common import (
    DBO_ABSTRACT,
    DBPEDIA_HI_PAGE_PREFIX,
    DBPEDIA_HI_PREFIX,
    DBPEDIA_ONTOLOGY_PREFIX,
    DBPEDIA_RESOURCE_PREFIX,
    DBO_WIKI_REDIRECTS,
    FACT_PREDICATE_HINTS,
    OWL_SAME_AS,
    RDF_TYPE,
    RDFS_LABEL,
    WIKIDATA_ENTITY_PREFIX,
    add_alias_rows,
    canonical_dbpedia_uri,
    connect_db,
    create_schema,
    guess_input_format,
    insert_or_merge_resource,
    insert_or_merge_staging_resource,
    insert_staging_fact,
    iter_tabular_rows,
    parse_list_field,
    reset_final_tables,
    reset_schema,
    sameas_language_uri,
    unique_preserve,
    uri_local_name,
)


TRIPLE_RE = re.compile(
    r'^<([^>]+)>\s+<([^>]+)>\s+(<[^>]+>|"(?:\\.|[^"\\])*"(?:@[A-Za-z0-9-]+|\^\^<[^>]+>)?)'
    r'(?:\s+<[^>]+>)?\s+\.$'
)
LITERAL_RE = re.compile(r'^"((?:\\.|[^"\\])*)"(?:@([A-Za-z0-9-]+)|\^\^<([^>]+)>)?$')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local DBpedia SQLite index for NHKG linking/fusion.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more translated DBpedia artifacts")
    parser.add_argument(
        "--input-format",
        choices=["translated_jsonl", "rdf_dump", "tabular", "hybrid"],
        required=True,
        help="How to interpret the supplied inputs",
    )
    parser.add_argument("--out-dir", default="data/dbpedia/index", help="Index output directory")
    parser.add_argument("--resource-db", default="", help="Explicit SQLite DB path (defaults under --out-dir)")
    parser.add_argument("--overwrite", action="store_true", help="Replace any existing DB/index metadata")
    parser.add_argument("--debug-samples", type=int, default=3, help="Sample rows to print")
    return parser


def parse_term(raw: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    raw = raw.strip()
    if raw.startswith("<") and raw.endswith(">"):
        return "uri", raw[1:-1], None, None
    match = LITERAL_RE.match(raw)
    if match:
        value = json.loads(f'"{match.group(1)}"')
        return "literal", value, match.group(2), match.group(3)
    return "raw", raw, None, None


def iter_nt_like(path: Path) -> Iterator[Tuple[str, str, str, str, Optional[str], Optional[str]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            match = TRIPLE_RE.match(raw)
            if not match:
                continue
            subject, predicate, obj_raw = match.groups()
            obj_kind, obj_value, lang, datatype = parse_term(obj_raw)
            yield subject, predicate, obj_kind, obj_value, lang, datatype


def iter_rdf_rows(path: Path) -> Iterator[Tuple[str, str, str, str, Optional[str], Optional[str]]]:
    suffix = path.suffix.lower()
    if suffix in {".nt", ".nq"}:
        yield from iter_nt_like(path)
        return
    if rdflib is None:
        raise SystemExit(
            f"RDF file '{path}' requires rdflib for parsing. Install rdflib or supply an .nt/.nq dump instead."
        )
    graph = rdflib.ConjunctiveGraph()
    graph.parse(str(path))
    for subject, predicate, obj in graph:
        subj = str(subject)
        pred = str(predicate)
        if isinstance(obj, rdflib.term.URIRef):
            yield subj, pred, "uri", str(obj), None, None
        elif isinstance(obj, rdflib.term.Literal):
            yield subj, pred, "literal", str(obj), obj.language, str(obj.datatype) if obj.datatype else None


def translated_jsonl_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL in {path}:{line_number}") from exc
            if isinstance(payload, dict):
                yield payload


def coerce_jsonl_record(row: dict, source_path: Path) -> Dict[str, object]:
    subject_uri = (
        row.get("uri")
        or row.get("canonical_uri")
        or row.get("resource_uri")
        or row.get("key")
        or ""
    )
    subject_uri = str(subject_uri).strip()
    canonical_uri = canonical_dbpedia_uri(subject_uri) or canonical_dbpedia_uri(str(row.get("canonical_uri", "")).strip())
    sameas_list = parse_list_field(row.get("sameas_language_uris"))
    wikidata_uri = str(row.get("wikidata") or row.get("wikidata_uri") or "").strip()
    if sameas_language_uri(subject_uri):
        sameas_list = unique_preserve([subject_uri, *sameas_list])
    aliases_hi = parse_list_field(row.get("aliases_hi"))
    aliases_en = parse_list_field(row.get("aliases_en"))
    redirects = parse_list_field(row.get("redirects"))
    disambiguations = parse_list_field(row.get("disambiguations") or row.get("disambiguation_titles"))
    translated_titles = parse_list_field(row.get("translated_titles") or row.get("titles_hi"))
    wikidata_labels = parse_list_field(row.get("wikidata_labels") or row.get("wikidata_aliases"))
    sameas_titles = [uri_local_name(uri) for uri in sameas_list]
    return {
        "subject_uri": subject_uri or canonical_uri,
        "canonical_uri": canonical_uri,
        "label_hi": row.get("label_hi") or row.get("hi") or "",
        "label_en": row.get("label_en") or row.get("en") or "",
        "aliases_hi": unique_preserve([*aliases_hi, *translated_titles, *disambiguations, *wikidata_labels, *sameas_titles]),
        "aliases_en": unique_preserve([*aliases_en, uri_local_name(canonical_uri or subject_uri)]),
        "redirects": unique_preserve([*redirects, *disambiguations]),
        "abstract_hi": row.get("abstract_hi") or "",
        "abstract_en": row.get("abstract_en") or "",
        "types_dbo": parse_list_field(row.get("types") or row.get("types_dbo")),
        "wikidata_uri": wikidata_uri if wikidata_uri.startswith(WIKIDATA_ENTITY_PREFIX) else "",
        "sameas_language_uris": sameas_list,
        "popularity_score": row.get("popularity_score"),
        "source_artifact": str(source_path),
        "source_lang": row.get("source_lang") or ("hi" if row.get("hi") or row.get("label_hi") else ""),
        "source_version": row.get("source_version") or "",
    }


def ingest_translated_jsonl(conn, path: Path, stats: Counter) -> None:
    for row in translated_jsonl_rows(path):
        record = coerce_jsonl_record(row, path)
        if not record["subject_uri"]:
            stats["jsonl_rows_missing_uri"] += 1
            continue
        insert_or_merge_staging_resource(conn, record)
        stats["jsonl_rows"] += 1
        if record.get("label_hi"):
            stats["label_hi_rows"] += 1
        if record.get("canonical_uri"):
            stats["canonical_rows"] += 1


def ingest_tabular(conn, path: Path, stats: Counter) -> None:
    for row in iter_tabular_rows(path):
        record = coerce_jsonl_record(row, path)
        if not record["subject_uri"]:
            stats["tabular_rows_missing_uri"] += 1
            continue
        insert_or_merge_staging_resource(conn, record)
        stats["tabular_rows"] += 1
        if record.get("label_hi"):
            stats["label_hi_rows"] += 1
        if record.get("canonical_uri"):
            stats["canonical_rows"] += 1


def ingest_rdf_dump(conn, path: Path, stats: Counter) -> None:
    for subject, predicate, obj_kind, obj_value, lang, datatype in iter_rdf_rows(path):
        if not (
            subject.startswith(DBPEDIA_HI_PREFIX)
            or subject.startswith(DBPEDIA_HI_PAGE_PREFIX)
            or subject.startswith(DBPEDIA_RESOURCE_PREFIX)
        ):
            continue
        stats["rdf_triples_seen"] += 1
        record: Dict[str, object] = {
            "subject_uri": subject,
            "source_artifact": str(path),
            "source_lang": "hi" if subject.startswith(DBPEDIA_HI_PREFIX) or subject.startswith(DBPEDIA_HI_PAGE_PREFIX) else "en",
        }
        changed = False

        if predicate == OWL_SAME_AS and obj_kind == "uri":
            canonical_uri = canonical_dbpedia_uri(obj_value)
            if canonical_uri:
                record["canonical_uri"] = canonical_uri
                if sameas_language_uri(subject):
                    record["sameas_language_uris"] = [subject]
                changed = True
                stats["sameas_to_canonical"] += 1
            else:
                language_uri = sameas_language_uri(obj_value)
                if language_uri:
                    record["sameas_language_uris"] = [language_uri]
                    changed = True
                    stats["sameas_language_links"] += 1
                elif str(obj_value).startswith(WIKIDATA_ENTITY_PREFIX):
                    record["wikidata_uri"] = obj_value
                    changed = True
                    stats["wikidata_links"] += 1
        elif predicate == RDFS_LABEL and obj_kind == "literal":
            if lang == "hi":
                record["label_hi"] = obj_value
                record["aliases_hi"] = [obj_value]
                changed = True
                stats["label_hi_rows"] += 1
            elif lang == "en":
                record["label_en"] = obj_value
                record["aliases_en"] = [obj_value]
                changed = True
                stats["label_en_rows"] += 1
        elif predicate == DBO_ABSTRACT and obj_kind == "literal":
            if lang == "hi":
                record["abstract_hi"] = obj_value
                changed = True
                stats["abstract_hi_rows"] += 1
            elif lang == "en":
                record["abstract_en"] = obj_value
                changed = True
                stats["abstract_en_rows"] += 1
        elif predicate == RDF_TYPE and obj_kind == "uri" and str(obj_value).startswith(DBPEDIA_ONTOLOGY_PREFIX):
            record["types_dbo"] = [obj_value]
            changed = True
            stats["type_rows"] += 1
        elif predicate == DBO_WIKI_REDIRECTS and obj_kind == "uri":
            record["redirects"] = [uri_local_name(obj_value)]
            changed = True
            stats["redirect_rows"] += 1
        elif predicate in FACT_PREDICATE_HINTS or predicate.startswith(DBPEDIA_ONTOLOGY_PREFIX):
            insert_staging_fact(
                conn,
                subject,
                predicate,
                obj_kind,
                str(obj_value),
                lang=lang,
                datatype=datatype,
            )
            stats["staging_facts"] += 1

        if changed:
            if subject.startswith(DBPEDIA_HI_PREFIX) or subject.startswith(DBPEDIA_HI_PAGE_PREFIX):
                record.setdefault("sameas_language_uris", [subject])
            insert_or_merge_staging_resource(conn, record)
            stats["staging_resources_touched"] += 1


def merge_aliases(*groups) -> List[str]:
    values: List[object] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, (list, tuple)):
            values.extend(group)
        else:
            values.append(group)
    return unique_preserve(values)


def finalize_index(conn, source_files: List[str]) -> Dict[str, object]:
    reset_final_tables(conn)
    stats = Counter()
    subject_to_canonical: Dict[str, str] = {}

    rows = conn.execute("SELECT * FROM staging_resources").fetchall()
    for row in rows:
        subject_uri = row["subject_uri"]
        canonical_uri = canonical_dbpedia_uri(row["canonical_uri"] or "") or canonical_dbpedia_uri(subject_uri)
        if not canonical_uri:
            stats["subjects_without_canonical_uri"] += 1
            continue

        subject_to_canonical[subject_uri] = canonical_uri
        sameas_uris = parse_list_field(row["sameas_language_uris_json"])
        if sameas_language_uri(subject_uri) and subject_uri not in sameas_uris and subject_uri != canonical_uri:
            sameas_uris.insert(0, subject_uri)

        alias_hi = merge_aliases(
            parse_list_field(row["aliases_hi_json"]),
            row["label_hi"],
            uri_local_name(subject_uri) if sameas_language_uri(subject_uri) else "",
        )
        alias_en = merge_aliases(
            parse_list_field(row["aliases_en_json"]),
            row["label_en"],
            uri_local_name(canonical_uri),
        )
        redirects = merge_aliases(parse_list_field(row["redirects_json"]))
        types_dbo = parse_list_field(row["types_dbo_json"])

        insert_or_merge_resource(
            conn,
            {
                "canonical_uri": canonical_uri,
                "label_hi": row["label_hi"] or "",
                "label_en": row["label_en"] or "",
                "aliases_hi": alias_hi,
                "aliases_en": alias_en,
                "redirects": redirects,
                "abstract_hi": row["abstract_hi"] or "",
                "abstract_en": row["abstract_en"] or "",
                "types_dbo": types_dbo,
                "wikidata_uri": row["wikidata_uri"] or "",
                "sameas_language_uris": sameas_uris,
                "popularity_score": row["popularity_score"],
                "source_artifact": row["source_artifact"] or "",
                "source_lang": row["source_lang"] or "",
                "source_version": row["source_version"] or "",
            },
        )
        stats["resources_with_canonical_uri"] += 1

    resources = conn.execute("SELECT * FROM resources").fetchall()
    for row in resources:
        canonical_uri = row["canonical_uri"]
        sameas_titles = [uri_local_name(uri) for uri in parse_list_field(row["sameas_language_uris_json"])]
        add_alias_rows(
            conn,
            canonical_uri,
            merge_aliases([row["label_hi"], *parse_list_field(row["aliases_hi_json"]), *sameas_titles]),
            "hi",
            "label_hi",
        )
        add_alias_rows(conn, canonical_uri, merge_aliases([row["label_en"], *parse_list_field(row["aliases_en_json"])]), "en", "label_en")
        add_alias_rows(conn, canonical_uri, parse_list_field(row["redirects_json"]), "hi", "redirect")
        add_alias_rows(conn, canonical_uri, parse_list_field(row["sameas_language_uris_json"]), "uri", "sameas_language")

    fact_rows = conn.execute("SELECT * FROM staging_facts").fetchall()
    for row in fact_rows:
        canonical_uri = subject_to_canonical.get(row["subject_uri"])
        if not canonical_uri:
            continue
        conn.execute(
            """
            INSERT INTO facts (canonical_uri, predicate, obj_kind, obj_value, lang, datatype)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_uri,
                row["predicate"],
                row["obj_kind"],
                row["obj_value"],
                row["lang"],
                row["datatype"],
            ),
        )
        stats["facts_materialized"] += 1

    conn.commit()
    stats["final_resource_count"] = conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
    stats["final_alias_count"] = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    stats["final_fact_count"] = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    stats["resources_with_hi_label"] = conn.execute("SELECT COUNT(*) FROM resources WHERE COALESCE(label_hi, '') <> ''").fetchone()[0]
    stats["resources_with_wikidata_bridge"] = conn.execute("SELECT COUNT(*) FROM resources WHERE COALESCE(wikidata_uri, '') <> ''").fetchone()[0]
    stats["resources_with_dbo_types"] = conn.execute("SELECT COUNT(*) FROM resources WHERE COALESCE(types_dbo_json, '') NOT IN ('', '[]')").fetchone()[0]
    stats["hindi_alias_rows"] = conn.execute("SELECT COUNT(*) FROM aliases WHERE lang = 'hi'").fetchone()[0]
    stats["redirect_alias_rows"] = conn.execute("SELECT COUNT(*) FROM aliases WHERE match_type = 'redirect'").fetchone()[0]
    return {
        "counts": dict(stats),
        "source_files": source_files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_version": "dbpedia-sqlite-v1",
        "available_languages": ["hi", "en"],
        "available_fields": [
            "canonical_uri",
            "label_hi",
            "label_en",
            "aliases_hi",
            "aliases_en",
            "redirects",
            "abstract_hi",
            "abstract_en",
            "types_dbo",
            "wikidata_uri",
            "sameas_language_uris",
            "facts",
        ],
    }


def handle_input(conn, path: Path, input_format: str, stats: Counter) -> None:
    mode = guess_input_format(path) if input_format == "hybrid" else input_format
    if mode == "translated_jsonl":
        ingest_translated_jsonl(conn, path, stats)
    elif mode == "tabular":
        ingest_tabular(conn, path, stats)
    elif mode == "rdf_dump":
        ingest_rdf_dump(conn, path, stats)
    else:  # pragma: no cover
        raise SystemExit(f"Unsupported input mode: {mode}")


def main() -> None:
    args = build_arg_parser().parse_args()

    input_paths = [Path(value).resolve() for value in args.input]
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"DBpedia input not found: {path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    resource_db = Path(args.resource_db).resolve() if args.resource_db else out_dir / "dbpedia_resources.sqlite"
    meta_path = out_dir / "dbpedia_index_meta.json"

    conn = connect_db(resource_db)
    create_schema(conn)
    if args.overwrite:
        reset_schema(conn)

    ingest_stats = Counter()
    for path in input_paths:
        handle_input(conn, path, args.input_format, ingest_stats)
        conn.commit()

    metadata = finalize_index(conn, [str(path) for path in input_paths])
    metadata["ingest_counts"] = dict(ingest_stats)
    metadata["resource_db"] = str(resource_db)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"[OK] dbpedia_index={resource_db}")
    print(f"[OK] meta={meta_path}")
    print(
        f"[OK] resources={metadata['counts'].get('final_resource_count', 0)} "
        f"aliases={metadata['counts'].get('final_alias_count', 0)} "
        f"facts={metadata['counts'].get('final_fact_count', 0)}"
    )
    if args.debug_samples > 0:
        rows = conn.execute(
            "SELECT canonical_uri, label_hi, label_en FROM resources ORDER BY canonical_uri LIMIT ?",
            (max(1, args.debug_samples),),
        ).fetchall()
        for row in rows:
            print(f"[DBG] {row['canonical_uri']} :: hi={row['label_hi'] or ''} :: en={row['label_en'] or ''}")
    conn.close()


if __name__ == "__main__":
    main()
