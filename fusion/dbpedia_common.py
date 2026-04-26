#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for local DBpedia indexing, linking, and fact fusion."""

from __future__ import annotations

import csv
import json
import sqlite3
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DBPEDIA_RESOURCE_PREFIX = "http://dbpedia.org/resource/"
DBPEDIA_ONTOLOGY_PREFIX = "http://dbpedia.org/ontology/"
DBPEDIA_HI_PREFIX = "http://hi.dbpedia.org/resource/"
DBPEDIA_HI_PAGE_PREFIX = "http://hi.dbpedia.org/page/"
WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
DBO_ABSTRACT = "http://dbpedia.org/ontology/abstract"
DBO_WIKI_REDIRECTS = "http://dbpedia.org/ontology/wikiPageRedirects"
GEO_LAT = "http://www.w3.org/2003/01/geo/wgs84_pos#lat"
GEO_LONG = "http://www.w3.org/2003/01/geo/wgs84_pos#long"

PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u0964": " ",
        "\u0965": " ",
        "-": " ",
        "_": " ",
        "/": " ",
        ",": " ",
        ";": " ",
        ":": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "|": " ",
    }
)

FACT_PREDICATE_HINTS = {
    RDFS_LABEL,
    RDF_TYPE,
    DBO_ABSTRACT,
    DBO_WIKI_REDIRECTS,
    GEO_LAT,
    GEO_LONG,
}


def json_loads_or(value: object, default):
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def normalize_lookup_text(text: object) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    raw = raw.translate(PUNCT_TRANSLATION)
    raw = " ".join(raw.split())
    raw = raw.replace("़", "")
    return raw.casefold().strip()


def unique_preserve(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        marker = text.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(text)
    return out


def parse_list_field(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return unique_preserve(value)
    if isinstance(value, tuple):
        return unique_preserve(list(value))
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            parsed = json_loads_or(raw, [])
            if isinstance(parsed, list):
                return unique_preserve(parsed)
        if "|" in raw:
            return unique_preserve(part.strip() for part in raw.split("|"))
        if ";" in raw:
            return unique_preserve(part.strip() for part in raw.split(";"))
        if "," in raw:
            return unique_preserve(part.strip() for part in raw.split(","))
        return [raw]
    return [str(value)]


def uri_local_name(uri: str) -> str:
    if "#" in uri:
        uri = uri.rsplit("#", 1)[1]
    else:
        uri = uri.rsplit("/", 1)[-1]
    return urllib.parse.unquote(uri.replace("_", " "))


def canonical_dbpedia_uri(uri: str) -> Optional[str]:
    text = str(uri or "").strip()
    if not text:
        return None
    if text.startswith(DBPEDIA_RESOURCE_PREFIX):
        return text
    return None


def sameas_language_uri(uri: str) -> Optional[str]:
    text = str(uri or "").strip()
    if not text:
        return None
    if text.startswith(DBPEDIA_HI_PREFIX) or text.startswith(DBPEDIA_HI_PAGE_PREFIX):
        return text
    if ".dbpedia.org/" in text and not text.startswith(DBPEDIA_RESOURCE_PREFIX):
        return text
    return None


def guess_input_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "translated_jsonl"
    if suffix in {".csv", ".tsv"}:
        return "tabular"
    if suffix in {".nt", ".ttl", ".nq"}:
        return "rdf_dump"
    return "translated_jsonl"


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS staging_resources (
            subject_uri TEXT PRIMARY KEY,
            canonical_uri TEXT,
            label_hi TEXT,
            label_en TEXT,
            aliases_hi_json TEXT,
            aliases_en_json TEXT,
            redirects_json TEXT,
            abstract_hi TEXT,
            abstract_en TEXT,
            types_dbo_json TEXT,
            wikidata_uri TEXT,
            sameas_language_uris_json TEXT,
            popularity_score REAL,
            source_artifact TEXT,
            source_lang TEXT,
            source_version TEXT
        );

        CREATE TABLE IF NOT EXISTS staging_facts (
            subject_uri TEXT NOT NULL,
            predicate TEXT NOT NULL,
            obj_kind TEXT NOT NULL,
            obj_value TEXT NOT NULL,
            lang TEXT,
            datatype TEXT
        );

        CREATE TABLE IF NOT EXISTS resources (
            canonical_uri TEXT PRIMARY KEY,
            label_hi TEXT,
            label_en TEXT,
            aliases_hi_json TEXT,
            aliases_en_json TEXT,
            redirects_json TEXT,
            abstract_hi TEXT,
            abstract_en TEXT,
            types_dbo_json TEXT,
            wikidata_uri TEXT,
            sameas_language_uris_json TEXT,
            popularity_score REAL,
            source_artifact TEXT,
            source_lang TEXT,
            source_version TEXT
        );

        CREATE TABLE IF NOT EXISTS aliases (
            alias_norm TEXT NOT NULL,
            alias TEXT NOT NULL,
            lang TEXT NOT NULL,
            canonical_uri TEXT NOT NULL,
            match_type TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            canonical_uri TEXT NOT NULL,
            predicate TEXT NOT NULL,
            obj_kind TEXT NOT NULL,
            obj_value TEXT NOT NULL,
            lang TEXT,
            datatype TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_alias_norm ON aliases(alias_norm);
        CREATE INDEX IF NOT EXISTS idx_alias_exact ON aliases(alias);
        CREATE INDEX IF NOT EXISTS idx_alias_uri ON aliases(canonical_uri);
        CREATE INDEX IF NOT EXISTS idx_fact_uri ON facts(canonical_uri);
        CREATE INDEX IF NOT EXISTS idx_fact_predicate ON facts(predicate);
        """
    )
    conn.commit()


def reset_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS staging_resources;
        DROP TABLE IF EXISTS staging_facts;
        DROP TABLE IF EXISTS resources;
        DROP TABLE IF EXISTS aliases;
        DROP TABLE IF EXISTS facts;
        """
    )
    conn.commit()
    create_schema(conn)


def merge_strings(left: object, right: object) -> str:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    return left_text or right_text


def merge_lists(left: object, right: object) -> List[str]:
    return unique_preserve(parse_list_field(left) + parse_list_field(right))


def insert_or_merge_staging_resource(conn: sqlite3.Connection, record: Dict[str, object]) -> None:
    subject_uri = str(record.get("subject_uri") or "").strip()
    if not subject_uri:
        return
    existing = conn.execute(
        "SELECT * FROM staging_resources WHERE subject_uri = ?",
        (subject_uri,),
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO staging_resources (
                subject_uri, canonical_uri, label_hi, label_en, aliases_hi_json, aliases_en_json,
                redirects_json, abstract_hi, abstract_en, types_dbo_json, wikidata_uri,
                sameas_language_uris_json, popularity_score, source_artifact, source_lang, source_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_uri,
                record.get("canonical_uri"),
                record.get("label_hi"),
                record.get("label_en"),
                json.dumps(parse_list_field(record.get("aliases_hi")), ensure_ascii=False),
                json.dumps(parse_list_field(record.get("aliases_en")), ensure_ascii=False),
                json.dumps(parse_list_field(record.get("redirects")), ensure_ascii=False),
                record.get("abstract_hi"),
                record.get("abstract_en"),
                json.dumps(parse_list_field(record.get("types_dbo")), ensure_ascii=False),
                record.get("wikidata_uri"),
                json.dumps(parse_list_field(record.get("sameas_language_uris")), ensure_ascii=False),
                record.get("popularity_score"),
                record.get("source_artifact"),
                record.get("source_lang"),
                record.get("source_version"),
            ),
        )
        return

    merged = {
        "canonical_uri": merge_strings(record.get("canonical_uri"), existing["canonical_uri"]),
        "label_hi": merge_strings(existing["label_hi"], record.get("label_hi")),
        "label_en": merge_strings(existing["label_en"], record.get("label_en")),
        "aliases_hi": merge_lists(existing["aliases_hi_json"], record.get("aliases_hi")),
        "aliases_en": merge_lists(existing["aliases_en_json"], record.get("aliases_en")),
        "redirects": merge_lists(existing["redirects_json"], record.get("redirects")),
        "abstract_hi": merge_strings(existing["abstract_hi"], record.get("abstract_hi")),
        "abstract_en": merge_strings(existing["abstract_en"], record.get("abstract_en")),
        "types_dbo": merge_lists(existing["types_dbo_json"], record.get("types_dbo")),
        "wikidata_uri": merge_strings(existing["wikidata_uri"], record.get("wikidata_uri")),
        "sameas_language_uris": merge_lists(existing["sameas_language_uris_json"], record.get("sameas_language_uris")),
        "popularity_score": record.get("popularity_score") if record.get("popularity_score") is not None else existing["popularity_score"],
        "source_artifact": merge_strings(existing["source_artifact"], record.get("source_artifact")),
        "source_lang": merge_strings(existing["source_lang"], record.get("source_lang")),
        "source_version": merge_strings(existing["source_version"], record.get("source_version")),
    }
    conn.execute(
        """
        UPDATE staging_resources
        SET canonical_uri = ?, label_hi = ?, label_en = ?, aliases_hi_json = ?, aliases_en_json = ?,
            redirects_json = ?, abstract_hi = ?, abstract_en = ?, types_dbo_json = ?, wikidata_uri = ?,
            sameas_language_uris_json = ?, popularity_score = ?, source_artifact = ?, source_lang = ?, source_version = ?
        WHERE subject_uri = ?
        """,
        (
            merged["canonical_uri"],
            merged["label_hi"],
            merged["label_en"],
            json.dumps(merged["aliases_hi"], ensure_ascii=False),
            json.dumps(merged["aliases_en"], ensure_ascii=False),
            json.dumps(merged["redirects"], ensure_ascii=False),
            merged["abstract_hi"],
            merged["abstract_en"],
            json.dumps(merged["types_dbo"], ensure_ascii=False),
            merged["wikidata_uri"],
            json.dumps(merged["sameas_language_uris"], ensure_ascii=False),
            merged["popularity_score"],
            merged["source_artifact"],
            merged["source_lang"],
            merged["source_version"],
            subject_uri,
        ),
    )


def insert_staging_fact(
    conn: sqlite3.Connection,
    subject_uri: str,
    predicate: str,
    obj_kind: str,
    obj_value: str,
    *,
    lang: Optional[str] = None,
    datatype: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO staging_facts (subject_uri, predicate, obj_kind, obj_value, lang, datatype)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (subject_uri, predicate, obj_kind, obj_value, lang, datatype),
    )


def insert_or_merge_resource(conn: sqlite3.Connection, record: Dict[str, object]) -> None:
    canonical_uri = str(record.get("canonical_uri") or "").strip()
    if not canonical_uri:
        return
    existing = conn.execute(
        "SELECT * FROM resources WHERE canonical_uri = ?",
        (canonical_uri,),
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO resources (
                canonical_uri, label_hi, label_en, aliases_hi_json, aliases_en_json, redirects_json,
                abstract_hi, abstract_en, types_dbo_json, wikidata_uri, sameas_language_uris_json,
                popularity_score, source_artifact, source_lang, source_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_uri,
                record.get("label_hi"),
                record.get("label_en"),
                json.dumps(parse_list_field(record.get("aliases_hi")), ensure_ascii=False),
                json.dumps(parse_list_field(record.get("aliases_en")), ensure_ascii=False),
                json.dumps(parse_list_field(record.get("redirects")), ensure_ascii=False),
                record.get("abstract_hi"),
                record.get("abstract_en"),
                json.dumps(parse_list_field(record.get("types_dbo")), ensure_ascii=False),
                record.get("wikidata_uri"),
                json.dumps(parse_list_field(record.get("sameas_language_uris")), ensure_ascii=False),
                record.get("popularity_score"),
                record.get("source_artifact"),
                record.get("source_lang"),
                record.get("source_version"),
            ),
        )
        return

    merged = {
        "label_hi": merge_strings(existing["label_hi"], record.get("label_hi")),
        "label_en": merge_strings(existing["label_en"], record.get("label_en")),
        "aliases_hi": merge_lists(existing["aliases_hi_json"], record.get("aliases_hi")),
        "aliases_en": merge_lists(existing["aliases_en_json"], record.get("aliases_en")),
        "redirects": merge_lists(existing["redirects_json"], record.get("redirects")),
        "abstract_hi": merge_strings(existing["abstract_hi"], record.get("abstract_hi")),
        "abstract_en": merge_strings(existing["abstract_en"], record.get("abstract_en")),
        "types_dbo": merge_lists(existing["types_dbo_json"], record.get("types_dbo")),
        "wikidata_uri": merge_strings(existing["wikidata_uri"], record.get("wikidata_uri")),
        "sameas_language_uris": merge_lists(existing["sameas_language_uris_json"], record.get("sameas_language_uris")),
        "popularity_score": record.get("popularity_score") if record.get("popularity_score") is not None else existing["popularity_score"],
        "source_artifact": merge_strings(existing["source_artifact"], record.get("source_artifact")),
        "source_lang": merge_strings(existing["source_lang"], record.get("source_lang")),
        "source_version": merge_strings(existing["source_version"], record.get("source_version")),
    }
    conn.execute(
        """
        UPDATE resources
        SET label_hi = ?, label_en = ?, aliases_hi_json = ?, aliases_en_json = ?, redirects_json = ?,
            abstract_hi = ?, abstract_en = ?, types_dbo_json = ?, wikidata_uri = ?,
            sameas_language_uris_json = ?, popularity_score = ?, source_artifact = ?, source_lang = ?, source_version = ?
        WHERE canonical_uri = ?
        """,
        (
            merged["label_hi"],
            merged["label_en"],
            json.dumps(merged["aliases_hi"], ensure_ascii=False),
            json.dumps(merged["aliases_en"], ensure_ascii=False),
            json.dumps(merged["redirects"], ensure_ascii=False),
            merged["abstract_hi"],
            merged["abstract_en"],
            json.dumps(merged["types_dbo"], ensure_ascii=False),
            merged["wikidata_uri"],
            json.dumps(merged["sameas_language_uris"], ensure_ascii=False),
            merged["popularity_score"],
            merged["source_artifact"],
            merged["source_lang"],
            merged["source_version"],
            canonical_uri,
        ),
    )


def reset_final_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM resources;
        DELETE FROM aliases;
        DELETE FROM facts;
        """
    )
    conn.commit()


def load_yaml_or_default(path: Optional[Path], default):
    if path is None or not path.exists() or yaml is None:
        return default
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if payload is not None else default


def iter_tabular_rows(path: Path) -> Iterator[Dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            if isinstance(row, dict):
                yield {str(k): str(v) for k, v in row.items() if k is not None}


def db_fetch_resource(conn: sqlite3.Connection, canonical_uri: str) -> Optional[Dict[str, object]]:
    row = conn.execute(
        "SELECT * FROM resources WHERE canonical_uri = ?",
        (canonical_uri,),
    ).fetchone()
    if row is None:
        return None
    return {
        "canonical_uri": row["canonical_uri"],
        "label_hi": row["label_hi"] or "",
        "label_en": row["label_en"] or "",
        "aliases_hi": parse_list_field(row["aliases_hi_json"]),
        "aliases_en": parse_list_field(row["aliases_en_json"]),
        "redirects": parse_list_field(row["redirects_json"]),
        "abstract_hi": row["abstract_hi"] or "",
        "abstract_en": row["abstract_en"] or "",
        "types_dbo": parse_list_field(row["types_dbo_json"]),
        "wikidata_uri": row["wikidata_uri"] or "",
        "sameas_language_uris": parse_list_field(row["sameas_language_uris_json"]),
        "popularity_score": row["popularity_score"],
        "source_artifact": row["source_artifact"] or "",
        "source_lang": row["source_lang"] or "",
        "source_version": row["source_version"] or "",
    }


def db_fetch_facts(conn: sqlite3.Connection, canonical_uri: str, predicates: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    if predicates:
        placeholders = ",".join("?" for _ in predicates)
        rows = conn.execute(
            f"SELECT * FROM facts WHERE canonical_uri = ? AND predicate IN ({placeholders})",
            [canonical_uri, *predicates],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM facts WHERE canonical_uri = ?",
            (canonical_uri,),
        ).fetchall()
    return [
        {
            "canonical_uri": row["canonical_uri"],
            "predicate": row["predicate"],
            "obj_kind": row["obj_kind"],
            "obj_value": row["obj_value"],
            "lang": row["lang"],
            "datatype": row["datatype"],
        }
        for row in rows
    ]


def db_fetch_candidates(conn: sqlite3.Connection, alias_norm: str, *, limit: int = 25) -> List[Dict[str, object]]:
    rows = conn.execute(
        """
        SELECT a.alias, a.lang, a.canonical_uri, a.match_type,
               r.label_hi, r.label_en, r.types_dbo_json, r.abstract_hi, r.abstract_en,
               r.wikidata_uri, r.sameas_language_uris_json, r.popularity_score
        FROM aliases AS a
        JOIN resources AS r ON r.canonical_uri = a.canonical_uri
        WHERE a.alias_norm = ?
        LIMIT ?
        """,
        (alias_norm, limit),
    ).fetchall()
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "canonical_uri": row["canonical_uri"],
                "alias": row["alias"],
                "lang": row["lang"],
                "match_type": row["match_type"],
                "label_hi": row["label_hi"] or "",
                "label_en": row["label_en"] or "",
                "types_dbo": parse_list_field(row["types_dbo_json"]),
                "abstract_hi": row["abstract_hi"] or "",
                "abstract_en": row["abstract_en"] or "",
                "wikidata_uri": row["wikidata_uri"] or "",
                "sameas_language_uris": parse_list_field(row["sameas_language_uris_json"]),
                "popularity_score": row["popularity_score"] or 0.0,
            }
        )
    return out


def db_fetch_exact_candidates(conn: sqlite3.Connection, alias: str, *, limit: int = 25) -> List[Dict[str, object]]:
    rows = conn.execute(
        """
        SELECT a.alias, a.lang, a.canonical_uri, a.match_type,
               r.label_hi, r.label_en, r.types_dbo_json, r.abstract_hi, r.abstract_en,
               r.wikidata_uri, r.sameas_language_uris_json, r.popularity_score
        FROM aliases AS a
        JOIN resources AS r ON r.canonical_uri = a.canonical_uri
        WHERE a.alias = ?
        LIMIT ?
        """,
        (alias, limit),
    ).fetchall()
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "canonical_uri": row["canonical_uri"],
                "alias": row["alias"],
                "lang": row["lang"],
                "match_type": row["match_type"],
                "label_hi": row["label_hi"] or "",
                "label_en": row["label_en"] or "",
                "types_dbo": parse_list_field(row["types_dbo_json"]),
                "abstract_hi": row["abstract_hi"] or "",
                "abstract_en": row["abstract_en"] or "",
                "wikidata_uri": row["wikidata_uri"] or "",
                "sameas_language_uris": parse_list_field(row["sameas_language_uris_json"]),
                "popularity_score": row["popularity_score"] or 0.0,
            }
        )
    return out


def add_alias_rows(conn: sqlite3.Connection, canonical_uri: str, values: Sequence[str], lang: str, match_type: str) -> None:
    inserted: Set[Tuple[str, str, str]] = set()
    for value in values:
        alias = str(value or "").strip()
        alias_norm = normalize_lookup_text(alias)
        if not alias or not alias_norm:
            continue
        key = (alias_norm, lang, match_type)
        if key in inserted:
            continue
        inserted.add(key)
        conn.execute(
            """
            INSERT INTO aliases (alias_norm, alias, lang, canonical_uri, match_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (alias_norm, alias, lang, canonical_uri, match_type),
        )
