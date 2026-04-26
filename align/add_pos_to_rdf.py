#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Append POS and optional syntax token nodes to an existing NHKG N-Quads file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    from .enrichment_common import (
        NS,
        clean_uri,
        load_sentence_lines,
        parse_nquad_line,
        parse_sentence_key_from_uri,
        to_nquad_bool,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
        word_uri,
    )
except ImportError:  # pragma: no cover
    from enrichment_common import (
        NS,
        clean_uri,
        load_sentence_lines,
        parse_nquad_line,
        parse_sentence_key_from_uri,
        to_nquad_bool,
        to_nquad_int,
        to_nquad_literal,
        to_nquad_uri,
        word_uri,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append POS-tagged nif:Word nodes and optional dependency arcs to an NHKG graph.",
    )
    parser.add_argument("--input", required=True, help="Input N-Quads, typically output of gold/to_rdf.py")
    parser.add_argument("--pos-json", required=True, help="POS cache from align/annotate_hindi_tokens.py")
    parser.add_argument(
        "--sentences",
        default="",
        help="Optional one-line-per-sentence file to aid sentence-key fallbacks",
    )
    parser.add_argument(
        "--include-syntax",
        action="store_true",
        help="Emit dependencyHead/dependencyRel/isDependencyRoot triples when available",
    )
    parser.add_argument(
        "--out",
        default="output/kg_with_pos.nq",
        help="Output N-Quads with appended nif:Word nodes",
    )
    return parser


def load_sentence_nodes(path: Path) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    sentence_key_to_uri: Dict[str, str] = {}
    graphs: Set[str] = set()
    seen_words: Set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            graphs.add(parsed.graph)
            if parsed.predicate == f"{NS['rdf']}type" and parsed.obj.kind == "uri":
                if parsed.obj.value == f"{NS['nif']}Sentence":
                    key = parse_sentence_key_from_uri(parsed.subject)
                    if key:
                        sentence_key_to_uri[key] = parsed.subject
                elif parsed.obj.value == f"{NS['nif']}Word":
                    seen_words.add(parsed.subject)

    return sentence_key_to_uri, graphs, seen_words


def load_pos_cache(path: Path) -> Dict[str, List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"POS cache not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("POS cache must be a JSON object")

    out: Dict[str, List[dict]] = {}

    if isinstance(payload.get("sentences"), dict):
        for doc_id, sent_map in payload["sentences"].items():
            if not isinstance(sent_map, dict):
                continue
            for sent_id, item in sent_map.items():
                tokens = item.get("tokens", []) if isinstance(item, dict) else item
                if isinstance(tokens, list):
                    out[f"{doc_id}::{sent_id}"] = tokens
        return out

    for key, value in payload.items():
        if key == "meta":
            continue
        if isinstance(value, list):
            out[str(key)] = value
            continue
        if not isinstance(value, dict):
            continue
        for sent_id, item in value.items():
            tokens = item.get("tokens", []) if isinstance(item, dict) else item
            if isinstance(tokens, list):
                out[f"{key}::{sent_id}"] = tokens
    return out


def valid_token(token: dict) -> bool:
    if not isinstance(token, dict):
        return False
    text = str(token.get("text", "")).strip()
    start = token.get("start")
    end = token.get("end")
    if not text or not isinstance(start, int) or not isinstance(end, int):
        return False
    return end >= start


def resolve_pos_tokens(
    sentence_key: str,
    pos_data: Dict[str, List[dict]],
    sentence_count: Optional[int] = None,
) -> List[dict]:
    candidates: List[str] = [sentence_key]
    if "::" in sentence_key:
        doc_id, sent_id = sentence_key.split("::", 1)
        candidates.extend(
            [
                doc_id,
                sent_id,
                f"batch_run::{sent_id}",
                f"doc0::{sent_id}",
            ]
        )
        if sentence_count is not None:
            try:
                sent_index = int(sent_id)
            except ValueError:
                sent_index = None
            if sent_index is not None and 0 <= sent_index < sentence_count:
                candidates.append(f"batch_run::{sent_index}")
    for key in candidates:
        tokens = pos_data.get(key)
        if tokens is not None:
            return tokens
    return []


def token_uri_for_cache_token(sentence_key_value: str, token: dict, fallback_index: int) -> str:
    token_index = token.get("token_id", fallback_index)
    try:
        token_index = int(token_index)
    except (TypeError, ValueError):
        token_index = fallback_index
    return word_uri(sentence_key_value, token_index, int(token["start"]), int(token["end"]))


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input N-Quads not found: {input_path}")

    pos_cache_path = Path(args.pos_json)
    sentence_lines: Optional[Sequence[str]] = None
    if args.sentences:
        sentence_path = Path(args.sentences)
        if not sentence_path.exists():
            raise SystemExit(f"Sentence text file not found: {sentence_path}")
        sentence_lines = load_sentence_lines(sentence_path)

    sentence_map, graphs, existing_words = load_sentence_nodes(input_path)
    if not sentence_map:
        raise SystemExit("No nif:Sentence nodes found in input graph; cannot anchor tokens")

    pos_data = load_pos_cache(pos_cache_path)
    if not pos_data:
        raise SystemExit("POS cache is empty")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    primary_graph = sorted(graphs)[0] if graphs else "http://ns.nhkg.org/graph/gold"
    with input_path.open("r", encoding="utf-8") as handle:
        quads = [line.rstrip("\n") for line in handle]
    seen_quads = set(quads)

    total_tokens = 0
    total_sentences = 0
    matched_sentences = 0
    unmatched_sentences = 0
    syntax_edges = 0
    syntax_roots = 0
    syntax_available = 0

    for sent_key, sent_uri in sorted(sentence_map.items()):
        total_sentences += 1
        tokens = resolve_pos_tokens(
            sent_key,
            pos_data,
            len(sentence_lines) if sentence_lines is not None else None,
        )
        if not tokens:
            unmatched_sentences += 1
            continue

        valid_tokens = [token for token in tokens if valid_token(token)]
        if not valid_tokens:
            unmatched_sentences += 1
            continue

        matched_sentences += 1
        token_uri_lookup: Dict[Tuple[int, int], str] = {}
        for idx, token in enumerate(valid_tokens):
            sent_index = int(token.get("sent_index", 0) or 0)
            word_index = int(token.get("word_index", idx + 1) or (idx + 1))
            token_uri_lookup[(sent_index, word_index)] = token_uri_for_cache_token(sent_key, token, idx)

        for idx, token in enumerate(valid_tokens):
            start = int(token["start"])
            end = int(token["end"])
            text = str(token.get("text", "")).strip()
            if not text:
                continue

            upos = str(token.get("upos", "") or "").strip()
            xpos = str(token.get("xpos", "") or "").strip()
            feats = str(token.get("ufeats", token.get("feats", "")) or "").strip()
            lemma = str(token.get("lemma", "") or text).strip()
            token_uri = token_uri_for_cache_token(sent_key, token, idx)

            if token_uri not in existing_words:
                additions = [
                    to_nquad_uri(token_uri, f"{NS['rdf']}type", f"{NS['nif']}Word", primary_graph),
                    to_nquad_literal(token_uri, f"{NS['nif']}anchorOf", text, primary_graph, lang="hi"),
                    to_nquad_int(token_uri, f"{NS['nif']}beginIndex", start, primary_graph),
                    to_nquad_int(token_uri, f"{NS['nif']}endIndex", end, primary_graph),
                    to_nquad_uri(token_uri, f"{NS['prov']}wasDerivedFrom", sent_uri, primary_graph),
                    to_nquad_uri(token_uri, f"{NS['nif']}referenceContext", sent_uri, primary_graph),
                ]
                if upos:
                    additions.append(to_nquad_literal(token_uri, f"{NS['nhkg']}upos", upos, primary_graph, lang="en"))
                if xpos:
                    additions.append(to_nquad_literal(token_uri, f"{NS['nhkg']}xpos", xpos, primary_graph, lang="en"))
                if feats:
                    additions.append(
                        to_nquad_literal(token_uri, f"{NS['nhkg']}ufeats", feats, primary_graph, lang="en")
                    )
                if lemma:
                    additions.append(to_nquad_literal(token_uri, f"{NS['nhkg']}lemma", lemma, primary_graph, lang="hi"))
                for quad in additions:
                    if quad not in seen_quads:
                        quads.append(quad)
                        seen_quads.add(quad)
                existing_words.add(token_uri)

            if args.include_syntax:
                head = token.get("head")
                deprel = str(token.get("deprel", "") or "").strip()
                sent_index = int(token.get("sent_index", 0) or 0)
                if isinstance(head, int) and head >= 0 and deprel:
                    syntax_available += 1
                    rel_quad = to_nquad_literal(
                        token_uri,
                        f"{NS['nhkg']}dependencyRel",
                        deprel,
                        primary_graph,
                        lang="en",
                    )
                    if rel_quad not in seen_quads:
                        quads.append(rel_quad)
                        seen_quads.add(rel_quad)

                    if head == 0:
                        root_quad = to_nquad_bool(token_uri, f"{NS['nhkg']}isDependencyRoot", True, primary_graph)
                        if root_quad not in seen_quads:
                            quads.append(root_quad)
                            seen_quads.add(root_quad)
                        syntax_roots += 1
                    else:
                        head_uri = token_uri_lookup.get((sent_index, int(head)))
                        if head_uri:
                            head_quad = to_nquad_uri(
                                token_uri,
                                f"{NS['nhkg']}dependencyHead",
                                head_uri,
                                primary_graph,
                            )
                            if head_quad not in seen_quads:
                                quads.append(head_quad)
                                seen_quads.add(head_quad)
                            syntax_edges += 1

            total_tokens += 1

    with out_path.open("w", encoding="utf-8") as handle:
        for quad in quads:
            handle.write(quad + "\n")

    print(f"[OK] Added POS tokens from {pos_cache_path}")
    print(f"[OK] input={input_path}")
    print(f"[OK] output={out_path}")
    print(f"[OK] sentences={total_sentences}, matched={matched_sentences}, unmatched={unmatched_sentences}")
    print(f"[OK] total_tokens={total_tokens}")
    if args.include_syntax:
        print(f"[OK] syntax_tokens={syntax_available}, dependency_edges={syntax_edges}, roots={syntax_roots}")


if __name__ == "__main__":
    main()
