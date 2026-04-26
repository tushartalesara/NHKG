#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create manual review packs for DBpedia linking outcomes."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create review packs for DBpedia links and misses.")
    parser.add_argument("--dbpedia-json", required=True, help="Link cache from align/link_dbpedia.py")
    parser.add_argument("--sentences", default="", help="Optional source sentence file for readable review context")
    parser.add_argument("--sample-size", type=int, default=50, help="Max items per bucket")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible sampling")
    parser.add_argument("--out", required=True, help="Combined output JSON review pack")
    return parser


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object at: {path}")
    return payload


def load_sentences(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def collect_rows(payload: dict) -> tuple[List[dict], List[dict]]:
    links: List[dict] = []
    mentions: List[dict] = []
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            sentence_key = f"{doc_id}::{sent_id}"
            for row in item.get("links", []):
                if isinstance(row, dict):
                    item_copy = dict(row)
                    item_copy.setdefault("sent_key", sentence_key)
                    links.append(item_copy)
            for row in item.get("mentions", []):
                if isinstance(row, dict):
                    item_copy = dict(row)
                    item_copy.setdefault("sent_key", sentence_key)
                    mentions.append(item_copy)
    return links, mentions


def sentence_text(sentences: List[str], sent_key_value: str) -> str:
    sent_id = sent_key_value.split("::", 1)[1] if "::" in sent_key_value else "0"
    try:
        index = int(sent_id)
    except ValueError:
        return ""
    if 0 <= index < len(sentences):
        return sentences[index]
    return ""


def to_review_item(row: dict, sentences: List[str]) -> dict:
    return {
        "sent_key": row.get("sent_key", ""),
        "sentence": sentence_text(sentences, str(row.get("sent_key", ""))),
        "mention": row.get("text", row.get("raw_text", "")),
        "mention_source": row.get("mention_source", ""),
        "mention_uri": row.get("mention_uri", ""),
        "predicted_uri": row.get("canonical_uri", row.get("predicted_uri", "")),
        "score": row.get("score", 0.0),
        "confidence": row.get("confidence", 0.0),
        "matched_via": row.get("matched_via", ""),
        "matched_label": row.get("matched_label", ""),
        "matched_lang": row.get("matched_lang", ""),
        "ner_label": row.get("ner_label", ""),
        "role": row.get("role", ""),
        "predicted_dbo_types": row.get("predicted_dbo_types", []),
        "top_candidates": row.get("top_candidates", []),
        "why_selected": row.get("why_selected", ""),
        "no_link_reason": row.get("no_link_reason", ""),
        "likely_linkable": row.get("likely_linkable", False),
        "review_label": "",
        "review_notes": "",
    }


def stratified_sample(rows: List[dict], sample_size: int, seed: int, key_name: str) -> List[dict]:
    if len(rows) <= sample_size:
        return rows
    random.seed(seed)
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_name, "unknown"))
        buckets[key].append(row)

    sampled: List[dict] = []
    keys = sorted(buckets)
    if keys:
        quota = max(1, sample_size // len(keys))
        for key in keys:
            pool = list(buckets[key])
            random.shuffle(pool)
            sampled.extend(pool[:quota])

    remaining = [row for row in rows if row not in sampled]
    random.shuffle(remaining)
    if len(sampled) < sample_size:
        sampled.extend(remaining[: sample_size - len(sampled)])
    return sampled[:sample_size]


def write_bucket(path: Path, items: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"items": items}, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = build_arg_parser().parse_args()
    cache_path = Path(args.dbpedia_json)
    if not cache_path.exists():
        raise SystemExit(f"DBpedia link cache not found: {cache_path}")

    payload = load_json(cache_path)
    links, mentions = collect_rows(payload)
    sentences = load_sentences(Path(args.sentences)) if args.sentences else []

    linked_items = [to_review_item(row, sentences) for row in links]
    missed_but_linkable = [to_review_item(row, sentences) for row in mentions if row.get("status") != "linked" and row.get("likely_linkable")]
    skipped_correctly = [to_review_item(row, sentences) for row in mentions if row.get("status") == "skipped" and not row.get("likely_linkable")]
    low_confidence = [to_review_item(row, sentences) for row in mentions if row.get("no_link_reason") == "low_confidence"]

    linked_sample = stratified_sample(linked_items, args.sample_size, args.seed, "matched_via")
    missed_sample = stratified_sample(missed_but_linkable, args.sample_size, args.seed, "no_link_reason")
    skipped_sample = stratified_sample(skipped_correctly, args.sample_size, args.seed, "no_link_reason")
    low_conf_sample = stratified_sample(low_confidence, args.sample_size, args.seed, "no_link_reason")

    report = {
        "meta": {
            "dbpedia_cache": str(cache_path.resolve()),
            "total_links": len(links),
            "total_mentions": len(mentions),
            "seed": args.seed,
        },
        "linked_mentions_sample": linked_sample,
        "missed_but_linkable_sample": missed_sample,
        "skipped_correctly_sample": skipped_sample,
        "low_confidence_candidates_sample": low_conf_sample,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    write_bucket(out_path.parent / "linked_mentions_sample.json", linked_sample)
    write_bucket(out_path.parent / "missed_but_linkable_sample.json", missed_sample)
    write_bucket(out_path.parent / "skipped_correctly_sample.json", skipped_sample)
    write_bucket(out_path.parent / "low_confidence_candidates_sample.json", low_conf_sample)

    print(f"[OK] wrote_dbpedia_audit={out_path}")
    print(
        f"[OK] linked_sample={len(linked_sample)} "
        f"missed_but_linkable_sample={len(missed_sample)} "
        f"skipped_correctly_sample={len(skipped_sample)} "
        f"low_confidence_sample={len(low_conf_sample)}"
    )


if __name__ == "__main__":
    main()
