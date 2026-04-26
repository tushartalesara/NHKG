#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a temporal review pack for timexes and event-event relations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from align.internal_quality_common import load_json, save_json, write_flat_csv
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.internal_quality_common import load_json, save_json, write_flat_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a manual review pack for temporal enrichment.")
    parser.add_argument("--time-json", required=True, help="Temporal cache path")
    parser.add_argument("--sample-size", type=int, default=100, help="Maximum number of items to include")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path")
    return parser


def annotation_schema() -> dict:
    return {
        "schema_version": "nhkg.review-annotation.v1",
        "review_type": "temporal_precision",
        "instructions": [
            "Use this pack to judge timex span correctness, normalization correctness, event-time links, relation labels, and evidence strength.",
            "For timex items, mark timex_span_correct and normalized_value_correct.",
            "Use normalized_value_correct values yes, no, unresolved_ok, or n.a.",
            "For relation items, mark relation_type_correct and evidence_strength_appropriate.",
            "Mark event_time_link_correct whenever the linked relation or event-time choice is justified.",
        ],
        "fields": [
            {"name": "timex_span_correct", "allowed_values": ["yes", "no", "n.a."]},
            {"name": "normalized_value_correct", "allowed_values": ["yes", "no", "unresolved_ok", "n.a."]},
            {"name": "event_time_link_correct", "allowed_values": ["yes", "no", "n.a."]},
            {"name": "relation_type_correct", "allowed_values": ["yes", "no", "n.a."]},
            {"name": "evidence_strength_appropriate", "allowed_values": ["yes", "no", "n.a."]},
            {"name": "review_notes", "allowed_values": ["free_text"]},
        ],
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = load_json(Path(args.time_json), {})
    review_rows: List[dict] = []
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            sent_key = f"{doc_id}::{sent_id}"
            sentence_text = item.get("text", "")
            for timex in item.get("timexes", []):
                if not isinstance(timex, dict):
                    continue
                review_rows.append(
                    {
                        "item_type": "timex",
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "sent_key": sent_key,
                        "sentence_text": sentence_text,
                        "text": timex.get("text"),
                        "normalized_value": timex.get("value"),
                        "timex_type": timex.get("type"),
                        "resolved": timex.get("resolution_status") == "resolved",
                        "resolution_status": timex.get("resolution_status"),
                        "unresolved_reason": "|".join(timex.get("unresolved_reasons", [])) if isinstance(timex.get("unresolved_reasons", []), list) else "",
                        "linked_events": "|".join(str(link.get("event_id", "")) for link in timex.get("linked_events", []) if isinstance(link, dict)),
                        "confidence": timex.get("confidence"),
                        "strategy": timex.get("engine"),
                        "evidence_strength": timex.get("resolution_status"),
                        "supporting_cue": "|".join(timex.get("notes", [])),
                        "reason_chosen": "|".join(timex.get("notes", [])),
                        "timex_span_correct": "",
                        "normalized_value_correct": "",
                        "event_time_link_correct": "",
                        "relation_type_correct": "",
                        "evidence_strength_appropriate": "",
                        "review_label": "",
                        "review_notes": "",
                    }
                )
            for relation in item.get("event_event_relations", []):
                if not isinstance(relation, dict):
                    continue
                review_rows.append(
                    {
                        "item_type": "event_event_relation",
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "sent_key": sent_key,
                        "sentence_text": sentence_text,
                        "source_event_id": relation.get("source_event_id"),
                        "target_event_id": relation.get("target_event_id"),
                        "text": f"{relation.get('source_event_id')} -> {relation.get('target_event_id')}",
                        "normalized_value": relation.get("relation"),
                        "timex_type": "RELATION",
                        "confidence": relation.get("confidence"),
                        "strategy": relation.get("strategy"),
                        "evidence_strength": relation.get("evidence_strength", ""),
                        "supporting_cue": relation.get("evidence", ""),
                        "reason_chosen": relation.get("reason", relation.get("evidence", "")),
                        "timex_span_correct": "",
                        "normalized_value_correct": "",
                        "event_time_link_correct": "",
                        "relation_type_correct": "",
                        "evidence_strength_appropriate": "",
                        "review_label": "",
                        "review_notes": "",
                    }
                )
    review_rows.sort(
        key=lambda row: (
            0 if row.get("item_type") == "event_event_relation" else 1,
            0 if str(row.get("evidence_strength", "")).lower() in {"weak", "shared_time_anchor"} else 1,
            float(row.get("confidence", 0.0) or 0.0),
        )
    )
    review_rows = review_rows[: max(0, args.sample_size)]
    save_json(
        Path(args.out),
        {
            "meta": {
                **annotation_schema(),
                "sample_size": len(review_rows),
            },
            "items": review_rows,
        },
    )
    if args.csv_out:
        write_flat_csv(
            Path(args.csv_out),
            review_rows,
            [
                "item_type",
                "doc_id",
                "sent_id",
                "sent_key",
                "source_event_id",
                "target_event_id",
                "text",
                "normalized_value",
                "timex_type",
                "resolved",
                "resolution_status",
                "unresolved_reason",
                "linked_events",
                "confidence",
                "strategy",
                "evidence_strength",
                "supporting_cue",
                "reason_chosen",
                "timex_span_correct",
                "normalized_value_correct",
                "event_time_link_correct",
                "relation_type_correct",
                "evidence_strength_appropriate",
                "review_label",
                "review_notes",
            ],
        )
    print(f"[OK] wrote_temporal_review_pack={args.out}")
    print(f"[OK] review_items={len(review_rows)}")


if __name__ == "__main__":
    main()
