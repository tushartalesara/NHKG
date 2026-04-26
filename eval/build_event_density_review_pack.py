#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a focused review pack for high-density and cap-hit event sentences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

try:
    from align.internal_quality_common import save_json, write_flat_csv
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.internal_quality_common import save_json, write_flat_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a manual review pack for event-density pressure cases.")
    parser.add_argument("--event-density-json", required=True, help="JSON emitted by eval/analyze_event_density.py")
    parser.add_argument("--sample-size", type=int, default=100, help="Maximum number of examples to keep")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path")
    return parser


def annotation_schema() -> dict:
    return {
        "schema_version": "nhkg.review-annotation.v1",
        "review_type": "event_density_oversplitting",
        "instructions": [
            "Use this pack to judge whether the event count per sentence is acceptable.",
            "Mark event_count_acceptable as yes or no.",
            "Enter recommended_event_count as an integer when you think the system oversplit the sentence.",
            "Mark oversplit as yes or no.",
            "Use reason from: copular_overgeneration, adjectival_overgeneration, trigger_duplication, multiword_duplication, other.",
        ],
        "fields": [
            {"name": "event_count_acceptable", "allowed_values": ["yes", "no"]},
            {"name": "recommended_event_count", "allowed_values": ["integer"]},
            {"name": "oversplit", "allowed_values": ["yes", "no"]},
            {
                "name": "reason",
                "allowed_values": [
                    "copular_overgeneration",
                    "adjectival_overgeneration",
                    "trigger_duplication",
                    "multiword_duplication",
                    "other",
                ],
            },
            {"name": "review_notes", "allowed_values": ["free_text"]},
        ],
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    with Path(args.event_density_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    review_rows: List[dict] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        review_rows.append(
            {
                "doc_id": example.get("doc_id"),
                "sent_id": example.get("sent_id"),
                "sent_key": example.get("sent_key"),
                "event_count": example.get("event_count"),
                "hit_event_cap": example.get("hit_event_cap"),
                "at_cap": example.get("at_cap", example.get("hit_event_cap")),
                "multi_event_warning": example.get("multi_event_warning"),
                "frames": example.get("frames", []),
                "event_ids": example.get("event_ids", []),
                "sentence_text": example.get("sentence_text", ""),
                "refinement_warnings": example.get("refinement_warnings", []),
                "confidence_summary": example.get("confidence_summary", {}),
                "reason_for_review": example.get("reason_for_review", ""),
                "event_count_acceptable": "",
                "recommended_event_count": "",
                "oversplit": "",
                "reason": "",
                "review_label": "",
                "review_notes": "",
            }
        )
    review_rows.sort(
        key=lambda row: (
            0 if row.get("hit_event_cap") else 1,
            0 if row.get("multi_event_warning") else 1,
            -int(row.get("event_count", 0) or 0),
            row.get("sent_key", ""),
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
                "doc_id",
                "sent_id",
                "sent_key",
                "event_count",
                "hit_event_cap",
                "at_cap",
                "multi_event_warning",
                "sentence_text",
                "reason_for_review",
                "event_count_acceptable",
                "recommended_event_count",
                "oversplit",
                "reason",
                "review_label",
                "review_notes",
            ],
        )
    print(f"[OK] wrote_event_density_review_pack={args.out}")
    print(f"[OK] review_items={len(review_rows)}")


if __name__ == "__main__":
    main()
