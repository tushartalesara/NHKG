#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a manual review pack for syntax-guided argument refinement."""

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
    parser = argparse.ArgumentParser(description="Create a manual review pack for argument refinement.")
    parser.add_argument("--refinement-json", required=True, help="Refinement cache path")
    parser.add_argument("--sample-size", type=int, default=100, help="Maximum number of events to include")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path")
    return parser


def annotation_schema() -> dict:
    return {
        "schema_version": "nhkg.review-annotation.v1",
        "review_type": "argument_precision",
        "instructions": [
            "Use this pack to judge trigger correctness, frame plausibility, argument-role correctness, and whether hard_review was justified.",
            "Mark event_valid as yes, partial, or no.",
            "Mark trigger_correct and frame_plausible as yes or no.",
            "Mark arguments_correct as all, partial, or none.",
            "Use primary_error from: bad_trigger, wrong_frame, overlapping_roles, span_mismatch, oversplit, other.",
        ],
        "fields": [
            {"name": "event_valid", "allowed_values": ["yes", "partial", "no"]},
            {"name": "trigger_correct", "allowed_values": ["yes", "no"]},
            {"name": "frame_plausible", "allowed_values": ["yes", "no"]},
            {"name": "arguments_correct", "allowed_values": ["all", "partial", "none"]},
            {
                "name": "primary_error",
                "allowed_values": ["bad_trigger", "wrong_frame", "overlapping_roles", "span_mismatch", "oversplit", "other"],
            },
            {"name": "review_notes", "allowed_values": ["free_text"]},
        ],
    }


def priority_rank(priority: str) -> int:
    label = str(priority or "info")
    if label == "hard_review":
        return 0
    if label == "caution":
        return 1
    return 2


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = load_json(Path(args.refinement_json), {})
    events = payload.get("events", []) if isinstance(payload, dict) else []
    review_rows: List[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        arguments = event.get("arguments", {}) or {}
        argument_rows = []
        for role, item in arguments.items():
            if not isinstance(item, dict):
                continue
            argument_rows.append(
                {
                    "role": role,
                    "original_text": item.get("original_text"),
                    "normalized_text": item.get("normalized_text"),
                    "cleaned_text": item.get("cleaned_text"),
                    "confidence": item.get("confidence"),
                    "warnings": item.get("warnings", []),
                    "warning_details": item.get("warning_details", []),
                }
            )
        review_rows.append(
            {
                "event_id": event.get("event_id"),
                "doc_id": event.get("doc_id"),
                "sent_id": event.get("sent_id"),
                "frame": event.get("frame"),
                "event_confidence": event.get("event_confidence"),
                "trigger_confidence": (event.get("trigger", {}) or {}).get("confidence"),
                "review_priority": event.get("review_priority", "info"),
                "manual_review": event.get("manual_review"),
                "warnings": event.get("warnings", []),
                "warning_details": event.get("warning_details", []),
                "trigger_text": (event.get("trigger", {}) or {}).get("original_text"),
                "trigger_head": (event.get("trigger", {}) or {}).get("normalized_head_text"),
                "argument_count": len(arguments),
                "arguments": argument_rows,
                "sentence_text": event.get("sentence_text", ""),
                "event_valid": "",
                "trigger_correct": "",
                "frame_plausible": "",
                "arguments_correct": "",
                "primary_error": "",
                "review_label": "",
                "review_notes": "",
            }
        )
    review_rows.sort(key=lambda row: (priority_rank(str(row.get("review_priority", "info"))), float(row.get("event_confidence", 0.0) or 0.0)))
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
                "event_id",
                "doc_id",
                "sent_id",
                "frame",
                "event_confidence",
                "trigger_confidence",
                "review_priority",
                "manual_review",
                "trigger_text",
                "trigger_head",
                "argument_count",
                "sentence_text",
                "event_valid",
                "trigger_correct",
                "frame_plausible",
                "arguments_correct",
                "primary_error",
                "review_label",
                "review_notes",
            ],
        )
    print(f"[OK] wrote_argument_review_pack={args.out}")
    print(f"[OK] review_items={len(review_rows)}")


if __name__ == "__main__":
    main()
