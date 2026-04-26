#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a lightweight manual review pack for canonical entity clustering."""

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
    parser = argparse.ArgumentParser(description="Create a manual review pack for canonical entity clusters.")
    parser.add_argument("--coref-json", required=True, help="Entity cluster cache path")
    parser.add_argument("--sample-size", type=int, default=100, help="Maximum number of clusters to include")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path")
    return parser


def annotation_schema() -> dict:
    return {
        "schema_version": "nhkg.review-annotation.v1",
        "review_type": "coref_unresolved_precision",
        "instructions": [
            "Use this pack to judge whether unresolved pronouns were correctly left unresolved.",
            "For unresolved pronouns, mark unresolved_correct as yes or no.",
            "Mark should_link as yes or no; if yes, fill antecedent_text.",
            "Mark reason_correct as yes or no based on the unresolved reason.",
            "For non-pronoun cluster rows, leave unresolved fields blank when not applicable.",
        ],
        "fields": [
            {"name": "unresolved_correct", "allowed_values": ["yes", "no"]},
            {"name": "should_link", "allowed_values": ["yes", "no"]},
            {"name": "antecedent_text", "allowed_values": ["free_text"]},
            {"name": "reason_correct", "allowed_values": ["yes", "no"]},
            {"name": "review_notes", "allowed_values": ["free_text"]},
        ],
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = load_json(Path(args.coref_json), {})
    documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
    review_rows: List[dict] = []
    for doc_id, item in (documents or {}).items():
        if not isinstance(item, dict):
            continue
        unresolved = item.get("unresolved_mentions", []) or []
        for cluster in item.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            mentions = [mention for mention in (cluster.get("mentions", []) or []) if isinstance(mention, dict)]
            pronoun_members = [mention for mention in mentions if str(mention.get("role_in_cluster", "")) == "pronoun"]
            non_pronoun_members = [mention for mention in mentions if str(mention.get("role_in_cluster", "")) != "pronoun"]
            review_rows.append(
                {
                    "doc_id": doc_id,
                    "cluster_id": cluster.get("cluster_id"),
                    "canonical_text": cluster.get("canonical_text"),
                    "canonical_normalized_text": cluster.get("canonical_normalized_text"),
                    "entity_type": cluster.get("predicted_entity_type"),
                    "confidence": cluster.get("confidence"),
                    "mention_count": len(mentions),
                    "pronoun_member_count": len(pronoun_members),
                    "members": mentions,
                    "pronoun_members": pronoun_members,
                    "non_pronoun_members": non_pronoun_members,
                    "evidence_summary": cluster.get("evidence_summary", []),
                    "conflicts": cluster.get("conflicts", []),
                    "unresolved_mentions_in_doc": unresolved,
                    "item_type": "cluster",
                    "unresolved_correct": "",
                    "should_link": "",
                    "antecedent_text": "",
                    "reason_correct": "",
                    "review_label": "",
                    "review_notes": "",
                }
            )
        for unresolved_item in unresolved:
            if not isinstance(unresolved_item, dict):
                continue
            review_rows.append(
                {
                    "doc_id": doc_id,
                    "cluster_id": "",
                    "canonical_text": "",
                    "canonical_normalized_text": "",
                    "entity_type": "",
                    "confidence": unresolved_item.get("best_score"),
                    "mention_count": 1,
                    "pronoun_member_count": 1,
                    "members": [],
                    "pronoun_members": [],
                    "non_pronoun_members": [],
                    "evidence_summary": unresolved_item.get("candidate_antecedents", []),
                    "conflicts": [],
                    "unresolved_mentions_in_doc": [],
                    "item_type": "unresolved_pronoun",
                    "pronoun_text": unresolved_item.get("text", ""),
                    "sentence_context": unresolved_item.get("sentence_context", ""),
                    "unresolved_reason": unresolved_item.get("reason", ""),
                    "candidate_antecedents": unresolved_item.get("candidate_antecedents", []),
                    "unresolved_correct": "",
                    "should_link": "",
                    "antecedent_text": "",
                    "reason_correct": "",
                    "review_label": "",
                    "review_notes": "",
                }
            )
    review_rows.sort(
        key=lambda row: (
            0 if str(row.get("item_type", "")) == "unresolved_pronoun" else 1,
            0 if int(row.get("pronoun_member_count", 0) or 0) > 0 else 1,
            -int(row.get("mention_count", 0) or 0),
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
                "cluster_id",
                "canonical_text",
                "entity_type",
                "confidence",
                "mention_count",
                "pronoun_member_count",
                "pronoun_text",
                "unresolved_reason",
                "unresolved_correct",
                "should_link",
                "antecedent_text",
                "reason_correct",
                "review_label",
                "review_notes",
            ],
        )
    print(f"[OK] wrote_coref_review_pack={args.out}")
    print(f"[OK] review_items={len(review_rows)}")


if __name__ == "__main__":
    main()
