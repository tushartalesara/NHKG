#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit consistency between extracted event counts and graph event-node counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from align.enrichment_common import NS, iter_input_events, parse_nquad_line, parse_sentence_key_from_uri, sentence_key
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.enrichment_common import NS, iter_input_events, parse_nquad_line, parse_sentence_key_from_uri, sentence_key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit event-count consistency between extracted JSONL and RDF graph outputs.")
    parser.add_argument("--input-jsonl", required=True, help="Extraction JSONL/JSON file")
    parser.add_argument("--graph", required=True, help="Base or final RDF graph in N-Quads")
    parser.add_argument("--event-density-json", default="", help="Optional event-density JSON for cross-checking totals")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--sample-size", type=int, default=25, help="Maximum number of discrepancy samples to retain")
    return parser


def extracted_counts(path: Path) -> tuple[Dict[str, int], int]:
    counts: Dict[str, int] = Counter()
    total = 0
    for event in iter_input_events(path):
        key = sentence_key(event.get("doc_id", "batch_run"), event.get("sent_id", 0))
        counts[key] += 1
        total += 1
    return dict(counts), total


def graph_counts(path: Path) -> tuple[Dict[str, int], int]:
    event_subjects: Set[str] = set()
    event_to_sentence: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_nquad_line(line)
            if parsed is None:
                continue
            local = parsed.subject.rsplit("/", 1)[-1]
            if not local.startswith("Event_"):
                continue
            event_subjects.add(parsed.subject)
            if parsed.predicate == f"{NS['prov']}wasDerivedFrom" and parsed.obj.kind == "uri":
                sent_key_value = parse_sentence_key_from_uri(parsed.obj.value)
                if sent_key_value:
                    event_to_sentence[parsed.subject] = sent_key_value
    counts: Dict[str, int] = Counter()
    unmapped = 0
    for subject in event_subjects:
        sent_key_value = event_to_sentence.get(subject)
        if sent_key_value:
            counts[sent_key_value] += 1
        else:
            unmapped += 1
    counts["__unmapped__"] = unmapped
    return dict(counts), len(event_subjects)


def load_event_density_total(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    totals = payload.get("totals", {})
    if not isinstance(totals, dict):
        return None
    try:
        return int(totals.get("events", 0))
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = build_arg_parser().parse_args()
    extraction_path = Path(args.input_jsonl)
    graph_path = Path(args.graph)
    density_path = Path(args.event_density_json) if args.event_density_json else None

    extracted_by_sentence, extracted_total = extracted_counts(extraction_path)
    graph_by_sentence, graph_total = graph_counts(graph_path)
    density_total = load_event_density_total(density_path)

    all_keys = sorted((set(extracted_by_sentence) | set(graph_by_sentence)) - {"__unmapped__"})
    mismatch_counter = Counter()
    discrepancy_rows: List[dict] = []
    for key in all_keys:
        extracted_count = int(extracted_by_sentence.get(key, 0))
        graph_count = int(graph_by_sentence.get(key, 0))
        if extracted_count == graph_count:
            continue
        reason = "graph_missing_events" if extracted_count > graph_count else "graph_has_extra_events"
        mismatch_counter[reason] += 1
        discrepancy_rows.append(
            {
                "sent_key": key,
                "extracted_event_count": extracted_count,
                "graph_event_node_count": graph_count,
                "difference": graph_count - extracted_count,
                "reason": reason,
            }
        )
    discrepancy_rows.sort(key=lambda row: (-abs(int(row["difference"])), row["sent_key"]))

    report = {
        "meta": {
            "schema_version": "nhkg.event-count-audit.v1",
            "stage_version": "1.0.0",
            "input_jsonl": str(extraction_path.resolve()),
            "graph": str(graph_path.resolve()),
            "event_density_json": str(density_path.resolve()) if density_path and density_path.exists() else "",
        },
        "counts": {
            "extracted_event_count": extracted_total,
            "graph_event_node_count": graph_total,
            "graph_unmapped_event_nodes": int(graph_by_sentence.get("__unmapped__", 0) or 0),
            "event_density_total": density_total,
            "mismatch_count": len(discrepancy_rows),
        },
        "mismatch_reasons": dict(sorted(mismatch_counter.items())),
        "per_sentence_extracted_counts": extracted_by_sentence,
        "per_sentence_graph_counts": {key: value for key, value in graph_by_sentence.items() if key != "__unmapped__"},
        "sample_discrepancies": discrepancy_rows[: max(0, args.sample_size)],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"[OK] wrote_event_count_audit={out_path}")
    print(
        f"[OK] extracted_events={extracted_total} graph_event_nodes={graph_total} "
        f"mismatches={len(discrepancy_rows)} unmapped_graph_events={int(graph_by_sentence.get('__unmapped__', 0) or 0)}"
    )


if __name__ == "__main__":
    main()
