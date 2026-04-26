#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze event density, cap hits, and multi-event warning pressure."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    from align.enrichment_common import collect_sentence_keys, iter_input_events, load_sentence_text_map, sentence_key
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from align.enrichment_common import collect_sentence_keys, iter_input_events, load_sentence_text_map, sentence_key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze event density, cap hits, and multi-event warning pressure.")
    parser.add_argument("--input-jsonl", required=True, help="Extraction JSONL/JSON file")
    parser.add_argument("--sentences", required=True, help="One sentence per line text file used during extraction")
    parser.add_argument("--refinement-json", default="", help="Optional refinement cache for multi-event warning analysis")
    parser.add_argument("--max-events", type=int, default=3, help="Extractor per-sentence max-events cap")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--csv-out", default="", help="Optional CSV export path")
    parser.add_argument("--sample-size", type=int, default=25, help="Number of high-density examples to retain")
    return parser


def load_multi_event_sentences(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}

    out: Dict[str, dict] = {}
    for event in payload.get("events", []) if isinstance(payload.get("events", []), list) else []:
        if not isinstance(event, dict):
            continue
        sent_key_value = str(event.get("sent_key", "")).strip() or sentence_key(event.get("doc_id", "batch_run"), event.get("sent_id", 0))
        out[sent_key_value] = event
    return out


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def build_sentence_texts(input_path: Path, sentence_path: Path) -> Dict[str, str]:
    keys = collect_sentence_keys(input_path)
    sentence_map = load_sentence_text_map(sentence_path, sentence_keys=keys)
    missing = [key for key in keys if not str(sentence_map.get(key, "")).strip()]
    if missing:
        raise SystemExit(
            "Sentence text lookup is incomplete for event-density analysis. "
            f"Missing sentence_text for {len(missing)} extracted sentences; first missing key: {missing[0]}"
        )
    return sentence_map


def collect_sentence_events(input_path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for event in iter_input_events(input_path):
        grouped[sentence_key(event.get("doc_id", "batch_run"), event.get("sent_id", 0))].append(event)
    return grouped


def build_examples(
    grouped_events: Dict[str, List[dict]],
    sentence_texts: Dict[str, str],
    multi_event_sentences: Dict[str, dict],
    sample_size: int,
    max_events: int,
) -> List[dict]:
    examples: List[dict] = []
    for sent_key_value, events in grouped_events.items():
        doc_id, sent_id = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
        event_count = len(events)
        refinement_event = multi_event_sentences.get(sent_key_value, {})
        warning_details = refinement_event.get("warning_details", []) if isinstance(refinement_event, dict) else []
        warning_labels = [
            str(item.get("label", item)) if isinstance(item, dict) else str(item)
            for item in (warning_details or refinement_event.get("warnings", []))
        ] if isinstance(refinement_event, dict) else []
        event_confidences = [float(event.get("event_confidence", 0.0) or 0.0) for event in events if isinstance(event, dict) and event.get("event_confidence") is not None]
        examples.append(
            {
                "doc_id": doc_id,
                "sent_id": sent_id,
                "sent_key": sent_key_value,
                "event_count": event_count,
                "hit_event_cap": event_count >= max_events,
                "at_cap": event_count >= max_events,
                "multi_event_warning": "multi_event_sentence" in warning_labels,
                "sentence_text": sentence_texts.get(sent_key_value, ""),
                "frames": [str((event or {}).get("frame", "")) for event in events if isinstance(event, dict)],
                "event_ids": [str((event or {}).get("event_id", "")) for event in events if isinstance(event, dict)],
                "refinement_warnings": warning_labels,
                "confidence_summary": {
                    "average_event_confidence": round(sum(event_confidences) / len(event_confidences), 4) if event_confidences else None,
                    "event_confidence_count": len(event_confidences),
                },
                "reason_for_review": (
                    "cap_hit_and_multi_event_warning"
                    if event_count >= max_events and "multi_event_sentence" in warning_labels
                    else "cap_hit"
                    if event_count >= max_events
                    else "multi_event_warning"
                    if "multi_event_sentence" in warning_labels
                    else "high_event_density"
                ),
            }
        )
    examples.sort(
        key=lambda item: (
            -int(item.get("event_count", 0) or 0),
            0 if item.get("hit_event_cap") else 1,
            0 if item.get("multi_event_warning") else 1,
            item.get("sent_key", ""),
        )
    )
    return examples[: max(0, sample_size)]


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id",
        "sent_id",
        "sent_key",
        "event_count",
        "hit_event_cap",
        "at_cap",
        "multi_event_warning",
        "sentence_text",
        "frames",
        "event_ids",
        "refinement_warnings",
        "average_event_confidence",
        "reason_for_review",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            confidence_summary = row.get("confidence_summary", {}) or {}
            writer.writerow(
                {
                    "doc_id": row.get("doc_id", ""),
                    "sent_id": row.get("sent_id", ""),
                    "sent_key": row.get("sent_key", ""),
                    "event_count": row.get("event_count", 0),
                    "hit_event_cap": row.get("hit_event_cap", False),
                    "at_cap": row.get("at_cap", False),
                    "multi_event_warning": row.get("multi_event_warning", False),
                    "sentence_text": row.get("sentence_text", ""),
                    "frames": "|".join(row.get("frames", [])),
                    "event_ids": "|".join(row.get("event_ids", [])),
                    "refinement_warnings": "|".join(row.get("refinement_warnings", [])),
                    "average_event_confidence": confidence_summary.get("average_event_confidence", ""),
                    "reason_for_review": row.get("reason_for_review", ""),
                }
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input_jsonl)
    sentence_path = Path(args.sentences)
    refinement_path = Path(args.refinement_json) if args.refinement_json else None

    grouped_events = collect_sentence_events(input_path)
    sentence_texts = build_sentence_texts(input_path, sentence_path)
    multi_event_sentences = load_multi_event_sentences(refinement_path)

    histogram = Counter()
    cap_hits = 0
    counts: Dict[str, int] = {}
    missing_sentence_text = 0
    for sent_key_value, events in grouped_events.items():
        event_count = len(events)
        counts[sent_key_value] = event_count
        histogram[event_count] += 1
        if event_count >= int(args.max_events):
            cap_hits += 1
        if not str(sentence_texts.get(sent_key_value, "")).strip():
            missing_sentence_text += 1

    values = list(counts.values())
    total_sentences = len(values)
    multi_event_warning_count = sum(1 for sent_key_value in counts if sent_key_value in multi_event_sentences)
    examples = build_examples(grouped_events, sentence_texts, multi_event_sentences, args.sample_size, int(args.max_events))

    report = {
        "meta": {
            "schema_version": "nhkg.event-density.v2",
            "stage_version": "1.1.0",
            "max_events": int(args.max_events),
            "input_jsonl": str(input_path.resolve()),
            "sentences": str(sentence_path.resolve()),
            "refinement_json": str(refinement_path.resolve()) if refinement_path and refinement_path.exists() else "",
        },
        "totals": {
            "sentences": total_sentences,
            "events": sum(values),
            "mean_per_sentence": round(statistics.mean(values), 4) if values else 0.0,
            "median_per_sentence": float(statistics.median(values)) if values else 0.0,
            "p95_per_sentence": round(percentile(values, 0.95), 4) if values else 0.0,
            "cap_hits": cap_hits,
            "cap_hit_rate": round(cap_hits / total_sentences, 4) if total_sentences else 0.0,
            "multi_event_warning_rate": round(multi_event_warning_count / total_sentences, 4) if total_sentences else 0.0,
            "missing_sentence_text": missing_sentence_text,
        },
        "histogram": {str(key): int(histogram[key]) for key in sorted(histogram)},
        "examples": examples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.csv_out:
        write_csv(Path(args.csv_out), examples)

    print(f"[OK] wrote_event_density={out_path}")
    print(
        f"[OK] sentences={total_sentences} events={report['totals']['events']} "
        f"mean={report['totals']['mean_per_sentence']:.4f} median={report['totals']['median_per_sentence']:.4f} "
        f"p95={report['totals']['p95_per_sentence']:.4f} cap_hits={cap_hits}"
    )


if __name__ == "__main__":
    main()
