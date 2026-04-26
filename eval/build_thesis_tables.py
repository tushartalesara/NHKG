#!/usr/bin/env python
"""
Build thesis-ready evaluation tables from a frozen NHKG evidence bundle.

This script is intentionally standalone and uses only the Python standard
library so it can be run directly from the repository checkout or from a copied
submission package.

Default inputs:
    thesis_submission/final_evidence_bundle_v1
    eval/annotation/event_trigger_slice_v11.csv
    eval/annotation/no_event_slice_v11.csv
    eval/annotation/silver_v1/task_a_emitted_event_pack_v1_adjudicated.jsonl
    eval/annotation/silver_v1/task_b_recover_now_no_event_pack_v1_audit.jsonl

Default output:
    eval/summary/final_thesis_tables

The script supports two modes at once:
1. Bundle summaries from frozen run artifacts.
2. Annotation-progress and label-summary tables from the prepared CSV slices.
3. Silver semantic evaluation summaries from the adjudicated Task A and Task B
   audit packs.

If the annotation CSVs are still mostly blank, the script reports progress and
schema coverage instead of pretending to compute final evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence


GENERIC_FAMILIES = {"xapana", "kara", "ho", "xe", "ja"}
EVENT_LABEL_COLUMNS = [
    "gold_eventhood",
    "gold_trigger_span_text",
    "gold_trigger_head_text",
    "system_trigger_ok",
    "system_predicate_center_ok",
    "frame_specificity_ok",
    "argument_minimal_ok",
    "adjudication_status",
]
NO_EVENT_LABEL_COLUMNS = [
    "gold_eventhood",
    "gold_num_events",
    "gold_min_trigger_text",
    "gold_trigger_head_text",
    "policy_bucket",
    "adjudication_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build thesis tables from a frozen NHKG evidence bundle."
    )
    parser.add_argument(
        "--bundle-dir",
        default="thesis_submission/final_evidence_bundle_v1",
        help="Path to the frozen thesis evidence bundle.",
    )
    parser.add_argument(
        "--event-annotation",
        default="eval/annotation/event_trigger_slice_v11.csv",
        help="Path to the event-trigger annotation CSV.",
    )
    parser.add_argument(
        "--no-event-annotation",
        default="eval/annotation/no_event_slice_v11.csv",
        help="Path to the no-event annotation CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="eval/summary/final_thesis_tables",
        help="Directory where thesis tables should be written.",
    )
    parser.add_argument(
        "--silver-task-a",
        default="eval/annotation/silver_v1/task_a_emitted_event_pack_v1_adjudicated.jsonl",
        help="Path to the adjudicated silver Task A JSONL.",
    )
    parser.add_argument(
        "--silver-task-b",
        default="eval/annotation/silver_v1/task_b_recover_now_no_event_pack_v1_audit.jsonl",
        help="Path to the silver Task B recover-now audit JSONL.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[MutableMapping[str, object]]:
    if not path.exists():
        return []
    rows: List[MutableMapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return default


def as_rate(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def format_rate(numerator: int, denominator: int) -> str:
    return f"{100.0 * as_rate(numerator, denominator):.2f}%"


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def family_of(frame: object) -> str:
    return str(frame or "").split(".", 1)[0]


def get_event_id(obj: Mapping[str, object]) -> str:
    return str(obj.get("event_id") or obj.get("id") or obj.get("eventId") or "")


def get_sent_id(obj: Mapping[str, object]) -> str:
    value = obj.get("sent_id")
    if value is None:
        value = obj.get("sentence_id")
    return str(value if value is not None else "")


def get_doc_id(obj: Mapping[str, object]) -> str:
    return str(obj.get("doc_id") or obj.get("document_id") or "")


def event_key(obj: Mapping[str, object]) -> str:
    return "||".join((get_doc_id(obj), get_sent_id(obj), get_event_id(obj)))


def iter_refinement_events(refinement: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    events = refinement.get("events")
    if isinstance(events, list):
        yield from events
        return
    refined_events = refinement.get("refined_events")
    if isinstance(refined_events, list):
        yield from refined_events


def detect_input_sentence_count(manifest: Mapping[str, object], fallback: int) -> int:
    inputs = manifest.get("inputs")
    if isinstance(inputs, Mapping):
        for key in ("sentence_count", "input_sentence_count", "input_count", "num_sentences"):
            value = inputs.get(key)
            if isinstance(value, int):
                return value
    for key in ("input_sentence_count", "input_count", "sentence_count", "num_sentences"):
        value = manifest.get(key)
        if isinstance(value, int):
            return value
    stages = manifest.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            for key in ("input_sentence_count", "input_count", "sentence_count", "num_sentences"):
                value = stage.get(key)
                if isinstance(value, int):
                    return value
    return fallback


def bundle_paths(bundle_dir: Path) -> Dict[str, Path]:
    return {
        "manifest": bundle_dir / "artifacts" / "manifest.json",
        "validation": bundle_dir / "artifacts" / "stats" / "canonical_full_graph.validation.json",
        "compute": bundle_dir / "artifacts" / "stats" / "compute_enrichment_stats.json",
        "audit": bundle_dir / "artifacts" / "stats" / "event_count_audit.json",
        "density": bundle_dir / "artifacts" / "stats" / "event_density.analysis.json",
        "extraction": bundle_dir / "artifacts" / "extraction" / "canonical_extraction.jsonl",
        "refinement": bundle_dir / "artifacts" / "refinement" / "refinement_cache.json",
    }


def summarize_bundle(bundle_dir: Path) -> Dict[str, object]:
    paths = bundle_paths(bundle_dir)
    manifest = load_json(paths["manifest"])
    validation = load_json(paths["validation"])
    compute = load_json(paths["compute"])
    audit = load_json(paths["audit"])
    density = load_json(paths["density"])
    refinement = load_json(paths["refinement"])
    extraction_rows = load_jsonl(paths["extraction"])

    input_count = detect_input_sentence_count(manifest, fallback=max(len(extraction_rows), 1))
    density_totals = density.get("totals") if isinstance(density.get("totals"), Mapping) else {}
    audit_counts = audit.get("counts") if isinstance(audit.get("counts"), Mapping) else {}
    custom_validation = (
        validation.get("custom_validation")
        if isinstance(validation.get("custom_validation"), Mapping)
        else {}
    )
    parse_validation = (
        validation.get("parse_validation")
        if isinstance(validation.get("parse_validation"), Mapping)
        else {}
    )
    shacl_validation = (
        validation.get("shacl_validation")
        if isinstance(validation.get("shacl_validation"), Mapping)
        else {}
    )

    event_bearing_sentences = as_int(
        density.get("num_sentences")
        or density.get("event_bearing_sentences")
        or density_totals.get("sentences")
    )
    extracted_events = as_int(
        density.get("total_events")
        or density.get("events")
        or density_totals.get("events")
        or audit.get("extracted_event_count")
        or audit_counts.get("extracted_event_count")
        or len(extraction_rows)
    )

    refinement_by_key = {
        event_key(row): row for row in iter_refinement_events(refinement)
    }

    family_counter: Counter[str] = Counter()
    trigger_category_counter: Counter[str] = Counter()
    review_priority_counter: Counter[str] = Counter()
    mismatch_count = 0

    for row in extraction_rows:
        frame_family = family_of(row.get("frame"))
        family_counter[frame_family] += 1

        trigger = row.get("trigger")
        category = ""
        if isinstance(trigger, Mapping):
            category = str(trigger.get("category") or "").strip().lower()
        trigger_category_counter[category or "<empty>"] += 1

        meta = row.get("meta")
        if isinstance(meta, Mapping):
            candidate_acceptance = meta.get("candidate_acceptance")
            if isinstance(candidate_acceptance, Mapping):
                if candidate_acceptance.get("final_trigger_matches_predicate_center") is False:
                    mismatch_count += 1

        ref = refinement_by_key.get(event_key(row))
        if isinstance(ref, Mapping):
            review_priority_counter[str(ref.get("review_priority") or "unknown")] += 1

    warning_counter = Counter(refinement.get("warning_counter") or {})
    if not warning_counter:
        refinement_summary = (
            refinement.get("summary")
            if isinstance(refinement.get("summary"), Mapping)
            else {}
        )
        if isinstance(refinement_summary.get("warning_counts"), Mapping):
            warning_counter = Counter(refinement_summary.get("warning_counts") or {})
    if not warning_counter:
        for ref_row in refinement_by_key.values():
            if isinstance(ref_row, Mapping):
                for warning in ref_row.get("warnings") or []:
                    warning_counter[str(warning)] += 1

    refined_event_count = refinement.get("events")
    if isinstance(refined_event_count, list):
        refined_event_count = len(refined_event_count)
    refined_event_count = as_int(refined_event_count, default=len(refinement_by_key))
    if not refined_event_count:
        refined_event_count = len(extraction_rows)

    compute_argument_refinement = (
        compute.get("argument_refinement")
        if isinstance(compute.get("argument_refinement"), Mapping)
        else {}
    )
    compute_refinement_quality = (
        compute.get("refinement_quality")
        if isinstance(compute.get("refinement_quality"), Mapping)
        else {}
    )

    generic_total = sum(
        count for family, count in family_counter.items() if family.lower() in GENERIC_FAMILIES
    )

    summary: Dict[str, object] = {
        "bundle_dir": str(bundle_dir),
        "run_name": manifest.get("run_name"),
        "run_id": manifest.get("run_id"),
        "input_sentences": input_count,
        "event_bearing_sentences": event_bearing_sentences,
        "event_bearing_sentence_rate": as_rate(event_bearing_sentences, input_count),
        "extracted_events": extracted_events,
        "events_per_input_sentence": extracted_events / float(input_count or 1),
        "validation": {
            "overall_ok": validation.get("overall_ok"),
            "parse_ok": validation.get("parse_ok")
            if validation.get("parse_ok") is not None
            else parse_validation.get("parse_ok"),
            "shacl_conforms": validation.get("shacl_conforms")
            if validation.get("shacl_conforms") is not None
            else shacl_validation.get("conforms"),
        },
        "count_reconciliation": {
            "extracted_event_count": audit.get("extracted_event_count")
            if audit.get("extracted_event_count") is not None
            else audit_counts.get("extracted_event_count"),
            "graph_event_node_count": audit.get("graph_event_node_count")
            if audit.get("graph_event_node_count") is not None
            else audit_counts.get("graph_event_node_count"),
            "event_density_total": audit.get("event_density_total")
            if audit.get("event_density_total") is not None
            else audit_counts.get("event_density_total"),
            "mismatch_count": audit.get("mismatch_count")
            if audit.get("mismatch_count") is not None
            else audit_counts.get("mismatch_count"),
        },
        "density": {
            "mean": density.get("mean_events_per_sentence")
            or density.get("mean_events_per_event_bearing_sentence")
            or density.get("mean")
            or density_totals.get("mean_per_sentence"),
            "median": density.get("median_events_per_sentence")
            or density.get("median")
            or density_totals.get("median_per_sentence"),
            "p95": density.get("p95_events_per_sentence")
            or density.get("p95")
            or density_totals.get("p95_per_sentence"),
            "histogram": density.get("histogram") or {},
            "cap_hits": density.get("cap_hits", density_totals.get("cap_hits", 0)),
        },
        "extraction": {
            "row_count": len(extraction_rows),
            "trigger_predicate_center_mismatch_count": mismatch_count,
            "trigger_predicate_center_mismatch_rate": as_rate(mismatch_count, len(extraction_rows)),
            "trigger_category_distribution": dict(trigger_category_counter.most_common()),
            "frame_family_distribution": dict(family_counter.most_common()),
            "generic_family_count": generic_total,
            "generic_family_share": as_rate(generic_total, len(extraction_rows)),
        },
        "refinement": {
            "processed_events": refined_event_count,
            "skipped_events": refinement.get("skipped_events", 0),
            "event_failures": refinement.get("event_failures", 0),
            "review_priority_distribution": dict(review_priority_counter.most_common()),
            "warning_distribution": dict(warning_counter.most_common()),
        },
        "compute": {
            "candidates_retrieved": compute.get("candidates_retrieved"),
            "final_candidates_emitted": compute.get("final_candidates_emitted"),
            "candidates_collapsed_as_duplicates": compute.get("candidates_collapsed_as_duplicates"),
            "candidates_downranked_as_helper_like": compute.get("candidates_downranked_as_helper_like"),
            "candidates_kept_as_lexical_predicate_centers": compute.get(
                "candidates_kept_as_lexical_predicate_centers"
            ),
            "average_event_confidence": compute.get("avg_event_confidence")
            or compute.get("average_event_confidence")
            or compute_argument_refinement.get("average_event_confidence")
            or compute_refinement_quality.get("average_event_confidence"),
            "hard_review_rate": compute.get("hard_review_rate")
            or compute_argument_refinement.get("hard_review_rate")
            or compute_refinement_quality.get("hard_review_rate"),
            "caution_rate": compute.get("caution_rate")
            or compute_argument_refinement.get("caution_rate")
            or compute_refinement_quality.get("caution_rate"),
            "info_only_rate": compute.get("info_only_rate")
            or compute_argument_refinement.get("info_only_rate")
            or compute_refinement_quality.get("info_only_rate"),
        },
    }
    return summary


def summarize_annotation_rows(
    rows: Sequence[Mapping[str, str]], label_columns: Sequence[str]
) -> Dict[str, object]:
    if not rows:
        return {
            "rows_total": 0,
            "rows_with_any_label": 0,
            "label_completion": {},
            "label_distributions": {},
        }

    completion: Dict[str, int] = {}
    distributions: Dict[str, Dict[str, int]] = {}
    rows_with_any_label = 0

    for column in label_columns:
        counter: Counter[str] = Counter()
        filled = 0
        for row in rows:
            value = str(row.get(column) or "").strip()
            if value:
                filled += 1
                counter[value] += 1
        completion[column] = filled
        distributions[column] = dict(counter.most_common())

    for row in rows:
        if any(str(row.get(column) or "").strip() for column in label_columns):
            rows_with_any_label += 1

    return {
        "rows_total": len(rows),
        "rows_with_any_label": rows_with_any_label,
        "row_completion_rate": as_rate(rows_with_any_label, len(rows)),
        "label_completion": completion,
        "label_distributions": distributions,
    }


def summarize_silver_task_a(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "rows_total": 0,
            "event_real_distribution": {},
            "trigger_correct_distribution": {},
            "frame_plausible_distribution": {},
            "arguments_usable_distribution": {},
            "duplicate_or_alternate_distribution": {},
            "overall_quality_distribution": {},
            "final_error_tag_distribution": {},
        }

    def count_field(name: str) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for row in rows:
            counter[str(row.get(name) or "").strip() or "<empty>"] += 1
        return dict(counter.most_common())

    tag_counter: Counter[str] = Counter()
    for row in rows:
        tags = row.get("final_error_tags")
        if isinstance(tags, list):
            for tag in tags:
                norm = str(tag or "").strip()
                if norm:
                    tag_counter[norm] += 1

    total = len(rows)
    event_real_distribution = count_field("final_event_real")
    trigger_correct_distribution = count_field("final_trigger_correct")
    frame_plausible_distribution = count_field("final_frame_plausible")
    arguments_usable_distribution = count_field("final_arguments_usable")
    duplicate_distribution = count_field("final_duplicate_or_alternate")
    overall_quality_distribution = count_field("final_overall_quality")

    return {
        "rows_total": total,
        "event_real_distribution": event_real_distribution,
        "trigger_correct_distribution": trigger_correct_distribution,
        "frame_plausible_distribution": frame_plausible_distribution,
        "arguments_usable_distribution": arguments_usable_distribution,
        "duplicate_or_alternate_distribution": duplicate_distribution,
        "overall_quality_distribution": overall_quality_distribution,
        "final_error_tag_distribution": dict(tag_counter.most_common()),
        "event_real_rate": {
            label: as_rate(count, total) for label, count in event_real_distribution.items()
        },
        "trigger_correct_rate": {
            label: as_rate(count, total)
            for label, count in trigger_correct_distribution.items()
        },
        "frame_plausible_rate": {
            label: as_rate(count, total)
            for label, count in frame_plausible_distribution.items()
        },
        "arguments_usable_rate": {
            label: as_rate(count, total)
            for label, count in arguments_usable_distribution.items()
        },
        "duplicate_or_alternate_rate": {
            label: as_rate(count, total) for label, count in duplicate_distribution.items()
        },
        "overall_quality_rate": {
            label: as_rate(count, total)
            for label, count in overall_quality_distribution.items()
        },
    }


def summarize_silver_task_b(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "rows_total": 0,
            "should_have_event_distribution": {},
            "recommended_event_type_distribution": {},
        }

    should_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    predicate_nonempty = 0
    for row in rows:
        should_counter[str(row.get("should_have_event") or "").strip() or "<empty>"] += 1
        type_counter[
            str(row.get("recommended_event_type") or "").strip() or "<empty>"
        ] += 1
        if str(row.get("suggested_predicate_text") or "").strip():
            predicate_nonempty += 1

    total = len(rows)
    should_dist = dict(should_counter.most_common())
    type_dist = dict(type_counter.most_common())
    return {
        "rows_total": total,
        "should_have_event_distribution": should_dist,
        "recommended_event_type_distribution": type_dist,
        "suggested_predicate_nonempty_count": predicate_nonempty,
        "should_have_event_rate": {
            label: as_rate(count, total) for label, count in should_dist.items()
        },
        "recommended_event_type_rate": {
            label: as_rate(count, total) for label, count in type_dist.items()
        },
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_markdown_table(
    path: Path, title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(format_value(cell) for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_bundle_tables(out_dir: Path, summary: Mapping[str, object]) -> None:
    write_json(out_dir / "overview_metrics.json", summary)

    validation = summary["validation"]
    reconciliation = summary["count_reconciliation"]
    density = summary["density"]
    extraction = summary["extraction"]
    refinement = summary["refinement"]
    compute = summary["compute"]

    write_markdown_table(
        out_dir / "01_run_overview.md",
        "Run overview",
        ["Metric", "Value"],
        [
            ("Run name", summary["run_name"]),
            ("Run id", summary["run_id"]),
            ("Input sentences", summary["input_sentences"]),
            (
                "Event-bearing sentences",
                f"{summary['event_bearing_sentences']} ({format_rate(summary['event_bearing_sentences'], summary['input_sentences'])})",
            ),
            ("Extracted events", summary["extracted_events"]),
            ("Events per input sentence", summary["events_per_input_sentence"]),
            ("overall_ok", validation["overall_ok"]),
            ("parse_ok", validation["parse_ok"]),
            ("shacl_conforms", validation["shacl_conforms"]),
            ("count mismatch", reconciliation["mismatch_count"]),
        ],
    )

    density_histogram = density["histogram"] if isinstance(density["histogram"], Mapping) else {}
    hist_rows = [
        {"event_count_bucket": bucket, "sentence_count": count}
        for bucket, count in density_histogram.items()
    ]
    write_csv(
        out_dir / "02_density_histogram.csv",
        ["event_count_bucket", "sentence_count"],
        hist_rows,
    )
    write_markdown_table(
        out_dir / "02_density_summary.md",
        "Density summary",
        ["Metric", "Value"],
        [
            ("Mean events per event-bearing sentence", density["mean"]),
            ("Median", density["median"]),
            ("P95", density["p95"]),
            ("Cap hits", density["cap_hits"]),
            ("Histogram", density["histogram"]),
        ],
    )

    trigger_categories = extraction["trigger_category_distribution"]
    trigger_rows = [
        {
            "trigger_category": category,
            "count": count,
            "share": format_rate(count, extraction["row_count"]),
        }
        for category, count in trigger_categories.items()
    ]
    write_csv(
        out_dir / "03_trigger_category_distribution.csv",
        ["trigger_category", "count", "share"],
        trigger_rows,
    )

    frame_families = extraction["frame_family_distribution"]
    family_rows = [
        {
            "frame_family": family,
            "count": count,
            "share": format_rate(count, extraction["row_count"]),
        }
        for family, count in list(frame_families.items())[:25]
    ]
    write_csv(
        out_dir / "04_frame_family_distribution.csv",
        ["frame_family", "count", "share"],
        family_rows,
    )

    warning_distribution = refinement["warning_distribution"]
    warning_rows = [
        {
            "warning": warning,
            "count": count,
            "share": format_rate(count, refinement["processed_events"]),
        }
        for warning, count in warning_distribution.items()
    ]
    write_csv(
        out_dir / "05_refinement_warning_distribution.csv",
        ["warning", "count", "share"],
        warning_rows,
    )

    write_markdown_table(
        out_dir / "06_quality_snapshot.md",
        "Quality snapshot",
        ["Metric", "Value"],
        [
            (
                "Trigger/predicate-center mismatch",
                f"{extraction['trigger_predicate_center_mismatch_count']} ({format_rate(extraction['trigger_predicate_center_mismatch_count'], extraction['row_count'])})",
            ),
            (
                "Generic-family share",
                f"{extraction['generic_family_count']} ({format_rate(extraction['generic_family_count'], extraction['row_count'])})",
            ),
            (
                "trigger_head_not_plausible",
                f"{warning_distribution.get('trigger_head_not_plausible', 0)} ({format_rate(as_int(warning_distribution.get('trigger_head_not_plausible', 0)), refinement['processed_events'])})",
            ),
            (
                "multi_event_sentence",
                f"{warning_distribution.get('multi_event_sentence', 0)} ({format_rate(as_int(warning_distribution.get('multi_event_sentence', 0)), refinement['processed_events'])})",
            ),
            ("Hard review rate", compute["hard_review_rate"]),
            ("Caution rate", compute["caution_rate"]),
            ("Info-only rate", compute["info_only_rate"]),
        ],
    )


def emit_annotation_tables(
    out_dir: Path,
    event_summary: Mapping[str, object],
    no_event_summary: Mapping[str, object],
) -> None:
    write_json(
        out_dir / "annotation_progress.json",
        {"event_trigger_slice": event_summary, "no_event_slice": no_event_summary},
    )

    rows = []
    for slice_name, summary in (
        ("event_trigger_slice", event_summary),
        ("no_event_slice", no_event_summary),
    ):
        rows.append(
            (
                slice_name,
                summary["rows_total"],
                summary["rows_with_any_label"],
                format_rate(summary["rows_with_any_label"], summary["rows_total"]),
            )
        )

    write_markdown_table(
        out_dir / "07_annotation_progress.md",
        "Annotation progress",
        ["Slice", "Rows total", "Rows with any label", "Row completion rate"],
        rows,
    )

    label_rows: List[Sequence[object]] = []
    for slice_name, summary in (
        ("event_trigger_slice", event_summary),
        ("no_event_slice", no_event_summary),
    ):
        completion = summary["label_completion"]
        distributions = summary["label_distributions"]
        for label_name, filled_count in completion.items():
            label_rows.append(
                (
                    slice_name,
                    label_name,
                    filled_count,
                    format_rate(filled_count, summary["rows_total"]),
                    distributions.get(label_name, {}),
                )
            )

    write_markdown_table(
        out_dir / "08_annotation_label_status.md",
        "Annotation label status",
        ["Slice", "Label", "Filled rows", "Completion", "Observed values"],
        label_rows,
    )


def emit_silver_tables(
    out_dir: Path,
    task_a_summary: Mapping[str, object],
    task_b_summary: Mapping[str, object],
) -> None:
    write_json(
        out_dir / "silver_semantic_summary.json",
        {
            "task_a_emitted_event_quality": task_a_summary,
            "task_b_recover_now_no_event_audit": task_b_summary,
        },
    )

    task_a_total = as_int(task_a_summary.get("rows_total"))
    task_b_total = as_int(task_b_summary.get("rows_total"))
    task_a_event_real = task_a_summary.get("event_real_distribution", {})
    task_a_trigger = task_a_summary.get("trigger_correct_distribution", {})
    task_a_frame = task_a_summary.get("frame_plausible_distribution", {})
    task_a_arguments = task_a_summary.get("arguments_usable_distribution", {})
    task_a_duplicate = task_a_summary.get("duplicate_or_alternate_distribution", {})
    task_a_overall = task_a_summary.get("overall_quality_distribution", {})
    task_b_should = task_b_summary.get("should_have_event_distribution", {})

    write_markdown_table(
        out_dir / "09_silver_task_a_semantic_quality.md",
        "Silver Task A semantic quality",
        ["Metric", "Value"],
        [
            ("Adjudicated emitted-event items", task_a_total),
            (
                "event_real_rate",
                f"{task_a_event_real.get('yes', 0)} ({format_rate(as_int(task_a_event_real.get('yes', 0)), task_a_total)})",
            ),
            (
                "trigger_correct_rate",
                f"{task_a_trigger.get('yes', 0)} ({format_rate(as_int(task_a_trigger.get('yes', 0)), task_a_total)})",
            ),
            (
                "frame_plausibility_rate",
                f"{task_a_frame.get('yes', 0)} ({format_rate(as_int(task_a_frame.get('yes', 0)), task_a_total)})",
            ),
            (
                "arguments_usable_rate",
                f"{task_a_arguments.get('yes', 0)} ({format_rate(as_int(task_a_arguments.get('yes', 0)), task_a_total)})",
            ),
            (
                "duplicate_or_alternate_rate",
                f"{task_a_duplicate.get('yes', 0)} ({format_rate(as_int(task_a_duplicate.get('yes', 0)), task_a_total)})",
            ),
            (
                "overall_usable_event_rate",
                f"{task_a_overall.get('usable', 0)} ({format_rate(as_int(task_a_overall.get('usable', 0)), task_a_total)})",
            ),
            ("event_real_distribution", task_a_event_real),
            ("trigger_correct_distribution", task_a_trigger),
            ("frame_plausible_distribution", task_a_frame),
            ("arguments_usable_distribution", task_a_arguments),
            ("duplicate_or_alternate_distribution", task_a_duplicate),
            ("overall_quality_distribution", task_a_overall),
        ],
    )

    task_a_tag_rows = [
        {
            "error_tag": tag,
            "count": count,
            "share": format_rate(count, task_a_total),
        }
        for tag, count in (task_a_summary.get("final_error_tag_distribution", {}) or {}).items()
    ]
    write_csv(
        out_dir / "09_silver_task_a_error_tags.csv",
        ["error_tag", "count", "share"],
        task_a_tag_rows,
    )

    write_markdown_table(
        out_dir / "10_silver_task_b_recover_now_audit.md",
        "Silver Task B recover-now missed-event audit",
        ["Metric", "Value"],
        [
            ("Audited no-event items", task_b_total),
            (
                "should_have_event_rate",
                f"{task_b_should.get('yes', 0)} ({format_rate(as_int(task_b_should.get('yes', 0)), task_b_total)})",
            ),
            (
                "no_event_rate",
                f"{task_b_should.get('no', 0)} ({format_rate(as_int(task_b_should.get('no', 0)), task_b_total)})",
            ),
            (
                "uncertain_rate",
                f"{task_b_should.get('uncertain', 0)} ({format_rate(as_int(task_b_should.get('uncertain', 0)), task_b_total)})",
            ),
            ("should_have_event_distribution", task_b_should),
            (
                "recommended_event_type_distribution",
                task_b_summary.get("recommended_event_type_distribution", {}),
            ),
        ],
    )

    task_b_type_rows = [
        {
            "recommended_event_type": label,
            "count": count,
            "share": format_rate(count, task_b_total),
        }
        for label, count in (
            task_b_summary.get("recommended_event_type_distribution", {}) or {}
        ).items()
    ]
    write_csv(
        out_dir / "10_silver_task_b_recommended_event_types.csv",
        ["recommended_event_type", "count", "share"],
        task_b_type_rows,
    )


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_summary = summarize_bundle(bundle_dir)
    event_annotation_rows = load_csv_rows(Path(args.event_annotation))
    no_event_annotation_rows = load_csv_rows(Path(args.no_event_annotation))
    silver_task_a_rows = load_jsonl(Path(args.silver_task_a))
    silver_task_b_rows = load_jsonl(Path(args.silver_task_b))
    event_annotation_summary = summarize_annotation_rows(
        event_annotation_rows, EVENT_LABEL_COLUMNS
    )
    no_event_annotation_summary = summarize_annotation_rows(
        no_event_annotation_rows, NO_EVENT_LABEL_COLUMNS
    )
    silver_task_a_summary = summarize_silver_task_a(silver_task_a_rows)
    silver_task_b_summary = summarize_silver_task_b(silver_task_b_rows)

    emit_bundle_tables(out_dir, bundle_summary)
    emit_annotation_tables(out_dir, event_annotation_summary, no_event_annotation_summary)
    emit_silver_tables(out_dir, silver_task_a_summary, silver_task_b_summary)

    index_lines = [
        "# Thesis evaluation table bundle",
        "",
        f"- Bundle source: `{bundle_dir}`",
        f"- Event annotation source: `{args.event_annotation}`",
        f"- No-event annotation source: `{args.no_event_annotation}`",
        f"- Silver Task A source: `{args.silver_task_a}`",
        f"- Silver Task B source: `{args.silver_task_b}`",
        "",
        "## Generated files",
        "",
        "- `overview_metrics.json`",
        "- `01_run_overview.md`",
        "- `02_density_histogram.csv`",
        "- `02_density_summary.md`",
        "- `03_trigger_category_distribution.csv`",
        "- `04_frame_family_distribution.csv`",
        "- `05_refinement_warning_distribution.csv`",
        "- `06_quality_snapshot.md`",
        "- `07_annotation_progress.md`",
        "- `08_annotation_label_status.md`",
        "- `annotation_progress.json`",
        "- `silver_semantic_summary.json`",
        "- `09_silver_task_a_semantic_quality.md`",
        "- `09_silver_task_a_error_tags.csv`",
        "- `10_silver_task_b_recover_now_audit.md`",
        "- `10_silver_task_b_recommended_event_types.csv`",
        "",
        "## Notes",
        "",
        "- Density reporting uses mean / median / p95 / histogram rather than `multi_event_warning_rate` as the headline.",
        "- Annotation files can stay partially blank; this script reports progress now and becomes an evaluation summarizer later once labels are filled.",
        "- Silver semantic evaluation is reported from adjudicated Task A outputs and the recover-now Task B audit, and should be described as LLM-assisted silver evaluation rather than human gold annotation.",
    ]
    (out_dir / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
