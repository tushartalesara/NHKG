#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare extraction saturation across multiple max-event caps without changing canonical defaults."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from eval.analyze_event_density import build_sentence_texts, collect_sentence_events, percentile
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from eval.analyze_event_density import build_sentence_texts, collect_sentence_events, percentile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled extraction comparison across multiple max-events caps and report saturation metrics."
    )
    parser.add_argument("--model", required=True, help="GGUF model path passed through to gold/pipeline.py")
    parser.add_argument("--input", required=True, help="One sentence per line input text used for extraction")
    parser.add_argument("--registry", required=True, help="Frame registry JSON used by gold/pipeline.py")
    parser.add_argument("--schemas", required=True, help="Schema directory used by gold/pipeline.py")
    parser.add_argument("--caps", nargs="+", type=int, default=[3, 5, 8], help="Max-event cap values to compare, for example 3 5 8")
    parser.add_argument("--ctx", type=int, default=2048, help="Context window passed through to gold/pipeline.py")
    parser.add_argument("--candidate-top-k", type=int, default=6, help="Candidate frame shortlist passed through to gold/pipeline.py")
    parser.add_argument("--subset-lines", type=int, default=0, help="Optional number of input lines to analyze from --start-line onward")
    parser.add_argument("--start-line", type=int, default=0, help="Optional zero-based starting line for subset analysis")
    parser.add_argument("--limit", type=int, default=0, help="Backward-compatible alias for --subset-lines")
    parser.add_argument("--tmp-dir", default="", help="Optional directory to hold temporary extraction files")
    parser.add_argument("--run-name-prefix", default="cap_sweep", help="Prefix used for temporary extraction filenames")
    parser.add_argument("--parallel", action="store_true", help="Run independent cap extractions in parallel when enough GPUs are available.")
    parser.add_argument("--max-workers", type=int, default=0, help="Maximum parallel cap workers; defaults to the number of provided --gpu-devices or 1.")
    parser.add_argument("--gpu-devices", nargs="*", default=[], help="Optional CUDA device ids used to pin parallel cap runs, for example 0 1.")
    parser.add_argument("--llama-gpu-layers", type=int, default=-1, help="n_gpu_layers passed through to gold/pipeline.py")
    parser.add_argument("--llama-main-gpu", type=int, default=-1, help="Optional main_gpu passed through to gold/pipeline.py")
    parser.add_argument("--llama-tensor-split", nargs="*", type=float, default=[], help="Optional tensor split ratios passed through to gold/pipeline.py")
    parser.add_argument("--llama-split-mode", choices=["none", "layer", "row"], default="layer", help="Split mode passed through to gold/pipeline.py when tensor split is set")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--csv-out", required=True, help="Output CSV summary path")
    parser.add_argument("--review-csv-out", default="", help="Optional CSV export path for cap-sensitive review examples")
    parser.add_argument("--review-limit", type=int, default=50, help="Maximum number of cap-sensitive example rows to retain in JSON")
    return parser


def load_input_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def select_lines(lines: Sequence[str], *, start_line: int, subset_lines: int) -> List[str]:
    start = max(0, int(start_line))
    if start >= len(lines):
        return []
    if subset_lines <= 0:
        return list(lines[start:])
    return list(lines[start : start + subset_lines])


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def run_pipeline(
    *,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    registry_path: Path,
    schema_dir: Path,
    ctx: int,
    max_events: int,
    candidate_top_k: int,
    llama_gpu_layers: int,
    llama_main_gpu: int,
    llama_tensor_split: Sequence[float],
    llama_split_mode: str,
    cuda_visible_devices: Optional[str] = None,
) -> float:
    effective_main_gpu = int(llama_main_gpu)
    effective_tensor_split = [float(value) for value in llama_tensor_split]
    if cuda_visible_devices is not None:
        if effective_main_gpu >= 0:
            effective_main_gpu = 0
        if len(effective_tensor_split) > 1:
            effective_tensor_split = []
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "gold" / "pipeline.py"),
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--out",
        str(output_path),
        "--registry",
        str(registry_path),
        "--schemas",
        str(schema_dir),
        "--ctx",
        str(ctx),
        "--llama-gpu-layers",
        str(llama_gpu_layers),
        "--max-events",
        str(max_events),
        "--candidate-top-k",
        str(candidate_top_k),
    ]
    if effective_main_gpu >= 0:
        cmd.extend(["--llama-main-gpu", str(effective_main_gpu)])
    if effective_tensor_split:
        cmd.extend(["--llama-tensor-split", *[str(value) for value in effective_tensor_split]])
        cmd.extend(["--llama-split-mode", str(llama_split_mode)])
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    started = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    return round(time.perf_counter() - started, 4)


def _run_cap_job(
    *,
    cap: int,
    model_path: Path,
    subset_input: Path,
    tmp_dir_path: Path,
    registry_path: Path,
    schema_dir: Path,
    ctx: int,
    candidate_top_k: int,
    llama_gpu_layers: int,
    llama_main_gpu: int,
    llama_tensor_split: Sequence[float],
    llama_split_mode: str,
    cuda_visible_devices: Optional[str],
) -> tuple[int, Dict[str, List[dict]], float]:
    output_jsonl = tmp_dir_path / f"cap_{cap}.jsonl"
    runtime_seconds = run_pipeline(
        model_path=model_path,
        input_path=subset_input,
        output_path=output_jsonl,
        registry_path=registry_path,
        schema_dir=schema_dir,
        ctx=ctx,
        max_events=int(cap),
        candidate_top_k=candidate_top_k,
        llama_gpu_layers=llama_gpu_layers,
        llama_main_gpu=llama_main_gpu,
        llama_tensor_split=llama_tensor_split,
        llama_split_mode=llama_split_mode,
        cuda_visible_devices=cuda_visible_devices,
    )
    grouped_events = collect_sentence_events(output_jsonl)
    return int(cap), grouped_events, runtime_seconds


def summarize_cap(cap: int, grouped_events: Dict[str, List[dict]], runtime_seconds: float) -> dict:
    counts = {sent_key: len(events) for sent_key, events in grouped_events.items()}
    values = list(counts.values())
    histogram = Counter(values)
    cap_hits = sum(1 for value in values if value >= cap)
    return {
        "cap": int(cap),
        "extracted_event_count": int(sum(values)),
        "sentences_analyzed": int(len(values)),
        "mean_events_per_sentence": round(statistics.mean(values), 4) if values else 0.0,
        "median_events_per_sentence": float(statistics.median(values)) if values else 0.0,
        "p95_events_per_sentence": round(percentile(values, 0.95), 4) if values else 0.0,
        "cap_hits": int(cap_hits),
        "cap_hit_rate": round(cap_hits / len(values), 4) if values else 0.0,
        "event_histogram": {str(key): int(histogram[key]) for key in sorted(histogram)},
        "multi_event_warning_rate": None,
        "runtime_seconds": runtime_seconds,
    }


def find_stabilized_at(counts_by_cap: Dict[int, int], ordered_caps: Sequence[int]) -> int | None:
    for index, cap in enumerate(ordered_caps):
        remaining = [counts_by_cap.get(value, 0) for value in ordered_caps[index:]]
        if remaining and len(set(remaining)) == 1:
            return int(cap)
    return None


def build_cap_sensitive_examples(
    *,
    cap_counts: Dict[int, Dict[str, int]],
    sentence_texts: Dict[str, str],
    ordered_caps: Sequence[int],
    limit: int,
) -> List[dict]:
    if not ordered_caps:
        return []
    smallest_cap = int(ordered_caps[0])
    largest_cap = int(ordered_caps[-1])
    all_keys = sorted({sent_key for counts in cap_counts.values() for sent_key in counts})
    rows: List[dict] = []
    for sent_key in all_keys:
        counts_by_cap = {int(cap): int(cap_counts.get(int(cap), {}).get(sent_key, 0)) for cap in ordered_caps}
        base_count = counts_by_cap.get(smallest_cap, 0)
        max_count = max(counts_by_cap.values()) if counts_by_cap else 0
        if max_count <= base_count:
            continue
        stabilized_at = find_stabilized_at(counts_by_cap, ordered_caps)
        rows.append(
            {
                "sent_key": sent_key,
                "sentence_text": sentence_texts.get(sent_key, ""),
                "counts_by_cap": {str(cap): count for cap, count in counts_by_cap.items()},
                f"delta_from_{smallest_cap}_to_{largest_cap}": max_count - base_count,
                "base_cap": smallest_cap,
                "base_cap_hit": base_count >= smallest_cap,
                "largest_cap": largest_cap,
                "largest_cap_count": counts_by_cap.get(largest_cap, 0),
                "stabilized_at_cap": stabilized_at,
                "reason_for_review": "count_rises_under_higher_caps" if max_count > base_count else "stable",
            }
        )
    rows.sort(
        key=lambda item: (
            -int(item.get(f"delta_from_{smallest_cap}_to_{largest_cap}", 0) or 0),
            0 if item.get("base_cap_hit") else 1,
            item.get("sent_key", ""),
        )
    )
    return rows[: max(0, int(limit))]


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cap",
        "extracted_event_count",
        "sentences_analyzed",
        "mean_events_per_sentence",
        "median_events_per_sentence",
        "p95_events_per_sentence",
        "cap_hits",
        "cap_hit_rate",
        "multi_event_warning_rate",
        "runtime_seconds",
        "event_histogram",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "cap": row.get("cap"),
                    "extracted_event_count": row.get("extracted_event_count"),
                    "sentences_analyzed": row.get("sentences_analyzed"),
                    "mean_events_per_sentence": row.get("mean_events_per_sentence"),
                    "median_events_per_sentence": row.get("median_events_per_sentence"),
                    "p95_events_per_sentence": row.get("p95_events_per_sentence"),
                    "cap_hits": row.get("cap_hits"),
                    "cap_hit_rate": row.get("cap_hit_rate"),
                    "multi_event_warning_rate": row.get("multi_event_warning_rate"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "event_histogram": json.dumps(row.get("event_histogram", {}), ensure_ascii=False, sort_keys=True),
                }
            )


def write_review_csv(path: Path, rows: Sequence[dict], ordered_caps: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sent_key",
        "sentence_text",
        *[f"count_cap_{int(cap)}" for cap in ordered_caps],
        "base_cap",
        "base_cap_hit",
        "largest_cap",
        "largest_cap_count",
        f"delta_from_{int(ordered_caps[0])}_to_{int(ordered_caps[-1])}" if ordered_caps else "delta",
        "stabilized_at_cap",
        "reason_for_review",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            counts_by_cap = row.get("counts_by_cap", {}) or {}
            payload = {
                "sent_key": row.get("sent_key", ""),
                "sentence_text": row.get("sentence_text", ""),
                "base_cap": row.get("base_cap"),
                "base_cap_hit": row.get("base_cap_hit"),
                "largest_cap": row.get("largest_cap"),
                "largest_cap_count": row.get("largest_cap_count"),
                f"delta_from_{int(ordered_caps[0])}_to_{int(ordered_caps[-1])}" if ordered_caps else "delta": row.get(
                    f"delta_from_{int(ordered_caps[0])}_to_{int(ordered_caps[-1])}", 0
                )
                if ordered_caps
                else row.get("delta", 0),
                "stabilized_at_cap": row.get("stabilized_at_cap"),
                "reason_for_review": row.get("reason_for_review", ""),
            }
            for cap in ordered_caps:
                payload[f"count_cap_{int(cap)}"] = counts_by_cap.get(str(int(cap)), 0)
            writer.writerow(payload)


def main() -> None:
    args = build_arg_parser().parse_args()
    model_path = Path(args.model).resolve()
    input_path = Path(args.input).resolve()
    registry_path = Path(args.registry).resolve()
    schema_dir = Path(args.schemas).resolve()
    caps = sorted({int(value) for value in (args.caps or []) if int(value) > 0})
    if not caps:
        raise SystemExit("compare_event_caps requires at least one positive value in --caps.")
    subset_lines = int(args.subset_lines or args.limit or 0)
    start_line = int(args.start_line or 0)
    input_lines = load_input_lines(input_path)
    selected_lines = select_lines(input_lines, start_line=start_line, subset_lines=subset_lines)
    if not selected_lines:
        raise SystemExit("No input lines were selected for compare_event_caps. Check --start-line/--subset-lines.")

    tmp_root = Path(args.tmp_dir).resolve() if args.tmp_dir else None
    if tmp_root:
        tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(tmp_root) if tmp_root else None, prefix=f"{args.run_name_prefix}_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        subset_input = tmp_dir_path / "subset_input.txt"
        write_lines(subset_input, selected_lines)

        results: List[dict] = []
        cap_counts: Dict[int, Dict[str, int]] = {}
        sentence_texts: Dict[str, str] = {}

        gpu_devices = [str(value) for value in (args.gpu_devices or []) if str(value).strip()]
        max_workers = int(args.max_workers or 0)
        if max_workers <= 0:
            max_workers = min(len(caps), len(gpu_devices)) if args.parallel and gpu_devices else 1
        max_workers = max(1, max_workers)
        run_parallel = bool(args.parallel and len(caps) > 1 and max_workers > 1)

        jobs = []
        for index, cap in enumerate(caps):
            assigned_device = gpu_devices[index % len(gpu_devices)] if gpu_devices else None
            jobs.append(
                {
                    "cap": int(cap),
                    "cuda_visible_devices": assigned_device,
                }
            )

        if run_parallel:
            results_by_cap: Dict[int, tuple[Dict[str, List[dict]], float]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_cap_job,
                        cap=job["cap"],
                        model_path=model_path,
                        subset_input=subset_input,
                        tmp_dir_path=tmp_dir_path,
                        registry_path=registry_path,
                        schema_dir=schema_dir,
                        ctx=int(args.ctx),
                        candidate_top_k=int(args.candidate_top_k),
                        llama_gpu_layers=int(args.llama_gpu_layers),
                        llama_main_gpu=int(args.llama_main_gpu),
                        llama_tensor_split=args.llama_tensor_split or [],
                        llama_split_mode=str(args.llama_split_mode),
                        cuda_visible_devices=job["cuda_visible_devices"],
                    ): job["cap"]
                    for job in jobs
                }
                for future in as_completed(futures):
                    cap, grouped_events, runtime_seconds = future.result()
                    results_by_cap[int(cap)] = (grouped_events, runtime_seconds)
            for cap in caps:
                grouped_events, runtime_seconds = results_by_cap[int(cap)]
                output_jsonl = tmp_dir_path / f"cap_{cap}.jsonl"
                if not sentence_texts:
                    sentence_texts = build_sentence_texts(output_jsonl, subset_input)
                cap_counts[int(cap)] = {sent_key: len(events) for sent_key, events in grouped_events.items()}
                results.append(summarize_cap(int(cap), grouped_events, runtime_seconds))
        else:
            for job in jobs:
                cap, grouped_events, runtime_seconds = _run_cap_job(
                    cap=job["cap"],
                    model_path=model_path,
                    subset_input=subset_input,
                    tmp_dir_path=tmp_dir_path,
                    registry_path=registry_path,
                    schema_dir=schema_dir,
                    ctx=int(args.ctx),
                    candidate_top_k=int(args.candidate_top_k),
                    llama_gpu_layers=int(args.llama_gpu_layers),
                    llama_main_gpu=int(args.llama_main_gpu),
                    llama_tensor_split=args.llama_tensor_split or [],
                    llama_split_mode=str(args.llama_split_mode),
                    cuda_visible_devices=job["cuda_visible_devices"],
                )
                output_jsonl = tmp_dir_path / f"cap_{cap}.jsonl"
                if not sentence_texts:
                    sentence_texts = build_sentence_texts(output_jsonl, subset_input)
                cap_counts[int(cap)] = {sent_key: len(events) for sent_key, events in grouped_events.items()}
                results.append(summarize_cap(int(cap), grouped_events, runtime_seconds))

        cap_sensitive_examples = build_cap_sensitive_examples(
            cap_counts=cap_counts,
            sentence_texts=sentence_texts,
            ordered_caps=caps,
            limit=int(args.review_limit),
        )

        report = {
            "meta": {
                "schema_version": "nhkg.cap-sweep.v2",
                "stage_version": "1.1.0",
                "model": str(model_path),
                "input": str(input_path),
                "input_line_count": len(input_lines),
                "registry": str(registry_path),
                "schemas": str(schema_dir),
                "caps": caps,
                "parallel": run_parallel,
                "max_workers": max_workers,
                "gpu_devices": gpu_devices,
                "ctx": int(args.ctx),
                "candidate_top_k": int(args.candidate_top_k),
                "llama_gpu_layers": int(args.llama_gpu_layers),
                "llama_main_gpu": int(args.llama_main_gpu),
                "llama_tensor_split": [float(value) for value in (args.llama_tensor_split or [])],
                "llama_split_mode": str(args.llama_split_mode),
                "start_line": start_line,
                "subset_lines": len(selected_lines),
                "selected_line_count": len(selected_lines),
            },
            "results": results,
            "cap_sensitive_examples": cap_sensitive_examples,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_csv(Path(args.csv_out), results)
    if args.review_csv_out:
        write_review_csv(Path(args.review_csv_out), cap_sensitive_examples, caps)

    print(f"[OK] wrote_cap_comparison={out_path}")
    print(f"[OK] caps={caps} subset_lines={len(selected_lines)}")
    if args.review_csv_out:
        print(f"[OK] wrote_cap_sensitive_examples_csv={Path(args.review_csv_out)}")
    for row in results:
        print(
            f"[OK] cap={row['cap']} events={row['extracted_event_count']} "
            f"sentences={row['sentences_analyzed']} mean={row['mean_events_per_sentence']:.4f} "
            f"median={row['median_events_per_sentence']:.4f} p95={row['p95_events_per_sentence']:.4f} "
            f"cap_hits={row['cap_hits']} runtime_seconds={row['runtime_seconds']:.4f}"
        )


if __name__ == "__main__":
    main()
