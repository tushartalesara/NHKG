#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run sentence-scoped Hindi NER as a post-extraction enrichment layer."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

try:
    from .enrichment_common import collect_sentence_keys, load_sentence_text_map, sentence_key
    from .internal_quality_common import DEFAULT_STAGE_VERSION, build_stage_metadata
except ImportError:  # pragma: no cover
    from enrichment_common import collect_sentence_keys, load_sentence_text_map, sentence_key
    from internal_quality_common import DEFAULT_STAGE_VERSION, build_stage_metadata


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_RULE_CONFIG = {
    "honorific_prefixes": ["श्री", "श्रीमती", "सुश्री", "डॉ", "डा", "प्रो"],
    "location_context_markers": ["शहर", "नगर", "जिला", "ज़िला", "गाँव", "गांव", "राज्य", "प्रदेश"],
}

TIME_PATTERNS = [
    re.compile(r"\b(?:आज|कल|परसों|नरसों|सुबह|शाम|रात|दोपहर|अभी|फिर|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)\b"),
    re.compile(r"\b(?:पिछले|अगले|इस)\s+(?:सप्ताह|हफ्ते|महीने|माह|साल|वर्ष|दिन)\b"),
    re.compile(r"\b\d{1,2}(?::\d{2})?\s*बजे\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"),
]
NUM_PATTERNS = [
    re.compile(r"\b[0-9०-९]+(?:[.,][0-9०-९]+)?\b"),
    re.compile(r"\b(?:एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह|सोलह|सत्रह|अठारह|उन्नीस|बीस|सौ|हजार|लाख|करोड़)\b"),
]


def load_yaml_config(path: Optional[Path], default: dict) -> dict:
    if path is None or not path.exists():
        return dict(default)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    merged = dict(default)
    if isinstance(payload, dict):
        merged.update(payload)
    return merged


def load_label_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("NER label map must be a JSON/YAML object mapping raw labels to canonical labels.")
    out: Dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key or "").strip().upper()
        value = str(raw_value or "").strip().upper()
        if key and value:
            out[key] = value
    return out


def compile_rule_patterns(config: dict) -> Dict[str, Optional[re.Pattern[str]]]:
    honorifics = [re.escape(str(item).strip()) for item in (config.get("honorific_prefixes") or []) if str(item).strip()]
    loc_markers = [re.escape(str(item).strip()) for item in (config.get("location_context_markers") or []) if str(item).strip()]
    honorific_pattern = None
    location_pattern = None
    if honorifics:
        honorific_pattern = re.compile(r"\b(?:%s)\.?\s+[^\s,;।!?]+(?:\s+[^\s,;।!?]+)?" % "|".join(honorifics))
    if loc_markers:
        location_pattern = re.compile(r"\b[^\s,;।!?]+(?:\s+[^\s,;।!?]+)?\s+(?:%s)\b" % "|".join(loc_markers))
    return {
        "honorific_pattern": honorific_pattern,
        "location_pattern": location_pattern,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hindi NER over sentence-aligned text and write align/ner_cache.json.")
    parser.add_argument("--input", default="", help="Extraction JSONL/JSON used to recover sentence keys")
    parser.add_argument("--sentences", default="", help="One sentence per line text file")
    parser.add_argument("--engine", choices=["local_hf", "rules", "none"], default="rules", help="NER backend to use. Prefer local_hf for real model-backed NER; rules is a conservative fallback baseline.")
    parser.add_argument("--model", default="", help="Local model path or Hugging Face model id for --engine local_hf")
    parser.add_argument("--device", default="cpu", help="Preferred device: cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--label-map", default="", help="Optional JSON/YAML map from raw model labels to canonical labels")
    parser.add_argument("--config", default="lexicons/ner_rules_config.yaml", help="Optional YAML config for conservative rule patterns")
    parser.add_argument("--out", default="align/ner_cache.json", help="Output JSON cache path")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of sample mentions to print")
    return parser


def load_sentence_entries(input_path: Optional[Path], sentence_path: Optional[Path]) -> List[Tuple[str, str]]:
    if sentence_path is None or not sentence_path.exists():
        raise SystemExit("NER needs sentence text. Pass --sentences with one sentence per line.")
    sentence_keys = collect_sentence_keys(input_path) if input_path else []
    sentence_map = load_sentence_text_map(sentence_path, sentence_keys=sentence_keys)
    if not sentence_map:
        raise SystemExit("No sentence text lines found in the supplied --sentences file.")
    if sentence_keys:
        ordered = [(key, sentence_map[key]) for key in sentence_keys if key in sentence_map]
        if ordered:
            return ordered
    return list(sentence_map.items())


def canonical_label(raw_label: str, label_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    label = str(raw_label or "").strip().upper()
    if not label or label == "O":
        return None
    if label_map and label in label_map:
        mapped = str(label_map[label]).strip().upper()
        return mapped or None
    if "-" in label:
        prefix, _, base = label.partition("-")
        if prefix in {"B", "I", "E", "S"} and base:
            label = base
            if label_map and label in label_map:
                mapped = str(label_map[label]).strip().upper()
                return mapped or None
    if any(part in label for part in ("PER", "PERSON")):
        return "PER"
    if any(part in label for part in ("ORG", "ORGANIZATION")):
        return "ORG"
    if any(part in label for part in ("LOC", "GPE", "PLACE", "FAC")):
        return "LOC"
    if any(part in label for part in ("DATE", "TIME", "TIMEX", "TEMP")):
        return "TIME"
    if any(part in label for part in ("NUM", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT")):
        return "NUM"
    if label.startswith("LABEL_"):
        return None
    return "MISC"


def bio_parts(raw_label: str) -> Tuple[str, str]:
    label = str(raw_label or "").strip()
    if not label or label.upper() == "O":
        return "O", ""
    if "-" in label:
        prefix, base = label.split("-", 1)
        prefix = prefix.upper()
        if prefix in {"B", "I", "E", "S"}:
            return prefix, base
    return "U", label


def detect_rule_mentions(sent_key: str, text: str, config: dict) -> List[dict]:
    mentions: List[dict] = []
    patterns = compile_rule_patterns(config)
    for pattern in TIME_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append({"sent_key": sent_key, "text": match.group(0), "start": match.start(), "end": match.end(), "label": "TIME", "canonical_label": "TIME", "raw_label": "RULE_TIME", "score": 0.72, "confidence": 0.72, "engine": "rules", "backend": "rules", "model_name": "rules", "token_indices": [], "source": "rules"})
    for pattern in NUM_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append({"sent_key": sent_key, "text": match.group(0), "start": match.start(), "end": match.end(), "label": "NUM", "canonical_label": "NUM", "raw_label": "RULE_NUM", "score": 0.70, "confidence": 0.70, "engine": "rules", "backend": "rules", "model_name": "rules", "token_indices": [], "source": "rules"})
    honorific_pattern = patterns.get("honorific_pattern")
    if honorific_pattern is not None:
        for match in honorific_pattern.finditer(text):
            mentions.append({"sent_key": sent_key, "text": match.group(0), "start": match.start(), "end": match.end(), "label": "PER", "canonical_label": "PER", "raw_label": "RULE_PER", "score": 0.55, "confidence": 0.55, "engine": "rules", "backend": "rules", "model_name": "rules", "token_indices": [], "source": "rules"})
    location_pattern = patterns.get("location_pattern")
    if location_pattern is not None:
        for match in location_pattern.finditer(text):
            mentions.append({"sent_key": sent_key, "text": match.group(0), "start": match.start(), "end": match.end(), "label": "LOC", "canonical_label": "LOC", "raw_label": "RULE_LOC", "score": 0.54, "confidence": 0.54, "engine": "rules", "backend": "rules", "model_name": "rules", "token_indices": [], "source": "rules"})
    return dedupe_mentions(mentions)


def dedupe_mentions(mentions: Iterable[dict]) -> List[dict]:
    best_by_key: Dict[Tuple[int, int, str], dict] = {}
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        try:
            start = int(mention.get("start"))
            end = int(mention.get("end"))
        except (TypeError, ValueError):
            continue
        label = str(mention.get("label", "")).strip()
        text = str(mention.get("text", "")).strip()
        if not label or not text or end <= start:
            continue
        key = (start, end, label)
        score = mention.get("confidence", mention.get("score"))
        try:
            score_value = float(score) if score is not None else -1.0
        except (TypeError, ValueError):
            score_value = -1.0
        current = best_by_key.get(key)
        current_score = -1.0
        if current is not None:
            try:
                current_score = float(current.get("confidence", current.get("score")))
            except (TypeError, ValueError):
                current_score = -1.0
        if current is None or score_value >= current_score:
            item = dict(mention)
            if "confidence" not in item and "score" in item:
                item["confidence"] = item.get("score")
            best_by_key[key] = item
    return [best_by_key[key] for key in sorted(best_by_key.keys(), key=lambda item: (item[0], item[1], item[2]))]


def run_local_hf(entries: Sequence[Tuple[str, str]], model_name: str, device: str, batch_size: int, label_map: Optional[Dict[str, str]] = None) -> Dict[str, List[dict]]:
    if not model_name:
        raise SystemExit("A model path or model id is required for --engine local_hf.")
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers and torch are required for --engine local_hf. Install them or use --engine rules.") from exc
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        device = "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    except Exception as exc:
        raise SystemExit(f"Failed to load token-classification model '{model_name}'. Use --engine rules if the model is unavailable locally.") from exc
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("The selected model needs a fast tokenizer so span offsets can be reconstructed reliably.")
    target_device = torch.device("cuda" if device == "cuda" else "cpu")
    model.to(target_device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    by_key: Dict[str, List[dict]] = {key: [] for key, _ in entries}
    for start_idx in range(0, len(entries), max(1, batch_size)):
        batch = list(entries[start_idx : start_idx + max(1, batch_size)])
        batch_keys = [key for key, _ in batch]
        batch_texts = [text for _, text in batch]
        encoded = tokenizer(batch_texts, return_offsets_mapping=True, padding=True, truncation=True, return_tensors="pt")
        offset_mappings = encoded.pop("offset_mapping")
        encoded = {name: tensor.to(target_device) for name, tensor in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu()
        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        for batch_index, sent_key_value in enumerate(batch_keys):
            raw_tokens = []
            text = batch_texts[batch_index]
            offset_rows = offset_mappings[batch_index].tolist()
            pred_row = predictions[batch_index].tolist()
            prob_row = probabilities[batch_index]
            for token_index, (offset_pair, pred_id) in enumerate(zip(offset_rows, pred_row)):
                start_char, end_char = int(offset_pair[0]), int(offset_pair[1])
                if end_char <= start_char:
                    continue
                raw_label = str(id2label.get(int(pred_id), "O"))
                score = float(prob_row[token_index, int(pred_id)].item())
                canon = canonical_label(raw_label, label_map=label_map)
                if canon is None:
                    continue
                raw_tokens.append({"start": start_char, "end": end_char, "raw_label": raw_label, "canonical": canon, "score": score, "token_index": token_index, "text": text[start_char:end_char]})
            merged = merge_model_tokens(sent_key_value, text, raw_tokens, model_name=model_name)
            by_key[sent_key_value].extend(merged)
    return {key: dedupe_mentions(value) for key, value in by_key.items()}


def merge_model_tokens(sent_key_value: str, text: str, raw_tokens: Sequence[dict], model_name: str) -> List[dict]:
    mentions: List[dict] = []
    current: Optional[dict] = None
    def flush() -> None:
        nonlocal current
        if current is None:
            return
        start = current["start"]
        end = current["end"]
        if end > start:
            mentions.append({"sent_key": sent_key_value, "text": text[start:end], "start": start, "end": end, "label": current["label"], "canonical_label": current["label"], "raw_label": "|".join(current["raw_labels"]), "score": round(sum(current["scores"]) / max(1, len(current["scores"])), 6), "confidence": round(sum(current["scores"]) / max(1, len(current["scores"])), 6), "engine": "local_hf", "backend": "local_hf", "model_name": model_name, "token_indices": list(current["token_indices"]), "source": "model"})
        current = None
    for token in raw_tokens:
        prefix, _ = bio_parts(token["raw_label"])
        label = token["canonical"]
        token_start = int(token["start"])
        token_end = int(token["end"])
        token_index = int(token["token_index"])
        contiguous = current is not None and token_start <= current["end"] + 1
        if current is None or label != current["label"] or prefix in {"B", "S"} or not contiguous:
            flush()
            current = {"label": label, "start": token_start, "end": token_end, "raw_labels": [token["raw_label"]], "scores": [float(token["score"])], "token_indices": [token_index]}
            continue
        current["end"] = max(current["end"], token_end)
        current["raw_labels"].append(token["raw_label"])
        current["scores"].append(float(token["score"]))
        current["token_indices"].append(token_index)
    flush()
    return dedupe_mentions(mentions)


def apply_rule_fallback(
    entries: Sequence[Tuple[str, str]],
    mentions_by_key: Dict[str, List[dict]],
    rule_config: dict,
) -> Tuple[Dict[str, List[dict]], dict]:
    fallback_sentence_count = 0
    fallback_mention_count = 0
    for sent_key_value, text in entries:
        existing = dedupe_mentions(mentions_by_key.get(sent_key_value, []))
        if existing:
            mentions_by_key[sent_key_value] = existing
            continue
        fallback_mentions = detect_rule_mentions(sent_key_value, text, rule_config)
        mentions_by_key[sent_key_value] = fallback_mentions
        if fallback_mentions:
            fallback_sentence_count += 1
            fallback_mention_count += len(fallback_mentions)
    return mentions_by_key, {
        "fallback_sentence_count": fallback_sentence_count,
        "fallback_mention_count": fallback_mention_count,
    }


def summarize_mentions(mentions_by_key: Dict[str, List[dict]]) -> dict:
    label_counts = Counter()
    model_mentions = 0
    rule_mentions = 0
    confidence_values: List[float] = []
    total_mentions = 0
    for mentions in mentions_by_key.values():
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            total_mentions += 1
            label = str(mention.get("canonical_label", mention.get("label", "MISC")) or "MISC")
            label_counts[label] += 1
            if mention.get("source") == "model":
                model_mentions += 1
            elif mention.get("source") == "rules":
                rule_mentions += 1
            score = mention.get("confidence", mention.get("score"))
            try:
                if score is not None:
                    confidence_values.append(float(score))
            except (TypeError, ValueError):
                continue
    return {
        "total_entity_mentions": total_mentions,
        "mentions_by_type": dict(sorted(label_counts.items())),
        "model_backed_mentions": model_mentions,
        "rule_backed_mentions": rule_mentions,
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 6) if confidence_values else None,
    }


def build_payload(entries: Sequence[Tuple[str, str]], mentions_by_key: Dict[str, List[dict]], *, engine: str, model_name: str, input_path: Optional[Path], sentence_path: Optional[Path], warnings: Sequence[str], label_map_path: Optional[Path], config_path: Optional[Path], summary: Optional[dict] = None, fallback_summary: Optional[dict] = None) -> Dict[str, object]:
    summary_payload = summary or summarize_mentions(mentions_by_key)
    payload: Dict[str, object] = {
        "meta": build_stage_metadata(
            stage_name="run_hindi_ner",
            stage_version=DEFAULT_STAGE_VERSION,
            engine=engine,
            source_paths={"input": input_path, "sentences": sentence_path},
            input_counts={"sentences": len(entries)},
            warnings=warnings,
            extra={
                "model_name": model_name,
                "label_map_path": str(label_map_path) if label_map_path else "",
                "rules_config_path": str(config_path) if config_path else "",
                "summary": summary_payload,
                "fallback_summary": fallback_summary or {},
            },
        ),
        "summary": {
            **summary_payload,
            "engine": engine,
            "model_name": model_name,
            "warnings": [str(item) for item in warnings if str(item)],
            "fallback_summary": fallback_summary or {},
        },
        "sentences": {},
    }
    for sent_key_value, text in entries:
        doc_id, sent_id = sent_key_value.split("::", 1) if "::" in sent_key_value else (sent_key_value, "0")
        payload["sentences"].setdefault(doc_id, {})
        payload["sentences"][doc_id][sent_id] = {"text": text, "entities": mentions_by_key.get(sent_key_value, [])}
    return payload


def print_summary(payload: Dict[str, object], debug_samples: int) -> None:
    total_sentences = 0
    total_mentions = 0
    per_label = Counter()
    model_mentions = 0
    rule_mentions = 0
    confidences: List[float] = []
    samples: List[str] = []
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            total_sentences += 1
            mentions = item.get("entities", [])
            if not isinstance(mentions, list):
                continue
            total_mentions += len(mentions)
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                per_label[str(mention.get("label", "MISC"))] += 1
                if mention.get("source") == "model":
                    model_mentions += 1
                elif mention.get("source") == "rules":
                    rule_mentions += 1
                score = mention.get("confidence", mention.get("score"))
                try:
                    if score is not None:
                        confidences.append(float(score))
                except (TypeError, ValueError):
                    pass
            if len(samples) < max(0, debug_samples) and mentions:
                preview = []
                for mention in mentions[:4]:
                    preview.append(f"{mention.get('text', '')}/{mention.get('label', '')}@{mention.get('start', '')}:{mention.get('end', '')}")
                samples.append(f"{sentence_key(doc_id, sent_id)} :: {' | '.join(preview)}")
    print(f"[OK] total_sentences={total_sentences}")
    print(f"[OK] total_entity_mentions={total_mentions}")
    print(f"[OK] mentions_per_label={dict(sorted(per_label.items()))}")
    print(f"[OK] mentions_from_model={model_mentions} mentions_from_rules={rule_mentions}")
    if confidences:
        print(f"[OK] average_confidence={statistics.mean(confidences):.4f}")
    for sample in samples:
        print(f"[DBG] {sample}")


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input) if args.input else None
    if input_path is not None and not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    sentence_path = Path(args.sentences) if args.sentences else None
    config_path = Path(args.config) if args.config else None
    label_map_path = Path(args.label_map) if args.label_map else None
    if label_map_path is not None and not label_map_path.exists():
        raise SystemExit(f"NER label map not found: {label_map_path}")
    entries = load_sentence_entries(input_path, sentence_path)
    rule_config = load_yaml_config(config_path, DEFAULT_RULE_CONFIG)
    label_map = load_label_map(label_map_path)

    mentions_by_key: Dict[str, List[dict]] = {key: [] for key, _ in entries}
    warnings: List[str] = []
    fallback_summary: dict = {}
    if args.engine == "local_hf":
        mentions_by_key = run_local_hf(entries, args.model, args.device.lower(), args.batch_size, label_map=label_map)
        mentions_by_key, fallback_summary = apply_rule_fallback(entries, mentions_by_key, rule_config)
        if fallback_summary.get("fallback_sentence_count", 0):
            warnings.append(
                "Rules fallback was used only for sentences where the model produced no span-level mentions."
            )
    elif args.engine == "rules":
        print("[WARN] Running in rules mode; only conservative TIME/NUM/PER/LOC heuristics will be emitted.", file=sys.stderr)
        warnings.append("Rules mode emits a conservative baseline only; it is not a full NER model.")
        for key, text in entries:
            mentions_by_key[key] = detect_rule_mentions(key, text, rule_config)
    else:
        print("[WARN] Engine 'none' selected; writing an empty NER cache.", file=sys.stderr)
        warnings.append("Engine none selected; cache contains no entity mentions.")

    summary = summarize_mentions(mentions_by_key)
    payload = build_payload(
        entries,
        mentions_by_key,
        engine=args.engine,
        model_name=args.model or args.engine,
        input_path=input_path,
        sentence_path=sentence_path,
        warnings=warnings,
        label_map_path=label_map_path,
        config_path=config_path,
        summary=summary,
        fallback_summary=fallback_summary,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[OK] wrote_ner_cache={out_path}")
    print_summary(payload, args.debug_samples)


if __name__ == "__main__":
    main()
