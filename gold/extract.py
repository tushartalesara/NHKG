#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 3.5: Gold Single-Sentence Extractor
Adds lightweight provenance fields and character-span derivation.

Usage:
  python gold/extract.py --model ./models/mistral.gguf --schema schemas/MARAA-1.0.schema.json --output output.jsonl "राम ने सीता को देखा"
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 for logs
sys.stderr.reconfigure(encoding="utf-8")

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    sys.stderr.write("Error: llama-cpp-python not installed.\n")
    sys.exit(1)

try:
    from gold.span import Tokenizer
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from gold.span import Tokenizer


PIPELINE_SCHEMA_VERSION = "2026-02-16"
CASE_MARKER_TOKENS = {
    "ने",
    "को",
    "से",
    "में",
    "पर",
    "तक",
    "का",
    "की",
    "के",
    "लिए",
    "साथ",
    "वाला",
    "वाले",
    "वालों",
}

def log(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def make_prompt(sentence: str, tokenizer: Tokenizer, frame_id: str, schema: dict) -> str:
    tokens, _ = tokenizer.tokenize(sentence)
    token_map = "\n".join([f"{i}: {t}" for i, t in enumerate(tokens)])

    role_defs = []
    args = schema.get("properties", {}).get("arguments", {}).get("properties", {})
    for role, details in args.items():
        desc = details.get("description", "No description provided.")
        role_defs.append(f"- {role}: {desc}")
    role_block = "\n".join(role_defs)

    return f"""<|system|>
You are an expert Semantic Role Labeling (SRL) system for Hindi.
Your task is to extract the Event Frame '{frame_id}' from the text.

ROLE DEFINITIONS:
{role_block}

INSTRUCTIONS:
1. Analyze the sentence meaning to find the entity that fits each Role.
2. Use the Token Map below to provide exact [start, end) span indices.
3. Include Hindi case markers immediately attached to a role mention in the span
   (for example: `राम ने`, `रावण को`, `अयोध्या से`).
4. Output strictly valid JSON.

<|user|>
Sentence: {sentence}

Token Map:
{token_map}

Extract the frame '{frame_id}'.
<|assistant|>
"""


def char_span_for_tokens(span, offsets):
    if not span or len(span) != 2:
        return None
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start or end > len(offsets):
        return None
    return [offsets[start][0], offsets[end - 1][1]]


def _expand_marker_span(tokens, span, max_markers: int = 2):
    if not span or len(span) != 2:
        return list(span) if isinstance(span, list) else [], False

    try:
        start, end = map(int, span)
    except Exception:
        return list(span), False

    if start < 0 or end <= start or end > len(tokens):
        return [start, end], False

    expanded_end = end
    expanded = False

    for _ in range(max_markers):
        if expanded_end >= len(tokens):
            break

        marker = tokens[expanded_end].strip()
        if marker in CASE_MARKER_TOKENS:
            expanded_end += 1
            expanded = True
            continue
        break

    return [start, expanded_end], expanded


def _expand_case_markers_in_extraction(extraction: dict, tokenizer: Tokenizer, sentence: str, warnings: list) -> None:
    tokens, _ = tokenizer.tokenize(sentence)
    arguments = extraction.get("arguments", {})
    if not isinstance(arguments, dict):
        return

    for role_name, arg_data in arguments.items():
        if not isinstance(arg_data, dict):
            continue

        span = arg_data.get("span")
        if not isinstance(span, list):
            continue

        new_span, changed = _expand_marker_span(tokens, span)
        if not changed:
            continue

        arg_data["span"] = new_span
        arg_data["text"] = tokenizer.resolve_span(new_span, tokens)
        warnings.append(f"{role_name}: span {span} expanded to include case marker(s) -> {new_span}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .gguf model file")
    ap.add_argument("--schema", required=True, help="Path to .schema.json file")
    ap.add_argument("--output", required=True, help="Path to output .jsonl file")
    ap.add_argument("--doc-id", default="batch_run")
    ap.add_argument("--sent-id", type=int, default=0)
    ap.add_argument("text", help="The Hindi sentence to process")
    ap.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()

    with open(args.schema, "r", encoding="utf-8") as f:
        schema_obj = json.load(f)

    frame_id = schema_obj.get("properties", {}).get("frame", {}).get("const", "UNKNOWN")

    try:
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema_obj))
    except Exception as e:
        log(f"Error: Grammar compilation failed: {e}")
        sys.exit(1)

    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx,
        n_gpu_layers=-1,
        verbose=False,
    )

    tokenizer = Tokenizer()
    prompt = make_prompt(args.text, tokenizer, frame_id, schema_obj)

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.1,
        grammar=grammar,
        stop=["<|user|>", "</s>"]
    )

    result_text = output["choices"][0]["text"]

    try:
        extraction = json.loads(result_text)
    except json.JSONDecodeError:
        log("Error: model output is not valid JSON")
        log(f"Output: {result_text}")
        return

    extraction.setdefault("doc_id", args.doc_id)
    extraction.setdefault("sent_id", args.sent_id)
    extraction.setdefault("frame", frame_id)
    extraction["event_id"] = f"{uuid.uuid4().hex[:12]}"

    tokenizer_warnings: list[str] = []
    _expand_case_markers_in_extraction(extraction, tokenizer, args.text, tokenizer_warnings)

    tokens, offsets = tokenizer.tokenize(args.text)
    for item in [extraction.get("trigger", {}), *extraction.get("arguments", {}).values()]:
        if isinstance(item, dict) and isinstance(item.get("span"), list):
            item["char_span"] = char_span_for_tokens(item.get("span"), offsets)

    valid, errors = tokenizer.validate_extraction(args.text, extraction)
    meta = extraction.setdefault("meta", {})
    meta["schema_version"] = PIPELINE_SCHEMA_VERSION
    meta["model"] = Path(args.model).name
    meta["run_id"] = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    meta["valid"] = valid
    if tokenizer_warnings:
        meta.setdefault("warnings", []).extend(tokenizer_warnings)
    if not valid:
        meta.setdefault("warnings", []).extend(errors)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(extraction, ensure_ascii=False) + "\n")

    if valid:
        log(f"Output appended to {out_path}")
    else:
        log(f"Warning: spans mismatch. Output still written to {out_path}")


if __name__ == "__main__":
    main()
