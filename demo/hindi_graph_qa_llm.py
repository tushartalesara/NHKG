#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from hindi_graph_qa_demo import (
    SAMPLE_QUESTIONS,
    EventRecord,
    clean_text,
    load_graph_events,
    parse_question,
    rank_actor_events,
    rank_entity_events,
    rank_theme_events,
)

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError as exc:
    raise SystemExit("llama-cpp-python is required for the LLM + KG demo.") from exc


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_hindi": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "grounded": {"type": "boolean"},
        "evidence_numbers": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["answer_hindi", "confidence", "grounded", "evidence_numbers"],
}


def retrieve_events(events: Sequence[EventRecord], question: str, top_k: int) -> List[EventRecord]:
    intent = parse_question(question)
    if intent.kind in {"what_did", "actor_with_trigger"}:
        return rank_actor_events(events, intent.entity, intent.trigger_hint)[:top_k]
    if intent.kind == "where":
        ranked = rank_actor_events(events, intent.entity)[: max(top_k * 2, top_k)]
        location_hits = []
        for event in ranked:
            if (
                event.roles.get("Destination")
                or event.roles.get("Location")
                or event.roles.get("Source")
                or event.roles.get("Theme")
            ):
                location_hits.append(event)
        return location_hits[:top_k]
    if intent.kind == "who_did":
        return rank_theme_events(events, intent.entity, intent.trigger_hint)[:top_k]
    return rank_entity_events(events, intent.entity or clean_text(question))[:top_k]


def format_event_evidence(index: int, event: EventRecord) -> str:
    role_parts = []
    for role_name in ("Actor", "Theme", "Destination", "Location", "Source"):
        values = event.roles.get(role_name, [])
        if values:
            role_parts.append(f"{role_name}={', '.join(values)}")
    role_text = "; ".join(role_parts) if role_parts else "??? ?????? ?????? ????"
    return (
        f"[{index}] ?????: {event.sentence}\n"
        f"    ??????: {event.trigger or event.label or '??????'}\n"
        f"    ????????: {role_text}"
    )


def build_prompt(question: str, evidence_block: str) -> str:
    return f"""<|system|>
You are a Hindi question answering assistant that must answer only from supplied event-graph evidence.

Rules:
1. Answer in Hindi only.
2. Use only the supplied evidence. Do not invent facts.
3. If the evidence is weak or incomplete, say that clearly.
4. Prefer short factual answers.
5. If there are multiple relevant events, summarize them briefly.
6. Return strictly valid JSON following the provided schema.

<|user|>
??????: {question}

?????? ?????-???????:
{evidence_block}

????? ???? ???:
- answer_hindi ??? ????????? ????? ???
- confidence ?? high/medium/low ??? ???
- grounded ?? true ???? ??? ????? ???????-?????? ??, ???? ?? false
- evidence_numbers ??? ????? ??? ?? ??????? ??????? ???

<|assistant|>
"""


def ensure_model_path(path_arg: str) -> str:
    model_path = path_arg or os.environ.get("NHKG_QA_MODEL", "").strip()
    if not model_path:
        raise SystemExit(
            "Model path required. Pass --model PATH_TO_GGUF or set NHKG_QA_MODEL."
        )
    if not Path(model_path).exists():
        raise SystemExit(f"Model file not found: {model_path}")
    return model_path


def generate_answer(
    llm: Llama,
    grammar: LlamaGrammar,
    question: str,
    retrieved_events: Sequence[EventRecord],
    *,
    max_tokens: int,
    temperature: float,
) -> dict:
    if not retrieved_events:
        return {
            "answer_hindi": "???? ?? ?????? ?? ??? ????? ??? ???????? ??????? ???? ?????",
            "confidence": "low",
            "grounded": False,
            "evidence_numbers": [],
        }

    evidence_block = "\n".join(
        format_event_evidence(index, event)
        for index, event in enumerate(retrieved_events, start=1)
    )
    prompt = build_prompt(question, evidence_block)
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        grammar=grammar,
        stop=["<|user|>", "</s>"],
    )
    raw_text = output["choices"][0]["text"]
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = {
            "answer_hindi": raw_text.strip() or "???? ?? ??? ????? ???? ?????",
            "confidence": "low",
            "grounded": False,
            "evidence_numbers": [],
        }
    return payload


def print_examples() -> None:
    print("LLM + KG ???? ?? ??? ????? ?? ??????:")
    for question in SAMPLE_QUESTIONS:
        print(f"- {question}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM + KG Hindi QnA demo over the NHKG graph.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("final_outputs/demo_graph/canonical_demo_graph.nq"),
        help="Path to the N-Quads graph file.",
    )
    parser.add_argument("--model", type=str, default="", help="Path to the GGUF model file. Falls back to NHKG_QA_MODEL.")
    parser.add_argument("--question", type=str, default="", help="Hindi question to answer.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved graph events to pass to the LLM.")
    parser.add_argument("--ctx", type=int, default=2048, help="Context window for llama.cpp.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="n_gpu_layers passed to llama_cpp.")
    parser.add_argument("--examples", action="store_true", help="Print sample Hindi questions and exit.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.examples:
        print_examples()
        return

    if not args.graph.exists():
        raise SystemExit(f"Graph file not found: {args.graph}")

    model_path = ensure_model_path(args.model)
    events = load_graph_events(args.graph)
    grammar = LlamaGrammar.from_json_schema(json.dumps(RESPONSE_SCHEMA))
    llm = Llama(
        model_path=model_path,
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    def run_once(question: str) -> None:
        retrieved = retrieve_events(events, question, top_k=max(1, args.top_k))
        payload = generate_answer(
            llm,
            grammar,
            question,
            retrieved,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n?????:")
        print(payload.get("answer_hindi", "").strip())
        print(f"\nconfidence: {payload.get('confidence', 'low')}")
        print(f"grounded: {payload.get('grounded', False)}")
        evidence_numbers = payload.get("evidence_numbers", [])
        if evidence_numbers:
            print("\n???????? ???????:")
            for index in evidence_numbers:
                if isinstance(index, int) and 1 <= index <= len(retrieved):
                    print(format_event_evidence(index, retrieved[index - 1]))
        elif retrieved:
            print("\n????? ???????????? ???????:")
            for index, event in enumerate(retrieved, start=1):
                print(format_event_evidence(index, event))

    if args.question:
        run_once(args.question)
        return

    print("NHKG LLM + KG Hindi QnA Demo")
    print("???? ???? ?? ??? 'exit' ?? 'quit' ??????")
    print_examples()
    while True:
        try:
            question = input("\n?????? > ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        run_once(question)


if __name__ == "__main__":
    main()
