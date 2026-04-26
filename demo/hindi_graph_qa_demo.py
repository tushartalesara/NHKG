#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from align.enrichment_common import parse_nquad_line


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SCHEMA_EVENT = "https://schema.org/Event"
PROV_DERIVED = "http://www.w3.org/ns/prov#wasDerivedFrom"
NIF_ANCHOR = "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf"
HAS_TRIGGER = "http://ns.nhkg.org/resource/hasTrigger"

ROLE_MAP = {
    "https://schema.org/actor": "Actor",
    "https://schema.org/location": "Location",
}

TRAILING_PUNCT_RE = re.compile(r"[?।!\s]+$")
LEADING_TRAILING_SPACE_RE = re.compile(r"\s+")

POSTPOSITIONS = (
    "के लिए",
    "के पास",
    "के साथ",
    "की ओर",
    "के ऊपर",
    "के नीचे",
    "के भीतर",
    "में",
    "से",
    "को",
    "पर",
    "तक",
    "ने",
)

SAMPLE_QUESTIONS = [
    "राम कहाँ गए?",
    "राम से जुड़ी घटनाएँ क्या हैं?",
    "मोहन ने क्या दिया?",
    "माँ ने क्या बनाया?",
    "पिताजी क्या लाए?",
    "रवि ने क्या बेचा?",
    "चोर को किसने पकड़ा?",
    "पुलिस ने क्या किया?",
    "न्यायाधीश ने क्या किया?",
]


@dataclass
class EventRecord:
    event_id: str
    label: str
    trigger: str
    sentence: str
    roles: Dict[str, List[str]]


@dataclass
class QuestionIntent:
    kind: str
    entity: str = ""
    trigger_hint: str = ""


def clean_text(text: str) -> str:
    trimmed = TRAILING_PUNCT_RE.sub("", str(text or "").strip())
    return LEADING_TRAILING_SPACE_RE.sub(" ", trimmed).strip()


def normalize_entity(text: str) -> str:
    value = clean_text(text)
    for suffix in sorted(POSTPOSITIONS, key=len, reverse=True):
        token = f" {suffix}"
        if value.endswith(token):
            value = value[: -len(token)].strip()
            break
    return value


def normalize_for_match(text: str) -> str:
    return normalize_entity(text).replace(" ", "")


def role_name_from_predicate(predicate: str) -> Optional[str]:
    if predicate in ROLE_MAP:
        return ROLE_MAP[predicate]
    if predicate.startswith("http://ns.nhkg.org/uhvn/"):
        return predicate.rsplit("/", 1)[-1]
    return None


def load_graph_events(graph_path: Path) -> List[EventRecord]:
    labels: Dict[str, str] = {}
    anchors: Dict[str, str] = {}
    event_ids = set()
    object_triples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with graph_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            quad = parse_nquad_line(raw_line)
            if not quad:
                continue
            subject = quad.subject
            predicate = quad.predicate
            obj = quad.obj
            if obj.kind == "literal":
                if predicate == RDFS_LABEL:
                    labels[subject] = str(obj.value)
                elif predicate == NIF_ANCHOR:
                    anchors[subject] = str(obj.value)
                continue

            object_triples[subject].append((predicate, obj.value))
            if predicate == RDF_TYPE and obj.value == SCHEMA_EVENT:
                event_ids.add(subject)

    raw_events: List[EventRecord] = []
    for event_id in sorted(event_ids):
        trigger = ""
        sentence = ""
        roles: Dict[str, List[str]] = defaultdict(list)
        for predicate, obj_uri in object_triples[event_id]:
            if predicate == HAS_TRIGGER:
                trigger = anchors.get(obj_uri, labels.get(obj_uri, ""))
            elif predicate == PROV_DERIVED:
                sentence = labels.get(obj_uri, "")
            else:
                role = role_name_from_predicate(predicate)
                if not role:
                    continue
                value = anchors.get(obj_uri, labels.get(obj_uri, ""))
                if value:
                    roles[role].append(value)

        raw_events.append(
            EventRecord(
                event_id=event_id,
                label=labels.get(event_id, ""),
                trigger=clean_text(trigger or labels.get(event_id, "")),
                sentence=clean_text(sentence),
                roles={key: list(dict.fromkeys(values)) for key, values in roles.items()},
            )
        )

    return dedupe_events(raw_events)


def event_quality_score(event: EventRecord) -> Tuple[int, int, int]:
    informative_roles = sum(len(event.roles.get(name, [])) for name in ("Actor", "Theme", "Destination", "Location", "Source"))
    trigger_clean = 0 if any(normalize_for_match(event.trigger) == normalize_for_match(value) for values in event.roles.values() for value in values) else 1
    return (trigger_clean, informative_roles, len(event.trigger))


def dedupe_events(events: Sequence[EventRecord]) -> List[EventRecord]:
    best_by_signature: Dict[Tuple[str, str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]], EventRecord] = {}
    for event in events:
        signature = (
            event.sentence,
            normalize_for_match(event.trigger),
            tuple(sorted(normalize_for_match(value) for value in event.roles.get("Actor", []))),
            tuple(sorted(normalize_for_match(value) for value in event.roles.get("Theme", []))),
            tuple(sorted(normalize_for_match(value) for value in event.roles.get("Destination", []))),
            tuple(sorted(normalize_for_match(value) for value in event.roles.get("Location", []))),
            tuple(sorted(normalize_for_match(value) for value in event.roles.get("Source", []))),
        )
        current = best_by_signature.get(signature)
        if current is None or event_quality_score(event) > event_quality_score(current):
            best_by_signature[signature] = event
    return list(best_by_signature.values())


def parse_question(question: str) -> QuestionIntent:
    q = clean_text(question)

    patterns = [
        (r"^(?P<entity>.+?)\s+कहाँ\s+.+$", "where"),
        (r"^(?P<theme>.+?)\s+को\s+किसने\s+(?P<trigger>.+)$", "who_did"),
        (r"^(?P<theme>.+?)\s+किसने\s+(?P<trigger>.+)$", "who_did"),
        (r"^(?P<entity>.+?)\s+से\s+जुड़ी\s+घटनाएँ\s+क्या\s+हैं$", "list_events"),
        (r"^(?P<entity>.+?)\s+से\s+संबंधित\s+घटनाएँ\s+क्या\s+हैं$", "list_events"),
        (r"^(?P<entity>.+?)\s+ने\s+क्या\s+किया$", "what_did"),
        (r"^(?P<entity>.+?)\s+क्या\s+(?P<trigger>.+)$", "actor_with_trigger"),
        (r"^(?P<entity>.+?)\s+ने\s+क्या\s+(?P<trigger>.+)$", "actor_with_trigger"),
    ]

    for pattern, kind in patterns:
        match = re.match(pattern, q)
        if not match:
            continue
        groups = match.groupdict()
        entity = normalize_entity(groups.get("entity") or groups.get("theme") or "")
        trigger_hint = clean_text(groups.get("trigger") or "")
        return QuestionIntent(kind=kind, entity=entity, trigger_hint=trigger_hint)

    fallback = normalize_entity(q.replace("क्या हुआ", "").strip())
    return QuestionIntent(kind="fallback", entity=fallback)


def values_match(values: Iterable[str], target: str) -> bool:
    target_norm = normalize_for_match(target)
    return any(normalize_for_match(value) == target_norm for value in values)


def contains_trigger(event: EventRecord, trigger_hint: str) -> bool:
    if not trigger_hint:
        return True
    hint_norm = normalize_for_match(trigger_hint)
    trigger_norm = normalize_for_match(event.trigger)
    return hint_norm in trigger_norm or trigger_norm in hint_norm


def summarize_event(event: EventRecord) -> str:
    parts = [f"ट्रिगर: {event.trigger or event.label or 'अज्ञात'}"]
    for role_name in ("Actor", "Theme", "Destination", "Location", "Source"):
        values = event.roles.get(role_name, [])
        if values:
            parts.append(f"{role_name}: {', '.join(values)}")
    parts.append(f"साक्ष्य: {event.sentence}")
    return " | ".join(parts)


def rank_actor_events(events: Sequence[EventRecord], actor: str, trigger_hint: str = "") -> List[EventRecord]:
    ranked = []
    for event in events:
        if not values_match(event.roles.get("Actor", []), actor):
            continue
        score = 10
        if contains_trigger(event, trigger_hint):
            score += 5
        if event.roles.get("Theme"):
            score += 2
        if event.roles.get("Destination") or event.roles.get("Location") or event.roles.get("Source"):
            score += 2
        if normalize_for_match(event.trigger) == normalize_for_match(actor):
            score -= 4
        ranked.append((score, event_quality_score(event), event))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def rank_theme_events(events: Sequence[EventRecord], theme: str, trigger_hint: str = "") -> List[EventRecord]:
    ranked = []
    for event in events:
        if not values_match(event.roles.get("Theme", []), theme):
            continue
        score = 10
        if contains_trigger(event, trigger_hint):
            score += 5
        if event.roles.get("Actor"):
            score += 3
        ranked.append((score, event_quality_score(event), event))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def rank_entity_events(events: Sequence[EventRecord], entity: str) -> List[EventRecord]:
    ranked = []
    entity_norm = normalize_for_match(entity)
    for event in events:
        hit = False
        score = 0
        for role_name, values in event.roles.items():
            for value in values:
                if normalize_for_match(value) == entity_norm:
                    hit = True
                    score += 3 if role_name == "Actor" else 2
        if entity_norm and entity_norm in normalize_for_match(event.trigger):
            hit = True
            score += 1
        if not hit:
            continue
        ranked.append((score, event_quality_score(event), event))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def answer_question(events: Sequence[EventRecord], question: str, top_k: int = 3) -> str:
    intent = parse_question(question)

    if intent.kind in {"what_did", "actor_with_trigger"}:
        matches = rank_actor_events(events, intent.entity, intent.trigger_hint)[:top_k]
        if not matches:
            return f'मुझे "{intent.entity}" के लिए इस ग्राफ में साफ उत्तर नहीं मिला।'
        lines = [f'ग्राफ के अनुसार, "{intent.entity}" से जुड़ी सबसे उपयुक्त घटनाएँ:']
        lines.extend(f"- {summarize_event(event)}" for event in matches)
        return "\n".join(lines)

    if intent.kind == "where":
        matches = rank_actor_events(events, intent.entity)[:top_k]
        location_hits = []
        for event in matches:
            destinations = (
                event.roles.get("Destination", [])
                + event.roles.get("Location", [])
                + event.roles.get("Source", [])
                + event.roles.get("Theme", [])
            )
            if destinations:
                location_hits.append((event, destinations))
        if not location_hits:
            return f'मुझे "{intent.entity}" के लिए इस ग्राफ में कोई साफ स्थान/गंतव्य उत्तर नहीं मिला।'
        lines = [f'ग्राफ के अनुसार, "{intent.entity}" से जुड़ी स्थान-सम्बंधित घटनाएँ:']
        for event, destinations in location_hits[:top_k]:
            lines.append(f"- गंतव्य/स्थान: {', '.join(destinations)} | ट्रिगर: {event.trigger} | साक्ष्य: {event.sentence}")
        return "\n".join(lines)

    if intent.kind == "who_did":
        matches = rank_theme_events(events, intent.entity, intent.trigger_hint)[:top_k]
        if not matches:
            return f'मुझे "{intent.entity}" थीम वाली घटना के लिए इस ग्राफ में कोई साफ actor नहीं मिला।'
        lines = [f'ग्राफ के अनुसार, "{intent.entity}" से जुड़ी घटना के संभावित actor:']
        for event in matches:
            actors = event.roles.get("Actor", [])
            actor_text = ", ".join(actors) if actors else "अज्ञात"
            lines.append(f"- actor: {actor_text} | ट्रिगर: {event.trigger} | साक्ष्य: {event.sentence}")
        return "\n".join(lines)

    if intent.kind in {"list_events", "fallback"}:
        matches = rank_entity_events(events, intent.entity)[:top_k]
        if not matches:
            return f'मुझे "{intent.entity or clean_text(question)}" के लिए इस ग्राफ में कोई स्पष्ट घटना नहीं मिली।'
        lines = [f'ग्राफ के अनुसार, "{intent.entity}" से जुड़ी घटनाएँ:']
        lines.extend(f"- {summarize_event(event)}" for event in matches)
        return "\n".join(lines)

    return "मैं अभी इस प्रश्न को इस डेमो में नहीं समझ पाया।"


def print_examples() -> None:
    print("प्रोफेसर डेमो के लिए सुझाए गए प्रश्न:")
    for question in SAMPLE_QUESTIONS:
        print(f"- {question}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hindi QnA demo over the NHKG event graph.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("final_outputs/demo_graph/canonical_demo_graph.nq"),
        help="Path to the N-Quads graph file.",
    )
    parser.add_argument("--question", type=str, default="", help="Hindi question to answer.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum number of answers to show.")
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

    events = load_graph_events(args.graph)

    if args.question:
        print(answer_question(events, args.question, top_k=max(1, args.top_k)))
        return

    print("NHKG Hindi Graph QnA Demo")
    print("खत्म करने के लिए 'exit' या 'quit' लिखें।")
    print_examples()
    while True:
        try:
            question = input("\nप्रश्न > ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        print(answer_question(events, question, top_k=max(1, args.top_k)))


if __name__ == "__main__":
    main()
