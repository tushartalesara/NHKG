#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 2.1: JSON Schema Generator (Automated Descriptions)
Automatically injects semantic definitions into schemas.

Usage:
  python gen_schemas.py --uhvn lexicons/uhvn_frames.json --map lexicons/uhvn2dbo_map.yaml --out schemas/
"""

import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------
# Standard Role Definitions (The Automation)
# ---------------------------
# These descriptions will be injected into EVERY frame that uses these roles.
ROLE_DESCRIPTIONS = {
    "Agent": "The entity that performs the action or causes the event (Hindi: Karta / Doer). Usually marked by 'ne'.",
    "Patient": "The entity that undergoes the action or is affected by it (Hindi: Karma / Victim). Usually marked by 'ko'.",
    "Theme": "The entity that is moved or described (Hindi: Vishay).",
    "Instrument": "The tool or method used to perform the action (Hindi: Karan / Tool). Usually marked by 'se'.",
    "Destination": "The place the entity ends up at (Hindi: Gantavya / Goal). Usually marked by 'ko' or 'mein'.",
    "Source": "The place the entity comes from (Hindi: Srot). Usually marked by 'se'.",
    "Time": "The time when the event happens (Hindi: Samay).",
    "Location": "The place where the event happens (Hindi: Sthaan).",
    "Beneficiary": "The entity for whose benefit the action is performed (Hindi: Hitadhikari)."
}

BASE_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "Document Identifier"},
        "sent_id": {"type": "integer", "description": "Sentence Index"},
        "frame": {"type": "string", "const": "PLACEHOLDER_CLASS_ID"},
        "trigger": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "span": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Token indices [start, end)"
                },
                "char_span": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Character offsets [start, end)"
                }
            },
            "required": ["text", "span"]
        },
        "arguments": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "required": []
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "meta": {
            "type": "object",
            "properties": {
                "schema_version": {"type": "string"},
                "model": {"type": "string"},
                "run_id": {"type": "string"},
                "frame_selection": {"type": "string"},
                "frame_candidates": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}}
            },
            "required": []
        }
    },
    "required": ["doc_id", "frame", "trigger", "arguments"]
}

ARG_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "span": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2
        },
        "char_span": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Character offsets [start, end)"
        }
    },
    "required": ["text", "span"]
}

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path):
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_schema_for_class(cid: str, cls_data: Dict, mapping_data: Dict) -> Dict:
    schema = json.loads(json.dumps(BASE_SCHEMA))

    # Set Frame Const
    schema["properties"]["frame"]["const"] = cid

    # Description for the LLM
    lemmas = [m["lemma"] for m in cls_data.get("members", [])[:5]]
    lemma_str = ", ".join(lemmas)
    schema["description"] = f"Extraction frame for '{cid}' events (e.g., {lemma_str})."

    # Build Arguments
    roles = cls_data.get("thematic_roles", [])
    mapped_roles = mapping_data.get("roles", {})

    for role in roles:
        role_name = role["name"]

        arg_def = json.loads(json.dumps(ARG_SCHEMA))

        # 1. Get Base Description from our Dictionary
        base_desc = ROLE_DESCRIPTIONS.get(role_name, f"Role: {role_name}")

        # 2. Append Mapping Info (if available)
        map_info = mapped_roles.get(role_name)
        if map_info:
            target_uri = map_info.get("uri") if isinstance(map_info, dict) else map_info
            base_desc += f" (Maps to {target_uri})"

        arg_def["description"] = base_desc

        # Add to properties
        schema["properties"]["arguments"]["properties"][role_name] = arg_def

    return schema

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uhvn", required=True, help="Path to uhvn_frames.json")
    ap.add_argument("--map", default="lexicons/uhvn2dbo_map.yaml", help="Path to mapping YAML")
    ap.add_argument("--out", required=True, help="Output directory for schemas")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    uhvn = load_json(args.uhvn)
    mapping = load_yaml(args.map)

    classes = uhvn.get("classes", {})
    print(f"Generating schemas for {len(classes)} classes...")

    count = 0
    for cid, cls_data in classes.items():
        cls_map = mapping.get(cid, {})
        schema = generate_schema_for_class(cid, cls_data, cls_map)

        out_path = out_dir / f"{cid}.schema.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        count += 1

    print(f"Generated {count} smart schemas in {out_dir}")

if __name__ == "__main__":
    main()
