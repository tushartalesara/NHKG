#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 1.2: PropBank -> UHVN Mapping Generator (Stubber)

Usage:
  python build_pb_map.py --uhvn lexicons/uhvn_frames.json --out lexicons/pb2uhvn_map.json

What it does:
  1. Reads your UHVN frames.
  2. For each verb member, creates a "Mock" PropBank roleset (e.g., 'marna.01').
  3. Guesses a mapping (Arg0->Agent, Arg1->Theme) based on common defaults.
  4. Writes a JSON map you can manually edit later.
"""

import argparse
import json
import logging
from pathlib import Path

# Common heuristic defaults for Hindi/PropBank
# We assume Arg0 is usually the "Doer" and Arg1 is the "Undergoer"
DEFAULT_ROLE_MAP = {
    "Arg0": ["Agent", "Causer", "Actor"],
    "Arg1": ["Patient", "Theme", "Beneficiary"],
    "Arg2": ["Instrument", "Source", "Destination"],
    "ArgM-LOC": ["Location"],
    "ArgM-TMP": ["Time"]
}

def load_uhvn(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def guess_mapping(roles):
    """
    Tries to map UHVN roles to PropBank Args based on simple heuristics.
    Returns a dict like {"Arg0": "Agent", "Arg1": "Patient"}
    """
    mapping = {}
    
    # Get list of role names available in this UHVN class
    available_roles = {r['name'] for r in roles}
    
    # 1. Assign Arg0 (The Agent-like role)
    for candidate in DEFAULT_ROLE_MAP["Arg0"]:
        if candidate in available_roles:
            mapping["Arg0"] = candidate
            available_roles.remove(candidate)
            break
            
    # 2. Assign Arg1 (The Patient-like role)
    for candidate in DEFAULT_ROLE_MAP["Arg1"]:
        if candidate in available_roles:
            mapping["Arg1"] = candidate
            available_roles.remove(candidate)
            break
            
    # 3. Assign Arg2 if any roles remain (Instrument/Source usually)
    for candidate in DEFAULT_ROLE_MAP["Arg2"]:
        if candidate in available_roles:
            mapping["Arg2"] = candidate
            break

    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uhvn", required=True, help="Path to uhvn_frames.json")
    ap.add_argument("--out", required=True, help="Output path for pb2uhvn_map.json")
    args = ap.parse_args()

    uhvn_data = load_uhvn(args.uhvn)
    classes = uhvn_data.get("classes", {})
    
    pb_map = {}
    
    print(f"Generating mappings for {len(classes)} classes...")

    for cid, cls_data in classes.items():
        # 1. Get the members (verbs)
        members = cls_data.get("members", [])
        if not members:
            continue
            
        # 2. Get the roles
        roles = cls_data.get("thematic_roles", [])
        
        # 3. Guess the mapping once for the class
        class_mapping = guess_mapping(roles)
        
        # 4. Create an entry for every member verb
        for m in members:
            lemma = m["lemma"]
            # Create a synthetic roleset ID. In real PropBank, this might be 'marna.01'
            roleset_id = f"{lemma}.01"
            
            pb_map[roleset_id] = {
                "lemma": lemma,
                "uhvn_class": cid,
                "mapping": class_mapping
            }

    # Output
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pb_map, f, indent=2, ensure_ascii=False)
        
    print(f"Success! Wrote {len(pb_map)} mappings to {out_path}")
    print("NOTE: Open this file and verify the 'mapping' fields are correct.")

if __name__ == "__main__":
    main()