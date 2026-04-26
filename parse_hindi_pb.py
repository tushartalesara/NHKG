#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 1.1 (V3): Robust Hindi PropBank Parser
Fixes: Handles UTF-16/UTF-8 files automatically.
Features: Scrapes 'Rel' examples for robust triggering.

Usage:
  python parse_hindi_pb_v3.py --input data/html_frames --out lexicons/
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

SUFFIX_MAP = {
    "gol": "Destination", "sou": "Source", "src": "Source",
    "loc": "Location", "tmp": "Time", "cau": "Cause",
    "mnr": "Manner", "adv": "Modifier"
}

KEYWORD_MAP = {
    "agent": "Agent", "doer": "Agent", "comer": "Agent", 
    "arriver": "Agent", "entity in motion": "Theme",
    "patient": "Patient", "victim": "Patient", "instrument": "Instrument"
}

def clean_line(text):
    # Remove hidden characters and excessive whitespace
    text = text.replace('\x00', '') # Remove null bytes (common in bad encoding reads)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    return re.sub(r'\s+', ' ', text).strip()

def map_role(arg_label, description):
    arg_clean = arg_label.lower().strip()
    desc_clean = description.lower()
    
    if "-" in arg_clean:
        suffix = arg_clean.split("-")[-1]
        if suffix in SUFFIX_MAP: return SUFFIX_MAP[suffix]
            
    for kw, role in KEYWORD_MAP.items():
        if kw in desc_clean: return role
            
    if "arg0" in arg_clean: return "Agent"
    if "arg1" in arg_clean: return "Theme"
    
    return "Unknown"

def read_file_content(path):
    """
    Tries multiple encodings to read the file correctly.
    """
    encodings = ["utf-8", "utf-16", "cp1252", "utf-8-sig"]
    
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
                # Simple heuristic: If it's valid HTML/Text, it should contain "Predicate" or "Roleset"
                if "Predicate" in content or "Roleset" in content:
                    return content
        except UnicodeError:
            continue
            
    # Fallback: Read binary and decode ignoring errors (last resort)
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_text_content(full_text):
    """
    Parses the text dump of the Hindi PropBank file using regex patterns.
    """
    frames = []
    
    # First collapse whitespace but preserve structure
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # Find all predicate sections
    # Pattern: "Predicate: <name>"
    pred_pattern = r'Predicate:\s*([^\s]+)'
    pred_matches = list(re.finditer(pred_pattern, full_text, re.IGNORECASE))
    
    for i, pred_match in enumerate(pred_matches):
        current_pred = pred_match.group(1).strip()
        
        # Get text from this predicate until the next one (or end)
        start_pos = pred_match.end()
        if i + 1 < len(pred_matches):
            end_pos = pred_matches[i + 1].start()
        else:
            end_pos = len(full_text)
        
        pred_section = full_text[start_pos:end_pos]
        
        # Find all rolesets in this predicate section
        # Pattern: "Roleset id: A.01 , to come"
        roleset_pattern = r'Roleset id:\s*([A-Za-z0-9\._]+)\s*,?\s*(.*?)(?=Roleset id:|$)'
        
        for rs_match in re.finditer(roleset_pattern, pred_section, re.IGNORECASE | re.DOTALL):
            rs_id = rs_match.group(1).strip()
            rs_content = rs_match.group(2).strip()
            
            # Extract description (text between roleset id and "Roles:")
            desc_match = re.match(r'^([^:]*?)\s*(?:Roles:|$)', rs_content, re.IGNORECASE)
            rs_desc = desc_match.group(1).strip() if desc_match else ""
            
            # Find the Roles section
            roles_match = re.search(r'Roles:\s*(.*?)(?=Example:|Roleset id:|Predicate:|$)', rs_content, re.IGNORECASE)
            
            current_roleset = {
                "class_id": rs_id,
                "lemma": current_pred,
                "description": rs_desc,
                "roles": [],
                "example_triggers": []
            }
            
            if roles_match:
                roles_text = roles_match.group(1)
                
                # Find all arguments in the Roles section
                # Pattern: "Arg1 : entity in motion/ 'comer'"
                arg_pattern = r'(Arg\d+[-a-z]*|Argm[-a-z]*)\s*:\s*([^A][^:]*?)(?=Arg\d|Argm|$)'
                
                for arg_match in re.finditer(arg_pattern, roles_text, re.IGNORECASE):
                    raw_arg = arg_match.group(1).strip()
                    desc = arg_match.group(2).strip()
                    
                    # Determine standard name (Agent, Patient, Destination)
                    std_role = map_role(raw_arg, desc)
                    
                    current_roleset["roles"].append({
                        "name": std_role,
                        "raw_arg": raw_arg,
                        "desc": desc
                    })
            
            # Extract example triggers (Rel: fields)
            rel_pattern = r'Rel\s*:\s*([^A-Za-z\d\n]*?)\s*(?=Arg\d|Argm|Example|Roleset|Predicate|$)'
            for rel_match in re.finditer(rel_pattern, rs_content, re.IGNORECASE):
                rel_text = rel_match.group(1).strip()
                
                # CLEANING: Remove brackets [] and extra spaces (annotation artifacts)
                rel_text = rel_text.replace('[', '').replace(']', '')
                rel_text = re.sub(r'\s+', ' ', rel_text).strip()
                
                if rel_text:
                    current_roleset["example_triggers"].append(rel_text)
            
            frames.append(current_roleset)
        
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(in_dir.glob("*.*"))
    all_classes = {}
    
    print(f"Processing {len(files)} files...")
    
    success_count = 0
    
    for fpath in files:
        if fpath.name.startswith('.'): continue # Skip hidden files
        
        # Robust Read
        content = read_file_content(fpath)
        
        if fpath.suffix in ['.html', '.htm']:
            soup = BeautifulSoup(content, "html.parser")
            # Use space separator to keep text on single line for regex parsing
            text_dump = soup.get_text(' ')
        else:
            text_dump = content

        frames = parse_text_content(text_dump)
        
        if frames:
            success_count += 1
            
        for fr in frames:
            cid = fr["class_id"]
            uhvn_cls = {
                "class_id": cid,
                "parent_id": None,
                "members": [{"lemma": fr["lemma"]}],
                "example_triggers": fr["example_triggers"],
                "thematic_roles": [],
                "frames": [{"description": {"primary": fr["description"]}, "syntax": [], "semantics": []}]
            }
            for r in fr["roles"]:
                uhvn_cls["thematic_roles"].append({"name": r["name"], "selrestrs": [], "raw": {"arg": r["raw_arg"], "desc": r["desc"]}})
                
            all_classes[cid] = uhvn_cls

    # Output
    output = {"classes": all_classes, "meta": {"source": "Hindi PropBank V3"}}
    out_path = out_dir / "uhvn_frames.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print(f"Extracted data from {success_count}/{len(files)} files.")
    print(f"Saved {len(all_classes)} frames to {out_path}")

if __name__ == "__main__":
    main()
