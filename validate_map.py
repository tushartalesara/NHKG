#!/usr/bin/env python3
"""
Milestone 1.3 Validator: Checks if your YAML mapping matches the UHVN Registry.
Usage: python validate_map.py
"""

import json
import yaml  # pip install PyYAML
import sys
from pathlib import Path

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    # Paths
    frames_path = Path("lexicons/uhvn_frames.json")
    map_path = Path("lexicons/uhvn2dbo_map.yaml")

    if not map_path.exists():
        print(f"❌ Error: {map_path} not found. Please create it first.")
        sys.exit(1)

    # Load Data
    print(f"Loading registry from {frames_path}...")
    uhvn = load_json(frames_path)
    
    print(f"Loading mapping from {map_path}...")
    with open(map_path, encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    # Validation Loop
    errors = 0
    warnings = 0

    classes = uhvn.get("classes", {})

    for class_id, map_data in mapping.items():
        if class_id == "DEFAULT": 
            continue

        # 1. Check if Class exists in UHVN
        if class_id not in classes:
            print(f"❌ Error: Class '{class_id}' in YAML does not exist in UHVN registry.")
            errors += 1
            continue

        # 2. Check if Roles exist in that Class
        uhvn_roles = {r["name"] for r in classes[class_id]["thematic_roles"]}
        mapped_roles = map_data.get("roles", {})

        for role_name in mapped_roles:
            if role_name not in uhvn_roles:
                print(f"⚠️  Warning: Role '{role_name}' in mapping for {class_id} is not defined in UHVN XML.")
                print(f"    Available roles: {uhvn_roles}")
                warnings += 1

    # Summary
    print("-" * 40)
    if errors > 0:
        print(f"❌ Validation FAILED with {errors} errors and {warnings} warnings.")
        sys.exit(1)
    else:
        print(f"✅ Validation PASSED. ({len(mapping)} classes mapped, {warnings} warnings)")

if __name__ == "__main__":
    main()