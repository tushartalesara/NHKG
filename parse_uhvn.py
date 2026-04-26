#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UHVN (Urdu/Hindi VerbNet) XML -> executable frame registry JSON.

Updates in this version:
1. Inheritance Resolution: Propagates Frames/Roles from Parent -> Child.
2. Lemma Normalization: Strips whitespace/artifacts from member lemmas.
3. Robust Subevent Detection: Preserves predicate order for process modeling.

Usage:
  python parse_uhvn.py --input /path/to/uhvn_xml --out ./lexicons

Outputs:
  ./lexicons/uhvn_frames.json
  ./lexicons/uhvn_roles.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import copy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET


# ---------------------------
# Logging
# ---------------------------

logger = logging.getLogger("uhvn_parser")


# ---------------------------
# Data models
# ---------------------------

@dataclass
class Role:
    name: str
    selrestrs: List[Dict[str, Any]]  # normalized form
    raw: Dict[str, Any]              # raw attributes for traceability


@dataclass
class Predicate:
    value: str
    args: List[Dict[str, Any]]
    subevent: Optional[str] = None   # "init" | "proc" | "res" | None
    raw: Dict[str, Any] = None


@dataclass
class Frame:
    description: Dict[str, Any]
    syntax: List[Dict[str, Any]]
    semantics: List[Predicate]


@dataclass
class UHVNClass:
    class_id: str
    parent_id: Optional[str]
    members: List[Dict[str, Any]]
    thematic_roles: List[Role]
    frames: List[Frame]
    raw: Dict[str, Any]


# ---------------------------
# Helpers
# ---------------------------

def strip_ns(tag: str) -> str:
    """Strip XML namespace prefix if present."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def children(elem: ET.Element, tag_name: str) -> List[ET.Element]:
    """Find children by local tag name, ignoring namespaces."""
    out = []
    for c in list(elem):
        if strip_ns(c.tag) == tag_name:
            out.append(c)
    return out


def first_child(elem: ET.Element, tag_name: str) -> Optional[ET.Element]:
    for c in list(elem):
        if strip_ns(c.tag) == tag_name:
            return c
    return None


def attr(elem: ET.Element, key: str, default: Optional[str] = None) -> Optional[str]:
    return elem.attrib.get(key, default)


def text(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    t = elem.text or ""
    t = t.strip()
    return t or None


def clean_lemma(lemma: Optional[str]) -> Optional[str]:
    """
    Normalize lemmas:
    1. Strip whitespace.
    2. Remove common suffixes like '-n', '-v' if present in raw data.
    """
    if not lemma:
        return None
    lemma = lemma.strip()
    # Example artifact removal (adjust based on actual dirty data inspection)
    # lemma = re.sub(r'-[nv]$', '', lemma) 
    return lemma or None


def norm_subevent(value: Optional[str]) -> Optional[str]:
    """
    Normalize subevent labels to: init/proc/res.
    """
    if not value:
        return None
    v = value.strip().lower()
    mapping = {
        "init": "init",
        "initial": "init",
        "initial_state": "init",
        "inception": "init",
        "proc": "proc",
        "process": "proc",
        "during": "proc",
        "res": "res",
        "result": "res",
        "result_state": "res",
        "final": "res",
    }
    return mapping.get(v, None)


_SUBEVENT_HINT_RE = re.compile(r"\b(init|proc|res)\b", re.IGNORECASE)


def detect_subevent_from_pred(pred_elem: ET.Element) -> Optional[str]:
    """
    Try to detect subevent stage from common attribute names or from predicate value strings.
    """
    for k in ("event", "subevent", "stage", "phase"):
        se = norm_subevent(attr(pred_elem, k))
        if se:
            return se

    v = attr(pred_elem, "value", "") or ""
    m = _SUBEVENT_HINT_RE.search(v)
    if m:
        return norm_subevent(m.group(1))
    return None


def parse_selrestrs(selrestrs_elem: ET.Element) -> List[Dict[str, Any]]:
    logic = (attr(selrestrs_elem, "logic") or "and").lower()
    items = []
    for sr in list(selrestrs_elem):
        if strip_ns(sr.tag) != "SELRESTR":
            continue
        items.append({
            "polarity": attr(sr, "Value"),  # "+" or "-"
            "type": attr(sr, "type"),
            "raw": dict(sr.attrib),
        })
    return [{"logic": logic, "items": items}]


def parse_themroles(class_elem: ET.Element) -> List[Role]:
    themroles = first_child(class_elem, "THEMROLES")
    if themroles is None:
        return []
    roles: List[Role] = []
    for tr in list(themroles):
        if strip_ns(tr.tag) != "THEMROLE":
            continue
        name = attr(tr, "type") or attr(tr, "role") or "UNKNOWN_ROLE"
        sel: List[Dict[str, Any]] = []
        selrestrs_elem = first_child(tr, "SELRESTRS")
        if selrestrs_elem is not None:
            sel = parse_selrestrs(selrestrs_elem)
        roles.append(Role(name=name, selrestrs=sel, raw=dict(tr.attrib)))
    return roles


def parse_members(class_elem: ET.Element) -> List[Dict[str, Any]]:
    members_elem = first_child(class_elem, "MEMBERS")
    if members_elem is None:
        return []
    out = []
    for m in list(members_elem):
        if strip_ns(m.tag) != "MEMBER":
            continue
        
        raw_lemma = attr(m, "name") or attr(m, "lemma") or attr(m, "word")
        if raw_lemma is None:
            raw_lemma = text(m)
            
        lemma = clean_lemma(raw_lemma)
        if lemma:
            out.append({
                "lemma": lemma,
                "raw": dict(m.attrib),
            })
    return out


def parse_syntax(frame_elem: ET.Element) -> List[Dict[str, Any]]:
    syntax_elem = first_child(frame_elem, "SYNTAX")
    if syntax_elem is None:
        return []
    out = []
    for n in list(syntax_elem):
        out.append({
            "tag": strip_ns(n.tag),
            "attrib": dict(n.attrib),
            "text": text(n),
        })
    return out


def parse_semantics(frame_elem: ET.Element) -> List[Predicate]:
    sem_elem = first_child(frame_elem, "SEMANTICS")
    if sem_elem is None:
        return []
    preds: List[Predicate] = []
    for p in list(sem_elem):
        if strip_ns(p.tag) != "PRED":
            continue
        pred_value = attr(p, "value") or attr(p, "name") or "UNKNOWN_PRED"
        subevent = detect_subevent_from_pred(p)
        args = []
        for a in list(p):
            if strip_ns(a.tag) != "ARGS":
                continue
            for arg_elem in list(a):
                if strip_ns(arg_elem.tag) != "ARG":
                    continue
                args.append({
                    "type": attr(arg_elem, "type"),     # e.g., "Agent", "Theme"
                    "value": attr(arg_elem, "value"),   # e.g., "Killer", "Victim"
                    "raw": dict(arg_elem.attrib),
                })
        preds.append(Predicate(value=pred_value, args=args, subevent=subevent, raw=dict(p.attrib)))
    return preds


def parse_frames(class_elem: ET.Element) -> List[Frame]:
    frames_elem = first_child(class_elem, "FRAMES")
    if frames_elem is None:
        return []
    out: List[Frame] = []
    for fr in list(frames_elem):
        if strip_ns(fr.tag) != "FRAME":
            continue
        desc = first_child(fr, "DESCRIPTION")
        description = dict(desc.attrib) if desc is not None else {}
        out.append(Frame(
            description=description,
            syntax=parse_syntax(fr),
            semantics=parse_semantics(fr),
        ))
    return out


def extract_class_id(elem: ET.Element) -> Optional[str]:
    for k in ("ID", "id", "VNCLASSID", "vnclassid", "name"):
        v = attr(elem, k)
        if v:
            return v
    return None


def find_classes(root: ET.Element) -> List[ET.Element]:
    classes = []
    for elem in root.iter():
        t = strip_ns(elem.tag).upper()
        if t in ("VNCLASS", "VNSUBCLASS", "CLASS", "SUBCLASS"):
            classes.append(elem)
    return classes


def parse_class_tree(class_elem: ET.Element, parent_id: Optional[str], acc: Dict[str, UHVNClass]) -> None:
    cid = extract_class_id(class_elem)
    if not cid:
        cid = f"NO_ID::{id(class_elem)}"
        logger.warning("Class without ID detected; using synthetic id=%s", cid)

    uclass = UHVNClass(
        class_id=cid,
        parent_id=parent_id,
        members=parse_members(class_elem),
        thematic_roles=parse_themroles(class_elem),
        frames=parse_frames(class_elem),
        raw={"tag": strip_ns(class_elem.tag), "attrib": dict(class_elem.attrib)},
    )
    acc[cid] = uclass

    for child in list(class_elem):
        t = strip_ns(child.tag).upper()
        if t in ("VNSUBCLASS", "SUBCLASS"):
            parse_class_tree(child, cid, acc)


def parse_uhvn_xml_file(path: Path) -> Dict[str, UHVNClass]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise RuntimeError(f"XML parse error in {path}: {e}") from e

    acc: Dict[str, UHVNClass] = {}

    top_level = []
    for c in list(root):
        if strip_ns(c.tag).upper() in ("VNCLASS", "CLASS"):
            top_level.append(c)

    if top_level:
        for cls in top_level:
            parse_class_tree(cls, parent_id=None, acc=acc)
        return acc

    all_classes = find_classes(root)
    if not all_classes:
        return acc

    def has_class_ancestor(e: ET.Element) -> bool:
        for cand in all_classes:
            for sub in list(cand):
                if sub is e:
                    return True
        return False

    roots = []
    for e in all_classes:
        if strip_ns(e.tag).upper() in ("VNCLASS", "CLASS") and not has_class_ancestor(e):
            roots.append(e)

    if not roots:
        roots = all_classes[:1]

    for r in roots:
        parse_class_tree(r, parent_id=None, acc=acc)

    return acc


def resolve_inheritance(classes: Dict[str, UHVNClass]) -> None:
    """
    Resolve inheritance for roles and frames.
    If a child class lacks roles or frames, it inherits them from the parent.
    We iterate until no more changes occur to handle multi-level depth.
    """
    logger.info("Resolving inheritance (parent -> child propagation)...")
    
    # Simple fixed-point iteration for deep hierarchies
    # (Depth is usually small < 5, so this is efficient enough)
    MAX_ITER = 10
    
    for _ in range(MAX_ITER):
        changed = False
        # Sort keys to ensure deterministic processing order
        for cid in sorted(classes.keys()):
            cls = classes[cid]
            
            # Skip if no parent or parent unknown
            if not cls.parent_id or cls.parent_id not in classes:
                continue
            
            parent = classes[cls.parent_id]
            
            # 1. Inherit Roles
            # Strategy: if child has NO roles, copy parent's.
            # (Some VerbNets merge roles, but for Hindi UHVN MVP, copy-on-missing is safer)
            if not cls.thematic_roles and parent.thematic_roles:
                cls.thematic_roles = copy.deepcopy(parent.thematic_roles)
                changed = True
                
            # 2. Inherit Frames
            # Strategy: if child has NO frames, copy parent's.
            if not cls.frames and parent.frames:
                cls.frames = copy.deepcopy(parent.frames)
                changed = True
        
        if not changed:
            break
    else:
        logger.warning("Inheritance resolution reached MAX_ITER. Check for cycles.")


def merge_role_inventory(classes: Dict[str, UHVNClass]) -> Dict[str, Dict[str, Any]]:
    role_inv: Dict[str, Dict[str, Any]] = {}
    for c in classes.values():
        for r in c.thematic_roles:
            if r.name not in role_inv:
                role_inv[r.name] = {
                    "name": r.name,
                    "selrestrs_examples": [],
                    "seen_in_classes": [],
                }
            if r.selrestrs:
                role_inv[r.name]["selrestrs_examples"].append(r.selrestrs)
            role_inv[r.name]["seen_in_classes"].append(c.class_id)

    for rn, info in role_inv.items():
        info["seen_in_classes"] = sorted(set(info["seen_in_classes"]))
    return role_inv


def predicate_subevent_stats(classes: Dict[str, UHVNClass]) -> Dict[str, Any]:
    total = 0
    with_stage = 0
    by_stage = {"init": 0, "proc": 0, "res": 0}
    for c in classes.values():
        for fr in c.frames:
            for p in fr.semantics:
                total += 1
                if p.subevent in by_stage:
                    with_stage += 1
                    by_stage[p.subevent] += 1
    return {"total_preds": total, "with_subevent": with_stage, "by_stage": by_stage}


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory containing UHVN XML files")
    ap.add_argument("--out", required=True, help="Output directory for JSON registries")
    ap.add_argument("--glob", default="**/*.xml", help="Glob pattern to find XML files")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(in_dir.glob(args.glob))
    if not xml_files:
        raise SystemExit(f"No XML files found in {in_dir} with glob {args.glob}")

    all_classes: Dict[str, UHVNClass] = {}

    # 1. Parse all files
    for xf in xml_files:
        logger.info("Parsing %s", xf)
        classes = parse_uhvn_xml_file(xf)
        for cid, cls in classes.items():
            if cid in all_classes:
                logger.warning("Duplicate class_id %s; keeping first. file=%s", cid, xf)
                continue
            all_classes[cid] = cls
            
    # 2. Resolve Inheritance (Fill missing holes in children)
    resolve_inheritance(all_classes)

    # 3. Serialize Output
    frames_out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(in_dir),
            "num_files": len(xml_files),
            "num_classes": len(all_classes),
            "subevent_stats": predicate_subevent_stats(all_classes),
        },
        "classes": {cid: asdict(cls) for cid, cls in all_classes.items()},
    }

    roles_out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_roles": None,
        },
        "roles": merge_role_inventory(all_classes),
    }
    roles_out["meta"]["num_roles"] = len(roles_out["roles"])

    frames_path = out_dir / "uhvn_frames.json"
    roles_path = out_dir / "uhvn_roles.json"
    frames_path.write_text(json.dumps(frames_out, ensure_ascii=False, indent=2), encoding="utf-8")
    roles_path.write_text(json.dumps(roles_out, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Wrote: %s", frames_path)
    logger.info("Wrote: %s", roles_path)


if __name__ == "__main__":
    main()