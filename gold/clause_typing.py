#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lightweight clause typing for precision-first candidate suppression."""

from __future__ import annotations

from typing import Dict, List, Sequence


HELPER_LIKE_CATEGORIES = {"auxiliary", "copula", "particle", "adp", "conjunction", "pronoun", "determiner", "closed_class"}
RECOVER_NOW_CLAUSE_TYPES = {
    "imperative_predicate",
    "change_of_state_resultative",
    "modal_lexical_event",
    "light_verb_compound_event",
}
COPULAR_STATE_TYPES = {
    "predicative_adjectival_state",
    "copular_nominal_classification",
    "static_copular_identity",
}
LEXICAL_CENTER_CATEGORIES = {"verb", "adjective", "noun", "proper_noun", "lexical", "number"}
RESULTATIVE_SUPPORT_PREFIXES = ("हो", "हुआ", "हुई", "हुए", "बन", "बना", "बनी", "बने")
MODAL_SUPPORT_PREFIXES = ("सक", "प", "पा", "जा")
STATIC_COPULA_FORMS = {"है", "हैं", "था", "थी", "थे", "थीं"}
IMPERATIVE_SUFFIXES = ("ो", "ओ", "ें", "एँ", "इए", "ईए", "ये", "िए")


def _unique_ordered(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _row_center_category(row: dict) -> str:
    return str(row.get("predicate_center_category", "") or row.get("anchor_category", "") or "")


def _row_center_index(row: dict) -> int:
    try:
        return int(row.get("predicate_center_index", row.get("anchor_index", -1)) or -1)
    except (TypeError, ValueError):
        return -1


def _token_text(token: dict) -> str:
    return str(token.get("text", "") or token.get("norm", "") or token.get("lemma", "") or "").strip()


def _token_norm(token: dict) -> str:
    return str(token.get("norm", "") or token.get("lemma", "") or token.get("text", "") or "").strip()


def _token_feat_values(token: dict, key: str) -> set:
    feats = token.get("morph") or token.get("feats") or token.get("features") or {}
    if isinstance(feats, dict):
        value = feats.get(key) or feats.get(key.lower()) or feats.get(key.upper())
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if str(item)}
        return {item for item in str(value).split(",") if item}
    if isinstance(feats, str):
        values = set()
        for part in feats.split("|"):
            if "=" not in part:
                continue
            feat_key, feat_value = part.split("=", 1)
            if feat_key == key:
                values.update(item for item in feat_value.split(",") if item)
        return values
    return set()


def _norm_matches_prefixes(token: dict, prefixes: Sequence[str]) -> bool:
    norm = _token_norm(token)
    return any(norm.startswith(prefix) for prefix in prefixes if prefix)


def _looks_imperative(token: dict) -> bool:
    upos = str(token.get("upos", "") or token.get("pos", "") or "").upper()
    category = str(token.get("category", "") or "")
    if upos not in {"VERB", "AUX"} and category not in {"verb", "light_verb", "auxiliary"}:
        return False
    if "Imp" in _token_feat_values(token, "Mood"):
        return True
    text = _token_text(token)
    return any(text.endswith(suffix) for suffix in IMPERATIVE_SUFFIXES if suffix)


def _row_priority(row: dict) -> float:
    category = _row_center_category(row)
    priority = {
        "verb": 1.0,
        "light_verb": 0.9 if row.get("multiword_predicate") else 0.55,
        "adjective": 0.82,
        "noun": 0.78,
        "proper_noun": 0.74,
        "lexical": 0.72,
        "number": 0.62,
        "copula": 0.2,
        "auxiliary": 0.18,
    }.get(category, 0.3)
    score = float(row.get("trigger_plausibility_score", 0.0) or 0.0)
    if "trigger_pattern_mismatch" in {str(flag) for flag in (row.get("mismatch_flags", []) or []) if str(flag)}:
        priority -= 0.12
    return round(priority + score, 4)


def _is_helper_like(category: str) -> bool:
    return str(category or "") in HELPER_LIKE_CATEGORIES


def _mergeable_with_group(group: dict, row: dict, *, has_copula: bool, config: dict) -> bool:
    indices = group.get("indices", [])
    if not indices:
        return False
    anchor_index = _row_center_index(row)
    if anchor_index < 0:
        return False
    last_index = int(indices[-1])
    same_center_window = int(config.get("same_center_merge_window", 1) or 1)
    copular_window = int(config.get("same_center_copular_window", max(2, same_center_window + 1)) or max(2, same_center_window + 1))
    distance = abs(anchor_index - last_index)
    row_category = _row_center_category(row)
    group_categories = {str(value) for value in (group.get("categories", []) or [])}
    if distance == 0:
        return True
    if distance <= same_center_window and (_is_helper_like(row_category) or any(_is_helper_like(value) for value in group_categories)):
        return True
    if has_copula and distance <= copular_window:
        allowed = HELPER_LIKE_CATEGORIES.union({"adjective", "noun", "proper_noun", "lexical", "light_verb"})
        combined = set(group_categories)
        combined.add(row_category)
        if combined.issubset(allowed):
            return True
    return False


def _group_summary(group_id: int, rows: Sequence[dict]) -> dict:
    ordered = sorted(rows, key=_row_priority, reverse=True)
    primary = ordered[0] if ordered else {}
    indices = sorted({_row_center_index(row) for row in rows if _row_center_index(row) >= 0})
    categories = sorted({_row_center_category(row) for row in rows if _row_center_category(row)})
    predicate_like = any(category not in HELPER_LIKE_CATEGORIES for category in categories)
    return {
        "group_id": f"center_group_{group_id}",
        "indices": indices,
        "categories": categories,
        "predicate_like": predicate_like,
        "primary_center": _row_center_index(primary),
        "primary_anchor_category": _row_center_category(primary),
        "best_score": float(primary.get("trigger_plausibility_score", 0.0) or 0.0),
    }


def classify_clause_profile(sentence_tokens: Sequence[dict], assessments: Sequence[dict], config: dict) -> dict:
    soft_threshold = float(config.get("soft_candidate_score", 0.3) or 0.3)
    coordination = {str(token.get("norm", "")) for token in sentence_tokens if str(token.get("norm", "")) in set(config.get("coordination_markers", []) or [])}
    subordination = {str(token.get("norm", "")) for token in sentence_tokens if str(token.get("norm", "")) in set(config.get("subordination_markers", []) or [])}
    copula_indices = [int(token.get("index", -1)) for token in sentence_tokens if str(token.get("category", "")) == "copula"]

    viable = [
        row
        for row in assessments
        if not row.get("hard_reject")
        and float(row.get("trigger_plausibility_score", 0.0) or 0.0) >= soft_threshold
        and _row_center_index(row) >= 0
    ]
    grouped_rows = sorted(viable, key=lambda row: (_row_center_index(row), -_row_priority(row)))
    groups: List[dict] = []
    for row in grouped_rows:
        if groups and _mergeable_with_group(groups[-1], row, has_copula=bool(copula_indices), config=config):
            groups[-1]["rows"].append(row)
            groups[-1]["indices"].append(_row_center_index(row))
            groups[-1]["categories"].add(_row_center_category(row))
        else:
            groups.append(
                {
                    "rows": [row],
                    "indices": [_row_center_index(row)],
                    "categories": {_row_center_category(row)},
                }
            )
    center_groups = [_group_summary(index, group.get("rows", [])) for index, group in enumerate(groups)]
    centers = _unique_ordered([group["primary_center"] for group in center_groups if int(group.get("primary_center", -1)) >= 0])
    center_group_map: Dict[int, str] = {}
    for group in center_groups:
        for index in group.get("indices", []):
            if int(index) >= 0:
                center_group_map[int(index)] = str(group.get("group_id", ""))
    primary_group = max(
        center_groups,
        key=lambda group: (group.get("best_score", 0.0), group.get("predicate_like", False)),
        default={"primary_center": -1, "primary_anchor_category": ""},
    )
    primary_center = int(primary_group.get("primary_center", -1) or -1)
    primary_anchor_category = str(primary_group.get("primary_anchor_category", "") or "")
    predicate_group_count = len([group for group in center_groups if group.get("predicate_like")])
    has_adjectival_center = any("adjective" in set(group.get("categories", [])) for group in center_groups)
    has_nominal_center = any(set(group.get("categories", [])).intersection({"noun", "proper_noun", "lexical", "number"}) for group in center_groups)
    token_norms = {_token_norm(token) for token in sentence_tokens if _token_norm(token)}
    has_imperative_signal = any(_looks_imperative(token) for token in sentence_tokens)
    has_resultative_support = any(
        _norm_matches_prefixes(token, RESULTATIVE_SUPPORT_PREFIXES)
        or ("Perf" in _token_feat_values(token, "Aspect") and str(token.get("category", "") or "") in {"verb", "copula", "auxiliary"})
        for token in sentence_tokens
    )
    has_modal_support = any(
        _norm_matches_prefixes(token, MODAL_SUPPORT_PREFIXES)
        and str(token.get("category", "") or "") in {"verb", "auxiliary", "light_verb", "copula"}
        for token in sentence_tokens
    )
    has_light_verb_compound = any(
        bool(row.get("multiword_predicate"))
        and _row_center_category(row) in LEXICAL_CENTER_CATEGORIES
        and str(row.get("predicate_center_category", "") or "") != "light_verb"
        for row in viable
    )

    clause_type = "multi_predicate_clause"
    max_per_clause = int(config.get("max_final_candidates_per_clause", 2) or 2)
    single_predicate_like = predicate_group_count <= 1 or len(center_groups) <= 1
    trigger_meta_hints: Dict[str, object] = {}

    if single_predicate_like:
        clause_type = "single_predicate_clause"
        if has_imperative_signal and primary_anchor_category in LEXICAL_CENTER_CATEGORIES:
            clause_type = "imperative_predicate"
            max_per_clause = 1
            trigger_meta_hints["directive"] = True
        elif has_resultative_support and (primary_anchor_category == "adjective" or has_adjectival_center):
            clause_type = "change_of_state_resultative"
            max_per_clause = 1
            trigger_meta_hints["change_of_state"] = True
        elif has_resultative_support and (primary_anchor_category in {"noun", "proper_noun", "lexical", "number"} or has_nominal_center):
            clause_type = "change_of_state_resultative"
            max_per_clause = 1
            trigger_meta_hints["change_of_state"] = True
        elif has_modal_support and primary_anchor_category in LEXICAL_CENTER_CATEGORIES:
            clause_type = "modal_lexical_event"
            max_per_clause = 1
            if any(_norm_matches_prefixes(token, ("सक", "पा")) for token in sentence_tokens):
                trigger_meta_hints["modality"] = "ability"
            elif any(_norm_matches_prefixes(token, ("जा",)) for token in sentence_tokens):
                trigger_meta_hints["modality"] = "passive_modal"
            else:
                trigger_meta_hints["modality"] = "possibility"
        elif has_light_verb_compound:
            clause_type = "light_verb_compound_event"
            max_per_clause = 1
            trigger_meta_hints["light_verb_compound"] = True
        elif copula_indices:
            if primary_anchor_category == "adjective" or (not primary_anchor_category and has_adjectival_center):
                clause_type = "predicative_adjectival_state"
                max_per_clause = int(config.get("max_events_per_single_adjectival_clause", 1) or 1)
            elif primary_anchor_category in {"noun", "proper_noun", "lexical", "number"} or has_nominal_center:
                if token_norms.intersection(STATIC_COPULA_FORMS):
                    clause_type = "static_copular_identity"
                else:
                    clause_type = "copular_nominal_classification"
                max_per_clause = int(config.get("max_events_per_single_copular_clause", 1) or 1)
            else:
                max_per_clause = int(config.get("max_events_per_single_predicate_clause", 2) or 2)
        else:
            max_per_clause = int(config.get("max_events_per_single_predicate_clause", 2) or 2)
    elif coordination or subordination:
        clause_type = "multi_predicate_clause"
        max_per_clause = int(config.get("max_final_candidates_per_clause", 2) or 2)

    compression_enabled_types = {str(value) for value in (config.get("compression_enabled_for_clause_types", []) or []) if str(value)}
    allow_multiple_if_clause_centers_gt = int(config.get("allow_multiple_if_clause_centers_gt", 1) or 1)
    compression_enabled = (
        clause_type in compression_enabled_types
        or clause_type in RECOVER_NOW_CLAUSE_TYPES
        or clause_type in COPULAR_STATE_TYPES
    ) and len(center_groups) <= max(1, allow_multiple_if_clause_centers_gt)
    clause_total_limit = max_per_clause if compression_enabled else int(config.get("max_final_candidates_per_clause", 2) or 2)

    return {
        "clause_type": clause_type,
        "recover_now_clause": clause_type in RECOVER_NOW_CLAUSE_TYPES,
        "predicate_centers": centers,
        "primary_center": primary_center,
        "primary_anchor_category": primary_anchor_category,
        "has_copula": bool(copula_indices),
        "coordination_markers_found": sorted(coordination),
        "subordination_markers_found": sorted(subordination),
        "single_predicate_like": single_predicate_like,
        "max_candidates_per_clause": max_per_clause,
        "clause_total_limit": clause_total_limit,
        "compression_enabled": compression_enabled,
        "center_groups": center_groups,
        "center_group_map": center_group_map,
        "trigger_meta_hints": trigger_meta_hints,
    }
