#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predicate-center-aware ranking helpers for precision-first candidate selection."""

from __future__ import annotations


HELPER_LIKE_CATEGORIES = {
    "auxiliary",
    "copula",
    "particle",
    "adp",
    "conjunction",
    "pronoun",
    "determiner",
    "closed_class",
}

LEXICAL_CENTER_LABELS = {
    "lexical_verb",
    "lexical_predicate",
    "light_verb_compound",
    "predicative_adjective",
    "predicative_nominal",
}
RECOVER_NOW_TYPES = {
    "imperative_predicate",
    "change_of_state_resultative",
    "modal_lexical_event",
    "light_verb_compound_event",
}


def _config_float(config: dict, key: str, default: float) -> float:
    try:
        return float(config.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _row_center_category(row: dict) -> str:
    return str(row.get("predicate_center_category", "") or row.get("anchor_category", "") or "")


def _row_center_index(row: dict) -> int:
    return int(row.get("predicate_center_index", row.get("anchor_index", -1)) or -1)


def _recover_now_center_rank(clause_type: str, row: dict) -> float:
    category = _row_center_category(row)
    multiword = bool(row.get("multiword_predicate"))
    if clause_type == "imperative_predicate":
        if category == "verb":
            return 0.0
        if multiword and category in {"adjective", "noun", "proper_noun", "lexical", "number"}:
            return 0.4
        if category in {"adjective", "noun", "proper_noun", "lexical", "number"}:
            return 1.4
        if category == "light_verb":
            return 2.0
    elif clause_type == "change_of_state_resultative":
        if category == "adjective":
            return 0.0
        if category in {"noun", "proper_noun", "lexical", "number"}:
            return 0.3
        if category == "verb":
            return 1.0
        if category == "light_verb":
            return 2.0
    elif clause_type == "modal_lexical_event":
        if category == "verb":
            return 0.0
        if multiword and category in {"adjective", "noun", "proper_noun", "lexical", "number"}:
            return 0.5
        if category in {"adjective", "noun", "proper_noun", "lexical", "number"}:
            return 1.2
        if category == "light_verb":
            return 2.0
    elif clause_type == "light_verb_compound_event":
        if category in {"adjective", "noun", "proper_noun", "lexical", "number"}:
            return 0.0
        if category == "verb":
            return 0.6
        if category == "light_verb":
            return 2.0
    return 0.0 if lexical_center_label(predicate_center_label(row, {"clause_type": clause_type})) else 1.0


def helper_like_anchor(row: dict) -> bool:
    category = _row_center_category(row)
    if category in HELPER_LIKE_CATEGORIES:
        return True
    return category == "light_verb" and not bool(row.get("multiword_predicate"))


def competition_group_id(row: dict, clause_profile: dict) -> str:
    anchor_index = _row_center_index(row)
    group_map = clause_profile.get("center_group_map", {}) if isinstance(clause_profile, dict) else {}
    if isinstance(group_map, dict):
        if anchor_index in group_map:
            return str(group_map[anchor_index])
        key = str(anchor_index)
        if key in group_map:
            return str(group_map[key])
    return f"anchor:{anchor_index}"


def mismatch_severity(row: dict) -> float:
    flags = {str(flag) for flag in (row.get("mismatch_flags", []) or []) if str(flag)}
    source_name = str(row.get("evidence_source", "") or "")
    severity = 0.0
    if "trigger_pattern_mismatch" in flags:
        severity += 1.35
    if "helper_only_anchor" in flags:
        severity += 1.5
    if "auxiliary_anchor" in flags:
        severity += 0.7
    if "copular_anchor" in flags:
        severity += 0.7
    if source_name in {"single_prefix", "multiword_prefix"}:
        severity = max(severity, 1.1)
    if int(row.get("lexical_token_count", 0) or 0) <= 0 and not bool(row.get("multiword_predicate")):
        severity = max(severity, 0.9)
    return round(severity, 4)


def predicate_center_label(row: dict, clause_profile: dict) -> str:
    category = _row_center_category(row)
    clause_type = str((clause_profile or {}).get("clause_type", "") or "")
    if helper_like_anchor(row):
        return "helper_like"
    if category == "verb":
        return "lexical_verb"
    if category == "light_verb":
        return "light_verb_compound" if bool(row.get("multiword_predicate")) else "light_verb"
    if category == "adjective" and clause_type in {"single_adjectival_clause", "single_copular_clause", "predicative_adjectival_state", "change_of_state_resultative"}:
        return "predicative_adjective"
    if category in {"noun", "proper_noun", "lexical", "number"} and clause_type in {"single_copular_clause", "copular_nominal_classification", "change_of_state_resultative"}:
        return "predicative_nominal"
    if category in {"noun", "proper_noun", "lexical", "number"}:
        return "lexical_predicate"
    if category == "adjective":
        return "lexical_adjective"
    return "other"


def predicate_center_priority(label: str) -> float:
    return {
        "lexical_verb": 1.0,
        "light_verb_compound": 0.92,
        "predicative_adjective": 0.88,
        "predicative_nominal": 0.84,
        "lexical_predicate": 0.8,
        "lexical_adjective": 0.66,
        "light_verb": 0.42,
        "helper_like": 0.1,
        "other": 0.3,
    }.get(str(label or ""), 0.25)


def lexical_center_label(label: str) -> bool:
    return str(label or "") in LEXICAL_CENTER_LABELS


def winner_sort_key(row: dict) -> tuple:
    return (
        float(row.get("final_candidate_score", 0.0) or 0.0),
        float(row.get("predicate_center_score", 0.0) or 0.0),
        1 if row.get("selected_by_frame_selection") else 0,
        -float(row.get("mismatch_severity", 0.0) or 0.0),
        int(row.get("lexical_token_count", 0) or 0),
        float(row.get("trigger_plausibility_score", 0.0) or 0.0),
        int(row.get("hits", 0) or 0),
    )


def apply_predicate_center_adjustments(row: dict, clause_profile: dict, config: dict, group_profile: dict) -> dict:
    label = predicate_center_label(row, clause_profile)
    clause_type = str((clause_profile or {}).get("clause_type", "") or "")
    center_category = _row_center_category(row)
    center_score = predicate_center_priority(label)
    recover_now_center_rank = _recover_now_center_rank(clause_type, row)
    primary_center = int((clause_profile or {}).get("primary_center", -1) or -1)
    anchor_index = _row_center_index(row)
    frame_family = str(row.get("frame_family", "") or str(row.get("frame_id", "")).split(".", 1)[0])
    breakdown = row.setdefault("score_breakdown", {})
    delta = 0.0

    predicate_center_bonus = 0.0
    if lexical_center_label(label):
        predicate_center_bonus += _config_float(config, "predicate_center_bonus", 0.1)
        if label in {"lexical_verb", "lexical_predicate"}:
            predicate_center_bonus += _config_float(config, "lexical_predicate_bonus", 0.08)
        elif label == "light_verb_compound":
            predicate_center_bonus += _config_float(config, "light_verb_compound_bonus", 0.06)
        elif label == "predicative_adjective":
            predicate_center_bonus += _config_float(
                config,
                "predicative_adjective_bonus",
                _config_float(config, "adjectival_predicate_bonus", 0.06),
            )
        elif label == "predicative_nominal":
            predicate_center_bonus += _config_float(config, "predicative_nominal_bonus", 0.05)
    if predicate_center_bonus:
        delta += predicate_center_bonus
        breakdown["predicate_center_bonus"] = round(predicate_center_bonus, 4)

    clause_center_penalty = 0.0
    if clause_type in RECOVER_NOW_TYPES and recover_now_center_rank > 0.0:
        clause_center_penalty = _config_float(config, "recover_now_center_misalignment_penalty", 0.08) * recover_now_center_rank
        delta -= clause_center_penalty
        breakdown["recover_now_center_misalignment_penalty"] = round(-clause_center_penalty, 4)

    recover_now_bonus = 0.0
    recover_now_bonus_defaults = {
        "imperative_predicate": 0.04,
        "change_of_state_resultative": 0.05,
        "modal_lexical_event": 0.04,
        "light_verb_compound_event": 0.04,
    }
    if clause_type in RECOVER_NOW_TYPES and lexical_center_label(label) and center_category != "light_verb":
        recover_now_bonus = _config_float(config, f"{clause_type}_bonus", recover_now_bonus_defaults.get(clause_type, 0.0))
        if recover_now_bonus:
            delta += recover_now_bonus
            breakdown["recover_now_clause_bonus"] = round(recover_now_bonus, 4)

    clause_root_bonus = 0.0
    if anchor_index >= 0 and anchor_index == primary_center and label not in {"helper_like", "other"}:
        clause_root_bonus = _config_float(config, "clause_root_bonus", 0.08)
        delta += clause_root_bonus
        breakdown["clause_root_bonus"] = round(clause_root_bonus, 4)

    helper_penalty = 0.0
    if label == "helper_like":
        helper_penalty = _config_float(config, "helper_like_penalty", 0.14)
    elif label == "light_verb" and bool(group_profile.get("has_clean_lexical_competitor")):
        helper_penalty = _config_float(config, "helper_like_penalty", 0.14) * 0.5
    if helper_penalty:
        delta -= helper_penalty
        breakdown["helper_like_penalty"] = round(-helper_penalty, 4)

    severity = mismatch_severity(row)
    mismatch_penalty = 0.0
    if severity:
        mismatch_penalty = severity * _config_float(config, "mismatch_penalty_weight", 0.14)
        if severity >= _config_float(config, "major_mismatch_threshold", 1.25):
            mismatch_penalty += _config_float(config, "severe_mismatch_penalty", 0.08)
        delta -= mismatch_penalty
        breakdown["mismatch_penalty"] = round(-mismatch_penalty, 4)

    lexical_winner_bonus = 0.0
    competition_penalty = 0.0
    best_clean_row = group_profile.get("best_clean_row")
    best_clean_label = predicate_center_label(best_clean_row, clause_profile) if isinstance(best_clean_row, dict) else ""
    best_clean_is_lexical = lexical_center_label(best_clean_label)
    best_clean_recover_now_rank = _recover_now_center_rank(clause_type, best_clean_row) if isinstance(best_clean_row, dict) else 0.0
    best_clean_family = (
        str(best_clean_row.get("frame_family", "") or str(best_clean_row.get("frame_id", "")).split(".", 1)[0])
        if isinstance(best_clean_row, dict)
        else ""
    )
    competitor_severity = float(best_clean_row.get("mismatch_severity", 0.0) or 0.0) if isinstance(best_clean_row, dict) else 0.0
    same_frame_family_competitor = bool(best_clean_row is not None and best_clean_row is not row and best_clean_family and best_clean_family == frame_family)
    if best_clean_row is row and lexical_center_label(label) and int(group_profile.get("size", 1) or 1) > 1:
        lexical_winner_bonus = _config_float(config, "same_center_lexical_winner_bonus", 0.06)
        delta += lexical_winner_bonus
        breakdown["same_center_lexical_winner_bonus"] = round(lexical_winner_bonus, 4)
    if best_clean_row is not None and best_clean_row is not row:
        same_center_penalty = _config_float(config, "same_center_competition_penalty", 0.1)
        if best_clean_is_lexical and not lexical_center_label(label):
            competition_penalty = same_center_penalty
            competition_penalty += _config_float(config, "same_center_helper_competitor_penalty", 0.08)
            if clause_type in RECOVER_NOW_TYPES:
                competition_penalty += _config_float(config, "recover_now_same_center_penalty", 0.06)
        elif clause_type in RECOVER_NOW_TYPES and recover_now_center_rank > best_clean_recover_now_rank:
            competition_penalty = same_center_penalty * 0.5
            competition_penalty += _config_float(config, "recover_now_center_competition_penalty", 0.06) * max(
                1.0, recover_now_center_rank - best_clean_recover_now_rank
            )
        elif severity > competitor_severity:
            competition_penalty = same_center_penalty * 0.6
            competition_penalty += _config_float(config, "same_center_mismatch_competitor_penalty", 0.08)
            if severity >= _config_float(config, "major_mismatch_threshold", 1.25):
                competition_penalty += _config_float(config, "same_center_major_mismatch_penalty", 0.05)
            if clause_type in RECOVER_NOW_TYPES:
                competition_penalty += _config_float(config, "recover_now_mismatch_penalty", 0.04)
        if same_frame_family_competitor and best_clean_is_lexical:
            competition_penalty += _config_float(config, "same_center_same_family_penalty", 0.05)
        if competition_penalty:
            delta -= competition_penalty
            breakdown["same_center_competition_penalty"] = round(-competition_penalty, 4)

    return {
        "predicate_center_label": label,
        "predicate_center_score": round(center_score + predicate_center_bonus + recover_now_bonus + clause_root_bonus + lexical_winner_bonus, 4),
        "mismatch_severity": round(severity, 4),
        "helper_penalty": round(helper_penalty, 4),
        "mismatch_penalty": round(mismatch_penalty, 4),
        "competition_penalty": round(competition_penalty, 4),
        "lexical_winner_bonus": round(lexical_winner_bonus, 4),
        "recover_now_bonus": round(recover_now_bonus, 4),
        "recover_now_center_rank": round(recover_now_center_rank, 4),
        "recover_now_center_penalty": round(clause_center_penalty, 4),
        "score_delta": round(delta, 4),
        "downranked_for_mismatch": mismatch_penalty > 0.0 or (competition_penalty > 0.0 and severity > 0.0),
        "downranked_as_helper_like": helper_penalty > 0.0,
        "has_cleaner_same_center_competitor": best_clean_row is not None and best_clean_row is not row,
        "best_clean_competitor_frame": str(best_clean_row.get("frame_id", "")) if best_clean_row is not None and best_clean_row is not row else "",
        "clean_competitor_is_lexical": bool(best_clean_is_lexical and best_clean_row is not None and best_clean_row is not row),
        "same_frame_family_competitor": same_frame_family_competitor,
    }
