#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Duplicate collapse and final acceptance for pre-extraction frame candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

try:
    from .predicate_center_ranking import (
        apply_predicate_center_adjustments,
        competition_group_id,
        lexical_center_label,
        mismatch_severity,
        predicate_center_label,
        winner_sort_key,
    )
except ImportError:  # pragma: no cover
    from gold.predicate_center_ranking import (
        apply_predicate_center_adjustments,
        competition_group_id,
        lexical_center_label,
        mismatch_severity,
        predicate_center_label,
        winner_sort_key,
    )


def _row_center_index(row: dict) -> int:
    return int(row.get("predicate_center_index", row.get("anchor_index", -1)) or -1)


def _row_center_span(row: dict) -> List[int]:
    span = row.get("predicate_center_span")
    if isinstance(span, list) and len(span) == 2:
        return span
    return row.get("anchor_span", []) if isinstance(row.get("anchor_span", []), list) else []


def _row_center_norm(row: dict) -> str:
    return str(row.get("predicate_center_norm", "") or row.get("anchor_norm", "") or "")


def _span_overlap_ratio(left: Sequence[int], right: Sequence[int]) -> float:
    if not (isinstance(left, list) and isinstance(right, list) and len(left) == 2 and len(right) == 2):
        return 0.0
    left_start, left_end = int(left[0]), int(left[1])
    right_start, right_end = int(right[0]), int(right[1])
    if left_end <= left_start or right_end <= right_start:
        return 0.0
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    if not overlap:
        return 0.0
    span = max(left_end - left_start, right_end - right_start)
    return overlap / float(span or 1)


def _similarity(left: dict, right: dict, config: dict, clause_profile: dict) -> float:
    score = 0.0
    left_center = _row_center_index(left)
    right_center = _row_center_index(right)
    if left_center == right_center:
        score += 0.35
    if abs(left_center - right_center) <= 1:
        score += 0.15
    overlap = _span_overlap_ratio(_row_center_span(left), _row_center_span(right))
    threshold = float(config.get("trigger_span_overlap_threshold", 0.5) or 0.5)
    if overlap >= threshold:
        score += 0.25
    if str(left.get("frame_family", "")) == str(right.get("frame_family", "")):
        score += 0.15
    if _row_center_norm(left) and _row_center_norm(left) == _row_center_norm(right):
        score += 0.1
    if clause_profile.get("single_predicate_like") and left_center == int(clause_profile.get("primary_center", -1)):
        if right_center == int(clause_profile.get("primary_center", -1)):
            score += 0.1
    return round(score, 4)


def _cluster_candidates(assessments: Sequence[dict], clause_profile: dict, config: dict) -> List[List[dict]]:
    if not assessments:
        return []
    min_similarity = float(config.get("duplicate_cluster_min_similarity", 0.6) or 0.6)
    clusters: List[List[dict]] = []
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for candidate in assessments:
        by_group[str(candidate.get("competition_group_id", "ungrouped"))].append(candidate)
    for group_rows in by_group.values():
        for candidate in sorted(group_rows, key=winner_sort_key, reverse=True):
            placed = False
            for cluster in clusters:
                if str(cluster[0].get("competition_group_id", "")) != str(candidate.get("competition_group_id", "")):
                    continue
                if _similarity(candidate, cluster[0], config, clause_profile) >= min_similarity:
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])
    return clusters


def select_final_candidates(
    assessments: Sequence[dict],
    clause_profile: dict,
    *,
    max_events: int,
    min_final_score: float,
    enable_candidate_dedup: bool,
    max_final_candidates_per_clause: int,
    config: dict,
) -> Tuple[List[dict], List[dict], dict]:
    decisions = [dict(row) for row in assessments]
    summary_counter = Counter()
    reason_counter = Counter()
    accepted_family_counter = Counter()
    surviving: List[dict] = []
    anchor_best_scores = {}
    grouped_candidates: Dict[str, List[dict]] = defaultdict(list)
    for row in decisions:
        row["competition_group_id"] = competition_group_id(row, clause_profile)
        row["mismatch_severity"] = mismatch_severity(row)
        grouped_candidates[str(row["competition_group_id"])].append(row)
        anchor = _row_center_index(row)
        if anchor < 0 or row.get("hard_reject"):
            continue
        anchor_best_scores[anchor] = max(
            float(row.get("trigger_plausibility_score", 0.0) or 0.0),
            float(anchor_best_scores.get(anchor, float("-inf"))),
        )
    summary_counter["candidates_grouped_by_clause_center"] = sum(len(rows) for rows in grouped_candidates.values() if len(rows) > 1)
    summary_counter["same_center_competition_groups"] = sum(1 for rows in grouped_candidates.values() if len(rows) > 1)
    group_profiles: Dict[str, dict] = {}
    for group_id, rows in grouped_candidates.items():
        clean_lexical_rows = [
            row
            for row in rows
            if lexical_center_label(predicate_center_label(row, clause_profile))
            and float(row.get("mismatch_severity", 0.0) or 0.0) <= 1.0
            and not row.get("hard_reject")
        ]
        clean_lexical_rows.sort(key=winner_sort_key, reverse=True)
        group_profiles[group_id] = {
            "size": len(rows),
            "best_clean_row": clean_lexical_rows[0] if clean_lexical_rows else None,
            "has_clean_lexical_competitor": bool(clean_lexical_rows),
        }

    for row in decisions:
        final_score = float(row.get("trigger_plausibility_score", 0.0) or 0.0)
        row["pre_bonus_final_score"] = round(max(0.0, final_score), 4)
        row["adjectival_bonus_applied"] = False
        row["adjectival_bonus_value"] = 0.0
        row["threshold_crossed_by_bonus"] = False
        row["would_pass_without_bonus"] = float(row["pre_bonus_final_score"]) >= float(min_final_score)
        clause_type = str(clause_profile.get("clause_type", "") or "")
        adjectival_bonus = float(config.get("adjectival_predicate_bonus", 0.0) or 0.0)
        bonus_clause_types = {str(value) for value in (config.get("adjectival_bonus_clause_types", []) or []) if str(value)}
        requires_center = bool(config.get("adjectival_bonus_requires_predicate_center", True))
        disallow_mismatch = bool(config.get("adjectival_bonus_disallow_trigger_mismatch", True))
        disallow_sources = {str(value) for value in (config.get("adjectival_bonus_disallow_sources", []) or []) if str(value)}
        anchor_index = _row_center_index(row)
        no_stronger_duplicate = float(row.get("trigger_plausibility_score", 0.0) or 0.0) >= float(anchor_best_scores.get(anchor_index, -1.0))
        mismatch_flags = [str(flag) for flag in (row.get("mismatch_flags", []) or []) if str(flag)]
        has_mismatch = bool(mismatch_flags) or str(row.get("evidence_source", "")) in disallow_sources
        eligible_for_adjectival_bonus = (
            not row.get("hard_reject")
            and adjectival_bonus > 0.0
            and clause_type in bonus_clause_types
            and str(row.get("anchor_category", "")) == "adjective"
            and no_stronger_duplicate
            and (not requires_center or anchor_index == int(clause_profile.get("primary_center", -999)))
            and (not disallow_mismatch or not has_mismatch)
        )
        if eligible_for_adjectival_bonus:
            final_score += adjectival_bonus
            row.setdefault("score_breakdown", {})
            row["score_breakdown"]["adjectival_predicate_bonus"] = round(adjectival_bonus, 4)
            row["adjectival_bonus_applied"] = True
            row["adjectival_bonus_value"] = round(adjectival_bonus, 4)
            row["threshold_crossed_by_bonus"] = (
                float(row["pre_bonus_final_score"]) < float(min_final_score) <= float(final_score)
            )
        if clause_profile.get("single_predicate_like") and _row_center_index(row) >= 0:
            if _row_center_index(row) != int(clause_profile.get("primary_center", -1)):
                penalty = float((config.get("mismatch_penalties", {}) or {}).get("outside_clause_center", 0.12) or 0.12)
                final_score -= penalty
                row.setdefault("score_breakdown", {})
                row["score_breakdown"]["outside_clause_center"] = round(-penalty, 4)
        ranking = apply_predicate_center_adjustments(
            row,
            clause_profile,
            config,
            group_profiles.get(str(row.get("competition_group_id", "")), {"size": 1, "best_clean_row": None, "has_clean_lexical_competitor": False}),
        )
        final_score += float(ranking.get("score_delta", 0.0) or 0.0)
        row["predicate_center_label"] = ranking.get("predicate_center_label", "")
        row["predicate_center_score"] = ranking.get("predicate_center_score", 0.0)
        row["mismatch_severity"] = ranking.get("mismatch_severity", row.get("mismatch_severity", 0.0))
        row["downranked_for_mismatch"] = bool(ranking.get("downranked_for_mismatch", False))
        row["downranked_as_helper_like"] = bool(ranking.get("downranked_as_helper_like", False))
        row["helper_penalty_applied"] = ranking.get("helper_penalty", 0.0)
        row["mismatch_penalty_applied"] = ranking.get("mismatch_penalty", 0.0)
        row["same_center_competition_penalty_applied"] = ranking.get("competition_penalty", 0.0)
        row["same_center_lexical_winner_bonus_applied"] = ranking.get("lexical_winner_bonus", 0.0)
        row["has_cleaner_same_center_competitor"] = bool(ranking.get("has_cleaner_same_center_competitor", False))
        row["best_clean_competitor_frame"] = ranking.get("best_clean_competitor_frame", "")
        row["clean_competitor_is_lexical"] = bool(ranking.get("clean_competitor_is_lexical", False))
        row["same_frame_family_competitor"] = bool(ranking.get("same_frame_family_competitor", False))
        row["competition_group_size"] = int(group_profiles.get(str(row.get("competition_group_id", "")), {}).get("size", 1) or 1)
        if row["downranked_for_mismatch"]:
            summary_counter["candidates_downranked_for_mismatch"] += 1
        if row["downranked_as_helper_like"]:
            summary_counter["candidates_downranked_as_helper_like"] += 1
        row["final_candidate_score"] = round(max(0.0, final_score), 4)
        if row.get("hard_reject"):
            row["decision"] = "rejected"
            row["decision_stage"] = "trigger_filter"
            row["rejection_reason"] = (row.get("reject_reasons") or ["closed_class_trigger"])[0]
            summary_counter["candidates_rejected_by_trigger_filter"] += 1
            reason_counter[str(row["rejection_reason"])] += 1
            continue
        if float(row.get("final_candidate_score", 0.0) or 0.0) < float(min_final_score):
            row["decision"] = "rejected"
            row["decision_stage"] = "trigger_filter"
            row["rejection_reason"] = "low_final_score"
            summary_counter["candidates_rejected_by_trigger_filter"] += 1
            reason_counter["low_final_score"] += 1
            continue
        row["decision"] = "survived_filter"
        row["decision_stage"] = "acceptance"
        surviving.append(row)

    clause_type = str(clause_profile.get("clause_type", "") or "")
    recover_now_clause = clause_type in RECOVER_NOW_CLAUSE_TYPES and bool(clause_profile.get("single_predicate_like"))

    same_family_survivor_groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in surviving:
        same_family_survivor_groups[
            (
                str(row.get("competition_group_id", "")),
                _row_center_index(row),
                _row_center_norm(row),
                str(row.get("frame_family", "") or ""),
            )
        ].append(row)
    precluster_survivors: List[dict] = []
    for rows in same_family_survivor_groups.values():
        if len(rows) <= 1:
            precluster_survivors.extend(rows)
            continue
        rows.sort(key=winner_sort_key, reverse=True)
        winner = rows[0]
        precluster_survivors.append(winner)
        for dropped in rows[1:]:
            dropped["decision"] = "rejected"
            dropped["decision_stage"] = "same_center_family_collapse"
            dropped["competition_winner_frame_id"] = winner.get("frame_id", "")
            dropped["competition_outcome_reason"] = "same_center_duplicate"
            dropped["lost_to_lexical_center"] = bool(
                winner.get("predicate_center_label")
                and lexical_center_label(str(winner.get("predicate_center_label", "")))
            )
            dropped["lost_due_to_mismatch"] = bool(dropped.get("downranked_for_mismatch"))
            dropped["lost_due_to_helper_like"] = bool(dropped.get("downranked_as_helper_like"))
            dropped["rejection_reason"] = "same_center_duplicate"
            summary_counter["candidates_collapsed_as_duplicates"] += 1
            summary_counter["candidates_dropped_as_same_center_duplicates"] += 1
            reason_counter["same_center_duplicate"] += 1
    surviving = precluster_survivors

    if recover_now_clause:
        same_center_survivor_groups: Dict[tuple, List[dict]] = defaultdict(list)
        for row in surviving:
            center_norm = _row_center_norm(row)
            key = (
                str(row.get("competition_group_id", "")),
                _row_center_index(row),
                center_norm or f"frame::{str(row.get('frame_family', '') or row.get('frame_id', ''))}",
            )
            same_center_survivor_groups[key].append(row)
        recover_now_survivors: List[dict] = []
        for rows in same_center_survivor_groups.values():
            if len(rows) <= 1:
                recover_now_survivors.extend(rows)
                continue
            rows.sort(key=winner_sort_key, reverse=True)
            winner = rows[0]
            recover_now_survivors.append(winner)
            for dropped in rows[1:]:
                if _similarity(dropped, winner, config, clause_profile) < 0.6:
                    recover_now_survivors.append(dropped)
                    continue
                dropped["decision"] = "rejected"
                dropped["decision_stage"] = "recover_now_same_center_collapse"
                dropped["competition_winner_frame_id"] = winner.get("frame_id", "")
                dropped["competition_outcome_reason"] = "same_center_duplicate"
                dropped["lost_to_lexical_center"] = bool(
                    winner.get("predicate_center_label")
                    and lexical_center_label(str(winner.get("predicate_center_label", "")))
                )
                dropped["lost_due_to_mismatch"] = bool(dropped.get("downranked_for_mismatch"))
                dropped["lost_due_to_helper_like"] = bool(dropped.get("downranked_as_helper_like"))
                dropped["rejection_reason"] = "same_center_duplicate"
                summary_counter["candidates_collapsed_as_duplicates"] += 1
                summary_counter["candidates_dropped_as_same_center_duplicates"] += 1
                reason_counter["same_center_duplicate"] += 1
        surviving = recover_now_survivors

    clusters = _cluster_candidates(surviving, clause_profile, config) if enable_candidate_dedup else [[row] for row in surviving]
    winners: List[dict] = []
    for cluster in clusters:
        cluster.sort(key=winner_sort_key, reverse=True)
        winner = cluster[0]
        winner["decision"] = "accepted_candidate"
        winner["decision_stage"] = "dedup"
        winner["won_same_center_competition"] = len(cluster) > 1
        winner["competition_loser_count"] = max(0, len(cluster) - 1)
        winners.append(winner)
        for dropped in cluster[1:]:
            dropped["decision"] = "rejected"
            dropped["decision_stage"] = "dedup"
            dropped["competition_winner_frame_id"] = winner.get("frame_id", "")
            dropped["competition_outcome_reason"] = "same_center_duplicate"
            dropped["lost_to_lexical_center"] = bool(
                winner.get("predicate_center_label") and lexical_center_label(str(winner.get("predicate_center_label", "")))
            )
            dropped["lost_due_to_mismatch"] = bool(dropped.get("downranked_for_mismatch"))
            dropped["lost_due_to_helper_like"] = bool(dropped.get("downranked_as_helper_like"))
            dropped["rejection_reason"] = "same_center_duplicate"
            summary_counter["candidates_collapsed_as_duplicates"] += 1
            summary_counter["candidates_dropped_as_same_center_duplicates"] += 1
            reason_counter["same_center_duplicate"] += 1

    per_center_limit = 1 if recover_now_clause else max(1, int(max_final_candidates_per_clause or 0) or int(clause_profile.get("max_candidates_per_clause", 1) or 1))
    clause_total_limit = 1 if recover_now_clause else int(clause_profile.get("clause_total_limit", max_events) or max_events)
    compression_enabled = bool(clause_profile.get("compression_enabled", False))
    emitted: List[dict] = []
    center_counts = defaultdict(int)
    for row in sorted(winners, key=winner_sort_key, reverse=True):
        center = _row_center_index(row)
        if compression_enabled and len(emitted) >= max(1, clause_total_limit):
            row["decision"] = "rejected"
            row["decision_stage"] = "clause_limit"
            row["rejection_reason"] = "clause_compression"
            summary_counter["clause_compression_applied_count"] += 1
            if clause_type in COPULAR_COMPRESSION_TYPES:
                summary_counter["candidates_rejected_as_copular_overgeneration"] += 1
                reason_counter["copular_overgeneration"] += 1
            elif recover_now_clause:
                row["rejection_reason"] = "recover_now_clause_limit"
                reason_counter["recover_now_clause_limit"] += 1
            else:
                reason_counter["clause_compression"] += 1
            continue
        if center_counts[center] >= per_center_limit:
            row["decision"] = "rejected"
            row["decision_stage"] = "clause_limit"
            if clause_type in COPULAR_COMPRESSION_TYPES:
                row["rejection_reason"] = "copular_overgeneration"
                summary_counter["candidates_rejected_as_copular_overgeneration"] += 1
                reason_counter["copular_overgeneration"] += 1
            elif recover_now_clause:
                row["rejection_reason"] = "recover_now_clause_limit"
                reason_counter["recover_now_clause_limit"] += 1
            else:
                row["rejection_reason"] = "clause_candidate_limit"
                reason_counter["clause_candidate_limit"] += 1
            continue
        if len(emitted) >= int(max_events):
            row["decision"] = "rejected"
            row["decision_stage"] = "max_events"
            row["rejection_reason"] = "max_events_limit"
            reason_counter["max_events_limit"] += 1
            continue
        row["decision"] = "emitted"
        row["decision_stage"] = "final"
        row["kept_as_lexical_predicate_center"] = lexical_center_label(str(row.get("predicate_center_label", "")))
        center_counts[center] += 1
        emitted.append(row)
        accepted_family_counter[str(row.get("frame_family", ""))] += 1
        if row.get("kept_as_lexical_predicate_center"):
            summary_counter["candidates_kept_as_lexical_predicate_centers"] += 1

    summary = {
        "candidates_retrieved": len(decisions),
        "candidates_rejected_by_trigger_filter": int(summary_counter.get("candidates_rejected_by_trigger_filter", 0) or 0),
        "candidates_rejected_as_copular_overgeneration": int(summary_counter.get("candidates_rejected_as_copular_overgeneration", 0) or 0),
        "candidates_collapsed_as_duplicates": int(summary_counter.get("candidates_collapsed_as_duplicates", 0) or 0),
        "candidates_grouped_by_clause_center": int(summary_counter.get("candidates_grouped_by_clause_center", 0) or 0),
        "candidates_dropped_as_same_center_duplicates": int(summary_counter.get("candidates_dropped_as_same_center_duplicates", 0) or 0),
        "candidates_downranked_for_mismatch": int(summary_counter.get("candidates_downranked_for_mismatch", 0) or 0),
        "candidates_downranked_as_helper_like": int(summary_counter.get("candidates_downranked_as_helper_like", 0) or 0),
        "candidates_kept_as_lexical_predicate_centers": int(summary_counter.get("candidates_kept_as_lexical_predicate_centers", 0) or 0),
        "clause_compression_applied_count": int(summary_counter.get("clause_compression_applied_count", 0) or 0),
        "same_center_competition_groups": int(summary_counter.get("same_center_competition_groups", 0) or 0),
        "final_candidates_considered": len(winners),
        "final_candidates_emitted": len(emitted),
        "rejection_reasons_distribution": dict(sorted(reason_counter.items())),
        "accepted_candidate_family_distribution": dict(sorted(accepted_family_counter.items())),
    }
    return emitted, decisions, summary
RECOVER_NOW_CLAUSE_TYPES = {
    "imperative_predicate",
    "change_of_state_resultative",
    "modal_lexical_event",
    "light_verb_compound_event",
}
COPULAR_COMPRESSION_TYPES = {
    "single_copular_clause",
    "single_adjectival_clause",
    "predicative_adjectival_state",
    "copular_nominal_classification",
    "static_copular_identity",
}
