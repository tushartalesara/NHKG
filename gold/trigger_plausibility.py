#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Config-backed trigger plausibility scoring for pre-extraction candidate control."""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


RECOVER_NOW_CLAUSE_TYPES = {
    "imperative_predicate",
    "change_of_state_resultative",
    "modal_lexical_event",
    "light_verb_compound_event",
}
LEXICAL_PREDICATE_CATEGORIES = {"verb", "adjective", "noun", "proper_noun", "lexical", "number", "light_verb"}
SUPPORT_ONLY_CATEGORIES = {"auxiliary", "copula", "particle", "adp", "conjunction", "pronoun", "determiner", "closed_class"}

import yaml

try:
    from .span import Tokenizer
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from gold.span import Tokenizer


IGNORED_PUNCTUATION = {
    ",",
    ".",
    ";",
    ":",
    "?",
    "!",
    "।",
    "॥",
    "\"",
    "'",
    "`",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "|",
}


def normalize_text(text: object) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    value = value.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    return " ".join(value.split()).strip()


def normalize_token(token: object) -> str:
    text = normalize_text(token).strip(".,;:!?।॥\"'`()[]{}<>|")
    return text


def deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_trigger_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "lexicons" / "trigger_plausibility.yaml"


def load_trigger_config(path: Optional[str]) -> dict:
    config_path = Path(path).resolve() if path else default_trigger_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Trigger plausibility config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Trigger plausibility config must be a mapping: {config_path}")
    payload["_resolved_path"] = str(config_path)
    return payload


def _normalized_lookup(values: Sequence[object]) -> set:
    return {normalize_token(value) for value in values if normalize_token(value)}


def build_lookup_tables(config: dict) -> dict:
    closed = config.get("closed_class_lemmas", {}) or {}
    return {
        "closed": {key: _normalized_lookup(values if isinstance(values, list) else []) for key, values in closed.items()},
        "auxiliary": _normalized_lookup(config.get("auxiliary_lemmas", []) or []),
        "copula": _normalized_lookup(config.get("copula_lemmas", []) or []),
        "light_verb": _normalized_lookup(config.get("light_verb_lemmas", []) or []),
        "verb_suffixes": tuple(str(value) for value in config.get("verb_suffixes", []) or [] if str(value)),
        "adjective_suffixes": tuple(str(value) for value in config.get("adjective_suffixes", []) or [] if str(value)),
        "coordination_markers": _normalized_lookup(config.get("coordination_markers", []) or []),
        "subordination_markers": _normalized_lookup(config.get("subordination_markers", []) or []),
    }


def classify_token(token_norm: str, token_raw: str, lookups: dict, *, upos: str = "") -> str:
    upos = str(upos or "").upper()
    if upos:
        if upos in {"PRON", "DET", "PART", "ADP", "CCONJ", "SCONJ", "PUNCT"}:
            return upos.lower()
        if upos in {"VERB", "ADJ", "NOUN", "PROPN", "NUM", "AUX"}:
            return {
                "VERB": "verb",
                "ADJ": "adjective",
                "NOUN": "noun",
                "PROPN": "proper_noun",
                "NUM": "number",
                "AUX": "auxiliary",
            }[upos]
    if not token_norm or token_norm in IGNORED_PUNCTUATION:
        return "punct"
    for group_name, values in (lookups.get("closed", {}) or {}).items():
        if token_norm in values:
            mapping = {
                "demonstratives_pronouns": "pronoun",
                "particles": "particle",
                "postpositions": "adp",
                "conjunctions": "conjunction",
                "discourse_particles": "particle",
            }
            return mapping.get(group_name, "closed_class")
    if token_norm in (lookups.get("copula") or set()):
        return "copula"
    if token_norm in (lookups.get("auxiliary") or set()):
        return "auxiliary"
    if token_norm in (lookups.get("light_verb") or set()):
        return "light_verb"
    if token_norm.isdigit():
        return "number"
    if any(token_norm.endswith(suffix) for suffix in (lookups.get("verb_suffixes") or ()) if suffix):
        return "verb"
    if any(token_norm.endswith(suffix) for suffix in (lookups.get("adjective_suffixes") or ()) if suffix):
        return "adjective"
    if any(char.isupper() for char in token_raw):
        return "proper_noun"
    return "lexical"


def sentence_tokens(sentence: str, config: dict) -> List[dict]:
    tokenizer = Tokenizer()
    raw_tokens, _ = tokenizer.tokenize(normalize_text(sentence))
    lookups = build_lookup_tables(config)
    out: List[dict] = []
    for raw in raw_tokens:
        norm = normalize_token(raw)
        if not norm or norm in IGNORED_PUNCTUATION:
            continue
        out.append(
            {
                "index": len(out),
                "text": raw,
                "norm": norm,
                "category": classify_token(norm, raw, lookups),
            }
        )
    return out


def _source_quality(source: str) -> float:
    return {
        "n_gram": 1.0,
        "split": 0.8,
        "single": 0.65,
        "single_prefix": 0.35,
        "multiword_prefix": 0.25,
    }.get(str(source or ""), 0.2)


def _best_evidence(candidate: dict) -> dict:
    evidence = candidate.get("evidence", []) or []
    best = None
    best_score = -1.0
    for row in evidence:
        if not isinstance(row, dict):
            continue
        span = row.get("span", [])
        start = int(span[0]) if isinstance(span, list) and len(span) == 2 else 0
        end = int(span[1]) if isinstance(span, list) and len(span) == 2 else start + 1
        quality = _source_quality(str(row.get("source", "")))
        score = quality + max(0, end - start) * 0.05
        if score > best_score:
            best = row
            best_score = score
    return best or {"span": [0, 1], "source": "unknown", "tokens": [], "span_type": "single"}


def _evidence_span_tokens(sentence_tokens_rows: Sequence[dict], evidence: dict) -> List[dict]:
    span = evidence.get("span", [])
    if not isinstance(span, list) or len(span) != 2:
        return []
    start, end = int(span[0]), int(span[1])
    return [row for row in sentence_tokens_rows if start <= int(row.get("index", -1)) < end]


def _choose_token_by_priority(
    evidence_tokens: Sequence[dict],
    priority_order: Sequence[Sequence[str]],
    *,
    prefer_rightmost: bool = False,
) -> Optional[dict]:
    rows = list(evidence_tokens)
    if not rows:
        return None
    for categories in priority_order:
        iterable = reversed(rows) if prefer_rightmost else rows
        for token in iterable:
            if str(token.get("category", "")) in set(categories):
                return token
    fallback_iterable = reversed(rows) if prefer_rightmost else rows
    for token in fallback_iterable:
        if token.get("category") not in {"punct"}:
            return token
    return rows[-1] if prefer_rightmost else rows[0]


def _choose_anchor_token(evidence_tokens: Sequence[dict]) -> Optional[dict]:
    priority_order = [
        ("verb",),
        ("adjective",),
        ("noun", "proper_noun", "lexical", "number"),
        ("light_verb",),
    ]
    return _choose_token_by_priority(evidence_tokens, priority_order, prefer_rightmost=False)


def _choose_predicate_center_token(evidence_tokens: Sequence[dict], clause_type: str = "") -> Optional[dict]:
    if not evidence_tokens:
        return None
    if clause_type == "change_of_state_resultative":
        priority_order = [
            ("adjective",),
            ("noun", "proper_noun", "lexical", "number"),
            ("verb",),
            ("light_verb",),
        ]
    else:
        priority_order = [
            ("verb",),
            ("adjective",),
            ("noun", "proper_noun", "lexical", "number"),
            ("light_verb",),
        ]
    return _choose_token_by_priority(evidence_tokens, priority_order, prefer_rightmost=False)


def _recover_now_clause_bonus(clause_type: str, config: dict) -> float:
    defaults = {
        "imperative_predicate": 0.06,
        "change_of_state_resultative": 0.07,
        "modal_lexical_event": 0.05,
        "light_verb_compound_event": 0.05,
    }
    key = f"{clause_type}_bonus"
    default = defaults.get(clause_type, 0.0)
    try:
        return float(config.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _frame_family(frame_id: str) -> str:
    return str(frame_id or "").split(".", 1)[0]


def _multiword_predicate(evidence_tokens: Sequence[dict]) -> bool:
    categories = {str(token.get("category", "")) for token in evidence_tokens}
    return len(evidence_tokens) > 1 and "light_verb" in categories and bool(categories.intersection({"verb", "adjective", "noun", "lexical"}))


def assess_candidates(
    sentence: str,
    ranked_candidates: Sequence[dict],
    *,
    selected_frame: str = "",
    clause_type: str = "",
    config: Optional[dict] = None,
) -> Tuple[List[dict], List[dict]]:
    cfg = config or load_trigger_config(None)
    penalties = cfg.get("mismatch_penalties", {}) or {}
    bonuses = cfg.get("source_bonus", {}) or {}
    lookups = build_lookup_tables(cfg)
    tokens = sentence_tokens(sentence, cfg)
    max_retrieval_score = max((float(item.get("score", 0.0) or 0.0) for item in ranked_candidates), default=1.0) or 1.0
    results: List[dict] = []

    for rank, candidate in enumerate(ranked_candidates, start=1):
        frame_id = str(candidate.get("frame_id", ""))
        evidence = _best_evidence(candidate)
        evidence_tokens = _evidence_span_tokens(tokens, evidence)
        anchor = _choose_anchor_token(evidence_tokens)
        predicate_center = _choose_predicate_center_token(evidence_tokens, clause_type=clause_type)
        anchor_category = str((anchor or {}).get("category", "closed_class"))
        anchor_text = str((anchor or {}).get("text", ""))
        anchor_norm = str((anchor or {}).get("norm", ""))
        anchor_index = int((anchor or {}).get("index", -1))
        predicate_center_category = str((predicate_center or anchor or {}).get("category", "closed_class"))
        predicate_center_text = str((predicate_center or anchor or {}).get("text", ""))
        predicate_center_norm = str((predicate_center or anchor or {}).get("norm", ""))
        predicate_center_index = int((predicate_center or anchor or {}).get("index", -1))
        lexical_count = sum(1 for token in evidence_tokens if str(token.get("category", "")) in {"verb", "adjective", "noun", "proper_noun", "lexical", "light_verb"})
        multiword_predicate = _multiword_predicate(evidence_tokens)
        effective_category = predicate_center_category or anchor_category
        anchor_is_support_only = anchor_category in SUPPORT_ONLY_CATEGORIES or anchor_category == "light_verb"
        predicate_center_is_richer = predicate_center_category in LEXICAL_PREDICATE_CATEGORIES and predicate_center_category != "light_verb"
        score = 0.0
        breakdown: Dict[str, float] = {}
        reject_reasons: List[str] = []
        mismatch_flags: List[str] = []

        retrieval_bonus = (float(candidate.get("score", 0.0) or 0.0) / max_retrieval_score) * float(bonuses.get("retrieval_score", 0.32) or 0.32)
        score += retrieval_bonus
        breakdown["retrieval_score"] = round(retrieval_bonus, 4)

        if lexical_count:
            lexical_bonus = float(bonuses.get("lexical_anchor", 0.22) or 0.22)
            score += lexical_bonus
            breakdown["lexical_anchor"] = round(lexical_bonus, 4)

        source_name = str(evidence.get("source", ""))
        if source_name == "n_gram":
            source_bonus = float(bonuses.get("contiguous_ngram", 0.18) or 0.18)
            score += source_bonus
            breakdown["contiguous_ngram"] = round(source_bonus, 4)
        elif source_name == "split":
            source_bonus = float(bonuses.get("discontinuous_match", 0.08) or 0.08)
            score += source_bonus
            breakdown["discontinuous_match"] = round(source_bonus, 4)
        elif source_name in {"single_prefix", "multiword_prefix"}:
            penalty_key = "variable_prefix" if source_name == "multiword_prefix" else "prefix_pattern"
            penalty = float(penalties.get(penalty_key, 0.16 if source_name == "single_prefix" else 0.18) or 0.0)
            score -= penalty
            breakdown[penalty_key] = round(-penalty, 4)
            mismatch_flags.append("trigger_pattern_mismatch")

        if multiword_predicate:
            multi_bonus = float(bonuses.get("multiword_predicate", 0.12) or 0.12)
            score += multi_bonus
            breakdown["multiword_predicate"] = round(multi_bonus, 4)

        if lexical_count > 1:
            span_bonus = float(bonuses.get("lexical_span_bonus", 0.08) or 0.08)
            score += span_bonus
            breakdown["lexical_span_bonus"] = round(span_bonus, 4)

        if selected_frame and frame_id == selected_frame:
            selected_bonus = float(bonuses.get("frame_selection_bonus", 0.08) or 0.08)
            score += selected_bonus
            breakdown["frame_selection_bonus"] = round(selected_bonus, 4)

        hard_reject = False
        if effective_category in {"pronoun", "determiner", "particle", "adp", "conjunction", "closed_class"} and not multiword_predicate:
            hard_reject = True
            reject_reasons.append("closed_class_trigger")
            penalty = float(penalties.get("closed_class_anchor", 1.0) or 1.0)
            score -= penalty
            breakdown["closed_class_anchor"] = round(-penalty, 4)

        recover_now_clause = clause_type in RECOVER_NOW_CLAUSE_TYPES
        support_penalty_scale = 1.0
        if recover_now_clause and lexical_count > 0 and predicate_center_category not in {"auxiliary", "copula", "particle", "adp", "conjunction", "pronoun", "determiner", "closed_class"}:
            support_penalty_scale = 0.45 if multiword_predicate else 0.6

        helper_only = effective_category in {"auxiliary", "copula"} and lexical_count == 0 and not multiword_predicate
        if helper_only:
            hard_reject = True
            reject_reasons.append("helper_only_trigger")
            penalty = float(penalties.get("helper_only", 0.7) or 0.7)
            score -= penalty
            breakdown["helper_only"] = round(-penalty, 4)
            mismatch_flags.append("helper_only_anchor")
        elif anchor_category == "auxiliary":
            base_penalty = float(penalties.get("auxiliary_anchor", 0.3) or 0.3) * support_penalty_scale
            penalty = base_penalty * (0.25 if predicate_center_is_richer else 1.0)
            score -= penalty
            breakdown["auxiliary_anchor"] = round(-penalty, 4)
            if not predicate_center_is_richer:
                mismatch_flags.append("auxiliary_anchor")
        elif anchor_category == "copula":
            base_penalty = float(penalties.get("copula_anchor", 0.24) or 0.24) * support_penalty_scale
            penalty = base_penalty * (0.25 if predicate_center_is_richer else 1.0)
            score -= penalty
            breakdown["copula_anchor"] = round(-penalty, 4)
            if not predicate_center_is_richer:
                mismatch_flags.append("copular_anchor")
        elif anchor_is_support_only and predicate_center_is_richer:
            support_penalty = float(penalties.get("support_material_present", 0.05) or 0.05)
            score -= support_penalty
            breakdown["support_material_present"] = round(-support_penalty, 4)

        recover_now_bonus = 0.0
        if not hard_reject and recover_now_clause and lexical_count > 0:
            recover_now_bonus = _recover_now_clause_bonus(clause_type, cfg)
            if recover_now_bonus:
                score += recover_now_bonus
                breakdown["recover_now_clause_bonus"] = round(recover_now_bonus, 4)

        trigger_score = round(max(0.0, min(1.5, score)), 4)
        if trigger_score < float(cfg.get("min_trigger_plausibility_score", 0.28) or 0.28) and not hard_reject:
            reject_reasons.append("low_trigger_plausibility")
        results.append(
            {
                "frame_id": frame_id,
                "frame_family": _frame_family(frame_id),
                "retrieval_rank": rank,
                "retrieval_score": round(float(candidate.get("score", 0.0) or 0.0), 4),
                "hits": int(candidate.get("hits", 0) or 0),
                "term_hits": int(candidate.get("term_hits", 0) or 0),
                "evidence": candidate.get("evidence", []) or [],
                "best_evidence": evidence,
                "evidence_source": source_name,
                "anchor_text": anchor_text,
                "anchor_norm": anchor_norm,
                "anchor_category": anchor_category,
                "anchor_index": anchor_index,
                "anchor_span": list(evidence.get("span", []) or []),
                "predicate_center_text": predicate_center_text,
                "predicate_center_norm": predicate_center_norm,
                "predicate_center_index": predicate_center_index,
                "predicate_center_span": [predicate_center_index, predicate_center_index + 1] if predicate_center_index >= 0 else list(evidence.get("span", []) or []),
                "predicate_center_category": predicate_center_category,
                "effective_trigger_category": effective_category,
                "recover_now_clause": recover_now_clause,
                "recover_now_bonus": round(recover_now_bonus, 4),
                "lexical_token_count": lexical_count,
                "multiword_predicate": multiword_predicate,
                "selected_by_frame_selection": bool(selected_frame and frame_id == selected_frame),
                "hard_reject": hard_reject,
                "trigger_plausibility_score": trigger_score,
                "score_breakdown": breakdown,
                "mismatch_flags": list(dict.fromkeys(mismatch_flags)),
                "reject_reasons": list(dict.fromkeys(reject_reasons)),
            }
        )
    return tokens, results


def compact_decision_row(row: dict, clause_type: str = "") -> dict:
    return {
        "frame_id": row.get("frame_id", ""),
        "frame_family": row.get("frame_family", ""),
        "clause_type": clause_type,
        "retrieval_rank": row.get("retrieval_rank"),
        "retrieval_score": row.get("retrieval_score"),
        "trigger_plausibility_score": row.get("trigger_plausibility_score"),
        "final_candidate_score": row.get("final_candidate_score"),
        "pre_bonus_final_score": row.get("pre_bonus_final_score"),
        "anchor_text": row.get("anchor_text", ""),
        "anchor_index": row.get("anchor_index", -1),
        "anchor_category": row.get("anchor_category", ""),
        "anchor_span": row.get("anchor_span", []),
        "predicate_center_text": row.get("predicate_center_text", ""),
        "predicate_center_norm": row.get("predicate_center_norm", ""),
        "predicate_center_index": row.get("predicate_center_index", -1),
        "predicate_center_span": row.get("predicate_center_span", []),
        "predicate_center_category": row.get("predicate_center_category", ""),
        "effective_trigger_category": row.get("effective_trigger_category", ""),
        "recover_now_clause": row.get("recover_now_clause", False),
        "recover_now_bonus": row.get("recover_now_bonus", 0.0),
        "competition_group_id": row.get("competition_group_id", ""),
        "competition_group_size": row.get("competition_group_size", 1),
        "predicate_center_label": row.get("predicate_center_label", ""),
        "predicate_center_score": row.get("predicate_center_score", 0.0),
        "same_center_lexical_winner_bonus_applied": row.get("same_center_lexical_winner_bonus_applied", 0.0),
        "mismatch_flags": row.get("mismatch_flags", []),
        "mismatch_severity": row.get("mismatch_severity", 0.0),
        "adjectival_bonus_applied": row.get("adjectival_bonus_applied", False),
        "adjectival_bonus_value": row.get("adjectival_bonus_value", 0.0),
        "threshold_crossed_by_bonus": row.get("threshold_crossed_by_bonus", False),
        "would_pass_without_bonus": row.get("would_pass_without_bonus", False),
        "downranked_for_mismatch": row.get("downranked_for_mismatch", False),
        "downranked_as_helper_like": row.get("downranked_as_helper_like", False),
        "helper_penalty_applied": row.get("helper_penalty_applied", 0.0),
        "mismatch_penalty_applied": row.get("mismatch_penalty_applied", 0.0),
        "same_center_competition_penalty_applied": row.get("same_center_competition_penalty_applied", 0.0),
        "has_cleaner_same_center_competitor": row.get("has_cleaner_same_center_competitor", False),
        "best_clean_competitor_frame": row.get("best_clean_competitor_frame", ""),
        "clean_competitor_is_lexical": row.get("clean_competitor_is_lexical", False),
        "same_frame_family_competitor": row.get("same_frame_family_competitor", False),
        "won_same_center_competition": row.get("won_same_center_competition", False),
        "competition_loser_count": row.get("competition_loser_count", 0),
        "competition_winner_frame_id": row.get("competition_winner_frame_id", ""),
        "competition_outcome_reason": row.get("competition_outcome_reason", ""),
        "kept_as_lexical_predicate_center": row.get("kept_as_lexical_predicate_center", False),
        "selected_by_frame_selection": row.get("selected_by_frame_selection", False),
        "decision": row.get("decision", ""),
        "decision_stage": row.get("decision_stage", ""),
        "rejection_reason": row.get("rejection_reason", ""),
        "score_breakdown": row.get("score_breakdown", {}),
    }


def sentence_decision_payload(
    *,
    sent_id: int,
    sentence: str,
    top_k_requested: int,
    clause_profile: dict,
    decisions: Sequence[dict],
    summary: dict,
    selected_frame: str,
    config_path: str,
) -> dict:
    clause_type = str(clause_profile.get("clause_type", "")) if isinstance(clause_profile, dict) else ""
    return {
        "sent_id": int(sent_id),
        "sent_key": f"batch_run::{int(sent_id)}",
        "sentence_text": sentence,
        "candidate_top_k_requested": int(top_k_requested),
        "selected_frame": selected_frame,
        "clause_profile": clause_profile,
        "summary": summary,
        "trigger_config_path": config_path,
        "candidate_decisions": [compact_decision_row(row, clause_type=clause_type) for row in decisions],
    }


def append_candidate_decision_log(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
