#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 4b: Gold Batch Pipeline (Real-World Hindi IE)

Upgrades:
- Token-level n-gram trigger matching with optional gap handling.
- Trigger normalization (Unicode + punctuation + lightweight verb normalization).
- Candidate ranking and optional two-step frame selection.
- Multi-event emission via a per-sentence `events` wrapper when needed.
- Per-event provenance enrichment for downstream KG materialization.
"""

import argparse
import json
import sys
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Force UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

try:
    import llama_cpp
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    sys.exit("Error: pip install llama-cpp-python")

try:
    from gold.span import Tokenizer
    from gold.trigger_plausibility import (
        append_candidate_decision_log,
        assess_candidates,
        build_lookup_tables,
        classify_token,
        load_trigger_config,
        sentence_decision_payload,
    )
    from gold.clause_typing import classify_clause_profile
    from gold.candidate_dedup import select_final_candidates
except ImportError:
    # Allow running from root dir
    sys.path.append(str(Path(__file__).parent.parent))
    from gold.span import Tokenizer
    from gold.trigger_plausibility import (
        append_candidate_decision_log,
        assess_candidates,
        build_lookup_tables,
        classify_token,
        load_trigger_config,
        sentence_decision_payload,
    )
    from gold.clause_typing import classify_clause_profile
    from gold.candidate_dedup import select_final_candidates

PIPELINE_SCHEMA_VERSION = "2026-02-16"

OPTIONAL_GAP_TOKENS = {"ही", "भी", "तो", "नहीं", "है", "हैं", "हूँ"}
VERB_SUFFIXES = [
    "या", "यी", "ए", "ता", "ती", "ते",
    "गया", "गई", "गए", "रहा", "रही", "रहे",
    "पाया", "पायी", "पाए",
    "दिया", "दी", "दिए",
]
TRIGGER_TERMINAL = "__frames__"
MAX_VARIABLE_PREFIX = 1
IGNORED_PUNCTUATION = {"।", "।", ",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "|", '"', "'", "`", "“", "”", "’", "‘"}
CASE_MARKER_TOKENS = {
    "ने",
    "को",
    "से",
    "में",
    "पर",
    "तक",
    "का",
    "की",
    "के",
    "लिए",
    "साथ",
    "वाला",
    "वालों",
    "वाले",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u0965", "\u0964")
    normalized = normalized.replace("\uFEFF", "")
    normalized = normalized.replace("\u200d", "")
    normalized = normalized.replace("\u200c", "")
    return " ".join(normalized.split())


def normalize_token(token: str) -> str:
    if not token:
        return ""
    tok = unicodedata.normalize("NFKC", token.strip())
    tok = tok.strip()
    tok = tok.strip("".join(sorted(IGNORED_PUNCTUATION)))
    return tok


def normalize_verb(token: str) -> str:
    if not token:
        return token
    for suffix in VERB_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[:-len(suffix)]
    return token


class FrameIndex:
    """Indexes frames using token-normalized trigger and lemma forms."""

    def __init__(self, registry_path: str):
        self.trie: Dict = {}
        self.token_frames: Dict[str, List[str]] = {}
        self.multiword_patterns_by_first: Dict[str, List[dict]] = {}
        self.trigger_with_variable_prefix: Dict[str, List[str]] = {}
        self.multiword_prefix_index: Dict[str, List[dict]] = {}
        self.frame_terms: Dict[str, set] = {}
        self.tokenizer = Tokenizer()
        self.load_registry(registry_path)

    def load_registry(self, path: str) -> None:
        print(f"[INFO] Building Trigger Index from {path}...", file=sys.stderr)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        classes = data.get("classes", {})
        count = 0

        for frame_id, cls in classes.items():
            terms = set()

            for member in cls.get("members", []):
                lemma = member.get("lemma")
                if lemma:
                    self._add_trigger(frame_id, str(lemma))
                    terms.add(normalize_token(lemma))

            for trigger in cls.get("example_triggers", []):
                if trigger:
                    self._add_trigger(frame_id, str(trigger))
                    for part in self._split_trigger(str(trigger)):
                        if part:
                            terms.add(part)

            self.frame_terms[frame_id] = terms
            count += 1

        print(f"[OK] Indexed triggers for {count} frames.", file=sys.stderr)

    def _split_trigger(self, text: str) -> List[str]:
        trigger_tokens = []
        tokens, _ = self.tokenizer.tokenize(normalize_text(text))
        for token in tokens:
            norm = normalize_token(token)
            if not norm or norm in IGNORED_PUNCTUATION:
                continue
            trigger_tokens.append(norm)
        return trigger_tokens

    @staticmethod
    def _candidate_token_variants(token: str) -> List[str]:
        token = normalize_token(token)
        if not token:
            return []

        variants = {token}
        verb_norm = normalize_verb(token)
        if verb_norm and verb_norm != token:
            variants.add(verb_norm)

        # Hindi participial variants commonly seen in corpora:
        # गए -> गया, लाए -> लाया, आए -> आया, आदि
        if token.endswith("ए") and len(token) > 1:
            variants.add(token[:-1] + "या")
        # सुनाई -> सुनाई? (already), सुनाई -> सुना
        if token.endswith("ई") and len(token) > 2 and token[-2] == "ा":
            variants.add(token[:-2] + "ा")

        return list(variants)

    @staticmethod
    def _is_viable_prefix_token(tok: str) -> bool:
        return bool(tok) and len(tok) > 1 and tok not in IGNORED_PUNCTUATION and tok not in CASE_MARKER_TOKENS

    @staticmethod
    def _token_variants(tokens: List[str]) -> List[tuple]:
        if not tokens:
            return []

        normalized = tuple(tokens)
        variants = {normalized}
        normalized_verb = tuple(normalize_verb(t) for t in tokens)
        variants.add(normalized_verb)
        return list(variants)

    def _add_trigger(self, frame_id: str, trigger: str) -> None:
        parts = self._split_trigger(trigger)
        if not parts:
            return

        for variant in self._token_variants(parts):
            if len(variant) == 1:
                term = variant[0]
                self.token_frames.setdefault(term, [])
                if frame_id not in self.token_frames[term]:
                    self.token_frames[term].append(frame_id)
                if self._is_viable_prefix_token(term):
                    prefix_bucket = self.trigger_with_variable_prefix.setdefault(term, [])
                    if frame_id not in prefix_bucket:
                        prefix_bucket.append(frame_id)
                continue

            node = self.trie
            for part in variant:
                node = node.setdefault(part, {})
            entries = node.setdefault(TRIGGER_TERMINAL, [])
            entry = {"frame_id": frame_id, "tokens": variant}
            if entry not in entries:
                entries.append(entry)

            first = variant[0]
            bucket = self.multiword_patterns_by_first.setdefault(first, [])
            if all(e["frame_id"] != frame_id or e["tokens"] != variant for e in bucket):
                bucket.append({"frame_id": frame_id, "tokens": variant})

            last = variant[-1]
            suffix_bucket = self.multiword_prefix_index.setdefault(last, [])
            suffix_entry = {"frame_id": frame_id, "tokens": variant}
            if all(e["frame_id"] != frame_id or e["tokens"] != variant for e in suffix_bucket):
                suffix_bucket.append(suffix_entry)

    def _match_discontinuous(self, tokens: List[str], pattern: tuple, start_idx: int, max_gap: int = 3) -> Optional[dict]:
        if start_idx >= len(tokens):
            return None
        if tokens[start_idx] != pattern[0]:
            return None

        pos = start_idx
        for part in pattern[1:]:
            found = None
            for next_pos in range(pos + 1, min(len(tokens), pos + max_gap + 1)):
                tok = tokens[next_pos]
                if tok in OPTIONAL_GAP_TOKENS and tok != part:
                    continue
                if tok == part:
                    found = next_pos
                    break
            if found is None:
                return None
            pos = found

        return {"start": start_idx, "end": pos + 1}

    def find_candidates(self, sentence: str, top_k: int = 6) -> List[dict]:
        norm_sentence = normalize_text(sentence)
        sentence_tokens_raw, _ = self.tokenizer.tokenize(norm_sentence)
        sentence_tokens = [normalize_token(t) for t in sentence_tokens_raw]
        sentence_tokens = [t for t in sentence_tokens if t and t not in IGNORED_PUNCTUATION]
        token_set = set(sentence_tokens)

        candidates: Dict[str, dict] = {}

        def add(frame_id: str, score: int, source: str, start: int, end: int, matched: List[str], span_type: str):
            info = candidates.setdefault(
                frame_id,
                {
                    "frame_id": frame_id,
                    "score": 0,
                    "hits": 0,
                    "evidence": [],
                },
            )
            info["score"] += score
            info["hits"] += 1
            info["evidence"].append(
                {
                    "source": source,
                    "span": [start, end],
                    "tokens": matched,
                    "span_type": span_type,
                }
            )

        for idx, tok in enumerate(sentence_tokens):
            seen_single = set()
            for tok_variant in self._candidate_token_variants(tok):
                for frame_id in self.token_frames.get(tok_variant, []):
                    if frame_id in seen_single:
                        continue
                    seen_single.add(frame_id)
                    add(frame_id, 2, "single", idx, idx + 1, [tok_variant], "single")

            seen_prefix = set()
            for tok_variant in self._candidate_token_variants(tok):
                for frame_id in self.trigger_with_variable_prefix.get(tok_variant, []):
                    if frame_id in seen_prefix:
                        continue
                    # Generic rule for `X + trigger` patterns where trigger is a single token.
                    # If trigger token is known but not necessarily root-mapped yet, still keep this as a low-weight hint.
                    if frame_id not in self.token_frames.get(tok_variant, []):
                        continue
                    seen_prefix.add(frame_id)
                    if idx < 1 or not self._is_viable_prefix_token(sentence_tokens[idx - 1]):
                        continue
                    add(
                        frame_id,
                        1,
                        "single_prefix",
                        idx - 1,
                        idx + 1,
                        sentence_tokens[idx - 1 : idx + 1],
                        "prefix",
                    )

                for entry in self.multiword_prefix_index.get(tok_variant, []):
                    frame_id = entry["frame_id"]
                    if frame_id in seen_prefix:
                        continue
                    for prefix_len in range(1, min(MAX_VARIABLE_PREFIX, idx) + 1):
                        start = idx - prefix_len
                        if not self._is_viable_prefix_token(sentence_tokens[start]):
                            continue
                        seen_prefix.add(frame_id)
                        add(
                            frame_id,
                            1,
                            "multiword_prefix",
                            start,
                            idx + 1,
                            sentence_tokens[start : idx + 1],
                            "variable_prefix",
                        )

        # Exact contiguous n-gram matches with trie.
        n = len(sentence_tokens)
        for start in range(n):
            node = self.trie
            for end in range(start, n):
                tok = sentence_tokens[end]
                if tok not in node:
                    break
                node = node[tok]
                terminal = node.get(TRIGGER_TERMINAL, [])
                for entry in terminal:
                    entry_tokens = entry["tokens"]
                    add(
                        entry["frame_id"],
                        5 + len(entry_tokens),
                        "n_gram",
                        start,
                        end + 1,
                        list(entry_tokens),
                        "contiguous",
                    )

        # Loose discontinuous matching (e.g. split constructions, close-by auxiliaries).
        for start, tok in enumerate(sentence_tokens):
            for pattern in self.multiword_patterns_by_first.get(tok, []):
                frame_id = pattern["frame_id"]
                pattern_tokens = pattern["tokens"]
                if len(pattern_tokens) < 2:
                    continue
                match = self._match_discontinuous(sentence_tokens, pattern_tokens, start)
                if not match:
                    continue
                add(
                    frame_id,
                    4 + len(pattern_tokens),
                    "split",
                    match["start"],
                    match["end"],
                    list(pattern_tokens),
                    "discontinuous",
                )

        for frame_id, terms in self.frame_terms.items():
            if frame_id not in candidates:
                continue
            overlap = len(terms.intersection(token_set))
            if overlap:
                candidates[frame_id]["score"] += overlap
                candidates[frame_id]["term_hits"] = overlap

        ranked = sorted(
            candidates.values(),
            key=lambda x: (x["score"], x.get("hits", 0), x["frame_id"]),
            reverse=True,
        )
        return ranked[:top_k]


class GoldPipeline:
    def __init__(
        self,
        model_path: str,
        schema_dir: str,
        registry_path: str,
        *,
        ctx: int = 2048,
        max_events: int = 3,
        max_candidates: int = 6,
        use_two_step_selection: bool = True,
        quarantine_path: Optional[str] = None,
        llama_gpu_layers: int = -1,
        llama_main_gpu: int = -1,
        llama_tensor_split: Optional[List[float]] = None,
        llama_split_mode: str = "layer",
        trigger_config: Optional[str] = None,
        candidate_decision_log: Optional[str] = None,
        min_trigger_score: float = -1.0,
        min_final_candidate_score: float = -1.0,
        enable_candidate_dedup: bool = True,
        max_final_candidates_per_clause: int = 0,
        run_id: Optional[str] = None,
    ):
        self.schema_dir = Path(schema_dir)
        self.tokenizer = Tokenizer()
        self.max_events = max_events
        self.max_candidates = max_candidates
        self.use_two_step_selection = use_two_step_selection
        self.run_id = str(run_id or "").strip() or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        self.model_id = Path(model_path).name
        self.quarantine_path = quarantine_path
        self.candidate_decision_log = candidate_decision_log

        self.index = FrameIndex(registry_path)
        self.trigger_config = load_trigger_config(trigger_config)
        if float(min_trigger_score) >= 0:
            self.trigger_config["min_trigger_plausibility_score"] = float(min_trigger_score)
        if float(min_final_candidate_score) >= 0:
            self.trigger_config["min_final_candidate_score"] = float(min_final_candidate_score)
        if int(max_final_candidates_per_clause or 0) > 0:
            self.trigger_config["max_final_candidates_per_clause"] = int(max_final_candidates_per_clause)
        self.trigger_config["enable_candidate_dedup"] = bool(enable_candidate_dedup)
        self.trigger_config_path = str(self.trigger_config.get("_resolved_path", trigger_config or ""))
        self.trigger_lookups = build_lookup_tables(self.trigger_config)

        print(f"[INFO] Loading Model: {model_path}...", file=sys.stderr)
        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": ctx,
            "n_gpu_layers": int(llama_gpu_layers),
            "verbose": False,
        }
        if int(llama_main_gpu) >= 0:
            llama_kwargs["main_gpu"] = int(llama_main_gpu)
        if llama_tensor_split:
            llama_kwargs["tensor_split"] = [float(value) for value in llama_tensor_split]
            split_mode_name = str(llama_split_mode or "layer").strip().lower()
            split_mode_lookup = {
                "none": getattr(llama_cpp, "LLAMA_SPLIT_MODE_NONE", None),
                "layer": getattr(llama_cpp, "LLAMA_SPLIT_MODE_LAYER", None),
                "row": getattr(llama_cpp, "LLAMA_SPLIT_MODE_ROW", None),
            }
            split_mode_value = split_mode_lookup.get(split_mode_name)
            if split_mode_value is not None:
                llama_kwargs["split_mode"] = split_mode_value
        self.llm = Llama(**llama_kwargs)

        self.grammar_cache: Dict[str, LlamaGrammar] = {}
        self.schema_cache: Dict[str, dict] = {}
        self.selection_grammar_cache: Dict[str, LlamaGrammar] = {}

    def _log_sentence_decision(
        self,
        *,
        sent_id: int,
        sentence: str,
        selected_frame: str,
        clause_profile: dict,
        decisions: List[dict],
        summary: dict,
    ) -> None:
        append_candidate_decision_log(
            self.candidate_decision_log,
            sentence_decision_payload(
                sent_id=sent_id,
                sentence=sentence,
                top_k_requested=self.max_candidates,
                clause_profile=clause_profile,
                decisions=decisions,
                summary=summary,
                selected_frame=selected_frame,
                config_path=self.trigger_config_path,
            ),
        )

    def _precision_filter(self, sentence: str, sent_id: int, ranked_candidates: List[dict], selected_frame: str) -> tuple:
        sentence_tokens, assessments = assess_candidates(
            sentence,
            ranked_candidates,
            selected_frame=selected_frame,
            config=self.trigger_config,
        )
        clause_profile = classify_clause_profile(sentence_tokens, assessments, self.trigger_config)
        clause_type = str(clause_profile.get("clause_type", "") or "")
        if clause_type in {"imperative_predicate", "change_of_state_resultative", "modal_lexical_event", "light_verb_compound_event"}:
            sentence_tokens, assessments = assess_candidates(
                sentence,
                ranked_candidates,
                selected_frame=selected_frame,
                clause_type=clause_type,
                config=self.trigger_config,
            )
            clause_profile = classify_clause_profile(sentence_tokens, assessments, self.trigger_config)
        accepted, decisions, summary = select_final_candidates(
            assessments,
            clause_profile,
            max_events=self.max_events,
            min_final_score=float(self.trigger_config.get("min_final_candidate_score", 0.46) or 0.46),
            enable_candidate_dedup=bool(self.trigger_config.get("enable_candidate_dedup", True)),
            max_final_candidates_per_clause=int(
                self.trigger_config.get("max_final_candidates_per_clause", clause_profile.get("max_candidates_per_clause", 2))
                or clause_profile.get("max_candidates_per_clause", 2)
            ),
            config=self.trigger_config,
        )
        self._log_sentence_decision(
            sent_id=sent_id,
            sentence=sentence,
            selected_frame=selected_frame,
            clause_profile=clause_profile,
            decisions=decisions,
            summary=summary,
        )
        return accepted, decisions, summary, clause_profile

    def _append_quarantine(self, payload: dict) -> None:
        if not self.quarantine_path:
            return
        try:
            with open(self.quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            print(f"[WARN] Failed to write quarantine row to {self.quarantine_path}", file=sys.stderr)

    def get_grammar(self, frame_id: str) -> Optional[LlamaGrammar]:
        if frame_id in self.grammar_cache:
            return self.grammar_cache[frame_id]

        schema_path = self.schema_dir / f"{frame_id}.schema.json"
        if not schema_path.exists():
            return None

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_obj = json.load(f)
            self.schema_cache[frame_id] = schema_obj
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema_obj))
            self.grammar_cache[frame_id] = grammar
            return grammar
        except Exception as e:
            print(f"[ERROR] Grammar compile failed for {frame_id}: {e}", file=sys.stderr)
            return None

    def make_prompt(self, sentence: str, frame_id: str) -> str:
        tokens, _ = self.tokenizer.tokenize(sentence)
        token_map = "\n".join([f"{i}: {token}" for i, token in enumerate(tokens)])

        schema = self.schema_cache.get(frame_id, {})
        args = schema.get("properties", {}).get("arguments", {}).get("properties", {})

        role_defs = []
        for role, details in args.items():
            desc = details.get("description", "No description.")
            role_defs.append(f"- {role}: {desc}")
        role_block = "\n".join(role_defs)

        return f"""<|system|>
You are an expert Semantic Role Labeling (SRL) system for Hindi.
Your task is to extract one event instance of frame '{frame_id}'.

ROLE DEFINITIONS:
{role_block}

INSTRUCTIONS:
1. Analyze the sentence and extract only arguments required by this frame.
2. Use token indices from [start, end) in the `span` fields.
3. Include Hindi case markers immediately attached to a role mention in the span text when they belong to that argument
   (for example: `राम ने`, `रावण को`, `अयोध्या से`).
4. Return strictly valid JSON.
5. Keep punctuation out of argument spans.

<|user|>
Sentence: {sentence}

Token Map:
{token_map}

Extract frame '{frame_id}'.
<|assistant|>
"""

    def make_selection_prompt(self, sentence: str, candidates: List[str]) -> str:
        candidate_lines = "\n".join([f"- {cid}" for cid in candidates])
        return f"""<|system|>
You are a frame selection agent.
Choose the best UHVN frame for the given sentence from the candidate list.
Return JSON: {{"frame": "FRAME_ID", "reason": "short rationale"}}.

Sentence: {sentence}

Candidates:
{candidate_lines}

<|assistant|>
"""

    def _selection_grammar(self, candidates: List[str]) -> LlamaGrammar:
        key = "|".join(sorted(candidates))
        if key in self.selection_grammar_cache:
            return self.selection_grammar_cache[key]

        schema = {
            "type": "object",
            "properties": {
                "frame": {
                    "type": "string",
                    "enum": candidates,
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": ["frame"],
            "additionalProperties": True,
        }
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
        self.selection_grammar_cache[key] = grammar
        return grammar

    def select_frame(self, sentence: str, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        prompt = self.make_selection_prompt(sentence, candidates)
        grammar = self._selection_grammar(candidates)
        output = self.llm(
            prompt,
            max_tokens=80,
            temperature=0.0,
            grammar=grammar,
            stop=["<|user|>", "</s>", "<|assistant|>"],
        )

        try:
            selected = json.loads(output["choices"][0]["text"])
            frame = selected.get("frame")
            if frame in candidates:
                return frame
        except Exception:
            pass

        return candidates[0]

    def _token_span_to_char_span(self, tokens, offsets, span: list) -> Optional[list]:
        if not span or len(span) != 2:
            return None
        start, end = span
        if start < 0 or end <= start or end > len(offsets):
            return None
        start_ch = offsets[start][0]
        end_ch = offsets[end - 1][1]
        return [start_ch, end_ch]

    @staticmethod
    def _expand_marker_span(tokens, span: list, max_markers: int = 2) -> tuple[list, bool]:
        if not span or len(span) != 2:
            return list(span) if isinstance(span, list) else [], False
        try:
            start, end = map(int, span)
        except Exception:
            return list(span), False

        if start < 0 or end <= start or end > len(tokens):
            return [start, end], False

        expanded_end = end
        expanded = False
        for _ in range(max_markers):
            if expanded_end >= len(tokens):
                break

            marker = normalize_token(tokens[expanded_end])
            if marker in CASE_MARKER_TOKENS:
                expanded_end += 1
                expanded = True
                continue
            break

        if not expanded:
            return [start, end], False
        return [start, expanded_end], True

    def _enrich_marker_spans(self, extraction: dict, tokens: list, warnings: List[str]) -> None:
        for key in ["trigger", "arguments"]:
            if key == "arguments":
                items = extraction.get("arguments", {})
                if isinstance(items, dict):
                    iterable = items.items()
                else:
                    continue
                for role_name, arg_data in iterable:
                    if not isinstance(arg_data, dict):
                        continue
                    span = arg_data.get("span")
                    if not isinstance(span, list):
                        continue
                    new_span, changed = self._expand_marker_span(tokens, span)
                    if not changed:
                        continue
                    old_span = span
                    arg_data["span"] = new_span
                    resolved = self.tokenizer.resolve_span(new_span, tokens)
                    if resolved is not None:
                        arg_data["text"] = resolved
                    warnings.append(
                        f"{role_name}: span {old_span} expanded to include case marker token(s) -> {new_span}"
                    )
            else:
                trigger = extraction.get("trigger")
                if not isinstance(trigger, dict):
                    continue
                span = trigger.get("span")
                if not isinstance(span, list):
                    continue
                new_span, changed = self._expand_marker_span(tokens, span, max_markers=1)
                if not changed:
                    continue
                trigger["span"] = new_span
                resolved = self.tokenizer.resolve_span(new_span, tokens)
                if resolved is not None:
                    trigger["text"] = resolved
                warnings.append(f"trigger: span {span} expanded to include case marker token(s) -> {new_span}")

    def _trigger_category_from_span(self, tokens: list, span: list) -> str:
        if not isinstance(span, list) or len(span) != 2:
            return ""
        try:
            start, end = map(int, span)
        except Exception:
            return ""
        if start < 0 or end <= start or end > len(tokens):
            return ""
        chosen = self._preferred_trigger_token(tokens, start, end)
        if not chosen:
            return ""
        norm = normalize_token(chosen)
        return classify_token(norm, chosen, self.trigger_lookups)

    def _preferred_trigger_index(self, tokens: list, start: int, end: int) -> int:
        if start < 0 or end <= start or end > len(tokens):
            return -1
        rows = []
        for index in range(start, end):
            token = tokens[index]
            norm = normalize_token(token)
            if not norm or norm in CASE_MARKER_TOKENS:
                continue
            category = classify_token(norm, token, self.trigger_lookups)
            rows.append((index, token, category))
        if not rows:
            return end - 1
        priority_order = [
            {"verb"},
            {"adjective"},
            {"noun", "proper_noun", "lexical", "number"},
            {"light_verb"},
        ]
        for categories in priority_order:
            for index, _, category in rows:
                if category in categories:
                    return index
        return rows[0][0]

    def _preferred_trigger_token(self, tokens: list, start: int, end: int) -> str:
        preferred_index = self._preferred_trigger_index(tokens, start, end)
        if 0 <= preferred_index < len(tokens):
            return tokens[preferred_index]
        return ""

    def _preferred_trigger_span(self, tokens: list, span: list) -> Optional[list]:
        if not isinstance(span, list) or len(span) != 2:
            return None
        try:
            start, end = map(int, span)
        except Exception:
            return None
        preferred_index = self._preferred_trigger_index(tokens, start, end)
        if 0 <= preferred_index < len(tokens):
            return [preferred_index, preferred_index + 1]
        return None

    def _candidate_anchor_span(self, candidate_decision: dict, tokens: list) -> Optional[list]:
        if not isinstance(candidate_decision, dict):
            return None
        predicate_center_span = candidate_decision.get("predicate_center_span")
        if isinstance(predicate_center_span, list) and len(predicate_center_span) == 2:
            try:
                start, end = map(int, predicate_center_span)
            except Exception:
                start, end = -1, -1
            if 0 <= start < end <= len(tokens):
                return [start, end]
        predicate_center_index = int(candidate_decision.get("predicate_center_index", -1) or -1)
        if 0 <= predicate_center_index < len(tokens):
            return [predicate_center_index, predicate_center_index + 1]
        if bool(candidate_decision.get("multiword_predicate")):
            best_evidence = candidate_decision.get("best_evidence", {}) or {}
            evidence_span = best_evidence.get("span")
            if isinstance(evidence_span, list) and len(evidence_span) == 2:
                try:
                    start, end = map(int, evidence_span)
                except Exception:
                    start, end = -1, -1
                if 0 <= start < end <= len(tokens):
                    return [start, end]
        anchor_index = int(candidate_decision.get("anchor_index", -1) or -1)
        if 0 <= anchor_index < len(tokens):
            return [anchor_index, anchor_index + 1]
        anchor_span = candidate_decision.get("anchor_span")
        if isinstance(anchor_span, list) and len(anchor_span) == 2:
            try:
                start, end = map(int, anchor_span)
            except Exception:
                return None
            if 0 <= start < end <= len(tokens):
                return [start, end]
        return None

    def _reanchor_trigger_from_candidate(
        self,
        extraction: dict,
        tokens: list,
        candidate_decision: Optional[dict],
        warnings: List[str],
    ) -> Optional[dict]:
        if not bool(self.trigger_config.get("trigger_reanchoring_enabled", True)):
            return None
        if not isinstance(candidate_decision, dict):
            return None
        trigger = extraction.get("trigger")
        if not isinstance(trigger, dict) or not isinstance(trigger.get("span"), list):
            return None

        current_span = list(trigger.get("span") or [])
        current_category = self._trigger_category_from_span(tokens, current_span)
        current_preferred_span = self._preferred_trigger_span(tokens, current_span)
        reanchor_categories = {
            str(value) for value in (self.trigger_config.get("trigger_reanchor_categories", []) or []) if str(value)
        }
        mismatch_flags = {
            str(flag) for flag in (candidate_decision.get("mismatch_flags", []) or []) if str(flag)
        }
        reanchor_flags = {
            str(value) for value in (self.trigger_config.get("trigger_reanchor_on_mismatch_flags", []) or []) if str(value)
        }
        lexical_trigger_categories = {"verb", "adjective", "noun", "proper_noun", "lexical", "light_verb", "number"}
        helper_like_categories = set(reanchor_categories)
        lexical_trigger_is_already_centered = (
            current_category in lexical_trigger_categories
            and isinstance(current_preferred_span, list)
            and current_preferred_span == current_span
        )
        mismatch_justifies_reanchor = bool(mismatch_flags.intersection(reanchor_flags)) and not lexical_trigger_is_already_centered

        target_span = self._candidate_anchor_span(candidate_decision, tokens)
        if not target_span:
            return None
        target_preferred_span = self._preferred_trigger_span(tokens, target_span) or target_span
        target_category = self._trigger_category_from_span(tokens, target_preferred_span)
        span_needs_narrowing = (
            isinstance(current_preferred_span, list)
            and current_preferred_span == target_preferred_span
            and current_span != target_preferred_span
        )
        center_mismatch = (
            isinstance(current_preferred_span, list)
            and current_preferred_span != target_preferred_span
        )
        should_reanchor = (
            current_category in reanchor_categories
            or mismatch_justifies_reanchor
            or span_needs_narrowing
            or center_mismatch
        )
        if not should_reanchor:
            return None
        if target_preferred_span == current_span:
            return None
        if target_category in helper_like_categories:
            return None

        old_text = trigger.get("text", "")
        trigger["span"] = target_preferred_span
        resolved = self.tokenizer.resolve_span(target_preferred_span, tokens)
        if resolved is not None:
            trigger["text"] = resolved
        trigger["category"] = target_category
        warnings.append(
            f"trigger reanchored from {current_span} ({old_text}) to {target_preferred_span} ({trigger.get('text', '')}) via accepted predicate center"
        )
        return {
            "applied": True,
            "reason": "trigger_reanchored_to_predicate_center",
            "from_span": current_span,
            "to_span": target_preferred_span,
            "from_category": current_category,
            "to_category": target_category,
            "from_text": old_text,
            "from_preferred_span": current_preferred_span or current_span,
            "to_text": trigger.get("text", ""),
            "mismatch_flags": sorted(mismatch_flags),
        }

    def _enrich_extraction(
        self,
        extraction: dict,
        sentence: str,
        sentence_id: int,
        tokens,
        offsets,
        frame_id: str,
        ranked_candidates: List[dict],
        selected_frame: str,
        warnings: List[str],
        candidate_decision: Optional[dict] = None,
        sentence_precision_summary: Optional[dict] = None,
        clause_profile: Optional[dict] = None,
        trigger_reanchoring: Optional[dict] = None,
    ) -> None:
        extraction["doc_id"] = "batch_run"
        extraction["sent_id"] = sentence_id
        extraction["frame"] = frame_id

        # Add char spans for traceability.
        trigger = extraction.get("trigger", {})
        if isinstance(trigger, dict) and isinstance(trigger.get("span"), list):
            char_span = self._token_span_to_char_span(tokens, offsets, trigger.get("span"))
            if char_span:
                trigger["char_span"] = char_span
            trigger["category"] = self._trigger_category_from_span(tokens, trigger.get("span"))

        for arg_data in extraction.get("arguments", {}).values():
            if not isinstance(arg_data, dict):
                continue
            span = arg_data.get("span")
            if isinstance(span, list):
                char_span = self._token_span_to_char_span(tokens, offsets, span)
                if char_span:
                    arg_data["char_span"] = char_span

        event_meta = extraction.setdefault("meta", {})
        event_meta["schema_version"] = PIPELINE_SCHEMA_VERSION
        event_meta["model"] = self.model_id
        event_meta["run_id"] = self.run_id
        event_meta["frame_selection"] = selected_frame
        event_meta["frame_candidates"] = [c["frame_id"] for c in ranked_candidates]
        event_meta["trigger_config_path"] = self.trigger_config_path
        if isinstance(candidate_decision, dict):
            final_trigger_preferred_span = self._preferred_trigger_span(tokens, trigger.get("span")) or trigger.get("span", [])
            event_meta["candidate_acceptance"] = {
                "frame_family": candidate_decision.get("frame_family", ""),
                "retrieval_rank": candidate_decision.get("retrieval_rank"),
                "retrieval_score": candidate_decision.get("retrieval_score"),
                "trigger_plausibility_score": candidate_decision.get("trigger_plausibility_score"),
                "final_candidate_score": candidate_decision.get("final_candidate_score"),
                "anchor_text": candidate_decision.get("anchor_text", ""),
                "anchor_index": candidate_decision.get("anchor_index", -1),
                "anchor_category": candidate_decision.get("anchor_category", ""),
                "anchor_span": candidate_decision.get("anchor_span", []),
                "predicate_center_text": candidate_decision.get("predicate_center_text", ""),
                "predicate_center_norm": candidate_decision.get("predicate_center_norm", ""),
                "predicate_center_index": candidate_decision.get("predicate_center_index", -1),
                "predicate_center_span": candidate_decision.get("predicate_center_span", []),
                "predicate_center_category": candidate_decision.get("predicate_center_category", ""),
                "effective_trigger_category": candidate_decision.get("effective_trigger_category", ""),
                "predicate_center_label": candidate_decision.get("predicate_center_label", ""),
                "predicate_center_score": candidate_decision.get("predicate_center_score", 0.0),
                "mismatch_severity": candidate_decision.get("mismatch_severity", 0.0),
                "recover_now_clause": candidate_decision.get("recover_now_clause", False),
                "recover_now_bonus": candidate_decision.get("recover_now_bonus", 0.0),
                "competition_group_id": candidate_decision.get("competition_group_id", ""),
                "clean_competitor_is_lexical": candidate_decision.get("clean_competitor_is_lexical", False),
                "same_frame_family_competitor": candidate_decision.get("same_frame_family_competitor", False),
                "same_center_lexical_winner_bonus_applied": candidate_decision.get("same_center_lexical_winner_bonus_applied", 0.0),
                "kept_as_lexical_predicate_center": candidate_decision.get("kept_as_lexical_predicate_center", False),
                "selected_by_frame_selection": candidate_decision.get("selected_by_frame_selection", False),
                "score_breakdown": candidate_decision.get("score_breakdown", {}),
            }
            event_meta["candidate_acceptance"]["final_trigger_text"] = trigger.get("text", "")
            event_meta["candidate_acceptance"]["final_trigger_span"] = trigger.get("span", [])
            event_meta["candidate_acceptance"]["final_trigger_char_span"] = trigger.get("char_span", [])
            event_meta["candidate_acceptance"]["final_trigger_category"] = trigger.get("category", "")
            event_meta["candidate_acceptance"]["final_trigger_preferred_span"] = final_trigger_preferred_span
            event_meta["candidate_acceptance"]["final_trigger_matches_predicate_center"] = (
                final_trigger_preferred_span == candidate_decision.get("predicate_center_span", [])
            )
        if isinstance(trigger_reanchoring, dict) and trigger_reanchoring.get("applied"):
            if isinstance(trigger_reanchoring.get("from_span"), list):
                from_char_span = self._token_span_to_char_span(tokens, offsets, trigger_reanchoring.get("from_span"))
                if from_char_span:
                    trigger_reanchoring["from_char_span"] = from_char_span
            if isinstance(trigger_reanchoring.get("to_span"), list):
                to_char_span = self._token_span_to_char_span(tokens, offsets, trigger_reanchoring.get("to_span"))
                if to_char_span:
                    trigger_reanchoring["to_char_span"] = to_char_span
            event_meta["trigger_reanchoring"] = trigger_reanchoring
        if isinstance(sentence_precision_summary, dict):
            event_meta["sentence_candidate_summary"] = {
                "candidates_retrieved": sentence_precision_summary.get("candidates_retrieved", 0),
                "candidates_rejected_by_trigger_filter": sentence_precision_summary.get("candidates_rejected_by_trigger_filter", 0),
                "candidates_rejected_as_copular_overgeneration": sentence_precision_summary.get("candidates_rejected_as_copular_overgeneration", 0),
                "candidates_collapsed_as_duplicates": sentence_precision_summary.get("candidates_collapsed_as_duplicates", 0),
                "candidates_grouped_by_clause_center": sentence_precision_summary.get("candidates_grouped_by_clause_center", 0),
                "candidates_dropped_as_same_center_duplicates": sentence_precision_summary.get("candidates_dropped_as_same_center_duplicates", 0),
                "candidates_downranked_for_mismatch": sentence_precision_summary.get("candidates_downranked_for_mismatch", 0),
                "candidates_downranked_as_helper_like": sentence_precision_summary.get("candidates_downranked_as_helper_like", 0),
                "candidates_kept_as_lexical_predicate_centers": sentence_precision_summary.get("candidates_kept_as_lexical_predicate_centers", 0),
                "clause_compression_applied_count": sentence_precision_summary.get("clause_compression_applied_count", 0),
                "final_candidates_considered": sentence_precision_summary.get("final_candidates_considered", 0),
                "final_candidates_emitted": sentence_precision_summary.get("final_candidates_emitted", 0),
                "accepted_candidate_family_distribution": sentence_precision_summary.get("accepted_candidate_family_distribution", {}),
                "rejection_reasons_distribution": sentence_precision_summary.get("rejection_reasons_distribution", {}),
            }
        if isinstance(clause_profile, dict):
            event_meta["clause_profile"] = {
                "clause_type": clause_profile.get("clause_type", ""),
                "predicate_centers": clause_profile.get("predicate_centers", []),
                "primary_center": clause_profile.get("primary_center"),
                "has_copula": clause_profile.get("has_copula", False),
                "center_groups": clause_profile.get("center_groups", []),
            }
            trigger_meta = event_meta.get("trigger_meta", {})
            if not isinstance(trigger_meta, dict):
                trigger_meta = {}
            hint_map = clause_profile.get("trigger_meta_hints", {})
            if isinstance(hint_map, dict):
                for key, value in hint_map.items():
                    if value not in ("", None, False, []):
                        trigger_meta[key] = value
            if isinstance(candidate_decision, dict):
                if candidate_decision.get("predicate_center_text"):
                    trigger_meta["predicate_center_text"] = candidate_decision.get("predicate_center_text", "")
                if candidate_decision.get("predicate_center_category"):
                    trigger_meta["predicate_center_category"] = candidate_decision.get("predicate_center_category", "")
            if trigger_meta:
                event_meta["trigger_meta"] = trigger_meta
        if warnings:
            existing = event_meta.get("warnings", [])
            if not isinstance(existing, list):
                existing = [str(existing)]
            existing.extend(warnings)
            event_meta["warnings"] = existing
        event_meta.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

        if "event_id" not in extraction:
            extraction["event_id"] = str(uuid.uuid4())[:12]

    def _extract_event(
        self,
        sentence: str,
        sentence_id: int,
        frame_id: str,
        ranked_candidates: List[dict],
        selected_frame: str,
        *,
        candidate_decision: Optional[dict] = None,
        sentence_precision_summary: Optional[dict] = None,
        clause_profile: Optional[dict] = None,
    ) -> Optional[dict]:
        grammar = self.get_grammar(frame_id)
        if not grammar:
            return None

        tokens, offsets = self.tokenizer.tokenize(sentence)
        prompt = self.make_prompt(sentence, frame_id)
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            grammar=grammar,
            stop=["<|user|>", "</s>", "<|assistant|>"],
        )
        raw_output = output["choices"][0]["text"]

        try:
            extraction = json.loads(raw_output)
        except json.JSONDecodeError:
            self._append_quarantine(
                {
                    "stage": "json_parse",
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "frame_id": frame_id,
                    "selected_frame": selected_frame,
                    "prompt": prompt,
                    "raw_output": raw_output,
                }
            )
            print(f"[WARN] JSON parse failure for sent={sentence_id}, frame={frame_id}", file=sys.stderr)
            return None

        warnings: List[str] = []
        self._enrich_marker_spans(extraction, tokens, warnings)
        trigger_reanchoring = self._reanchor_trigger_from_candidate(extraction, tokens, candidate_decision, warnings)
        is_valid, validation_errors = self.tokenizer.validate_extraction(sentence, extraction)
        if not is_valid:
            warnings.extend(validation_errors)
            self._append_quarantine(
                {
                    "stage": "schema_validation",
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "frame_id": frame_id,
                    "selected_frame": selected_frame,
                    "prompt": prompt,
                    "warnings": validation_errors,
                    "raw_output": raw_output,
                }
            )

        if is_valid:
            print(f"[OK] Sent {sentence_id}: {frame_id} extracted", file=sys.stderr)
        else:
            print(f"[WARN] Sent {sentence_id}: {frame_id} span mismatch -> {validation_errors}", file=sys.stderr)

        self._enrich_extraction(
            extraction=extraction,
            sentence=sentence,
            sentence_id=sentence_id,
            tokens=tokens,
            offsets=offsets,
            frame_id=frame_id,
            ranked_candidates=ranked_candidates,
            selected_frame=selected_frame,
            warnings=warnings,
            candidate_decision=candidate_decision,
            sentence_precision_summary=sentence_precision_summary,
            clause_profile=clause_profile,
            trigger_reanchoring=trigger_reanchoring,
        )
        return extraction

    def process_sentence(self, sent_id: int, text: str) -> List[dict]:
        text = text.strip()
        if not text:
            return []

        ranked = self.index.find_candidates(text, top_k=self.max_candidates)
        if not ranked:
            self._log_sentence_decision(
                sent_id=sent_id,
                sentence=text,
                selected_frame="",
                clause_profile={
                    "clause_type": "no_candidates",
                    "predicate_centers": [],
                    "primary_center": -1,
                    "has_copula": False,
                },
                decisions=[],
                summary={
                    "candidates_retrieved": 0,
                    "candidates_rejected_by_trigger_filter": 0,
                    "candidates_rejected_as_copular_overgeneration": 0,
                    "candidates_collapsed_as_duplicates": 0,
                    "final_candidates_considered": 0,
                    "final_candidates_emitted": 0,
                    "rejection_reasons_distribution": {},
                },
            )
            print(f"[INFO] Sent {sent_id}: no trigger candidates", file=sys.stderr)
            return []

        candidate_ids = [entry["frame_id"] for entry in ranked]
        selected = self.select_frame(text, candidate_ids) if self.use_two_step_selection else candidate_ids[0]
        accepted_candidates, decisions, precision_summary, clause_profile = self._precision_filter(text, sent_id, ranked, selected or "")
        if not accepted_candidates:
            print(
                f"[INFO] Sent {sent_id}: candidates={candidate_ids}, selected={selected}, clause_type={clause_profile.get('clause_type')}, extracted=0 after precision filter",
                file=sys.stderr,
            )
            return []

        events: List[dict] = []
        for candidate in accepted_candidates:
            frame_id = str(candidate.get("frame_id", ""))
            event = self._extract_event(
                text,
                sent_id,
                frame_id,
                ranked,
                selected or frame_id,
                candidate_decision=candidate,
                sentence_precision_summary=precision_summary,
                clause_profile=clause_profile,
            )
            if event:
                events.append(event)

        print(
            f"[INFO] Sent {sent_id}: candidates={candidate_ids}, selected={selected}, clause_type={clause_profile.get('clause_type')}, accepted={precision_summary.get('final_candidates_considered', 0)}, extracted={len(events)}",
            file=sys.stderr,
        )
        return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="Text file (one sentence per line)")
    ap.add_argument("--out", required=True, help="Output JSONL file")
    ap.add_argument("--registry", default="lexicons/uhvn_frames.json")
    ap.add_argument("--schemas", default="schemas/")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--llama-gpu-layers", type=int, default=-1, help="n_gpu_layers passed to llama_cpp.Llama; keep -1 to offload all layers that fit.")
    ap.add_argument("--llama-main-gpu", type=int, default=-1, help="Optional main_gpu passed to llama_cpp.Llama when using CUDA.")
    ap.add_argument("--llama-tensor-split", nargs="+", type=float, default=[], help="Optional tensor split ratios for explicit multi-GPU llama_cpp runs, for example 0.5 0.5.")
    ap.add_argument("--llama-split-mode", choices=["none", "layer", "row"], default="layer", help="Split mode used with --llama-tensor-split for multi-GPU llama_cpp runs.")
    ap.add_argument("--max-events", type=int, default=3, help="Maximum events emitted per sentence")
    ap.add_argument("--candidate-top-k", type=int, default=6, help="Top-k frames to score per sentence")
    ap.add_argument("--trigger-config", default=str((Path(__file__).resolve().parent.parent / "lexicons" / "trigger_plausibility.yaml").resolve()), help="YAML config for trigger plausibility scoring, clause typing, and duplicate collapse.")
    ap.add_argument("--candidate-decision-log", default="", help="Optional JSONL file capturing per-sentence candidate acceptance decisions.")
    ap.add_argument("--min-trigger-score", type=float, default=-1.0, help="Optional override for config min_trigger_plausibility_score.")
    ap.add_argument("--min-final-candidate-score", type=float, default=-1.0, help="Optional override for config min_final_candidate_score.")
    ap.add_argument("--disable-candidate-dedup", action="store_true", help="Disable duplicate / near-duplicate candidate collapse before final event emission.")
    ap.add_argument("--max-final-candidates-per-clause", type=int, default=0, help="Optional override for the post-retrieval per-clause acceptance cap.")
    ap.add_argument("--no-frame-selection", action="store_true", help="Skip two-step frame selection")
    ap.add_argument("--quarantine", default="", help="Optional JSONL file for malformed/invalid rows")
    ap.add_argument("--run-id", default="", help="Optional stable run id propagated to every emitted event in this extraction process.")

    args = ap.parse_args()

    pipeline = GoldPipeline(
        args.model,
        args.schemas,
        args.registry,
        ctx=args.ctx,
        max_events=args.max_events,
        max_candidates=args.candidate_top_k,
        use_two_step_selection=not args.no_frame_selection,
        quarantine_path=args.quarantine or None,
        llama_gpu_layers=args.llama_gpu_layers,
        llama_main_gpu=args.llama_main_gpu,
        llama_tensor_split=args.llama_tensor_split or None,
        llama_split_mode=args.llama_split_mode,
        trigger_config=args.trigger_config,
        candidate_decision_log=args.candidate_decision_log or None,
        min_trigger_score=args.min_trigger_score,
        min_final_candidate_score=args.min_final_candidate_score,
        enable_candidate_dedup=not args.disable_candidate_dedup,
        max_final_candidates_per_clause=args.max_final_candidates_per_clause,
        run_id=args.run_id or None,
    )

    print(f"[INFO] Starting Batch Processing on {args.input}...", file=sys.stderr)
    start_time = time.time()
    count = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            sentence = line.strip()
            if not sentence:
                continue

            if line_num % 10 == 0:
                print(".", end="", file=sys.stderr, flush=True)

            events = pipeline.process_sentence(line_num, sentence)
            if not events:
                continue

            for event in events:
                fout.write(json.dumps(event, ensure_ascii=False) + "\n")
                count += 1

    duration = time.time() - start_time
    print(f"\n[OK] Done! Extracted {count} event objects in {duration:.2f}s.", file=sys.stderr)


if __name__ == "__main__":
    main()
