#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milestone 3.1: Span Indexing & Alignment Logic (Fixed for Hindi)
Implements Option A: Half-open intervals [start, end)

Fixes: Uses explicit whitespace/punctuation splitting instead of regex \w matching.
"""

import re
from typing import List, Tuple, Optional

class Tokenizer:
    def __init__(self):
        # Split on whitespace OR punctuation.
        # Captures the delimiter so we can track offsets correctly.
        # Includes Hindi Danda (।) and standard punctuation.
        self._split_pattern = re.compile(r'(\s+|[.,;!?।"\'|()\[\]{}<>\u0964\u0965])')

    def tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Returns:
            tokens: List of strings (punctuation isolated, whitespace removed)
            offsets: List of (char_start, char_end) tuples for each token
        """
        tokens = []
        offsets = []
        
        current_pos = 0
        # re.split with capturing group returns [text, delimiter, text, delimiter...]
        parts = self._split_pattern.split(text)
        
        for part in parts:
            if not part:
                continue
                
            # If part is purely whitespace, skip as token but advance position
            if part.strip() == "":
                current_pos += len(part)
                continue
                
            # It's a valid token (word or punctuation)
            start = current_pos
            end = current_pos + len(part)
            tokens.append(part)
            offsets.append((start, end))
            
            current_pos += len(part)
            
        return tokens, offsets

    def resolve_span(self, span: List[int], tokens: List[str]) -> Optional[str]:
        """
        Reconstructs text from token indices [start, end).
        """
        if not span or len(span) != 2:
            return None
            
        start_idx, end_idx = span
        
        # Boundary checks
        if start_idx < 0 or end_idx > len(tokens) or start_idx >= end_idx:
            return None
            
        # Join with space (Canonical reconstruction)
        return " ".join(tokens[start_idx:end_idx])

    def validate_extraction(self, text: str, extraction: dict) -> Tuple[bool, List[str]]:
        """
        Validates that 'text' fields match the 'span' indices.
        """
        tokens, offsets = self.tokenize(text)
        errors = []
        
        # 1. Check Trigger
        trig = extraction.get("trigger", {})
        if not self._check_span(trig, tokens, "Trigger", errors):
            pass

        # 2. Check Arguments
        args = extraction.get("arguments", {})
        for role, arg_data in args.items():
            self._check_span(arg_data, tokens, f"Arg:{role}", errors)
            
        return len(errors) == 0, errors

    def _check_span(self, item: dict, tokens: List[str], label: str, errors: List[str]) -> bool:
        """Helper to validate a single text/span pair."""
        if not item:
            return True 
            
        claimed_text = item.get("text")
        span = item.get("span")
        
        if span is None:
            if claimed_text:
                errors.append(f"[{label}] Missing span for text '{claimed_text}'")
                return False
            return True

        # Resolve actual text from indices
        actual_text = self.resolve_span(span, tokens)
        
        if actual_text is None:
            errors.append(f"[{label}] Invalid span indices {span} (max {len(tokens)})")
            return False
            
        # Fuzzy match: We ignore slight whitespace differences
        if claimed_text.strip() != actual_text.strip():
            errors.append(f"[{label}] Mismatch. Span {span} -> '{actual_text}', but claimed '{claimed_text}'")
            return False
            
        return True

# ---------------------------
# Test Block
# ---------------------------
if __name__ == "__main__":
    t = Tokenizer()
    sent = "राम ने रावण को मारा।"
    tokens, offsets = t.tokenize(sent)
    
    print(f"Sentence: {sent}")
    print(f"Tokens:   {tokens}")
    print(f"Indices:  {[i for i in range(len(tokens))]}")
    print("-" * 30)
    
    # Valid Extraction (Option A: [start, end))
    # 'राम' is index 0. End is 1. -> [0, 1]
    valid_ex = {
        "trigger": {"text": "मारा", "span": [4, 5]}, 
        "arguments": {
            "Agent": {"text": "राम", "span": [0, 1]},
            "Patient": {"text": "रावण", "span": [2, 3]}
        }
    }
    
    ok, errs = t.validate_extraction(sent, valid_ex)
    print(f"Valid Extraction Test: {'PASSED' if ok else 'FAILED'}")
    if not ok: print(errs)

    print("-" * 30)

    # Bad Extraction
    bad_ex = {
        "trigger": {"text": "मारा", "span": [4, 5]},
        "arguments": {
            "Agent": {"text": "लक्ष्मण", "span": [0, 1]} # Text says Lakshman, Span points to Ram
        }
    }
    
    ok, errs = t.validate_extraction(sent, bad_ex)
    print(f"Bad Extraction Test:   {'PASSED' if ok else 'FAILED'}")
    if not ok: print(errs)