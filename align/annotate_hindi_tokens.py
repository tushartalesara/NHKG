#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Hindi POS and optional syntax caches for post-extraction enrichment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    from .enrichment_common import collect_sentence_keys, load_sentence_text_map, sentence_key
    from .internal_quality_common import DEFAULT_STAGE_VERSION, build_stage_metadata
except ImportError:  # pragma: no cover
    from enrichment_common import collect_sentence_keys, load_sentence_text_map, sentence_key
    from internal_quality_common import DEFAULT_STAGE_VERSION, build_stage_metadata

try:
    import stanza
except ImportError:  # pragma: no cover
    stanza = None

try:
    from gold.span import Tokenizer as SpanTokenizer
except Exception:  # pragma: no cover
    SpanTokenizer = None


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class PosToken:
    token_id: int
    text: str
    start: int
    end: int
    upos: str
    xpos: str = ""
    feats: str = ""
    ufeats: str = ""
    lemma: str = ""
    head: Optional[int] = None
    deprel: str = ""
    sent_index: int = 0
    word_index: int = 0
    stanza_word_id: Optional[int] = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a UTF-8 Hindi POS cache with optional dependency parses.")
    parser.add_argument("--input", default="", help="Extraction JSONL/JSON used to recover sentence keys")
    parser.add_argument("--sentences", default="", help="One sentence per line text file")
    parser.add_argument("--out", default="align/pos_cache.json", help="Output JSON cache consumed by RDF/WordNet enrichment")
    parser.add_argument("--syntax-out", default="", help="Optional sidecar JSON containing only syntax-oriented fields")
    parser.add_argument("--engine", choices=["auto", "stanza", "fallback"], default="auto", help="Annotation backend to use")
    parser.add_argument("--depparse", dest="depparse", action="store_true", help="Run dependency parsing in addition to POS and lemma")
    parser.add_argument("--no-depparse", dest="depparse", action="store_false", help="Disable dependency parsing and emit POS/lemma only")
    parser.set_defaults(depparse=False)
    parser.add_argument("--stanza-dir", default="", help="Optional Stanza resources directory; auto-detected when omitted")
    parser.add_argument("--download-model", action="store_true", help="Download the Hindi Stanza model if it is missing")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt automatic Stanza downloads")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--device", default="", help="Preferred device, for example cpu or cuda")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of sentence-level diagnostic samples to print")
    return parser


def resolve_stanza_dir(cli_value: str) -> Optional[Path]:
    candidates: List[Path] = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())
    env_dir = os.environ.get("STANZA_RESOURCES_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend([Path.cwd() / "stanza_resources", Path.home() / "stanza_resources", Path.home() / "AppData" / "Local" / "StanfordNLP" / "stanza_resources"])
    for candidate in candidates:
        if (candidate / "resources.json").exists():
            return candidate
    return None


def build_stanza_pipeline(*, depparse: bool, download_model: bool, no_download: bool, stanza_dir: Optional[Path], device: str, cpu: bool):
    if stanza is None:
        return None
    processors = "tokenize,pos,lemma,depparse" if depparse else "tokenize,pos,lemma"
    pipeline_args = {
        "processors": processors,
        "tokenize_no_ssplit": True,
        "verbose": False,
        "use_gpu": False if cpu or device.lower() == "cpu" else device.lower() == "cuda",
    }
    if stanza_dir is not None:
        pipeline_args["dir"] = str(stanza_dir)
    if no_download:
        pipeline_args["download_method"] = None
    if download_model:
        download_args = {"package": "hdtb"}
        if stanza_dir is not None:
            download_args["dir"] = str(stanza_dir)
        stanza.download("hi", **download_args)
    try:
        return stanza.Pipeline("hi", **pipeline_args)
    except Exception:
        if no_download:
            raise
        if not download_model:
            download_args = {"package": "hdtb"}
            if stanza_dir is not None:
                download_args["dir"] = str(stanza_dir)
            stanza.download("hi", **download_args)
            return stanza.Pipeline("hi", **pipeline_args)
        raise


def fallback_tokens(sentence: str) -> List[Tuple[str, Tuple[int, int]]]:
    if SpanTokenizer is not None:
        tokenizer = SpanTokenizer()
        tokens, offsets = tokenizer.tokenize(sentence)
        return list(zip(tokens, offsets))
    items: List[Tuple[str, Tuple[int, int]]] = []
    cursor = 0
    for raw in sentence.split():
        start = sentence.find(raw, cursor)
        if start < 0:
            start = cursor
        end = start + len(raw)
        items.append((raw, (start, end)))
        cursor = end
    return items


def fallback_tokenize(sentence: str) -> List[PosToken]:
    out: List[PosToken] = []
    for idx, (token, (start, end)) in enumerate(fallback_tokens(sentence)):
        out.append(PosToken(token_id=idx, text=token, start=int(start), end=int(end), upos="", xpos="", feats="", ufeats="", lemma=token, head=None, deprel="", sent_index=0, word_index=idx + 1, stanza_word_id=None))
    return out


def _int_or_none(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def tokenize_with_stanza(pipeline, sentence: str) -> List[PosToken]:
    doc = pipeline(sentence)
    out: List[PosToken] = []
    token_counter = 0
    for sent_index, sent in enumerate(getattr(doc, "sentences", []) or []):
        for word_index, word in enumerate(getattr(sent, "words", []) or [], start=1):
            text = getattr(word, "text", "") or ""
            start = _int_or_none(getattr(word, "start_char", None))
            end = _int_or_none(getattr(word, "end_char", None))
            if start is None or end is None:
                continue
            out.append(PosToken(token_id=token_counter, text=text, start=start, end=end, upos=(getattr(word, "upos", "") or "").strip(), xpos=(getattr(word, "xpos", "") or "").strip(), feats=(getattr(word, "feats", "") or "").strip(), ufeats=(getattr(word, "feats", "") or "").strip(), lemma=(getattr(word, "lemma", "") or text).strip(), head=_int_or_none(getattr(word, "head", None)), deprel=(getattr(word, "deprel", "") or "").strip(), sent_index=sent_index, word_index=word_index, stanza_word_id=_int_or_none(getattr(word, "id", None))))
            token_counter += 1
    if not out:
        return fallback_tokenize(sentence)
    return out


def load_sentence_entries(input_path: Optional[Path], sentence_path: Optional[Path]) -> List[Tuple[str, str]]:
    if sentence_path is None or not sentence_path.exists():
        raise SystemExit("Sentence text input is required. Pass --sentences with one sentence per line.")
    sentence_keys = collect_sentence_keys(input_path) if input_path else []
    sentence_map = load_sentence_text_map(sentence_path, sentence_keys=sentence_keys)
    if not sentence_map:
        raise SystemExit("No sentence text lines found in the supplied --sentences file.")
    if sentence_keys:
        ordered = [(key, sentence_map[key]) for key in sentence_keys if key in sentence_map]
        if ordered:
            return ordered
    return list(sentence_map.items())


def build_syntax_payload(payload: Dict[str, object], *, meta: dict) -> Dict[str, object]:
    syntax_payload: Dict[str, object] = {"meta": dict(meta), "sentences": {}}
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        syntax_payload["sentences"].setdefault(doc_id, {})
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            tokens = item.get("tokens", [])
            slim_tokens = []
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                slim_tokens.append({"token_id": token.get("token_id"), "text": token.get("text", ""), "start": token.get("start"), "end": token.get("end"), "head": token.get("head"), "deprel": token.get("deprel", ""), "sent_index": token.get("sent_index", 0), "word_index": token.get("word_index", 0), "stanza_word_id": token.get("stanza_word_id")})
            syntax_payload["sentences"][doc_id][sent_id] = {"text": item.get("text", ""), "tokens": slim_tokens}
    return syntax_payload


def print_diagnostics(*, payload: Dict[str, object], used_engine: str, depparse: bool, debug_samples: int) -> None:
    token_total = 0
    upos_tokens = 0
    lemma_tokens = 0
    syntax_tokens = 0
    sample_lines: List[str] = []
    sentence_total = 0
    for doc_id, sent_map in (payload.get("sentences", {}) or {}).items():
        if not isinstance(sent_map, dict):
            continue
        for sent_id, item in sent_map.items():
            if not isinstance(item, dict):
                continue
            sentence_total += 1
            tokens = item.get("tokens", [])
            if not isinstance(tokens, list):
                continue
            token_total += len(tokens)
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                if str(token.get("upos", "")).strip():
                    upos_tokens += 1
                if str(token.get("lemma", "")).strip():
                    lemma_tokens += 1
                head = token.get("head")
                deprel = str(token.get("deprel", "")).strip()
                if isinstance(head, int) and head >= 0 and deprel:
                    syntax_tokens += 1
            if len(sample_lines) < max(0, debug_samples):
                preview = []
                for token in tokens[: min(8, len(tokens))]:
                    if not isinstance(token, dict):
                        continue
                    chunk = f"{token.get('text', '')}/{token.get('upos', '')}:{token.get('lemma', '')}"
                    if depparse:
                        chunk += f"->{token.get('head', '')}:{token.get('deprel', '')}"
                    preview.append(chunk)
                sample_lines.append(f"{sentence_key(doc_id, sent_id)} :: {' | '.join(preview)}")
    print(f"[OK] engine={used_engine} sentences={sentence_total} tokens={token_total}")
    if token_total:
        print(f"[OK] tokens_with_upos={upos_tokens} ({upos_tokens / token_total:.2%})")
        print(f"[OK] tokens_with_lemma={lemma_tokens} ({lemma_tokens / token_total:.2%})")
        if depparse:
            print(f"[OK] tokens_with_head_deprel={syntax_tokens} ({syntax_tokens / token_total:.2%})")
    for sample in sample_lines:
        print(f"[DBG] {sample}")


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input) if args.input else None
    if input_path is not None and not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    sentence_path = Path(args.sentences) if args.sentences else None
    entries = load_sentence_entries(input_path, sentence_path)

    requires_stanza = args.engine in {"auto", "stanza"} or args.depparse
    if args.depparse and args.engine == "fallback":
        raise SystemExit("Dependency parsing requires Stanza. Re-run with --engine stanza or --engine auto.")

    tokenizer: Callable[[str], List[PosToken]]
    used_engine = "fallback"
    warnings: List[str] = []

    if requires_stanza:
        if stanza is None:
            if args.depparse or args.engine == "stanza":
                raise SystemExit("Stanza is not installed. Install it or rerun with --engine fallback without --depparse.")
            tokenizer = fallback_tokenize
            warnings.append("Stanza unavailable; emitted fallback tokenization only.")
        else:
            try:
                stanza_dir = resolve_stanza_dir(args.stanza_dir)
                pipeline = build_stanza_pipeline(depparse=args.depparse, download_model=args.download_model, no_download=args.no_download, stanza_dir=stanza_dir, device=args.device or ("cpu" if args.cpu else ""), cpu=args.cpu)
                tokenizer = lambda sentence: tokenize_with_stanza(pipeline, sentence)
                used_engine = "stanza"
            except Exception as exc:
                if args.depparse or args.engine == "stanza":
                    raise SystemExit("Stanza Hindi resources are unavailable. Pass --download-model, --stanza-dir, or rerun without --depparse in fallback mode.") from exc
                print(f"[WARN] Falling back to simple tokenization because Stanza failed: {exc}", file=sys.stderr)
                tokenizer = fallback_tokenize
                warnings.append(f"Fell back to simple tokenization because Stanza failed: {exc}")
    else:
        tokenizer = fallback_tokenize

    payload: Dict[str, object] = {"meta": {}, "sentences": {}}
    for key, sentence in entries:
        doc_id, sent_id = key.split("::", 1) if "::" in key else (key, "0")
        payload["sentences"].setdefault(doc_id, {})
        payload["sentences"][doc_id][sent_id] = {"text": sentence, "tokens": [asdict(token) for token in tokenizer(sentence)]}

    effective_depparse = bool(args.depparse and used_engine == "stanza")
    payload["meta"] = build_stage_metadata(
        stage_name="annotate_hindi_tokens",
        stage_version=DEFAULT_STAGE_VERSION,
        engine=used_engine,
        source_paths={"input": input_path, "sentences": sentence_path},
        input_counts={"sentences": len(entries)},
        warnings=warnings,
        extra={"depparse": effective_depparse, "stanza_dir": str(resolve_stanza_dir(args.stanza_dir) or ""), "requested_engine": args.engine},
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if args.syntax_out:
        syntax_payload = build_syntax_payload(payload, meta=build_stage_metadata(stage_name="annotate_hindi_tokens.syntax", stage_version=DEFAULT_STAGE_VERSION, engine=used_engine, source_paths={"input": input_path, "sentences": sentence_path}, input_counts={"sentences": len(entries)}, warnings=warnings, extra={"depparse": effective_depparse, "requested_engine": args.engine}))
        syntax_path = Path(args.syntax_out)
        syntax_path.parent.mkdir(parents=True, exist_ok=True)
        with syntax_path.open("w", encoding="utf-8") as handle:
            json.dump(syntax_payload, handle, ensure_ascii=False, indent=2)
        print(f"[OK] wrote_syntax_cache={syntax_path}")

    print(f"[OK] wrote_pos_cache={out_path}")
    print_diagnostics(payload=payload, used_engine=used_engine, depparse=effective_depparse, debug_samples=max(0, args.debug_samples))


if __name__ == "__main__":
    main()
