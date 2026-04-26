# NHKG: Hindi Event Knowledge Graph Construction

NHKG is a hybrid pipeline for converting raw Hindi text into a validated, event-centric RDF knowledge graph. The system combines frame-guided candidate retrieval, schema-constrained LLM extraction, deterministic precision filtering, linguistic enrichment, RDF/N-Quads materialization, and graph validation.

The final output is a Hindi event knowledge graph containing event nodes, triggers, participant roles, linguistic enrichment, provenance, and validation evidence.

## Repository Contents

| Path | Purpose |
|---|---|
| `gold/` | Core event extraction pipeline, frame retrieval, schema-constrained extraction, trigger plausibility, clause typing, candidate filtering, and RDF conversion. |
| `align/` | Linguistic enrichment stages: POS/syntax, NER, coreference/entity clustering, temporal enrichment, WordNet enrichment, and RDF alignment utilities. |
| `fusion/` | DBpedia/indexing and fact-fusion utilities used by optional enrichment stages. |
| `eval/` | Graph validation, count reconciliation, event-density analysis, and thesis-table generation scripts. |
| `schemas/` | Frame-specific JSON schemas for constrained extraction plus SHACL validation shapes. |
| `lexicons/` | UHVN/PropBank-style frame registry, role mappings, trigger plausibility config, and enrichment mappings. |
| `data/nhkg_mtpsem_dataset/` | Dataset used for building/evaluating the knowledge graph. |
| `presentation/` | Thesis defense PPTX and PDF. |
| `demo/` | Graph-grounded Hindi Q&A/demo scripts. |

## Repository Scope

This repository is organized as a clean release of the main NHKG pipeline. It focuses on the code, schemas, lexicons, dataset, aggregate results, and defense materials needed to understand and reproduce the system-level workflow.

## Dataset

The KG-building dataset is included at:

```text
data/nhkg_mtpsem_dataset/
```

Important files:

- `nhkg_input_5000.txt`: final 5,000-sentence input used for the large run
- `nhkg_input_1000.txt`: smaller input split
- `nhkg_input_5000.meta.json`: metadata for the 5,000-sentence input
- `manifest.json`: dataset manifest
- Hindi WordNet/IWN gloss/example files used for lexical-semantic resources

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The extraction pipeline expects a local GGUF LLM model path when running `gold/pipeline.py`. Model weights are not included in this repository.

## Main Pipeline

At a high level:

```text
Hindi text
-> frame retrieval and candidate proposal
-> schema-constrained LLM extraction
-> trigger/clause-aware candidate filtering
-> linguistic enrichment
-> RDF/N-Quads graph construction
-> custom checks + RDF parse validation + SHACL validation
```

Core scripts:

- `gold/pipeline.py`: main extraction pipeline
- `gold/to_rdf.py`: JSONL event output to RDF/N-Quads
- `align/annotate_hindi_tokens.py`: token/syntax annotation
- `align/run_hindi_ner.py`: NER enrichment
- `align/temporal_enrich.py`: temporal extraction/enrichment
- `align/wordnet_enrich.py`: WordNet enrichment
- `eval/validate_graph.py`: custom structural checks, RDF parse validation, and SHACL validation
- `eval/audit_event_count_consistency.py`: extraction-vs-graph count reconciliation
- `eval/analyze_event_density.py`: event density analysis

## Results

Final large-run summary:

| Metric | Value |
|---|---:|
| Input sentences | 5,000 |
| Event-bearing sentences | 2,568 (51.36%) |
| Extracted events | 2,705 |
| Events per input sentence | 0.5410 |
| Count mismatch | 0 |
| RDF parse validation | Passed |
| SHACL validation | Conforms |
| Overall validation status | True |

Quality/diagnostic snapshot:

| Metric | Value |
|---|---:|
| Trigger/predicate-center mismatch | 16 (0.59%) |
| Implausible trigger heads | 6 (0.22%) |
| Generic-family share | 923 (34.12%) |
| Multi-event sentence warning | 270 (9.98%) |
| Hard review rate | 3.77% |
| Caution rate | 22.00% |
| Info-only rate | 74.23% |

Semantic evaluation summary:

| Task A: emitted-event quality, N=300 | Value |
|---|---:|
| Event real rate | 263 (87.67%) |
| Trigger correct rate | 150 (50.00%) |
| Frame plausible rate | 125 (41.67%) |
| Arguments usable rate | 83 (27.67%) |
| Duplicate/alternate rate | 80 (26.67%) |
| Overall usable event rate | 71 (23.67%) |

| Task B: recover-now no-event audit, N=50 | Value |
|---|---:|
| Should have had an event | 33 (66.00%) |
| True no-event | 15 (30.00%) |
| Uncertain | 2 (4.00%) |

Most common missed-event families:

| Family | Count |
|---|---:|
| change_of_state | 13 |
| modal_lexical | 12 |
| light_verb_compound | 8 |
| imperative | 6 |
| state | 5 |
| action | 4 |
| other | 1 |
| classification | 1 |

Interpretation: NHKG is structurally stable and validation-ready at scale. The remaining bottleneck is semantic survivor quality, especially trigger specificity, frame choice, and argument usability.

## Validation Layers

NHKG uses three validation layers:

1. **Custom structural checks**: NHKG-specific consistency checks such as event trigger presence, token offset validity, temporal relation completeness, role target sanity, and DBpedia/NER type compatibility.
2. **RDF parse validation**: verifies that the produced N-Quads file is syntactically valid RDF and can be loaded by `rdflib`.
3. **SHACL validation**: checks formal shape constraints for event nodes, token nodes, entity mentions, canonical entities, temporal nodes, and provenance/reference links.

## Notes for Reviewers

The central contribution is a reproducible Hindi event-to-knowledge-graph pipeline, not a claim of perfect semantic extraction. Structural validation is strong, while semantic evaluation identifies the remaining bottlenecks in trigger specificity, frame choice, and argument usability.
