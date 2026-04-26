# Evaluation Scripts

This folder contains lightweight CLI scripts for MTP evaluation.

## 1) Prepare evaluation records

Flatten wrapper outputs (`{"events":[...]}` or plain event objects) into one event per line:

```bash
python eval/prepare_dataset.py \
  --input output/real_kg_from_env.jsonl \
  --output outputs/metrics/real_kg_from_env_flat.jsonl \
  --write-splits \
  --split 0.8 0.1 0.1
```

## 2) Run baseline (MTP comparison)

```bash
python eval/run_baselines.py \
  --method trigger \
  --input data/real_test.txt \
  --out outputs/baselines/trigger_baseline.jsonl \
  --registry lexicons/uhvn_frames.json \
  --max-events 2 \
  --emit-wrapper
```

`--method indie` and `--method mrebel` are scaffolded. In this environment they write a warning payload with installation guidance.

## 3) Compute metrics

```bash
python eval/compute_metrics.py \
  --gold output/real_kg.jsonl \
  --pred output/real_kg_from_env.jsonl \
  --schemas schemas \
  --sameas output/sameas.nq \
  --pred-nq output/real_kg_from_env.nq \
  --match-mode strict \
  --out outputs/metrics/real_eval.json
```

Use `--match-mode lenient` when gold/prediction spans are in different coordinate systems
(e.g., gold uses token spans while model output uses character spans).

## 4) Render report

```bash
python eval/report.py \
  --metrics outputs/metrics/real_eval.json \
  --out outputs/metrics/real_eval_report.md
```

## Outputs

The metrics JSON contains:
- event matching precision/recall/F1
- argument precision/recall/F1 (text + span variants)
- ontology conformance flags
- hallucination rate
- EL@1 estimate from mention links

## 5) Build annotation-ready review packs

The pack builders now emit explicit annotation schemas and CSV columns for precision-oriented review.

```bash
python eval/build_argument_validation_review_pack.py \
  --refinement-json output/run/refinement/refinement_cache.json \
  --out output/run/review/argument_review_pack.json \
  --csv-out output/run/review/argument_review_pack.csv

python eval/build_coref_review_pack.py \
  --coref-json output/run/coref/entity_cluster_cache.json \
  --out output/run/review/coref_review_pack.json \
  --csv-out output/run/review/coref_review_pack.csv

python eval/build_temporal_review_pack.py \
  --time-json output/run/time/time_cache.json \
  --out output/run/review/temporal_review_pack.json \
  --csv-out output/run/review/temporal_review_pack.csv

python eval/build_event_density_review_pack.py \
  --event-density-json output/run/stats/event_density.analysis.json \
  --out output/run/review/event_density_review_pack.json \
  --csv-out output/run/review/event_density_review_pack.csv
```

## 6) Build a stratified extraction review sample

Use this to add a less challenge-biased evaluation sample from the full extraction output.

```bash
python eval/build_stratified_extraction_review_sample.py \
  --input-jsonl output/run/extraction/canonical_extraction.jsonl \
  --sentences data/nhkg_mtpsem_dataset/nhkg_input_1000.txt \
  --ner-json output/run/ner/ner_cache.json \
  --time-json output/run/time/time_cache.json \
  --event-density-json output/run/stats/event_density.analysis.json \
  --out output/run/review/extraction_sample.json \
  --csv-out output/run/review/extraction_sample.csv
```

Default bucket sizes are:
- 40 random sentences
- 20 with NER mentions
- 20 with timexes
- 20 high-density sentences

## 7) Compile an evaluation summary

Once review packs have been annotated, combine them with the automatic stats into a single report.

```bash
python eval/summarize_run_evaluation.py \
  --validation-json output/run/stats/canonical_full_graph.validation.json \
  --event-audit-json output/run/stats/event_count_audit.json \
  --stats-json output/run/stats/compute_enrichment_stats.json \
  --event-density-json output/run/stats/event_density.analysis.json \
  --argument-review-json output/run/review/argument_review_pack.json \
  --coref-review-json output/run/review/coref_review_pack.json \
  --temporal-review-json output/run/review/temporal_review_pack.json \
  --event-density-review-json output/run/review/event_density_review_pack.json \
  --extraction-sample-json output/run/review/extraction_sample.json \
  --out output/run/stats/evaluation_summary.json \
  --markdown-out output/run/stats/evaluation_summary.md
```

This summary combines:
- automatic acceptance checks
- human-reviewed precision metrics
- oversplitting indicators
- a compact thesis-ready markdown table
