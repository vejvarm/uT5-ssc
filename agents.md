# AGENTS (uT5-ssc)

Use this file to align contributors working on the `uT5-ssc` fine-tuning, evaluation, schema-serving, and analysis pipeline.

## Architecture Snapshot
- The main train/eval/predict entrypoint is `seq2seq/run_seq2seq.py`, which reads a JSON config, sets visible devices, resolves checkpoints, and dispatches Hugging Face training or inference.
- Experiment setup is config-driven under `configs/`, with directories grouped by research question, pretrained backbone, cleanliness condition, language, schema format, and train/eval/predict stage.
- Dataset argument definitions and schema-serialization knobs live in `seq2seq/utils/dataset.py` and `seq2seq/utils/args.py`.
- Dataset loading, preprocessing, and metric binding are centralized in `seq2seq/utils/dataset_loader.py`.
- Spider4SSC, Spider, CoSQL, and SM3 dataset adapters live under `seq2seq/datasets/`, and task metrics live under `seq2seq/metrics/`.
- Trainer customization and prediction post-processing live in `seq2seq/utils/trainer.py`, `seq2seq/utils/spider.py`, and `seq2seq/utils/cosql.py`.
- Graph/schema extraction and local KG-serving helpers live in `seq2seq/serve_neo4j_graphs.py`, `seq2seq/serve_rdf4j_graphs.py`, `seq2seq/extract_neo4j_schemas.py`, and `.scripts/init.sh`.
- Analysis, reporting, and result-comparison utilities live at the repository root and in `docs/` and `reports_summary/`.

## Repository Working Rules
- Prefer config-driven changes for checkpoints, datasets, schema serialization, decoding settings, and output directories; update the relevant JSON config before hardcoding new behavior in Python.
- Keep dataset names, metric pairings, and language-specific preprocessing aligned across `seq2seq/utils/dataset.py`, `seq2seq/utils/dataset_loader.py`, `seq2seq/datasets/`, and `seq2seq/metrics/`.
- Preserve the expected Spider4SSC and SM3 data layout under `data/` unless a migration is explicitly requested.
- Treat `results/`, `wandb/`, `logs/`, and large prediction artifacts as generated outputs unless the task explicitly requires tracked changes there.
- Keep KG setup assumptions intact for SPARQL and Cypher execution workflows; if graph-serving behavior changes, update `.scripts/init.sh` or the serving helpers in the same change.
- Add or update targeted tests under `seq2seq/utils/tests/` when changing reusable preprocessing, identifier mapping, or helper logic.
- Do not overwrite user analysis work at the repository root without confirming scope; ad hoc notebooks, plots, and summary scripts may be intentionally in progress.

## Standard Repository Flow
1. Identify whether the change affects config selection, dataset preprocessing, training/evaluation flow, schema serving, or post-hoc analysis.
2. Implement the smallest correct change in shared `seq2seq/` code first, then update configs or analysis scripts only where the public workflow actually changes.
3. Validate with a targeted unit test or CLI smoke check first, then run broader experiment-specific validation only if the change touches runtime behavior.
4. Record the work in local `documentation.md` with the newest entry at the top.

## Validation Guidance
- Targeted unit test: `python3 -m unittest seq2seq.utils.tests.test_cypher_identifier_mapping`
- CLI smoke check: `python3 -m seq2seq.run_seq2seq --help`
- Optional syntax smoke check: `python3 -m py_compile seq2seq/run_seq2seq.py seq2seq/utils/dataset.py seq2seq/utils/dataset_loader.py seq2seq/utils/trainer.py`
- Representative run pattern: `python3 -m seq2seq.run_seq2seq configs/5_ut5-ep19-dirty/sql_compact.json`

## Role Split (optional)
- Fine-Tuning Agent: training configs, checkpoint selection, `seq2seq/run_seq2seq.py`, and Trainer behavior.
- Dataset Agent: dataset adapters, schema serialization, preprocessing, and identifier mapping utilities.
- Metric Agent: exact-match, execution, and task-specific evaluation logic under `seq2seq/metrics/`.
- KG/Infra Agent: Neo4j or RDF4J serving, schema extraction, and environment bootstrap scripts.
- Analysis Agent: result summarization, plotting, mismatch analysis, and report generation.
