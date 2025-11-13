# uT5-SSC Supplementary Repo
Code, configs, and analysis assets for paper *Are We Too Focused on a Single Query Language? Investigating Text-to-SQL/SPARQL/Cypher Task Complexity via Fine-tuning Unbiased T5 Models*. 

Everything is organized so you can recreate the SQL/SPARQL/Cypher fine-tuning runs, evaluation sweeps, and downstream analyses reported in the manuscript.

## Repository Highlights
- `seq2seq/`: training loop (`run_seq2seq.py`), dataset wrappers for Spider4SSC, schema/identifier utilities, and KG serving helpers.
- `configs/`: JSON configs grouped by research question (RQ1–RQ5), language, schema serialization strategy, and cleanliness settings (clean/injected/dirty).
- `.scripts/` and `results/`: lightweight bash helpers and saved trainer states, logits, and prediction dumps used by the paper.
- `RQ*.ipynb` and `RQ*.py`: figure/table generation notebooks plus token/complexity studies.

## Environment & Data
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Running any training config triggers an automatic download of the Spider4SSC bundle (SQL/SPARQL/Cypher splits plus RDF/Neo4j DBs) via the Hugging Face datasets loader under `seq2seq/datasets/ssc`. Set `HF_DATASETS_CACHE` if you want to reuse the archive across experiments.

## Serving the Knowledge Graphs
Refer to [.scripts/init.sh](.scripts/init.sh) for how to set up and initilalize the Neo4j and RDF4j databases with Spider4SSC knowledge graphs.

## Training & Evaluation Cheatsheet
```bash
# SQL, SPARQL, or Cypher single-language runs
python seq2seq/run_seq2seq.py configs/5_ut5-ep19-clean/sql_compact.json
python seq2seq/run_seq2seq.py configs/5_ut5-ep19-clean/sparql_compact.json
python seq2seq/run_seq2seq.py configs/5_ut5-ep19-clean/cypher_compact.json

# Joint-language or ablation studies
python seq2seq/run_seq2seq.py configs/rq5_joint_clean/sql_norange.json

# Batch evaluation (produces preds_and_labels_*.json inside /work/results)
bash .scripts/eval_ssc_ut5_ep19-clean-2048.sh
```
Each config sets model checkpoints (uT5 base, EP19), schema serialization (`compact`, `norange`, `no-schema`), context length, and output directories so you can match the exact paper runs. Swap configs across the `5_*` and `6_*` folders to reproduce different training/eval conditions.

## Analysis Artifacts
- Token complexity (RQ4): `RQ4_token_count_analysis_new.ipynb`, driven by `seq2seq/count_tokens.py`.
- Cross-language comparisons (RQ1–RQ3): see `RQ1andRQ2_compare_runs-bias.py`, `RQ4_compare_langs_and_schemas.py`, and `RQ4andRQ5_gather_eval_results.ipynb`.
All notebooks assume the prediction JSON files emitted into `/work/results/ut5-base/Spider4SSC/...` by the configs above.

For questions or replication clarifications, cite the paper and reference this repository in supplementary materials.
