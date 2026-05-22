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
Refer to [.scripts/init.sh](.scripts/init.sh) for how to set up and initialize the Neo4j and RDF4j databases with Spider4SSC knowledge graphs.

## Reproducibility Guide
For a fuller reproduction checklist covering the conda environment, datastore services, fine-tuning/evaluation commands, and tokenizer audits, see [docs/reproducibility.md](docs/reproducibility.md).

## Training & Evaluation Cheatsheet
```bash
# SQL, SPARQL, or Cypher single-language runs
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sql_compact.json
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sparql_compact.json
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/cypher_compact.json

# Joint-language or ablation studies
python -m seq2seq.run_seq2seq configs/rq5_joint_clean/sql_norange.json

# Batch evaluation (produces preds_and_labels_*.json inside /work/results)
bash .scripts/eval_ssc_ut5_ep19-clean-2048.sh
```
Each config sets model checkpoints (uT5 base, EP19), schema serialization (`compact`, `norange`, `no-schema`), context length, and output directories so you can match the exact paper runs. Swap configs across the `5_*` and `6_*` folders to reproduce different training/eval conditions.

## Reproducing the C4 Spider4SSC runs

The C4-pretrained checkpoints are produced in the sibling repository
`~/git/balanced-plms`. The SSC configs committed here currently cover two C4
variants:

* `c4-clean`, which expects
  `/work/results/t5/ut5-ep19-c4-clean/final_mixed_grouped_dirty0p`
* `c4-dirty-10p`, which expects
  `/work/results/t5/ut5-ep19-c4-dirty-10p/final_mixed_grouped`

Before launching the fine-tuning jobs, make sure those pretraining runs have
completed in `balanced-plms` and that the Spider4SSC RDF4j and Neo4j services
are initialized as described in `.scripts/init.sh`.

### Run the full single-language C4 fine-tuning sweeps

Each wrapper below launches all 9 single-language runs for one C4 variant:

* SQL: `compact`, `norange`, `no-schema`
* SPARQL: `compact`, `norange`, `no-schema`
* Cypher: `compact`, `norange`, `no-schema`

```bash
cd ~/git/uT5-ssc
source ~/miniconda3/bin/activate ./.conda

bash .scripts/run_ssc_ut5_ep19-c4-clean.sh
bash .scripts/run_ssc_ut5_ep19-c4-dirty-10p.sh
```

The exact config folders are:

* `configs/5_ut5-ep19-c4-clean/`
* `configs/5_ut5-ep19-c4-dirty-10p/`

The outputs land under:

* `/work/results/ut5-base/Spider4SSC/c4-clean/ep19/...`
* `/work/results/ut5-base/Spider4SSC/c4-dirty-10p/ep19/...`

If you want to launch a single run instead of the whole sweep, call
`seq2seq.run_seq2seq` directly on an individual config, for example:

```bash
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-c4-clean/sql_compact.json
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-c4-dirty-10p/sql_compact.json
```

At the moment, this repository does not contain a committed
`c4-dirty-15p` SSC fine-tuning config set.

### Expanded pool6x C4 ladder (`0%`, `10%`, `30%`, `60%`)

The expanded C4 ladder is trained from a separate pooled dataset root in the
sibling `balanced-plms` repository:

* `/work/datasets/c4-en-noclean-match-owt-clean-dirtypool6x`

It expects these pretraining checkpoints:

* `/work/results/t5/ut5-ep19-c4-clean-pool6x/final_mixed_grouped_dirty0p`
* `/work/results/t5/ut5-ep19-c4-dirty-10p-pool6x/final_mixed_grouped_dirty10p`
* `/work/results/t5/ut5-ep19-c4-dirty-30p-pool6x/final_mixed_grouped_dirty30p`
* `/work/results/t5/ut5-ep19-c4-dirty-60p-pool6x/final_mixed_grouped_dirty60p`

Launch the full 9-run SSC sweeps with:

```bash
cd ~/git/uT5-ssc
source ~/miniconda3/bin/activate ./.conda

bash .scripts/run_ssc_ut5_ep19-c4-clean-pool6x.sh
bash .scripts/run_ssc_ut5_ep19-c4-dirty-10p-pool6x.sh
bash .scripts/run_ssc_ut5_ep19-c4-dirty-30p-pool6x.sh
bash .scripts/run_ssc_ut5_ep19-c4-dirty-60p-pool6x.sh
```

The corresponding config folders are:

* `configs/5_ut5-ep19-c4-clean-pool6x/`
* `configs/5_ut5-ep19-c4-dirty-10p-pool6x/`
* `configs/5_ut5-ep19-c4-dirty-30p-pool6x/`
* `configs/5_ut5-ep19-c4-dirty-60p-pool6x/`

The outputs land under:

* `/work/results/ut5-base/Spider4SSC/c4-clean-pool6x/ep19/...`
* `/work/results/ut5-base/Spider4SSC/c4-dirty-10p-pool6x/ep19/...`
* `/work/results/ut5-base/Spider4SSC/c4-dirty-30p-pool6x/ep19/...`
* `/work/results/ut5-base/Spider4SSC/c4-dirty-60p-pool6x/ep19/...`

If you only want one run, call `seq2seq.run_seq2seq` on an individual config,
for example:

```bash
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-c4-dirty-30p-pool6x/sql_compact.json
```

The legacy `c4-clean` and `c4-dirty-10p` SSC configs remain available
separately and continue to point at the original non-pool6x pretraining runs.

## Analysis Artifacts
- Token complexity (RQ4): `RQ4_token_count_analysis_new.ipynb`, driven by `seq2seq/count_tokens.py`.
- Cross-language comparisons (RQ1–RQ3): see `RQ1andRQ2_compare_runs-bias.py`, `RQ4_compare_langs_and_schemas.py`, and `RQ4andRQ5_gather_eval_results.ipynb`.
All notebooks assume the prediction JSON files emitted into `/work/results/ut5-base/Spider4SSC/...` by the configs above.

## Target-Query Tokenizer Audit
The target-query tokenizer round-trip audit is implemented in `seq2seq/analyze_target_tokenizer_roundtrip.py`. It uses the same T5 query extra tokens and generated-prediction decode cleanup as `seq2seq/run_seq2seq.py` and `seq2seq/utils/spider.py`.

```bash
conda run -p ./.conda env \
  TOKENIZER_NAME_OR_PATH=/home/vejvar-martin-nj/git/balanced-plms/results/t5/unbiased-openwebtext-10k/clean \
  python seq2seq/analyze_target_tokenizer_roundtrip.py \
    --split dev \
    --output-dir results/tokenizer_roundtrip/dev
```

The corrected dev audit reports execution-equivalent round trips for all 608 gold queries in each language: SQL, SPARQL, and Cypher. SPARQL remains much more token-fragmented than SQL, but deterministic tokenizer round trips do not corrupt gold-query semantics. See [docs/reproducibility.md](docs/reproducibility.md#target-query-tokenizer-round-trip-audit) for outputs, interpretation, and validation commands.

For questions or replication clarifications, cite the paper and reference this repository in supplementary materials.
