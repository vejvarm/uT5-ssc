# Reproducibility Guide

This guide documents the commands needed to reproduce the Spider4SSC fine-tuning,
evaluation, and target-query tokenizer audit workflows used by this repository.
It complements the shorter quick-start notes in `README.md`.

## Environment

The local experiments use the repository-local conda environment:

```bash
cd ~/git/uT5-ssc
conda activate ./.conda
```

Validate Python, PyTorch, and CUDA before running GPU jobs:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device_0", torch.cuda.get_device_name(0))
PY
```

The training and evaluation entrypoint is `seq2seq/run_seq2seq.py`. Config files
under `configs/` define the model checkpoint, query language, schema
serialisation, context length, and output directory.

## Data And Datastore Services

Training configs load Spider4SSC through the dataset wrapper. Execution-based
evaluation and tokenizer round-trip audits also require the database files and
the graph datastore services:

- SQLite files are read directly from `data/Spider4SSC/database/...`.
- SPARQL queries execute against RDF4J repositories.
- Cypher queries execute against Neo4j databases.

The repository expects Spider4SSC at `data/Spider4SSC` by default. If the data
live somewhere else, pass `--dataset-root /path/to/Spider4SSC` to analysis
scripts that expose this option.

Initialise the graph services using `.scripts/init.sh` as a reference. Read the
script before running it because it creates Docker containers, creates or uses a
`neo4j` system user, and writes under `/neo4j`. The script starts Neo4j and
RDF4J Docker containers, copies the Spider4SSC database folder into the Neo4j
import root, and populates the selected split:

```bash
bash .scripts/init.sh
```

For a manual refresh after the services are already running:

```bash
python -m seq2seq.serve_rdf4j_graphs data/Spider4SSC --split dev
python -m seq2seq.serve_neo4j_graphs data/Spider4SSC --split dev
```

Use `--split all` when evaluating train, dev, and test graphs in the same
environment.

## Fine-Tuning And Evaluation

Run a single-language fine-tuning config with:

```bash
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sql_compact.json
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sparql_compact.json
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/cypher_compact.json
```

Run a batch evaluation wrapper, for example:

```bash
bash .scripts/eval_ssc_ut5_ep19-clean.sh
```

Generated metrics and prediction dumps are written to the `output_dir` specified
inside each config, commonly under `/work/results/ut5-base/Spider4SSC/...`.

## Target-Query Tokenizer Round-Trip Audit

The target-query tokenizer audit checks whether gold SQL, SPARQL, and Cypher
queries survive the same T5 tokenisation and generated-prediction decoding path
used by training and evaluation. It answers a narrow question: whether
tokenisation itself corrupts valid gold-query semantics.

The audit implementation is:

- `seq2seq/analyze_target_tokenizer_roundtrip.py`
- shared tokenizer helpers in `seq2seq/utils/query_tokenizer.py`
- tests in `seq2seq/utils/tests/test_target_tokenizer_roundtrip.py`

The shared tokenizer setup intentionally mirrors `seq2seq/run_seq2seq.py`:

```python
("{", "}", " <=", " <", "^^")
```

Decoded text is cleaned with the same generated-prediction cleanup used by
`seq2seq/utils/spider.py`: remove `</s>` and `<pad>`, collapse doubled braces,
repair `< >` to `<>`, then strip outer whitespace.

Run the dev audit with the uT5 tokenizer checkpoint:

```bash
conda run -p ./.conda env \
  TOKENIZER_NAME_OR_PATH=google-t5/t5-base \
  python seq2seq/analyze_target_tokenizer_roundtrip.py \
    --split dev \
    --output-dir results/tokenizer_roundtrip/dev
```

If your Spider4SSC data are not under `data/Spider4SSC`, add:

```bash
--dataset-root /path/to/Spider4SSC
```

Outputs:

- `results/tokenizer_roundtrip/dev/target_token_stats.csv`
- `results/tokenizer_roundtrip/dev/roundtrip_summary.json`
- `results/tokenizer_roundtrip/dev/roundtrip_examples.jsonl`
- `results/tokenizer_roundtrip/dev/roundtrip_examples.md`

Current corrected dev audit results:

| Language | Mean target tokens | Mean tokens per lexical item | Execution-equivalent round trips |
| --- | ---: | ---: | ---: |
| SQL | 35.89 | 1.93 | 608/608 |
| SPARQL | 109.20 | 2.96 | 608/608 |
| Cypher | 79.85 | 2.88 | 608/608 |

Interpretation: SPARQL and Cypher targets are more fragmented than SQL targets,
but deterministic tokenizer round trips are execution-equivalent for all dev-set
gold queries. The audit therefore supports a limited claim: tokenisation
fragmentation may contribute to learning difficulty, but destructive
tokenisation alone does not explain the SPARQL performance gap.

## Validation Commands

Run the focused tests:

```bash
conda run -p ./.conda python -m unittest seq2seq.utils.tests.test_target_tokenizer_roundtrip
```

Run a compile check over the tokenizer/audit path:

```bash
conda run -p ./.conda python -m py_compile \
  seq2seq/analyze_target_tokenizer_roundtrip.py \
  seq2seq/utils/query_tokenizer.py \
  seq2seq/utils/spider.py \
  seq2seq/run_seq2seq.py \
  seq2seq/count_tokens.py
```
