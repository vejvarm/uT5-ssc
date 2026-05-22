# DOCUMENTATION

Use this file as the reverse-chronological engineering log for `uT5-ssc`.

### 2026-05-22 (local) - Added Spider4SSC target-query tokenizer round-trip audit
- Scope: `seq2seq/analyze_target_tokenizer_roundtrip.py`, `seq2seq/utils/tests/test_target_tokenizer_roundtrip.py`.
- Problem: reviewer R5.5 raised a tokenizer-confound concern: SPARQL might appear harder because the T5 tokenizer fragments RDF/SPARQL syntax, rather than because of query-language or data-model difficulty.
- Changes:
  - added a dev-split audit command that tokenizes and decodes native Spider4SSC SQL/SPARQL/Cypher gold target queries using the same fast T5 tokenizer setup and extra tokens as `seq2seq/count_tokens.py`,
  - computes per-language target token statistics, exact and normalized round-trip match rates, execution-equivalence rates, and Wilson 95% confidence intervals,
  - delegates execution equivalence to the existing `third_party.test_suite.exec_eval.eval_exec_match` path so SQL, SPARQL, and Cypher use the same postprocessing and execution semantics as the paper pipeline,
  - writes `target_token_stats.csv`, `roundtrip_summary.json`, `roundtrip_examples.jsonl`, and `roundtrip_examples.md` under `results/tokenizer_roundtrip/dev/`.
- Run command:
  - `TOKENIZER_NAME_OR_PATH=/home/vejvar-martin-nj/git/balanced-plms/results/t5/unbiased-openwebtext-10k/clean python3 seq2seq/analyze_target_tokenizer_roundtrip.py --split dev --output-dir results/tokenizer_roundtrip/dev`
- Validation:
  - `python3 -m unittest seq2seq.utils.tests.test_target_tokenizer_roundtrip` -> `OK (4 tests)`,
  - `python3 -m py_compile seq2seq/analyze_target_tokenizer_roundtrip.py seq2seq/utils/tests/test_target_tokenizer_roundtrip.py` -> `OK`,
  - `python3 seq2seq/analyze_target_tokenizer_roundtrip.py --help` -> `OK`.

### 2026-04-04 00:34 (local) - Added repository contributor guide and engineering log scaffold
- Scope: `agents.md`, `documentation.md`.
- Problem: the repository did not have a project-specific contributor guide or a local engineering log, so follow-up work on Spider4SSC fine-tuning, evaluation, and schema-serving had no single place documenting structure, conventions, or validation paths.
- Root cause: repository knowledge was spread across `README.md`, `configs/`, `seq2seq/`, and ad hoc analysis scripts, but there was no maintained summary of the codebase surfaces or expected working flow.
- Changes:
  - added `agents.md` with a `uT5-ssc`-specific architecture snapshot covering the config-driven experiment layout, the `seq2seq/run_seq2seq.py` entrypoint, dataset and metric modules, schema-serving helpers, and analysis/reporting surfaces,
  - documented repository rules around config-first changes, dataset and metric consistency, generated-artifact handling, KG-serving assumptions, and targeted test updates,
  - created `documentation.md` as the reverse-chronological local engineering log and seeded it with this initial entry.
- Validation:
  - reviewed `README.md`, `seq2seq/run_seq2seq.py`, `seq2seq/utils/dataset.py`, `seq2seq/utils/dataset_loader.py`, `seq2seq/utils/trainer.py`, `.scripts/init.sh`, and representative configs under `configs/5_ut5-ep19-dirty/` to confirm the documented structure and workflow,
  - `python3 -m unittest seq2seq.utils.tests.test_cypher_identifier_mapping` -> `OK (3 tests)`,
  - `python3 -m py_compile seq2seq/run_seq2seq.py seq2seq/utils/dataset.py seq2seq/utils/dataset_loader.py seq2seq/utils/trainer.py` -> `OK`.
