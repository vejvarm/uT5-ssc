# DOCUMENTATION

Use this file as the reverse-chronological engineering log for `uT5-ssc`.

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
