# Execution-equivalence semantics (Spider4SSC: SQLite / RDF4J / Neo4j)

This document formalises how this codebase defines *execution-result equality* (a.k.a. denotation equivalence) when evaluating/predicting across the three query languages used in Spider4SSC:

- **SQL** executed on **SQLite** (`.sqlite`)
- **SPARQL** executed on **RDF4J** (`.ttl` data loaded into an RDF4J repository)
- **Cypher** executed on **Neo4j** (`.ttl` data loaded into a Neo4j database)

The goal is to make our execution-equivalence checking reproducible by specifying:

- query pre-processing and post-processing applied during evaluation/prediction,
- how results are materialised and normalised into a common Python representation,
- the exact equality semantics used for comparison (ordering, duplicates, NULLs, datatypes),
- timeouts and failure handling.

## Scope (what we actually use)

This document focuses on the execution + equivalence protocol implemented in `third_party/test_suite/exec_eval.py` (and its direct dependencies).

- We **do not** use `third_party/test_suite/exec_eval_sync.py` nor `third_party/test_suite/exec_eval_multi_.py`.
- We **do not** use any separate “classical” evaluator for execution-equivalence; equality is defined by `exec_eval.py:result_eq`.

**Code references (authoritative implementation):**

- Equality + execution (core): `third_party/test_suite/exec_eval.py` (`eval_exec_match`, `exec_on_db`, `result_eq`)
- Metric harness (calls into `exec_eval.py`, does not redefine equality): `third_party/test_suite/evaluation.py` (`Evaluator.evaluate_one`)
- DISTINCT stripping (used by `exec_eval.py`): `third_party/test_suite/parse.py` (`remove_distinct`)
- SPARQL transport (used by `exec_eval.py`): `third_party/test_suite/rdf4j_connector.py` (`RDF4jConnector.query_rdf4j`)
- Cypher transport (used by `exec_eval.py`): `third_party/test_suite/neo4j_connector.py` (`Neo4jConnector.query_neo4j`)
- Upstream query-string cleanup (outside equality semantics): `seq2seq/utils/spider.py` (`SpiderTrainer._post_process_function`)

---

## 1) Execution protocol (`exec_eval.py`)

### 1.1 Query normalisation before execution

Before calling `third_party/test_suite/exec_eval.py:eval_exec_match`, our trainer/pipeline may clean model outputs to reduce trivial execution failures. This is upstream of `exec_eval.py` and does **not** change the equality semantics in §2–§4.

- Strip decoder artefacts / fix tokenisation: remove `</s>` and `<pad>`, fix `{{`/`}}` and `< >`.
- **SPARQL**: remove custom datatype suffixes in literals via `replace_custom_datatypes(..., keep_xsd=False)` (e.g. `"foo"^^ex:SomeType` → `"foo"`).
- **Cypher**: if an identifier-shortening mapping was used during training, restore original identifiers via `IdentifierMapping.restore_query(...)`.
- If `target_with_db_id=True`, drop the leading `{db_id} |` prefix before evaluation by splitting on the first `|`.

Inside `third_party/test_suite/exec_eval.py:eval_exec_match`, both gold and predicted queries are normalised before execution:

1. **Operator spacing fix** (`postprocess`):
   - `"> =" → ">="`, `"< =" → "<="`, `"! =" → "!="`
2. **DISTINCT is intentionally ignored** (`keep_distinct=False`) during execution evaluation (and therefore duplicates can be present and are compared under bag semantics; see §4):
   - SQL: removes the `DISTINCT` token via `sqlparse` parsing.
   - SPARQL/Cypher: literal string replacement of `DISTINCT`/`distinct`.
3. **Value “plugging” is disabled** in our runs (`plug_value=False`):
   - we evaluate the predicted query **as-is** (no substitution of gold literal values into the prediction).

### 1.2 Which databases are executed (test-suite semantics)

Execution accuracy follows the Spider “test suite” protocol:
1. A “db path” is provided per example (`db_dir/db_id/db_id.{sqlite|ttl}`).
2. For execution evaluation, we run the gold and predicted query against **every DB file in the same directory** that matches the language extension:
   - SQL: all `*.sqlite`
   - SPARQL/Cypher: all `*.ttl`

Concretely (from `eval_exec_match`):

- `db_dir = dirname(db)` and then `db_paths = [join(db_dir, f) for f in listdir(db_dir) if f.endswith(ext)]`.

The prediction is considered correct only if it matches the gold denotation **on every DB/graph variant** in that directory.

### 1.3 Execution and timeout semantics

For each `db_path` in the directory:

- Execute gold and predicted queries concurrently (`asyncio.gather`) using `exec_on_db`.
- Each individual execution is time-limited by `asyncio.wait_for(..., timeout=TIMEOUT)`.
  - Default `TIMEOUT` is **60 seconds** (`third_party/test_suite/exec_eval.py:TIMEOUT = 60`).

Failure handling:

- **Gold query**: must execute successfully on every DB variant; otherwise an `AssertionError` is raised.
- **Predicted query**: if it throws an exception (including timeouts), it is treated as **non-equivalent** for that DB variant.

### 1.4 Result materialisation (engine → Python denotation)

All engines are normalised into a single representation:

> **Denotation** = `List[Tuple[Any, ...]]` (a list of rows; each row is a Python tuple of cell values).

Engine-specific steps:

- **SQLite (SQL)**: `sqlite3` cursor executes, then `fetchall()` returns `List[Tuple]`.
  - `text_factory` is set to decode bytes with `errors="ignore"` (to avoid decode crashes).
  - A deterministic rewrite is applied for time-dependent SQL constructs:
    - `YEAR(CURDATE())` is rewritten to the constant `2020` (see `replace_cur_year`).
- **RDF4J (SPARQL)**: query results are returned in SPARQL JSON format and converted as follows:
  - **Backend**: queries are sent to an RDF4J server; the `.ttl` file path is used to derive the RDF4J repository ID.
    - Repository ID: `Path(kg_path).stem`
    - Namespaces: before execution, all namespaces returned by RDF4J for that repository are prepended as `PREFIX` declarations (see `RDF4jConnector.query_rdf4j`).
    - Assumption: the repository is already created and populated with the corresponding KG before evaluation (see `.scripts/init.sh` and `README.md`).
  - For each binding row, values are ordered according to `head.vars`.
  - Typed literals are cast to Python primitives using the datatype URI:
    - `xsd:decimal`, `xsd:float`, `xsd:double` → `float`
    - `xsd:integer`, `xsd:long`, `xsd:int` → `int`
    - `xsd:boolean` → `bool`
    - `xsd:string` → `str`
    - `xsd:dateTime` → `neo4j.time.DateTime`
    - `xsd:date` → `neo4j.time.Date`
    - `xsd:time` → `neo4j.time.Time`
    - unknown datatypes: kept as the raw value string returned by RDF4J
  - Values without an RDF datatype (e.g. URIs) are kept as strings.
  - Unbound variables become `None`.
  - Special-case: if RDF4J returns the string `"0"` for a variable whose name contains `"aggregation"`, it is mapped to `None` (see `convert_rdflib_value`).
- **Neo4j (Cypher)**: results are obtained via the Neo4j Python driver and converted to:
  - **Backend**: queries are sent to a Neo4j server; the `.ttl` file path is used to derive the Neo4j database name.
    - Default database name: `Path(kg_path).stem` with underscores removed.
    - Optional override: if `NEO4J_OVERRIDE_KG` is set, that database is used for all evaluations.
    - Assumption: the database is already created and populated with the corresponding KG before evaluation (see `.scripts/init.sh` and `README.md`).
  - `res_out = [tuple(record) for record in result]`, where each `record` is already a list of values.
  - Neo4j `null` becomes Python `None`.

Because all three engines are converted to the same denotation representation (`List[Tuple[Any, ...]]`), the equality check in `exec_eval.py:result_eq` applies uniformly (and can be used cross-engine) without any additional canonicalisation beyond the conversions above.

---

## 2) Equality semantics between denotations (ordering, duplicates, NULLs, types)

### 2.1 When does row ordering matter?

We use a *gold-query heuristic*:

- `order_matters = ("order by" in g_str.lower())`

This check is done inside `exec_eval.py:eval_exec_match` after the query-string normalisation in §1.1 (i.e., on the postprocessed gold string `g_str`). If the gold query contains `ORDER BY`, we require the predicted result to match **row order**; otherwise we compare results as **unordered**.

### 2.2 Duplicates (“bag semantics”)

Unless `ORDER BY` is present, we compare results using **bag (multiset) semantics**:

- duplicates are preserved,
- two results are equal only if each distinct row occurs with the same multiplicity in both.

This is implemented by a multiset equality check (`multiset_eq`) after candidate column permutations are applied.

### 2.3 Column order and column names

We do **not** compare column names.

Instead, we treat the *projected columns* as potentially permuted across engines and allow **column-order permutation**:

- Two denotations are equal if there exists a **permutation of columns** that makes them equal (subject to row ordering rules above).

This addresses cross-engine differences such as:

- different projection-order defaults,
- different column/variable naming conventions (especially in SPARQL JSON bindings).

### 2.4 NULL / unbound values

All NULL-like values are normalised to Python `None` before comparison:

- SQL `NULL` → `None`
- SPARQL unbound variable → `None`
- Neo4j `null` → `None`

Equality then uses standard Python tuple equality, so `None` equals `None` and is distinct from strings like `"NULL"`.

### 2.5 Datatype casting (strings vs numerics, dates/times)

Comparison is *type-aware*: values are compared using Python equality after normalisation.

- SPARQL typed literals are cast based on their XML Schema datatype URI (see `DATATYPES` and `convert_rdflib_value`).
- Cypher values returned by the Neo4j driver are used as-is.
- SQLite returns Python types produced by `sqlite3`.

Important consequence:

- The string `"1"` is **not equal** to the integer `1`.
- Floats are compared **exactly** (no tolerance; see §2.6).

### 2.6 Floating point tolerance

There is **no floating-point tolerance** in `result_eq`:

- floats must match exactly under Python equality after the normalisation described above.

### 2.7 Canonicalisation

We do **not** fully canonicalise result sets (e.g., by sorting rows/columns deterministically and then comparing).

Instead, we implement equivalence by searching over **candidate column permutations** and then applying either:

- exact sequence equality (if `order_matters=True`), or
- multiset equality (if `order_matters=False`).

---

## 3) Formal definition of denotation equivalence

Let a denotation be a finite list of rows, each row an *n*-tuple:

- `G = [g1, ..., gk]`, `P = [p1, ..., pk]`
- `gi, pi ∈ V^n` (same arity required; `k` rows required to match exactly)
- `π` is a permutation of the column indices `{0, ..., n-1}`
- `π(p)` permutes tuple elements: `π(p)[i] = p[π(i)]`

We define equality as:

1. **Shape constraints**:
   - If both are empty, they are equal.
   - Otherwise, equal only if they have the same number of rows and columns.
2. **Order-sensitive case** (`order_matters=True`):
   - `G ≡ P` iff there exists a permutation `π` such that:
     - `G == [π(p1), ..., π(pk)]` as sequences (same row order).
3. **Order-insensitive case** (`order_matters=False`):
   - `G ≡ P` iff there exists a permutation `π` such that:
     - the multiset of rows in `G` equals the multiset of rows in `[π(p1), ..., π(pk)]`.

All value comparisons are done via Python equality on the normalised values (§1.5).

---

## 4) Pseudocode (as implemented)

Below is a faithful outline of `third_party/test_suite/exec_eval.py:result_eq` (names simplified):

```python 
def exec_equivalent(gold_rows, pred_rows, order_matters):
    # 1) Empty / shape checks
    if len(gold_rows) == 0 and len(pred_rows) == 0:
        return True
    if len(gold_rows) != len(pred_rows):
        return False
    if len(pred_rows[0]) != len(gold_rows[0]):
        return False

    # 2) Quick reject (ignores column order inside each row)
    if not quick_rej(gold_rows, pred_rows, order_matters):
        return False

    # 3) Try to find a column permutation that makes them equal
    for pi in constrained_column_permutations(gold_rows, pred_rows):
        pred_pi = [permute(row, pi) for row in pred_rows]
        if order_matters:
            if gold_rows == pred_pi:
                return True
        else:
            if set(gold_rows) == set(pred_pi) and multiset_equal(gold_rows, pred_pi):
                return True
    return False
```

`constrained_column_permutations` uses value-based constraints (and random sampling of up to 20 rows when the arity is > 3) to reduce the number of permutations explored, but does not change the semantics: it still searches for a column permutation that makes the denotations equal.

---

## 5) Practical notes / known limitations

- **No float tolerance**: if your workload contains floating computations with engine-specific rounding, this evaluator may produce false negatives.
- **ORDER BY heuristic**: ordering is only enforced if the gold query contains the substring `"order by"`. Queries where ordering is semantically relevant but not expressed via `ORDER BY` are treated as unordered.
- **DISTINCT disabled by default**: because we remove DISTINCT tokens before execution, this evaluator does not test DISTINCT correctness.
