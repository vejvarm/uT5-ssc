#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from seq2seq.utils.db_id_filter import filter_records_by_db_id, load_db_id_filter

try:
    from third_party.test_suite import exec_eval
    from third_party.test_suite import parse as test_suite_parse
    _IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    exec_eval = None  # type: ignore[assignment]
    test_suite_parse = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


SUPPORTED_LANGS = {"sql", "sparql", "cypher", "postgresql"}
SCHEMA_MARKERS = ("no_schema",)
MODE_DEFAULTS = {
    "test": {
        "input_pattern": "predictions_test.json",
        "output_suffix": "predictions_test_exec_analysis.json",
        "summary_filename": "predictions_test_exec_summary.json",
        "fail_info_dirname": "fail_info",
        "use_label_as_gold": False,
    },
    "eval": {
        "input_pattern": "predictions_eval_None.json",
        "output_suffix": "predictions_eval_None_exec_analysis.json",
        "summary_filename": "predictions_eval_None_exec_summary.json",
        "fail_info_dirname": "fail_info_eval",
        "use_label_as_gold": True,
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_exec_eval() -> None:
    if exec_eval is None or test_suite_parse is None:
        raise ModuleNotFoundError(
            "Missing dependencies for execution analysis. "
            "Install requirements.txt or ensure third_party/test_suite deps are available."
        ) from _IMPORT_ERROR


def _db_file_path(db_root: str, db_id: str, lang: str) -> str:
    if not db_root:
        return ""
    if db_root.endswith(".sqlite") or db_root.endswith(".ttl"):
        return db_root
    ext = ".sqlite" if ("sql" in lang or "postgresql" in lang) else ".ttl"
    return str(Path(db_root) / db_id / f"{db_id}{ext}")


def _schema_from_path(path: Path, root: Path) -> Optional[str]:
    try:
        rel_parts = path.relative_to(root).parts
    except Exception:
        rel_parts = path.parts
    for idx, part in enumerate(rel_parts):
        lowered = part.lower()
        if lowered in SCHEMA_MARKERS:
            return lowered
        if lowered == "schema":
            if idx + 1 < len(rel_parts):
                schema_type = rel_parts[idx + 1].lower()
                if schema_type.endswith(".json"):
                    return "schema/unknown"
                return f"schema/{schema_type}"
    return None


def _row_sort_key(row: List[Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"type": "float", "value": str(value)}
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    if hasattr(value, "isoformat"):
        try:
            return {"type": type(value).__name__, "value": value.isoformat()}
        except Exception:
            return {"type": type(value).__name__, "value": str(value)}
    return {"type": type(value).__name__, "value": str(value)}


def _normalize_rows(denotation: List[Tuple[Any, ...]]) -> List[List[Any]]:
    return [[_normalize_value(v) for v in row] for row in denotation]


def _normalize_rows_unordered(
    denotation: List[Tuple[Any, ...]], order_matters: bool
) -> List[List[Any]]:
    unordered_rows: List[List[Any]] = []
    for row in denotation:
        unordered = exec_eval.unorder_row(row)
        unordered_rows.append([_normalize_value(v) for v in unordered])
    if not order_matters:
        unordered_rows.sort(key=_row_sort_key)
    return unordered_rows


def _denotation_stats(denotation: List[Tuple[Any, ...]]) -> Dict[str, int]:
    if not denotation:
        return {"row_count": 0, "column_count": 0}
    return {"row_count": len(denotation), "column_count": len(denotation[0])}


def _exception_payload(exc: Any) -> Dict[str, Any]:
    if isinstance(exc, BaseException):
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": type(exc).__name__, "message": str(exc)}


def _preprocess_query(query: str, lang: str) -> Tuple[Optional[str], Optional[Exception]]:
    if query is None:
        return "", None
    q = query.strip()
    if not q:
        return "", None
    q = exec_eval.postprocess(q)
    try:
        q = test_suite_parse.remove_distinct(q, lang)
    except Exception as exc:
        return None, exc
    return q, None


def _compare_denotations(
    gold: List[Tuple[Any, ...]], pred: List[Tuple[Any, ...]], order_matters: bool
) -> Tuple[bool, Dict[str, Any]]:
    if len(gold) == 0 and len(pred) == 0:
        return True, {"reason": "both_empty"}
    if len(gold) != len(pred):
        return False, {
            "reason": "row_count_mismatch",
            "gold_rows": len(gold),
            "pred_rows": len(pred),
        }
    if not gold:
        return True, {"reason": "both_empty"}
    gold_cols = len(gold[0])
    pred_cols = len(pred[0]) if pred else 0
    if pred_cols != gold_cols:
        return False, {
            "reason": "column_count_mismatch",
            "gold_cols": gold_cols,
            "pred_cols": pred_cols,
        }
    if not exec_eval.quick_rej(gold, pred, order_matters=order_matters):
        return False, {"reason": "quick_reject"}
    if exec_eval.result_eq(gold, pred, order_matters=order_matters):
        return True, {"reason": "match"}
    return False, {"reason": "permute_mismatch"}


def _run_exec(db_path: str, query: str, lang: str) -> Tuple[str, Any]:
    return asyncio.run(exec_eval.exec_on_db(db_path, query, lang=lang))


def _build_exec_payload(
    flag: str,
    denotation_or_exc: Any,
    order_matters: bool,
) -> Dict[str, Any]:
    if flag == "result":
        stats = _denotation_stats(denotation_or_exc)
        return {
            "status": "result",
            **stats,
            "normalized_rows": _normalize_rows(denotation_or_exc),
            "normalized_rows_unordered": _normalize_rows_unordered(
                denotation_or_exc, order_matters
            ),
        }
    return {
        "status": "exception",
        "exception": _exception_payload(denotation_or_exc),
    }


def _build_skipped_payload(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def _sql_db_exists(path: str) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.is_file()


def _resolve_gold_query(
    record: Dict[str, Any], lang: str, use_label_as_gold: bool
) -> Tuple[str, str, str]:
    query_field = record.get("query") or ""
    gold_sql = record.get("sql") or ""
    label_field = record.get("label") or ""

    if use_label_as_gold:
        gold_query = label_field
        source_field = "label"
        if not gold_query.strip():
            if query_field.strip():
                gold_query = query_field
                source_field = "query"
            elif gold_sql.strip():
                gold_query = gold_sql
                source_field = "sql"
        return gold_query, lang, source_field

    if "sql" in lang:
        return query_field or gold_sql, "sql", "query_or_sql"
    return gold_sql, "sql", "sql"


def analyze_record(
    record: Dict[str, Any], use_label_as_gold: bool = False
) -> Dict[str, Any]:
    _require_exec_eval()
    output = dict(record)
    lang = (record.get("lang") or "").lower()
    prediction = record.get("prediction") or ""
    db_root = record.get("db_path") or ""
    db_id = record.get("db_id") or ""
    gold_query, gold_lang, gold_source_field = _resolve_gold_query(
        record, lang, use_label_as_gold
    )
    output["gold_query_source_field"] = gold_source_field
    output["gold_query_lang"] = gold_lang

    output["pred_query_processed"], pred_parse_err = _preprocess_query(prediction, lang)
    output["gold_query_processed"], gold_parse_err = _preprocess_query(
        gold_query, gold_lang
    )
    # Keep legacy key for downstream consumers that expect this field name.
    output["gold_sql_processed"] = output["gold_query_processed"]

    order_matters = False
    if output["gold_query_processed"]:
        order_matters = "order by" in output["gold_query_processed"].lower()
    output["order_matters"] = order_matters

    pred_db_path = _db_file_path(db_root, db_id, lang)
    gold_db_path = _db_file_path(db_root, db_id, gold_lang)
    output["db_pred_path"] = pred_db_path
    output["db_gold_path"] = gold_db_path

    fail_info: Dict[str, Any] = {"category": "match"}

    pred_exec_payload: Dict[str, Any] = _build_skipped_payload("uninitialized")
    gold_exec_payload: Dict[str, Any] = _build_skipped_payload("uninitialized")
    pred_flag: Optional[str] = None
    gold_flag: Optional[str] = None
    pred_denotation: Optional[Any] = None
    gold_denotation: Optional[Any] = None

    if not lang:
        fail_info = {"category": "missing_lang"}
        pred_exec_payload = _build_skipped_payload("missing_lang")
        gold_exec_payload = _build_skipped_payload("missing_lang")
    elif lang not in SUPPORTED_LANGS:
        fail_info = {"category": "unsupported_lang", "lang": lang}
        pred_exec_payload = _build_skipped_payload("unsupported_lang")
        gold_exec_payload = _build_skipped_payload("unsupported_lang")
    else:
        if not prediction.strip():
            pred_exec_payload = _build_skipped_payload("missing_prediction")
        elif pred_parse_err is not None:
            pred_exec_payload = _build_skipped_payload("pred_parse_error")
        else:
            if "sql" in lang and not _sql_db_exists(pred_db_path):
                pred_exec_payload = _build_skipped_payload("missing_db_path")
            else:
                pred_flag, pred_denotation = _run_exec(
                    pred_db_path, output["pred_query_processed"], lang
                )
                pred_exec_payload = _build_exec_payload(
                    pred_flag, pred_denotation, order_matters
                )

        if not gold_query.strip():
            gold_exec_payload = _build_skipped_payload("missing_gold_sql")
        elif gold_parse_err is not None:
            gold_exec_payload = _build_skipped_payload("gold_parse_error")
        else:
            if "sql" in gold_lang and not _sql_db_exists(gold_db_path):
                gold_exec_payload = _build_skipped_payload("missing_db_path")
            else:
                gold_flag, gold_denotation = _run_exec(
                    gold_db_path, output["gold_query_processed"], gold_lang
                )
                gold_exec_payload = _build_exec_payload(
                    gold_flag, gold_denotation, order_matters
                )

        if pred_flag == "result" and gold_flag == "result":
            match, mismatch = _compare_denotations(
                gold_denotation, pred_denotation, order_matters
            )
            output["match"] = 1 if match else 0
            if not match:
                fail_info = {"category": "result_mismatch", **mismatch}
        else:
            output["match"] = 0

        if not prediction.strip():
            fail_info = {"category": "missing_prediction"}
        elif not gold_query.strip():
            fail_info = {"category": "missing_gold_sql"}
        elif pred_parse_err is not None:
            fail_info = {
                "category": "pred_parse_error",
                "exception": _exception_payload(pred_parse_err),
            }
        elif gold_parse_err is not None:
            fail_info = {
                "category": "gold_parse_error",
                "exception": _exception_payload(gold_parse_err),
            }
        elif gold_exec_payload.get("status") == "exception":
            fail_info = {
                "category": "gold_exec_error",
                "exception": gold_exec_payload.get("exception"),
            }
        elif pred_exec_payload.get("status") == "exception":
            fail_info = {
                "category": "pred_exec_error",
                "exception": pred_exec_payload.get("exception"),
            }
        elif gold_exec_payload.get("status") == "skipped":
            fail_info = {"category": gold_exec_payload.get("reason", "gold_skipped")}
        elif pred_exec_payload.get("status") == "skipped":
            fail_info = {"category": pred_exec_payload.get("reason", "pred_skipped")}
        elif output.get("match") == 1:
            fail_info = {"category": "match"}

    output["pred_exec"] = pred_exec_payload
    output["gold_exec"] = gold_exec_payload
    output["fail_info"] = fail_info
    return output


def _fail_group_key(fail_info: Dict[str, Any]) -> str:
    category = fail_info.get("category", "unknown")
    reason = fail_info.get("reason")
    if reason:
        return f"{category}__{reason}"
    return category


def _reduced_exec_payload(exec_payload: Dict[str, Any]) -> Dict[str, Any]:
    status = exec_payload.get("status")
    if status == "result":
        rows = exec_payload.get("normalized_rows") or []
        return {
            "status": "result",
            "normalized_rows": rows[:5],
        }
    if status == "exception":
        return {
            "status": "exception",
            "exception": exec_payload.get("exception"),
        }
    return {"status": status}


def analyze_file(
    path: Path,
    max_entries: Optional[int] = None,
    use_label_as_gold: bool = False,
    db_id_filter: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    data = filter_records_by_db_id(data, db_id_filter)
    if max_entries is not None:
        data = data[:max_entries]

    analyzed: List[Dict[str, Any]] = []
    category_counts: Counter = Counter()
    match_count = 0
    lang_counts: Counter = Counter()
    fail_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in data:
        lang = (record.get("lang") or "").lower()
        if lang:
            lang_counts[lang] += 1
        analyzed_record = analyze_record(
            record, use_label_as_gold=use_label_as_gold
        )
        analyzed.append(analyzed_record)
        group_key = _fail_group_key(analyzed_record["fail_info"])
        category_counts[group_key] += 1
        if analyzed_record.get("match") == 1:
            match_count += 1
        fail_groups[group_key].append(
            {
                "db_id": analyzed_record.get("db_id"),
                "context": analyzed_record.get("context"),
                "pred_query_processed": analyzed_record.get("pred_query_processed"),
                "gold_query_processed": analyzed_record.get("gold_query_processed"),
                "gold_sql_processed": analyzed_record.get("gold_sql_processed"),
                "pred_exec": _reduced_exec_payload(analyzed_record.get("pred_exec", {})),
                "gold_exec": _reduced_exec_payload(analyzed_record.get("gold_exec", {})),
            }
        )

    total = len(analyzed)
    summary = {
        "total": total,
        "matches": match_count,
        "match_rate": (match_count / total) if total else 0.0,
        "fail_categories": dict(category_counts),
        "lang_counts": dict(lang_counts),
    }
    return analyzed, summary, fail_groups


def summarize_analyzed_file(path: Path) -> Dict[str, Any]:
    analyzed = json.loads(path.read_text())
    if not isinstance(analyzed, list):
        raise ValueError(f"Expected a list in {path}, got {type(analyzed)}")

    category_counts: Counter = Counter()
    match_count = 0
    lang_counts: Counter = Counter()
    for record in analyzed:
        lang = (record.get("lang") or "").lower()
        if lang:
            lang_counts[lang] += 1
        category_counts[_fail_group_key(record.get("fail_info", {}))] += 1
        if record.get("match") == 1:
            match_count += 1

    total = len(analyzed)
    return {
        "total": total,
        "matches": match_count,
        "match_rate": (match_count / total) if total else 0.0,
        "fail_categories": dict(category_counts),
        "lang_counts": dict(lang_counts),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative execution error analysis for predictions_test/eval outputs."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root folder to search for prediction files.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_DEFAULTS.keys()),
        default="test",
        help="Preset for defaults: test uses SQL gold; eval uses label gold with same language as prediction.",
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default=None,
        help="Filename to search recursively (defaults depend on --mode).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Output filename to write next to each prediction input (defaults depend on --mode).",
    )
    parser.add_argument(
        "--fail-info-dirname",
        type=str,
        default=None,
        help="Directory name for grouped failures next to each prediction file (defaults depend on --mode).",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap on number of entries per file (for quick tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without running execution.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path for a global summary JSON (defaults depend on --mode).",
    )
    parser.add_argument(
        "--db-id-list",
        type=str,
        default=None,
        help="Optional comma-separated db_id list to analyze.",
    )
    parser.add_argument(
        "--db-id-file",
        type=Path,
        default=None,
        help="Optional newline-separated db_id file to analyze.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Build the global summary from existing execution-analysis files without running execution.",
    )
    args = parser.parse_args()

    mode_defaults = MODE_DEFAULTS[args.mode]
    input_pattern = args.input_pattern or mode_defaults["input_pattern"]
    output_suffix = args.output_suffix or mode_defaults["output_suffix"]
    fail_info_dirname = args.fail_info_dirname or mode_defaults["fail_info_dirname"]
    use_label_as_gold = bool(mode_defaults["use_label_as_gold"])
    db_id_filter = load_db_id_filter(args.db_id_list, args.db_id_file)

    root = Path(args.root)
    prediction_files = sorted(root.rglob(input_pattern))

    if args.dry_run:
        for p in prediction_files:
            print(p)
        return
    if not args.summarize_existing:
        _require_exec_eval()

    global_summary = {
        "root": str(root),
        "generated_at": _now_iso(),
        "files": {},
        "by_language": {},
        "by_schema": {},
        "by_language_schema": {},
    }
    by_language = defaultdict(lambda: {"total": 0, "matches": 0, "fail_categories": Counter()})
    by_schema = defaultdict(lambda: {"total": 0, "matches": 0, "fail_categories": Counter()})
    by_lang_schema = defaultdict(lambda: {"total": 0, "matches": 0, "fail_categories": Counter()})

    for path in prediction_files:
        if args.summarize_existing:
            summary = summarize_analyzed_file(path)
            rel_path = str(path.with_name(path.name.replace("_exec_analysis", "")).relative_to(root))
        else:
            analyzed, summary, fail_groups = analyze_file(
                path,
                max_entries=args.max_entries,
                use_label_as_gold=use_label_as_gold,
                db_id_filter=db_id_filter,
            )
            output_path = path.with_name(output_suffix)
            write_json(output_path, analyzed)
            rel_path = str(path.relative_to(root))

        global_summary["files"][rel_path] = summary

        dominant_lang = None
        if summary["lang_counts"]:
            dominant_lang = max(summary["lang_counts"].items(), key=lambda x: x[1])[0]
        if dominant_lang:
            lang_bucket = by_language[dominant_lang]
            lang_bucket["total"] += summary["total"]
            lang_bucket["matches"] += summary["matches"]
            lang_bucket["fail_categories"].update(summary["fail_categories"])

        schema = _schema_from_path(path, root)
        if schema:
            schema_bucket = by_schema[schema]
            schema_bucket["total"] += summary["total"]
            schema_bucket["matches"] += summary["matches"]
            schema_bucket["fail_categories"].update(summary["fail_categories"])
            if dominant_lang:
                key = f"{dominant_lang}__{schema}"
                lang_schema_bucket = by_lang_schema[key]
                lang_schema_bucket["total"] += summary["total"]
                lang_schema_bucket["matches"] += summary["matches"]
                lang_schema_bucket["fail_categories"].update(summary["fail_categories"])

        if not args.summarize_existing:
            fail_info_dir = path.parent / fail_info_dirname
            if fail_info_dir.exists():
                for existing in fail_info_dir.glob("*.json"):
                    existing.unlink()
            else:
                fail_info_dir.mkdir(parents=True, exist_ok=True)
            for group_key, entries in fail_groups.items():
                out_path = fail_info_dir / f"{group_key}.json"
                write_json(out_path, entries)

    for lang, stats in by_language.items():
        total = stats["total"]
        matches = stats["matches"]
        global_summary["by_language"][lang] = {
            "total": total,
            "matches": matches,
            "match_rate": (matches / total) if total else 0.0,
            "fail_categories": dict(stats["fail_categories"]),
        }
    for schema, stats in by_schema.items():
        total = stats["total"]
        matches = stats["matches"]
        global_summary["by_schema"][schema] = {
            "total": total,
            "matches": matches,
            "match_rate": (matches / total) if total else 0.0,
            "fail_categories": dict(stats["fail_categories"]),
        }
    for key, stats in by_lang_schema.items():
        total = stats["total"]
        matches = stats["matches"]
        global_summary["by_language_schema"][key] = {
            "total": total,
            "matches": matches,
            "match_rate": (matches / total) if total else 0.0,
            "fail_categories": dict(stats["fail_categories"]),
        }

    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else root / mode_defaults["summary_filename"]
    )
    write_json(summary_path, global_summary)


if __name__ == "__main__":
    main()
