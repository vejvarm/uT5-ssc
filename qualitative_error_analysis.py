#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from third_party.test_suite import exec_eval
    from third_party.test_suite import parse as test_suite_parse
    _IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    exec_eval = None  # type: ignore[assignment]
    test_suite_parse = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


SUPPORTED_LANGS = {"sql", "sparql", "cypher", "postgresql"}


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


def analyze_record(record: Dict[str, Any]) -> Dict[str, Any]:
    _require_exec_eval()
    output = dict(record)
    lang = (record.get("lang") or "").lower()
    prediction = record.get("prediction") or ""
    gold_sql = record.get("sql") or ""
    db_root = record.get("db_path") or ""
    db_id = record.get("db_id") or ""

    output["pred_query_processed"], pred_parse_err = _preprocess_query(prediction, lang)
    output["gold_sql_processed"], gold_parse_err = _preprocess_query(gold_sql, "sql")

    order_matters = False
    if output["gold_sql_processed"]:
        order_matters = "order by" in output["gold_sql_processed"].lower()
    output["order_matters"] = order_matters

    pred_db_path = _db_file_path(db_root, db_id, lang)
    gold_db_path = _db_file_path(db_root, db_id, "sql")
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

        if not gold_sql.strip():
            gold_exec_payload = _build_skipped_payload("missing_gold_sql")
        elif gold_parse_err is not None:
            gold_exec_payload = _build_skipped_payload("gold_parse_error")
        else:
            if not _sql_db_exists(gold_db_path):
                gold_exec_payload = _build_skipped_payload("missing_db_path")
            else:
                gold_flag, gold_denotation = _run_exec(
                    gold_db_path, output["gold_sql_processed"], "sql"
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
        elif not gold_sql.strip():
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


def analyze_file(path: Path, max_entries: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    if max_entries is not None:
        data = data[:max_entries]

    analyzed: List[Dict[str, Any]] = []
    category_counts: Counter = Counter()
    match_count = 0
    lang_counts: Counter = Counter()

    for record in data:
        lang = (record.get("lang") or "").lower()
        if lang:
            lang_counts[lang] += 1
        analyzed_record = analyze_record(record)
        analyzed.append(analyzed_record)
        category_counts[analyzed_record["fail_info"]["category"]] += 1
        if analyzed_record.get("match") == 1:
            match_count += 1

    total = len(analyzed)
    summary = {
        "total": total,
        "matches": match_count,
        "match_rate": (match_count / total) if total else 0.0,
        "fail_categories": dict(category_counts),
        "lang_counts": dict(lang_counts),
    }
    return analyzed, summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative error analysis for predictions_test.json outputs."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root folder to search for predictions_test.json files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="predictions_test_exec_analysis.json",
        help="Output filename to write next to each predictions_test.json.",
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
        help="Optional path for a global summary JSON.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    prediction_files = sorted(root.rglob("predictions_test.json"))

    if args.dry_run:
        for p in prediction_files:
            print(p)
        return
    _require_exec_eval()

    global_summary = {
        "root": str(root),
        "generated_at": _now_iso(),
        "files": {},
        "by_language": {},
    }
    by_language = defaultdict(lambda: {"total": 0, "matches": 0, "fail_categories": Counter()})

    for path in prediction_files:
        analyzed, summary = analyze_file(path, max_entries=args.max_entries)
        output_path = path.with_name(args.output_suffix)
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

    for lang, stats in by_language.items():
        total = stats["total"]
        matches = stats["matches"]
        global_summary["by_language"][lang] = {
            "total": total,
            "matches": matches,
            "match_rate": (matches / total) if total else 0.0,
            "fail_categories": dict(stats["fail_categories"]),
        }

    summary_path = Path(args.summary_path) if args.summary_path else root / "predictions_test_exec_summary.json"
    write_json(summary_path, global_summary)


if __name__ == "__main__":
    main()
