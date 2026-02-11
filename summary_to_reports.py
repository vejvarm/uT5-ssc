#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt


def _get_colors_bright():
    return {
        "cypher": [0x44 / 255, 0x77 / 255, 0xAA / 255],  # blue
        "sparql": [0x22 / 255, 0x88 / 255, 0x33 / 255],  # green
        "sql": [0xCC / 255, 0xBB / 255, 0x44 / 255],  # yellow
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _shade(color, factor: float):
    return [_clamp(c * factor) for c in color]


def _lang_from_label(label: str) -> Optional[str]:
    lowered = label.lower()
    for lang in ("cypher", "sparql", "sql"):
        if lowered.startswith(f"{lang}__") or lowered.startswith(f"{lang}-") or lowered == lang:
            return lang
    return None


def _schema_from_label(label: str) -> Optional[str]:
    lowered = label.lower()
    if "no_schema" in lowered:
        return "no_schema"
    if "schema/compact" in lowered or lowered.endswith("compact"):
        return "schema/compact"
    if "schema/norange" in lowered or lowered.endswith("norange"):
        return "schema/norange"
    return None


PRETRAIN_MARKERS = ("clean", "dirty", "injected10p", "injected30p")
SCHEMA_ORDER = ("no_schema", "schema/norange", "schema/compact")
LANG_ORDER = ("sparql", "sql", "cypher")
HARDNESS_ORDER = ("easy", "medium", "hard", "extra")
ERROR_SCHEMA_GROUP_ORDER = (
    "no_schema",
    "schema/norange",
    "schema/compact",
    "short_norange",
    "short_compact",
    "strip_root_norange",
    "strip_root_compact",
)
EXEC_ERROR_PATTERNS = [
    ("Multiple result columns with same name", r"Multiple result columns with the same name are not supported"),
    ("Undefined aggregation variable", r"Variable `?aggregation_.*?`? not defined"),
    ("Missing RETURN clause", r"Query cannot conclude with WITH"),
    ("Aggregation datatype mismatch", r"can only handle numerical values, duration, or null"),
    ("Expression without alias in WITH", r"Expression in WITH must be aliased"),
    ("Variable referenced but never bound", r"Variable `?.*?`? not defined"),
    ("Invalid syntax/input error", r"Invalid input"),
    ("Property access on non-existent map", r"Property access on .*? is not supported"),
    ("Type mismatch", r"Type mismatch"),
    ("General processing exception", r"general processing exception"),
]

try:
    from third_party.spider.process_sql import (
        get_schema as _spider_get_schema,
        Schema as _SpiderSchema,
        get_sql as _spider_get_sql,
    )
    from third_party.spider.evaluation import (
        count_component1 as _spider_count_component1,
        count_component2 as _spider_count_component2,
        count_others as _spider_count_others,
    )
    _SPIDER_HARDNESS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    _spider_get_schema = None  # type: ignore[assignment]
    _SpiderSchema = None  # type: ignore[assignment]
    _spider_get_sql = None  # type: ignore[assignment]
    _spider_count_component1 = None  # type: ignore[assignment]
    _spider_count_component2 = None  # type: ignore[assignment]
    _spider_count_others = None  # type: ignore[assignment]
    _SPIDER_HARDNESS_IMPORT_ERROR = exc

try:
    from spider4ssc_complexity import compute_complexity as _compute_complexity
    _COMPLEXITY_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    _compute_complexity = None  # type: ignore[assignment]
    _COMPLEXITY_IMPORT_ERROR = exc


def _ctxlen_from_path(rel_path: str) -> Optional[int]:
    # clean runs can contain extra context-length folders like .../clean/.../1024/cypher/... or .../2048/cypher/...
    parts = [p.lower() for p in Path(rel_path).parts]
    if "clean" not in parts:
        return None
    for p in parts:
        if p.isdigit() and p in ("1024", "2048"):
            return int(p)
    return None


def _lang_display(lang: Optional[str], ctx_len: Optional[int]) -> Optional[str]:
    if not lang:
        return lang
    if lang == "cypher" and ctx_len in (1024, 2048):
        return f"cypher-{ctx_len}"
    return lang


def _lang_display_sort_key(lang: Optional[str]) -> Tuple[int, int]:
    # Ensures base language ordering sparql, sql, cypher, with cypher variants after cypher.
    if not lang:
        return (99, 0)
    base = lang.split("-", 1)[0]
    try:
        base_idx = LANG_ORDER.index(base)
    except ValueError:
        base_idx = 99
    suffix = 0
    if "-" in lang:
        try:
            suffix = int(lang.split("-", 1)[1])
        except Exception:
            suffix = 0
    return (base_idx, suffix)


def _pretrain_sort_key(pretrain: Optional[str]) -> int:
    if pretrain in PRETRAIN_MARKERS:
        return PRETRAIN_MARKERS.index(pretrain)
    return 99


def _schema_sort_key(schema: Optional[str]) -> int:
    if schema in SCHEMA_ORDER:
        return SCHEMA_ORDER.index(schema)
    return 99


def _error_schema_group_sort_key(schema_group: Optional[str]) -> int:
    if schema_group in ERROR_SCHEMA_GROUP_ORDER:
        return ERROR_SCHEMA_GROUP_ORDER.index(schema_group)
    return 99


def _cypher_variant_from_path(rel_path: str, lang: Optional[str]) -> str:
    if lang != "cypher":
        return "base"
    parts = [p.lower() for p in Path(rel_path).parts]
    if "cypher_short" in parts:
        return "short"
    if "cypher_strip_root" in parts:
        return "strip_root"
    return "base"


def _error_schema_group(schema: Optional[str], cypher_variant: str) -> str:
    base_schema = schema or "unknown_schema"
    if cypher_variant == "base":
        return base_schema
    if base_schema in ("schema/norange", "schema/compact"):
        return f"{cypher_variant}_{base_schema.split('/')[-1]}"
    return f"{cypher_variant}_{base_schema.replace('/', '_')}"


def _split_label_from_summary_path(path: Path) -> str:
    name = path.name.lower()
    if "eval" in name:
        return "eval"
    if "test" in name:
        return "test"
    return path.stem


def _analysis_path_for_prediction_file(prediction_path: Path) -> Optional[Path]:
    explicit = prediction_path.with_name(f"{prediction_path.stem}_exec_analysis.json")
    if explicit.exists():
        return explicit
    candidates = sorted(prediction_path.parent.glob("predictions*_exec_analysis.json"))
    if candidates:
        return candidates[0]
    return None


def _categorize_exec_error_message(message: str) -> str:
    if not message:
        return "No message"
    for label, pattern in EXEC_ERROR_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return label
    return "Uncategorized"


def _increment_exec_detail_count(counts: Counter, side: str, exec_payload: Dict[str, Any]) -> None:
    if not isinstance(exec_payload, dict):
        counts[f"{side}::Other/Unmapped"] += 1
        return
    if exec_payload.get("status") != "exception":
        counts[f"{side}::Other/Unmapped"] += 1
        return
    exc = exec_payload.get("exception", {}) or {}
    if not isinstance(exc, dict):
        exc = {"message": str(exc)}
    message = str(exc.get("message", ""))
    detail = _categorize_exec_error_message(message)
    counts[f"{side}::{detail}"] += 1


def _extract_exec_error_counts_from_analysis(analysis_path: Path) -> Counter:
    counts: Counter = Counter()
    try:
        payload = json.loads(analysis_path.read_text())
    except Exception:
        return counts
    if not isinstance(payload, list):
        return counts

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        fail_info = entry.get("fail_info", {}) or {}
        fail_category = fail_info.get("category") if isinstance(fail_info, dict) else None
        if fail_category == "pred_exec_error":
            _increment_exec_detail_count(
                counts, "pred_exec_error", entry.get("pred_exec", {}) or {}
            )
        elif fail_category == "gold_exec_error":
            _increment_exec_detail_count(
                counts, "gold_exec_error", entry.get("gold_exec", {}) or {}
            )
    return counts


def _collect_exec_error_details_by_file(
    files_summary: Dict[str, Dict[str, Any]], summary_root: Path
) -> Dict[str, Counter]:
    out: Dict[str, Counter] = {}
    for rel_path in files_summary.keys():
        pred_path = summary_root / rel_path
        analysis_path = _analysis_path_for_prediction_file(pred_path)
        if analysis_path is None:
            continue
        out[rel_path] = _extract_exec_error_counts_from_analysis(analysis_path)
    return out


def _require_spider_hardness() -> None:
    if (
        _spider_get_schema is None
        or _SpiderSchema is None
        or _spider_get_sql is None
        or _spider_count_component1 is None
        or _spider_count_component2 is None
        or _spider_count_others is None
    ):
        raise ModuleNotFoundError(
            "Spider hardness dependencies are unavailable. "
            "Use the project Python env (`.conda/bin/python`) with requirements installed."
        ) from _SPIDER_HARDNESS_IMPORT_ERROR


def _require_complexity() -> None:
    if _compute_complexity is None:
        raise ModuleNotFoundError(
            "spider4ssc_complexity is unavailable. "
            "Install it in the active environment (e.g., `.conda/bin/python`)."
        ) from _COMPLEXITY_IMPORT_ERROR


def _spider_eval_hardness(sql_ast: Dict[str, Any]) -> str:
    count_comp1 = _spider_count_component1(sql_ast)
    count_comp2 = _spider_count_component2(sql_ast)
    count_others = _spider_count_others(sql_ast)

    if count_comp1 <= 1 and count_others == 0 and count_comp2 == 0:
        return "easy"
    if (count_others <= 2 and count_comp1 <= 1 and count_comp2 == 0) or (
        count_comp1 <= 2 and count_others < 2 and count_comp2 == 0
    ):
        return "medium"
    if (
        (count_others > 2 and count_comp1 <= 2 and count_comp2 == 0)
        or (2 < count_comp1 <= 3 and count_others <= 2 and count_comp2 == 0)
        or (count_comp1 <= 1 and count_others == 0 and count_comp2 <= 1)
    ):
        return "hard"
    return "extra"


def _sqlite_path_from_record(record: Dict[str, Any]) -> Optional[Path]:
    db_path_raw = (record.get("db_path") or "").strip()
    db_id = (record.get("db_id") or "").strip()
    if not db_path_raw:
        return None
    db_path = Path(db_path_raw)
    if db_path.suffix == ".sqlite":
        return db_path
    if not db_id:
        return None
    return db_path / db_id / f"{db_id}.sqlite"


def _resolve_gold_query_text(record: Dict[str, Any]) -> str:
    source = (record.get("gold_query_source_field") or "").strip()
    if source:
        raw = record.get(source)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    for key in ("gold_query_processed", "gold_sql_processed", "label", "query", "sql"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _is_standard_single_lang_file(rel_path: str, lang: Optional[str]) -> bool:
    if not lang:
        return False
    parts = [p.lower() for p in Path(rel_path).parts]
    if "single" not in parts:
        # Support compact RQ3 layout: clean/<lang>/schema/.../predictions_*.json
        if "joint" in parts:
            return False
        for idx, part in enumerate(parts[:-1]):
            if part in PRETRAIN_MARKERS and idx + 1 < len(parts) and parts[idx + 1] == lang:
                return True
        return lang in parts

    idx = parts.index("single")
    # Canonical layout: .../single/<lang>/...
    if idx + 1 < len(parts) and parts[idx + 1] == lang:
        return True
    # Context-length variant layout: .../single/1024/<lang>/... or .../single/2048/<lang>/...
    if idx + 2 < len(parts) and parts[idx + 1] in ("1024", "2048") and parts[idx + 2] == lang:
        return True
    return False


def _build_rq3_hardness_rows(
    files_summary: Dict[str, Dict[str, Any]],
    summary_root: Path,
    pretrain_filter: Optional[str] = "clean",
    schema_filter: Optional[str] = "schema/compact",
    include_ctx_variants: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    _require_spider_hardness()
    _require_complexity()
    rows: List[Dict[str, Any]] = []
    stats = Counter()
    schema_cache: Dict[str, Any] = {}
    hardness_cache: Dict[Tuple[str, str], Optional[str]] = {}
    complexity_cache: Dict[Tuple[str, str], Optional[float]] = {}

    def _hardness_for_sql(sql_text: str, sqlite_path: Path) -> Optional[str]:
        key = (str(sqlite_path), sql_text)
        if key in hardness_cache:
            return hardness_cache[key]
        try:
            schema_obj = schema_cache.get(str(sqlite_path))
            if schema_obj is None:
                schema_obj = _SpiderSchema(_spider_get_schema(str(sqlite_path)))
                schema_cache[str(sqlite_path)] = schema_obj
            sql_ast = _spider_get_sql(schema_obj, sql_text)
            hardness = _spider_eval_hardness(sql_ast)
        except Exception:
            hardness = None
        hardness_cache[key] = hardness
        return hardness

    def _complexity_for_query(query_text: str, query_language: str) -> Optional[float]:
        key = (query_language, query_text)
        if key in complexity_cache:
            return complexity_cache[key]
        try:
            res = _compute_complexity(
                query=query_text,
                query_language=query_language,
                include_sqomplexity=False,
            )
            score = res.get("complexity_score") if isinstance(res, dict) else None
            score = float(score) if score is not None else None
            if score is not None and score > 0:
                complexity_cache[key] = score
            else:
                complexity_cache[key] = None
        except Exception:
            complexity_cache[key] = None
        return complexity_cache[key]

    for rel_path, summary_stats in files_summary.items():
        lang = _dominant_lang(summary_stats.get("lang_counts", {}))
        pretrain = _pretrain_from_path(rel_path)
        schema = _schema_from_path(rel_path)
        ctx_len = _ctxlen_from_path(rel_path)

        if pretrain_filter and pretrain != pretrain_filter:
            continue
        if schema_filter and schema != schema_filter:
            continue
        if not include_ctx_variants and pretrain == "clean" and ctx_len in (1024, 2048):
            continue
        if not _is_standard_single_lang_file(rel_path, lang):
            continue

        pred_path = summary_root / rel_path
        analysis_path = _analysis_path_for_prediction_file(pred_path)
        if analysis_path is None or not analysis_path.exists():
            stats["missing_analysis_file"] += 1
            continue

        try:
            payload = json.loads(analysis_path.read_text())
        except Exception:
            stats["analysis_json_load_error"] += 1
            continue
        if not isinstance(payload, list):
            stats["analysis_not_list"] += 1
            continue

        for record in payload:
            if not isinstance(record, dict):
                stats["analysis_bad_record"] += 1
                continue
            rec_lang = (record.get("lang") or lang or "").lower()
            if rec_lang not in LANG_ORDER:
                stats["unsupported_lang"] += 1
                continue
            sql_text = (record.get("sql") or "").strip()
            if not sql_text and rec_lang == "sql":
                # SQL dataset records use `query` as the gold SQL.
                sql_text = (record.get("query") or "").strip()
            if not sql_text:
                stats["missing_sql"] += 1
                continue
            sqlite_path = _sqlite_path_from_record(record)
            if sqlite_path is None:
                stats["missing_db_path"] += 1
                continue
            if not sqlite_path.is_file():
                stats["sqlite_not_found"] += 1
                continue
            hardness = _hardness_for_sql(sql_text, sqlite_path)
            if hardness is None:
                stats["hardness_parse_error"] += 1
                continue
            match_raw = record.get("match", 0)
            if isinstance(match_raw, bool):
                match = 1 if match_raw else 0
            else:
                try:
                    match = 1 if int(match_raw) == 1 else 0
                except Exception:
                    match = 0

            # Compute complexity in the query's native language.
            # For predict/test non-sql runs, gold_query_lang is "sql", so we keep "-" for non-sql rows.
            complexity_score: Optional[float] = None
            inverse_complexity: Optional[float] = None
            gold_query_lang = (record.get("gold_query_lang") or "").lower().strip()
            if gold_query_lang and gold_query_lang == rec_lang:
                query_text = _resolve_gold_query_text(record)
                if query_text:
                    complexity_score = _complexity_for_query(query_text, rec_lang)
                    if complexity_score is None:
                        stats["complexity_calc_error"] += 1
                    else:
                        inverse_complexity = 1.0 / complexity_score
            elif rec_lang == "sql":
                # Backward compatibility for older files missing gold_query_lang.
                query_text = _resolve_gold_query_text(record)
                if query_text:
                    complexity_score = _complexity_for_query(query_text, "sql")
                    if complexity_score is None:
                        stats["complexity_calc_error"] += 1
                    else:
                        inverse_complexity = 1.0 / complexity_score
            else:
                stats["complexity_not_available_for_lang"] += 1

            rows.append(
                {
                    "file": rel_path,
                    "lang": rec_lang,
                    "pretrain": pretrain or "unknown_pretrain",
                    "schema": schema or "unknown_schema",
                    "hardness": hardness,
                    "match": match,
                    "complexity_score": complexity_score,
                    "inverse_complexity": inverse_complexity,
                }
            )
            stats["used_rows"] += 1

    return pd.DataFrame(rows), dict(stats)


def _build_rq3_ex_tables(df_rq3_rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    if df_rq3_rows.empty:
        return {
            "rq3_long": empty,
            "rq3_ex_matrix": empty,
            "rq3_counts_matrix": empty,
            "rq3_complexity_matrix": empty,
            "rq3_inverse_complexity_matrix": empty,
            "rq3_complexity_display": empty,
            "rq3_inverse_complexity_display": empty,
        }

    grouped = (
        df_rq3_rows.groupby(["lang", "hardness"], dropna=False, as_index=False)
        .agg(
            total=("match", "size"),
            matches=("match", "sum"),
            complexity_score=("complexity_score", "mean"),
            inverse_complexity=("inverse_complexity", "mean"),
            complexity_n=("complexity_score", lambda s: int(s.notna().sum())),
        )
    )
    grouped["exec_accuracy"] = grouped["matches"] / grouped["total"]

    ex_matrix = pd.pivot_table(
        grouped,
        index="lang",
        columns="hardness",
        values="exec_accuracy",
        aggfunc="mean",
    ).reindex(index=list(LANG_ORDER), columns=list(HARDNESS_ORDER))
    counts_matrix = pd.pivot_table(
        grouped,
        index="lang",
        columns="hardness",
        values="total",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=list(LANG_ORDER), columns=list(HARDNESS_ORDER), fill_value=0)
    complexity_matrix = pd.pivot_table(
        grouped,
        index="lang",
        columns="hardness",
        values="complexity_score",
        aggfunc="mean",
    ).reindex(index=list(LANG_ORDER), columns=list(HARDNESS_ORDER))
    inverse_complexity_matrix = pd.pivot_table(
        grouped,
        index="lang",
        columns="hardness",
        values="inverse_complexity",
        aggfunc="mean",
    ).reindex(index=list(LANG_ORDER), columns=list(HARDNESS_ORDER))

    ex_matrix["Average"] = ex_matrix[list(HARDNESS_ORDER)].mean(axis=1, skipna=True)
    ex_average_row = ex_matrix[list(HARDNESS_ORDER) + ["Average"]].mean(
        axis=0, skipna=True
    )
    ex_matrix.loc["Average"] = ex_average_row

    counts_matrix["Total"] = counts_matrix[list(HARDNESS_ORDER)].sum(axis=1)
    counts_total_row = counts_matrix[list(HARDNESS_ORDER) + ["Total"]].sum(
        axis=0, skipna=True
    )
    counts_matrix.loc["Total"] = counts_total_row
    complexity_matrix["Average"] = complexity_matrix[list(HARDNESS_ORDER)].mean(
        axis=1, skipna=True
    )
    complexity_avg_row = complexity_matrix[list(HARDNESS_ORDER) + ["Average"]].mean(
        axis=0, skipna=True
    )
    complexity_matrix.loc["Average"] = complexity_avg_row
    inverse_complexity_matrix["Average"] = inverse_complexity_matrix[
        list(HARDNESS_ORDER)
    ].mean(axis=1, skipna=True)
    inverse_avg_row = inverse_complexity_matrix[
        list(HARDNESS_ORDER) + ["Average"]
    ].mean(axis=0, skipna=True)
    inverse_complexity_matrix.loc["Average"] = inverse_avg_row

    ex_matrix = ex_matrix.round(4)
    counts_matrix = counts_matrix.astype(int)
    complexity_matrix = complexity_matrix.round(4)
    inverse_complexity_matrix = inverse_complexity_matrix.round(4)
    complexity_display = complexity_matrix.copy()
    inverse_display = inverse_complexity_matrix.copy()
    for col in complexity_display.columns:
        complexity_display[col] = complexity_display[col].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.2f}"
        )
    for col in inverse_display.columns:
        inverse_display[col] = inverse_display[col].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.2f}"
        )
    grouped = grouped.sort_values(
        ["lang", "hardness"],
        key=lambda s: s.map(
            lambda x: LANG_ORDER.index(x) if x in LANG_ORDER else 99
        )
        if s.name == "lang"
        else s.map(lambda x: HARDNESS_ORDER.index(x) if x in HARDNESS_ORDER else 99),
    )
    return {
        "rq3_long": grouped,
        "rq3_ex_matrix": ex_matrix,
        "rq3_counts_matrix": counts_matrix,
        "rq3_complexity_matrix": complexity_matrix,
        "rq3_inverse_complexity_matrix": inverse_complexity_matrix,
        "rq3_complexity_display": complexity_display,
        "rq3_inverse_complexity_display": inverse_display,
    }


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _dominant_lang(lang_counts: Dict[str, int]) -> Optional[str]:
    if not lang_counts:
        return None
    return max(lang_counts.items(), key=lambda x: x[1])[0]


def _schema_from_path(rel_path: str) -> Optional[str]:
    parts = Path(rel_path).parts
    for idx, part in enumerate(parts):
        lowered = part.lower()
        if lowered == "no_schema":
            return "no_schema"
        if lowered == "schema":
            if idx + 1 < len(parts):
                return f"schema/{parts[idx + 1].lower()}"
    return None


def _pretrain_from_path(rel_path: str) -> Optional[str]:
    parts = [p.lower() for p in Path(rel_path).parts]
    for marker in PRETRAIN_MARKERS:
        if marker in parts:
            return marker
    return None


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    palette = _get_colors_bright()
    lang_counts: Dict[str, int] = {}
    colors = []
    default_color = [0.6, 0.6, 0.6]
    schema_factors = {
        "no_schema": 1.20,
        "schema/norange": 0.90,
        "schema/compact": 0.70,
    }
    for label in df[x].astype(str).tolist():
        lang = _lang_from_label(label)
        schema = _schema_from_label(label)
        if lang is None:
            colors.append(default_color)
            continue
        idx = lang_counts.get(lang, 0)
        lang_counts[lang] = idx + 1
        base = palette.get(lang)
        if base is None:
            colors.append(default_color)
            continue
        factor = 1.0
        if idx == 1:
            factor = 1.15
        elif idx == 2:
            factor = 0.85
        elif idx >= 3:
            factor = 0.7
        schema_factor = schema_factors.get(schema, 1.0)
        colors.append(_shade(base, factor * schema_factor))
    plt.bar(df[x], df[y], color=colors)
    plt.title(title)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_by_pretrain_compact(df_files: pd.DataFrame, out_dir: Path) -> None:
    if df_files.empty:
        return
    if "exclude_from_aggregates" in df_files.columns:
        df_files = df_files[~df_files["exclude_from_aggregates"]].copy()
    compact = df_files[df_files["schema"] == "schema/compact"].copy()
    if compact.empty:
        return
    for pretrain in PRETRAIN_MARKERS:
        subset = compact[compact["pretrain"] == pretrain].copy()
        if subset.empty:
            continue
        subset["lang"] = pd.Categorical(subset["lang"], categories=LANG_ORDER, ordered=True)
        subset = subset.sort_values("lang")
        _plot_bar(
            subset,
            "lang",
            "match_rate",
            f"Match Rate by Language ({pretrain}, schema/compact)",
            out_dir / f"match_rate_by_language_{pretrain}_schema_compact.png",
        )


def _summary_dict_to_df(summary_dict: Dict[str, Dict[str, Any]], key_name: str) -> pd.DataFrame:
    rows = []
    for key, stats in summary_dict.items():
        rows.append(
            {
                key_name: key,
                "total": stats.get("total", 0),
                "matches": stats.get("matches", 0),
                "match_rate": stats.get("match_rate", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        if key_name == "lang":
            df["lang"] = pd.Categorical(df["lang"], categories=LANG_ORDER, ordered=True)
            df = df.sort_values("lang")
        if key_name == "schema":
            df["schema"] = pd.Categorical(df["schema"], categories=SCHEMA_ORDER, ordered=True)
            df = df.sort_values("schema")
    return df


def _sort_by_lang(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    df["_lang"] = df[col].astype(str).str.split("__").str[0].str.split("-").str[0]
    df["_lang"] = pd.Categorical(df["_lang"], categories=LANG_ORDER, ordered=True)
    df = df.sort_values(["_lang", col]).drop(columns=["_lang"])
    return df


def _sort_by_schema(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    df["_schema"] = df[col].astype(str).str.split("__").str[-1].str.split("-").str[-1]
    df["_schema"] = df["_schema"].replace(
        {"norange": "schema/norange", "compact": "schema/compact", "no_schema": "no_schema"}
    )
    df["_schema"] = pd.Categorical(df["_schema"], categories=SCHEMA_ORDER, ordered=True)
    df = df.sort_values(["_schema", col]).drop(columns=["_schema"])
    return df


def _aggregate_from_files(files_summary: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    rows = []
    for rel_path, stats in files_summary.items():
        lang = _dominant_lang(stats.get("lang_counts", {}))
        schema = _schema_from_path(rel_path)
        pretrain = _pretrain_from_path(rel_path)
        ctx_len = _ctxlen_from_path(rel_path)
        lang_disp = _lang_display(lang, ctx_len)
        # Clean: treat 1024/2048 cypher runs as separate experiments and exclude from aggregates/graphs.
        exclude = bool(pretrain == "clean" and lang == "cypher" and ctx_len in (1024, 2048))
        rows.append(
            {
                "file": rel_path,
                "lang": lang,
                "lang_display": lang_disp,
                "schema": schema,
                "pretrain": pretrain,
                "ctx_len": ctx_len,
                "exclude_from_aggregates": exclude,
                "total": stats.get("total", 0),
                "matches": stats.get("matches", 0),
                "match_rate": stats.get("match_rate", 0.0),
            }
        )

    df_files = pd.DataFrame(rows)

    def _agg(df: pd.DataFrame, keys: Tuple[str, ...]) -> pd.DataFrame:
        grouped = df.groupby(list(keys), dropna=False).agg(total=("total", "sum"), matches=("matches", "sum"))
        grouped["match_rate"] = grouped["matches"] / grouped["total"]
        return grouped.reset_index()

    df_main = df_files[~df_files["exclude_from_aggregates"]].copy() if not df_files.empty else df_files

    df_by_lang = _agg(df_main, ("lang",))
    df_by_schema = _agg(df_main, ("schema",))
    df_by_lang_schema = _agg(df_main, ("lang", "schema"))
    if not df_by_lang_schema.empty:
        df_by_lang_schema["lang_schema"] = df_by_lang_schema["lang"].astype(str) + "__" + df_by_lang_schema["schema"].astype(str)
        df_by_lang_schema = df_by_lang_schema[["lang_schema", "total", "matches", "match_rate"]]

    df_lang_pretrain = _agg(df_main, ("lang", "pretrain"))
    df_schema_pretrain = _agg(df_main, ("schema", "pretrain"))
    df_lang_schema_pretrain_main = _agg(df_main, ("lang", "schema", "pretrain"))

    # Sheet-only: cypher-1024 / cypher-2048.
    df_lang_schema_pretrain_variants = pd.DataFrame()
    if not df_files.empty:
        df_variants = df_files[df_files["exclude_from_aggregates"]].copy()
        if not df_variants.empty:
            df_lang_schema_pretrain_variants = _agg(
                df_variants, ("lang_display", "schema", "pretrain")
            ).rename(columns={"lang_display": "lang"})

    # Stable ordering for tables/plots.
    if not df_by_lang.empty:
        df_by_lang["lang"] = pd.Categorical(df_by_lang["lang"], categories=LANG_ORDER, ordered=True)
        df_by_lang = df_by_lang.sort_values("lang")
    if not df_by_schema.empty:
        df_by_schema["schema"] = pd.Categorical(df_by_schema["schema"], categories=SCHEMA_ORDER, ordered=True)
        df_by_schema = df_by_schema.sort_values("schema")
    if not df_by_lang_schema.empty:
        df_by_lang_schema = _sort_by_lang(df_by_lang_schema, "lang_schema")
        df_by_lang_schema = _sort_by_schema(df_by_lang_schema, "lang_schema")

    if not df_lang_pretrain.empty:
        df_lang_pretrain["lang"] = pd.Categorical(df_lang_pretrain["lang"], categories=LANG_ORDER, ordered=True)
        df_lang_pretrain = df_lang_pretrain.sort_values(["lang", "pretrain"])
    if not df_schema_pretrain.empty:
        df_schema_pretrain["schema"] = pd.Categorical(df_schema_pretrain["schema"], categories=SCHEMA_ORDER, ordered=True)
        df_schema_pretrain = df_schema_pretrain.sort_values(["schema", "pretrain"])

    df_lang_schema_pretrain_sheet = df_lang_schema_pretrain_main.copy()
    if not df_lang_schema_pretrain_sheet.empty:
        df_lang_schema_pretrain_sheet = df_lang_schema_pretrain_sheet.sort_values(
            ["lang", "schema", "pretrain"],
            key=lambda s: s.map(_lang_display_sort_key) if s.name == "lang" else s,
        )
    if not df_lang_schema_pretrain_variants.empty:
        df_lang_schema_pretrain_variants = df_lang_schema_pretrain_variants.sort_values(
            ["lang", "schema", "pretrain"],
            key=lambda s: s.map(_lang_display_sort_key) if s.name == "lang" else s,
        )
        df_lang_schema_pretrain_sheet = pd.concat(
            [df_lang_schema_pretrain_sheet, df_lang_schema_pretrain_variants], ignore_index=True
        )
        df_lang_schema_pretrain_sheet = df_lang_schema_pretrain_sheet.sort_values(
            ["lang", "schema", "pretrain"],
            key=lambda s: s.map(_lang_display_sort_key) if s.name == "lang" else s,
        )

    # Keep by_file sheet comprehensive (includes excluded rows).
    if not df_files.empty:
        df_files["lang"] = pd.Categorical(df_files["lang"], categories=LANG_ORDER, ordered=True)
        df_files = df_files.sort_values(["lang", "schema", "pretrain", "ctx_len", "file"])

    return {
        "df_files": df_files,
        "df_main": df_main,
        "df_by_lang": df_by_lang,
        "df_by_schema": df_by_schema,
        "df_by_lang_schema": df_by_lang_schema,
        "df_lang_pretrain": df_lang_pretrain,
        "df_schema_pretrain": df_schema_pretrain,
        "df_lang_schema_pretrain": df_lang_schema_pretrain_sheet,
    }


def _build_error_rows(
    files_summary: Dict[str, Dict[str, Any]],
    exec_error_details_by_file: Optional[Dict[str, Counter]] = None,
) -> pd.DataFrame:
    rows = []
    for rel_path, stats in files_summary.items():
        lang = _dominant_lang(stats.get("lang_counts", {}))
        schema = _schema_from_path(rel_path)
        cypher_variant = _cypher_variant_from_path(rel_path, lang)
        schema_group = _error_schema_group(schema, cypher_variant)
        pretrain = _pretrain_from_path(rel_path)
        ctx_len = _ctxlen_from_path(rel_path)
        # Keep consistency with aggregate logic in this script.
        exclude = bool(pretrain == "clean" and lang == "cypher" and ctx_len in (1024, 2048))
        if exclude:
            continue

        fail_categories = stats.get("fail_categories", {}) or {}
        details = (exec_error_details_by_file or {}).get(rel_path, Counter())
        total = int(stats.get("total", 0) or 0)
        matches = int(stats.get("matches", 0) or 0)
        fail_total = max(total - matches, 0)
        for error_category, count in fail_categories.items():
            if error_category == "match":
                continue
            cnt = int(count or 0)
            if cnt <= 0:
                continue
            if error_category in ("pred_exec_error", "gold_exec_error"):
                granular_counts = {
                    key: int(val)
                    for key, val in details.items()
                    if key.startswith(f"{error_category}::") and int(val) > 0
                }
                if granular_counts:
                    granular_total = sum(granular_counts.values())
                    for granular_label, granular_count in sorted(granular_counts.items()):
                        rows.append(
                            {
                                "file": rel_path,
                                "lang": lang or "unknown_lang",
                                "schema": schema or "unknown_schema",
                                "schema_group": schema_group,
                                "pretrain": pretrain or "unknown_pretrain",
                                "error_category": granular_label,
                                "count": granular_count,
                                "total": total,
                                "matches": matches,
                                "fail_total": fail_total,
                            }
                        )
                    if cnt > granular_total:
                        rows.append(
                            {
                                "file": rel_path,
                                "lang": lang or "unknown_lang",
                                "schema": schema or "unknown_schema",
                                "schema_group": schema_group,
                                "pretrain": pretrain or "unknown_pretrain",
                                "error_category": f"{error_category}::Other/Unmapped",
                                "count": cnt - granular_total,
                                "total": total,
                                "matches": matches,
                                "fail_total": fail_total,
                            }
                        )
                    continue
            rows.append(
                {
                    "file": rel_path,
                    "lang": lang or "unknown_lang",
                    "schema": schema or "unknown_schema",
                    "schema_group": schema_group,
                    "pretrain": pretrain or "unknown_pretrain",
                    "error_category": str(error_category),
                    "count": cnt,
                    "total": total,
                    "matches": matches,
                    "fail_total": fail_total,
                }
            )
    return pd.DataFrame(rows)


def _build_error_pivots(df_errors: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df_errors.empty:
        empty = pd.DataFrame()
        return {"error_long": empty, "error_counts": empty, "error_fail_share": empty}

    long_df = (
        df_errors.groupby(
            ["pretrain", "schema_group", "error_category"], dropna=False, as_index=False
        )
        .agg(count=("count", "sum"))
    )
    long_df["group_total_errors"] = long_df.groupby(
        ["pretrain", "schema_group"], dropna=False
    )["count"].transform("sum")
    long_df["share_within_group"] = (
        (long_df["count"] / long_df["group_total_errors"]).fillna(0.0) * 100.0
    ).round(2)

    counts = pd.pivot_table(
        long_df,
        index="error_category",
        columns=["pretrain", "schema_group"],
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    ordered_cols = sorted(
        counts.columns,
        key=lambda c: (
            _pretrain_sort_key(c[0]),
            _error_schema_group_sort_key(c[1]),
            str(c[0]),
            str(c[1]),
        ),
    )
    counts = counts.reindex(columns=ordered_cols, fill_value=0).astype(int)
    counts.columns = [f"{pretrain}__{schema_group}" for pretrain, schema_group in counts.columns]
    counts["total"] = counts.sum(axis=1).astype(int)
    counts = (
        counts.assign(_error_category=counts.index.astype(str))
        .sort_values(["total", "_error_category"], ascending=[False, True])
        .drop(columns=["_error_category"])
    )

    # Normalize each (pretrain, schema) column by total failures in that group.
    shares = counts.drop(columns=["total"]).copy().astype(float)
    for col in shares.columns:
        col_total = float(shares[col].sum())
        if col_total > 0:
            shares[col] = (shares[col] / col_total) * 100.0
        else:
            shares[col] = 0.0
    overall_total = float(counts["total"].sum())
    if overall_total > 0:
        shares["overall_share"] = (counts["total"] / overall_total) * 100.0
    else:
        shares["overall_share"] = 0.0
    shares = shares.round(2)
    shares = shares.loc[counts.index]

    return {
        "error_long": long_df.sort_values(
            ["pretrain", "schema_group", "count", "error_category"],
            ascending=[True, True, False, True],
        ),
        "error_counts": counts,
        "error_fail_share": shares,
    }


def _ordered_languages(values) -> list[str]:
    present = [str(v) for v in values if pd.notna(v)]
    unique = sorted(set(present))
    return sorted(unique, key=lambda x: (LANG_ORDER.index(x) if x in LANG_ORDER else 99, x))


def _build_error_pivots_by_lang(df_errors: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    if df_errors.empty or "lang" not in df_errors.columns:
        return out
    for lang in _ordered_languages(df_errors["lang"].tolist()):
        subset = df_errors[df_errors["lang"] == lang].copy()
        if subset.empty:
            continue
        out[lang] = _build_error_pivots(subset)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graphs, Excel pivots, and error-category CSVs from execution-summary JSON.")
    parser.add_argument("summary_json", type=str, help="Path to predictions_*_exec_summary.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for graphs and Excel. (defaults to `reports_summary` subfolder in the same folder as the `summary_json`)")
    parser.add_argument(
        "--split-label",
        type=str,
        default=None,
        help="Optional split label for output filenames (e.g., test or eval). If omitted, inferred from summary filename.",
    )
    parser.add_argument(
        "--rq3-pretrain",
        type=str,
        default="clean",
        help="Pretrain marker filter for RQ3 hardness tables (default: clean). Set to empty string to disable filtering.",
    )
    parser.add_argument(
        "--rq3-schema",
        type=str,
        default="schema/compact",
        help="Schema filter for RQ3 hardness tables (default: schema/compact). Set to empty string to disable filtering.",
    )
    parser.add_argument(
        "--rq3-include-ctx-variants",
        action="store_true",
        help="Include clean context-length variants (1024/2048) in RQ3 hardness tables.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if args.out_dir is None:
        out_dir = summary_path.parent.joinpath("reports_summary")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    files_summary = summary.get("files", {})
    split_label = args.split_label or _split_label_from_summary_path(summary_path)
    summary_root = Path(summary.get("root", str(summary_path.parent)))
    if not summary_root.is_absolute():
        summary_root = (summary_path.parent / summary_root).resolve()
    exec_error_details_by_file = _collect_exec_error_details_by_file(files_summary, summary_root)
    rq3_pretrain_filter = args.rq3_pretrain.strip() or None
    rq3_schema_filter = args.rq3_schema.strip() or None

    dfs = _aggregate_from_files(files_summary)
    df_files = dfs["df_files"]
    df_by_lang = dfs["df_by_lang"]
    df_by_schema = dfs["df_by_schema"]
    df_by_lang_schema = dfs["df_by_lang_schema"]
    df_lang_pretrain = dfs["df_lang_pretrain"]
    df_schema_pretrain = dfs["df_schema_pretrain"]
    df_lang_schema_pretrain = dfs["df_lang_schema_pretrain"]
    df_errors = _build_error_rows(
        files_summary, exec_error_details_by_file=exec_error_details_by_file
    )
    error_tables = _build_error_pivots(df_errors)
    df_error_long = error_tables["error_long"]
    df_error_counts = error_tables["error_counts"]
    df_error_fail_share = error_tables["error_fail_share"]
    error_tables_by_lang = _build_error_pivots_by_lang(df_errors)
    rq3_rows = pd.DataFrame()
    rq3_tables = {
        "rq3_long": pd.DataFrame(),
        "rq3_ex_matrix": pd.DataFrame(),
        "rq3_counts_matrix": pd.DataFrame(),
        "rq3_complexity_matrix": pd.DataFrame(),
        "rq3_inverse_complexity_matrix": pd.DataFrame(),
        "rq3_complexity_display": pd.DataFrame(),
        "rq3_inverse_complexity_display": pd.DataFrame(),
    }
    rq3_stats: Dict[str, int] = {}
    rq3_error: Optional[str] = None
    try:
        rq3_rows, rq3_stats = _build_rq3_hardness_rows(
            files_summary,
            summary_root,
            pretrain_filter=rq3_pretrain_filter,
            schema_filter=rq3_schema_filter,
            include_ctx_variants=args.rq3_include_ctx_variants,
        )
        rq3_tables = _build_rq3_ex_tables(rq3_rows)
    except ModuleNotFoundError as exc:
        rq3_error = str(exc)
    df_rq3_long = rq3_tables["rq3_long"]
    df_rq3_ex = rq3_tables["rq3_ex_matrix"]
    df_rq3_counts = rq3_tables["rq3_counts_matrix"]
    df_rq3_complexity = rq3_tables["rq3_complexity_matrix"]
    df_rq3_inv_complexity = rq3_tables["rq3_inverse_complexity_matrix"]
    df_rq3_complexity_disp = rq3_tables["rq3_complexity_display"]
    df_rq3_inv_complexity_disp = rq3_tables["rq3_inverse_complexity_display"]

    excel_path = out_dir / "prediction_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_by_lang.to_excel(writer, sheet_name="by_language", index=False)
        df_by_schema.to_excel(writer, sheet_name="by_schema", index=False)
        df_by_lang_schema.to_excel(writer, sheet_name="by_language_schema", index=False)
        df_files.to_excel(writer, sheet_name="by_file", index=False)
        df_lang_pretrain.to_excel(writer, sheet_name="by_lang_pretrain", index=False)
        df_schema_pretrain.to_excel(writer, sheet_name="by_schema_pretrain", index=False)
        df_lang_schema_pretrain.to_excel(writer, sheet_name="by_lang_schema_pretrain", index=False)
        df_error_counts.to_excel(writer, sheet_name="error_counts_pivot")
        df_error_fail_share.to_excel(writer, sheet_name="error_fail_share_pivot")
        df_error_long.to_excel(writer, sheet_name="error_long", index=False)
        for lang, tables in error_tables_by_lang.items():
            tables["error_counts"].to_excel(writer, sheet_name=f"err_cnt_{lang}")
            tables["error_fail_share"].to_excel(writer, sheet_name=f"err_share_{lang}")
        df_rq3_ex.to_excel(writer, sheet_name="rq3_ex_hardness")
        df_rq3_counts.to_excel(writer, sheet_name="rq3_counts_hardness")
        df_rq3_complexity.to_excel(writer, sheet_name="rq3_complexity_hardness")
        df_rq3_inv_complexity.to_excel(writer, sheet_name="rq3_inv_complexity_hardness")
        df_rq3_complexity_disp.to_excel(writer, sheet_name="rq3_complexity_display")
        df_rq3_inv_complexity_disp.to_excel(writer, sheet_name="rq3_inv_complexity_display")
        df_rq3_long.to_excel(writer, sheet_name="rq3_hardness_long", index=False)

    if not df_by_lang.empty:
        _plot_bar(df_by_lang, "lang", "match_rate", "Match Rate by Language", out_dir / "match_rate_by_language.png")
    if not df_by_schema.empty:
        _plot_bar(df_by_schema, "schema", "match_rate", "Match Rate by Schema", out_dir / "match_rate_by_schema.png")
    if not df_by_lang_schema.empty:
        _plot_bar(df_by_lang_schema, "lang_schema", "match_rate", "Match Rate by Lang+Schema", out_dir / "match_rate_by_lang_schema.png")
    if not df_lang_pretrain.empty:
        _plot_bar(df_lang_pretrain, "pretrain", "match_rate", "Match Rate by Pretrain (Lang aggregated)", out_dir / "match_rate_by_pretrain.png")
    _plot_by_pretrain_compact(df_files, out_dir)

    error_counts_csv = out_dir / f"error_categories_pivot_counts_{split_label}.csv"
    error_share_csv = out_dir / f"error_categories_pivot_fail_share_{split_label}.csv"
    df_error_counts.to_csv(error_counts_csv)
    df_error_fail_share.to_csv(error_share_csv)
    by_lang_outputs = []
    for lang, tables in error_tables_by_lang.items():
        lang_counts_csv = out_dir / f"error_categories_pivot_counts_{lang}_{split_label}.csv"
        lang_share_csv = out_dir / f"error_categories_pivot_fail_share_{lang}_{split_label}.csv"
        tables["error_counts"].to_csv(lang_counts_csv)
        tables["error_fail_share"].to_csv(lang_share_csv)
        by_lang_outputs.append((lang_counts_csv, lang_share_csv))

    print(f"Wrote Excel: {excel_path}")
    print(f"Wrote error counts CSV: {error_counts_csv}")
    print(f"Wrote error share CSV: {error_share_csv}")
    for counts_path, share_path in by_lang_outputs:
        print(f"Wrote lang error counts CSV: {counts_path}")
        print(f"Wrote lang error share CSV: {share_path}")

    rq3_tag_parts = [rq3_pretrain_filter or "allpretrain", (rq3_schema_filter or "allschema").replace("/", "-")]
    rq3_tag = "_".join(rq3_tag_parts)
    rq3_ex_csv = out_dir / f"rq3_ex_by_hardness_lang_{rq3_tag}_{split_label}.csv"
    rq3_counts_csv = out_dir / f"rq3_counts_by_hardness_lang_{rq3_tag}_{split_label}.csv"
    rq3_complexity_csv = out_dir / f"rq3_complexity_by_hardness_lang_{rq3_tag}_{split_label}.csv"
    rq3_inv_complexity_csv = out_dir / f"rq3_inverse_complexity_by_hardness_lang_{rq3_tag}_{split_label}.csv"
    rq3_long_csv = out_dir / f"rq3_hardness_long_{rq3_tag}_{split_label}.csv"
    rq3_latex = out_dir / f"rq3_ex_by_hardness_lang_{rq3_tag}_{split_label}.tex"
    rq3_complexity_latex = out_dir / f"rq3_complexity_by_hardness_lang_{rq3_tag}_{split_label}.tex"
    rq3_inv_complexity_latex = out_dir / f"rq3_inverse_complexity_by_hardness_lang_{rq3_tag}_{split_label}.tex"
    df_rq3_ex.to_csv(rq3_ex_csv)
    df_rq3_counts.to_csv(rq3_counts_csv)
    df_rq3_complexity_disp.to_csv(rq3_complexity_csv)
    df_rq3_inv_complexity_disp.to_csv(rq3_inv_complexity_csv)
    df_rq3_long.to_csv(rq3_long_csv, index=False)
    if not df_rq3_ex.empty:
        df_rq3_ex.round(2).to_latex(rq3_latex, float_format="%.2f")
    if not df_rq3_complexity_disp.empty:
        df_rq3_complexity_disp.to_latex(rq3_complexity_latex, escape=False)
    if not df_rq3_inv_complexity_disp.empty:
        df_rq3_inv_complexity_disp.to_latex(rq3_inv_complexity_latex, escape=False)
    print(f"Wrote RQ3 EX CSV: {rq3_ex_csv}")
    print(f"Wrote RQ3 counts CSV: {rq3_counts_csv}")
    print(f"Wrote RQ3 complexity CSV: {rq3_complexity_csv}")
    print(f"Wrote RQ3 inverse complexity CSV: {rq3_inv_complexity_csv}")
    print(f"Wrote RQ3 long CSV: {rq3_long_csv}")
    if not df_rq3_ex.empty:
        print(f"Wrote RQ3 LaTeX table: {rq3_latex}")
    if not df_rq3_complexity_disp.empty:
        print(f"Wrote RQ3 complexity LaTeX table: {rq3_complexity_latex}")
    if not df_rq3_inv_complexity_disp.empty:
        print(f"Wrote RQ3 inverse complexity LaTeX table: {rq3_inv_complexity_latex}")
    if rq3_stats:
        print(f"RQ3 hardness row stats: {json.dumps(rq3_stats, sort_keys=True)}")
    if df_rq3_long.empty:
        print(
            "RQ3 hardness tables are empty. "
            f"Filters pretrain={rq3_pretrain_filter!r}, schema={rq3_schema_filter!r}, "
            f"include_ctx_variants={args.rq3_include_ctx_variants}. "
            "Check folder layout and/or pass --rq3-pretrain '' --rq3-schema ''."
        )
    if rq3_error is not None:
        print(f"RQ3 hardness generation skipped: {rq3_error}")
    print(f"Wrote graphs to: {out_dir}")


if __name__ == "__main__":
    main()
