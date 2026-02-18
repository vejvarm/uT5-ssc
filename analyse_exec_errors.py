#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
import pandas as pd
from pathlib import Path
from typing import List, Optional, Set

TYPE_LABELS = {
    "int",
    "str",
    "float",
    "double",
    "bool",
    "boolean",
    "num",
    "number",
    "date",
    "datetime",
    "string",
    "long",
}

ERROR_CATEGORY_ORDER = [
    "Invalid White Space",
    "Catastrophic Repetition",
    "Schema Hallucination",
    "Multiple Result Columns with Same Name",
    "Undefined variable or alias",
    "Missing RETURN Clause",
]


def _clean_config_name(path_obj: Path) -> str:
    stem = path_obj.stem
    stem = re.sub(r"^cypher_", "", stem)
    stem = re.sub(r"_pred_exec_error$", "", stem)
    clean_name = stem.replace("_", " ").title().replace("Norange", "No Range")
    if "Compact" not in clean_name and "No Range" not in clean_name:
        clean_name = path_obj.stem
    clean_name = {
        "Compact": "Full Compact",
        "Short No Range": "No Range (w/o redundancy)",
        "Short Compact": "Compact (w/o redundancy)",
    }.get(clean_name, clean_name)
    return clean_name


def _parse_allowed_labels(context: str) -> Set[str]:
    labels = set()
    if not context:
        return labels

    parts = [p.strip() for p in context.split(" | ")]
    for seg in parts[2:]:
        if ":" not in seg:
            continue
        raw_label = seg.split(":", 1)[0].strip()
        if not raw_label:
            continue
        short = raw_label[len("ROOT__") :] if raw_label.startswith("ROOT__") else raw_label
        labels.update({raw_label.lower(), short.lower(), f"ROOT__{short}".lower()})
    return labels


def _extract_query_labels(query: str) -> List[str]:
    if not query:
        return []
    # Matches label tokens in patterns like (T1:ROOT__singer) and (:artist)
    return [
        m.group(1).lower()
        for m in re.finditer(
            r"\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*)?:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
            query,
        )
    ]


def _has_invalid_whitespace(query: str) -> bool:
    if not query:
        return False
    whitespace_patterns = [
        r"\.\s+[A-Za-z_]",  # T1. pets__pettype
        r"[A-Za-z0-9_]_\s+[A-Za-z0-9_]",  # t1_ pettype
        r"[A-Za-z0-9_]__\s+[A-Za-z0-9_]",  # documents__ template_id
        r"\bROOT__\s+[A-Za-z_]",  # ROOT__ singer
        r"\b[A-Za-z0-9_]+\s+__[A-Za-z0-9_]",  # documents __template_id
        r"\[\s*:\s+[A-Za-z_]",  # [: concert__STADIUM_ID]
    ]
    return any(re.search(pattern, query) for pattern in whitespace_patterns)


def _is_catastrophic_repetition(query: str, msg: str) -> bool:
    query = query or ""
    if not query:
        return False

    repetition_patterns = [
        r"(__([A-Za-z0-9]+))(?:__\2){3,}",  # cars__make__make__make...
        r"\b([A-Za-z0-9]+)(?:_\1){4,}\b",  # left_left_left_left_left
        r"\bas\s+[A-Za-z_][A-Za-z0-9_]*\s+as\s+[A-Za-z_][A-Za-z0-9_]*",  # as X as X
        r"(sum|avg|max|min|count)\s*\([^)]*\s+\1\s*\(",  # sum(... sum(...
    ]
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in repetition_patterns):
        return True

    # Very long repeated alias/property tails (common in runaway generation loops).
    long_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{40,}", query)
    for tok in long_tokens:
        parts = [p for p in tok.split("_") if p]
        if len(parts) < 8:
            continue
        counts = defaultdict(int)
        for p in parts:
            counts[p] += 1
        if max(counts.values()) >= 5:
            return True

    # Parser failures around repeated "as" are typically repetition artifacts.
    if re.search(r"Invalid input 'as'", msg or "", re.IGNORECASE) and query.count(" as ") >= 3:
        return True

    return False


def _is_schema_hallucination(query: str, context: str, msg: str) -> bool:
    query = query or ""
    labels = _extract_query_labels(query)
    if any(label in TYPE_LABELS for label in labels):
        return True

    allowed_labels = _parse_allowed_labels(context or "")
    if any(label not in allowed_labels for label in labels):
        return True

    # Type tokens hallucinated as properties (e.g., T1.int, T1.str).
    if re.search(r"\.\s*(int|str|float|double|bool|num|date|datetime|string|long)\b", query, re.IGNORECASE):
        return True

    # Schema-metadata leak into query syntax (e.g., match (TEACHER_ID]-> ...).
    if re.search(r"\([A-Za-z_][A-Za-z0-9_]*\]->", query):
        return True

    if re.search(
        r"Relationship types in a relationship type expressions may not be combined using ':'",
        msg or "",
        re.IGNORECASE,
    ):
        return True

    if re.search(r"Invalid input 'TEACHER_ID'", msg or "", re.IGNORECASE):
        return True

    return False


def _categorize_entry(entry: dict) -> Optional[str]:
    pred_exec = entry.get("pred_exec", {})
    if pred_exec.get("status") != "exception":
        return None

    msg = pred_exec.get("exception", {}).get("message", "") or ""
    if not msg:
        return "Uncategorized"

    query = entry.get("pred_query_processed", "") or ""
    context = entry.get("context", "") or ""

    if re.search(r"Multiple result columns with the same name are not supported", msg, re.IGNORECASE):
        return "Multiple Result Columns with Same Name"

    if re.search(r"Query cannot conclude with WITH", msg, re.IGNORECASE):
        return "Missing RETURN Clause"

    # Merge:
    # - Variable Referenced but Never Bound
    # - Expression without Alias in WITH
    # - Undefined Aggregation Variable
    if (
        re.search(r"Variable `?aggregation_.*?`? not defined", msg, re.IGNORECASE)
        or re.search(r"Expression in WITH must be aliased", msg, re.IGNORECASE)
        or re.search(r"Variable `?.*?`? not defined", msg, re.IGNORECASE)
    ):
        return "Undefined variable or alias"

    syntax_like = bool(
        re.search(
            r"Invalid input|Failed to parse string literal|Query cannot conclude with MATCH|Relationship types in a relationship type expressions may not be combined using ':'",
            msg,
            re.IGNORECASE,
        )
    )
    if syntax_like:
        if _has_invalid_whitespace(query):
            return "Invalid White Space"
        if _is_catastrophic_repetition(query, msg):
            return "Catastrophic Repetition"
        if _is_schema_hallucination(query, context, msg):
            return "Schema Hallucination"
        # Keep syntax-like fallbacks inside schema hallucination bucket to avoid fragmenting the table.
        return "Schema Hallucination"

    return "Uncategorized"


def analyze_errors(file_paths):
    all_file_counts = {}

    for file_path in file_paths:
        path_obj = Path(file_path)
        categorised_queries = defaultdict(list)
        clean_name = _clean_config_name(path_obj)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        stats = defaultdict(int)
        
        for entry in data:
            category = _categorize_entry(entry)
            if category is None:
                continue
            stats[category] += 1
            categorised_queries[category].append(entry)
        
        categorised_dir = path_obj.parent.joinpath("categorised")
        categorised_dir.mkdir(exist_ok=True)
        categorised_output = categorised_dir.joinpath(f"{path_obj.stem}_categorised.json")
        with open(categorised_output, 'w', encoding='utf-8') as f:
            json.dump(dict(categorised_queries), f, indent=2, ensure_ascii=False)
            print(f"Categorised queries saved to {categorised_output}")

        all_file_counts[clean_name] = dict(stats)

    # Create DataFrame
    counts_df = pd.DataFrame.from_dict(all_file_counts).fillna(0).astype(int) if all_file_counts else pd.DataFrame()
    
    if not counts_df.empty:
        extras = [row for row in counts_df.index if row not in ERROR_CATEGORY_ORDER]
        counts_df = counts_df.reindex(ERROR_CATEGORY_ORDER + sorted(extras)).fillna(0).astype(int)

    # Calculate Total Row
    if not counts_df.empty:
        counts_df.loc["Total Execution Errors"] = counts_df.sum(axis=0)

    return counts_df

def main():
    parser = argparse.ArgumentParser(description="Analyze Cypher Execution Errors")
    parser.add_argument("files", nargs='+', type=str, help="List of pred_exec_error JSON files")
    parser.add_argument("--csv", type=str, default="exec_error_summary.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    if not args.files:
        print("Please provide at least one JSON file.")
        return

    df = analyze_errors(args.files)
    
    print("\n--- Execution Error Analysis ---")
    print(df)
    
    df.to_csv(args.csv)
    print(f"\nSummary saved to {args.csv}")

if __name__ == "__main__":
    main()
