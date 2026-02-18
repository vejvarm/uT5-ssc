import json
import pathlib
import re
from collections import defaultdict
import pandas as pd
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
        r"\.\s+[A-Za-z_]",
        r"[A-Za-z0-9_]_\s+[A-Za-z0-9_]",
        r"[A-Za-z0-9_]__\s+[A-Za-z0-9_]",
        r"\bROOT__\s+[A-Za-z_]",
        r"\b[A-Za-z0-9_]+\s+__[A-Za-z0-9_]",
        r"\[\s*:\s+[A-Za-z_]",
    ]
    return any(re.search(pattern, query) for pattern in whitespace_patterns)


def _is_catastrophic_repetition(query: str, msg: str) -> bool:
    query = query or ""
    if not query:
        return False

    repetition_patterns = [
        r"(__([A-Za-z0-9]+))(?:__\2){3,}",
        r"\b([A-Za-z0-9]+)(?:_\1){4,}\b",
        r"\bas\s+[A-Za-z_][A-Za-z0-9_]*\s+as\s+[A-Za-z_][A-Za-z0-9_]*",
        r"(sum|avg|max|min|count)\s*\([^)]*\s+\1\s*\(",
    ]
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in repetition_patterns):
        return True

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

    if re.search(r"\.\s*(int|str|float|double|bool|num|date|datetime|string|long)\b", query, re.IGNORECASE):
        return True
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
        return "Schema Hallucination"

    return "Uncategorized"

def analyze_errors(file_paths):
    results = {}
    all_file_counts = {}

    for file_path in file_paths:
        path_obj = pathlib.Path(file_path)
        
        # Robust naming: fallback to stem if directory structure assumes too much
        try:
            # Matches your specific structure: .../schema/compact/...
            # Grabs: "schema_compact_filename"
            file_name = f"{path_obj.parent.parent.name}_{path_obj.parent.name}_{path_obj.stem}" 
        except IndexError:
            file_name = path_obj.stem

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        stats = defaultdict(lambda: {"count": 0, "examples": []})
        
        for entry in data:
            category = _categorize_entry(entry)
            if category is None:
                continue
            stats[category]["count"] += 1

            msg = entry.get("pred_exec", {}).get("exception", {}).get("message", "") or ""
            clean_msg = msg.replace("\n", " ")[:200] + "..."
            if len(stats[category]["examples"]) < 3 and clean_msg not in stats[category]["examples"]:
                stats[category]["examples"].append(clean_msg)

        # Store detailed results for this file
        results[file_name] = stats
        
        # Flatten counts for the DataFrame and store in master dict
        all_file_counts[file_name] = {k: v["count"] for k, v in stats.items()}

    counts_df = pd.DataFrame.from_dict(all_file_counts).fillna(0).astype(int) if all_file_counts else pd.DataFrame()
    if not counts_df.empty:
        extras = [row for row in counts_df.index if row not in ERROR_CATEGORY_ORDER]
        counts_df = counts_df.reindex(ERROR_CATEGORY_ORDER + sorted(extras)).fillna(0).astype(int)

    if not counts_df.empty:
        counts_df.loc["Total Execution Errors"] = counts_df.sum(axis=0)

    return results, counts_df

# execution example
if __name__ == "__main__":
    clean_cypher_compact_path = "/work/results/ut5-base/Spider4SSC/predict/clean/ep19/single/cypher/schema/compact/fail_info/pred_exec_error.json"
    clean_cypher_norange_path = "/work/results/ut5-base/Spider4SSC/predict/clean/ep19/single/cypher/schema/norange/fail_info/pred_exec_error.json"
    
    files = [clean_cypher_compact_path, clean_cypher_norange_path]

    results, counts_df = analyze_errors(files)

    print(json.dumps(results, indent=2))
    print(counts_df)
