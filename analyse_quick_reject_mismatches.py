#!/usr/bin/env python3
import argparse
import json
import re
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

def load_json(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_labels(query: str) -> Set[str]:
    """Extracts node labels like (n:Label) from a Cypher query."""
    if not query:
        return set()
    return set(re.findall(r'\(\w+\s*:\s*([\w_]+)\)', query, re.IGNORECASE))

def clean_config_name(filepath: Path, suffix: str) -> str:
    stem = filepath.stem
    stem = re.sub(r"^cypher_", "", stem)
    stem = re.sub(fr"{re.escape(suffix)}$", "", stem)
    clean_name = stem.replace("_", " ").title().replace("Norange", "No Range")
    return {
        "Compact": "Full Compact",
        "Short No Range": "No Range (w/o redundancy)",
        "Short Compact": "Compact (w/o redundancy)",
    }.get(clean_name, clean_name)

def categorize_error(entry: Dict[str, Any]) -> str:
    """
    Analyzes a single entry to determine the cause of the Value Mismatch (Quick Reject).
    Returns a string category.
    """
    pred_query = entry.get('pred_query_processed', '').lower()
    gold_query = entry.get('gold_query_processed', '').lower()
    
    # 1. CHECK FOR SCHEMA HALLUCINATIONS FIRST
    # (Even if columns are wrong, if the table is fake, that's the root cause)
    pred_labels = extract_labels(pred_query)
    gold_labels = extract_labels(gold_query)
    
    diff_labels = pred_labels - gold_labels
    
    if diff_labels:
        # Check for Types (int, str, etc.) - specific to Compact
        type_keywords = {'int', 'str', 'float', 'bool', 'match', 'num', 'date', 'datetime'}
        if any(l in type_keywords for l in diff_labels):
            return "Schema Hallucination: Type (e.g., :int)"
        
        # Check for Synonyms (non-ROOT) vs Wrong Table (ROOT)
        all_are_root = all(l.startswith('root__') for l in diff_labels)
        if all_are_root:
            return "Wrong Table Selection"
        else:
            return "Schema Hallucination: Synonym"

    # 2. CHECK FOR COLUMN / PROPERTY MISMATCH
    # Extract properties T1.prop
    pred_props = set(re.findall(r'\.\s*([\w_]+)', pred_query))
    gold_props = set(re.findall(r'\.\s*([\w_]+)', gold_query))
    
    # If the sets of accessed properties don't match, it's a column selection error
    # (We filter out common mismatches that might just be aliasing artifacts if needed, 
    # but exact property name matching is usually required in Cypher)
    if pred_props != gold_props:
        return "Wrong Column/Property"

    # 3. CHECK FOR FILTER VALUES
    # Extract string literals 'Value'
    pred_literals = set(re.findall(r"['\"](.*?)['\"]", pred_query))
    gold_literals = set(re.findall(r"['\"](.*?)['\"]", gold_query))
    if pred_literals != gold_literals:
        return "Wrong Filter Value"

    # 4. CHECK FOR AGGREGATION / ORDERING
    pred_aggs = re.findall(r'(count|sum|avg|max|min)\s*\(', pred_query)
    gold_aggs = re.findall(r'(count|sum|avg|max|min)\s*\(', gold_query)
    if sorted(pred_aggs) != sorted(gold_aggs):
        return "Aggregation/Order Error"

    if 'order by' in gold_query and 'order by' not in pred_query:
        return "Aggregation/Order Error"

    # Fallback
    return "Other Value Mismatch"

def process_file(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    data = load_json(filepath)
    results = []
    categorised_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    config_name = clean_config_name(filepath, "_result_mismatch__quick_reject")
    
    for entry in data:
        error_type = categorize_error(entry)
        results.append({
            "Configuration": config_name,
            "Error Category": error_type
        })
        categorised_queries[error_type].append(entry)
        
    return pd.DataFrame(results), dict(categorised_queries)

def main():
    parser = argparse.ArgumentParser(description="Analyze Cypher Value Mismatches (Quick Reject)")
    parser.add_argument("files", nargs='+', type=Path, help="List of JSON files to analyze")
    parser.add_argument("--output", type=str, default="value_mismatch_summary.csv", help="Output CSV filename")
    args = parser.parse_args()

    all_dfs = []
    for f in args.files:
        if not f.exists():
            print(f"Skipping {f}: File not found")
            continue
        print(f"Processing {f.name}...")
        df, categorised_queries = process_file(f)
        all_dfs.append(df)

        categorised_dir = f.parent.joinpath("categorised")
        categorised_dir.mkdir(exist_ok=True)
        categorised_output = categorised_dir.joinpath(f"{f.stem}_categorised.json")
        with open(categorised_output, "w", encoding="utf-8") as out:
            json.dump(categorised_queries, out, indent=2, ensure_ascii=False)
        print(f"Categorised queries saved to {categorised_output}")
    
    if not all_dfs:
        print("No data processed.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Create the summary pivot table
    summary = final_df.groupby(['Configuration', 'Error Category']).size().unstack(fill_value=0)
    
    # Calculate totals
    summary['Total Errors'] = summary.sum(axis=1)
    
    print("\n--- Value Mismatch Analysis Summary ---")
    print(summary)
    
    summary.to_csv(args.output)
    print(f"\nSummary saved to {args.output}")

if __name__ == "__main__":
    main()
