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
    Analyzes a single entry to determine the cause of the row count mismatch.
    Returns a string category.
    """
    pred_query = entry.get('pred_query_processed', '').lower()
    gold_query = entry.get('gold_query_processed', '').lower()
    
    # Pre-computation
    pred_labels = extract_labels(pred_query)
    gold_labels = extract_labels(gold_query)
    
    # 1. CHECK FOR SCHEMA HALLUCINATIONS
    # Find labels in prediction that are NOT in gold
    # (We assume Gold contains the "correct" schema subset)
    diff_labels = pred_labels - gold_labels
    
    if diff_labels:
        # A. Type Hallucinations (Specific to Compact prompts)
        # Check for common data types disguised as labels
        type_keywords = {'int', 'str', 'float', 'bool', 'match', 'num', 'date', 'datetime', 'string', 'long', 'double'}
        if any(l in type_keywords for l in diff_labels):
            return "Schema Hallucination: Data Type (e.g., :int, :str)"
        
        # B. Synonym/General Hallucinations vs Wrong Table
        # Rule: If the hallucinated label starts with 'root__', it's likely a valid table 
        # but the WRONG one (e.g. root__singer instead of root__concert).
        # If it DOES NOT start with 'root__', it's a pure hallucination (e.g. :artist).
        
        # Check if ALL mismatch labels are valid ROOT tables (start with root__)
        all_are_root = all(l.startswith('root__') for l in diff_labels)
        
        if all_are_root:
            return "Logic Error: Wrong Table Selection"
        else:
            return "Schema Hallucination: Synonym/Invalid Label"

    # 2. CHECK FOR LOGIC ERRORS (Labels are correct/subset)
    
    # C. Missing Relationship / Join Pattern
    # Count relationship arrows -> or <-
    pred_arrows = pred_query.count('->') + pred_query.count('<-')
    gold_arrows = gold_query.count('->') + gold_query.count('<-')
    
    if pred_arrows < gold_arrows:
        return "Logic Error: Missing Relationship/Pattern"
        
    # D. Missing Filter Logic (WHERE clause)
    if pred_query.count('where') < gold_query.count('where'):
        return "Logic Error: Missing Filter"

    # E. Missing Aggregation
    # Check for aggregation keywords
    aggs = ['count(', 'sum(', 'avg(', 'max(', 'min(']
    pred_has_agg = any(a in pred_query for a in aggs)
    gold_has_agg = any(a in gold_query for a in aggs)
    
    if gold_has_agg and not pred_has_agg:
        return "Logic Error: Missing Aggregation"

    # F. Missing LIMIT
    if 'limit' in gold_query and 'limit' not in pred_query:
        return "Logic Error: Missing LIMIT"

    # Default fallback
    return "Logic Error: Other (Complex Mismatch)"

def process_file(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    data = load_json(filepath)
    results = []
    categorised_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    config_name = clean_config_name(filepath, "_result_mismatch__row_count_mismatch")
    
    for entry in data:
        error_type = categorize_error(entry)
        results.append({
            "Configuration": config_name,
            "Error Category": error_type
        })
        categorised_queries[error_type].append(entry)
        
    return pd.DataFrame(results), dict(categorised_queries)

def main():
    parser = argparse.ArgumentParser(description="Analyze Cypher Row Count Mismatches")
    parser.add_argument("files", nargs='+', type=Path, help="List of JSON files to analyze")
    parser.add_argument("--output", type=str, default="error_analysis_summary.csv", help="Output CSV filename")
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
    
    print("\n--- Error Analysis Summary ---")
    print(summary)
    
    summary.to_csv(args.output)
    print(f"\nSummary saved to {args.output}")

if __name__ == "__main__":
    main()
