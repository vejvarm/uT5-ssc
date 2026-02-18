#!/usr/bin/env python3
import argparse
import json
import re
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_json(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    Analyzes a single entry to determine if columns are missing or extra.
    """
    pred_query = entry.get('pred_query_processed', '').lower()
    gold_query = entry.get('gold_query_processed', '').lower()
    
    # Helper to count items in the RETURN clause
    def get_return_count(query):
        if 'return' not in query:
            return 0
        # Split by 'return' and take the last part
        # (Be careful of subqueries, but this heuristic works for most Spider/Kaggle style queries)
        return_part = query.split('return')[-1]
        
        # Strip out subsequent clauses like ORDER BY, LIMIT, SKIP
        return_part = re.split(r'order by|limit|skip', return_part)[0]
        
        # Count commas + 1 (e.g. "a, b" -> 1 comma -> 2 cols)
        # Empty string check
        if not return_part.strip():
            return 0
        return return_part.count(',') + 1

    pred_cols = get_return_count(pred_query)
    gold_cols = get_return_count(gold_query)
    
    # 1. Compare Counts
    if pred_cols < gold_cols:
        return "Missing Property Column"
    elif pred_cols > gold_cols:
        return "Extra Property Column"
    else:
        # If counts are equal, it shouldn't be in this file! 
        # But if it is, it might be a mismatch in column *names* that 
        # the evaluator flagged as column mismatch (rare).
        return "Other Column Error"

def process_file(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    data = load_json(filepath)
    results = []
    categorised_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    config_name = clean_config_name(filepath, "_result_mismatch__column_count_mismatch")
    
    for entry in data:
        error_type = categorize_error(entry)
        results.append({
            "Configuration": config_name,
            "Error Category": error_type
        })
        categorised_queries[error_type].append(entry)
        
    return pd.DataFrame(results), dict(categorised_queries)

def main():
    parser = argparse.ArgumentParser(description="Analyze Cypher Column Count Mismatches")
    parser.add_argument("files", nargs='+', type=Path, help="List of JSON files to analyze")
    parser.add_argument("--output", type=str, default="column_mismatch_summary.csv", help="Output CSV filename")
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
    
    summary = final_df.groupby(['Configuration', 'Error Category']).size().unstack(fill_value=0)
    summary['Total Errors'] = summary.sum(axis=1)
    
    print("\n--- Column Count Mismatch Summary ---")
    print(summary)
    
    summary.to_csv(args.output)
    print(f"\nSummary saved to {args.output}")

if __name__ == "__main__":
    main()
