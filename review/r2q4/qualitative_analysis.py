import json
import pathlib
import re
from collections import defaultdict
import pandas as pd

def analyze_errors(file_paths):
    # Regex patterns for categorization
    error_patterns = [
        ("Multiple result columns with same name", r"Multiple result columns with the same name are not supported"),
        ("Undefined aggregation variable", r"Variable `?aggregation_.*?`? not defined"),
        ("Missing RETURN clause", r"Query cannot conclude with WITH"),
        ("Aggregation Datatype mismatch", r"can only handle numerical values, duration, or null"),
        ("Expression without alias in WITH", r"Expression in WITH must be aliased"),
        ("Variable referenced but never bound", r"Variable `?.*?`? not defined"),
        ("Invalid Syntax / Input Error", r"Invalid input"),
        ("Property access on non-existent map", r"Property access on .*? is not supported"),
        ("Type mismatch", r"Type mismatch"),
        ("General Processing Exception", r"general processing exception")
    ]

    results = {}
    all_file_counts = {} # Container to hold counts for every file

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
            pred_exec = entry.get("pred_exec", {})
            if pred_exec.get("status") != "exception":
                continue
                
            msg = pred_exec.get("exception", {}).get("message", "")
            if not msg: continue

            matched = False
            for error_name, pattern in error_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    stats[error_name]["count"] += 1
                    
                    # Store unique examples (up to 3)
                    clean_msg = msg.replace("\n", " ")[:200] + "..."
                    if len(stats[error_name]["examples"]) < 3 and clean_msg not in stats[error_name]["examples"]:
                         stats[error_name]["examples"].append(clean_msg)
                    matched = True
                    break
            
            if not matched:
                stats["Uncategorized"]["count"] += 1
                clean_msg = msg.replace("\n", " ")[:200] + "..."
                if len(stats["Uncategorized"]["examples"]) < 3:
                    stats["Uncategorized"]["examples"].append(clean_msg)

        # Store detailed results for this file
        results[file_name] = stats
        
        # Flatten counts for the DataFrame and store in master dict
        all_file_counts[file_name] = {k: v["count"] for k, v in stats.items()}

    # --- DATAFRAME CREATION (Outside the loop) ---
    # Create DF from dict where Keys=Columns (FileNames) and sub-keys=Rows (ErrorTypes)
    counts_df = pd.DataFrame.from_dict(all_file_counts).fillna(0).astype(int)
    
    # Sort index alphabetically to keep error types grouped consistently, or you can hardcode order
    counts_df = counts_df.sort_index()

    # Calculate Total Row (Summing down the columns, which is axis=0)
    if not counts_df.empty:
        counts_df.loc["Total"] = counts_df.sum(axis=0)

    return results, counts_df

# execution example
if __name__ == "__main__":
    clean_cypher_compact_path = "/work/results/ut5-base/Spider4SSC/predict/clean/ep19/single/cypher/schema/compact/fail_info/pred_exec_error.json"
    clean_cypher_norange_path = "/work/results/ut5-base/Spider4SSC/predict/clean/ep19/single/cypher/schema/norange/fail_info/pred_exec_error.json"
    
    files = [clean_cypher_compact_path, clean_cypher_norange_path]

    results, counts_df = analyze_errors(files)

    print(json.dumps(results, indent=2))
    print(counts_df)