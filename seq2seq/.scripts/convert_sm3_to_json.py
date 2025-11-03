import argparse
import csv
import json
import re
from pathlib import Path

def extract_namespaces(sparql_query):
    # Extracts all namespace labels from PREFIX lines in a SPARQL query
    pattern = r"PREFIX\s+(\w+):\s*<[^>]+>"
    return [match.group(1) for match in re.finditer(pattern, sparql_query)]

def remove_prefix_lines(sparql_query):
    # Remove all lines starting with PREFIX (may appear mid-query)
    return re.sub(r'PREFIX\s+\w+:\s*<[^>]+>\s*', '', sparql_query, flags=re.IGNORECASE).lstrip('\n')

def csv_to_json(input_path: Path, output_path: Path):
    with input_path.open(newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        output = []
        for row in reader:
            item = dict(row)  # Copy all fields
            sparql = row.get("sparql", "")
            # Extract namespaces from original SPARQL
            item["namespaces"] = extract_namespaces(sparql)
            # Remove all PREFIX lines for the new sparql field
            sparql_no_prefix = remove_prefix_lines(sparql)
            item["sparql"] = sparql_no_prefix.strip()
            # Optional: if you want db_id = class, add/overwrite here:
            if "class" in row:
                item["db_id"] = row["class"]
            output.append(item)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Convert SM3 CSV to JSON format for text-to-SSC.")
    parser.add_argument('--input', type=Path, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=Path, required=False, help='Path to the output JSON file (optional)')
    args = parser.parse_args()

    input_path = args.input
    if args.output is not None:
        output_path = args.output
    else:
        # Default: same as input, but with .json extension
        output_path = input_path.with_suffix('.json')

    csv_to_json(input_path, output_path)
    print(f"JSON written to: {output_path}")

if __name__ == "__main__":
    main()
