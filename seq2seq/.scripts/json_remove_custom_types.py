import argparse
import json
from pathlib import Path
import re
import shutil
from seq2seq.utils.helpers import replace_custom_datatypes
# Pattern for 'something'^^prefix: or "something"^^prefix: or ^^<URI>

def process_file(path, keep_xsd=True):
    backup_path = path.with_name(path.stem + "_customtypes.json")
    shutil.copy(str(path), str(backup_path))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for item in data:
        if "sparql" in item:
            orig = item["sparql"]
            new = replace_custom_datatypes(orig, keep_xsd=keep_xsd)
            if orig != new:
                item["sparql"] = new
                changed += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return changed, len(data), backup_path

def main():
    parser = argparse.ArgumentParser(description="Remove custom datatypes from SPARQL in dev/train jsons.")
    parser.add_argument("data_dir", type=Path, help="Folder containing train.json and dev.json")
    parser.add_argument("--plain", action="store_true", help="Remove all datatypes, do NOT convert to xsd:string.")
    args = parser.parse_args()

    for fname in ["train.json", "dev.json"]:
        path = args.data_dir / fname
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        changed, total, backup_path = process_file(path, keep_xsd=not args.plain)
        print(f"{fname}: {changed}/{total} SPARQL queries changed. Original backed up as {backup_path.name}")

if __name__ == "__main__":
    main()
