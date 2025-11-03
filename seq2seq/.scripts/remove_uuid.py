import argparse
import json
from pathlib import Path
import re
import shutil

UUID_PATTERN = re.compile(r"(['\"])(.+?)\1\^\^uuid:")

def refactor_sparql(sparql):
    # Replace any quoted string literal followed by ^^uuid:
    def repl(match):
        val = match.group(2)
        # Always output as double quotes
        return f"\"{val}\""
    return UUID_PATTERN.sub(repl, sparql)

def process_file(path):
    backup_path = path.with_name(path.stem + "_uuid.json")
    shutil.copy(str(path), str(backup_path))  # Backup original
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for item in data:
        if "sparql" in item:
            orig = item["sparql"]
            new = refactor_sparql(orig)
            if orig != new:
                item["sparql"] = new
                changed += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return changed, len(data), backup_path

def main():
    parser = argparse.ArgumentParser(description="Backup and clean dev/train jsons by removing ^^uuid: in SPARQL fields.")
    parser.add_argument("data_dir", type=Path, help="Folder with train.json and dev.json")
    args = parser.parse_args()

    for fname in ["train.json", "dev.json"]:
        path = args.data_dir / fname
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        changed, total, backup_path = process_file(path)
        print(f"{fname}: {changed}/{total} SPARQL queries changed. Original backed up as {backup_path.name}")

if __name__ == "__main__":
    main()
