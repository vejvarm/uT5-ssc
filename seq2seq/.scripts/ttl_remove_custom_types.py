import argparse
import re
from pathlib import Path
import shutil
import sys

SYNTH_DATATYPE_REGEX = re.compile(r'(["\'])(.+?)\1\^\^<https://knacc\.umbc\.edu/dae-young/kim/ontologies/synthea#[^>]+>')
DATATYPE_DEF_REGEX = re.compile(r'^<https://knacc\.umbc\.edu/dae-young/kim/ontologies/synthea#[^>]+>\s+a\s+rdfs:Datatype\s*\.\s*$', re.MULTILINE)

def replace_in_file(ttl_path, remove_datatype_defs=True):
    with open(ttl_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all synthea custom datatypes with xsd:string
    new_content = SYNTH_DATATYPE_REGEX.sub(lambda m: f"\"{m.group(2)}\"^^xsd:string", content)

    # Optionally remove the custom datatype definitions themselves
    if remove_datatype_defs:
        new_content = DATATYPE_DEF_REGEX.sub('', new_content)

    if new_content != content:
        with open(ttl_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Replaced in: {ttl_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Backup folder, then convert all Synthea custom datatypes to xsd:string in TTLs recursively.")
    parser.add_argument("folder", type=Path, help="Root folder containing TTL files")
    args = parser.parse_args()

    folder = args.folder.resolve()
    parent = folder.parent
    original_name = folder.name
    backup_name = original_name + "_folder"
    backup_path = parent / backup_name

    # 1. Backup/rename the folder
    if backup_path.exists():
        print(f"Backup folder {backup_path} already exists. Please move or remove it first.")
        sys.exit(1)
    print(f"Backing up {folder} to {backup_path} ...")
    shutil.move(str(folder), str(backup_path))

    # 2. Copy it back to the original location for processing
    print(f"Copying backup {backup_path} back to {folder} for in-place editing ...")
    shutil.copytree(str(backup_path), str(folder))

    # 3. Replace in all TTLs inside the newly created (original name) folder
    count = 0
    for ttl_path in folder.rglob("*.ttl"):
        if replace_in_file(ttl_path):
            count += 1

    print(f"Done. Changed {count} TTL files in {folder}.")
    print(f"Original data safely backed up as {backup_path}")

if __name__ == "__main__":
    main()
